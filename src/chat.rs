use anyhow::{Context, Result, anyhow, bail};
use jsonschema::Validator;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
    sync::OnceLock,
};
use uuid::Uuid;

pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const TOOL_START: &str = "<minimax:tool_call>";
pub const TOOL_END: &str = "</minimax:tool_call>";

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_text: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub current_date: Option<String>,
    #[serde(default)]
    pub current_location: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AssistantToolCall {
    pub id: String,
    pub r#type: &'static str,
    pub function: AssistantFunctionCall,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AssistantFunctionCall {
    pub name: String,
    /// OpenAI represents arguments as a JSON-encoded string.
    pub arguments: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedAssistant {
    pub reasoning: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<AssistantToolCall>,
}

pub fn visible_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| match part {
                Value::String(text) => Some(text.clone()),
                Value::Object(object)
                    if object.get("type").and_then(Value::as_str) == Some("text") =>
                {
                    object
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(other) => other.to_string(),
    }
}

fn json_jinja(value: &Value, output: &mut String) -> Result<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => write!(output, "{value}")?,
        Value::String(value) => output.push_str(&serde_json::to_string(value)?),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                json_jinja(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                output.push_str(&serde_json::to_string(key)?);
                output.push_str(": ");
                json_jinja(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn tool_function(tool: &Value) -> Result<&Value> {
    tool.get("function")
        .context("each tool must contain a function object")
}

// llama.cpp round-trips OpenAI tools through `common_chat_tool`, retaining only
// these three function fields and supplying defaults before Jinja rendering.
fn render_tool_function(tool: &Value, output: &mut String) -> Result<()> {
    let function = tool_function(tool)?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .context("each tool function must have a string name")?;
    let description = match function.get("description") {
        None => "",
        Some(description) => description
            .as_str()
            .context("each tool function description must be a string")?,
    };

    output.push_str("{\"name\": ");
    output.push_str(&serde_json::to_string(name)?);
    output.push_str(", \"description\": ");
    output.push_str(&serde_json::to_string(description)?);
    output.push_str(", \"parameters\": ");
    if let Some(parameters) = function.get("parameters") {
        json_jinja(parameters, output)?;
    } else {
        output.push_str("{}");
    }
    output.push('}');
    Ok(())
}

#[derive(Clone, Debug)]
struct ToolDefinition {
    parameters: Value,
    validator: Validator,
}

#[derive(Clone, Debug, Default)]
pub struct ToolRegistry {
    definitions: HashMap<String, ToolDefinition>,
}

impl ToolRegistry {
    pub fn from_tools(tools: &[Value]) -> Result<Self> {
        let mut definitions = HashMap::with_capacity(tools.len());
        for tool in tools {
            let function = tool_function(tool)?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .context("each tool function must have a string name")?
                .to_owned();
            let parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));
            let validator = jsonschema::validator_for(&parameters).map_err(|error| {
                anyhow!("tool {name} has an invalid parameters schema: {error}")
            })?;
            if definitions
                .insert(
                    name.clone(),
                    ToolDefinition {
                        parameters,
                        validator,
                    },
                )
                .is_some()
            {
                bail!("duplicate tool function name: {name}")
            }
        }
        Ok(Self { definitions })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}

impl ToolChoice {
    fn parse(choice: Option<&Value>) -> Result<Self> {
        match choice {
            None | Some(Value::Null) => Ok(Self::Auto),
            Some(Value::String(choice)) => match choice.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                _ => bail!(
                    "invalid tool_choice {choice:?}; expected \"auto\", \"none\", \"required\", or a function object"
                ),
            },
            Some(Value::Object(choice)) => {
                if choice.get("type").and_then(Value::as_str) != Some("function") {
                    bail!("tool_choice.type must be \"function\"")
                }
                let name = choice
                    .get("function")
                    .and_then(Value::as_object)
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .context("tool_choice.function.name must be a non-empty string")?;
                Ok(Self::Function(name.to_owned()))
            }
            Some(_) => {
                bail!("tool_choice must be \"auto\", \"none\", \"required\", or a function object")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ToolPolicy {
    tools: Vec<Value>,
    registry: ToolRegistry,
    requires_call: bool,
}

impl ToolPolicy {
    pub fn from_request(tools: &[Value], choice: Option<&Value>) -> Result<Self> {
        let registry = ToolRegistry::from_tools(tools)?;
        match ToolChoice::parse(choice)? {
            ToolChoice::Auto => Ok(Self {
                tools: tools.to_vec(),
                registry,
                requires_call: false,
            }),
            ToolChoice::None => Ok(Self {
                tools: Vec::new(),
                registry: ToolRegistry::default(),
                requires_call: false,
            }),
            ToolChoice::Required => {
                if tools.is_empty() {
                    bail!("tool_choice \"required\" requires at least one tool")
                }
                Ok(Self {
                    tools: tools.to_vec(),
                    registry,
                    requires_call: true,
                })
            }
            ToolChoice::Function(name) => {
                let tool = tools
                    .iter()
                    .find(|tool| {
                        tool_function(tool)
                            .ok()
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                            == Some(name.as_str())
                    })
                    .cloned()
                    .with_context(|| format!("tool_choice selects unknown function {name:?}"))?;
                let tools = vec![tool];
                Ok(Self {
                    registry: ToolRegistry::from_tools(&tools)?,
                    tools,
                    requires_call: true,
                })
            }
        }
    }

    pub fn tools(&self) -> &[Value] {
        &self.tools
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    pub fn requires_call(&self) -> bool {
        self.requires_call
    }

    pub fn validate_calls(&self, calls: &[AssistantToolCall]) -> Result<()> {
        if self.requires_call && calls.is_empty() {
            bail!("tool_choice requires at least one valid tool call")
        }
        Ok(())
    }
}

fn incoming_tool_call(call: &Value) -> Result<(String, Map<String, Value>)> {
    let call = call.get("function").unwrap_or(call);
    let name = call
        .get("name")
        .and_then(Value::as_str)
        .context("assistant tool call is missing function.name")?
        .to_owned();
    let arguments = match call.get("arguments") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(arguments)) => arguments.clone(),
        Some(Value::String(arguments)) => match serde_json::from_str::<Value>(arguments) {
            Ok(Value::Object(arguments)) => arguments,
            Ok(_) => bail!("assistant tool-call arguments must be a JSON object"),
            Err(error) => bail!("invalid assistant tool-call arguments: {error}"),
        },
        Some(_) => bail!("assistant tool-call arguments must be an object or JSON string"),
    };
    Ok((name, arguments))
}

/// Render MiniMax-M2.7's native chat template, including tool definitions and
/// prior assistant/tool turns. This mirrors chat_template.jinja in the model.
pub fn render_prompt(
    messages: &[ChatMessage],
    tools: &[Value],
    requires_tool_call: bool,
) -> Result<String> {
    if messages.is_empty() {
        bail!("messages must not be empty")
    }
    let (system, conversation) = if messages
        .first()
        .is_some_and(|message| message.role == "system")
    {
        (messages.first(), &messages[1..])
    } else {
        (None, messages)
    };

    let mut prompt = String::from("]~!b[]~b]system\n");
    let system_text = system
        .map(|message| visible_text(message.content.as_ref()))
        .filter(|content| !content.is_empty())
        .unwrap_or_else(|| {
            "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
                .to_owned()
        });
    prompt.push_str(&system_text);
    if let Some(date) = system.and_then(|message| message.current_date.as_deref()) {
        write!(prompt, "\nCurrent date: {date}")?;
    }
    if let Some(location) = system.and_then(|message| message.current_location.as_deref()) {
        write!(prompt, "\nCurrent location: {location}")?;
    }

    if !tools.is_empty() {
        prompt.push_str("\n\n# Tools\n");
        prompt.push_str(if requires_tool_call {
            "You must call one or more tools to assist with the user query.\n"
        } else {
            "You may call one or more tools to assist with the user query.\n"
        });
        prompt.push_str("Here are the tools available in JSONSchema format:\n\n<tools>\n");
        for tool in tools {
            prompt.push_str("<tool>");
            render_tool_function(tool, &mut prompt)?;
            prompt.push_str("</tool>\n");
        }
        prompt.push_str(
            "</tools>\n\nWhen making tool calls, use XML format to invoke tools and pass parameters:\n\n\
             <minimax:tool_call>\n<invoke name=\"tool-name-1\">\n\
             <parameter name=\"param-key-1\">param-value-1</parameter>\n\
             <parameter name=\"param-key-2\">param-value-2</parameter>\n...\n\
             </invoke>\n</minimax:tool_call>",
        );
    }
    prompt.push_str("[e~[\n");

    let last_user_index = conversation
        .iter()
        .rposition(|message| message.role == "user");
    let mut assistant_had_tool_call = false;
    for (index, message) in conversation.iter().enumerate() {
        match message.role.as_str() {
            "user" => {
                prompt.push_str("]~b]user\n");
                prompt.push_str(&visible_text(message.content.as_ref()));
                prompt.push_str("[e~[\n");
                assistant_had_tool_call = false;
            }
            "assistant" => {
                prompt.push_str("]~b]ai\n");
                let mut content = visible_text(message.content.as_ref());
                let explicit_reasoning = message
                    .reasoning_content
                    .as_ref()
                    .or(message.reasoning.as_ref())
                    .or(message.reasoning_text.as_ref())
                    .cloned();
                let reasoning = if let Some(reasoning) = explicit_reasoning {
                    Some(reasoning)
                } else if let Some((before, after)) = content.split_once(THINK_END) {
                    let reasoning = before
                        .rsplit_once(THINK_START)
                        .map_or(before, |(_, reasoning)| reasoning)
                        .trim_matches('\n')
                        .to_owned();
                    content = after.trim_matches('\n').to_owned();
                    Some(reasoning)
                } else {
                    None
                };
                if reasoning
                    .as_ref()
                    .is_some_and(|reasoning| !reasoning.is_empty())
                    && last_user_index.is_none_or(|last_user| index > last_user)
                {
                    write!(
                        prompt,
                        "<think>\n{}\n</think>\n\n",
                        reasoning.as_deref().unwrap_or_default()
                    )?;
                }
                if !content.is_empty() {
                    prompt.push_str(&content);
                }
                assistant_had_tool_call = !message.tool_calls.is_empty();
                if assistant_had_tool_call {
                    prompt.push_str("\n<minimax:tool_call>\n");
                    for call in &message.tool_calls {
                        let (name, arguments) = incoming_tool_call(call)?;
                        writeln!(prompt, "<invoke name=\"{name}\">")?;
                        for (name, value) in arguments {
                            write!(prompt, "<parameter name=\"{name}\">")?;
                            if let Value::String(value) = value {
                                prompt.push_str(&value);
                            } else {
                                json_jinja(&value, &mut prompt)?;
                            }
                            writeln!(prompt, "</parameter>")?;
                        }
                        prompt.push_str("</invoke>\n");
                    }
                    prompt.push_str("</minimax:tool_call>");
                }
                prompt.push_str("[e~[\n");
            }
            "tool" => {
                if !assistant_had_tool_call {
                    bail!(
                        "tool message{} has no preceding assistant tool call",
                        message
                            .tool_call_id
                            .as_deref()
                            .map_or(String::new(), |id| format!(" {id}"))
                    )
                }
                if index == 0 || conversation[index - 1].role != "tool" {
                    prompt.push_str("]~b]tool");
                }
                write!(
                    prompt,
                    "\n<response>{}</response>",
                    visible_text(message.content.as_ref())
                )?;
                if index + 1 == conversation.len() || conversation[index + 1].role != "tool" {
                    prompt.push_str("[e~[\n");
                }
            }
            // Ignore unsupported roles rather than accidentally rendering them as users.
            _ => {}
        }
    }
    prompt.push_str("]~b]ai\n<think>\n");
    Ok(prompt)
}

fn invoke_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r#"(?s)<\s*invoke\s+name\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^>\s]+))\s*>(.*?)<\s*/\s*invoke\s*>"#,
        )
        .unwrap()
    })
}

fn parameter_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r#"(?s)<\s*parameter\s+name\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^>\s]+))\s*>(.*?)<\s*/\s*parameter\s*>"#,
        )
        .unwrap()
    })
}

fn local_schema_ref<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    let pointer = reference.strip_prefix('#')?;
    if pointer.is_empty() {
        Some(root)
    } else {
        root.pointer(pointer)
    }
}

// MiniMax emits schema-declared strings as raw XML text rather than JSON string
// literals. Mirror llama.cpp's string-schema detection before decoding other
// parameter bodies as JSON.
fn schema_resolves_to_string(
    root: &Value,
    schema: &Value,
    visited_refs: &mut HashSet<String>,
) -> bool {
    let Some(schema) = schema.as_object() else {
        return false;
    };

    if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
        && visited_refs.insert(reference.to_owned())
        && local_schema_ref(root, reference)
            .is_some_and(|resolved| schema_resolves_to_string(root, resolved, visited_refs))
    {
        return true;
    }

    match schema.get("type") {
        Some(Value::String(schema_type)) if schema_type == "string" => return true,
        Some(Value::Array(schema_types))
            if schema_types
                .iter()
                .any(|schema_type| schema_type == "string") =>
        {
            return true;
        }
        _ => {}
    }

    for keyword in ["oneOf", "anyOf"] {
        if schema
            .get(keyword)
            .and_then(Value::as_array)
            .is_some_and(|alternatives| {
                alternatives.iter().any(|alternative| {
                    schema_resolves_to_string(root, alternative, &mut visited_refs.clone())
                })
            })
        {
            return true;
        }
    }
    if schema
        .get("allOf")
        .and_then(Value::as_array)
        .is_some_and(|schemas| {
            !schemas.is_empty()
                && schemas.iter().all(|schema| {
                    schema_resolves_to_string(root, schema, &mut visited_refs.clone())
                })
        })
    {
        return true;
    }
    if schema.get("const").is_some_and(Value::is_string)
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.iter().any(Value::is_string))
        || ["pattern", "minLength", "maxLength"]
            .iter()
            .any(|keyword| schema.contains_key(*keyword))
    {
        return true;
    }
    schema
        .get("format")
        .and_then(Value::as_str)
        .is_some_and(|format| {
            matches!(
                format,
                "date"
                    | "time"
                    | "date-time"
                    | "uri"
                    | "email"
                    | "hostname"
                    | "ipv4"
                    | "ipv6"
                    | "uuid"
            ) || format.starts_with("uuid")
        })
}

fn declared_property_schema<'a>(
    root: &'a Value,
    schema: &'a Value,
    name: &str,
    visited_refs: &mut HashSet<String>,
) -> Option<&'a Value> {
    let schema = schema.as_object()?;
    if let Some(property) = schema
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(name))
    {
        return Some(property);
    }
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
        && visited_refs.insert(reference.to_owned())
        && let Some(resolved) = local_schema_ref(root, reference)
        && let Some(property) = declared_property_schema(root, resolved, name, visited_refs)
    {
        return Some(property);
    }
    for keyword in ["allOf", "oneOf", "anyOf"] {
        if let Some(schemas) = schema.get(keyword).and_then(Value::as_array) {
            for schema in schemas {
                if let Some(property) =
                    declared_property_schema(root, schema, name, &mut visited_refs.clone())
                {
                    return Some(property);
                }
            }
        }
    }
    None
}

fn additional_property_schema<'a>(
    root: &'a Value,
    schema: &'a Value,
    visited_refs: &mut HashSet<String>,
) -> Option<&'a Value> {
    let schema = schema.as_object()?;
    if let Some(additional) = schema.get("additionalProperties")
        && additional.is_object()
    {
        return Some(additional);
    }
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
        && visited_refs.insert(reference.to_owned())
        && let Some(resolved) = local_schema_ref(root, reference)
        && let Some(additional) = additional_property_schema(root, resolved, visited_refs)
    {
        return Some(additional);
    }
    for keyword in ["allOf", "oneOf", "anyOf"] {
        if let Some(schemas) = schema.get(keyword).and_then(Value::as_array) {
            for schema in schemas {
                if let Some(additional) =
                    additional_property_schema(root, schema, &mut visited_refs.clone())
                {
                    return Some(additional);
                }
            }
        }
    }
    None
}

fn raw_starts_structured_json(raw: &str) -> bool {
    matches!(raw.trim_start().chars().next(), Some('{' | '[' | '"'))
}

fn decode_parameter(definition: &ToolDefinition, name: &str, raw: &str) -> Option<Value> {
    let parameter_schema = declared_property_schema(
        &definition.parameters,
        &definition.parameters,
        name,
        &mut HashSet::new(),
    )
    .or_else(|| {
        additional_property_schema(
            &definition.parameters,
            &definition.parameters,
            &mut HashSet::new(),
        )
    });

    if parameter_schema.is_some_and(|schema| {
        schema_resolves_to_string(&definition.parameters, schema, &mut HashSet::new())
    }) {
        return Some(Value::String(raw.to_owned()));
    }
    serde_json::from_str(raw).ok().or_else(|| {
        (parameter_schema.is_none() && !raw_starts_structured_json(raw))
            .then(|| Value::String(raw.to_owned()))
    })
}

fn parse_invoke(
    invoke: &regex::Captures<'_>,
    registry: &ToolRegistry,
) -> Option<AssistantToolCall> {
    let name = invoke
        .get(1)
        .or_else(|| invoke.get(2))
        .or_else(|| invoke.get(3))?
        .as_str()
        .trim()
        .to_owned();
    let definition = registry.definitions.get(&name)?;
    let body = invoke.get(4).map_or("", |capture| capture.as_str());
    let mut arguments = Map::new();
    // Preserve source parameter order and JSON spelling so concatenated stream
    // fragments reconstruct the same arguments string as the complete parser.
    let mut encoded_arguments = String::from("{");
    let mut cursor = 0;
    for parameter in parameter_regex().captures_iter(body) {
        let whole_parameter = parameter.get(0)?;
        if !body[cursor..whole_parameter.start()].trim().is_empty() {
            return None;
        }
        cursor = whole_parameter.end();
        let parameter_name = parameter
            .get(1)
            .or_else(|| parameter.get(2))
            .or_else(|| parameter.get(3))?
            .as_str()
            .trim();
        if parameter_name.is_empty() || arguments.contains_key(parameter_name) {
            return None;
        }
        let raw = parameter.get(4).map_or("", |capture| capture.as_str());
        let value = decode_parameter(definition, parameter_name, raw)?;
        if !arguments.is_empty() {
            encoded_arguments.push(',');
        }
        encoded_arguments.push_str(
            &serde_json::to_string(parameter_name)
                .expect("serializing a parameter name cannot fail"),
        );
        encoded_arguments.push(':');
        match parameter_encoding(definition, parameter_name) {
            ParameterEncoding::String => encoded_arguments
                .push_str(&serde_json::to_string(raw).expect("serializing a string cannot fail")),
            ParameterEncoding::Json => encoded_arguments.push_str(raw),
            ParameterEncoding::Deferred if raw_starts_structured_json(raw) => {
                encoded_arguments.push_str(raw)
            }
            ParameterEncoding::Deferred => encoded_arguments.push_str(&value.to_string()),
        }
        arguments.insert(parameter_name.to_owned(), value);
    }
    if !body[cursor..].trim().is_empty() {
        return None;
    }

    encoded_arguments.push('}');
    let arguments = Value::Object(arguments);
    if !definition.validator.is_valid(&arguments) {
        return None;
    }
    Some(AssistantToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        r#type: "function",
        function: AssistantFunctionCall {
            name,
            arguments: encoded_arguments,
        },
    })
}

/// Parse only complete, schema-valid invokes. Invalid native markup remains
/// ordinary content in the caller and is never exposed as an executable call.
pub fn parse_tool_calls(xml: &str, registry: &ToolRegistry) -> Vec<AssistantToolCall> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    for invoke in invoke_regex().captures_iter(xml) {
        let Some(whole_invoke) = invoke.get(0) else {
            return Vec::new();
        };
        if !xml[cursor..whole_invoke.start()].trim().is_empty() {
            return Vec::new();
        }
        cursor = whole_invoke.end();
        if let Some(call) = parse_invoke(&invoke, registry) {
            calls.push(call);
        }
    }
    if !xml[cursor..].trim().is_empty() {
        return Vec::new();
    }
    calls
}

pub fn parse_assistant(raw: &str, registry: &ToolRegistry) -> ParsedAssistant {
    // llama.cpp's current MiniMax template analysis identifies these exact
    // wrappers: `<think>\n`, `\n</think>\n\n`, and
    // `<minimax:tool_call>\n`. Remove at most that fixed whitespace; everything
    // else belongs to the assistant payload.
    let raw = if let Some(raw) = raw.strip_prefix(THINK_START) {
        raw.strip_prefix('\n').unwrap_or(raw)
    } else {
        raw
    };
    let (reasoning, remainder, reasoning_ends_at_tool) =
        if let Some((reasoning, remainder)) = raw.split_once(THINK_END) {
            let reasoning = reasoning.strip_suffix('\n').unwrap_or(reasoning);
            let mut remainder = remainder;
            for _ in 0..2 {
                let Some(stripped) = remainder.strip_prefix('\n') else {
                    break;
                };
                remainder = stripped;
            }
            (reasoning, remainder, false)
        } else if let Some(tool_start) = raw.find(TOOL_START) {
            (&raw[..tool_start], &raw[tool_start..], true)
        } else {
            (raw, "", false)
        };

    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut cursor = 0;
    while let Some(relative_start) = remainder[cursor..].find(TOOL_START) {
        let start = cursor + relative_start;
        content.push_str(&remainder[cursor..start]);
        let body_start = start + TOOL_START.len();
        let Some(relative_end) = remainder[body_start..].find(TOOL_END) else {
            content.push_str(&remainder[start..]);
            cursor = remainder.len();
            break;
        };
        let end = body_start + relative_end;
        let block = &remainder[body_start..end];
        let parsed = parse_tool_calls(block, registry);
        if parsed.is_empty() {
            content.push_str(&remainder[start..end + TOOL_END.len()]);
        } else {
            // A single newline immediately before a valid native tool section
            // is the section separator emitted by the MiniMax template.
            if content.ends_with('\n') {
                content.pop();
            }
            tool_calls.extend(parsed);
        }
        cursor = end + TOOL_END.len();
    }
    content.push_str(&remainder[cursor..]);

    let reasoning = if reasoning_ends_at_tool && !tool_calls.is_empty() {
        reasoning.strip_suffix('\n').unwrap_or(reasoning)
    } else {
        reasoning
    };
    ParsedAssistant {
        reasoning: (!reasoning.is_empty()).then(|| reasoning.to_owned()),
        content: (!content.is_empty()).then_some(content),
        tool_calls,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MarkerIds {
    pub think_start: u32,
    pub think_end: u32,
    pub tool_start: u32,
    pub tool_end: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AssistantToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub r#type: Option<&'static str>,
    pub function: AssistantFunctionCallDelta,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AssistantFunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDelta {
    Reasoning(String),
    Content(String),
    ToolCall(AssistantToolCallDelta),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamMode {
    Reasoning,
    Content,
    Tool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolXmlState {
    BetweenInvokes,
    BetweenParameters,
    ParameterValue,
    UnknownInvoke,
    InvalidInvoke,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParameterEncoding {
    String,
    Json,
    Deferred,
}

#[derive(Debug)]
enum ParameterStreamState {
    EscapedString,
    JsonPending(String),
    JsonStructured(JsonBoundary),
    JsonScalar,
    DeferredPending(String),
    DeferredString,
    DeferredStructured(JsonBoundary),
    Invalid,
}

#[derive(Debug, Default)]
struct JsonBoundary {
    stack: Vec<char>,
    started: bool,
    root_string: bool,
    in_string: bool,
    escaped: bool,
    complete: bool,
}

impl JsonBoundary {
    fn push(&mut self, text: &str) -> Option<String> {
        let mut safe_end = 0;
        for (index, character) in text.char_indices() {
            let end = index + character.len_utf8();
            if self.complete {
                if character.is_whitespace() {
                    safe_end = end;
                    continue;
                }
                return None;
            }
            if !self.started {
                if character.is_whitespace() {
                    safe_end = end;
                    continue;
                }
                self.started = true;
                match character {
                    '{' | '[' => self.stack.push(character),
                    '"' => {
                        self.root_string = true;
                        self.in_string = true;
                    }
                    _ => return None,
                }
                safe_end = end;
                continue;
            }
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else {
                    match character {
                        '\\' => self.escaped = true,
                        '"' => {
                            self.in_string = false;
                            if self.root_string && self.stack.is_empty() {
                                self.complete = true;
                            }
                        }
                        _ => {}
                    }
                }
                safe_end = end;
                continue;
            }
            match character {
                '"' => self.in_string = true,
                '{' | '[' => self.stack.push(character),
                '}' => {
                    if self.stack.pop() != Some('{') {
                        return None;
                    }
                    if self.stack.is_empty() {
                        self.complete = true;
                    }
                }
                ']' => {
                    if self.stack.pop() != Some('[') {
                        return None;
                    }
                    if self.stack.is_empty() {
                        self.complete = true;
                    }
                }
                _ => {}
            }
            safe_end = end;
        }
        Some(text[..safe_end].to_owned())
    }

    fn is_complete(&self) -> bool {
        self.complete && !self.in_string && self.stack.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeferredClassification {
    Pending,
    String,
    Structured,
}

fn could_be_json_number_prefix(text: &str) -> bool {
    let text = text.trim_start();
    let value_end = text.find(char::is_whitespace).unwrap_or(text.len());
    if !text[value_end..].chars().all(char::is_whitespace) {
        return false;
    }
    let bytes = &text.as_bytes()[..value_end];
    let mut index = 0;
    if bytes.get(index) == Some(&b'-') {
        index += 1;
    }
    let Some(&first) = bytes.get(index) else {
        return true;
    };
    if first == b'0' {
        index += 1;
    } else if first.is_ascii_digit() && first != b'0' {
        index += 1;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
    } else {
        return false;
    }
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == bytes.len() {
            return true;
        }
    }
    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
    }
    index == bytes.len()
}

fn classify_deferred(raw: &str) -> DeferredClassification {
    let trimmed = raw.trim_start();
    let Some(first) = trimmed.chars().next() else {
        return DeferredClassification::Pending;
    };
    if matches!(first, '{' | '[' | '"') {
        return DeferredClassification::Structured;
    }
    if serde_json::from_str::<Value>(trimmed)
        .is_ok_and(|value| !matches!(value, Value::Array(_) | Value::Object(_) | Value::String(_)))
    {
        return DeferredClassification::Pending;
    }
    let literal_prefix = match first {
        't' => "true".starts_with(trimmed),
        'f' => "false".starts_with(trimmed),
        'n' => "null".starts_with(trimmed),
        '-' | '0'..='9' => could_be_json_number_prefix(trimmed),
        _ => false,
    };
    if literal_prefix {
        DeferredClassification::Pending
    } else {
        DeferredClassification::String
    }
}

impl ParameterStreamState {
    fn new(encoding: ParameterEncoding) -> Self {
        match encoding {
            ParameterEncoding::String => Self::EscapedString,
            ParameterEncoding::Json => Self::JsonPending(String::new()),
            ParameterEncoding::Deferred => Self::DeferredPending(String::new()),
        }
    }

    fn opens_string(&self) -> bool {
        matches!(self, Self::EscapedString)
    }

    fn push(&mut self, text: &str) -> Option<String> {
        let state = std::mem::replace(self, Self::Invalid);
        match state {
            Self::EscapedString => {
                *self = Self::EscapedString;
                Some(escaped_json_string_fragment(text))
            }
            Self::JsonPending(mut pending) => {
                pending.push_str(text);
                match pending.trim_start().chars().next() {
                    None => {
                        *self = Self::JsonPending(pending);
                        Some(String::new())
                    }
                    Some('{' | '[' | '"') => {
                        let mut boundary = JsonBoundary::default();
                        let output = boundary.push(&pending)?;
                        *self = Self::JsonStructured(boundary);
                        Some(output)
                    }
                    Some(_) => {
                        *self = Self::JsonScalar;
                        Some(String::new())
                    }
                }
            }
            Self::JsonStructured(mut boundary) => {
                let output = boundary.push(text)?;
                *self = Self::JsonStructured(boundary);
                Some(output)
            }
            Self::JsonScalar => {
                *self = Self::JsonScalar;
                Some(String::new())
            }
            Self::DeferredPending(mut pending) => {
                pending.push_str(text);
                match classify_deferred(&pending) {
                    DeferredClassification::Pending => {
                        *self = Self::DeferredPending(pending);
                        Some(String::new())
                    }
                    DeferredClassification::String => {
                        let output = format!("\"{}", escaped_json_string_fragment(&pending));
                        *self = Self::DeferredString;
                        Some(output)
                    }
                    DeferredClassification::Structured => {
                        let mut boundary = JsonBoundary::default();
                        let output = boundary.push(&pending)?;
                        *self = Self::DeferredStructured(boundary);
                        Some(output)
                    }
                }
            }
            Self::DeferredString => {
                *self = Self::DeferredString;
                Some(escaped_json_string_fragment(text))
            }
            Self::DeferredStructured(mut boundary) => {
                let output = boundary.push(text)?;
                *self = Self::DeferredStructured(boundary);
                Some(output)
            }
            Self::Invalid => None,
        }
    }

    fn finish(&self, raw: &str, value: &Value) -> Option<String> {
        match self {
            Self::EscapedString | Self::DeferredString => Some("\"".to_owned()),
            Self::JsonPending(_) | Self::JsonScalar => Some(raw.to_owned()),
            Self::JsonStructured(boundary) | Self::DeferredStructured(boundary)
                if boundary.is_complete() =>
            {
                Some(String::new())
            }
            Self::DeferredPending(_) => Some(value.to_string()),
            Self::JsonStructured(_) | Self::DeferredStructured(_) | Self::Invalid => None,
        }
    }
}

struct StreamingParameter {
    name: String,
    raw: String,
    stream: ParameterStreamState,
}

struct StreamingInvoke {
    index: usize,
    name: String,
    arguments: Map<String, Value>,
    parameter_names: HashSet<String>,
    parameter_count: usize,
}

struct CompletedStreamingCall {
    index: usize,
    name: String,
    arguments: Value,
}

fn find_tag_end(text: &str) -> Option<usize> {
    let mut quote = None;
    let mut awaiting_value = false;
    let mut unquoted_value = false;
    for (index, character) in text.char_indices() {
        if let Some(expected) = quote {
            if character == expected {
                quote = None;
            }
            continue;
        }
        if unquoted_value {
            if character == '>' {
                return Some(index);
            }
            continue;
        }
        if awaiting_value {
            if character.is_whitespace() {
                continue;
            }
            awaiting_value = false;
            if matches!(character, '"' | '\'') {
                quote = Some(character);
            } else if character == '>' {
                return Some(index);
            } else {
                unquoted_value = true;
            }
            continue;
        }
        match character {
            '=' => awaiting_value = true,
            '>' => return Some(index),
            _ => {}
        }
    }
    None
}

fn invoke_open_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r#"(?s)^<\s*invoke\s+name\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^>\s]+))\s*>$"#)
            .unwrap()
    })
}

fn parameter_open_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r#"(?s)^<\s*parameter\s+name\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^>\s]+))\s*>$"#)
            .unwrap()
    })
}

fn invoke_close_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?s)<\s*/\s*invoke\s*>").unwrap())
}

fn invoke_close_tag_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?s)^<\s*/\s*invoke\s*>$").unwrap())
}

fn parameter_close_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?s)<\s*/\s*parameter\s*>").unwrap())
}

fn capture_name(captures: &regex::Captures<'_>) -> Option<String> {
    captures
        .get(1)
        .or_else(|| captures.get(2))
        .or_else(|| captures.get(3))
        .map(|capture| capture.as_str().trim().to_owned())
}

fn leading_whitespace_len(text: &str) -> usize {
    text.char_indices()
        .find_map(|(index, character)| (!character.is_whitespace()).then_some(index))
        .unwrap_or(text.len())
}

fn possible_close_tag_prefix(text: &str, name: &str) -> bool {
    let mut characters = text.chars();
    if characters.next() != Some('<') {
        return false;
    }
    let mut next = characters.next();
    while next.is_some_and(char::is_whitespace) {
        next = characters.next();
    }
    let Some('/') = next else {
        return next.is_none();
    };
    next = characters.next();
    while next.is_some_and(char::is_whitespace) {
        next = characters.next();
    }
    for expected in name.chars() {
        match next {
            Some(actual) if actual == expected => next = characters.next(),
            None => return true,
            _ => return false,
        }
    }
    while next.is_some_and(char::is_whitespace) {
        next = characters.next();
    }
    next.is_none()
}

fn safe_text_before_close_prefix(text: &str, name: &str) -> usize {
    text.char_indices()
        .find_map(|(index, character)| {
            (character == '<' && possible_close_tag_prefix(&text[index..], name)).then_some(index)
        })
        .unwrap_or(text.len())
}

fn escaped_json_string_fragment(text: &str) -> String {
    let encoded = serde_json::to_string(text).expect("serializing a string cannot fail");
    encoded[1..encoded.len() - 1].to_owned()
}

fn parameter_encoding(definition: &ToolDefinition, name: &str) -> ParameterEncoding {
    let schema = declared_property_schema(
        &definition.parameters,
        &definition.parameters,
        name,
        &mut HashSet::new(),
    )
    .or_else(|| {
        additional_property_schema(
            &definition.parameters,
            &definition.parameters,
            &mut HashSet::new(),
        )
    });
    match schema {
        Some(schema)
            if schema_resolves_to_string(&definition.parameters, schema, &mut HashSet::new()) =>
        {
            ParameterEncoding::String
        }
        Some(_) => ParameterEncoding::Json,
        None => ParameterEncoding::Deferred,
    }
}

fn tool_call_header(index: usize, name: String) -> StreamDelta {
    StreamDelta::ToolCall(AssistantToolCallDelta {
        index,
        id: Some(format!("call_{}", Uuid::new_v4().simple())),
        r#type: Some("function"),
        function: AssistantFunctionCallDelta {
            name: Some(name),
            arguments: Some("{".to_owned()),
        },
    })
}

fn tool_arguments_delta(index: usize, arguments: String) -> StreamDelta {
    StreamDelta::ToolCall(AssistantToolCallDelta {
        index,
        id: None,
        r#type: None,
        function: AssistantFunctionCallDelta {
            name: None,
            arguments: Some(arguments),
        },
    })
}

pub struct ChatStreamParser {
    markers: MarkerIds,
    mode: StreamMode,
    prefix_newlines: usize,
    pending_newline: bool,
    tool_visible_prefix: Option<StreamMode>,
    tool_buffer: String,
    tool_pending: String,
    tool_state: ToolXmlState,
    active_invoke: Option<StreamingInvoke>,
    active_parameter: Option<StreamingParameter>,
    completed_calls: Vec<CompletedStreamingCall>,
    next_tool_index: usize,
    registry: ToolRegistry,
}

impl ChatStreamParser {
    pub fn new(markers: MarkerIds, registry: ToolRegistry) -> Self {
        Self {
            markers,
            mode: StreamMode::Reasoning,
            prefix_newlines: 0,
            pending_newline: false,
            tool_visible_prefix: None,
            tool_buffer: String::new(),
            tool_pending: String::new(),
            tool_state: ToolXmlState::BetweenInvokes,
            active_invoke: None,
            active_parameter: None,
            completed_calls: Vec::new(),
            next_tool_index: 0,
            registry,
        }
    }

    pub fn push(&mut self, id: u32, text: String) -> Vec<StreamDelta> {
        if id == self.markers.think_start {
            self.prefix_newlines = 1;
            return Vec::new();
        }
        if id == self.markers.think_end {
            // `\n</think>\n\n` is the fixed MiniMax reasoning delimiter.
            // Hold one reasoning newline until this marker arrives, then consume
            // at most two content-side newlines before forwarding payload text.
            self.pending_newline = false;
            self.mode = StreamMode::Content;
            self.prefix_newlines = 2;
            return Vec::new();
        }
        if id == self.markers.tool_start {
            self.start_tool_section();
            return Vec::new();
        }
        if id == self.markers.tool_end {
            return self.close_tool_section();
        }
        self.push_text(text)
    }

    fn start_tool_section(&mut self) {
        // The template places one newline between payload content and a native
        // tool section. Delay that one byte so a valid section can consume it;
        // preserve it if the section later proves invalid or unterminated.
        self.tool_visible_prefix = self.pending_newline.then_some(self.mode);
        self.pending_newline = false;
        self.prefix_newlines = 0;
        self.mode = StreamMode::Tool;
        self.tool_buffer.clear();
        self.tool_pending.clear();
        self.tool_state = ToolXmlState::BetweenInvokes;
        self.active_invoke = None;
        self.active_parameter = None;
        self.completed_calls.clear();
    }

    fn payload_delta(mode: StreamMode, text: String) -> StreamDelta {
        match mode {
            StreamMode::Reasoning => StreamDelta::Reasoning(text),
            StreamMode::Content => StreamDelta::Content(text),
            StreamMode::Tool => unreachable!("tool XML is not assistant payload text"),
        }
    }

    fn push_payload_text(&mut self, text: String) -> Vec<StreamDelta> {
        let mut text = text.as_str();
        while self.prefix_newlines > 0 {
            let Some(stripped) = text.strip_prefix('\n') else {
                if !text.is_empty() {
                    self.prefix_newlines = 0;
                }
                break;
            };
            text = stripped;
            self.prefix_newlines -= 1;
        }

        let mut output = String::new();
        if self.pending_newline {
            output.push('\n');
            self.pending_newline = false;
        }
        if let Some(stripped) = text.strip_suffix('\n') {
            output.push_str(stripped);
            self.pending_newline = true;
        } else {
            output.push_str(text);
        }
        if output.is_empty() {
            Vec::new()
        } else {
            vec![Self::payload_delta(self.mode, output)]
        }
    }

    fn fail_tool_section(&mut self) {
        self.tool_state = ToolXmlState::Failed;
        self.active_invoke = None;
        self.active_parameter = None;
        self.tool_pending.clear();
    }

    fn reject_active_invoke(&mut self) {
        self.tool_state = ToolXmlState::InvalidInvoke;
        self.active_invoke = None;
        self.active_parameter = None;
    }

    fn process_tool_text(&mut self) -> Vec<StreamDelta> {
        let mut deltas = Vec::new();
        loop {
            match self.tool_state {
                ToolXmlState::BetweenInvokes => {
                    let whitespace = leading_whitespace_len(&self.tool_pending);
                    self.tool_pending.drain(..whitespace);
                    if self.tool_pending.is_empty() {
                        break;
                    }
                    let Some(tag_end) = find_tag_end(&self.tool_pending) else {
                        break;
                    };
                    let tag = self.tool_pending[..=tag_end].to_owned();
                    let Some(name) = invoke_open_regex()
                        .captures(&tag)
                        .and_then(|captures| capture_name(&captures))
                        .filter(|name| !name.is_empty())
                    else {
                        self.fail_tool_section();
                        break;
                    };
                    self.tool_pending.drain(..=tag_end);
                    if self.registry.definitions.contains_key(&name) {
                        let index = self.next_tool_index;
                        self.next_tool_index += 1;
                        self.active_invoke = Some(StreamingInvoke {
                            index,
                            name: name.clone(),
                            arguments: Map::new(),
                            parameter_names: HashSet::new(),
                            parameter_count: 0,
                        });
                        self.tool_state = ToolXmlState::BetweenParameters;
                        deltas.push(tool_call_header(index, name));
                    } else {
                        self.tool_state = ToolXmlState::UnknownInvoke;
                    }
                }
                ToolXmlState::BetweenParameters => {
                    let whitespace = leading_whitespace_len(&self.tool_pending);
                    self.tool_pending.drain(..whitespace);
                    if self.tool_pending.is_empty() {
                        break;
                    }
                    let Some(tag_end) = find_tag_end(&self.tool_pending) else {
                        break;
                    };
                    let tag = self.tool_pending[..=tag_end].to_owned();
                    if invoke_close_tag_regex().is_match(&tag) {
                        self.tool_pending.drain(..=tag_end);
                        let Some(invoke) = self.active_invoke.take() else {
                            self.fail_tool_section();
                            break;
                        };
                        let arguments = Value::Object(invoke.arguments);
                        let Some(definition) = self.registry.definitions.get(&invoke.name) else {
                            self.fail_tool_section();
                            break;
                        };
                        if !definition.validator.is_valid(&arguments) {
                            self.tool_state = ToolXmlState::BetweenInvokes;
                            continue;
                        }
                        self.completed_calls.push(CompletedStreamingCall {
                            index: invoke.index,
                            name: invoke.name,
                            arguments,
                        });
                        self.tool_state = ToolXmlState::BetweenInvokes;
                        continue;
                    }

                    let Some(parameter_name) = parameter_open_regex()
                        .captures(&tag)
                        .and_then(|captures| capture_name(&captures))
                        .filter(|name| !name.is_empty())
                    else {
                        self.reject_active_invoke();
                        continue;
                    };
                    let Some(invoke) = self.active_invoke.as_mut() else {
                        self.fail_tool_section();
                        break;
                    };
                    if !invoke.parameter_names.insert(parameter_name.clone()) {
                        self.reject_active_invoke();
                        continue;
                    }
                    let Some(definition) = self.registry.definitions.get(&invoke.name) else {
                        self.fail_tool_section();
                        break;
                    };
                    let encoding = parameter_encoding(definition, &parameter_name);
                    let stream = ParameterStreamState::new(encoding);
                    let mut arguments = String::new();
                    if invoke.parameter_count > 0 {
                        arguments.push(',');
                    }
                    arguments.push_str(
                        &serde_json::to_string(&parameter_name)
                            .expect("serializing a parameter name cannot fail"),
                    );
                    arguments.push(':');
                    if stream.opens_string() {
                        arguments.push('"');
                    }
                    invoke.parameter_count += 1;
                    let index = invoke.index;
                    self.active_parameter = Some(StreamingParameter {
                        name: parameter_name,
                        raw: String::new(),
                        stream,
                    });
                    self.tool_pending.drain(..=tag_end);
                    self.tool_state = ToolXmlState::ParameterValue;
                    deltas.push(tool_arguments_delta(index, arguments));
                }
                ToolXmlState::ParameterValue => {
                    if let Some((value_end, close_end)) = parameter_close_regex()
                        .find(&self.tool_pending)
                        .map(|close| (close.start(), close.end()))
                    {
                        let value_text = self.tool_pending[..value_end].to_owned();
                        let stream_valid = self.stream_parameter_text(&value_text, &mut deltas);
                        self.tool_pending.drain(..close_end);
                        if !stream_valid {
                            self.reject_active_invoke();
                            continue;
                        }

                        let Some(parameter) = self.active_parameter.take() else {
                            self.fail_tool_section();
                            break;
                        };
                        let Some((index, invoke_name)) = self
                            .active_invoke
                            .as_ref()
                            .map(|invoke| (invoke.index, invoke.name.clone()))
                        else {
                            self.fail_tool_section();
                            break;
                        };
                        let Some(definition) = self.registry.definitions.get(&invoke_name) else {
                            self.fail_tool_section();
                            break;
                        };
                        let Some(value) =
                            decode_parameter(definition, &parameter.name, &parameter.raw)
                        else {
                            self.reject_active_invoke();
                            continue;
                        };
                        let Some(suffix) = parameter.stream.finish(&parameter.raw, &value) else {
                            self.reject_active_invoke();
                            continue;
                        };
                        self.active_invoke
                            .as_mut()
                            .expect("active invoke was checked above")
                            .arguments
                            .insert(parameter.name, value);
                        self.tool_state = ToolXmlState::BetweenParameters;
                        if !suffix.is_empty() {
                            deltas.push(tool_arguments_delta(index, suffix));
                        }
                        continue;
                    }

                    let safe = safe_text_before_close_prefix(&self.tool_pending, "parameter");
                    if safe == 0 {
                        break;
                    }
                    let value_text = self.tool_pending[..safe].to_owned();
                    self.tool_pending.drain(..safe);
                    if !self.stream_parameter_text(&value_text, &mut deltas) {
                        self.reject_active_invoke();
                    }
                }
                ToolXmlState::UnknownInvoke | ToolXmlState::InvalidInvoke => {
                    if let Some(close) = invoke_close_regex().find(&self.tool_pending) {
                        let close_end = close.end();
                        self.tool_pending.drain(..close_end);
                        self.tool_state = ToolXmlState::BetweenInvokes;
                        continue;
                    }
                    let safe = safe_text_before_close_prefix(&self.tool_pending, "invoke");
                    if safe == 0 {
                        break;
                    }
                    self.tool_pending.drain(..safe);
                }
                ToolXmlState::Failed => {
                    self.tool_pending.clear();
                    break;
                }
            }
        }
        deltas
    }

    fn stream_parameter_text(&mut self, text: &str, deltas: &mut Vec<StreamDelta>) -> bool {
        let Some(index) = self.active_invoke.as_ref().map(|invoke| invoke.index) else {
            self.fail_tool_section();
            return false;
        };
        let Some(parameter) = self.active_parameter.as_mut() else {
            self.fail_tool_section();
            return false;
        };
        parameter.raw.push_str(text);
        let Some(output) = parameter.stream.push(text) else {
            return false;
        };
        if !output.is_empty() {
            deltas.push(tool_arguments_delta(index, output));
        }
        true
    }

    fn completed_calls_match(&self, calls: &[AssistantToolCall]) -> bool {
        calls.len() == self.completed_calls.len()
            && calls
                .iter()
                .zip(&self.completed_calls)
                .all(|(call, streamed)| {
                    call.function.name == streamed.name
                        && serde_json::from_str::<Value>(&call.function.arguments)
                            .is_ok_and(|arguments| arguments == streamed.arguments)
                })
    }

    fn close_tool_section(&mut self) -> Vec<StreamDelta> {
        let mut deltas = self.process_tool_text();
        let buffered = std::mem::take(&mut self.tool_buffer);
        let parsed_calls = parse_tool_calls(&buffered, &self.registry);
        // The complete parser remains the final schema/markup authority. Keep
        // every streamed object open until this succeeds, so malformed output
        // can never reconstruct a complete executable arguments object.
        let valid = self.tool_state == ToolXmlState::BetweenInvokes
            && self.tool_pending.is_empty()
            && self.active_invoke.is_none()
            && self.active_parameter.is_none()
            && !parsed_calls.is_empty()
            && self.completed_calls_match(&parsed_calls);

        self.mode = StreamMode::Content;
        self.prefix_newlines = 0;
        self.pending_newline = false;
        if valid {
            self.tool_visible_prefix = None;
            for call in &self.completed_calls {
                deltas.push(tool_arguments_delta(call.index, "}".to_owned()));
            }
        } else {
            if let Some(mode) = self.tool_visible_prefix.take() {
                deltas.insert(0, Self::payload_delta(mode, "\n".to_owned()));
            }
            deltas.push(StreamDelta::Content(format!(
                "{TOOL_START}{buffered}{TOOL_END}"
            )));
        }
        self.tool_pending.clear();
        self.active_invoke = None;
        self.active_parameter = None;
        self.completed_calls.clear();
        self.tool_state = ToolXmlState::BetweenInvokes;
        deltas
    }

    /// Route decoded text that no longer represents a complete source token.
    /// Stop matching can cut through a protocol marker, in which case the
    /// surviving prefix must remain ordinary text rather than trigger it.
    pub fn push_text(&mut self, text: String) -> Vec<StreamDelta> {
        if text.is_empty() {
            return Vec::new();
        }
        match self.mode {
            StreamMode::Reasoning | StreamMode::Content => self.push_payload_text(text),
            StreamMode::Tool => {
                self.tool_buffer.push_str(&text);
                self.tool_pending.push_str(&text);
                self.process_tool_text()
            }
        }
    }

    /// Flush delimiter candidates or an unterminated tool block. Any
    /// speculative tool deltas deliberately lack the closing argument-object
    /// brace, so an invalid or length-limited block cannot reconstruct an
    /// executable call.
    pub fn finish(&mut self) -> Vec<StreamDelta> {
        match self.mode {
            mode @ (StreamMode::Reasoning | StreamMode::Content) => {
                self.prefix_newlines = 0;
                if std::mem::take(&mut self.pending_newline) {
                    vec![Self::payload_delta(mode, "\n".to_owned())]
                } else {
                    Vec::new()
                }
            }
            StreamMode::Tool => {
                self.mode = StreamMode::Content;
                let buffered = std::mem::take(&mut self.tool_buffer);
                let mut deltas = self
                    .tool_visible_prefix
                    .take()
                    .map(|mode| Self::payload_delta(mode, "\n".to_owned()))
                    .into_iter()
                    .collect::<Vec<_>>();
                deltas.push(StreamDelta::Content(format!("{TOOL_START}{buffered}")));
                self.tool_pending.clear();
                self.active_invoke = None;
                self.active_parameter = None;
                self.completed_calls.clear();
                self.tool_state = ToolXmlState::BetweenInvokes;
                deltas
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(role: &str, content: Value) -> ChatMessage {
        ChatMessage {
            role: role.to_owned(),
            content: Some(content),
            reasoning_content: None,
            reasoning: None,
            reasoning_text: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            current_date: None,
            current_location: None,
        }
    }

    fn text_message(role: &str, content: &str) -> ChatMessage {
        message(role, json!(content))
    }

    fn function_tool(name: &str) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        })
    }

    fn empty_function_tool(name: &str) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        })
    }

    fn typed_tool() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "typed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "integer": {"type": "integer"},
                        "number": {"type": "number"},
                        "boolean": {"type": "boolean"},
                        "array": {"type": "array", "items": {"type": "integer"}},
                        "object": {
                            "type": "object",
                            "properties": {"nested": {"type": "boolean"}},
                            "required": ["nested"],
                            "additionalProperties": false
                        },
                        "null": {"type": "null"}
                    },
                    "required": [
                        "text", "integer", "number", "boolean", "array", "object", "null"
                    ],
                    "additionalProperties": false
                }
            }
        })
    }

    fn registry(tools: &[Value]) -> ToolRegistry {
        ToolRegistry::from_tools(tools).expect("valid tool schemas")
    }

    fn typed_invoke() -> &'static str {
        concat!(
            "<invoke name=\"typed\">",
            "<parameter name=\"text\">  keep\n spaces  </parameter>",
            "<parameter name=\"integer\">1</parameter>",
            "<parameter name=\"number\">1.25</parameter>",
            "<parameter name=\"boolean\">true</parameter>",
            "<parameter name=\"array\">[1, 2]</parameter>",
            "<parameter name=\"object\">{\"nested\": false}</parameter>",
            "<parameter name=\"null\">null</parameter>",
            "</invoke>"
        )
    }

    fn typed_arguments() -> Value {
        json!({
            "text": "  keep\n spaces  ",
            "integer": 1,
            "number": 1.25,
            "boolean": true,
            "array": [1, 2],
            "object": {"nested": false},
            "null": null
        })
    }

    fn incoming_call(name: &str, arguments: Value) -> Value {
        json!({
            "id": "call_previous",
            "type": "function",
            "function": {"name": name, "arguments": arguments}
        })
    }

    fn parsed_arguments(call: &AssistantToolCall) -> Value {
        serde_json::from_str(&call.function.arguments).expect("valid JSON arguments")
    }

    #[derive(Default)]
    struct AccumulatedToolCall {
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    }

    fn accumulate_tool_deltas(
        calls: &mut Vec<AccumulatedToolCall>,
        deltas: impl IntoIterator<Item = StreamDelta>,
    ) {
        for delta in deltas {
            let StreamDelta::ToolCall(delta) = delta else {
                continue;
            };
            if calls.len() <= delta.index {
                calls.resize_with(delta.index + 1, AccumulatedToolCall::default);
            }
            let call = &mut calls[delta.index];
            if let Some(id) = delta.id {
                call.id = Some(id);
            }
            if let Some(name) = delta.function.name {
                call.name = Some(name);
            }
            if let Some(arguments) = delta.function.arguments {
                call.arguments.push_str(&arguments);
            }
        }
    }

    #[derive(Default)]
    struct AccumulatedAssistant {
        reasoning: String,
        content: String,
        tool_calls: Vec<AccumulatedToolCall>,
    }

    impl AccumulatedAssistant {
        fn push(&mut self, deltas: impl IntoIterator<Item = StreamDelta>) {
            for delta in deltas {
                match delta {
                    StreamDelta::Reasoning(text) => self.reasoning.push_str(&text),
                    StreamDelta::Content(text) => self.content.push_str(&text),
                    StreamDelta::ToolCall(delta) => {
                        if self.tool_calls.len() <= delta.index {
                            self.tool_calls
                                .resize_with(delta.index + 1, AccumulatedToolCall::default);
                        }
                        let call = &mut self.tool_calls[delta.index];
                        if let Some(id) = delta.id {
                            call.id = Some(id);
                        }
                        if let Some(name) = delta.function.name {
                            call.name = Some(name);
                        }
                        if let Some(arguments) = delta.function.arguments {
                            call.arguments.push_str(&arguments);
                        }
                    }
                }
            }
        }
    }

    fn push_stream_text(
        parser: &mut ChatStreamParser,
        accumulated: &mut AccumulatedAssistant,
        text: &str,
    ) {
        for character in text.chars() {
            accumulated.push(parser.push(99, character.to_string()));
        }
    }

    fn stream_assistant(raw: &str, registry: &ToolRegistry) -> AccumulatedAssistant {
        let markers = markers();
        let marker_text = [
            (THINK_START, markers.think_start),
            (THINK_END, markers.think_end),
            (TOOL_START, markers.tool_start),
            (TOOL_END, markers.tool_end),
        ];
        let mut parser = ChatStreamParser::new(markers, registry.clone());
        let mut accumulated = AccumulatedAssistant::default();
        let mut cursor = 0;
        while cursor < raw.len() {
            let next = marker_text
                .iter()
                .filter_map(|(marker, id)| {
                    raw[cursor..]
                        .find(marker)
                        .map(|offset| (cursor + offset, *id, *marker))
                })
                .min_by_key(|(offset, _, _)| *offset);
            let Some((offset, id, marker)) = next else {
                push_stream_text(&mut parser, &mut accumulated, &raw[cursor..]);
                break;
            };
            if cursor < offset {
                push_stream_text(&mut parser, &mut accumulated, &raw[cursor..offset]);
            }
            accumulated.push(parser.push(id, marker.to_owned()));
            cursor = offset + marker.len();
        }
        accumulated.push(parser.finish());
        accumulated
    }

    fn assert_stream_matches_complete(raw: &str, registry: &ToolRegistry) -> ParsedAssistant {
        let complete = parse_assistant(raw, registry);
        let streamed = stream_assistant(raw, registry);
        assert_eq!(
            streamed.reasoning,
            complete.reasoning.clone().unwrap_or_default(),
            "reasoning differed for {raw:?}"
        );
        assert_eq!(
            streamed.content,
            complete.content.clone().unwrap_or_default(),
            "content differed for {raw:?}"
        );
        assert_eq!(streamed.tool_calls.len(), complete.tool_calls.len());
        for (streamed, complete) in streamed.tool_calls.iter().zip(&complete.tool_calls) {
            assert_eq!(
                streamed.name.as_deref(),
                Some(complete.function.name.as_str())
            );
            assert_eq!(streamed.arguments, complete.function.arguments);
        }
        complete
    }

    fn markers() -> MarkerIds {
        MarkerIds {
            think_start: 10,
            think_end: 11,
            tool_start: 12,
            tool_end: 13,
        }
    }

    #[test]
    fn visible_text_normalizes_openai_content_parts() {
        let content = json!([
            "Hello",
            {"type": "text", "text": " world"},
            {"type": "image_url", "image_url": {"url": "ignored"}},
            {"type": "text", "text": 42},
            null
        ]);

        assert_eq!(visible_text(None), "");
        assert_eq!(visible_text(Some(&Value::Null)), "");
        assert_eq!(visible_text(Some(&content)), "Hello world");
        assert_eq!(
            visible_text(Some(&json!({"key": "value"}))),
            r#"{"key":"value"}"#
        );
    }

    #[test]
    fn renders_the_native_default_conversation_envelope() {
        let prompt =
            render_prompt(&[text_message("user", "Hello")], &[], false).expect("render prompt");

        assert_eq!(
            prompt,
            concat!(
                "]~!b[]~b]system\n",
                "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.",
                "[e~[\n",
                "]~b]user\nHello[e~[\n",
                "]~b]ai\n<think>\n"
            )
        );
    }

    #[test]
    fn renders_system_metadata_and_multipart_text() {
        let mut system = message(
            "system",
            json!(["Follow ", {"type": "text", "text": "policy."}]),
        );
        system.current_date = Some("2026-04-01".to_owned());
        system.current_location = Some("London".to_owned());

        let prompt = render_prompt(&[system, text_message("user", "Hi")], &[], false)
            .expect("render prompt");
        assert!(prompt.starts_with(
            "]~!b[]~b]system\nFollow policy.\nCurrent date: 2026-04-01\nCurrent location: London[e~[\n"
        ));
        assert!(prompt.contains("]~b]user\nHi[e~[\n"));
    }

    #[test]
    fn renders_native_tool_instructions() {
        let tools = vec![function_tool("read")];
        let prompt = render_prompt(&[text_message("user", "Open README.md")], &tools, false)
            .expect("render tools");

        assert!(prompt.contains("# Tools\n"));
        assert!(prompt.contains("<tools>\n<tool>"));
        assert!(prompt.contains(r#""name": "read""#));
        assert!(prompt.contains("<invoke name=\"tool-name-1\">"));
        assert!(registry(&tools).definitions.contains_key("read"));
    }

    #[test]
    fn renders_canonical_tool_prompt_like_llama_cpp() {
        let request: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/minimax-tool-prompt-request.json"
        ))
        .expect("valid OpenAI request fixture");
        let messages = serde_json::from_value::<Vec<ChatMessage>>(request["messages"].clone())
            .expect("valid messages");
        let tools = request["tools"].as_array().expect("tool array");

        let prompt = render_prompt(&messages, tools, false).expect("render prompt");

        assert_eq!(
            prompt,
            include_str!("../tests/fixtures/minimax-tool-prompt.txt")
        );
    }

    #[test]
    fn resolves_tool_choice_before_rendering_and_parsing() {
        let tools = vec![function_tool("read"), empty_function_tool("write")];

        let auto = ToolPolicy::from_request(&tools, Some(&json!("auto"))).expect("auto");
        assert_eq!(auto.tools().len(), 2);
        assert!(!auto.requires_call());

        let none = ToolPolicy::from_request(&tools, Some(&json!("none"))).expect("none");
        assert!(none.tools().is_empty());
        assert!(
            parse_tool_calls(
                "<invoke name=\"read\"><parameter name=\"path\">README.md</parameter></invoke>",
                none.registry()
            )
            .is_empty()
        );

        let required =
            ToolPolicy::from_request(&tools, Some(&json!("required"))).expect("required");
        assert!(required.requires_call());
        let prompt = render_prompt(
            &[text_message("user", "Use a tool")],
            required.tools(),
            required.requires_call(),
        )
        .expect("required prompt");
        assert!(prompt.contains("You must call one or more tools"));
        assert!(required.validate_calls(&[]).is_err());

        let explicit_choice = json!({
            "type": "function",
            "function": {"name": "write"}
        });
        let explicit = ToolPolicy::from_request(&tools, Some(&explicit_choice)).expect("explicit");
        assert_eq!(explicit.tools().len(), 1);
        assert_eq!(
            tool_function(&explicit.tools()[0]).unwrap()["name"],
            "write"
        );
        assert!(explicit.requires_call());
        let explicit_prompt = render_prompt(
            &[text_message("user", "Use the selected tool")],
            explicit.tools(),
            explicit.requires_call(),
        )
        .expect("explicit prompt");
        assert!(explicit_prompt.contains(r#""name": "write""#));
        assert!(!explicit_prompt.contains(r#""name": "read""#));

        let body = concat!(
            "<invoke name=\"read\"><parameter name=\"path\">README.md</parameter></invoke>",
            "<invoke name=\"write\"></invoke>"
        );
        let calls = parse_tool_calls(body, explicit.registry());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write");
        explicit.validate_calls(&calls).expect("selected call");

        let markers = markers();
        let mut parser = ChatStreamParser::new(markers, explicit.registry().clone());
        parser.push(markers.tool_start, TOOL_START.to_owned());
        let mut streamed = Vec::new();
        accumulate_tool_deltas(&mut streamed, parser.push(99, body.to_owned()));
        accumulate_tool_deltas(
            &mut streamed,
            parser.push(markers.tool_end, TOOL_END.to_owned()),
        );
        assert_eq!(streamed.len(), 1);
        assert_eq!(streamed[0].name.as_deref(), Some("write"));
    }

    #[test]
    fn rejects_unsatisfiable_or_invalid_tool_choices() {
        assert!(ToolPolicy::from_request(&[], Some(&json!("required"))).is_err());

        let tools = vec![function_tool("read")];
        let invalid = [
            json!("any"),
            json!(7),
            json!({}),
            json!({"type": "function"}),
            json!({"type": "function", "function": {"name": "missing"}}),
            json!({"type": "custom", "function": {"name": "read"}}),
        ];
        for choice in invalid {
            assert!(
                ToolPolicy::from_request(&tools, Some(&choice)).is_err(),
                "accepted invalid tool_choice: {choice}"
            );
        }
    }

    #[test]
    fn an_empty_tool_set_never_accepts_native_tool_markup() {
        let policy = ToolPolicy::from_request(&[], None).expect("no tools");
        let raw = concat!(
            "</think>",
            "<minimax:tool_call><invoke name=\"arbitrary\"></invoke>",
            "</minimax:tool_call>"
        );

        let parsed = parse_assistant(raw, policy.registry());
        assert!(parsed.tool_calls.is_empty());
        assert!(
            parsed
                .content
                .is_some_and(|content| content.contains("arbitrary"))
        );

        let markers = markers();
        let mut parser = ChatStreamParser::new(markers, policy.registry().clone());
        parser.push(markers.tool_start, TOOL_START.to_owned());
        let mut deltas = parser.push(99, "<invoke name=\"arbitrary\"></invoke>".to_owned());
        deltas.extend(parser.push(markers.tool_end, TOOL_END.to_owned()));
        assert!(
            !deltas
                .iter()
                .any(|delta| matches!(delta, StreamDelta::ToolCall(_)))
        );
    }

    #[test]
    fn preserves_prior_tool_exchange_but_omits_stale_reasoning() {
        let mut assistant = text_message("assistant", "");
        assistant.reasoning_content = Some("private prior reasoning".to_owned());
        assistant.tool_calls = vec![incoming_call(
            "weather",
            json!(r#"{"city":"Paris","days":2}"#),
        )];
        let mut tool_result = text_message("tool", "sunny");
        tool_result.tool_call_id = Some("call_previous".to_owned());
        let messages = vec![
            text_message("user", "Check the weather"),
            assistant,
            tool_result,
            text_message("user", "Summarize it"),
        ];

        let prompt = render_prompt(&messages, &[], false).expect("render follow-up");
        assert!(!prompt.contains("private prior reasoning"));
        assert!(prompt.contains("<invoke name=\"weather\">"));
        assert!(prompt.contains("<parameter name=\"city\">Paris</parameter>"));
        assert!(prompt.contains("<parameter name=\"days\">2</parameter>"));
        assert!(
            prompt
                .contains("]~b]tool\n<response>sunny</response>[e~[\n]~b]user\nSummarize it[e~[\n")
        );
    }

    #[test]
    fn preserves_reasoning_on_the_latest_assistant_turn() {
        let mut assistant = text_message("assistant", "visible answer");
        assistant.reasoning_content = Some("latest reasoning".to_owned());

        let prompt = render_prompt(&[text_message("user", "Question"), assistant], &[], false)
            .expect("render assistant turn");
        assert!(
            prompt.contains("]~b]ai\n<think>\nlatest reasoning\n</think>\n\nvisible answer[e~[\n")
        );
    }

    #[test]
    fn extracts_embedded_reasoning_from_prior_assistant_content() {
        let assistant = text_message(
            "assistant",
            "<think>\nembedded reasoning\n</think>\nvisible answer",
        );

        let prompt = render_prompt(&[text_message("user", "Question"), assistant], &[], false)
            .expect("render embedded reasoning");
        assert!(
            prompt
                .contains("]~b]ai\n<think>\nembedded reasoning\n</think>\n\nvisible answer[e~[\n")
        );
    }

    #[test]
    fn groups_consecutive_tool_results_in_one_native_tool_turn() {
        let mut assistant = text_message("assistant", "");
        assistant.tool_calls = vec![
            incoming_call("first", json!({})),
            incoming_call("second", json!({})),
        ];
        let messages = vec![
            text_message("user", "Run both"),
            assistant,
            text_message("tool", "result A"),
            text_message("tool", "result B"),
        ];

        let prompt = render_prompt(&messages, &[], false).expect("render tool results");
        assert_eq!(prompt.matches("]~b]tool").count(), 1);
        assert!(prompt.contains(
            "]~b]tool\n<response>result A</response>\n<response>result B</response>[e~[\n"
        ));
    }

    #[test]
    fn rejects_empty_conversations_and_orphan_tool_results() {
        assert!(
            render_prompt(&[], &[], false)
                .expect_err("empty conversation")
                .to_string()
                .contains("messages must not be empty")
        );

        let mut orphan = text_message("tool", "result");
        orphan.tool_call_id = Some("call_missing".to_owned());
        let error = render_prompt(&[text_message("user", "Hi"), orphan], &[], false)
            .expect_err("orphan tool result");
        assert!(
            error
                .to_string()
                .contains("no preceding assistant tool call")
        );
        assert!(error.to_string().contains("call_missing"));
    }

    #[test]
    fn rejects_non_object_prior_tool_arguments() {
        let mut assistant = text_message("assistant", "");
        assistant.tool_calls = vec![incoming_call("read", json!("[1, 2]"))];

        let error = render_prompt(&[text_message("user", "Hi"), assistant], &[], false)
            .expect_err("non-object arguments");
        assert!(
            error
                .to_string()
                .contains("tool-call arguments must be a JSON object")
        );
    }

    #[test]
    fn parses_xml_quote_styles_and_filters_unknown_tools() {
        let xml = r#"
            <invoke name='read'>
                <parameter name=path> README.md </parameter>
                <parameter name='options'> {"line": 1} </parameter>
            </invoke>
            <invoke name="delete"><parameter name="path">README.md</parameter></invoke>
        "#;
        let registry = registry(&[function_tool("read")]);

        let calls = parse_tool_calls(xml, &registry);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            parsed_arguments(&calls[0]),
            json!({"path": " README.md ", "options": {"line": 1}})
        );
    }

    #[test]
    fn decodes_tool_parameters_with_the_supplied_schema() {
        let registry = registry(&[typed_tool()]);

        let calls = parse_tool_calls(typed_invoke(), &registry);
        assert_eq!(calls.len(), 1);
        assert_eq!(parsed_arguments(&calls[0]), typed_arguments());

        let raw = format!("</think>{TOOL_START}{}{TOOL_END}", typed_invoke());
        let parsed = parse_assistant(&raw, &registry);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed_arguments(&parsed.tool_calls[0]), typed_arguments());
    }

    #[test]
    fn rejects_malformed_and_schema_invalid_tool_calls() {
        let registry = registry(&[typed_tool()]);
        let invalid_calls = [
            typed_invoke().replace("<parameter name=\"null\">null</parameter>", ""),
            typed_invoke().replace(
                "<parameter name=\"integer\">1</parameter>",
                "<parameter name=\"integer\">1.5</parameter>",
            ),
            typed_invoke().replace(
                "<parameter name=\"array\">[1, 2]</parameter>",
                "<parameter name=\"array\">[1,</parameter>",
            ),
            typed_invoke().replace(
                "</invoke>",
                "<parameter name=\"extra\">1</parameter></invoke>",
            ),
            typed_invoke().replace(
                "</invoke>",
                "<parameter name=\"integer\">2</parameter></invoke>",
            ),
            typed_invoke().replace("</invoke>", "<malformed></invoke>"),
        ];

        for xml in invalid_calls {
            assert!(
                parse_tool_calls(&xml, &registry).is_empty(),
                "invalid call was executable: {xml}"
            );
            let raw = format!("</think>{TOOL_START}{xml}{TOOL_END}");
            let parsed = parse_assistant(&raw, &registry);
            assert!(parsed.tool_calls.is_empty());
            assert!(parsed.content.is_some_and(|content| content.contains(&xml)));
        }
    }

    #[test]
    fn validates_schema_valued_additional_properties() {
        let tool = json!({
            "type": "function",
            "function": {
                "name": "extend",
                "parameters": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                    "required": ["label"],
                    "additionalProperties": {"type": "integer"}
                }
            }
        });
        let registry = registry(&[tool]);
        let valid = concat!(
            "<invoke name=\"extend\">",
            "<parameter name=\"label\"> exact </parameter>",
            "<parameter name=\"retries\">2</parameter>",
            "</invoke>"
        );
        let calls = parse_tool_calls(valid, &registry);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            parsed_arguments(&calls[0]),
            json!({"label": " exact ", "retries": 2})
        );

        let invalid = valid.replace(">2</parameter>", ">two</parameter>");
        assert!(parse_tool_calls(&invalid, &registry).is_empty());
    }

    #[test]
    fn preserves_payload_whitespace_after_fixed_protocol_delimiters() {
        let registry = registry(&[]);
        let fixtures = [
            (
                "plan\n</think>\n\n  leading spaces",
                Some("plan"),
                Some("  leading spaces"),
            ),
            (
                "  plan with a payload newline\n\n</think>\n\nanswer with trailing newlines\n\n",
                Some("  plan with a payload newline\n"),
                Some("answer with trailing newlines\n\n"),
            ),
            (
                concat!(
                    "<think>\n  inspect\n</think>\n\n",
                    "```rust\n",
                    "fn main() {\n",
                    "    println!(\"hello\");\n",
                    "}\n",
                    "```\n"
                ),
                Some("  inspect"),
                Some("```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n"),
            ),
        ];

        for (raw, reasoning, content) in fixtures {
            let parsed = assert_stream_matches_complete(raw, &registry);
            assert_eq!(parsed.reasoning.as_deref(), reasoning);
            assert_eq!(parsed.content.as_deref(), content);
            assert!(parsed.tool_calls.is_empty());
        }
    }

    #[test]
    fn tool_only_protocol_whitespace_is_not_visible_content() {
        let registry = registry(&[function_tool("read")]);
        let raw = concat!(
            "\n</think>\n\n",
            "<minimax:tool_call>\n",
            "<invoke name=\"read\">\n",
            "<parameter name=\"path\">README.md</parameter>\n",
            "</invoke>\n",
            "</minimax:tool_call>"
        );

        let parsed = assert_stream_matches_complete(raw, &registry);
        assert_eq!(parsed.reasoning, None);
        assert_eq!(parsed.content, None);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "read");
        assert_eq!(
            parsed_arguments(&parsed.tool_calls[0]),
            json!({"path": "README.md"})
        );
    }

    #[test]
    fn parses_reasoning_visible_content_and_tool_calls_separately() {
        let raw = concat!(
            "<think>\nInspect first.\n</think>\n",
            "Preface\n",
            "<minimax:tool_call><invoke name=\"read\">",
            "<parameter name=\"path\">README.md</parameter>",
            "</invoke></minimax:tool_call>\n",
            "Suffix"
        );
        let registry = registry(&[function_tool("read")]);

        let parsed = parse_assistant(raw, &registry);
        assert_eq!(parsed.reasoning.as_deref(), Some("Inspect first."));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "read");
        let content = parsed.content.expect("visible content");
        assert!(content.contains("Preface"));
        assert!(content.contains("Suffix"));
        assert!(!content.contains(TOOL_START));
    }

    #[test]
    fn parses_a_tool_call_even_without_a_think_end_marker() {
        let raw = concat!(
            "Need the file.\n",
            "<minimax:tool_call><invoke name=\"read\"></invoke>",
            "</minimax:tool_call>"
        );
        let registry = registry(&[empty_function_tool("read")]);

        let parsed = assert_stream_matches_complete(raw, &registry);
        assert_eq!(parsed.reasoning.as_deref(), Some("Need the file."));
        assert_eq!(parsed.content, None);
        assert_eq!(parsed.tool_calls.len(), 1);
    }

    #[test]
    fn preserves_unrecognized_tool_markup_as_visible_content() {
        let raw = concat!(
            "Plan.\n</think>\n",
            "<minimax:tool_call><invoke name=\"delete\"></invoke>",
            "</minimax:tool_call>"
        );
        let registry = registry(&[function_tool("read")]);

        let parsed = parse_assistant(raw, &registry);
        assert_eq!(parsed.reasoning.as_deref(), Some("Plan."));
        assert!(parsed.tool_calls.is_empty());
        let content = parsed.content.expect("unrecognized block remains visible");
        assert!(content.contains(TOOL_START));
        assert!(content.contains("name=\"delete\""));
        assert!(content.contains(TOOL_END));
    }

    #[test]
    fn stream_parser_routes_arbitrary_chunks_by_semantic_mode() {
        let markers = markers();
        let mut parser = ChatStreamParser::new(markers, registry(&[function_tool("read")]));

        assert!(
            parser
                .push(markers.think_start, THINK_START.into())
                .is_empty()
        );
        assert!(matches!(
            parser.push(99, "reasoning".into()).as_slice(),
            [StreamDelta::Reasoning(text)] if text == "reasoning"
        ));
        assert!(parser.push(markers.think_end, THINK_END.into()).is_empty());
        assert!(matches!(
            parser.push(99, "answer".into()).as_slice(),
            [StreamDelta::Content(text)] if text == "answer"
        ));

        assert!(
            parser
                .push(markers.tool_start, TOOL_START.into())
                .is_empty()
        );
        let mut calls = Vec::new();
        let first = parser.push(99, "<inv".to_owned());
        assert!(first.is_empty());
        accumulate_tool_deltas(&mut calls, first);
        for chunk in [
            "oke name=\"read\"><parameter name=\"path\">",
            "README.md</parameter></invoke>",
        ] {
            accumulate_tool_deltas(&mut calls, parser.push(99, chunk.to_owned()));
        }
        accumulate_tool_deltas(&mut calls, parser.push(markers.tool_end, TOOL_END.into()));
        assert_eq!(calls.len(), 1);
        assert!(
            calls[0]
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with("call_"))
        );
        assert_eq!(calls[0].name.as_deref(), Some("read"));
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].arguments).expect("complete arguments"),
            json!({"path": "README.md"})
        );

        assert!(matches!(
            parser.push(99, "after".into()).as_slice(),
            [StreamDelta::Content(text)] if text == "after"
        ));
        assert!(parser.finish().is_empty());
    }

    #[test]
    fn stream_parser_emits_tool_arguments_before_the_closing_marker() {
        let markers = markers();
        let mut parser = ChatStreamParser::new(markers, registry(&[function_tool("read")]));

        assert!(
            parser
                .push(markers.tool_start, TOOL_START.to_owned())
                .is_empty()
        );
        let deltas = parser.push(
            99,
            concat!(
                "<invoke name=\"read\"><parameter name=\"path\">",
                "a/large/argument/prefix"
            )
            .to_owned(),
        );
        assert!(
            deltas.iter().any(|delta| matches!(
                delta,
                StreamDelta::ToolCall(AssistantToolCallDelta {
                    function: AssistantFunctionCallDelta {
                        arguments: Some(arguments),
                        ..
                    },
                    ..
                }) if arguments.contains("a/large/argument/prefix")
            )),
            "tool-call output stayed buffered until the closing marker"
        );
    }

    #[test]
    fn bytewise_streaming_reconstructs_the_non_streaming_typed_call() {
        let markers = markers();
        let registry = registry(&[typed_tool()]);
        let mut parser = ChatStreamParser::new(markers, registry.clone());
        let mut streamed = Vec::new();
        let mut ordinary_content = String::new();

        parser.push(markers.tool_start, TOOL_START.to_owned());
        for byte in typed_invoke().bytes() {
            let deltas = parser.push(99, char::from(byte).to_string());
            for delta in &deltas {
                if let StreamDelta::Content(content) = delta {
                    ordinary_content.push_str(content);
                }
            }
            accumulate_tool_deltas(&mut streamed, deltas);
        }
        let deltas = parser.push(markers.tool_end, TOOL_END.to_owned());
        for delta in &deltas {
            if let StreamDelta::Content(content) = delta {
                ordinary_content.push_str(content);
            }
        }
        accumulate_tool_deltas(&mut streamed, deltas);

        let complete = parse_assistant(
            &format!("</think>{TOOL_START}{}{TOOL_END}", typed_invoke()),
            &registry,
        );
        assert!(
            ordinary_content.is_empty(),
            "valid native XML leaked as content"
        );
        assert_eq!(streamed.len(), complete.tool_calls.len());
        assert_eq!(streamed[0].name.as_deref(), Some("typed"));
        assert_eq!(
            streamed[0].name.as_deref(),
            Some(complete.tool_calls[0].function.name.as_str())
        );
        assert_eq!(
            streamed[0].arguments, complete.tool_calls[0].function.arguments,
            "streaming and non-streaming argument fragments diverged"
        );
    }

    #[test]
    fn tool_call_deltas_have_openai_streaming_shape() {
        let StreamDelta::ToolCall(header) = tool_call_header(3, "read".to_owned()) else {
            unreachable!()
        };
        let header = serde_json::to_value(header).expect("serialize header");
        assert_eq!(header["index"], 3);
        assert_eq!(header["type"], "function");
        assert!(
            header["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("call_"))
        );
        assert_eq!(header["function"]["name"], "read");
        assert_eq!(header["function"]["arguments"], "{");

        let StreamDelta::ToolCall(arguments) =
            tool_arguments_delta(3, "\"path\":\"README.md\"".to_owned())
        else {
            unreachable!()
        };
        assert_eq!(
            serde_json::to_value(arguments).expect("serialize arguments"),
            json!({
                "index": 3,
                "function": {"arguments": "\"path\":\"README.md\""}
            })
        );
    }

    #[test]
    fn stream_parser_treats_partial_marker_tokens_as_plain_text() {
        let mut parser = ChatStreamParser::new(markers(), registry(&[]));
        assert!(matches!(
            parser.push_text("<minimax:".to_owned()).as_slice(),
            [StreamDelta::Reasoning(text)] if text == "<minimax:"
        ));
        assert!(parser.finish().is_empty());
    }

    #[test]
    fn stream_parser_indexes_parallel_calls_and_reconstructs_each_one() {
        let markers = markers();
        let registry = registry(&[empty_function_tool("first"), function_tool("read")]);
        let body = concat!(
            "<invoke name=\"first\"></invoke>\n",
            "<invoke name=\"read\">",
            "<parameter name=\"path\">README.md</parameter>",
            "</invoke>"
        );
        let mut parser = ChatStreamParser::new(markers, registry.clone());
        let mut streamed = Vec::new();

        parser.push(markers.tool_start, TOOL_START.to_owned());
        accumulate_tool_deltas(&mut streamed, parser.push(99, body.to_owned()));
        accumulate_tool_deltas(
            &mut streamed,
            parser.push(markers.tool_end, TOOL_END.to_owned()),
        );

        let complete = parse_tool_calls(body, &registry);
        assert_eq!(streamed.len(), 2);
        assert_eq!(streamed.len(), complete.len());
        for (streamed, complete) in streamed.iter().zip(complete) {
            assert_eq!(
                streamed.name.as_deref(),
                Some(complete.function.name.as_str())
            );
            assert_eq!(
                streamed.arguments, complete.function.arguments,
                "streaming and non-streaming argument fragments diverged"
            );
        }
    }

    #[test]
    fn stream_parser_never_forwards_a_json_value_that_can_close_the_arguments_object() {
        let markers = markers();
        let tool = json!({
            "type": "function",
            "function": {
                "name": "integer",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                    "additionalProperties": false
                }
            }
        });
        let registry = registry(&[tool]);
        let mut parser = ChatStreamParser::new(markers, registry.clone());
        let mut streamed = Vec::new();

        parser.push(markers.tool_start, TOOL_START.to_owned());
        accumulate_tool_deltas(
            &mut streamed,
            parser.push(
                99,
                "<invoke name=\"integer\"><parameter name=\"n\">1}".to_owned(),
            ),
        );
        assert_eq!(streamed.len(), 1);
        assert!(
            serde_json::from_str::<Value>(&streamed[0].arguments).is_err(),
            "malformed native JSON closed the streamed argument object"
        );

        accumulate_tool_deltas(
            &mut streamed,
            parser.push(99, "</parameter></invoke>".to_owned()),
        );
        let closing = parser.push(markers.tool_end, TOOL_END.to_owned());
        assert!(
            closing
                .iter()
                .any(|delta| matches!(delta, StreamDelta::Content(_)))
        );
        accumulate_tool_deltas(&mut streamed, closing);
        assert!(serde_json::from_str::<Value>(&streamed[0].arguments).is_err());
        assert!(
            parse_tool_calls(
                "<invoke name=\"integer\"><parameter name=\"n\">1}</parameter></invoke>",
                &registry
            )
            .is_empty()
        );
    }

    #[test]
    fn stream_parser_rejects_bytes_after_a_json_container_root_closes() {
        let markers = markers();
        let tool = json!({
            "type": "function",
            "function": {
                "name": "object",
                "parameters": {
                    "type": "object",
                    "properties": {"payload": {"type": "object"}},
                    "required": ["payload"],
                    "additionalProperties": false
                }
            }
        });
        let registry = registry(&[tool]);
        let mut parser = ChatStreamParser::new(markers, registry);
        let mut streamed = Vec::new();

        parser.push(markers.tool_start, TOOL_START.to_owned());
        accumulate_tool_deltas(
            &mut streamed,
            parser.push(
                99,
                concat!(
                    "<invoke name=\"object\"><parameter name=\"payload\">",
                    "{\"x\":1}"
                )
                .to_owned(),
            ),
        );
        assert!(serde_json::from_str::<Value>(&streamed[0].arguments).is_err());
        assert!(parser.push(99, "}".to_owned()).is_empty());
        parser.push(99, "</parameter></invoke>".to_owned());
        let closing = parser.push(markers.tool_end, TOOL_END.to_owned());
        assert!(
            closing
                .iter()
                .any(|delta| matches!(delta, StreamDelta::Content(_)))
        );
        accumulate_tool_deltas(&mut streamed, closing);
        assert!(serde_json::from_str::<Value>(&streamed[0].arguments).is_err());
    }

    #[test]
    fn stream_parser_keeps_valid_parallel_calls_after_an_invalid_sibling() {
        let markers = markers();
        let invalid_tool = json!({
            "type": "function",
            "function": {
                "name": "integer",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                    "additionalProperties": false
                }
            }
        });
        let registry = registry(&[invalid_tool, empty_function_tool("ok")]);
        let body = concat!(
            "<invoke name=\"integer\">",
            "<parameter name=\"n\">wrong</parameter>",
            "</invoke>",
            "<invoke name=\"ok\"></invoke>"
        );
        let mut parser = ChatStreamParser::new(markers, registry.clone());
        let mut streamed = Vec::new();

        parser.push(markers.tool_start, TOOL_START.to_owned());
        accumulate_tool_deltas(&mut streamed, parser.push(99, body.to_owned()));
        let closing = parser.push(markers.tool_end, TOOL_END.to_owned());
        assert!(
            !closing
                .iter()
                .any(|delta| matches!(delta, StreamDelta::Content(_)))
        );
        accumulate_tool_deltas(&mut streamed, closing);

        let complete = parse_tool_calls(body, &registry);
        assert_eq!(complete.len(), 1);
        assert_eq!(complete[0].function.name, "ok");
        let streamed_ok = streamed
            .iter()
            .find(|call| call.name.as_deref() == Some("ok"))
            .expect("valid sibling was streamed");
        assert_eq!(streamed_ok.arguments, "{}");
    }

    #[test]
    fn stream_parser_respects_greater_than_inside_quoted_parameter_names() {
        let markers = markers();
        let tool = json!({
            "type": "function",
            "function": {
                "name": "special",
                "parameters": {
                    "type": "object",
                    "properties": {"a>b": {"type": "string"}},
                    "required": ["a>b"],
                    "additionalProperties": false
                }
            }
        });
        let registry = registry(&[tool]);
        let body = concat!(
            "<invoke name=\"special\">",
            "<parameter name=\"a>b\">value</parameter>",
            "</invoke>"
        );
        let mut parser = ChatStreamParser::new(markers, registry.clone());
        let mut streamed = Vec::new();

        parser.push(markers.tool_start, TOOL_START.to_owned());
        accumulate_tool_deltas(&mut streamed, parser.push(99, body.to_owned()));
        let closing = parser.push(markers.tool_end, TOOL_END.to_owned());
        assert!(
            !closing
                .iter()
                .any(|delta| matches!(delta, StreamDelta::Content(_)))
        );
        accumulate_tool_deltas(&mut streamed, closing);

        let complete = parse_tool_calls(body, &registry);
        assert_eq!(streamed.len(), 1);
        assert_eq!(streamed[0].arguments, complete[0].function.arguments);
        assert_eq!(
            serde_json::from_str::<Value>(&streamed[0].arguments).expect("valid arguments"),
            json!({"a>b": "value"})
        );
    }

    #[test]
    fn stream_parser_classifies_and_streams_schema_free_values() {
        let markers = markers();
        let tool = json!({
            "type": "function",
            "function": {
                "name": "dynamic",
                "parameters": {
                    "type": "object",
                    "additionalProperties": true
                }
            }
        });
        let registry = registry(&[tool]);
        let mut parser = ChatStreamParser::new(markers, registry.clone());
        let mut streamed = Vec::new();

        parser.push(markers.tool_start, TOOL_START.to_owned());
        let prefix = parser.push(
            99,
            concat!(
                "<invoke name=\"dynamic\"><parameter name=\"payload\">",
                "a large schema-free string prefix"
            )
            .to_owned(),
        );
        assert!(prefix.iter().any(|delta| matches!(
            delta,
            StreamDelta::ToolCall(AssistantToolCallDelta {
                function: AssistantFunctionCallDelta {
                    arguments: Some(arguments),
                    ..
                },
                ..
            }) if arguments.contains("a large schema-free string prefix")
        )));
        accumulate_tool_deltas(&mut streamed, prefix);
        accumulate_tool_deltas(
            &mut streamed,
            parser.push(99, "</parameter></invoke>".to_owned()),
        );
        accumulate_tool_deltas(
            &mut streamed,
            parser.push(markers.tool_end, TOOL_END.to_owned()),
        );

        let body = concat!(
            "<invoke name=\"dynamic\"><parameter name=\"payload\">",
            "a large schema-free string prefix",
            "</parameter></invoke>"
        );
        let complete = parse_tool_calls(body, &registry);
        assert_eq!(streamed[0].arguments, complete[0].function.arguments);
        assert_eq!(
            serde_json::from_str::<Value>(&streamed[0].arguments).expect("valid arguments"),
            json!({"payload": "a large schema-free string prefix"})
        );

        let mut parser = ChatStreamParser::new(markers, registry.clone());
        let mut structured = Vec::new();
        parser.push(markers.tool_start, TOOL_START.to_owned());
        let prefix = parser.push(
            99,
            concat!(
                "<invoke name=\"dynamic\"><parameter name=\"payload\">",
                "{\"items\":[1,2"
            )
            .to_owned(),
        );
        assert!(prefix.iter().any(|delta| matches!(
            delta,
            StreamDelta::ToolCall(AssistantToolCallDelta {
                function: AssistantFunctionCallDelta {
                    arguments: Some(arguments),
                    ..
                },
                ..
            }) if arguments.contains("{\"items\":[1,2")
        )));
        accumulate_tool_deltas(&mut structured, prefix);
        accumulate_tool_deltas(
            &mut structured,
            parser.push(99, ",3]}</parameter></invoke>".to_owned()),
        );
        accumulate_tool_deltas(
            &mut structured,
            parser.push(markers.tool_end, TOOL_END.to_owned()),
        );
        let structured_body = concat!(
            "<invoke name=\"dynamic\"><parameter name=\"payload\">",
            "{\"items\":[1,2,3]}",
            "</parameter></invoke>"
        );
        let complete = parse_tool_calls(structured_body, &registry);
        assert_eq!(structured[0].arguments, complete[0].function.arguments);
    }

    #[test]
    fn stream_parser_applies_tool_schemas() {
        let markers = markers();
        let registry = registry(&[typed_tool()]);
        let mut parser = ChatStreamParser::new(markers, registry.clone());

        parser.push(markers.tool_start, TOOL_START.to_owned());
        let mut calls = Vec::new();
        accumulate_tool_deltas(&mut calls, parser.push(99, typed_invoke().to_owned()));
        accumulate_tool_deltas(
            &mut calls,
            parser.push(markers.tool_end, TOOL_END.to_owned()),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_deref(), Some("typed"));
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].arguments).expect("complete arguments"),
            typed_arguments()
        );

        let invalid = typed_invoke().replace(
            "<parameter name=\"integer\">1</parameter>",
            "<parameter name=\"integer\">wrong</parameter>",
        );
        let mut parser = ChatStreamParser::new(markers, registry);
        let mut invalid_call = Vec::new();
        parser.push(markers.tool_start, TOOL_START.to_owned());
        accumulate_tool_deltas(&mut invalid_call, parser.push(99, invalid.clone()));
        let closing_deltas = parser.push(markers.tool_end, TOOL_END.to_owned());
        assert!(matches!(
            closing_deltas.as_slice(),
            [StreamDelta::Content(content)]
                if content == &format!("{TOOL_START}{invalid}{TOOL_END}")
        ));
        accumulate_tool_deltas(&mut invalid_call, closing_deltas);
        assert_eq!(invalid_call.len(), 1);
        assert!(
            serde_json::from_str::<Value>(&invalid_call[0].arguments).is_err(),
            "a schema-invalid streamed call became executable"
        );
    }

    #[test]
    fn stream_parser_preserves_invalid_and_unterminated_tool_blocks() {
        let markers = markers();
        let mut parser = ChatStreamParser::new(markers, registry(&[function_tool("read")]));
        let unknown = "<invoke name=\"delete\"></invoke>";

        parser.push(markers.tool_start, String::new());
        parser.push(99, unknown.to_owned());
        assert!(matches!(
            parser.push(markers.tool_end, String::new()).as_slice(),
            [StreamDelta::Content(text)]
                if text == &format!("{TOOL_START}{unknown}{TOOL_END}")
        ));

        parser.push(markers.tool_start, String::new());
        parser.push(99, "<invoke name=\"read\">".to_owned());
        assert!(matches!(
            parser.finish().as_slice(),
            [StreamDelta::Content(text)]
                if text == "<minimax:tool_call><invoke name=\"read\">"
        ));
        assert!(parser.finish().is_empty());
    }
}
