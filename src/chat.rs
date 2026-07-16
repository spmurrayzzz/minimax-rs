use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{collections::HashSet, fmt::Write, sync::OnceLock};
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

pub fn tool_names(tools: &[Value]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(|tool| {
            tool.get("function")?
                .get("name")?
                .as_str()
                .map(str::to_owned)
        })
        .collect()
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
pub fn render_prompt(messages: &[ChatMessage], tools: &[Value]) -> Result<String> {
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
        prompt.push_str(
            "\n\n# Tools\nYou may call one or more tools to assist with the user query.\n\
             Here are the tools available in JSONSchema format:\n\n<tools>\n",
        );
        for tool in tools {
            prompt.push_str("<tool>");
            json_jinja(tool_function(tool)?, &mut prompt)?;
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

pub fn parse_tool_calls(xml: &str, valid_names: &HashSet<String>) -> Vec<AssistantToolCall> {
    let mut calls = Vec::new();
    for invoke in invoke_regex().captures_iter(xml) {
        let name = invoke
            .get(1)
            .or_else(|| invoke.get(2))
            .or_else(|| invoke.get(3))
            .map(|capture| capture.as_str().trim().to_owned())
            .unwrap_or_default();
        if name.is_empty() || (!valid_names.is_empty() && !valid_names.contains(&name)) {
            continue;
        }
        let body = invoke.get(4).map_or("", |capture| capture.as_str());
        let mut arguments = Map::new();
        for parameter in parameter_regex().captures_iter(body) {
            let parameter_name = parameter
                .get(1)
                .or_else(|| parameter.get(2))
                .or_else(|| parameter.get(3))
                .map(|capture| capture.as_str().trim().to_owned())
                .unwrap_or_default();
            if !parameter_name.is_empty() {
                arguments.insert(
                    parameter_name,
                    Value::String(
                        parameter
                            .get(4)
                            .map_or("", |capture| capture.as_str())
                            .trim()
                            .to_owned(),
                    ),
                );
            }
        }
        calls.push(AssistantToolCall {
            id: format!("call_{}", Uuid::new_v4().simple()),
            r#type: "function",
            function: AssistantFunctionCall {
                name,
                arguments: Value::Object(arguments).to_string(),
            },
        });
    }
    calls
}

pub fn parse_assistant(raw: &str, valid_names: &HashSet<String>) -> ParsedAssistant {
    let raw = raw.strip_prefix(THINK_START).unwrap_or(raw);
    let (reasoning, remainder) = if let Some((reasoning, remainder)) = raw.split_once(THINK_END) {
        (reasoning.trim_matches('\n'), remainder)
    } else if let Some(tool_start) = raw.find(TOOL_START) {
        (raw[..tool_start].trim_matches('\n'), &raw[tool_start..])
    } else {
        (raw.trim_matches('\n'), "")
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
        let parsed = parse_tool_calls(block, valid_names);
        if parsed.is_empty() {
            content.push_str(&remainder[start..end + TOOL_END.len()]);
        } else {
            tool_calls.extend(parsed);
        }
        cursor = end + TOOL_END.len();
    }
    content.push_str(&remainder[cursor..]);

    let reasoning = (!reasoning.is_empty()).then(|| reasoning.to_owned());
    let content = content.trim_matches('\n').trim().to_owned();
    ParsedAssistant {
        reasoning,
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

#[derive(Debug, Clone)]
pub enum StreamDelta {
    Reasoning(String),
    Content(String),
    ToolCalls(Vec<AssistantToolCall>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamMode {
    Reasoning,
    Content,
    Tool,
}

pub struct ChatStreamParser {
    markers: MarkerIds,
    mode: StreamMode,
    tool_buffer: String,
    valid_names: HashSet<String>,
}

impl ChatStreamParser {
    pub fn new(markers: MarkerIds, valid_names: HashSet<String>) -> Self {
        Self {
            markers,
            mode: StreamMode::Reasoning,
            tool_buffer: String::new(),
            valid_names,
        }
    }

    pub fn push(&mut self, id: u32, text: String) -> Vec<StreamDelta> {
        if id == self.markers.think_start {
            return Vec::new();
        }
        if id == self.markers.think_end {
            self.mode = StreamMode::Content;
            return Vec::new();
        }
        if id == self.markers.tool_start {
            self.mode = StreamMode::Tool;
            self.tool_buffer.clear();
            return Vec::new();
        }
        if id == self.markers.tool_end {
            let buffered = std::mem::take(&mut self.tool_buffer);
            let calls = parse_tool_calls(&buffered, &self.valid_names);
            self.mode = StreamMode::Content;
            return if calls.is_empty() {
                vec![StreamDelta::Content(format!(
                    "{TOOL_START}{buffered}{TOOL_END}"
                ))]
            } else {
                vec![StreamDelta::ToolCalls(calls)]
            };
        }
        if text.is_empty() {
            return Vec::new();
        }
        match self.mode {
            StreamMode::Reasoning => vec![StreamDelta::Reasoning(text)],
            StreamMode::Content => vec![StreamDelta::Content(text)],
            StreamMode::Tool => {
                self.tool_buffer.push_str(&text);
                Vec::new()
            }
        }
    }

    /// Flush an unterminated tool block as visible content. This keeps a
    /// length-limited stream consistent with the non-streaming parser rather
    /// than silently dropping everything after the opening marker.
    pub fn finish(&mut self) -> Vec<StreamDelta> {
        if self.mode != StreamMode::Tool {
            return Vec::new();
        }
        self.mode = StreamMode::Content;
        let buffered = std::mem::take(&mut self.tool_buffer);
        vec![StreamDelta::Content(format!("{TOOL_START}{buffered}"))]
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
        let prompt = render_prompt(&[text_message("user", "Hello")], &[]).expect("render prompt");

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

        let prompt =
            render_prompt(&[system, text_message("user", "Hi")], &[]).expect("render prompt");
        assert!(prompt.starts_with(
            "]~!b[]~b]system\nFollow policy.\nCurrent date: 2026-04-01\nCurrent location: London[e~[\n"
        ));
        assert!(prompt.contains("]~b]user\nHi[e~[\n"));
    }

    #[test]
    fn renders_tool_definitions_without_coupling_to_json_key_order() {
        let tools = vec![function_tool("read")];
        let prompt =
            render_prompt(&[text_message("user", "Open README.md")], &tools).expect("render tools");

        assert!(prompt.contains("# Tools\n"));
        assert!(prompt.contains("<tools>\n<tool>"));
        assert!(prompt.contains(r#""name": "read""#));
        assert!(prompt.contains("<invoke name=\"tool-name-1\">"));
        assert_eq!(tool_names(&tools), HashSet::from(["read".to_owned()]));
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

        let prompt = render_prompt(&messages, &[]).expect("render follow-up");
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

        let prompt = render_prompt(&[text_message("user", "Question"), assistant], &[])
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

        let prompt = render_prompt(&[text_message("user", "Question"), assistant], &[])
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

        let prompt = render_prompt(&messages, &[]).expect("render tool results");
        assert_eq!(prompt.matches("]~b]tool").count(), 1);
        assert!(prompt.contains(
            "]~b]tool\n<response>result A</response>\n<response>result B</response>[e~[\n"
        ));
    }

    #[test]
    fn rejects_empty_conversations_and_orphan_tool_results() {
        assert!(
            render_prompt(&[], &[])
                .expect_err("empty conversation")
                .to_string()
                .contains("messages must not be empty")
        );

        let mut orphan = text_message("tool", "result");
        orphan.tool_call_id = Some("call_missing".to_owned());
        let error = render_prompt(&[text_message("user", "Hi"), orphan], &[])
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

        let error = render_prompt(&[text_message("user", "Hi"), assistant], &[])
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
        let valid_names = HashSet::from(["read".to_owned()]);

        let calls = parse_tool_calls(xml, &valid_names);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            parsed_arguments(&calls[0]),
            json!({"path": "README.md", "options": "{\"line\": 1}"})
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
        let valid_names = HashSet::from(["read".to_owned()]);

        let parsed = parse_assistant(raw, &valid_names);
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
        let valid_names = HashSet::from(["read".to_owned()]);

        let parsed = parse_assistant(raw, &valid_names);
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
        let valid_names = HashSet::from(["read".to_owned()]);

        let parsed = parse_assistant(raw, &valid_names);
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
        let mut parser = ChatStreamParser::new(markers, HashSet::from(["read".to_owned()]));

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
        for chunk in [
            "<inv",
            "oke name=\"read\"><parameter name=\"path\">",
            "README.md</parameter></invoke>",
        ] {
            assert!(parser.push(99, chunk.to_owned()).is_empty());
        }
        let deltas = parser.push(markers.tool_end, TOOL_END.into());
        let [StreamDelta::ToolCalls(calls)] = deltas.as_slice() else {
            panic!("expected a streamed tool call")
        };
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(parsed_arguments(&calls[0]), json!({"path": "README.md"}));

        assert!(matches!(
            parser.push(99, "after".into()).as_slice(),
            [StreamDelta::Content(text)] if text == "after"
        ));
        assert!(parser.finish().is_empty());
    }

    #[test]
    fn stream_parser_preserves_invalid_and_unterminated_tool_blocks() {
        let markers = markers();
        let mut parser = ChatStreamParser::new(markers, HashSet::from(["read".to_owned()]));
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
