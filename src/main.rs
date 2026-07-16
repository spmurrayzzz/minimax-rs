mod chat;
mod model;
mod tokenizer;

use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use candle_core::{Device, quantized::gguf_file};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(name = "minimax-server", about = "MiniMax-M2.7 GGUF inference server")]
struct Args {
    /// Directory containing the four split GGUF files.
    #[arg(long, default_value = "/storage/models/minimax-m2.7-gguf/UD-Q4_K_XL")]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8080")]
    host: String,
    /// Parse every GGUF shard and validate the model without starting HTTP.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine>,
}

const PREFILL_CHUNK: usize = 512;
const DEFAULT_MAX_TOKENS: usize = 4096;
const MAX_CONTEXT: usize = 196_608;

struct ModelState {
    model: model::Model,
    /// KV cache contents. A later agent/tool turn commonly extends this exact
    /// token prefix, so retain it instead of re-prefilling the whole session.
    cached_ids: Vec<u32>,
    next_logits: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationFinish {
    Eos,
    Length,
    Cancelled,
}

impl GenerationFinish {
    fn openai_reason(self) -> &'static str {
        match self {
            Self::Eos | Self::Cancelled => "stop",
            Self::Length => "length",
        }
    }
}

struct GenerationResult {
    text: String,
    token_count: usize,
    finish: GenerationFinish,
    cached_prompt_tokens: usize,
}

/// Owns the loaded execution graph and tokenizer behind the API boundary.
struct Engine {
    ready: bool,
    state: std::sync::Mutex<Option<ModelState>>,
    tokenizer: tokenizer::MiniMaxTokenizer,
}

impl Engine {
    fn open(model_dir: &Path, load_model: bool) -> Result<Self> {
        let mut shards: Vec<_> = std::fs::read_dir(model_dir)
            .with_context(|| format!("cannot read model directory {}", model_dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
            .collect();
        shards.sort();
        if shards.len() != 4 {
            bail!(
                "expected 4 split GGUF files in {}, found {}",
                model_dir.display(),
                shards.len()
            );
        }

        let mut tensors = 0usize;
        for (i, path) in shards.iter().enumerate() {
            let mut file = std::fs::File::open(path)?;
            let content = gguf_file::Content::read(&mut file)
                .with_context(|| format!("invalid GGUF shard {}", path.display()))?;
            tensors += content.tensor_infos.len();
            info!(shard = i + 1, tensors = content.tensor_infos.len(), path = %path.display(), "validated GGUF shard");
        }
        if tensors < 800 {
            bail!("model appears incomplete: only {tensors} tensors found");
        }

        let devices = (0..2)
            .map(Device::new_cuda)
            .collect::<candle_core::Result<Vec<_>>>()?;
        let tokenizer = tokenizer::MiniMaxTokenizer::from_gguf(&shards[0])?;
        let loaded = if load_model {
            Some(ModelState {
                model: model::Model::load(&shards, &devices)?,
                cached_ids: Vec::new(),
                next_logits: Vec::new(),
            })
        } else {
            None
        };
        Ok(Self {
            ready: load_model,
            state: std::sync::Mutex::new(loaded),
            tokenizer,
        })
    }

    fn marker_ids(&self) -> chat::MarkerIds {
        chat::MarkerIds {
            think_start: self.tokenizer.think_start,
            think_end: self.tokenizer.think_end,
            tool_start: self.tokenizer.tool_start,
            tool_end: self.tokenizer.tool_end,
        }
    }

    /// Generate tokens while invoking `on_token` immediately after selecting
    /// each token and before evaluating the next one. Returning false cancels
    /// generation (normally because the streaming client disconnected).
    fn generate_with<F>(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        _temperature: f32,
        mut on_token: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(u32, String) -> bool,
    {
        if prompt.is_empty() {
            bail!("prompt must contain at least one token id")
        }
        if prompt.len() >= MAX_CONTEXT {
            bail!(
                "prompt has {} tokens, context limit is {MAX_CONTEXT}",
                prompt.len()
            )
        }
        let max_tokens = max_tokens.min(MAX_CONTEXT - prompt.len());
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model is not loaded"))?;

        let reusable = !state.cached_ids.is_empty()
            && prompt.len() >= state.cached_ids.len()
            && prompt.starts_with(&state.cached_ids)
            && !state.next_logits.is_empty();
        let cached_prompt_tokens = if reusable { state.cached_ids.len() } else { 0 };
        if !reusable {
            state.model.reset();
            state.cached_ids.clear();
            state.next_logits.clear();
        }
        debug!(
            prompt_tokens = prompt.len(),
            cached_prompt_tokens, "starting generation"
        );

        let mut next = state.next_logits.clone();
        for chunk in prompt[state.cached_ids.len()..].chunks(PREFILL_CHUNK) {
            let offset = state.cached_ids.len();
            next = state.model.forward(chunk, offset)?.to_vec1::<f32>()?;
            state.cached_ids.extend_from_slice(chunk);
        }
        if next.is_empty() {
            bail!("no logits available for prompt")
        }
        state.next_logits = next.clone();

        let mut generated = Vec::new();
        let mut emitted = String::new();
        let mut finish = GenerationFinish::Length;
        for _ in 0..max_tokens {
            let (id, _) = next
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .context("empty logits")?;
            let id = id as u32;
            if id == self.tokenizer.eos {
                finish = GenerationFinish::Eos;
                break;
            }
            generated.push(id);

            // Decode cumulatively so byte-level tokens split across a Unicode
            // scalar are not independently turned into replacement characters.
            let decoded = self.tokenizer.decode(&generated)?;
            let stable = decoded.trim_end_matches('\u{fffd}');
            let delta = stable.strip_prefix(&emitted).unwrap_or_default().to_owned();
            emitted = stable.to_owned();
            if !on_token(id, delta) {
                finish = GenerationFinish::Cancelled;
                break;
            }

            let offset = state.cached_ids.len();
            next = state.model.forward(&[id], offset)?.to_vec1::<f32>()?;
            state.cached_ids.push(id);
            state.next_logits = next.clone();
        }

        Ok(GenerationResult {
            text: self.tokenizer.decode(&generated)?,
            token_count: generated.len(),
            finish,
            cached_prompt_tokens,
        })
    }
}

#[derive(Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: Prompt,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
}
fn default_temperature() -> f32 {
    1.0
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    Batch(Vec<Prompt>),
}

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<chat::ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
}

impl ChatRequest {
    fn output_limit(&self) -> usize {
        self.max_tokens
            .or(self.max_completion_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS)
    }
}
#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}
#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatOutMessage,
    finish_reason: &'static str,
}
#[derive(Serialize)]
struct ChatOutMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<chat::AssistantToolCall>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}
#[derive(Serialize)]
struct Choice {
    text: String,
    index: usize,
    logprobs: Option<()>,
    finish_reason: &'static str,
}
#[derive(Clone, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<PromptTokensDetails>,
}
#[derive(Clone, Serialize)]
struct PromptTokensDetails {
    cached_tokens: usize,
}
#[derive(Serialize)]
struct ErrorResponse {
    error: ApiError,
}
#[derive(Serialize)]
struct ApiError {
    message: String,
    r#type: &'static str,
    code: Option<&'static str>,
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status":"ok"}))
}
async fn models(State(state): State<AppState>) -> impl IntoResponse {
    Json(
        serde_json::json!({"object":"list","data":[{"id":"MiniMax-M2.7","object":"model","owned_by":"MiniMax","ready":state.engine.ready}]}),
    )
}

fn created_at() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn generation_usage(prompt_tokens: usize, generation: &GenerationResult) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens: generation.token_count,
        total_tokens: prompt_tokens + generation.token_count,
        prompt_tokens_details: Some(PromptTokensDetails {
            cached_tokens: generation.cached_prompt_tokens,
        }),
    }
}

fn finish_reason(generation: &GenerationResult, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        "tool_calls"
    } else {
        generation.finish.openai_reason()
    }
}

fn sse(value: &serde_json::Value) -> String {
    format!("data: {value}\n\n")
}

fn send_sse(
    sender: &mpsc::Sender<std::result::Result<String, Infallible>>,
    value: serde_json::Value,
) -> bool {
    sender.blocking_send(Ok(sse(&value))).is_ok()
}

fn stream_response(
    receiver: mpsc::Receiver<std::result::Result<String, Infallible>>,
) -> axum::response::Response {
    axum::response::Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(ReceiverStream::new(receiver)))
        .unwrap()
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    let prompts = match req.prompt {
        Prompt::Text(text) => match state.engine.tokenizer.encode(&text) {
            Ok(ids) => vec![ids],
            Err(error) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    &format!("tokenization failed: {error}"),
                );
            }
        },
        Prompt::Tokens(tokens) => vec![tokens],
        Prompt::Batch(batch) => batch
            .into_iter()
            .filter_map(|prompt| match prompt {
                Prompt::Text(text) => state.engine.tokenizer.encode(&text).ok(),
                Prompt::Tokens(tokens) => Some(tokens),
                Prompt::Batch(_) => None,
            })
            .collect(),
    };
    let Some(prompt) = prompts.into_iter().next() else {
        return api_error(StatusCode::BAD_REQUEST, "prompt must not be empty");
    };
    if prompt.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "prompt must not be empty");
    }
    let prompt_tokens = prompt.len();
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature = req.temperature;
    let model_name = req.model.unwrap_or_else(|| "MiniMax-M2.7".into());
    let id = format!("cmpl-{}", Uuid::new_v4());
    let created = created_at();

    if req.stream {
        let (sender, receiver) = mpsc::channel(64);
        let engine = state.engine.clone();
        let stream_id = id.clone();
        let stream_model = model_name.clone();
        tokio::task::spawn_blocking(move || {
            let token_sender = sender.clone();
            let generation =
                engine.generate_with(&prompt, max_tokens, temperature, move |_token, text| {
                    text.is_empty()
                        || send_sse(
                            &token_sender,
                            serde_json::json!({
                                "id": stream_id,
                                "object": "text_completion",
                                "created": created,
                                "model": stream_model,
                                "choices": [{
                                    "text": text,
                                    "index": 0,
                                    "logprobs": null,
                                    "finish_reason": null
                                }]
                            }),
                        )
                });
            match generation {
                Ok(generation) => {
                    let final_chunk = serde_json::json!({
                        "id": id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "text": "",
                            "index": 0,
                            "logprobs": null,
                            "finish_reason": generation.finish.openai_reason()
                        }],
                        "usage": generation_usage(prompt_tokens, &generation)
                    });
                    let _ = send_sse(&sender, final_chunk);
                }
                Err(error) => {
                    let _ = send_sse(
                        &sender,
                        serde_json::json!({"error": {"message": error.to_string(), "type": "server_error"}}),
                    );
                }
            }
            let _ = sender.blocking_send(Ok("data: [DONE]\n\n".into()));
        });
        return stream_response(receiver);
    }

    let engine = state.engine.clone();
    let generation = match tokio::task::spawn_blocking(move || {
        engine.generate_with(&prompt, max_tokens, temperature, |_token, _text| true)
    })
    .await
    {
        Ok(Ok(generation)) => generation,
        Ok(Err(error)) => return api_error(StatusCode::SERVICE_UNAVAILABLE, &error.to_string()),
        Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string()),
    };
    let usage = generation_usage(prompt_tokens, &generation);
    Json(CompletionResponse {
        id,
        object: "text_completion",
        created,
        model: model_name,
        choices: vec![Choice {
            text: generation.text,
            index: 0,
            logprobs: None,
            finish_reason: generation.finish.openai_reason(),
        }],
        usage,
    })
    .into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> axum::response::Response {
    if req.messages.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let max_tokens = req.output_limit();
    let tools = if req
        .tool_choice
        .as_ref()
        .is_some_and(|choice| choice == "none")
    {
        Vec::new()
    } else {
        req.tools.clone()
    };
    let valid_tool_names = chat::tool_names(&tools);
    let prompt = match chat::render_prompt(&req.messages, &tools) {
        Ok(prompt) => prompt,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let ids = match state.engine.tokenizer.encode(&prompt) {
        Ok(ids) => ids,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                &format!("tokenization failed: {error}"),
            );
        }
    };
    let prompt_tokens = ids.len();
    let temperature = req.temperature;
    let model_name = req.model.unwrap_or_else(|| "MiniMax-M2.7".into());
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = created_at();

    if req.stream {
        let (sender, receiver) = mpsc::channel(128);
        let initial = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": null
            }]
        });
        let _ = sender.try_send(Ok(sse(&initial)));

        let engine = state.engine.clone();
        let markers = engine.marker_ids();
        let stream_id = id.clone();
        let stream_model = model_name.clone();
        tokio::task::spawn_blocking(move || {
            let mut parser = chat::ChatStreamParser::new(markers, valid_tool_names.clone());
            let mut streamed_tool_calls = false;
            let token_sender = sender.clone();
            let generation = engine.generate_with(&ids, max_tokens, temperature, |token, text| {
                for delta in parser.push(token, text) {
                    let delta = match delta {
                        chat::StreamDelta::Reasoning(reasoning) => {
                            serde_json::json!({"reasoning_content": reasoning})
                        }
                        chat::StreamDelta::Content(content) => {
                            serde_json::json!({"content": content})
                        }
                        chat::StreamDelta::ToolCalls(calls) => {
                            streamed_tool_calls = true;
                            let calls = calls
                                .into_iter()
                                .enumerate()
                                .map(|(index, call)| {
                                    serde_json::json!({
                                        "index": index,
                                        "id": call.id,
                                        "type": call.r#type,
                                        "function": call.function
                                    })
                                })
                                .collect::<Vec<_>>();
                            serde_json::json!({"tool_calls": calls})
                        }
                    };
                    if !send_sse(
                        &token_sender,
                        serde_json::json!({
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": stream_model,
                            "choices": [{
                                "index": 0,
                                "delta": delta,
                                "finish_reason": null
                            }]
                        }),
                    ) {
                        return false;
                    }
                }
                true
            });
            match generation {
                Ok(generation) => {
                    let parsed = chat::parse_assistant(&generation.text, &valid_tool_names);
                    // If an unusual token split prevented the incremental parser
                    // from emitting a completed call, emit it once before finish.
                    if !streamed_tool_calls && !parsed.tool_calls.is_empty() {
                        let calls = parsed
                            .tool_calls
                            .iter()
                            .cloned()
                            .enumerate()
                            .map(|(index, call)| {
                                serde_json::json!({
                                    "index": index,
                                    "id": call.id,
                                    "type": call.r#type,
                                    "function": call.function
                                })
                            })
                            .collect::<Vec<_>>();
                        let _ = send_sse(
                            &sender,
                            serde_json::json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"tool_calls": calls},
                                    "finish_reason": null
                                }]
                            }),
                        );
                    }
                    let reason = finish_reason(&generation, !parsed.tool_calls.is_empty());
                    let _ = send_sse(
                        &sender,
                        serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": reason
                            }],
                            "usage": generation_usage(prompt_tokens, &generation)
                        }),
                    );
                }
                Err(error) => {
                    let _ = send_sse(
                        &sender,
                        serde_json::json!({"error": {"message": error.to_string(), "type": "server_error"}}),
                    );
                }
            }
            let _ = sender.blocking_send(Ok("data: [DONE]\n\n".into()));
        });
        return stream_response(receiver);
    }

    let engine = state.engine.clone();
    let generation = match tokio::task::spawn_blocking(move || {
        engine.generate_with(&ids, max_tokens, temperature, |_token, _text| true)
    })
    .await
    {
        Ok(Ok(generation)) => generation,
        Ok(Err(error)) => return api_error(StatusCode::SERVICE_UNAVAILABLE, &error.to_string()),
        Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string()),
    };
    let parsed = chat::parse_assistant(&generation.text, &valid_tool_names);
    let reason = finish_reason(&generation, !parsed.tool_calls.is_empty());
    let usage = generation_usage(prompt_tokens, &generation);
    Json(ChatResponse {
        id,
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatOutMessage {
                role: "assistant",
                content: parsed.content,
                reasoning_content: parsed.reasoning,
                tool_calls: parsed.tool_calls,
            },
            finish_reason: reason,
        }],
        usage,
    })
    .into_response()
}

fn api_error(status: StatusCode, message: &str) -> axum::response::Response {
    (
        status,
        Json(ErrorResponse {
            error: ApiError {
                message: message.into(),
                r#type: "server_error",
                code: None,
            },
        }),
    )
        .into_response()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::parse();
    let engine = Arc::new(Engine::open(&args.model, !args.dry_run)?);
    if args.dry_run {
        println!("validated 4 GGUF shards for MiniMax-M2.7 on 2 CUDA devices");
        return Ok(());
    }
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(AppState { engine });
    let listener = TcpListener::bind(&args.host).await?;
    info!(address = %args.host, "MiniMax server listening");
    axum::serve(listener, app).await?;
    Ok(())
}
