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
use candle_core::{D, Device, Tensor, quantized::gguf_file};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(name = "minimax-server", about = "MiniMax-M2.7 GGUF inference server")]
struct Args {
    /// Directory containing the four split GGUF files.
    #[arg(long, env = "MINIMAX_MODEL_DIR", value_name = "DIR")]
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
    next_logits: Option<Tensor>,
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

struct InferenceMetrics {
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    prefill_tokens: usize,
    prefill_time: Duration,
    decode_tokens: usize,
    decode_time: Duration,
}

impl InferenceMetrics {
    fn log(&self, request_id: &str, finish: GenerationFinish) {
        let total_tokens = self.prefill_tokens + self.decode_tokens;
        let total_time = self.prefill_time + self.decode_time;
        info!(
            request_id,
            prompt_tokens = self.prompt_tokens,
            cached_prompt_tokens = self.cached_prompt_tokens,
            prefill_tokens = self.prefill_tokens,
            prefill_time_ms = rounded_metric(duration_ms(self.prefill_time)),
            prefill_ms_per_token = rounded_metric(milliseconds_per_token(
                self.prefill_time,
                self.prefill_tokens,
            )),
            prefill_tokens_per_second = rounded_metric(tokens_per_second(
                self.prefill_time,
                self.prefill_tokens,
            )),
            decode_tokens = self.decode_tokens,
            decode_time_ms = rounded_metric(duration_ms(self.decode_time)),
            decode_ms_per_token = rounded_metric(milliseconds_per_token(
                self.decode_time,
                self.decode_tokens,
            )),
            decode_tokens_per_second = rounded_metric(tokens_per_second(
                self.decode_time,
                self.decode_tokens,
            )),
            total_tokens,
            total_time_ms = rounded_metric(duration_ms(total_time)),
            finish = ?finish,
            "inference timings"
        );
    }
}

fn rounded_metric(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn milliseconds_per_token(duration: Duration, tokens: usize) -> f64 {
    if tokens == 0 {
        0.0
    } else {
        duration_ms(duration) / tokens as f64
    }
}

fn tokens_per_second(duration: Duration, tokens: usize) -> f64 {
    if tokens == 0 || duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

/// Owns the loaded execution graph and tokenizer behind the API boundary.
struct Engine {
    ready: bool,
    state: std::sync::Mutex<Option<ModelState>>,
    tokenizer: tokenizer::MiniMaxTokenizer,
}

impl Engine {
    fn open(model_dir: &Path, load_model: bool) -> Result<Self> {
        let shards = minimax::model_files::discover_gguf_shards(model_dir)?;

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
        // Inference uses exactly one CUDA stream per GPU and synchronizes the
        // host-staged device boundary explicitly. Avoid cudarc's per-allocation
        // cross-stream events, which otherwise dominate single-token decode.
        for device in &devices {
            unsafe { device.as_cuda_device()?.disable_event_tracking() };
        }
        let tokenizer = tokenizer::MiniMaxTokenizer::from_gguf(&shards[0])?;
        let loaded = if load_model {
            Some(ModelState {
                model: model::Model::load(&shards, &devices)?,
                cached_ids: Vec::new(),
                next_logits: None,
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
        request_id: &str,
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
            && state.next_logits.is_some();
        let cached_prompt_tokens = if reusable { state.cached_ids.len() } else { 0 };
        if !reusable {
            state.model.reset();
            state.cached_ids.clear();
            state.next_logits = None;
        }
        debug!(
            request_id,
            prompt_tokens = prompt.len(),
            cached_prompt_tokens,
            "starting generation"
        );

        // CUDA launches are asynchronous, so synchronize the final prompt
        // logits before recording prefill time. The first argmax would impose
        // the same dependency; doing it here gives prefill and decode cleanly
        // separated timings.
        let prefill_tokens = prompt.len() - state.cached_ids.len();
        let prefill_started = Instant::now();
        let mut next = state.next_logits.clone();
        for chunk in prompt[state.cached_ids.len()..].chunks(PREFILL_CHUNK) {
            let offset = state.cached_ids.len();
            next = Some(state.model.forward(chunk, offset)?);
            state.cached_ids.extend_from_slice(chunk);
        }
        let mut next = next.context("no logits available for prompt")?;
        if prefill_tokens > 0 {
            next.device().synchronize()?;
        }
        let prefill_time = if prefill_tokens > 0 {
            prefill_started.elapsed()
        } else {
            Duration::ZERO
        };
        state.next_logits = Some(next.clone());

        let decode_started = Instant::now();
        let mut generated = Vec::new();
        let mut decode_state = tokenizer::DecodeState::default();
        let mut finish = GenerationFinish::Length;
        let mut logits_pending = false;
        for _ in 0..max_tokens {
            // Reduce the 200k vocabulary on-device; copying every logit over
            // PCIe and scanning it on the CPU costs roughly 0.35 ms/token.
            let id = next.argmax(D::Minus1)?.to_scalar::<u32>()?;
            logits_pending = false;
            if id == self.tokenizer.eos {
                finish = GenerationFinish::Eos;
                break;
            }
            generated.push(id);

            // DecodeStream retains only the short byte/prefix window needed
            // for UTF-8 boundaries instead of decoding the full history.
            let delta = self
                .tokenizer
                .decode_step(&mut decode_state, id)?
                .unwrap_or_default();
            if !on_token(id, delta) {
                finish = GenerationFinish::Cancelled;
                break;
            }

            let offset = state.cached_ids.len();
            next = state.model.forward(&[id], offset)?;
            logits_pending = true;
            state.cached_ids.push(id);
            state.next_logits = Some(next.clone());
        }
        // Usually the following iteration's scalar argmax synchronizes each
        // decode step. A length-limited request leaves its last logits pending,
        // so wait for them before calculating throughput and releasing the
        // cache to the next request.
        if logits_pending {
            next.device().synchronize()?;
        }
        let decode_time = decode_started.elapsed();

        let result = GenerationResult {
            text: self.tokenizer.decode(&generated)?,
            token_count: generated.len(),
            finish,
            cached_prompt_tokens,
        };
        InferenceMetrics {
            prompt_tokens: prompt.len(),
            cached_prompt_tokens,
            prefill_tokens,
            prefill_time,
            decode_tokens: result.token_count,
            decode_time,
        }
        .log(request_id, finish);
        Ok(result)
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
            let generation = engine.generate_with(
                &id,
                &prompt,
                max_tokens,
                temperature,
                move |_token, text| {
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
                },
            );
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
    let request_id = id.clone();
    let generation = match tokio::task::spawn_blocking(move || {
        engine.generate_with(
            &request_id,
            &prompt,
            max_tokens,
            temperature,
            |_token, _text| true,
        )
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
            let generation =
                engine.generate_with(&id, &ids, max_tokens, temperature, |token, text| {
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
    let request_id = id.clone();
    let generation = match tokio::task::spawn_blocking(move || {
        engine.generate_with(
            &request_id,
            &ids,
            max_tokens,
            temperature,
            |_token, _text| true,
        )
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
