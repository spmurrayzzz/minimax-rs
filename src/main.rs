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
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::net::TcpListener;
use tracing::info;
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

/// Owns the loaded execution graph and tokenizer behind the API boundary.
struct Engine {
    ready: bool,
    model: std::sync::Mutex<Option<model::Model>>,
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

        // Candle's CUDA backend selects the compiled kernel for the actual device. In
        // particular, CUDA builds must be made with a CUDA toolkit supporting sm_120.
        let devices = (0..2)
            .map(|ordinal| Device::new_cuda(ordinal))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let tokenizer = tokenizer::MiniMaxTokenizer::from_gguf(&shards[0])?;
        let loaded = if load_model {
            Some(model::Model::load(&shards, &devices)?)
        } else {
            None
        };
        Ok(Self {
            ready: load_model,
            model: std::sync::Mutex::new(loaded),
            tokenizer,
        })
    }

    fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<(String, usize)> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        let model = model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model is not loaded"))?;
        if prompt.is_empty() {
            bail!("prompt must contain at least one token id")
        }
        model.reset();
        // Prefill in parallel chunks. This keeps prompt processing parallel
        // while avoiding oversized MoE kernel launches for agent prompts.
        // llama.cpp defaults: logical n_batch=2048, physical n_ubatch=512.
        // This is the physical chunk used by its prompt decode loop.
        const PREFILL_CHUNK: usize = 512;
        let mut next = Vec::new();
        for (pos, chunk) in prompt.chunks(PREFILL_CHUNK).enumerate() {
            let offset = pos * PREFILL_CHUNK;
            next = model.forward(chunk, offset)?.to_vec1::<f32>()?;
        }
        let mut generated = Vec::new();
        for pos in 0..max_tokens {
            let (id, _) = next
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            if id as u32 == self.tokenizer.eos {
                break;
            }
            generated.push(id as u32);
            next = model
                .forward(&[id as u32], prompt.len() + pos)?
                .to_vec1::<f32>()?;
        }
        Ok((self.tokenizer.decode(&generated)?, generated.len()))
    }
}

#[derive(Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: Prompt,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
}
fn default_max_tokens() -> usize {
    16
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
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
}
#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    // OpenAI clients may send either a string or an array of content parts.
    content: serde_json::Value,
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
    content: String,
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
#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
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

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    if req.stream {
        return error(StatusCode::NOT_IMPLEMENTED, "streaming is not implemented");
    }
    let prompts = match req.prompt {
        Prompt::Text(text) => match state.engine.tokenizer.encode(&text) {
            Ok(ids) => vec![ids],
            Err(e) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    &format!("tokenization failed: {e}"),
                );
            }
        },
        Prompt::Tokens(t) => vec![t],
        Prompt::Batch(batch) => batch
            .into_iter()
            .filter_map(|p| match p {
                Prompt::Text(text) => state.engine.tokenizer.encode(&text).ok(),
                Prompt::Tokens(t) => Some(t),
                Prompt::Batch(_) => None,
            })
            .collect(),
    };
    if prompts.is_empty() {
        return error(StatusCode::BAD_REQUEST, "prompt must not be empty");
    }
    let result = match state
        .engine
        .generate(&prompts[0], req.max_tokens, req.temperature)
    {
        Ok(result) => result,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()),
    };
    let model = req.model.unwrap_or_else(|| "MiniMax-M2.7".into());
    let (text, completion_tokens) = result;
    let n = prompts[0].len();
    Json(CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion",
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model,
        choices: vec![Choice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens: n,
            completion_tokens,
            total_tokens: n + completion_tokens,
        },
    })
    .into_response()
}
async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> axum::response::Response {
    if req.messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    // MiniMax-M2.7's GGUF chat template. Rendering only `role: content`
    // produces valid tokens but not a prompt the instruct model was trained on.
    let content = |value: &serde_json::Value| match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect::<Vec<_>>()
            .join(""),
        other => other.to_string(),
    };
    let mut prompt = String::from("]~!b[]~b]system\n");
    if let Some(system) = req.messages.iter().find(|m| m.role == "system") {
        prompt.push_str(&content(&system.content));
    } else {
        prompt.push_str(
            "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.",
        );
    }
    prompt.push_str("[e~[\n");
    for message in req.messages.iter().filter(|m| m.role != "system") {
        prompt.push_str(if message.role == "assistant" {
            "]~b]ai\n"
        } else {
            "]~b]user\n"
        });
        prompt.push_str(&content(&message.content));
        prompt.push_str("[e~[\n");
    }
    prompt.push_str("]~b]ai\n<think>\n");
    let ids = match state.engine.tokenizer.encode(&prompt) {
        Ok(ids) => ids,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                &format!("tokenization failed: {e}"),
            );
        }
    };
    let (text, completion_tokens) =
        match state.engine.generate(&ids, req.max_tokens, req.temperature) {
            Ok(result) => result,
            Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()),
        };
    let model = req.model.unwrap_or_else(|| "MiniMax-M2.7".into());
    let prompt_tokens = ids.len();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    if req.stream {
        let first = serde_json::json!({"id": id, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": null}]});
        let last = serde_json::json!({"id": id, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]});
        return axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(Body::from(format!(
                "data: {first}\n\ndata: {last}\n\ndata: [DONE]\n\n"
            )))
            .unwrap();
    }
    Json(ChatResponse {
        id,
        object: "chat.completion",
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatOutMessage {
                role: "assistant",
                content: text,
            },
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}

fn error(status: StatusCode, message: &str) -> axum::response::Response {
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
