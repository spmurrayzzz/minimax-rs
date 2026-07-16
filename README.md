# minimax

Rust/Candle CUDA inference server for MiniMax-M2.7 split GGUF weights.

## Run

```bash
CUDA_COMPUTE_CAP=120f cargo run --release --features cuda -- \
  --model /storage/models/minimax-m2.7-gguf/UD-Q4_K_XL \
  --host 0.0.0.0:8000
```

Weights load across two GPUs and startup may take several minutes. Tokenization is reconstructed from the GGUF metadata; no external tokenizer file is used.

## Test

```bash
BASE_URL=http://127.0.0.1:8000 ./scripts/test-completion.sh
```

Endpoints: `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`.

Chat completions support true token-by-token SSE streaming, `reasoning_content`, OpenAI function tools, MiniMax XML tool-call parsing, prior assistant/tool turns, and both `max_tokens` and `max_completion_tokens`. The default output limit is 4096 tokens. Exact-prefix KV reuse accelerates subsequent agent/tool turns and is reported as `usage.prompt_tokens_details.cached_tokens`.

The implementation uses 512-token parallel prefill chunks. The custom WMMA MoE prefill path is disabled because it is numerically unstable across expert segments on `sm_120`; the active fallback remains a batched parallel CUDA kernel.
