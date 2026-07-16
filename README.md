# minimax

Custom inference engine for MiniMax-M2.X models in GGUF format. The engine is written in Rust and uses Candle/CUDA for GPU acceleration.

## Notes

- I built this as a completely bespoke solution for running my own specific hardware configuration, it's not meant to be portable.
- The engine is designed to run on a single machine with multiple GPUs. It does not support distributed inference across multiple machines (yet).
- Lots of AI slop right now in the vendored Candle/CUDA code, due to the need for sm120-specific patching. I will be cleaning this up over time, it's not clear whether any of it really belongs upstream though.

## Run

Set `MINIMAX_MODEL_DIR` to the directory containing the four GGUF shards. Every model-dependent command also accepts `--model DIR`, which takes precedence over the environment variable.

```bash
export MINIMAX_MODEL_DIR="<directory-containing-the-gguf-shards>"
CUDA_COMPUTE_CAP=120f cargo run --release --features cuda -- \
  --host 0.0.0.0:8000
```

Weights load across two GPUs and startup may take several minutes. Tokenization is reconstructed from the GGUF metadata, but this part of the stack still needs some work to unravel some of the temporary hacks.

## Test

```bash
BASE_URL=http://127.0.0.1:8000 ./scripts/test-completion.sh
cargo test gguf_tokenizer_round_trip -- --ignored
```

The tokenizer integration test uses `MINIMAX_MODEL_DIR` and is ignored by default because it requires the external model weights. Model-dependent examples use the same environment variable and `--model` option.

Endpoints: `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`.

Chat completions support SSE streaming, `reasoning_content`, OpenAI function tools, MiniMax XML tool-call parsing, prior assistant/tool turns, and both `max_tokens` and `max_completion_tokens`. The default output limit is 131,072 tokens. Exact-prefix KV reuse accelerates subsequent agent/tool turns and is reported as `usage.prompt_tokens_details.cached_tokens`.

The implementation uses 512-token parallel prefill chunks. The repaired WMMA MoE path is enabled for prefill batches of at least 192 tokens; shorter batches use the generic parallel CUDA kernel because it has lower fixed overhead.

Each completed request emits an info-level `inference timings` event with prompt/cache token counts, prefill and decode durations, milliseconds per token, tokens per second, total inference time, and finish reason. Prefill throughput counts only the uncached prompt suffix, so cache hits remain distinguishable from model evaluation.
