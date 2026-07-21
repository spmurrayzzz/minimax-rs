# minimax-rs

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

Sampling defaults to MiniMax's recommended `temperature=1.0`, `top_p=0.95`, and `top_k=40`. Server-wide defaults can be changed with `--temp` (or `--temperature`), `--top-p`, and `--top-k`; the corresponding `temperature`, `top_p`, and `top_k` fields on either completion endpoint override those defaults individually for that request. Use `temperature=0` for greedy decoding, `top_p=1` to disable nucleus filtering, or `top_k=0` to disable top-k filtering.

## Test

```bash
cargo test
BASE_URL=http://127.0.0.1:8000 ./scripts/test-completion.sh
cargo test gguf_tokenizer_round_trip -- --ignored
```

The default unit suite does not load model weights or initialize CUDA devices. It covers API/chat protocol behavior, tokenizer splitting, cache reuse, causal masking, and shard discovery without asserting GPU or layer placement. The tokenizer integration test uses `MINIMAX_MODEL_DIR` and is ignored by default because it requires the external model weights. Model-dependent examples use the same environment variable and `--model` option.

Endpoints: `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`.

Chat completions support SSE streaming, `reasoning_content`, OpenAI function tools, MiniMax XML tool-call parsing, prior assistant/tool turns, and both `max_tokens` and `max_completion_tokens`. The default output limit is 131,072 tokens. Longest-common-token-prefix KV reuse accelerates subsequent agent/tool turns and is reported as `usage.prompt_tokens_details.cached_tokens`.

The implementation defaults to 512-token parallel prefill chunks with CUDA FlashAttention over the retained KV prefix; `MINIMAX_PREFILL_CHUNK` can raise the physical chunk size to at most 2,048. Q4_K/Q5_K expert projections use direct quantized MMQ for prefill batches of at least 32 tokens, while shorter batches retain the lower-overhead generic kernel.

Each completed request emits an info-level `inference timings` event with prompt/cache token counts, prefill and decode durations, milliseconds per token, tokens per second, total inference time, and finish reason. Prefill throughput counts only the uncached prompt suffix, so cache hits remain distinguishable from model evaluation.
