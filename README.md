# minimax-rs

A bespoke Rust/Candle CUDA inference server for MiniMax-M2.7 split GGUF weights. It exposes an OpenAI-compatible API subset and is optimized for one specific two-GPU Blackwell workstation rather than portability.

## Scope and requirements

The current execution graph is specific to MiniMax-M2.7: model dimensions, 62 layers, four GGUF shards, and a two-GPU 31/31 layer split are fixed in code.

Tested setup:

- Two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs (`sm_120`)
- CUDA 13.3 toolkit with a 595-series driver exposing CUDA 13.2 compatibility
- A recent Rust toolchain with Rust 2024 edition support
- A directory containing exactly four regular `.gguf` shard files and no unrelated GGUF files

The server does not download or distribute model weights. It runs on one machine, requires exactly two CUDA devices, and does not support distributed inference. Loading the current quantization uses roughly 128 GB for weights and can take several minutes.

## Run

Set the model directory, then start the server on its default address, `127.0.0.1:8000`:

```bash
export MINIMAX_MODEL_DIR="<directory-containing-the-four-gguf-shards>"
CUDA_COMPUTE_CAP=120f cargo run --release
```

The server and weight-backed examples also accept `--model DIR`, which takes precedence over `MINIMAX_MODEL_DIR`:

```bash
CUDA_COMPUTE_CAP=120f cargo run --release -- --model /path/to/model
```

To inspect shard metadata, construct the tokenizer, and verify that two CUDA devices can be initialized without loading model tensors or starting HTTP:

```bash
CUDA_COMPUTE_CAP=120f cargo run --release -- --dry-run
```

Use `--host 0.0.0.0:8000` only when remote access is intentional. The server has no built-in authentication or TLS and should otherwise remain on localhost or behind a trusted authenticated proxy.

Sampling defaults to MiniMax's recommended `temperature=1.0`, `top_p=0.95`, and `top_k=40`. Server defaults can be changed with `--temp` (or `--temperature`), `--top-p`, and `--top-k`; request fields override them individually. Use `temperature=0` for greedy decoding, `top_p=1` to disable nucleus filtering, or `top_k=0` to disable top-k filtering.

## API

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Minimal chat request:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Supported behavior includes:

- Text completions from one string prompt or one token-ID array; prompt batches are rejected.
- Chat messages with a leading `system` or `developer` message, prior assistant tool calls, and `tool` results.
- SSE streaming of content, `reasoning_content`, and incremental OpenAI-shaped tool-call deltas.
- OpenAI function tools, `tool_choice` (`auto`, `none`, `required`, or a named function), JSON Schema argument validation, and MiniMax native XML tool-call parsing.
- `stop` as one string or up to four strings, plus GGUF and MiniMax end-of-generation tokens.
- `max_tokens` on both endpoints and `max_completion_tokens` on chat requests.
- Longest-common-token-prefix KV reuse, reported as `usage.prompt_tokens_details.cached_tokens`.

This is an OpenAI-compatible subset, not a complete implementation. Requests are serialized through one model mutex and share one global KV cache, so an unrelated request can displace a reusable prefix. The total context limit is 196,608 tokens. The default output limit is 131,072 tokens and is clipped to the context remaining after the prompt.

## Implementation notes

Prompt prefill defaults to 512-token physical chunks with CUDA FlashAttention over the retained KV prefix. `MINIMAX_PREFILL_CHUNK` accepts values from 1 through 2,048; absent or invalid values use 512.

By default, Q4_K/Q5_K expert projections use direct Q8_1 MMQ for prefill batches of at least 32 tokens. Shorter batches use the lower-overhead generic parallel kernel, while other supported GGUF expert types retain the repaired F16 WMMA path for batches of at least 192 tokens.

Each finished generation emits an info-level `inference timings` event with prompt, cache, prefill, and decode token counts; durations; latency and throughput; total inference time; and finish reason. Prefill throughput counts only the uncached prompt suffix.

## Test

Run the default suite:

```bash
cargo test
```

It does not load model weights or initialize CUDA devices, although compiling the test binaries still requires the configured CUDA toolchain. It covers API and chat protocol behavior, tokenizer splitting and token types, sampling, stop matching, cache reuse, shard discovery, and attention workspace sizing.

Run the tokenizer integration test with external weights:

```bash
MINIMAX_MODEL_DIR=/path/to/model \
  cargo test gguf_tokenizer_round_trip -- --ignored
```

Run the ignored CUDA attention correctness and near-context-limit tests on the target hardware:

```bash
cargo test --test gqa_prefill -- --ignored --nocapture
```

Against a running server on the default address:

```bash
./scripts/test-completion.sh
```

Set `BASE_URL` when the server uses another address.

## License

This repository does not currently include a code license. Model weights are external and remain subject to the [MiniMax-M2.7 model license](https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE).
