# minimax-rs

A bespoke Rust/Candle CUDA inference server for MiniMax-M2.7 split GGUF weights. It exposes an OpenAI-compatible API subset and is optimized for one specific two-GPU Blackwell workstation rather than portability.

## Scope and requirements

The execution graph is specific to MiniMax-M2.7: model dimensions, 62 layers, four GGUF shards, and exactly two GPUs are fixed in code. Two execution modes are available:

- `pipeline` (default) keeps the established 31/31 layer split.
- `tensor` starts one process per GPU; each process owns all 62 rank-local weight shards and head-sharded KV caches. Hidden states are reduced across TP=2 after each attention and MoE block.

Tested setup:

- Two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs (`sm_120`)
- CUDA 13.3 toolkit with a 595-series driver exposing CUDA 13.2 compatibility
- A recent Rust toolchain with Rust 2024 edition support
- A directory containing exactly four regular `.gguf` shard files and no unrelated GGUF files

The server does not download or distribute model weights. It runs on one machine, requires exactly two CUDA devices, and does not support multi-machine inference. Loading the current quantization uses roughly 128 GB for weights and can take several minutes.

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

Select tensor parallelism explicitly with `--parallelism tensor` or `MINIMAX_PARALLELISM=tensor`:

```bash
CUDA_COMPUTE_CAP=120f cargo run --release -- --parallelism tensor
```

`pipeline` remains the default rollback path. Request and response schemas are identical in both modes.

To inspect shard metadata, construct the tokenizer, and validate the selected GPU topology without loading model weights or starting HTTP:

```bash
# Initialize both pipeline CUDA devices.
CUDA_COMPUTE_CAP=120f cargo run --release -- --dry-run

# Spawn both TP ranks, initialize NCCL/CUDA IPC, and run collective self-tests.
CUDA_COMPUTE_CAP=120f cargo run --release -- --parallelism tensor --dry-run
```

Use `--host 0.0.0.0:8000` only when remote access is intentional. The server has no built-in authentication or TLS and should otherwise remain on localhost or behind a trusted authenticated proxy.

Sampling defaults to MiniMax's recommended `temperature=1.0`, `top_p=0.95`, and `top_k=40`. Server defaults can be changed with `--temp` (or `--temperature`), `--top-p`, and `--top-k`; request fields override them individually. Use `temperature=0` for greedy decoding, `top_p=1` to disable nucleus filtering, or `top_k=0` to disable top-k filtering.

## API

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/perplexity` (deterministic validation extension)

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

`POST /v1/perplexity` accepts the same single string or token-ID-array `prompt` shape as text completions. It resets the shared cache, scores every token after the first with teacher forcing, and returns the token IDs plus `ln p(token[i] | token[:i])` values. The first token is context rather than a scored target. This endpoint is intentionally non-OpenAI and intended for parity validation; scoring is token-by-token and displaces the server's reusable generation cache.

This is an OpenAI-compatible subset, not a complete implementation. Requests are serialized through one model mutex and share one global KV cache, so an unrelated request can displace a reusable prefix. The total context limit is 196,608 tokens. The default output limit is 131,072 tokens and is clipped to the context remaining after the prompt.

## Implementation notes

Prompt prefill defaults to 512-token physical chunks with CUDA FlashAttention over the retained KV prefix. `MINIMAX_PREFILL_CHUNK` accepts values from 1 through 2,048; absent or invalid values use 512.

By default, Q4_K/Q5_K expert projections use direct Q8_1 MMQ for prefill batches of at least 32 tokens. Shorter batches use the lower-overhead generic parallel kernel, while other supported GGUF expert types retain the repaired F16 WMMA path for batches of at least 192 tokens.

Tensor mode keeps the HTTP controller CUDA-free. Cache reset, rewind, prefill, and decode commands are accepted only after both long-lived rank workers report the same cache length. A rank error terminates both workers so a partially advanced cache cannot be reused; `/health` then returns HTTP 503 and `/v1/models` reports `ready: false`. Rank 0 retains logits and performs both on-device greedy selection and request-local stochastic sampling, avoiding full-vocabulary JSON transfers.

Target validation confirmed the decode advantage: a warmed 5-token prompt plus 100 greedy tokens had median endpoint throughput of 140.2 tensor-mode tokens/s versus 125.7 pipeline-mode tokens/s, an 11.5% improvement. Pipeline mode remained substantially faster for warmed 512/513-token prefill (about 229/232 ms versus 381/392 ms) and completed a mixed 125-request prefill/decode matrix in 30.2 seconds versus tensor mode's 36.4 seconds. Mode selection therefore remains explicit: `pipeline` is the default for the broader workload and rollback path, while `tensor` is the decode-latency option.

Each finished generation emits an info-level `inference timings` event with prompt, cache, prefill, and decode token counts; durations; latency and throughput; total inference time; and finish reason. Prefill throughput counts only the uncached prompt suffix.

## Test

Run the default suite:

```bash
cargo fmt
cargo check
cargo test
```

The default suite does not load model weights or initialize CUDA devices, although compiling the test binaries still requires the configured CUDA toolchain. It covers API and chat protocol behavior, tokenizer splitting and token types, sampling, stop matching, cache reuse, shard discovery, and attention workspace sizing.

Validate tensor-mode process and collective startup without loading weights:

```bash
CUDA_COMPUTE_CAP=120f cargo run --release -- --parallelism tensor --dry-run
```

Run the tokenizer integration test with external weights:

```bash
MINIMAX_MODEL_DIR=/path/to/model \
  cargo test gguf_tokenizer_round_trip -- --ignored
```

Run the ignored tensor-parallel and CUDA attention tests on the target hardware (the TP layer test also requires weights):

```bash
cargo test --test tp_collective -- --ignored --nocapture
cargo test --test tp_layer -- --ignored --nocapture
cargo test --test gqa_prefill -- --ignored --nocapture
```

Against a running server on the default address:

```bash
./scripts/test-completion.sh
```

Run the fuller server validation matrix, optionally including repeated deterministic stress cycles:

```bash
./scripts/test-server.py
./scripts/test-server.py --stress-cycles 25
```

The server validation script covers fresh 1/5/39/512/513-token prompts, cache reuse and rewind, stop/EOG cache exclusion, deterministic and stochastic sampling, teacher-forced perplexity scoring, the llama.cpp golden, and completion/reasoning/tool-call SSE reconstruction. Set `BASE_URL` for `test-completion.sh`, or pass `--base-url` to `test-server.py`, when the server uses another address.

Capture a paired teacher-forced perplexity comparison without loading both execution modes at once. First start pipeline mode and save its artifact:

```bash
./scripts/validate-ppl.py corpus.txt --output pipeline-ppl.json
```

Restart the server in tensor mode, then score exactly the same corpus against that reference:

```bash
./scripts/validate-ppl.py corpus.txt \
  --reference pipeline-ppl.json \
  --output tensor-ppl.json
```

The comparison reports aggregate PPL delta, paired mean/p50/p95/p99/max token-NLL differences, and the worst token positions. It reports without imposing an arbitrary acceptance policy by default. Add any combination of `--max-relative-ppl-delta`, `--max-mean-abs-token-nll-delta`, and `--max-p99-abs-token-nll-delta` to turn selected metrics into exit-code gates. Use `--token-ids` when the corpus file is a JSON token-ID array; this removes tokenizer behavior from the model-math comparison.

## License

Except for code under [`vendor/`](vendor/), this repository is licensed under the [Apache License 2.0](LICENSE). Copyright 2026 Stephen Murray.

Vendored crates retain their upstream licenses; see the [vendored-code license inventory](vendor/README.md) and the license files in each vendored crate. Model weights are external and remain subject to the [MiniMax-M2.7 model license](https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE).
