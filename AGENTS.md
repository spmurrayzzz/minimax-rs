# AGENTS.md

## Project

`minimax` is a Rust/Candle CUDA inference server for MiniMax-M2.7 split GGUF weights. The execution graph is fixed to four shards and exactly two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs (`sm_120`), with layers split 31/31.

## Commands

Commands that access weights require `MINIMAX_MODEL_DIR` or an explicit `--model DIR`. There is no built-in model path. The shard directory must contain exactly four regular `.gguf` files.

Run standard checks after Rust changes:

```bash
cargo fmt
cargo check
cargo test
```

Build for the target machine:

```bash
CUDA_COMPUTE_CAP=120f cargo build --release
```

Parse shard metadata, construct the tokenizer, and initialize both CUDA devices without loading model tensors or starting HTTP:

```bash
CUDA_COMPUTE_CAP=120f cargo run --release -- --dry-run
```

The server defaults to `127.0.0.1:8080`. Run the API smoke test against an already-running server with:

```bash
./scripts/test-completion.sh
```

Set `BASE_URL` only when using another address. The server has no authentication or TLS; do not expose it directly to an untrusted network.

Full startup loads roughly 128 GB of weights and can take several minutes. Do not repeatedly launch inference servers during debugging without first killing the previous process.

## Important implementation details

### Model and CUDA

- Model dimensions are hardcoded for MiniMax-M2.7: GQA, Q/K RMSNorm, 64-dimension partial RoPE, 62 layers, KV caching, and sigmoid-gated top-8-of-256 MoE routing.
- Model layers are selectively loaded from all shards onto both GPUs; GGUF shard boundaries can split an individual layer.
- RoPE positions and frequencies must be constructed in F32. F16 positions lose integer precision after 2,048 and overflow before the 196,608-token context limit.
- Activations at the layer 30/31 boundary and final activations returning from GPU 1 to GPU 0 are staged through host memory. Direct Candle peer copies exposed stale allocations on this machine, with the first divergent tensor at layer 31.
- `vendor/candle-transformers` contains the MiniMax sigmoid router and expert-bias selection behavior.
- `vendor/candle-kernels` contains a PTX 9.3-to-9.2 compatibility workaround for the installed CUDA 13.3 toolkit and 595-series driver exposing CUDA 13.2 compatibility.
- `vendor/cudarc` binds the owning CUDA context before freeing untracked async allocations. This is required because inference disables allocation-event tracking and requests can move between worker threads while using two contexts.
- Preserve batched prompt ingestion. Physical prefill chunks default to 512 tokens; `MINIMAX_PREFILL_CHUNK` accepts 1 through 2,048. Do not replace prefill with token-by-token forwarding.
- By default, Q4_K/Q5_K MoE prefill uses direct Q8_1 MMQ at 32 or more tokens, other supported GGUF types use the repaired F16 WMMA path at 192 or more tokens, and shorter batches use the generic parallel kernel. CUB provides stable route grouping. `MINIMAX_MMQ_MIN_TOKENS` and `MINIMAX_WMMA_MIN_TOKENS` override the thresholds for targeted experiments.

### Tokenizer, API, and cache

- GGUF metadata supplies vocabulary, merges, token types, and configured EOG IDs. Model-specific code supplies the canonical MiniMax/GPT-4o split regex, byte-level encoding with `use_regex=false`, recognized EOG spellings, and protocol-marker handling. Generic GPT-2 byte-level splitting is not equivalent.
- Preserve the OpenAI-compatible `/v1/completions` and `/v1/chat/completions` subset. Completion prompts accept one string or one token-ID array; valid batch-shaped prompts must return HTTP 400 before streaming begins.
- `stop` accepts one string or at most four strings. A selected EOG token is neither emitted nor evaluated. When the incremental matcher recognizes a stop on a selected token, do not evaluate or cache that completing token; preserve the tested final-decoder-tail behavior. `cached_ids` must always describe exactly the positions present in every layer cache.
- Stream content, reasoning, and tool-call deltas incrementally; never buffer the complete response before emitting SSE. UTF-8 completion, stop-prefix matching, and protocol parsing may temporarily withhold partial text.
- MiniMax tool use requires the native `# Tools`/`<tools>` system section and `<minimax:tool_call><invoke ...>` parsing. Preserve canonical tool rendering (`name`, `description`, and `parameters`), JSON Schema validation, `tool_choice` policies, incremental OpenAI tool-call deltas, prior assistant calls, grouped tool results, and assistant whitespace.
- A leading `developer` message is treated like the system message. Preserve the existing handling of later policy roles and the tested reasoning retention/omission semantics.
- Sampling precedence is built-in defaults, then server flags, then per-request fields. `temperature=0` must retain the deterministic on-device argmax path.
- Model state retains one exact token-prefix KV cache across requests. Rewind every layer to the longest common prefix when a prompt diverges and reset only when no prefix is reusable. A direct sequential agent follow-up with a shared prefix and no intervening request should report nonzero `usage.prompt_tokens_details.cached_tokens`.

## Validation expectations

When changing inference code, test an actual request rather than only startup. At minimum verify:

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"Hello","max_tokens":1,"temperature":0}'
```

Test fresh 1-, 5-, 39-, and 512-token prompts plus a prompt crossing the physical chunk boundary, such as 513 tokens. Use distinct first token IDs between cases so the global prefix cache does not turn them into suffix-only evaluations. Multi-token prefill must return logits only for the final position. `scripts/test-completion.sh` does not cover every one of these lengths.

When attention kernels change, run the ignored CUDA tests on the target hardware:

```bash
cargo test --test gqa_prefill -- --ignored --nocapture
```

The deterministic llama.cpp golden for chat message `test` has 39 prompt tokens. Its first 24 greedy token IDs are:

```text
758, 3100, 3700, 494, 4500, 4969, 1204, 355, 258, 4160, 1618, 4838,
46, 517, 23413, 1352, 7623, 36238, 46, 25209, 687, 5177, 23077, 46
```

Both minimax-rs and llama.cpp decode these to `The user says "test". This is a simple test message. The assistant should respond appropriately. There's no policy violation.` Compare logits or generated tokens against llama.cpp whenever model math or CUDA kernels change.

## Current limitations

- Requests are serialized through one model/KV-cache mutex; concurrent or continuous batching is not implemented.
- There is one global prefix cache rather than per-client sessions, so unrelated requests can displace reusable state.
- Completion prompt batches and distributed multi-machine inference are not supported.
- The HTTP API is an OpenAI-compatible subset and has no built-in authentication or TLS.
