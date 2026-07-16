# AGENTS.md

## Project

`minimax` is a Rust/Candle CUDA inference server for MiniMax-M2.7 split GGUF weights. The target hardware is two RTX Pro Blackwell 6000 GPUs (`sm_120`).

## Commands

Run fast checks after Rust changes:

```bash
cargo fmt
cargo check
```

Build for the target machine:

```bash
CUDA_COMPUTE_CAP=120f cargo build --release
```

Run model validation without loading weights:

```bash
cargo run --release -- --dry-run
```

Run the API smoke test against a running server:

```bash
BASE_URL=http://127.0.0.1:8000 ./scripts/test-completion.sh
```

Full startup loads roughly 128 GB of weights and can take several minutes. Do not repeatedly launch inference servers during debugging without killing the previous process.

## Important implementation details

- GGUF shards are in `/storage/models/minimax-m2.7-gguf/UD-Q4_K_XL/` by default.
- Model layers are selectively loaded across both GPUs; shard boundaries can split individual layers.
- `vendor/candle-transformers` contains the MiniMax sigmoid-router and expert-bias changes.
- `vendor/candle-kernels` contains a PTX compatibility workaround for the installed CUDA 13.3 toolkit and CUDA 13.2 driver.
- `vendor/candle-nn` routes GGUF MoE prefill through the generic parallel CUDA kernel. The WMMA path is disabled because real top-8 batches produce numerically wrong, nondeterministic results across expert segments (Q5_K also races for a single expert). Do not replace this with token-by-token prompt ingestion: the fallback remains batched and parallel.
- Prompt prefill uses 512-token physical chunks, matching llama.cpp's default `n_ubatch=512`; llama.cpp's logical default `n_batch` is 2048.
- Cross-GPU activations are staged through host memory at the layer 30/31 boundary. Candle's direct peer copy exposed stale allocations on this two-Blackwell setup; direct P2P made the first divergent tensor appear at layer 31.
- The tokenizer is reconstructed exclusively from GGUF metadata. `tokenizer.ggml.pre=minimax-m2` requires the MiniMax/GPT-4o split regex followed by byte-level encoding with `use_regex=false`; generic GPT-2 byte-level splitting is not equivalent.
- Use `CUDA_COMPUTE_CAP=120f` for builds on the development box.
- The model uses GQA, Q/K RMSNorm, 64-dimension partial RoPE, KV caching, and top-8-of-256 MoE routing. RoPE tables must be constructed in F32; F16 position construction loses precision after 2048 and overflows below the 196K context limit.
- Preserve the OpenAI-compatible `/v1/completions` and `/v1/chat/completions` interfaces. Chat SSE must emit each token as it is generated; never regress to buffering the full response.
- MiniMax tool use requires rendering the native `# Tools`/`<tools>` system section and parsing `<minimax:tool_call><invoke ...>` output into OpenAI `tool_calls`. Preserve `reasoning_content`, prior assistant tool calls, and `tool` result messages.
- The model state retains an exact token-prefix KV cache across requests. Agent follow-up turns should report a nonzero `usage.prompt_tokens_details.cached_tokens`; reset safely whenever the new prompt does not extend the cached IDs.

## Validation expectations

When changing inference code, test an actual request—not only model startup. At minimum verify:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"Hello","max_tokens":1,"temperature":0}'
```

Test 1, 5, 39, and 512-token prefill inputs, plus at least one input crossing the 512-token chunk boundary. Multi-token prefill must return logits only for the final position.

The deterministic llama.cpp golden for chat message `test` has 39 prompt tokens. Its first 24 greedy token IDs are:

```text
758, 3100, 3700, 494, 4500, 4969, 1204, 355, 258, 4160, 1618, 4838,
46, 517, 23413, 1352, 7623, 36238, 46, 25209, 687, 5177, 23077, 46
```

Both implementations decode these to `The user says "test". This is a simple test message. The assistant should respond appropriately. There's no policy violation.`

## Current limitations

- Sampling is currently greedy; temperature/top-p/top-k are not fully implemented.
- Requests are serialized through one model/KV-cache mutex; concurrent scheduling is not implemented.
- Compare logits or generated tokens against llama.cpp when changing model math or CUDA kernels.
