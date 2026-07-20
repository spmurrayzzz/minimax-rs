# AGENTS.md

## Project

`minimax` is a Rust/Candle CUDA inference server for MiniMax-M2.7 split GGUF weights. The target hardware is two RTX Pro Blackwell 6000 GPUs (`sm_120`).

## Commands

Commands that access weights require `MINIMAX_MODEL_DIR` or an explicit `--model DIR`. Set the environment variable to the directory containing the four GGUF shards before running them.

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

- The GGUF shard directory has no built-in filesystem default; supply it through `--model DIR` or `MINIMAX_MODEL_DIR`.
- Model layers are selectively loaded across both GPUs; shard boundaries can split individual layers.
- `vendor/candle-transformers` contains the MiniMax sigmoid-router and expert-bias changes.
- `vendor/candle-kernels` contains a PTX compatibility workaround for the installed CUDA 13.3 toolkit and CUDA 13.2 driver.
- `vendor/cudarc` binds the owning CUDA context before freeing untracked async allocations. This is required because the server disables allocation-event tracking and moves requests between worker threads while using two GPU contexts.
- Q4_K/Q5_K GGUF MoE prefill batches of at least 32 tokens use the llama.cpp-style direct Q8_1 MMQ path; shorter batches use the generic parallel CUDA kernel. Other GGUF types retain the repaired F16 WMMA fallback at 192 tokens. CUB provides stable expert grouping beyond the old shared-memory argsort limit. MMQ is deterministic in benchmark runs, and its numerical differences from the generic path remain within the benchmark tolerance. Do not replace these paths with token-by-token prompt ingestion.
- Prompt prefill defaults to 512-token physical chunks, matching llama.cpp's default `n_ubatch=512`. `MINIMAX_PREFILL_CHUNK` can select a validated size up to 2048, which is llama.cpp's logical `n_batch` default.
- Cross-GPU activations are staged through host memory at the layer 30/31 boundary. Candle's direct peer copy exposed stale allocations on this two-Blackwell setup; direct P2P made the first divergent tensor appear at layer 31.
- The tokenizer is reconstructed exclusively from GGUF metadata. `tokenizer.ggml.pre=minimax-m2` requires the MiniMax/GPT-4o split regex followed by byte-level encoding with `use_regex=false`; generic GPT-2 byte-level splitting is not equivalent.
- Use `CUDA_COMPUTE_CAP=120f` for builds on the development box.
- The model uses GQA, Q/K RMSNorm, 64-dimension partial RoPE, KV caching, and top-8-of-256 MoE routing. RoPE tables must be constructed in F32; F16 position construction loses precision after 2048 and overflows below the 196K context limit.
- Preserve the OpenAI-compatible `/v1/completions` and `/v1/chat/completions` interfaces. Chat SSE must emit each token as it is generated; never regress to buffering the full response.
- MiniMax tool use requires rendering the native `# Tools`/`<tools>` system section and parsing `<minimax:tool_call><invoke ...>` output into OpenAI `tool_calls`. Preserve `reasoning_content`, prior assistant tool calls, and `tool` result messages.
- The model state retains an exact token-prefix KV cache across requests. Rewind every layer to the longest common token prefix when a prompt diverges, and reset only when no prefix is reusable. Agent follow-up turns should report a nonzero `usage.prompt_tokens_details.cached_tokens`.

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
