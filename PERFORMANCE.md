# Performance notes

Measured on two NVIDIA RTX PRO 6000 Blackwell GPUs (`sm_120`) with the
UD-Q4_K_XL MiniMax-M2.7 GGUF. Requests were greedy, warmed, and uncached unless
noted otherwise.

## Results

| Metric | Before | After |
|---|---:|---:|
| Decode, 39-token prompt | ~43 tok/s | ~88 tok/s |
| TTFT, 39-token prompt | 97–117 ms | ~95 ms |
| TTFT, 512-token prompt | ~1.34 s | ~0.65 s |
| TTFT, 1024-token prompt | ~2.71 s | ~1.34 s |
| TTFT, 4096-token prompt | not recorded | ~6.14 s |
| Decode, 4096-token prompt | not recorded | ~56 tok/s |

The final decode measurement used 96 streamed tokens and reported 87.56–87.92
tok/s over three runs. The 512-token TTFT was 654–684 ms over three runs.

## Changes that mattered

- Disabled cudarc allocation-event tracking for the server's one-stream-per-GPU
  execution model. The vendored cudarc fix binds the owning context before an
  untracked async free, which is required when requests move between worker
  threads or execution switches GPUs.
- Replaced full-history `ConcatKvCache` copies with a growable `KvCache`.
- Performed true grouped-query attention by grouping six Q heads per KV head,
  rather than materializing six copies of K and V.
- Added exact fused single-token CUDA paths for Q/K/V projection, Q/K RMSNorm,
  F16 conversion, and partial RoPE. The fused projection shares one activation
  quantization across all three Q8_0 matrices.
- Repaired the GGUF WMMA MoE kernel's 64-logical-lane dequantization and shared
  memory races. It is enabled only at 192 or more prefill tokens, where it wins
  on this GPU; short prompts retain the generic kernel.
- Kept logits on-device and used GPU argmax instead of copying/scanning all
  200,064 logits on the CPU.
- Switched streaming text generation to `DecodeStream`, avoiding a full token
  history decode after every token.

Set `MINIMAX_WMMA_MIN_TOKENS=999999` to force the generic MoE prefill path for
numerical comparisons. WMMA uses F16 tensor-core inputs rather than the generic
path's Q8_1 activation quantization, so long-prompt greedy continuations can
change. Projection checks show small errors versus a dequantized F16 reference,
and validation covered coherent long prompts, native tool use, determinism, and
the 24-token llama.cpp chat golden.

## Reproduce

```bash
CUDA_COMPUTE_CAP=120f cargo build --release
./scripts/bench.py --length 39 --max-tokens 96 --repeats 3
./scripts/bench.py --length 512 --max-tokens 1 --repeats 3
./scripts/bench.py --length 1024 --max-tokens 1 --repeats 3
BASE_URL=http://127.0.0.1:8000 ./scripts/test-completion.sh
```

Kernel and layer microbenchmarks are in `examples/`.
