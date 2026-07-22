# Vendored code

Code in this directory is not covered by the repository-root `LICENSE`. Each
vendored crate retains its upstream license, and local modifications are made
available under the same license choice as that crate.

| Path | Upstream baseline | License | Included license files |
| --- | --- | --- | --- |
| `candle-kernels/` | [Candle 0.11.0 at `31f35b1`](https://github.com/huggingface/candle/tree/31f35b147389700ed2a178ee66a91c3cc25cc80d/candle-kernels) | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| `candle-nn/` | [Candle 0.11.0 at `31f35b1`](https://github.com/huggingface/candle/tree/31f35b147389700ed2a178ee66a91c3cc25cc80d/candle-nn) | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| `candle-transformers/` | [Candle 0.11.0 at `31f35b1`](https://github.com/huggingface/candle/tree/31f35b147389700ed2a178ee66a91c3cc25cc80d/candle-transformers) | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |
| `cudarc/` | [cudarc 0.19.8 at `e1ab1e5`](https://github.com/chelsea0x3b/cudarc/tree/e1ab1e59f441db6f0a5aef0d5a078696f802a166) | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE` |

The Candle crates were published from one workspace, so their crates.io
archives did not contain the workspace-root license files. Exact copies from
the recorded Candle source revision are included here. cudarc's published
license files were already present and are retained unchanged.

These are modified vendored copies rather than pristine upstream releases.
The source comments and attribution links supplied by upstream are retained;
Git history records this project's changes.
