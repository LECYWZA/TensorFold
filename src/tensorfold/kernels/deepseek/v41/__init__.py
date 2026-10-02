"""DeepSeek-V4.1-Flash's kernels (M1).

The v4 engines' kernels under ``tensorfold.kernels.deepseek.v4`` are keyed to a pool row every 4 or 128 positions
and to bf16 keys; V4.1 pools every 1 or 2 positions and stores FP4 main KV, so its pool, indexer and attention
rows get their own package here rather than a flag in the old one.
"""