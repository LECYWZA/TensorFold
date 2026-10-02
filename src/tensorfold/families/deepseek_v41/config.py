"""DeepSeek-V4.1-Flash's settings from the checkpoint's config.json, and the gap list this skeleton still owes.

V4.1 is a new ``model_type`` (``deepseek_v41`` / ``deepseek_v41_text``): its config names source layers whose KV and
indexer K later layers reuse, a hierarchical candidate pool, 1- and 2-position pool ratios, per-layer Engram tables
and a vision tower. This module reads that config as data and states, machine-checkably, what an engine must still
implement; ``unsupported()`` is the task list, not a comment. Nothing here imports MLX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
# pool ratios the architecture uses per layer: a window-only layer, or a pool row every 1 or 2 positions
RATIOS = (0, 1, 2)


@dataclass(frozen=True)
class Gap:
    """One piece of the architecture with no engine behind it yet."""

    key: str
    title: str
    detail: str
    lands_in: str


@dataclass
class Config:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    sliding_window: int
    compress_ratios: list[int]
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    rope_original: int
    beta_fast: float
    beta_slow: float
    kv_source_layer_ids: list[int]
    index_source_layer_ids: list[int]
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    moe_intermediate_size: int
    scoring_func: str
    topk_method: str
    routed_scaling_factor: float
    swiglu_limit: float
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    engram_layer_ids: list[int]
    engram_num_embeddings: list[int]
    engram_vocab_size: int
    engram_compressed_vocab_size: int
    engram_max_ngram_size: int
    engram_n_heads: int
    engram_head_dim: int
    num_nextn_predict_layers: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: list[int]
    dspark_markov_rank: int
    dspark_n_routed_experts: int
    dspark_num_experts_per_tok: int
    eos_token_id: list[int]
    image_token_id: int | None = None
    vision: dict[str, Any] = field(default_factory=dict)

    def ratio(self, layer: int) -> int:
        return int(self.compress_ratios[layer]) if layer < len(self.compress_ratios) else 0

    def pool_ratios(self) -> set[int]:
        return {int(r) for r in self.compress_ratios if int(r)}

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        """The V4.1 text config, refusing anything this reading has not seen before."""

        t = dict(config.get("text_config") or config)
        kind = str(t.get("model_type") or config.get("model_type") or "")
        if kind not in MODEL_TYPES:
            raise ValueError(f"deepseek_v41: model_type {kind!r}; this family reads {MODEL_TYPES}")
        layers = int(t["num_hidden_layers"])
        ratios = [int(r) for r in t.get("compress_ratios") or []]
        if len(ratios) < layers:
            raise ValueError(f"deepseek_v41: compress_ratios covers {len(ratios)} of {layers} layers")
        unknown = sorted({r for r in ratios if r not in RATIOS})
        if unknown:
            raise ValueError(f"deepseek_v41: compress_ratios use {unknown}; this reading knows {RATIOS} "
                             f"(0 window-only, 1 and 2 pool every such position)")

        def ids(key: str) -> list[int]:
            found = [int(x) for x in t.get(key) or []]
            bad = [x for x in found if not 0 <= x < layers]
            if bad:
                raise ValueError(f"deepseek_v41: {key} {bad} outside the model's {layers} layers")
            return found

        rope = dict(t.get("rope_scaling") or {})
        eos = t.get("eos_token_id", config.get("eos_token_id"))
        vision = dict(config.get("vision_config") or t.get("vision_config") or {})
        return cls(
            hidden_size=int(t["hidden_size"]), num_hidden_layers=layers, vocab_size=int(t["vocab_size"]),
            max_position_embeddings=int(t.get("max_position_embeddings", 0)),
            rms_norm_eps=float(t.get("rms_norm_eps", 1e-6)), num_attention_heads=int(t["num_attention_heads"]),
            num_key_value_heads=int(t.get("num_key_value_heads", 1)), head_dim=int(t["head_dim"]),
            qk_rope_head_dim=int(t["qk_rope_head_dim"]), q_lora_rank=int(t["q_lora_rank"]),
            o_lora_rank=int(t["o_lora_rank"]), o_groups=int(t["o_groups"]),
            sliding_window=int(t.get("sliding_window", 128)), compress_ratios=ratios,
            rope_theta=float(t.get("rope_theta", 10000.0)),
            compress_rope_theta=float(t.get("compress_rope_theta", t.get("rope_theta", 10000.0))),
            rope_factor=float(rope.get("factor", 1.0)),
            rope_original=int(rope.get("original_max_position_embeddings", 0)),
            beta_fast=float(rope.get("beta_fast", 32)), beta_slow=float(rope.get("beta_slow", 1)),
            kv_source_layer_ids=ids("kv_source_layer_ids"),
            index_source_layer_ids=ids("index_source_layer_ids"),
            candidate_source_layer_id=int(t.get("candidate_source_layer_id", -1)),
            candidate_topk_blocks=int(t.get("candidate_topk_blocks", 0)),
            candidate_block_size=int(t.get("candidate_block_size", 0)),
            index_n_heads=int(t["index_n_heads"]), index_head_dim=int(t["index_head_dim"]),
            index_topk=int(t["index_topk"]), n_routed_experts=int(t["n_routed_experts"]),
            num_experts_per_tok=int(t["num_experts_per_tok"]), n_shared_experts=int(t.get("n_shared_experts", 1)),
            moe_intermediate_size=int(t["moe_intermediate_size"]),
            scoring_func=str(t.get("scoring_func", "sqrtsoftplus")), topk_method=str(t.get("topk_method", "")),
            routed_scaling_factor=float(t.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(t.get("swiglu_limit") or 0.0), hc_mult=int(t.get("hc_mult", 4)),
            hc_eps=float(t.get("hc_eps", 1e-6)), hc_sinkhorn_iters=int(t.get("hc_sinkhorn_iters", 20)),
            engram_layer_ids=ids("engram_layer_ids"),
            engram_num_embeddings=[int(n) for n in t.get("engram_num_embeddings") or []],
            engram_vocab_size=int(t.get("engram_vocab_size", 0)),
            engram_compressed_vocab_size=int(t.get("engram_compressed_vocab_size", 0)),
            engram_max_ngram_size=int(t.get("engram_max_ngram_size", 0)),
            engram_n_heads=int(t.get("engram_n_heads", 0)), engram_head_dim=int(t.get("engram_head_dim", 0)),
            num_nextn_predict_layers=int(t.get("num_nextn_predict_layers", 0)),
            dspark_block_size=int(t.get("dspark_block_size", 0)),
            dspark_noise_token_id=int(t.get("dspark_noise_token_id", 0)),
            dspark_target_layer_ids=[int(x) for x in t.get("dspark_target_layer_ids") or []],
            dspark_markov_rank=int(t.get("dspark_markov_rank", 0)),
            dspark_n_routed_experts=int(t.get("dspark_n_routed_experts", 0)),
            dspark_num_experts_per_tok=int(t.get("dspark_num_experts_per_tok", 0)),
            eos_token_id=list(eos) if isinstance(eos, list) else ([int(eos)] if eos is not None else []),
            image_token_id=None if t.get("image_token_id") is None and config.get("image_token_id") is None
            else int(t.get("image_token_id", config.get("image_token_id", 0))),
            vision=vision)


def unsupported(config: dict[str, Any]) -> list[Gap]:
    """Every V4.1 feature with no engine behind it, as the list of work an engine owes this family."""

    cfg = Config.from_dict(config)
    gaps = [
        Gap("cross-layer-kv",
            "Cross-layer main KV and indexer K (the causal encoder/decoder split in this config)",
            f"kv_source_layer_ids {cfg.kv_source_layer_ids} publish main KV and index_source_layer_ids "
            f"{cfg.index_source_layer_ids} publish indexer K for later layers to reuse. The v4 engine builds one "
            f"independent cache per layer that writes its own keys, pool and index rows, so it must instead give "
            f"source layers a shared store and dependent layers a read plan, and count that memory once.",
            "families/deepseek_v41/attention.py and caches.py (new): a per-layer mode table from these ids"),
        Gap("candidate-pool",
            "Hierarchical sparse indexer over a candidate pool",
            f"Layer {cfg.candidate_source_layer_id} builds a candidate pool of {cfg.candidate_topk_blocks} blocks of "
            f"{cfg.candidate_block_size} positions; deeper indexing layers select inside that pool. The v4 indexer "
            f"selects rows from one flat pool, so the second stage, its block geometry and its cost per row are new.",
            "families/deepseek_v41/attention.py (new): pool builder, block selection and its exactness check"),
        Gap("pool-ratios",
            f"Pool ratios {sorted(cfg.pool_ratios())} instead of the v4 engine's 4 and 128",
            f"compress_ratios {cfg.compress_ratios[:6]}... give a pool row every 1 or 2 positions. The v4 compressor "
            f"kernel, its projection ring, the indexer's row width and the memory-growth estimate are all keyed to a "
            f"ratio of 4 (or 128), so the pooling math and every ring size change with them.",
            "families/deepseek_v41/compressor.py, kernels/deepseek/v41/pool.py (new)"),
        Gap("fp4-kv",
            "FP4 main KV cache (E2M1 with an E4M3 scale per 16 channels)",
            "The model card puts the global KV at 890 bytes a token. The MLX engine stores bf16 keys in a ring and "
            "the CUDA --kv-dtype switch belongs to Flash Next only, so V4.1 needs its own quantized KV write/read "
            "path whose per-row arithmetic still matches a serial one-row call.",
            "families/deepseek_v41/caches.py plus a KV kernel in kernels/deepseek/v41"),
        Gap("engram",
            "Engram conditional memory (token-lookup tables)",
            f"layers {cfg.engram_layer_ids} read tables of {cfg.engram_num_embeddings or 'unknown'} entries with "
            f"{cfg.engram_max_ngram_size}-gram hashing, {cfg.engram_n_heads} heads of {cfg.engram_head_dim} "
            f"(vocabulary {cfg.engram_vocab_size}, compressed to {cfg.engram_compressed_vocab_size}). Nothing in "
            f"the v4 family reads these weights: they change the loader, the resident-weight estimate, admission "
            f"and the whole-block forward (a lookup kernel, not a linear).",
            "families/deepseek_v41/engram.py plus weight_bytes() and the admission estimate"),
        Gap("weight-format",
            "No MLX checkpoint of the released weights",
            "The official checkpoint ships fp8 (ue8m0 scales, 32x32 blocks) with fp4 experts, while this engine "
            "reads MLX affine 4-bit weights in groups of 64 with mxfp4 routed experts. A conversion must be "
            "published (mlx-community-style) or the loader must read the official layout directly.",
            "src/tensorfold/families/deepseek_v41/convert.py or a published MLX conversion"),
        Gap("vision",
            "Vision tower (DeepSeek-ViT) and image input",
            "vision_config carries a 32-layer tower (hidden 1024, patch 14, 3x3 downsample) with its own projector "
            "and image_token_id, so --vision, the image budget and the prompt-cache rules for image rows come with "
            "the family rather than after it.",
            "families/deepseek_v41/vision.py, reusing the shared image path"),
        Gap("rms-eps",
            "rms_norm_eps 1e-20 in the exact kernels",
            f"The config sets rms_norm_eps {cfg.rms_norm_eps:g} where the v4 engine assumes 1e-6. At 1e-20 the "
            f"epsilon is effectively absent, so every fused norm, its fp32 accumulation order and the load-time "
            f"window check must be re-verified against the released implementation rather than inherited.",
            "kernels/deepseek/v41: norm paths and their row/width equality checks"),
        Gap("swa-replay",
            "SWA Bounded Replay (strategy, not a config field)",
            "The card rebuilds missing sliding-window KV by replaying the last n_win tokens instead of persisting "
            "them. This engine keeps the window ring resident and moves an offset, so replay may be unnecessary "
            "here; the open question is only whether a replayed window is bit-identical to the written one, which "
            "needs the real weights to settle.",
            "a load-time or test-time comparison, not necessarily new kernels"),
    ]
    return gaps


def reusable() -> list[str]:
    """What the v4 family already supplies, so the gap list above is not read as a rewrite."""

    return [
        "the lane-engine protocol and its exactness checks (exact_width, window_costs, shared rows, rollback by offset)",
        "128-token window with sinks and position-indexed caches",
        "compressor/indexer scaffolding, to be re-keyed to the new pool ratios",
        "hyper-connection blocks and row linears shared with GLM-5.3 (KERNEL_DEPENDENCIES)",
        "DSpark drafting: the converter already reads block size, noise token and target layers from these configs",
        "the tokenizer and chat/tool encoding (DeepSeek's own template)",
        "tests/dsv4_fakes.py-style model-free tests for the engine's invariants",
    ]


def summary(config: dict[str, Any]) -> str:
    """A one-screen report: what the config is, and what an engine owes it."""

    cfg = Config.from_dict(config)
    ratios = {}
    for r in cfg.compress_ratios:
        ratios[int(r)] = ratios.get(int(r), 0) + 1
    lines = [
        f"DeepSeek-V4.1-Flash: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
        f"vocab {cfg.vocab_size}, window {cfg.sliding_window}, pool ratios {ratios}",
        f"  attention {cfg.num_attention_heads} heads / {cfg.num_key_value_heads} shared KV head of {cfg.head_dim}, "
        f"o_lora_rank {cfg.o_lora_rank} in {cfg.o_groups} groups, rope {cfg.rope_theta:g}"
        f"/{cfg.compress_rope_theta:g} (YaRN factor {cfg.rope_factor:g} from {cfg.rope_original})",
        f"  sharing   kv sources {cfg.kv_source_layer_ids}, index sources {cfg.index_source_layer_ids}, "
        f"candidate pool from layer {cfg.candidate_source_layer_id} "
        f"({cfg.candidate_topk_blocks} x {cfg.candidate_block_size})",
        f"  MoE       {cfg.n_routed_experts} routed experts, {cfg.num_experts_per_tok} a token, "
        f"{cfg.n_shared_experts} shared, {cfg.scoring_func}",
        f"  drafter   {cfg.num_nextn_predict_layers} MTP layers, DSpark block {cfg.dspark_block_size} "
        f"over layers {cfg.dspark_target_layer_ids}",
        f"  engram    layers {cfg.engram_layer_ids}, {cfg.engram_num_embeddings} entries, "
        f"{cfg.engram_max_ngram_size}-gram, {cfg.engram_n_heads}x{cfg.engram_head_dim}",
        f"  vision    {'yes' if cfg.vision else 'no'}",
        f"  gaps      {len(unsupported(config))}",
    ]
    for gap in unsupported(config):
        lines.append(f"    - {gap.key}: {gap.title} -> {gap.lands_in}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m tensorfold.families.deepseek_v41.config <model dir or config.json>``: the gap report."""

    import json
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m tensorfold.families.deepseek_v41.config <model dir or config.json>", file=sys.stderr)
        return 2
    path = Path(args[0])
    config = json.loads((path / "config.json").read_text() if path.is_dir() else path.read_text())
    print(summary(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())