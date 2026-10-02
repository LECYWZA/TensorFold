"""DeepSeek-V4.1-Flash (model_type ``deepseek_v41``): the skeleton a CED engine is built on.

The config is read and validated today (see ``config.unsupported()`` for the work still owed); ``load`` refuses
until that work lands, so ``tensorfold info`` answers before anyone downloads 276 GB of weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
TITLE = "DeepSeek-V4.1-Flash"
LANES = True
# the released checkpoint (fp8 with fp4 experts); an MLX affine 4-bit/g64 conversion is still to be published
MODELS = ("deepseek-ai/DeepSeek-V4.1-Flash",)
# the MLX engine reads MLX-quantized weights, as the other MLX families do
QUANT_METHODS = {"mlx": ("mlx",)}
KERNEL_PACKAGE = "tensorfold.kernels.deepseek.v41"
KERNEL_VERSION = "v1"
LEAST_MLX = (0, 32, 2)


def check(model_dir: str | Path) -> None:
    """Refuse anything this reading of the architecture has not seen, from config.json alone (no MLX import).

    The gate is structural on purpose: a well-formed V4.1 config is understood today, and ``load`` reports the
    engine still owed for it. What must never pass is a config whose fields mean something this reading ignores.
    """

    from tensorfold.families import read_config
    from tensorfold.families.deepseek_v41.config import Config

    Config.from_dict(read_config(model_dir))


def load(model_dir: Path, **options: Any) -> tuple[Any, Any]:
    """The MLX engine and its tokenizer: not written yet (M0)."""

    from tensorfold.families.deepseek_v41.config import summary, unsupported
    from tensorfold.families import read_config

    config = read_config(model_dir)
    gaps = unsupported(config)
    raise NotImplementedError(
        f"deepseek_v41 has no engine yet: its config is read, {len(gaps)} pieces are still owed "
        f"({', '.join(gap.key for gap in gaps)}). Run "
        f"`python -m tensorfold.families.deepseek_v41.config {model_dir}` for the list.\n{summary(config)}")


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load (M1)."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}