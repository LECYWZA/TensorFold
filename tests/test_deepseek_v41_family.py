"""DeepSeek-V4.1-Flash's skeleton: the config gate, the gap list, and why this needs a family of its own.

The fixture is the released checkpoint's config.json, so these tests run without MLX, without a GPU and without
weights: they check what the engine reads, what it still owes, and that none of it is silent.
"""

import json
from pathlib import Path

import pytest

from tensorfold import families
from tensorfold.families import deepseek_v41
from tensorfold.families.deepseek_v4 import config as v4
from tensorfold.families.deepseek_v41 import config as v41

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41"
OFFICIAL = json.loads((FIXTURE / "config.json").read_text())


def test_the_family_is_discovered_by_its_model_types():
    found = families.families()
    assert "deepseek_v41" in found and "deepseek_v41_text" in found
    family = families.detect(FIXTURE)
    assert family.model_type == "deepseek_v41"
    assert family.title == "DeepSeek-V4.1-Flash"
    assert family.lanes is True


def test_the_released_config_parses():
    cfg = v41.Config.from_dict(OFFICIAL)
    assert cfg.num_hidden_layers == 40
    assert cfg.pool_ratios() == {1, 2}
    assert cfg.kv_source_layer_ids == [2, 8, 14, 20]
    assert cfg.index_source_layer_ids == [2, 8, 14, 20, 24, 28, 32, 36]
    assert cfg.candidate_source_layer_id == 20
    assert (cfg.engram_layer_ids, cfg.engram_max_ngram_size) == ([1, 14], 4)
    assert cfg.rms_norm_eps == 1e-20
    assert cfg.dspark_block_size == 5 and cfg.dspark_target_layer_ids == [37, 38, 39]
    assert cfg.vision["num_hidden_layers"] == 32


def test_the_v4_engine_refuses_it_which_is_why_this_family_exists():
    with pytest.raises(ValueError) as refused:
        v4.Config.from_dict(OFFICIAL)
    assert "compress_ratios" in str(refused.value)


def test_config_reads_only_its_own_model_type():
    other = {**OFFICIAL, "text_config": {**OFFICIAL["text_config"], "model_type": "deepseek_v4_text"}}
    with pytest.raises(ValueError) as refused:
        v41.Config.from_dict(other)
    assert "model_type" in str(refused.value)


def test_source_layer_ids_outside_the_stack_are_refused():
    broken = {**OFFICIAL, "text_config": {**OFFICIAL["text_config"], "kv_source_layer_ids": [2, 8, 99]}}
    with pytest.raises(ValueError) as refused:
        v41.Config.from_dict(broken)
    assert "outside the model's 40 layers" in str(refused.value)


def test_every_unimplemented_piece_is_named_and_placed():
    gaps = v41.unsupported(OFFICIAL)
    assert {gap.key for gap in gaps} == {"cross-layer-kv", "candidate-pool", "pool-ratios", "fp4-kv", "engram",
                                        "weight-format", "vision", "rms-eps", "swa-replay"}
    for gap in gaps:
        assert gap.lands_in, f"{gap.key} does not say where the work goes"


def test_the_gap_report_is_built_from_this_checkpoint():
    text = v41.summary(OFFICIAL)
    assert "40 layers" in text and "{0: 5, 2: 18, 1: 20}" in text
    assert "kv sources [2, 8, 14, 20]" in text and "layers [1, 14]" in text


def test_check_reads_the_released_config_today_and_load_says_what_is_owed():
    deepseek_v41.check(FIXTURE)                     # no MLX import, no weights: the gate answers now
    with pytest.raises(NotImplementedError) as owed:
        deepseek_v41.load(FIXTURE)
    assert "no engine yet" in str(owed.value)


def test_the_official_checkpoint_is_not_an_mlx_one_yet():
    assert families.quant_method(OFFICIAL) == "fp8"
    with pytest.raises(ValueError) as refused:
        families.require_readable(families.detect(FIXTURE), OFFICIAL, "mlx")
    assert "recipe" in str(refused.value)