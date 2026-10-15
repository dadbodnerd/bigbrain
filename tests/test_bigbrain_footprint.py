# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""tools/bigbrain/footprint.py: K+V elements one token adds to a model's cache, per architecture family. The first cases pin the
rule for the hybrid Qwen, Nemotron-NAS and Gemma 2/3; the golden table at the end pins the answer for seven real-world
configs, so a change to the rule shows up as a changed number here and not as a silently different plan."""
import pytest

from tools.bigbrain.footprint import kv_shape


def test_kv_shape_counts_only_full_attention_layers_of_the_hybrid_qwen():
    types = (["linear_attention"] * 3 + ["full_attention"]) * 16
    cfg = {"model_type": "qwen3_5_text", "num_hidden_layers": 64, "num_attention_heads": 24,
           "num_key_value_heads": 4, "head_dim": 256, "layer_types": types}
    assert kv_shape(cfg) == (16 * 4 * 256 * 2, 0, None)


def test_kv_shape_reads_nemotron_block_configs():
    blocks = ([{"n_heads_in_group": 8, "no_op": False}] * 49
              + [{"n_heads_in_group": None, "no_op": True}] * 31)
    cfg = {"model_type": "nemotron-nas", "num_hidden_layers": 80, "num_attention_heads": 64,
           "hidden_size": 8192, "block_attention": blocks}
    assert kv_shape(cfg) == (49 * 8 * 128 * 2, 0, None)


def test_kv_shape_splits_gemma2_into_global_and_sliding_layers():
    cfg = {"model_type": "gemma2", "num_hidden_layers": 42, "num_attention_heads": 16,
           "num_key_value_heads": 8, "head_dim": 256, "sliding_window": 4096}
    per = 8 * 256 * 2
    assert kv_shape(cfg) == (21 * per, 21 * per, 4096)


def test_kv_shape_ignores_qwen2s_unused_sliding_window():
    cfg = {"model_type": "qwen2", "num_hidden_layers": 48, "num_attention_heads": 40,
           "num_key_value_heads": 8, "head_dim": 128, "sliding_window": 131072}
    assert kv_shape(cfg) == (48 * 8 * 128 * 2, 0, None)


def test_kv_shape_answers_zero_rather_than_guessing():
    assert kv_shape({}) == (0, 0, None)


GOLDEN = [
    ({"model_type": "gpt_oss", "num_hidden_layers": 36, "num_attention_heads": 64, "num_key_value_heads": 8,
      "head_dim": 64, "sliding_window": 128, "layer_types": ["sliding_attention", "full_attention"] * 18},
     (18 * 8 * 64 * 2, 18 * 8 * 64 * 2, 128)),
    ({"model_type": "gemma3", "num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 8,
      "head_dim": 256, "sliding_window": 1024, "sliding_window_pattern": 6},
     (8 * 8 * 256 * 2, 40 * 8 * 256 * 2, 1024)),
    ({"model_type": "llama", "num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096},
     (32 * 32 * 128 * 2, 0, None)),
    ({"model_type": "qwen3_5_text", "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
      "head_dim": 256, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16},
     (16 * 4 * 256 * 2, 0, None)),
    ({"model_type": "nemotron-nas", "num_hidden_layers": 80, "num_attention_heads": 64, "hidden_size": 8192,
      "block_attention": [{"n_heads_in_group": 8, "no_op": False}] * 49 + [{"n_heads_in_group": None, "no_op": True}] * 31},
     (49 * 8 * 128 * 2, 0, None)),
    ({"model_type": "gemma2", "num_hidden_layers": 42, "num_attention_heads": 16, "num_key_value_heads": 8,
      "head_dim": 256, "sliding_window": 4096},
     (21 * 8 * 256 * 2, 21 * 8 * 256 * 2, 4096)),
    ({}, (0, 0, None)),
]


@pytest.mark.parametrize("cfg,expected", GOLDEN, ids=[c.get("model_type", "empty") for c, _ in GOLDEN])
def test_kv_shape_golden_table(cfg, expected):
    assert kv_shape(cfg) == expected
