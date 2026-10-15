# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""tools/bigbrain/footprint.py -- K+V elements one token adds to a model's cache, from its config.json.

The one rule build_registry.py uses for the registry's `kv_elements_per_token`, and so the one the capacity planner prices
the KV cache with. It is a lower bound by design: every branch returns what the architecture really caches, and a missing
field answers 0, never a guess, because an invented cache size could refuse a plan that fits.
Its tests are tests/test_bigbrain_footprint.py (one case per architecture family).
"""
from __future__ import annotations

from typing import Optional, Tuple

#: Architectures whose sliding-window layers alternate with global ones when
#: config.json carries no explicit layer_types.
_ALTERNATING_SWA = {"gemma2": 2}


def kv_shape(cfg: dict) -> Tuple[int, int, Optional[int]]:
    """(full-context elements per token, sliding elements per token, window).

    Every branch returns what the architecture really caches; when a field is
    missing the answer is 0, never a guess -- vram_fit's estimate is a lower
    bound, and an invented cache size could refuse a load that fits."""
    heads = cfg.get("num_attention_heads") or 0
    kv_heads = cfg.get("num_key_value_heads") or heads
    head_dim = cfg.get("head_dim") or ((cfg.get("hidden_size") or 0) // heads if heads else 0)
    layers = cfg.get("num_hidden_layers") or 0
    per_layer = int(kv_heads) * int(head_dim) * 2
    window = cfg.get("sliding_window")
    window = int(window) if isinstance(window, (int, float)) and window > 0 else None

    blocks = cfg.get("block_attention")
    if isinstance(blocks, list) and blocks:
        # Nemotron-NAS: each block says whether it has attention at all and how
        # many query heads share a KV head.
        full = 0
        for att in blocks:
            if not isinstance(att, dict) or att.get("no_op") or att.get("replace_with_linear"):
                continue
            group = att.get("n_heads_in_group")
            if not group:
                continue
            full += (int(heads) // int(group)) * int(head_dim) * 2
        return full, 0, None

    types = cfg.get("layer_types")
    if isinstance(types, list) and types:
        full = sum(1 for t in types if t in ("full_attention", "attention"))
        sliding = sum(1 for t in types if t == "sliding_attention")
        return full * per_layer, (sliding * per_layer if window else 0), (window if sliding else None)

    model_type = str(cfg.get("model_type") or "")
    if window and model_type in _ALTERNATING_SWA:
        every = _ALTERNATING_SWA[model_type]
        full_layers = layers // every
        return full_layers * per_layer, (layers - full_layers) * per_layer, window
    if window and model_type == "gemma3":
        pattern = cfg.get("sliding_window_pattern") or 6
        full_layers = layers // int(pattern)
        return full_layers * per_layer, (layers - full_layers) * per_layer, window
    # Everything else caches every layer at full length. qwen2's config carries
    # a sliding_window it does not use unless use_sliding_window is true.
    return layers * per_layer, 0, None
