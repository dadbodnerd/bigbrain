# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""The capacity-aware BigBrain planner (tools/bigbrain/controller.py).

Layers are split across the chosen GPUs in proportion to each GPU's USABLE
memory (free - headroom - the node's fixed costs), respecting non-uniform
layers (embedding on node 0, lm_head on the last, hybrid models' cache only on
the attention layers), and refused with a reason when they cannot hold the
model at the asked context. Pure: no ssh, no cards.

    C:/Python314/python.exe -m pytest tests/test_bigbrain_planner.py -q -p no:cacheprovider
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bigbrain import controller as ctl  # noqa: E402

MIB = 1024 * 1024


def shape(layers=10, layer_mib=100.0, embed_mib=0.0, head_mib=0.0, kv_per_token_per_layer=0,
          layer_types=None, cache_bytes=2.0, name="M"):
    """A synthetic model: `layers` blocks of layer_mib each."""
    info = {"layers": layers, "layer_mib": [layer_mib] * layers, "embed_mib": embed_mib,
            "head_mib": head_mib, "kv_elements_per_token": kv_per_token_per_layer * (
                layers if not layer_types else sum(1 for t in layer_types if "full" in t)),
            "layer_types": layer_types}
    return ctl.model_shape(name, info, cache_bytes=cache_bytes)


def gpu(host, idx, free, total=24576, name_="RTX 3090", card_key=None, model_dir="/m", peers=(),
        host_name=None):
    return ctl.GpuCapacity(host=host, gpu=idx, total_mib=total, free_mib=free,
                           host_name=host_name or host, card_key=card_key or f"{host}_{idx}",
                           name=name_, nvlink_peers=tuple(peers), model_dir=model_dir,
                           python_bin="/py")


KW = dict(headroom_mib=0, node_overhead_mib=0)


# ── the model shape ─────────────────────────────────────────────────────

def test_shape_estimates_embed_and_head_from_config_facts_when_unmeasured():
    s = ctl.model_shape("X", {"layers": 4, "size_gib": 4.0, "hidden_size": 1024, "vocab_size": 32768,
                              "kv_elements_per_token": 4 * 2048})
    # embed = vocab x hidden x 2 B = 64 MiB; head (8-bit default) = 32 MiB
    assert s.embed_mib == pytest.approx(64.0)
    assert s.head_mib == pytest.approx(32.0)
    assert sum(s.layer_mib) == pytest.approx(4096 - 96)
    assert len(set(round(x, 6) for x in s.layer_mib)) == 1          # uniform when unmeasured
    # fp16 cache: 2048 elements per token per layer x 2 bytes
    assert s.kv_bytes_per_token == [4096.0] * 4


def test_hybrid_shape_puts_cache_only_on_attention_layers():
    types = ["linear_attention"] * 3 + ["full_attention"]
    s = ctl.model_shape("H", {"layers": 4, "layer_mib": [10.0] * 4, "embed_mib": 0, "head_mib": 0,
                              "kv_elements_per_token": 1000, "layer_types": types})
    assert s.kv_bytes_per_token == [0.0, 0.0, 0.0, 2000.0]
    assert s.hybrid is True
    assert s.recurrent_mib[0] > 0 and s.recurrent_mib[3] == 0


def test_shape_without_layers_or_kv_facts_is_refused():
    with pytest.raises(ctl.PlanRefused, match="layers"):
        ctl.model_shape("X", {"size_gib": 4.0, "kv_elements_per_token": 10})
    with pytest.raises(ctl.PlanRefused, match="kv"):
        ctl.model_shape("X", {"layers": 4, "size_gib": 4.0})


# ── proportional split ──────────────────────────────────────────────────

def test_split_is_proportional_to_usable_memory_not_even():
    s = shape(layers=30, layer_mib=100.0)
    # usable 2000 vs 1000: twice the room, twice the layers
    p = ctl.plan_capacity(s, [gpu("a", 0, 2000), gpu("b", 0, 1000)], ctx=1024, **KW)
    assert p.ok, p.reason
    assert [n["layers"] for n in p.nodes] == [[0, 20], [20, 30]]
    assert p.split == "proportional"
    assert p.nodes[0]["need_mib"] == 2000 and p.nodes[1]["need_mib"] == 1000


def test_even_capacity_gives_an_even_split():
    s = shape(layers=28, layer_mib=50.0)
    p = ctl.plan_capacity(s, [gpu("a", 0, 5000), gpu("b", 0, 5000)], ctx=1024, **KW)
    assert [n["layers"] for n in p.nodes] == [[0, 14], [14, 28]]


def test_embedding_and_head_take_room_on_the_end_nodes():
    """Node 0 carries the embedding, the last node the lm_head: with equal free
    memory, those nodes get fewer blocks than a middle node."""
    s = shape(layers=30, layer_mib=100.0, embed_mib=500.0, head_mib=500.0)
    p = ctl.plan_capacity(s, [gpu("a", 0, 1500), gpu("b", 0, 1500), gpu("c", 0, 1500)], ctx=512, **KW)
    assert p.ok, p.reason
    counts = [n["layers"][1] - n["layers"][0] for n in p.nodes]
    assert counts[1] > counts[0] and counts[1] > counts[2]
    assert sum(counts) == 30
    assert p.nodes[0]["weights_mib"] >= 500 and p.nodes[2]["weights_mib"] >= 500


def test_cache_scales_with_context_and_moves_the_split():
    """Each layer's KV grows with the context; a hybrid model's linear layers do
    not, so the context changes how many attention layers a card can take."""
    s = shape(layers=10, layer_mib=100.0, kv_per_token_per_layer=512)   # 1 KiB/token/layer fp16
    small = ctl.plan_capacity(s, [gpu("a", 0, 2000), gpu("b", 0, 2000)], ctx=1024, **KW)
    assert small.ok and small.nodes[0]["cache_mib"] == 5      # 5 layers x 1 MiB
    big = ctl.plan_capacity(s, [gpu("a", 0, 2000), gpu("b", 0, 2000)], ctx=1024000, **KW)
    assert not big.ok and "1,024,000" in big.reason


def test_non_uniform_layers_are_respected():
    """Measured per-layer sizes: the heavy layers get spread, never assumed even."""
    info = {"layers": 6, "layer_mib": [500, 500, 100, 100, 100, 100], "embed_mib": 0, "head_mib": 0,
            "kv_elements_per_token": 0}
    s = ctl.model_shape("U", info)
    p = ctl.plan_capacity(s, [gpu("a", 0, 1000), gpu("b", 0, 1000)], ctx=128, **KW)
    assert p.ok, p.reason
    # an even split by layer COUNT ([0,3] / [3,6]) would put 1,100 MiB on a 1,000 MiB card
    assert [n["layers"] for n in p.nodes] == [[0, 1], [1, 6]]
    assert [n["need_mib"] for n in p.nodes] == [500, 900]


def test_the_bottleneck_fallback_finds_a_split_proportional_cutting_misses():
    """Big layers make the proportional cut overflow one card; the min-max
    fallback still finds the contiguous split that fits."""
    info = {"layers": 3, "layer_mib": [750, 260, 2590], "embed_mib": 0, "head_mib": 0,
            "kv_elements_per_token": 0}
    s = ctl.model_shape("B", info)
    # proportional target for node 0 is 900 MiB: the nearest cut (1,010) overflows its 1,000
    p = ctl.plan_capacity(s, [gpu("a", 0, 1000), gpu("b", 0, 3000)], ctx=128, **KW)
    assert p.ok, p.reason
    assert p.split == "bottleneck"
    assert [n["layers"] for n in p.nodes] == [[0, 1], [1, 3]]
    assert all(n["need_mib"] <= n["usable_mib"] for n in p.nodes)


def test_headroom_and_node_overhead_count_against_every_gpu():
    s = shape(layers=10, layer_mib=100.0)
    p = ctl.plan_capacity(s, [gpu("a", 0, 1100)], ctx=128, headroom_mib=100, node_overhead_mib=0)
    assert p.ok
    p = ctl.plan_capacity(s, [gpu("a", 0, 1100)], ctx=128, headroom_mib=100, node_overhead_mib=50)
    assert not p.ok
    n = p.nodes[0]
    assert n["usable_mib"] == 1000 and n["headroom_mib"] == 100 and n["overhead_mib"] == 50


# ── refusals, each with a reason that names the numbers ─────────────────

def test_refuses_when_the_gpus_cannot_hold_the_model():
    s = shape(layers=10, layer_mib=1000.0)
    p = ctl.plan_capacity(s, [gpu("a", 0, 4000), gpu("b", 0, 4000)], ctx=4096, **KW)
    assert not p.ok
    assert "cannot hold" in p.reason and "short" in p.reason
    assert p.nodes and all("free_mib" in n for n in p.nodes)        # per-GPU numbers still shown
    assert p.cluster is None


def test_refuses_a_gpu_whose_pc_lacks_the_model():
    s = shape(layers=10, layer_mib=10.0)
    p = ctl.plan_capacity(s, [gpu("a", 0, 4000), gpu("b", 0, 4000, model_dir="", host_name="beta")],
                          ctx=128, **KW)
    assert not p.ok and "beta" in p.reason and "not on" in p.reason


def test_refuses_duplicates_and_more_gpus_than_layers():
    s = shape(layers=2, layer_mib=10.0)
    p = ctl.plan_capacity(s, [gpu("a", 0, 4000), gpu("a", 0, 4000)], ctx=128, **KW)
    assert not p.ok and "twice" in p.reason
    p = ctl.plan_capacity(s, [gpu("a", 0, 4000), gpu("a", 1, 4000), gpu("b", 0, 4000)], ctx=128, **KW)
    assert not p.ok and "layers" in p.reason


def test_refuses_a_node_that_cannot_even_hold_its_fixed_part():
    s = shape(layers=10, layer_mib=10.0, embed_mib=3000.0)
    p = ctl.plan_capacity(s, [gpu("a", 0, 2000), gpu("b", 0, 20000)], ctx=128, **KW)
    assert not p.ok and "node 0" in p.reason and "embedding" in p.reason


# ── the cluster plan and the links ──────────────────────────────────────

def test_the_capacity_plan_builds_a_valid_cluster_plan():
    s = shape(layers=30, layer_mib=100.0)
    p = ctl.plan_capacity(s, [gpu("10.0.0.1", 0, 2000, model_dir="/a/M"),
                              gpu("10.0.0.2", 1, 1000, model_dir="/b/M")],
                          ctx=8192, stage_root="/tmp/bigbrain/bb-1", http_port=16980,
                          api_token="T0K", **KW)
    c = p.cluster
    assert c is not None and c.validate() == []
    assert c.max_num_tokens == 8192 and c.stage_root == "/tmp/bigbrain/bb-1"
    assert [(n.host, n.gpu, n.slice_start, n.slice_end, n.model_dir, n.python_bin) for n in c.nodes] == [
        ("10.0.0.1", 0, 0, 20, "/a/M", "/py"), ("10.0.0.2", 1, 20, 30, "/b/M", "/py")]
    c0, c1 = c.build_commands()
    assert "--http-port 16980" in c0 and "--api-token T0K" in c0
    assert "--http-port" not in c1 and "--api-token" not in c1
    # serving: no one-shot prompt; device order pinned to nvidia-smi's
    assert "--prompt" not in c0
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in c0 and c0.startswith("cd /tmp/bigbrain/bb-1 && ")


def test_links_say_same_host_or_lan_with_an_estimate():
    s = shape(layers=30, layer_mib=10.0)
    gpus = [gpu("g", 0, 9000, peers=(1,)), gpu("g", 1, 9000, peers=(0,)), gpu("r", 1, 9000)]
    p = ctl.plan_capacity(s, gpus, ctx=128, link_mb_s={"g": 110, "r": 1150}, **KW)
    assert p.ok
    assert p.links[0]["kind"] == "same-host" and p.links[0]["nvlink"] is True
    assert p.links[1]["kind"] == "lan" and p.links[1]["est_mb_s"] == 110      # the slower end
    assert p.links[0]["est_mb_s"] > p.links[1]["est_mb_s"]


def test_to_dict_has_the_contract_shape():
    s = shape(layers=4, layer_mib=10.0)
    d = ctl.plan_capacity(s, [gpu("10.0.0.1", 0, 9000, host_name="alpha", card_key="ALPHA_GPU0")],
                          ctx=32768, **KW).to_dict()
    assert {"ok", "reason", "model", "ctx", "nodes", "total_need_mib", "links"} <= set(d)
    n = d["nodes"][0]
    assert {"rank", "host", "host_name", "gpu", "card_key", "layers", "need_mib", "free_mib"} <= set(n)
    assert n["host_name"] == "alpha" and n["card_key"] == "ALPHA_GPU0" and n["layers"] == [0, 4]
    assert d["ctx"] == 32768 and d["links"] == []
    assert "cluster" not in d and "api_token" not in str(d)


# ── auto GPU choice: fewest machines, fewest GPUs, NVLink pairs first ───

def test_auto_prefers_one_machine_over_two():
    s = shape(layers=20, layer_mib=1000.0)               # 20 GiB of weights
    cands = [gpu("beta", 1, 12000), gpu("alpha", 0, 12000, peers=(1,)), gpu("alpha", 1, 12000, peers=(0,)),
             gpu("beta", 2, 12000)]
    p = ctl.choose_gpus(s, cands, ctx=128, **KW)
    assert p.ok, p.reason
    assert {n["host"] for n in p.nodes} == {"alpha"}            # beta 1+2 also fits: NVLink wins the tie
    assert [n["gpu"] for n in p.nodes] == [0, 1]


def test_auto_takes_the_fewest_gpus_that_fit():
    s = shape(layers=10, layer_mib=1000.0)
    cands = [gpu("a", 0, 12000), gpu("a", 1, 12000), gpu("a", 2, 23000)]
    p = ctl.choose_gpus(s, cands, ctx=128, **KW)
    assert p.ok and [n["gpu"] for n in p.nodes] == [2]


def test_auto_spans_machines_only_when_it_must_and_groups_each_host():
    s = shape(layers=40, layer_mib=1000.0)               # 40 GiB: no single PC here holds it
    cands = [gpu("alpha", 0, 18000), gpu("beta", 1, 22000), gpu("alpha", 1, 18000), gpu("beta", 2, 11000)]
    p = ctl.choose_gpus(s, cands, ctx=128, **KW)
    assert p.ok, p.reason
    hosts = [n["host"] for n in p.nodes]
    # each host's GPUs are adjacent: one LAN hop, not two
    assert sum(1 for a, b in zip(hosts, hosts[1:]) if a != b) == 1
    assert len(p.links) == len(p.nodes) - 1


def test_auto_refuses_when_nothing_fits_and_says_how_short():
    s = shape(layers=10, layer_mib=10000.0)
    p = ctl.choose_gpus(s, [gpu("a", 0, 20000), gpu("b", 0, 20000)], ctx=128, **KW)
    assert not p.ok and "short" in p.reason


def test_auto_with_no_candidates_says_so():
    p = ctl.choose_gpus(shape(), [], ctx=128, **KW)
    assert not p.ok and "no eligible GPU" in p.reason


# ── the CLI keeps the even split for a raw --model-dir ───────────────────

def test_cli_even_split_unchanged_for_model_dir():
    plan = ctl.plan_from_args(["--model-dir", "/m", "--total-blocks", "28", "--gpus", "h0:0", "--gpus", "h1:1"])
    assert [(n.slice_start, n.slice_end) for n in plan.nodes] == [(0, 14), (14, 28)]


def test_cli_capacity_split_uses_injected_readings(tmp_path, monkeypatch):
    import json
    from tools.bigbrain import registry as reg
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"version": 1, "nodes": {
        "a": {"host": "10.0.0.1", "launch": True, "python": "/pa", "model_root": "/m",
              "gpus": [{"index": 0, "total_mib": 24576, "card_key": "A0"}]},
        "b": {"host": "10.0.0.2", "launch": True, "python": "/pb", "model_root": "/m",
              "gpus": [{"index": 0, "total_mib": 24576, "card_key": "B0"}]}},
        "models": {"M": {"layers": 30, "layer_mib": [100.0] * 30, "embed_mib": 0, "head_mib": 0,
                         "kv_elements_per_token": 0, "paths": {"a": "/m/M", "b": "/m/M"}}}}),
        encoding="utf-8")
    monkeypatch.setenv(reg.ENV_PATH, str(p))
    free = {("10.0.0.1", 0): 2000, ("10.0.0.2", 0): 1000}
    monkeypatch.setattr(ctl, "_cli_free_mib", lambda reg_, cards: dict(free))
    plan = ctl.plan_from_args(["--model", "M", "--gpus", "10.0.0.1:0", "--gpus", "10.0.0.2:0",
                               "--ctx", "1024", "--node-overhead-mib", "0", "--headroom-mib", "0"])
    assert [(n.slice_start, n.slice_end) for n in plan.nodes] == [(0, 20), (20, 30)]
    assert [n.python_bin for n in plan.nodes] == ["/pa", "/pb"]
