# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""The controller side of the speed work: two bugs it once had (serving mode launched node 0 with a flag
worker.py rejected; server.py was never staged), the slot/speculation flags it passes to the nodes, and the
capacity planner pricing N slots and gpt-oss's sliding-window state. No GPU, no ssh."""
from __future__ import annotations

import ast
import shlex
from pathlib import Path

import pytest

import tools.bigbrain.controller as ctl
from tools.bigbrain import registry as bbreg
from tools.bigbrain import server as srv
from tools.bigbrain import worker as wk
from tools.bigbrain.controller import PACKAGE_FILES, ClusterPlan, NodeSpec, plan_from_args, stage_package
from tools.bigbrain.paged import PagedSlots

PKG = Path(__file__).resolve().parent.parent / "tools" / "bigbrain"
MIB = 1024 * 1024


def _node(rank, host, gpu, start, end, in_port):
    return NodeSpec(rank=rank, host=host, gpu=gpu, slice_start=start, slice_end=end, in_port=in_port)


def _serve_plan(**kw):
    nodes = [_node(0, "h0", 0, 0, 18, 0), _node(1, "h1", 0, 18, 36, 6001)]
    base = dict(model_dir="/m", nodes=nodes, total_blocks=36, token_port=7000, http_port=16980,
                api_token="T0K", python_bin="python")
    base.update(kw)
    return ClusterPlan(**base)


# ---- BUG 2: server.py (and everything it and the worker import) must be staged

def _relative_imports(path: Path):
    """The tools/bigbrain modules a file imports: `from .x import ...` / `from . import x`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                mods.add(node.module.split(".")[0])
            else:
                mods.update(a.name for a in node.names)
    return mods


def test_server_py_is_staged():
    assert "tools/bigbrain/server.py" in PACKAGE_FILES
    shipped = []
    stage_package("h0", "u", str(PKG), "/tmp/bigbrain", stager=lambda h, u, rp, c: shipped.append(rp))
    assert "/tmp/bigbrain/tools/bigbrain/server.py" in shipped


def test_the_staged_package_is_closed_under_relative_imports():
    """Any module a staged module imports from tools/bigbrain must be staged too -- a node would
    die with ImportError otherwise (this is how server.py went missing)."""
    staged = {Path(f).stem for f in PACKAGE_FILES if f.startswith("tools/bigbrain/")}
    seen, todo = set(), ["worker", "server"]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        for dep in _relative_imports(PKG / f"{m}.py"):
            if (PKG / f"{dep}.py").exists():
                assert dep in staged, f"{m}.py imports {dep}.py, which the controller does not stage"
                todo.append(dep)
    assert {"worker", "server", "pipe", "paged", "spec", "engine", "draft"} <= staged


def test_every_staged_file_exists():
    for rel in PACKAGE_FILES:
        if rel.endswith("tools/__init__.py"):
            continue            # the repo has no tools/__init__.py; stage_package ships it empty
        assert (PKG.parent.parent / rel).is_file(), rel


# ---- BUG 1: the serving command line must be accepted by what it runs

def _argv(cmd: str, module: str):
    return shlex.split(cmd.split(f" -m {module} ", 1)[1])


def test_serving_commands_parse_node_zero_as_the_server_and_the_rest_as_workers():
    plan = _serve_plan(slots=4, spec="ngram", spec_k=4)
    c0, c1 = plan.build_commands()
    a0 = srv.build_arg_parser().parse_args(_argv(c0, "tools.bigbrain.worker"))
    assert a0.http_port == 16980 and a0.api_token == "T0K" and a0.rank == 0
    a1 = wk.build_arg_parser().parse_args(_argv(c1, "tools.bigbrain.worker"))
    assert a1.rank == 1 and wk.effective_slots(a1) == 4 and wk.effective_max_draft(a1) == 4
    # the worker's own parser has no HTTP flags: that was the bug -- worker.main now hands them to the server
    with pytest.raises(SystemExit):
        wk.build_arg_parser().parse_args(_argv(c0, "tools.bigbrain.worker"))


def test_serving_launch_of_node_zero_reaches_server_main(monkeypatch):
    got = []
    monkeypatch.setattr(srv, "main", lambda argv=None: (got.append(list(argv)), 0)[1])
    plan = _serve_plan()
    argv0 = _argv(plan.build_commands()[0], "tools.bigbrain.worker")
    assert wk.main(argv0) == 0 and got == [argv0]


# ---- slots + speculation flags

def test_a_plain_plan_adds_no_speed_flags():
    for c in _serve_plan().build_commands():
        assert "--slots" not in c and "--spec" not in c and "--max-draft" not in c


def test_batched_plan_passes_the_same_slots_to_every_node_and_speculation_to_node_zero():
    plan = _serve_plan(slots=8, spec="model", spec_k=5, spec_model="/m/gpt-oss-20b",
                       draft_device="cuda:1", inflight=3, mb_size=2)
    c0, c1 = plan.build_commands()
    for c in (c0, c1):
        assert "--slots 8" in c and "--max-draft 5" in c
    for flag in ("--spec-draft model", "--spec-k 5", "--spec-model /m/gpt-oss-20b",
                 "--draft-device cuda:1", "--inflight 3", "--mb-size 2"):
        assert flag in c0, flag
        assert flag not in c1, flag
    plan2 = _serve_plan(slots=2)
    assert "--max-draft" not in plan2.build_commands()[0] and "--slots 2" in plan2.build_commands()[1]


def test_validate_checks_the_speed_settings():
    assert _serve_plan(slots=4, spec="ngram").validate() == []
    assert any("slots" in p for p in _serve_plan(slots=0).validate())
    assert any("spec must" in p for p in _serve_plan(spec="medusa").validate())
    assert any("spec_k" in p for p in _serve_plan(spec="ngram", spec_k=40).validate())
    assert any("spec_model" in p for p in _serve_plan(spec="model").validate())
    one_shot = _serve_plan(slots=4, http_port=0, api_token="")
    assert any("serving engine" in p for p in one_shot.validate())


def test_cli_serve_and_speed_flags_reach_the_plan():
    plan = plan_from_args(["--model-dir", "/m", "--total-blocks", "36", "--token-port", "7000",
                           "--node", "h0:0:0:18:0", "--node", "h1:0:18:36:6001",
                           "--serve-port", "16980", "--api-token", "T", "--slots", "4",
                           "--spec-draft", "ngram", "--spec-k", "6"])
    assert (plan.http_port, plan.api_token, plan.slots, plan.spec, plan.spec_k) == (16980, "T", 4, "ngram", 6)
    assert plan.validate() == []
    c0 = plan.build_commands()[0]
    assert "--http-port 16980" in c0 and "--slots 4" in c0 and "--spec-draft ngram" in c0


# ---- the planner: slots and gpt-oss's sliding-window state

GPTOSS = {"arch": "gpt_oss", "layers": 36, "layer_mib": [1600.0] * 36, "embed_mib": 1100.0,
          "head_mib": 1100.0, "hidden_size": 2880, "vocab_size": 201088,
          "num_key_value_heads": 8, "head_dim": 64,
          "layer_types": ["sliding_attention", "full_attention"] * 18,
          "kv_elements_per_token": 18 * 8 * 64 * 2,        # the FULL layers only (kv_shape)
          "hybrid": False}


def test_registry_says_gpt_oss_has_recurrent_state_even_though_it_is_not_hybrid():
    assert not bbreg.is_hybrid(GPTOSS)
    assert bbreg.has_swa_state(GPTOSS) and bbreg.has_recurrent_state(GPTOSS)
    assert not bbreg.has_recurrent_state({"arch": "llama", "layer_types": ["full_attention"] * 4})
    assert bbreg.has_recurrent_state({"arch": "qwen3_5_text"})
    # window 128 -> 768 state positions x 8 heads x 64 x (K,V) x 2 bytes = 1.5 MiB per layer
    assert bbreg.swa_state_mib(GPTOSS) == pytest.approx(768 * 8 * 64 * 2 * 2 / MIB)


def test_gpt_oss_sliding_layers_are_priced_as_state_not_as_paged_kv():
    sh = ctl.model_shape("g", GPTOSS)
    assert sh.recurrent_state and not sh.hybrid
    sliding = [i for i in range(36) if i % 2 == 0]
    full = [i for i in range(36) if i % 2 == 1]
    assert all(sh.kv_bytes_per_token[i] == 0 for i in sliding)
    assert all(sh.recurrent_mib[i] == pytest.approx(1.5) for i in sliding)
    assert all(sh.recurrent_mib[i] == 0 for i in full)
    # the KV facts cover the 18 full layers: 8 heads x 64 x 2 x 2 bytes per token each
    assert all(sh.kv_bytes_per_token[i] == pytest.approx(8 * 64 * 2 * 2) for i in full)
    # total KV at 32k is the full layers' alone (the sliding layers add a context-independent state)
    total_kv = sum(sh.cache_mib(i, 32768) for i in range(36))
    assert total_kv == pytest.approx(18 * 8 * 64 * 2 * 2 * 32768 / MIB + 18 * 1.5)


def test_slots_multiply_the_cache_and_the_state_but_not_the_weights():
    sh = ctl.model_shape("g", GPTOSS)
    one = sh.cache_mib(1, 32768)
    four = sh.cache_mib(1, 32768, slots=4)
    assert four == pytest.approx(PagedSlots.tokens_for(4, 32768) * 8 * 64 * 2 * 2 / MIB)
    assert four > 4 * one - 1                       # 4 slots + a slack page each
    assert sh.cache_mib(0, 32768, slots=4) == pytest.approx(4 * 1.5)
    assert sh.costs(32768)[0] + 1 > sh.layer_mib[0]
    # a single-sequence run keeps the exact legacy size
    assert sh.cache_mib(1, 32768, slots=1) == pytest.approx(32768 * 8 * 64 * 2 * 2 / MIB)


def _gpus(n, free):
    return [ctl.GpuCapacity(host=f"h{i}", gpu=0, total_mib=24576, free_mib=free, host_name=f"h{i}",
                            card_key=f"c{i}", name="RTX 3090", model_dir="/m", python_bin="/py")
            for i in range(n)]


def test_the_planner_refuses_slots_that_do_not_fit_and_says_why():
    sh = ctl.model_shape("g", GPTOSS)
    kw = dict(headroom_mib=0, node_overhead_mib=0)
    ok1 = ctl.plan_capacity(sh, _gpus(3, 22000), 32768, **kw)
    assert ok1.ok, ok1.reason
    many = ctl.plan_capacity(sh, _gpus(3, 22000), 32768, slots=8, **kw)
    assert not many.ok and "cannot hold" in many.reason
    cache1 = sum(sh.cache_mib(i, 32768) for i in range(36))
    cache8 = sum(sh.cache_mib(i, 32768, slots=8) for i in range(36))
    assert cache8 > 7 * cache1 and f"{cache8:,.0f} cache" in many.reason
    ok4 = ctl.plan_capacity(sh, _gpus(3, 23000), 16384, slots=4, http_port=16980, api_token="T", **kw)
    assert ok4.ok and ok4.cluster.slots == 4
    assert "--slots 4" in ok4.cluster.build_commands()[1]


def test_the_draft_model_reserves_room_on_node_zero():
    sh = ctl.model_shape("g", GPTOSS)
    kw = dict(headroom_mib=0, node_overhead_mib=0, http_port=16980, api_token="T")
    without = ctl.plan_capacity(sh, _gpus(3, 30000), 8192, slots=2, spec="ngram", **kw)
    with_draft = ctl.plan_capacity(sh, _gpus(3, 30000), 8192, slots=2, spec="model", spec_k=4,
                                   spec_model="/m20", draft_mib=11000, **kw)
    assert without.ok and with_draft.ok
    # node 0 carries the 11 GB draft, so it gets fewer layers than without one
    assert with_draft.nodes[0]["layers"][1] - with_draft.nodes[0]["layers"][0] < \
        without.nodes[0]["layers"][1] - without.nodes[0]["layers"][0]
    assert with_draft.cluster.spec == "model" and with_draft.cluster.spec_model == "/m20"
    # a draft model sharing the card with no footprint given is flagged
    vague = ctl.plan_capacity(sh, _gpus(3, 30000), 8192, spec="model", spec_model="/m20", **kw)
    assert any("draft_mib" in w for w in vague.warnings)


def _two_on_one_host():
    mk = lambda host, gpu: ctl.GpuCapacity(host=host, gpu=gpu, total_mib=24576, free_mib=30000, host_name=host,  # noqa: E731
                                           card_key=f"{host}{gpu}", name="RTX 3090", model_dir="/m",
                                           python_bin="/py")
    return [mk("h0", 0), mk("h0", 1), mk("h1", 0)]


def test_a_draft_device_must_be_a_card_of_the_plan_and_carries_the_draft():
    """The draft's card is a card the reservation holds, parks and unloads: it has to be one of the plan's
    own cards on node 0's machine, and THAT node (not node 0) pays for the draft."""
    sh = ctl.model_shape("g", GPTOSS)
    kw = dict(headroom_mib=0, node_overhead_mib=0, http_port=16980, api_token="T", slots=2,
              spec="model", spec_model="/m20", draft_mib=11000)
    on0 = ctl.plan_capacity(sh, _two_on_one_host(), 8192, **kw)
    on1 = ctl.plan_capacity(sh, _two_on_one_host(), 8192, draft_device="cuda:1", **kw)
    assert on0.ok and on1.ok
    n = lambda plan, i: plan.nodes[i]["layers"][1] - plan.nodes[i]["layers"][0]   # noqa: E731
    assert n(on0, 0) < n(on1, 0) and n(on1, 1) < n(on0, 1)       # the draft moves from node 0 to node 1
    assert on1.cluster.draft_device == "cuda:1"
    # a device that is not in the plan (or on another machine) is refused with the reason
    for dev in ("cuda:3", "cuda:2"):
        bad = ctl.plan_capacity(sh, _two_on_one_host(), 8192, draft_device=dev, **kw)
        assert not bad.ok and "not one of the plan's cards" in bad.reason
    only_other_host = _two_on_one_host()[2:]
    assert not ctl.plan_capacity(sh, only_other_host, 8192, draft_device="cuda:1", **kw).ok


def test_the_draft_footprint_counts_its_weights_and_its_own_cache():
    d20 = {"arch": "gpt_oss", "layers": 24, "size_gib": 11.2, "hidden_size": 2880, "vocab_size": 201088,
           "num_key_value_heads": 8, "head_dim": 64, "layer_types": ["sliding_attention", "full_attention"] * 12,
           "kv_elements_per_token": 12 * 8 * 64 * 2}
    one = ctl.draft_footprint_mib(d20, 8192, slots=1)
    eight = ctl.draft_footprint_mib(d20, 32768, slots=8)
    assert 11.2 * 1024 < one < 11.2 * 1024 + 800
    # 8 slots x (32k + a slack page) of KV on the 12 full layers, plus a sliding-window state per slot
    kv = 12 * 8 * 64 * 2 * 2 * ctl.PagedSlots.tokens_for(8, 32768) / MIB
    assert eight == pytest.approx(11.2 * 1024 + kv + 12 * 8 * 1.5, rel=0.02)
    assert ctl.draft_footprint_mib(d20, 8192, 1, separate_device=True) == pytest.approx(one + 512)


def test_choose_gpus_counts_the_slots_in_its_quick_bound():
    sh = ctl.model_shape("g", GPTOSS)
    cands = _gpus(4, 22000)
    kw = dict(headroom_mib=0, node_overhead_mib=0)
    assert ctl.choose_gpus(sh, cands, 32768, **kw).ok
    bad = ctl.choose_gpus(sh, cands, 65536, slots=8, **kw)
    assert not bad.ok and "tokens" in bad.reason
    # the draft model counts too
    no_room = ctl.choose_gpus(sh, _gpus(3, 20000), 8192, spec="model", spec_model="/m", draft_mib=90000, **kw)
    assert not no_room.ok and "tokens" in no_room.reason


def test_a_gated_delta_net_keeps_k_extra_states_per_slot_a_sliding_window_does_not():
    qwen = {"arch": "qwen3_5_text", "layers": 4, "layer_mib": [100.0] * 4, "kv_elements_per_token": 4096,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]}
    sh = ctl.model_shape("q", qwen)
    lin, full = 0, 3
    assert sh.cache_mib(lin, 1024, slots=2, history=0) == pytest.approx(2 * 8.0)
    assert sh.cache_mib(lin, 1024, slots=2, history=4) == pytest.approx(2 * 8.0 * 5)
    assert sh.cache_mib(full, 1024, slots=2, history=4) == sh.cache_mib(full, 1024, slots=2, history=0)
    g = ctl.model_shape("g", GPTOSS)
    assert g.cache_mib(0, 1024, slots=2, history=4) == g.cache_mib(0, 1024, slots=2, history=0)   # SWA: in place


def test_watchdog_flag_reaches_every_node_and_is_off_by_default():
    base = ["--model-dir", "/m", "--total-blocks", "28", "--gpus", "h0:0", "--gpus", "h1:1"]
    plain = ctl.plan_from_args(base)
    assert plain.watchdog_s == 0.0 and not any("--watchdog-s" in c for c in plain.build_commands())
    plan = ctl.plan_from_args(base + ["--watchdog-s", "30"])
    cmds = plan.build_commands()
    assert len(cmds) == 2 and all(" --watchdog-s 30" in c for c in cmds) and plan.validate() == []
    too_small = ctl.plan_from_args(base + ["--watchdog-s", "1"])
    assert any("watchdog_s" in p for p in too_small.validate())
