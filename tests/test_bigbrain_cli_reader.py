# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""controller._cli_free_mib: the card reading behind the CLI's `--split capacity`. By default it is the built-in nvidia-smi-over-ssh
reader (nvidia_smi_free_mib); site.toml's [capacity] reader = "module:function" swaps in your own. A reader that cannot be
loaded, or an ssh that fails, is a PlanRefused that names the way out (`--split even`), not a bare traceback."""
import json
import subprocess
import sys
import types

import pytest

from bigbrain import controller as ctl
from bigbrain import registry as reg

REG = {"version": 1, "nodes": {
    "a": {"host": "10.0.0.1", "user": "ua", "launch": True, "python": "/pa", "model_root": "/m",
          "gpus": [{"index": 0, "total_mib": 24576, "card_key": "A0"}, {"index": 1, "total_mib": 24576, "card_key": "A1"}]},
    "b": {"host": "10.0.0.2", "launch": True, "python": "/pb", "model_root": "/m",
          "gpus": [{"index": 0, "total_mib": 24576, "card_key": "B0"}]}},
    "models": {"M": {"layers": 30, "layer_mib": [100.0] * 30, "embed_mib": 0, "head_mib": 0,
                     "kv_elements_per_token": 0, "paths": {"a": "/m/M", "b": "/m/M"}}}}


@pytest.fixture
def registry_env(tmp_path, monkeypatch):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(REG), encoding="utf-8")
    monkeypatch.setenv(reg.ENV_PATH, str(p))


def _site(tmp_path, monkeypatch, text):
    f = tmp_path / "site.toml"
    f.write_text(text, encoding="utf-8")
    monkeypatch.setenv("BIGBRAIN_SITE", str(f))


# -- the built-in reader --------------------------------------------------------------------------------------------------------
def test_the_builtin_reader_asks_each_pc_once_with_its_own_user_and_address():
    calls = []

    def run(user, host, cmd):
        calls.append((user, host, cmd))
        return {"10.0.0.1": "0, 20000\n1, 1500\n", "10.0.0.2": "0, 9000\n"}[host]

    free = ctl.nvidia_smi_free_mib(REG, [("a", 0), ("a", 1), ("b", 0)], run=run)
    assert free == {("a", 0): 20000, ("a", 1): 1500, ("b", 0): 9000}
    assert len(calls) == 2, "one read-only ssh per PC"
    assert calls[0][:2] == ("ua", "10.0.0.1") and calls[0][2] == ctl.FREE_MEM_QUERY
    assert calls[1][0], "a PC with no `user` in the registry falls back to the site ssh user"
    assert "nvidia-smi" in ctl.FREE_MEM_QUERY and "memory.free" in ctl.FREE_MEM_QUERY and "rm " not in ctl.FREE_MEM_QUERY


def test_the_builtin_reader_leaves_out_a_card_the_pc_did_not_report():
    free = ctl.nvidia_smi_free_mib(REG, [("a", 0), ("a", 5)], run=lambda u, h, c: "0, 100\nnot a row\n")
    assert free == {("a", 0): 100}


def test_an_ssh_failure_is_a_plan_refusal_that_names_split_even():
    def run(user, host, cmd):
        raise subprocess.CalledProcessError(255, "ssh")
    with pytest.raises(ctl.PlanRefused) as e:
        ctl.nvidia_smi_free_mib(REG, [("a", 0)], run=run)
    assert "--split even" in str(e.value) and "CalledProcessError" in str(e.value)


def test_with_no_site_setting_the_builtin_reader_is_the_one_used(monkeypatch):
    seen = []
    monkeypatch.setattr(ctl, "nvidia_smi_free_mib", lambda reg_, cards: seen.append(list(cards)) or {c: 1 for c in cards})
    assert ctl._cli_free_mib(REG, [("a", 0)]) == {("a", 0): 1} and seen == [[("a", 0)]]


# -- the optional reader from site.toml --------------------------------------------------------------------------------------------
def test_a_configured_reader_replaces_the_builtin_one(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch, '[capacity]\nreader = "my_scheduler_probe:free_mib"\n')
    fake = types.SimpleNamespace(free_mib=lambda reg_, cards: {c: 5000 for c in cards})
    monkeypatch.setitem(sys.modules, "my_scheduler_probe", fake)
    monkeypatch.setattr(ctl, "nvidia_smi_free_mib", lambda *a: pytest.fail("the built-in reader must not run"))
    assert ctl._cli_free_mib({}, [("h", 0)]) == {("h", 0): 5000}


def test_a_configured_reader_that_cannot_be_loaded_names_split_even(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch, '[capacity]\nreader = "no_such_module_xyz:f"\n')
    with pytest.raises(ctl.PlanRefused) as e:
        ctl._cli_free_mib({}, [("10.0.0.1", 0)])
    msg = str(e.value)
    assert "--split even" in msg and "no_such_module_xyz:f" in msg and "ModuleNotFoundError" in msg


def test_a_configured_reader_with_a_missing_function_is_refused_the_same_way(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch, '[capacity]\nreader = "my_scheduler_probe2:nope"\n')
    monkeypatch.setitem(sys.modules, "my_scheduler_probe2", types.SimpleNamespace())
    with pytest.raises(ctl.PlanRefused) as e:
        ctl._cli_free_mib({}, [("h", 0)])
    assert "--split even" in str(e.value) and "AttributeError" in str(e.value)


# -- through the CLI ------------------------------------------------------------------------------------------------------------
def test_the_cli_exits_with_the_refusal_not_a_traceback(registry_env, tmp_path, monkeypatch, capsys):
    _site(tmp_path, monkeypatch, '[capacity]\nreader = "no_such_module_xyz:f"\n')
    with pytest.raises(SystemExit) as e:
        ctl.plan_from_args(["--model", "M", "--gpus", "a:0", "--gpus", "b:0", "--ctx", "1024",
                            "--node-overhead-mib", "0", "--headroom-mib", "0"])
    assert e.value.code == 2                                         # argparse's error exit
    err = capsys.readouterr().err
    assert "--split even" in err and "Traceback" not in err


def test_the_even_split_needs_no_reader(registry_env, monkeypatch):
    monkeypatch.setattr(ctl, "_cli_free_mib", lambda *a: pytest.fail("--split even must not read the cards"))
    plan = ctl.plan_from_args(["--model-dir", "/m", "--total-blocks", "28", "--gpus", "h0:0", "--gpus", "h1:1"])
    assert [(n.slice_start, n.slice_end) for n in plan.nodes] == [(0, 14), (14, 28)]


def test_the_capacity_split_uses_the_reading(registry_env, monkeypatch):
    monkeypatch.setattr(ctl, "_cli_free_mib", lambda reg_, cards: {c: 24000 for c in cards})
    plan = ctl.plan_from_args(["--model", "M", "--gpus", "a:0", "--gpus", "b:0", "--ctx", "1024",
                               "--node-overhead-mib", "0", "--headroom-mib", "0"])
    assert [n.host for n in plan.nodes] == ["10.0.0.1", "10.0.0.2"]
    assert plan.nodes[0].user == "ua" and plan.nodes[0].slice_start == 0 and plan.nodes[-1].slice_end == 30


def test_the_capacity_split_accepts_slots_and_a_serve_port_together(registry_env, monkeypatch):
    """Regression: several slots need the serving engine, and the capacity planner validates its plan before the CLI copies the
    serving flags onto it, so `--split capacity --slots 4 --serve-port N` used to be refused with "give --serve-port"."""
    monkeypatch.setattr(ctl, "_cli_free_mib", lambda reg_, cards: {c: 24000 for c in cards})
    plan = ctl.plan_from_args(["--model", "M", "--gpus", "a:0", "--gpus", "b:0", "--ctx", "1024",
                               "--node-overhead-mib", "0", "--headroom-mib", "0",
                               "--slots", "4", "--serve-port", "16980", "--api-token", "T"])
    assert (plan.http_port, plan.api_token, plan.slots) == (16980, "T", 4)
    assert plan.validate() == []
    assert "--http-port 16980" in plan.build_commands()[0] and "--slots 4" in plan.build_commands()[1]
