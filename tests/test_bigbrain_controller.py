# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Tests for tools/bigbrain/controller.py (v1).

All tests use fake runners -- no GPU, no network, no ssh.
"""
import pytest

from tools.bigbrain.controller import (
    ClusterPlan, NodeSpec, Controller, plan_from_args,
)


def _node(rank, host, gpu, start, end, in_port):
    return NodeSpec(rank=rank, host=host, gpu=gpu, slice_start=start,
                    slice_end=end, in_port=in_port)


def _plan(nodes, total_blocks, token_port=0, model_dir="/m", python_bin="python"):
    return ClusterPlan(model_dir=model_dir, nodes=nodes, total_blocks=total_blocks,
                       token_port=token_port, python_bin=python_bin)


class TestValidate:
    def test_valid_two_node_plan_has_no_problems(self):
        # a multi-node plan carries a token port distinct from its in-ports
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 20, 6001)], 20, token_port=7000)
        assert plan.validate() == []

    def test_gap_in_slice_coverage(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 15, 25, 6001)], 25)
        assert any("gap" in p for p in plan.validate())

    def test_overlapping_slices(self):
        plan = _plan([_node(0, "h0", 0, 0, 20, 0),
                      _node(1, "h1", 0, 10, 30, 6001)], 30)
        assert any("overlap" in p for p in plan.validate())

    def test_non_contiguous_ranks(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(2, "h1", 0, 10, 20, 6001)], 20)
        assert any("rank" in p for p in plan.validate())

    def test_non_first_node_with_in_port_zero(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 20, 0)], 20)
        assert any("in_port" in p for p in plan.validate())

    def test_duplicate_in_ports(self):
        # two listeners on the SAME host cannot share a port
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 20, 6001),
                      _node(2, "h1", 1, 20, 30, 6001)], 30, token_port=7000)
        assert any("duplicate in_port" in p for p in plan.validate())

    def test_different_hosts_may_reuse_an_in_port(self):
        # per-host uniqueness (f29793a3 / the WIP 5bc171fb): separate machines,
        # separate sockets
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 20, 6001),
                      _node(2, "h2", 0, 20, 30, 6001)], 30, token_port=7000)
        assert plan.validate() == []

    def test_last_node_must_cover_all_blocks(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 18, 6001)], 20)
        assert any("model has 20" in p for p in plan.validate())

    def test_token_port_collision_with_in_port_is_flagged(self):
        # token port 6001 == the in_port of a node on node 0's OWN host: node 0
        # binds the token port there, so the two listeners would collide.
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h0", 1, 10, 20, 6001)], 20, token_port=6001)
        assert any("collides" in p for p in plan.validate())

    def test_token_port_equal_to_an_in_port_elsewhere_is_fine(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 20, 6001)], 20, token_port=6001)
        assert plan.validate() == []

    def test_http_port_needs_a_token_and_its_own_port(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h0", 1, 10, 20, 6001)], 20, token_port=7000)
        plan.http_port = 7000
        assert any("http_port" in p for p in plan.validate())
        plan.http_port, plan.api_token = 16980, ""
        assert any("api_token" in p for p in plan.validate())
        plan.api_token = "t"
        assert plan.validate() == []

    def test_teardown_command_targets_only_this_stage_dir(self):
        from tools.bigbrain.controller import teardown_command
        c = teardown_command("/tmp/bigbrain/bb-1", "TERM")
        assert "kill -TERM" in c and "/tmp/bigbrain/bb-1" in c and "/proc/$p/cwd" in c
        assert c.rstrip().endswith("true")
        with pytest.raises(ValueError):
            teardown_command("/tmp/x", "HUP; rm -rf /")

    def test_teardown_pattern_matches_both_launch_forms_and_not_itself(self):
        """v0.1.1: the controller's own launch is `-m tools.bigbrain.worker`, a wrapper script's is `python -c "... from
        tools.bigbrain import worker ..."` . One pattern finds both; it does not match unrelated
        processes nor the shell that carries it."""
        import re
        from tools.bigbrain.controller import NODE_PROCESS_PATTERN, teardown_command
        rx = re.compile(NODE_PROCESS_PATTERN)
        launched_by_controller = "/venv/bin/python -m tools.bigbrain.worker --model-dir /m --rank 1 --total 4"
        launched_by_run_node = "/venv/bin/python -m tools.bigbrain.worker --rank 0 --http-port 16980"
        launched_as_server = "/venv/bin/python -m tools.bigbrain.server --rank 0 --http-port 16980"
        launched_by_wrapper_worker = ("/venv/bin/python -c import sys; sys.path.insert(0, chr(46)); import launch_wrapper; "
                                      "launch_wrapper.install(); from tools.bigbrain import worker; sys.exit(worker.main()) --rank 1")
        launched_by_wrapper_server = ("bash -c cd /tmp/bigbrain_stage/r && WRAPPER_OUT=t.json /venv/bin/python -c 'import sys; "
                                      "import launch_wrapper; from tools.bigbrain import server; sys.exit(server.main())' --rank 0")
        for line in (launched_by_controller, launched_by_run_node, launched_as_server,
                     launched_by_wrapper_worker, launched_by_wrapper_server):
            assert rx.search(line), line
        for line in ("/venv/bin/python -m tools.bigbrain.controller --gpus a:0",       # the controller CLI itself is not a node
                     "/venv/bin/python -m tools.bigbrain.bench grid --url http://x",
                     "/venv/bin/python main.py --config configs/tabby-5000.yml",
                     "grep tools.bigbrainXworker", "tail -f node_0.log"):
            assert not rx.search(line), line
        cmd = teardown_command("/tmp/bigbrain_stage/run1")
        assert f'pgrep -f "{NODE_PROCESS_PATTERN}"' in cmd
        assert not rx.search(cmd), "the shell running the teardown must not match its own pattern"

    def test_token_port_zero_is_flagged_for_multi_node(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 20, 6001)], 20, token_port=0)
        assert any("token_port" in p for p in plan.validate())

    def test_distinct_token_port_is_clean(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 20, 6001)], 20, token_port=7000)
        assert plan.validate() == []

    def test_per_node_model_dir_and_python_overrides(self):
        # mixed cluster: node 0 and node 1 have the model and the venv python at
        # different paths. build_commands must honour the per-node overrides
        # and fall back to the plan-level value where a node has none.
        n0 = _node(0, "h0", 0, 0, 10, 0)
        n1 = _node(1, "h1", 0, 10, 20, 6001)
        n0.model_dir = "/mnt/a/model"
        n0.python_bin = "/venvA/bin/python"
        n1.model_dir = "/mnt/b/model"
        n1.python_bin = "/venvB/bin/python"
        plan = _plan([n0, n1], 20, token_port=7000,
                     model_dir="/plan/default", python_bin="/plan/py")
        c0, c1 = plan.build_commands()
        assert "--model-dir /mnt/a/model" in c0 and "/venvA/bin/python" in c0
        assert "--model-dir /mnt/b/model" in c1 and "/venvB/bin/python" in c1
        assert "/plan/default" not in c0 and "/plan/default" not in c1

    def test_per_node_override_falls_back_to_plan_value(self):
        n0 = _node(0, "h0", 0, 0, 10, 0)
        n1 = _node(1, "h1", 0, 10, 20, 6001)
        n0.model_dir = "/only/node0/model"  # node 1 keeps the plan default
        plan = _plan([n0, n1], 20, token_port=7000, model_dir="/plan/default")
        c0, c1 = plan.build_commands()
        assert "--model-dir /only/node0/model" in c0
        assert "--model-dir /plan/default" in c1


class TestBuildCommands:
    def test_two_node_wiring(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 1, 10, 20, 6001)], 20, token_port=7000)
        c0, c1 = plan.build_commands()
        # node 0 -> node 1
        assert "--next-host h1" in c0 and "--next-port 6001" in c0
        assert "--token-host" not in c0 and "--in-port" not in c0
        # node 0 BINDS the token port (pre-load) for the last node to connect
        # back to -- a regression dropping this makes node 0 never open it.
        assert "--token-port 7000" in c0
        # node 1 (last) -> node 0 token port, no next
        assert "--token-host h0" in c1 and "--token-port 7000" in c1
        assert "--next-host" not in c1
        assert "--in-port 6001" in c1
        # common bits
        for c in (c0, c1):
            assert "-m tools.bigbrain.worker" in c
            assert "--total 2" in c
            assert "--model-dir /m" in c

    def test_three_node_wiring(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 20, 6001),
                      _node(2, "h2", 0, 20, 30, 6002)], 30, token_port=7000)
        c0, c1, c2 = plan.build_commands()
        assert "--next-host h1" in c0 and "--next-port 6001" in c0
        assert "--next-host h2" in c1 and "--next-port 6002" in c1
        assert "--token-host h0" in c2 and "--token-port 7000" in c2
        assert "--next-host" not in c2
        assert "--slice 10 20" in c1 and "--rank 1" in c1
        # node 0 binds the token port; middle/last nodes never do
        assert "--token-port 7000" in c0 and "--token-port" not in c1

    def test_single_node_has_no_links(self):
        plan = _plan([_node(0, "h0", 0, 0, 20, 0)], 20)
        c0 = plan.build_commands()[0]
        assert "--next-host" not in c0 and "--token-host" not in c0
        assert "--total 1" in c0


class _FakeHandle:
    def __init__(self, code=0):
        self._code = code
        self.polled = 0

    def poll(self):
        self.polled += 1
        return None if self.polled < 2 else self._code


class _FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, host, user, cmd):
        self.calls.append((host, user, cmd))
        return _FakeHandle(0)


class TestController:
    def test_launch_starts_in_reverse_rank_order(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h1", 0, 10, 20, 6001),
                      _node(2, "h2", 0, 20, 30, 6002)], 30)
        fr = _FakeRunner()
        Controller(plan, runner=fr, dry_run=True).launch()
        # reverse rank order: last node (listener for token port) first
        assert [c[0] for c in fr.calls] == ["h2", "h1", "h0"]

    def test_wait_polls_until_done(self):
        plan = _plan([_node(0, "h0", 0, 0, 20, 0)], 20)
        fr = _FakeRunner()
        ctl = Controller(plan, runner=fr, dry_run=False, stage=False)
        ctl.launch()
        assert ctl.wait() == 0

    def test_signal_death_is_not_reported_as_success(self):
        # A node killed by SIGKILL (OOM) exits -9; wait() must surface that as
        # a failure (128+9), not max(0, -9) == 0.
        plan = _plan([_node(0, "h0", 0, 0, 20, 0)], 20)

        class _SigHandle:
            def __init__(self):
                self._polled = 0

            def poll(self):
                self._polled += 1
                return None if self._polled < 2 else -9

        def sig_runner(host, user, cmd):
            return _SigHandle()

        ctl = Controller(plan, runner=sig_runner, dry_run=False, stage=False)
        ctl.launch()
        assert ctl.wait() == 137

    def test_dry_run_wait_is_immediate_zero(self):
        plan = _plan([_node(0, "h0", 0, 0, 20, 0)], 20)
        assert Controller(plan, dry_run=True).wait() == 0


class TestPlanFromArgs:
    def test_parses_repeated_node_specs(self):
        plan = plan_from_args([
            "--model-dir", "/m", "--total-blocks", "20", "--token-port", "7000",
            "--node", "h0:0:0:10:0", "--node", "h1:1:10:20:6001",
        ])
        assert plan.total_blocks == 20
        assert plan.token_port == 7000
        assert [(n.rank, n.host, n.gpu, n.slice_start, n.slice_end, n.in_port)
                for n in plan.nodes] == [
            (0, "h0", 0, 0, 10, 0),
            (1, "h1", 1, 10, 20, 6001),
        ]
        assert plan.validate() == []

    def test_bad_node_spec_raises(self):
        with pytest.raises(ValueError):
            plan_from_args(["--model-dir", "/m", "--total-blocks", "10",
                            "--node", "h0:0:10"])


class TestStagingAndFlags:
    def test_prompt_is_shell_quoted(self):
        plan = _plan([_node(0, "h0", 0, 0, 20, 0)], 20)
        plan.prompt = 'say "hi" there'
        c0 = plan.build_commands()[0]
        # shlex.quote wraps in single quotes; the raw double quotes survive inside
        assert "--prompt 'say \"hi\" there'" in c0
        assert "--prompt \"say" not in c0  # no naive double-quote wrapping

    def test_mem_fraction_emitted_only_when_set(self):
        plan = _plan([_node(0, "h0", 0, 0, 20, 0)], 20)
        assert "--mem-fraction" not in plan.build_commands()[0]
        plan.mem_fraction = 0.4
        assert "--mem-fraction 0.4" in plan.build_commands()[0]

    def test_command_cd_prefix_points_at_stage_root(self):
        plan = _plan([_node(0, "h0", 0, 0, 20, 0)], 20)
        c0 = plan.build_commands()[0]
        assert c0.startswith("cd /tmp/bigbrain && ")
        assert "-m tools.bigbrain.worker" in c0

    def test_stage_package_stages_all_files(self, tmp_path, monkeypatch):
        # Use the real package dir so content round-trips byte-for-byte.
        from pathlib import Path
        from tools.bigbrain.controller import PACKAGE_FILES, stage_package
        pkg_dir = str(Path(__file__).resolve().parent.parent / "tools" / "bigbrain")
        calls = []

        def fake_stager(host, user, remote_path, content):
            calls.append((host, user, remote_path, content))

        staged = stage_package("h0", "u", pkg_dir, "/tmp/bigbrain", stager=fake_stager)
        assert len(staged) == len(PACKAGE_FILES) == 9
        # every file went to the right remote path with the right bytes
        for (host, user, remote_path, content) in calls:
            assert (host, user) == ("h0", "u")
            rel = remote_path[len("/tmp/bigbrain/"):]
            src = Path(pkg_dir).parent.parent / rel
            expect = src.read_text(encoding="utf-8") if src.exists() else ""
            assert content == expect, rel

    def test_launch_stages_once_per_host_then_starts_reverse(self):
        plan = _plan([_node(0, "h0", 0, 0, 10, 0),
                      _node(1, "h0", 1, 10, 20, 6001),  # same host, 2nd GPU
                      _node(2, "h1", 0, 20, 30, 6002)], 30)
        fr = _FakeRunner()
        staged_hosts = []

        def fake_stager(host, user, remote_path, content):
            staged_hosts.append(host)

        ctl = Controller(plan, runner=fr, dry_run=False, stager=fake_stager)
        ctl.launch()
        # staged exactly ONCE per unique host (one stage_package call = one copy of PACKAGE_FILES);
        # h0 appears twice in nodes but is staged a single time
        from tools.bigbrain.controller import PACKAGE_FILES
        assert staged_hosts.count("h0") == len(PACKAGE_FILES)
        assert staged_hosts.count("h1") == len(PACKAGE_FILES)
        assert len(set(staged_hosts)) == 2
        # nodes still start in reverse rank order
        assert [c[0] for c in fr.calls] == ["h1", "h0", "h0"]

    def test_dry_run_launch_does_not_stage(self):
        plan = _plan([_node(0, "h0", 0, 0, 20, 0)], 20)
        staged = []
        ctl = Controller(plan, dry_run=True,
                         stager=lambda h, u, p, c: staged.append(p))
        ctl.launch()
        assert staged == []


class TestPlanAuto:
    def test_even_split(self):
        from tools.bigbrain.controller import plan_auto
        plan = plan_auto("/m", [("h0", 0), ("h1", 0)], total_blocks=28)
        assert [(n.slice_start, n.slice_end) for n in plan.nodes] == [(0, 14), (14, 28)]
        assert plan.nodes[0].in_port == 0 and plan.nodes[1].in_port == 16900
        assert plan.token_port == 16901
        assert plan.total_blocks == 28
        assert plan.validate() == []

    def test_uneven_split_gives_extra_to_early_nodes(self):
        from tools.bigbrain.controller import plan_auto
        # 28 layers over 3 gpus -> 10, 9, 9
        plan = plan_auto("/m", [("h0", 0), ("h1", 0), ("h2", 0)], total_blocks=28)
        assert [(n.slice_start, n.slice_end) for n in plan.nodes] == \
            [(0, 10), (10, 19), (19, 28)]
        assert plan.validate() == []

    def test_multi_machine_mixed_gpus(self):
        from tools.bigbrain.controller import plan_auto
        plan = plan_auto("/m", [("10.0.0.11", 0), ("10.0.0.12", 1),
                                ("10.0.0.13", 0)], total_blocks=64)
        assert [n.host for n in plan.nodes] == \
            ["10.0.0.11", "10.0.0.12", "10.0.0.13"]
        assert [n.gpu for n in plan.nodes] == [0, 1, 0]
        # contiguous coverage 0..64
        assert plan.nodes[0].slice_start == 0
        assert plan.nodes[-1].slice_end == 64
        assert plan.validate() == []

    def test_single_gpu_is_whole_model(self):
        from tools.bigbrain.controller import plan_auto
        plan = plan_auto("/m", [("h0", 0)], total_blocks=28)
        assert [(n.slice_start, n.slice_end) for n in plan.nodes] == [(0, 28)]
        assert plan.nodes[0].in_port == 0
        assert plan.validate() == []

    def test_more_gpus_than_layers_raises(self):
        from tools.bigbrain.controller import plan_auto
        import pytest
        with pytest.raises(ValueError):
            plan_auto("/m", [("h0", 0), ("h1", 0), ("h2", 0), ("h3", 0)],
                      total_blocks=2)

    def test_gpus_cli_parses_and_plans(self):
        plan = plan_from_args([
            "--model-dir", "/m", "--total-blocks", "28", "--token-port", "0",
            "--gpus", "h0:0", "--gpus", "h1:1",
        ])
        assert [(n.host, n.gpu) for n in plan.nodes] == [("h0", 0), ("h1", 1)]
        assert [(n.slice_start, n.slice_end) for n in plan.nodes] == [(0, 14), (14, 28)]
        assert plan.validate() == []

    def test_gpus_and_node_mutually_exclusive(self):
        import pytest
        with pytest.raises(SystemExit):
            plan_from_args([
                "--model-dir", "/m", "--gpus", "h0:0",
                "--node", "h0:0:0:14:0",
            ])

    def test_neither_gpus_nor_node_errors(self):
        import pytest
        with pytest.raises(SystemExit):
            plan_from_args(["--model-dir", "/m"])