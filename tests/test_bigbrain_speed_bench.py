# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/bench.py against the in-process fake pipeline: the grid runs, every cell is
checked against the plain run, and a deliberately corrupted cell is caught."""
from __future__ import annotations

import json

import pytest

from bigbrain_speed_fakes import V
from bigbrain_speed_harness import KEY, pipes, served  # noqa: F401
from bigbrain import bench

PROMPTS = ["abcabcabdabcab", "jjijjijji", "a", "cdefghcdefgh", "bbbbbb"]


def _torch():
    return pytest.importorskip("torch")


def test_compare_outputs_reports_the_first_divergence_with_the_plain_margin():
    base = {"0": [1, 2, 3, 4], "1": [5, 6, 7]}
    same = bench.compare_outputs(base, {"0": [1, 2, 3, 4], "1": [5, 6, 7]})
    assert same["all_identical"] and same["identical"] == 2
    bad = bench.compare_outputs(base, {"0": [1, 2, 9, 4], "1": [5, 6]}, {"0": [0.5, 0.4, 0.001, 0.3]})
    assert not bad["all_identical"] and bad["identical"] == 0
    d = {r["prompt"]: r for r in bad["diverged"]}
    assert d[0]["first_divergence"] == 2 and d[0]["plain_margin"] == 0.001
    assert d[0]["plain_token"] == 3 and d[0]["other_token"] == 9
    assert d[1]["first_divergence"] == 2           # a shorter output diverges where it ends
    assert not bench.compare_outputs({}, {})["all_identical"]


def test_synthetic_and_review_prompts(tmp_path):
    ps = bench.synthetic_prompts(3, 400)
    assert len(ps) == 3 and all(p.startswith(bench.DEFAULT_REVIEW_PROMPT) and len(p) < 600 for p in ps)
    assert ps[0] != ps[1]
    (tmp_path / "a.py").write_text("print('a')\n" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("print('b')\n")
    (tmp_path / "c.txt").write_text("nope")
    rp = bench.review_prompts(str(tmp_path), ["**/*.py"], 50, 10)
    assert len(rp) == 2 and "File: a.py" in rp[0] and "b" in rp[1]


def test_the_grid_runs_every_cell_and_every_cell_matches_the_plain_run(served):
    _torch()
    p, app, base = served([(0, 2), (2, 4)], slots=4, max_draft=4, spec="none")
    client = bench.Client(base, KEY)
    grid = bench.run_grid(client, PROMPTS, [1, 2, 4], ["none", "ngram"], spec_k=4, max_tokens=14)
    cells = {(c["concurrency"], c["spec"]): c for c in grid["cells"]}
    assert set(cells) == {(1, "none"), (2, "none"), (4, "none"), (1, "ngram"), (2, "ngram"), (4, "ngram")}
    assert grid["cells"][0]["concurrency"] == 1 and grid["cells"][0]["spec"] == "none"   # the plain run first
    for key, c in cells.items():
        assert c["completed"] == len(PROMPTS) and c["failed"] == 0, key
        assert c["tokens"] == len(PROMPTS) * 14
        assert c["parity"]["all_identical"], (key, c["parity"])
        assert c["agg_tok_s"] > 0 and c["per_seq_tok_s"] > 0 and c["first_token_s"]["mean"] is not None
        assert [n["rank"] for n in c["nodes"]] == [0, 1]
        for n in c["nodes"]:
            assert 0 <= n["busy_pct"] <= 100 and n["hop_wait_s"] >= 0
        assert c["engine"]["decode_tokens"] == len(PROMPTS) * 14
    # speculation: only the ngram cells proposed anything
    assert cells[(1, "none")]["spec_stats"]["proposed"] == 0
    assert cells[(1, "ngram")]["spec_stats"]["proposed"] > 0
    assert 0 <= cells[(1, "ngram")]["spec_stats"]["acceptance_rate"] <= 1
    assert cells[(4, "none")]["engine"]["inflight"]["max"] >= 2
    table = bench.format_table(grid)
    assert "IDENTICAL" in table and table.count("\n") >= 7


def test_a_corrupted_cell_is_caught_by_the_parity_check(served, monkeypatch):
    _torch()
    p, app, base = served([(0, 4)], slots=2, max_draft=4)
    client = bench.Client(base, KEY)
    real = client.post
    calls = {"n": 0}

    def lying_post(path, body):
        code, obj = real(path, body)
        if path == "/v1/bigbrain/generate" and body.get("spec") == "ngram" and code == 200:
            calls["n"] += 1
            obj["tokens"] = list(obj["tokens"])
            obj["tokens"][5] = (obj["tokens"][5] + 1) % V
        return code, obj
    monkeypatch.setattr(client, "post", lying_post)
    grid = bench.run_grid(client, PROMPTS[:2], [1], ["none", "ngram"], 4, 10)
    bad = [c for c in grid["cells"] if c["spec"] == "ngram"][0]
    assert not bad["parity"]["all_identical"] and bad["parity"]["diverged"][0]["first_divergence"] == 5
    assert bad["parity"]["diverged"][0]["plain_margin"] is not None       # the plain cell recorded diag margins
    assert "identical" in bench.format_table(grid)


def test_a_draft_cell_on_a_server_without_a_draft_model_is_skipped_not_fatal(served):
    _torch()
    p, app, base = served([(0, 4)], slots=2, max_draft=4)
    grid = bench.run_grid(bench.Client(base, KEY), PROMPTS[:2], [1], ["none", "draft"], 4, 8)
    draft = [c for c in grid["cells"] if c["spec"] == "draft"][0]
    assert "draft model" in draft["skipped"] and draft["http_status"] == 400
    assert "skipped" in bench.format_table(grid)


def test_cli_grid_writes_a_result_file_and_compare_reads_it(served, tmp_path, capsys):
    _torch()
    p, app, base = served([(0, 2), (2, 4)], slots=2, max_draft=4)
    pf = tmp_path / "prompts.json"
    pf.write_text(json.dumps(PROMPTS[:3]))
    kf = tmp_path / "key"
    kf.write_text(KEY)
    out = tmp_path / "grid.json"
    rc = bench.main(["grid", "--url", base, "--key-file", str(kf), "--concurrency", "1", "2",
                     "--spec", "none", "ngram", "--max-tokens", "10", "--prompts-file", str(pf),
                     "--out", str(out)])
    assert rc == 0
    d = json.loads(out.read_text())
    assert len(d["cells"]) == 4 and d["meta"]["prompts"] == 3
    assert "key" not in d["meta"]["args"]
    # a second run, checked against the first as its plain baseline
    out2 = tmp_path / "grid2.json"
    rc = bench.main(["grid", "--url", base, "--key-file", str(kf), "--concurrency", "2", "--spec", "ngram",
                     "--max-tokens", "10", "--prompts-file", str(pf), "--baseline", str(out), "--out", str(out2)])
    assert rc == 0
    assert bench.main(["compare", str(out), str(out2)]) == 0
    capsys.readouterr()


def test_the_documented_workflow_plain_single_sequence_server_then_batched_server(served, tmp_path):
    """T1 of the GPU plan, in mock form: (A) a PLAIN single-sequence server (one slot, no speculation,
    the old traffic) records plain.json; (B) a batched speculative server is checked against it."""
    _torch()
    from bigbrain_speed_harness import FakeWorkerEngine, make_node
    from bigbrain.server import BigBrainApp, serve_in_thread
    node = make_node(__import__("bigbrain_speed_fakes").KINDS_MIXED, 0, 4, 0, 1, slots=1, max_draft=0)

    class _Plain(FakeWorkerEngine):
        pass
    eng = FakeWorkerEngine(type("P", (), {"head": node, "engine": None})())
    assert eng.batch is None and eng.concurrency == 1
    app = BigBrainApp(eng, api_key=KEY, queue_size=4, retry_after=1)
    httpd, port, _ = serve_in_thread(app)
    try:
        plain_file = tmp_path / "plain.json"
        kf = tmp_path / "key"
        kf.write_text(KEY)
        pf = tmp_path / "p.json"
        pf.write_text(json.dumps(PROMPTS))
        rc = bench.main(["grid", "--url", f"http://127.0.0.1:{port}", "--key-file", str(kf), "--concurrency", "1",
                         "--spec", "none", "--max-tokens", "12", "--prompts-file", str(pf), "--out", str(plain_file)])
        assert rc == 0
        plain = json.loads(plain_file.read_text())
        assert plain["cells"][0]["outputs"]
        assert plain["cells"][0]["nodes"][0]["steps"] > 0           # the plain path reports its node timers too
        # a speculation cell on the single-sequence server is refused, and recorded as skipped
        g = bench.run_grid(bench.Client(f"http://127.0.0.1:{port}", KEY), PROMPTS[:2], [1], ["none", "ngram"], 4, 6)
        assert "batched" in [c for c in g["cells"] if c["spec"] == "ngram"][0]["skipped"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
    p, app2, base = served([(0, 2), (2, 4)], slots=4, max_draft=4, spec="ngram")
    out = tmp_path / "grid.json"
    rc = bench.main(["grid", "--url", base, "--key-file", str(kf), "--concurrency", "1", "4", "--spec", "none", "ngram",
                     "--max-tokens", "12", "--prompts-file", str(pf), "--baseline", str(plain_file), "--out", str(out)])
    assert rc == 0, "a cell diverged from the plain single-sequence server"
    grid = json.loads(out.read_text())
    assert all(c["parity"]["all_identical"] for c in grid["cells"])
