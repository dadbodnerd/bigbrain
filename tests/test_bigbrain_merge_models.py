# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/merge_models.py: add measured models to the registry and mark what is gone from disk, touching nothing else.

    C:/Python314/python.exe -m pytest tests/test_bigbrain_merge_models.py -q -p no:cacheprovider
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigbrain import merge_models as mm  # noqa: E402


def reg():
    return {"_gpus_note": "pinned by uuid", "nodes": {"beta": {"gpus": [{"index": 0, "uuid": "uuid-a"}, {"index": 2, "uuid": "uuid-b"}]}},
            "models": {"Old": {"size_gib": 10.0, "paths": {"alpha": "/g/old", "beta": "/r/old"}},
                       "Gone": {"size_gib": 5.0, "paths": {"beta": "/r/gone"}}}}


UPD = {"add": {"New-4.0bpw": {"bits": 4.0, "size_gib": 50.0, "paths": {"alpha": "/g/new"}, "partial_hosts": ["beta"]}},
       "missing": [{"model": "Gone", "host": "beta", "path": "/r/gone"}], "notes": ["n1"]}


def test_models_are_added_and_missing_folders_marked_but_nothing_is_deleted_or_renumbered():
    old = reg()
    new, report = mm.merge(old, UPD, now="T")
    assert new["models"]["New-4.0bpw"]["size_gib"] == 50.0 and new["models"]["Gone"]["missing_on"] == ["beta"]
    assert new["models"]["Gone"]["paths"] == {"beta": "/r/gone"}                       # kept, so the apps can say it is gone
    assert "missing_on" not in new["models"]["Old"] and new["nodes"] == old["nodes"] and new["_gpus_note"] == "pinned by uuid"
    assert new["_models_refreshed"] == {"at": "T", "by": "bigbrain/merge_models.py", "notes": ["n1"]}
    assert mm.validate(old, new) == [] and any("added New-4.0bpw" in r for r in report)
    assert old == reg()                                                                 # the input is not mutated


def test_an_existing_entry_is_completed_never_replaced_and_a_second_run_changes_nothing():
    first, _ = mm.merge(reg(), UPD, now="T")
    first["models"]["New-4.0bpw"]["size_gib"] = 51.5                                    # an operator's edit survives
    again, report = mm.merge(first, UPD, now="T")
    assert again["models"]["New-4.0bpw"]["size_gib"] == 51.5 and report == []
    fill, rep = mm.merge(reg(), {"add": {"Old": {"size_gib": 99.0, "layers": 30, "paths": {"gamma": "/s/old"}}}}, now="T")
    assert fill["models"]["Old"]["size_gib"] == 10.0 and fill["models"]["Old"]["layers"] == 30
    assert fill["models"]["Old"]["paths"] == {"alpha": "/g/old", "beta": "/r/old", "gamma": "/s/old"} and rep


def test_a_folder_that_is_present_again_loses_its_missing_mark_when_an_audit_ran():
    r = reg()
    r["models"]["Gone"]["missing_on"] = ["beta"]
    new, report = mm.merge(r, {"missing": []}, now="T")
    assert "missing_on" not in new["models"]["Gone"] and any("present again" in x for x in report)
    untouched, report = mm.merge(r, {"add": {}}, now="T")                                # no audit in the update: marks stay
    assert untouched["models"]["Gone"]["missing_on"] == ["beta"] and report == []


def test_path_fixes_move_a_folder_and_validation_refuses_a_changed_node_block_or_a_dropped_model():
    new, report = mm.merge(reg(), {"path_fixes": {"Old": {"beta": "/r/moved"}}}, now="T")
    assert new["models"]["Old"]["paths"]["beta"] == "/r/moved" and report
    bad = reg()
    bad["nodes"]["beta"]["gpus"][0]["index"] = 1
    del bad["models"]["Old"]
    problems = mm.validate(reg(), bad)
    assert any("node block" in p for p in problems) and any("Old was dropped" in p for p in problems)


def test_the_cli_dry_runs_then_writes_a_backup_and_keeps_crlf(tmp_path, capsys):
    p = tmp_path / "model_registry.json"
    p.write_bytes((json.dumps(reg(), indent=1) + "\n").replace("\n", "\r\n").encode("utf-8"))
    u = tmp_path / "u.json"
    u.write_text(json.dumps(UPD), encoding="utf-8")
    before = p.read_bytes()
    assert mm.main(["--registry", str(p), "--update", str(u)]) == 0 and p.read_bytes() == before and "dry run" in capsys.readouterr().out
    assert mm.main(["--registry", str(p), "--update", str(u), "--write"]) == 0
    baks = [f for f in os.listdir(tmp_path) if ".bak-" in f]
    assert len(baks) == 1 and (tmp_path / baks[0]).read_bytes() == before
    out = p.read_bytes()
    assert b"\r\n" in out and b"\n" not in out.replace(b"\r\n", b"") and "New-4.0bpw" in json.loads(out.decode("utf-8"))["models"]
    assert mm.main(["--registry", str(p), "--update", str(u), "--write"]) == 0 and len([f for f in os.listdir(tmp_path) if ".bak-" in f]) == 1
