# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/exllamav3/dflash2_tp.py: the DFlash2 + tensor-parallel patch for exllamav3 1.6.x, on a synthetic exllamav3 package that
holds exactly the anchors of the six files. (The real 1.6.0 / 1.6.1 packages are checked by patches/verify.py.)"""
import importlib.util
import os
import pathlib
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "patches"))

spec = importlib.util.spec_from_file_location("dflash2_tp", os.path.join(ROOT, "patches", "exllamav3", "dflash2_tp.py"))
d2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d2)

FILES = {
    "model/model_tp_fn.py": "import torch\n\n\n" + d2.TP_FN_ANCHOR + "\n",
    "model/model_tp.py": "class ExLlamaTP:\n    def other(self):\n        pass\n\n" + d2.TP_ANCHOR + "\n",
    "model/model.py": "import torch\n\n\nclass Model:\n" + d2.MODEL_ANCHOR + "\n        return 1\n",
    "modules/arch_specific/dflash2.py":
        "import torch\n\n\nclass Selector:\n    def walk_block(self, hidden, cuda, bsz, rows, unary, cands, gate, anchor_ids, return_confidence, ext):\n"
        + d2.SEL_OLD,
    "architecture/dflash2.py":
        "class DFlash2Model:\n    def attach_to(self, target):\n" + d2.ARCH_ATTACH_OLD +
        "\n    def sample_from_state(self, state, params):\n" + d2.ARCH_SAMPLE_OLD + "        return lm\n",
    "generator/generator.py": "class Generator:\n    def f(self):\n" + d2.GEN_OLD + "\n        return window\n",
}


def package(root, version="1.6.1"):
    pkg = root / "lib" / "python3.12" / "site-packages" / "exllamav3"
    for rel, text in FILES.items():
        p = pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    if version is not None:                        # exllamav3 ships its version as `__version__ = "1.6.1"` in version.py
        (pkg / "version.py").write_text(f'__version__ = "{version}"', encoding="utf-8", newline="\n")
    return pkg


def test_every_file_of_the_patch_is_covered_by_the_fixture():
    assert set(d2.SUBSTITUTIONS) == set(FILES) and len(FILES) == 6


def test_apply_check_revert_on_the_package_dir(tmp_path, capsys):
    pkg = package(tmp_path)
    t = str(pkg)
    assert d2.main(["--target", t]) == 1
    assert d2.main(["--target", t, "--apply"]) == 0
    for rel in FILES:
        text = (pkg / rel).read_text(encoding="utf-8")
        assert d2.MARK in text, rel
        compile(text, rel, "exec")
        assert (pkg / (rel + ".bb-orig-dflash2_tp")).read_text(encoding="utf-8") == FILES[rel]
    assert d2.main(["--target", t]) == 0
    assert "dflash2_tp: applied" in capsys.readouterr().out
    assert d2.main(["--target", t, "--revert"]) == 0
    assert all((pkg / rel).read_text(encoding="utf-8") == text for rel, text in FILES.items())
    assert d2.main(["--target", t]) == 1


def test_target_may_be_the_venv_root_or_site_packages(tmp_path):
    pkg = package(tmp_path)
    assert d2.main(["--target", str(tmp_path)]) == 1                                    # venv root: found under lib/python3*/site-packages
    assert d2.main(["--target", str(pkg.parent)]) == 1                                  # site-packages
    assert d2.resolve_package(str(tmp_path)) == str(pkg)


def test_one_drifted_file_refuses_the_whole_patch_and_writes_nothing(tmp_path):
    pkg = package(tmp_path)
    (pkg / "generator" / "generator.py").write_text("class Generator:\n    pass\n", encoding="utf-8")
    before = {rel: (pkg / rel).read_bytes() for rel in FILES}
    assert d2.main(["--target", str(pkg)]) == 2
    assert d2.main(["--target", str(pkg), "--apply"]) == 2
    assert {rel: (pkg / rel).read_bytes() for rel in FILES} == before
    assert not list(pkg.rglob("*.bb-orig-*"))


def test_the_marker_is_in_every_patched_file_or_check_could_never_say_applied(tmp_path):
    pkg = package(tmp_path)
    assert d2.main(["--target", str(pkg), "--apply"]) == 0
    missing = [rel for rel in FILES if d2.MARK not in (pkg / rel).read_text(encoding="utf-8")]
    assert missing == []


# -- the version gate: exllamav3 1.6.2 has DFlash2 tensor parallel built in, so this patch must never go on it ------------------
@pytest.mark.parametrize("version", ["1.6.2", "1.6.2+cu128.torch2.9.0", "1.6.3", "1.7.0", "2.0.0",     # built in (or later)
                                     "1.6.0.dev1", "1.5.4", "1.4.8", "0.9.0", "garbage", ""])        # outside the verified two
def test_apply_refuses_everything_but_1_6_0_and_1_6_1(tmp_path, version, capsys):
    pkg = package(tmp_path, version=version)
    before = {rel: (pkg / rel).read_bytes() for rel in FILES}
    assert d2.version_gate(str(pkg)) is not None
    assert d2.main(["--target", str(pkg), "--apply"]) == 2
    assert "REFUSING" in capsys.readouterr().out
    assert {rel: (pkg / rel).read_bytes() for rel in FILES} == before, "a refused apply writes nothing"
    assert not list(pkg.rglob("*.bb-orig-*")) and not (pkg / ".bb-patch-dflash2_tp.json").exists()


@pytest.mark.parametrize("version", ["1.6.0", "1.6.1", "1.6.1+cu128.torch2.9.0", "1.6.0+cu128.torch2.9.0"])
def test_apply_still_goes_on_1_6_0_and_1_6_1(tmp_path, version):
    pkg = package(tmp_path, version=version)
    assert d2.version_gate(str(pkg)) is None
    assert d2.main(["--target", str(pkg), "--apply"]) == 0
    assert d2.main(["--target", str(pkg)]) == 0


def test_a_missing_version_py_is_unsupported_and_apply_refuses(tmp_path, capsys):
    pkg = package(tmp_path, version=None)
    assert d2.read_version(str(pkg)) is None
    kind, msg = d2.version_gate(str(pkg))
    assert kind == "unsupported" and "version.py" in msg
    assert d2.main(["--target", str(pkg), "--apply"]) == 2


@pytest.mark.parametrize("version", ["1.6.2", "1.6.2+cu128.torch2.9.0", "1.6.3", "1.7.0"])
def test_check_on_a_version_that_has_it_built_in_says_so_and_exits_1(tmp_path, capsys, version):
    pkg = package(tmp_path, version=version)
    capsys.readouterr()
    assert d2.main(["--target", str(pkg), "--check"]) == 1
    out = capsys.readouterr().out
    assert "built in, never patch" in out and "dflash2_tp:" in out
    assert "DRIFTED" not in out, "a version with the feature built in is not upstream drift"


def test_the_real_1_6_2_layout_does_not_look_like_drift(tmp_path, capsys):
    """1.6.2's files lack our anchors (the feature is built in), so without the gate --check would say DRIFTED (exit 2) -- the
    bug the gate fixes. Same files, version 1.6.1: that IS drift; version 1.6.2: not."""
    pkg = package(tmp_path, version="1.6.1")
    (pkg / "architecture" / "dflash2.py").write_text("class DFlash2Model:\n    def attach_to(self, target):\n        pass\n", encoding="utf-8")
    assert d2.main(["--target", str(pkg)]) == 2
    (pkg / "version.py").write_text('__version__ = "1.6.2"', encoding="utf-8")
    assert d2.main(["--target", str(pkg)]) == 1


def test_our_patch_on_a_built_in_version_is_loud(tmp_path, capsys):
    pkg = package(tmp_path, version="1.6.1")
    assert d2.main(["--target", str(pkg), "--apply"]) == 0
    (pkg / "version.py").write_text('__version__ = "1.6.2"', encoding="utf-8")
    capsys.readouterr()
    assert d2.main(["--target", str(pkg)]) == 2
    assert "revert it" in capsys.readouterr().out
    assert d2.main(["--target", str(pkg), "--revert"]) == 0                        # the way out still works


def test_check_on_an_older_version_reports_the_files_and_a_note(tmp_path, capsys):
    pkg = package(tmp_path, version="1.5.4")
    capsys.readouterr()
    assert d2.main(["--target", str(pkg)]) == 1
    out = capsys.readouterr().out
    assert "NOT patched" in out and "outside the verified versions" in out and "--apply refuses" in out


def test_pins_toml_and_the_script_hold_the_same_two_numbers():
    import tomllib
    pins = tomllib.loads((pathlib.Path(ROOT) / "patches" / "PINS.toml").read_text(encoding="utf-8"))
    gate = pins["upstream"]["exllamav3"]["dflash2_tp"]
    assert tuple(gate["apply_versions"]) == d2.ALLOWED_VERSIONS == ("1.6.0", "1.6.1")
    assert gate["built_in_from"] == d2.BUILT_IN_FROM == "1.6.2"


def test_the_example_hosts_pin_a_1_6_2_venv_absent():
    import tomllib
    hosts = tomllib.loads((pathlib.Path(ROOT) / "patches" / "hosts.example.toml").read_text(encoding="utf-8"))["hosts"]
    entries = [e for e in hosts["beta"]["exllamav3"] if e["venv"] == "~/tabbyAPI/venv162"]
    assert len(entries) == 1 and entries[0]["absent"] == ["dflash2_tp"] and entries[0]["version"].startswith("1.6.2")
    assert "applied" not in entries[0]
