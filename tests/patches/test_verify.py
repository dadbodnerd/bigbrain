# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/verify.py and PINS.toml: the pins are internally consistent, a local host is verified read-only, and the program that
runs a patch script on a remote host over `python3 -` behaves exactly like the script run directly."""
import hashlib
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(HERE).parent.parent
sys.path.insert(0, str(ROOT / "patches"))

import verify  # noqa: E402

HOSTS_EXAMPLE = ROOT / "patches" / "hosts.example.toml"
PINS = verify.load_pins(hosts_path=HOSTS_EXAMPLE)

UPSTREAM_MODEL = '''import pathlib

async def unload_model(skip_wait: bool = False, shutdown: bool = False):
    pass


async def load_model_gen(model_path: pathlib.Path, **kwargs):
    """Generator to load a model"""
    global container

    async with load_lock:
        container = None

        new_container = await ExllamaV3Container.create(model_path.resolve(), hf_model, **kwargs)

        load_status = new_container.load_gen(load_progress, **kwargs)

        async with status_display.suspended():
            progress = get_loading_progress_bar()
            progress.start()

            try:
                async for module, modules in load_status:
                    yield module, modules, "model"

                container = new_container
            finally:
                progress.stop()

        sampling.clear_model_overrides()
'''


# -- PINS.toml is consistent --------------------------------------------------------------------------------------
def test_every_patch_in_the_pins_exists_with_a_known_kind():
    assert set(PINS["patches"]) == {"failed_load", "close_stc", "lora", "auth_redact", "windows_launcher", "dflash2_tp"}
    for name, spec in PINS["patches"].items():
        assert spec["kind"] in ("script", "patchfile"), name
        assert (ROOT / "patches" / spec["path"]).is_file(), name
        assert spec["applies_to"] in ("tabbyapi", "exllamav3"), name
        assert spec["files"] and spec["license"] and spec["summary"], name
        for req in spec.get("requires", []):
            assert req in PINS["patches"], (name, req)


def test_every_host_entry_names_known_patches_and_has_the_keys_verify_needs():
    assert set(PINS["hosts"]) == {"alpha", "beta", "gamma"}
    for hname, h in PINS["hosts"].items():
        assert h.get("local") is True or h.get("ssh"), hname
        for entry in h.get("tabbyapi", []):
            assert entry["dir"]
            for n in entry.get("applied", []) + entry.get("absent", []):
                assert PINS["patches"][n]["applies_to"] == "tabbyapi", (hname, n)
        for entry in h.get("exllamav3", []):
            assert entry["venv"]
            for n in entry.get("applied", []) + entry.get("absent", []):
                assert PINS["patches"][n]["applies_to"] == "exllamav3", (hname, n)


def test_close_stc_is_only_pinned_where_failed_load_is_too():
    for h in PINS["hosts"].values():
        for entry in h.get("tabbyapi", []):
            if "close_stc" in entry.get("applied", []):
                assert "failed_load" in entry["applied"]


def test_the_tabbyapi_pin_is_a_full_commit_and_the_licences_are_named():
    assert len(PINS["upstream"]["tabbyapi"]["commit"]) == 40
    assert PINS["upstream"]["tabbyapi"]["license"] == "AGPL-3.0"
    assert PINS["upstream"]["exllamav3"]["license"] == "MIT"
    assert PINS["upstream"]["exllamav3"]["copyright"] == "Copyright (c) 2025 Turboderp"


# -- the bootstrap program == running the script ------------------------------------------------------------------
@pytest.fixture
def tabby(tmp_path):
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "model.py").write_text(UPSTREAM_MODEL, encoding="utf-8", newline="\n")
    return tmp_path


def _run_bootstrap(script, args):
    prog = verify.bootstrap_program(script, args)
    p = subprocess.run([sys.executable, "-B", "-"], input=prog.encode(), capture_output=True)
    return p.returncode, p.stdout.decode().replace("\r\n", "\n").strip()


def test_the_stdin_program_gives_the_same_exit_codes_and_output_as_the_script(tabby):
    script = ROOT / "patches" / "tabbyapi" / "failed_load.py"
    direct = subprocess.run([sys.executable, "-B", str(script), "--target", str(tabby), "--check"], capture_output=True, text=True)
    rc, out = _run_bootstrap(script, ["--target", str(tabby), "--check"])
    assert (rc, out) == (direct.returncode, direct.stdout.strip()) and rc == 1 and "NOT patched" in out
    # and the program is read-only for --check: nothing new in the tree
    assert sorted(os.listdir(tabby)) == ["common"] and sorted(os.listdir(tabby / "common")) == ["model.py"]


def test_the_stdin_program_sees_drift_as_exit_2(tabby):
    (tabby / "common" / "model.py").write_text("print('something else')\n", encoding="utf-8")
    rc, out = _run_bootstrap(ROOT / "patches" / "tabbyapi" / "failed_load.py", ["--target", str(tabby), "--check"])
    assert rc == 2 and "DRIFTED" in out


# -- verifying a local host ---------------------------------------------------------------------------------------
def _tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _pins_for(tabby, venv, commit, applied=(), absent=(), version="1.6.1"):
    pins = {"patches": PINS["patches"], "upstream": {"tabbyapi": {"commit": commit}},
            "hosts": {"here": {"local": True,
                               "tabbyapi": [{"dir": str(tabby), "applied": list(applied), "absent": list(absent)}],
                               "exllamav3": [{"venv": str(venv), "version": version}]}}}
    return pins


@pytest.fixture
def local_host(tabby, tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"], ["add", "-A"],
                ["commit", "-qm", "upstream"]):
        subprocess.run(["git", "-C", str(tabby), *cmd], check=True, capture_output=True)
    commit = subprocess.run(["git", "-C", str(tabby), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    venv = tmp_path / "venv"
    (venv / "lib" / "python3.12" / "site-packages" / "exllamav3-1.6.1.dist-info").mkdir(parents=True)
    pkg = venv / "lib" / "python3.12" / "site-packages" / "exllamav3"
    for rel in ("model/model_tp.py", "model/model.py", "model/model_tp_fn.py", "modules/arch_specific/dflash2.py",
                "architecture/dflash2.py", "generator/generator.py"):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text("# nothing to anchor on\n", encoding="utf-8")
    return tabby, venv, commit


def test_a_matching_local_host_reports_no_difference_and_changes_nothing(local_host):
    tabby, venv, commit = local_host
    before = (_tree_hash(tabby / "common"), _tree_hash(venv))
    pins = _pins_for(tabby, venv, commit, absent=["failed_load", "close_stc"])
    report, diffs, unreachable = verify.verify(pins, ["here"])
    assert diffs == [] and unreachable == []
    assert {r["what"] for r in report} >= {"upstream commit", "exllamav3 version", "patch failed_load"}
    assert (_tree_hash(tabby / "common"), _tree_hash(venv)) == before


def test_differences_are_reported_not_fixed(local_host):
    tabby, venv, commit = local_host
    before = _tree_hash(tabby / "common")
    pins = _pins_for(tabby, venv, "0" * 40, applied=["failed_load"], version="1.5.4")
    report, diffs, unreachable = verify.verify(pins, ["here"])
    text = "\n".join(diffs)
    assert "TabbyAPI is at" in text and "patch failed_load is not applied" in text and "exllamav3 is 1.6.1" in text
    assert _tree_hash(tabby / "common") == before                    # reported, never fixed


def test_an_unreachable_host_is_reported_and_nothing_else_is_claimed(monkeypatch):
    pins = {"patches": PINS["patches"], "upstream": PINS["upstream"],
            "hosts": {"far": {"ssh": "nobody@192.0.2.1", "tabbyapi": [{"dir": "~/x", "applied": ["failed_load"]}]}}}
    monkeypatch.setattr(verify.Runner, "reachable", lambda self: False)
    report, diffs, unreachable = verify.verify(pins, ["far"])
    assert unreachable == ["far"] and report == [] and diffs == []


def test_rpath_quotes_for_the_remote_shell():
    assert verify.rpath("~/tabbyAPI") == '"$HOME"/tabbyAPI'
    assert verify.rpath("/opt/a b") == "'/opt/a b'"
    assert verify.rpath("~/x y;rm") == '"$HOME"/' + "'x y;rm'"


# -- exllamav3 1.6.2 has DFlash2-TP built in: not drift, never patched ------------------------------------------------------------
def _venv_with_version(root, version):
    site = root / "venv" / "lib" / "python3.12" / "site-packages"
    (site / f"exllamav3-{version}+cu128.torch2.9.0.dist-info").mkdir(parents=True)
    pkg = site / "exllamav3"
    (pkg / "model").mkdir(parents=True)
    (pkg / "model" / "model_tp.py").write_text("# 1.6.2 carries this itself\n", encoding="utf-8")
    (pkg / "version.py").write_text(f'__version__ = "{version}"', encoding="utf-8")
    return root / "venv"


def test_a_1_6_2_venv_pinned_absent_is_not_a_difference(tmp_path):
    venv = _venv_with_version(tmp_path, "1.6.2")
    pins = {"patches": PINS["patches"], "upstream": PINS["upstream"],
            "hosts": {"here": {"local": True, "exllamav3": [{"venv": str(venv), "version": "1.6.2+cu128.torch2.9.0",
                                                              "absent": ["dflash2_tp"]}]}}}
    report, diffs, unreachable = verify.verify(pins, ["here"])
    assert diffs == [] and unreachable == []
    row = [r for r in report if r["what"] == "patch dflash2_tp"][0]
    assert row["ok"] and row["found"] == "not applied (built in)" and row["expected"] == "not applied"


def test_a_1_6_2_venv_pinned_applied_is_a_difference(tmp_path):
    venv = _venv_with_version(tmp_path, "1.6.2")
    pins = {"patches": PINS["patches"], "upstream": PINS["upstream"],
            "hosts": {"here": {"local": True, "exllamav3": [{"venv": str(venv), "applied": ["dflash2_tp"]}]}}}
    report, diffs, _ = verify.verify(pins, ["here"])
    assert len(diffs) == 1 and "patch dflash2_tp is not applied" in diffs[0]


def test_the_stdin_program_says_built_in_for_a_1_6_2_package(tmp_path):
    venv = _venv_with_version(tmp_path, "1.6.2")
    script = ROOT / "patches" / "exllamav3" / "dflash2_tp.py"
    rc, out = _run_bootstrap(script, ["--target", str(venv), "--check"])
    assert rc == 1 and "built in, never patch" in out
    rc, out = _run_bootstrap(script, ["--target", str(venv), "--apply"])
    assert rc == 2 and "REFUSING" in out
