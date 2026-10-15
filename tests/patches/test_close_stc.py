# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/tabbyapi/close_stc.py: the fix for the Windows "Too many open files" failure (exllamav3 1.6.0 leaks 8 fopen()
streams per safetensors file on every FAILED model load and the Windows C runtime allows 512). The command line is
--target + --check/--apply/--revert and --stdio-limit.

Driven on a synthetic copy of TabbyAPI's common/model.py and main.py (the real files are never touched):
- the hook lands at the end of _free_failed_part, the patched function really calls part.config.stc.close(),
  survives a close() that raises, and still unloads first;
- it is idempotent, refuses when TabbyAPI has changed or the failed-load patch is missing, keeps CRLF files CRLF,
  backs up before writing, and --check changes nothing and prints the diff;
- --stdio-limit adds the 8192 ceiling to main.py only on Windows.
"""
import ast
import importlib.util
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "patches"))
TOOL = os.path.join(ROOT, "patches", "tabbyapi", "close_stc.py")

MODEL_PY = '''"""model"""
import pathlib


def _free_failed_part(part, label: str):
    """Free one part of a load that failed."""

    children = getattr(part, "mp_children", None) or []
    if children:
        part.destroy_tp_context()

    # Then the weights and caches this process holds.
    try:
        part.unload()
    except Exception as exc:
        xlogger.warning(f"Failed-load cleanup: {label}.unload() raised ({exc}); unloading module by module")
        for module in getattr(part, "modules", None) or []:
            try:
                module.unload()
            except Exception:
                pass


async def _discard_failed_load(failed, exc: BaseException):
    """Give back everything a FAILED load took."""
    return None
'''

MAIN_PY = '''"""The main module."""
import os

import argparse
import platform
from loguru import logger

from common.tabby_config import config


async def entrypoint_async():
    return None
'''


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("close_stc", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tabby(tmp_path):
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "model.py").write_text(MODEL_PY, encoding="utf-8", newline="\n")
    (tmp_path / "main.py").write_text(MAIN_PY, encoding="utf-8", newline="\n")
    return tmp_path


def load_free_failed_part(src, warnings):
    """The patched _free_failed_part, compiled out of the patched source with a fake xlogger."""
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_free_failed_part")
    ns = {"xlogger": types.SimpleNamespace(warning=warnings.append)}
    exec(compile(ast.Module([fn], []), "model.py", "exec"), ns)
    return ns["_free_failed_part"]


class Part:
    def __init__(self, close_raises=False, order=None):
        self.order = order if order is not None else []
        outer = self

        class Stc:
            def close(self):
                outer.order.append("close")
                if close_raises:
                    raise AssertionError("new_tensors is not None")

        self.config = types.SimpleNamespace(stc=Stc())

    def unload(self):
        self.order.append("unload")


def test_the_hook_is_added_at_the_end_of_free_failed_part(tool):
    new = tool.patch_model_source(MODEL_PY)
    assert new.count("part.config.stc.close()") == 1 and tool.MARKER in new
    compile(new, "model.py", "exec")
    fn_end = new.index("async def _discard_failed_load")
    assert new.index("part.config.stc.close()") < fn_end
    assert new.index("part.unload()") < new.index("part.config.stc.close()")      # unload first, then the files


def test_the_patched_function_closes_the_weight_files_after_unloading(tool):
    warnings = []
    fn = load_free_failed_part(tool.patch_model_source(MODEL_PY), warnings)
    part = Part()
    fn(part, "model")
    assert part.order == ["unload", "close"] and warnings == []


def test_a_close_that_raises_is_a_warning_not_a_second_failure(tool):
    warnings = []
    fn = load_free_failed_part(tool.patch_model_source(MODEL_PY), warnings)
    part = Part(close_raises=True)
    fn(part, "draft_model")
    assert part.order == ["unload", "close"]
    assert len(warnings) == 1 and "draft_model weight files" in warnings[0]


def test_a_part_without_a_config_is_survived(tool):
    warnings = []
    fn = load_free_failed_part(tool.patch_model_source(MODEL_PY), warnings)
    bare = types.SimpleNamespace(unload=lambda: None)
    fn(bare, "vision_model")
    assert len(warnings) == 1


def test_already_patched_and_changed_sources_are_refused(tool):
    new = tool.patch_model_source(MODEL_PY)
    with pytest.raises(ValueError, match="already patched"):
        tool.patch_model_source(new)
    with pytest.raises(ValueError, match="apply patches/tabbyapi/failed_load.py first"):
        tool.patch_model_source("async def _discard_failed_load(failed, exc: BaseException):\n    pass\n")
    with pytest.raises(ValueError, match="matched 0 times"):
        tool.patch_model_source("def _free_failed_part(part, label):\n    pass\n")
    with pytest.raises(ValueError, match="does not end where expected"):
        tool.patch_model_source(MODEL_PY.replace("                pass\n", "                x = 1\n"))


def test_the_stdio_ceiling_goes_in_front_of_entrypoint_async(tool):
    new = tool.patch_main_source(MAIN_PY)
    compile(new, "main.py", "exec")
    assert new.count("_setmaxstdio(8192)") == 1 and tool.MARKER in new
    assert new.index("_setmaxstdio(8192)") < new.index("async def entrypoint_async")
    assert new.index("from common.tabby_config import config") < new.index("_setmaxstdio(8192)")
    assert 'platform.system() == "Windows"' in new
    with pytest.raises(ValueError, match="already patched"):
        tool.patch_main_source(new)
    with pytest.raises(ValueError, match="does not import platform"):
        tool.patch_main_source(MAIN_PY.replace("import platform\n", ""))


def test_check_changes_nothing_and_shows_the_diff(tool, tabby, capsys):
    before = {p: (tabby / p).read_bytes() for p in ("common/model.py", "main.py")}
    assert tool.main(["--target", str(tabby), "--check", "--diff", "--stdio-limit"]) == 1      # not applied
    out = capsys.readouterr().out
    assert "NOT patched" in out and "+        part.config.stc.close()" in out and "_setmaxstdio(8192)" in out
    assert {p: (tabby / p).read_bytes() for p in before} == before
    assert not [f for f in os.listdir(tabby / "common") if ".bb-orig-" in f or ".bak-" in f]
    assert not (tabby / ".bb-patch-close_stc.json").exists()


def test_apply_close_hook_only_by_default_with_a_backup(tool, tabby, capsys):
    assert tool.main(["--target", str(tabby), "--apply"]) == 0
    assert "part.config.stc.close()" in (tabby / "common" / "model.py").read_text(encoding="utf-8")
    assert (tabby / "main.py").read_text(encoding="utf-8") == MAIN_PY            # --stdio-limit not asked for
    assert (tabby / "common" / "model.py.bb-orig-close_stc").read_text(encoding="utf-8") == MODEL_PY
    capsys.readouterr()
    assert tool.main(["--target", str(tabby), "--apply"]) == 0                   # idempotent
    assert "already applied" in capsys.readouterr().out
    assert tool.main(["--target", str(tabby)]) == 0
    assert "PATCHED" in capsys.readouterr().out
    assert len([f for f in os.listdir(tabby / "common") if ".bb-orig-" in f]) == 1


def test_stdio_limit_is_applied_on_request_and_reverts_with_the_rest(tool, tabby):
    assert tool.main(["--target", str(tabby), "--apply", "--stdio-limit"]) == 0
    assert "_setmaxstdio(8192)" in (tabby / "main.py").read_text(encoding="utf-8")
    assert tool.main(["--target", str(tabby), "--revert"]) == 0
    assert (tabby / "main.py").read_text(encoding="utf-8") == MAIN_PY
    assert (tabby / "common" / "model.py").read_text(encoding="utf-8") == MODEL_PY


def test_the_stdio_limit_can_be_added_after_the_close_hook(tool, tabby):
    assert tool.main(["--target", str(tabby), "--apply"]) == 0
    assert tool.main(["--target", str(tabby), "--check", "--stdio-limit"]) == 1               # model.py applied, main.py not: PARTIAL
    assert tool.main(["--target", str(tabby), "--apply", "--stdio-limit"]) == 0
    assert tool.main(["--target", str(tabby), "--check", "--stdio-limit"]) == 0


def test_crlf_files_stay_crlf(tool, tabby):
    p = tabby / "common" / "model.py"
    p.write_bytes(MODEL_PY.replace("\n", "\r\n").encode("utf-8"))
    assert tool.main(["--target", str(tabby), "--apply"]) == 0
    raw = p.read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert b"part.config.stc.close()" in raw


def test_a_tabby_without_the_failed_load_patch_is_refused_and_left_alone(tool, tabby, capsys):
    p = tabby / "common" / "model.py"
    p.write_text("async def _discard_failed_load(failed, exc):\n    pass\n", encoding="utf-8")
    before = p.read_bytes()
    assert tool.main(["--target", str(tabby), "--check"]) == 1                    # not applied, and cannot be yet: not "drifted"
    assert "cannot be applied yet" in capsys.readouterr().out
    assert tool.main(["--target", str(tabby), "--apply"]) == 2
    assert "REFUSING" in capsys.readouterr().out and p.read_bytes() == before
    assert not [f for f in os.listdir(tabby / "common") if ".bb-orig-" in f]


def test_close_stc_must_be_reverted_before_failed_load(tool, tabby, capsys):
    """The two patches edit the same file; the state records make the order matter, and a wrong order is refused."""
    ptf_spec = importlib.util.spec_from_file_location("failed_load_for_stack",
                                                      os.path.join(ROOT, "patches", "tabbyapi", "failed_load.py"))
    # a model.py that both can patch: the failed-load patch's anchors, then close_stc on top of its result
    ptf = importlib.util.module_from_spec(ptf_spec)
    ptf_spec.loader.exec_module(ptf)
    upstream = (
        "import pathlib\n\nasync def unload_model(skip_wait: bool = False, shutdown: bool = False):\n    pass\n\n\n"
        "async def load_model_gen(model_path: pathlib.Path, **kwargs):\n    global container\n\n    async with load_lock:\n"
        "        container = None\n\n        new_container = await ExllamaV3Container.create(model_path.resolve(), hf_model, **kwargs)\n\n"
        "        async with status_display.suspended():\n            progress = get_loading_progress_bar()\n"
        "            progress.start()\n\n            try:\n                async for module, modules in new_container.load_gen():\n"
        "                    yield module, modules, \"model\"\n\n                container = new_container\n"
        "            finally:\n                progress.stop()\n")
    (tabby / "common" / "model.py").write_text(upstream, encoding="utf-8", newline="\n")
    assert ptf.main(["--target", str(tabby), "--apply"]) == 0
    assert tool.main(["--target", str(tabby), "--apply"]) == 0
    capsys.readouterr()
    assert ptf.main(["--target", str(tabby), "--revert"]) == 2                  # close_stc is on top of it
    assert "changed since the apply" in capsys.readouterr().out
    assert tool.main(["--target", str(tabby), "--revert"]) == 0
    assert ptf.main(["--target", str(tabby), "--revert"]) == 0
    assert (tabby / "common" / "model.py").read_text(encoding="utf-8") == upstream




def test_adopting_a_stacked_pair_needs_the_top_patch_first(tool, tabby, capsys):
    """A common/model.py patched by earlier copies of the scripts: upstream -> failed_load (.bak-1) -> close_stc (.bak-2).
    --adopt of close_stc finds .bak-2 (the failed_load output); failed_load cannot be adopted while close_stc sits on top, and
    after close_stc is reverted by the kit it can (with .bak-1). Everything then reverts down to upstream."""
    spec = importlib.util.spec_from_file_location("failed_load_adopt", os.path.join(ROOT, "patches", "tabbyapi", "failed_load.py"))
    fl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fl)
    upstream = (
        "import pathlib\n\nasync def unload_model(skip_wait: bool = False, shutdown: bool = False):\n    pass\n\n\n"
        "async def load_model_gen(model_path: pathlib.Path, **kwargs):\n    global container\n\n    async with load_lock:\n"
        "        container = None\n\n        new_container = await ExllamaV3Container.create(model_path.resolve(), hf_model, **kwargs)\n\n"
        "        async with status_display.suspended():\n            progress = get_loading_progress_bar()\n"
        "            progress.start()\n\n            try:\n                async for module, modules in new_container.load_gen():\n"
        "                    yield module, modules, \"model\"\n\n                container = new_container\n"
        "            finally:\n                progress.stop()\n")
    after_fl = fl.patch_source(upstream)
    after_cs = tool.patch_model_source(after_fl)
    common = tabby / "common"
    (common / "model.py.bak-20261010-165943").write_text(upstream, encoding="utf-8", newline="\n")
    (common / "model.py.bak-20261014-084849").write_text(after_fl, encoding="utf-8", newline="\n")
    (common / "model.py").write_text(after_cs, encoding="utf-8", newline="\n")

    assert tool.main(["--target", str(tabby), "--adopt"]) == 0
    assert "model.py.bak-20261014-084849" in capsys.readouterr().out
    assert fl.main(["--target", str(tabby), "--adopt"]) == 2                     # close_stc is on top of it
    assert "no old backup" in capsys.readouterr().out
    assert fl.main(["--target", str(tabby), "--revert"]) == 2
    assert tool.main(["--target", str(tabby), "--revert"]) == 0                  # the kit takes close_stc off ...
    assert (common / "model.py").read_text(encoding="utf-8") == after_fl
    assert fl.main(["--target", str(tabby), "--adopt"]) == 0                     # ... and now failed_load can be adopted
    assert "model.py.bak-20261010-165943" in capsys.readouterr().out
    assert tool.main(["--target", str(tabby), "--apply"]) == 0                   # close_stc goes back on, kit-style
    assert (common / "model.py").read_text(encoding="utf-8") == after_cs
    assert tool.main(["--target", str(tabby), "--revert"]) == 0
    assert fl.main(["--target", str(tabby), "--revert"]) == 0
    assert (common / "model.py").read_text(encoding="utf-8") == upstream
