# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/tabbyapi/lora.py: the LoRA wiring for TabbyAPI's ExLlamaV3 backend, on a synthetic backends/exllamav3/model.py
that holds exactly the four anchors (the real file is checked by patches/verify.py and the live proof in docs/PROVENANCE.md)."""
import importlib.util
import os
import pathlib
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "patches"))

spec = importlib.util.spec_from_file_location("lora", os.path.join(ROOT, "patches", "tabbyapi", "lora.py"))
lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lora)

BACKEND = (
    "import pathlib\n"
    "from typing import Any, Dict, List\n"
    + lora.IMPORT_OLD +
    "\n\nclass ExllamaV3Container:\n"
    "    def __init__(self, model_directory):\n"
    + lora.INIT_OLD +
    "        self.model = None\n\n"
    + lora.STUBS_OLD +
    "\n    async def unload(self, loras_only: bool = False, **kwargs):\n"
    + lora.UNLOAD_OLD +
    "        self.model = None\n"
)


@pytest.fixture
def tabby(tmp_path):
    (tmp_path / "backends" / "exllamav3").mkdir(parents=True)
    (tmp_path / "backends" / "exllamav3" / "model.py").write_text(BACKEND, encoding="utf-8", newline="\n")
    return tmp_path


def test_the_wiring_replaces_the_stubs_and_compiles():
    out = lora.patch_source(BACKEND)
    compile(out, "model.py", "exec")
    assert "from exllamav3.model.lora import LoRA" in out and "self.loras = []" in out
    assert "LoRA loading is not hooked up" not in out
    assert "adapter = LoRA.from_directory(self.model, str(directory), scaling)" in out
    assert out.count("TENSOR PARALLEL AND LoRA ARE MUTUALLY EXCLUSIVE") == 1


def test_it_refuses_on_a_changed_upstream():
    with pytest.raises(ValueError, match=r"'import' anchor matched 0 times"):
        lora.patch_source(BACKEND.replace(lora.IMPORT_OLD, ""))
    with pytest.raises(ValueError, match=r"'load_loras/get_loras' anchor matched 0 times"):
        lora.patch_source(BACKEND.replace(lora.STUBS_OLD, ""))


def test_command_line_round_trip(tabby, capsys):
    t = str(tabby)
    assert lora.main(["--target", t]) == 1
    assert lora.main(["--target", t, "--apply"]) == 0
    assert lora.main(["--target", t]) == 0
    assert "PATCHED" in capsys.readouterr().out
    assert lora.main(["--target", t, "--revert"]) == 0
    assert (tabby / "backends" / "exllamav3" / "model.py").read_text(encoding="utf-8") == BACKEND


def test_there_is_no_hard_coded_target_any_more():
    src = pathlib.Path(lora.__file__).read_text(encoding="utf-8")
    assert "expanduser" not in src and "~/aio/tabbyAPI\"" not in src
    assert not hasattr(lora, "TARGET")
