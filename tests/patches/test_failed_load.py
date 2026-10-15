# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/tabbyapi/failed_load.py: a FAILED TabbyAPI model load gives back everything it took.
The command line is --target + --check/--apply/--revert.

Two layers:
  * the patch itself (anywhere): every anchor must match once, the result compiles, a second run refuses, and the
    wrap re-indents exactly the load block;
  * the patched TabbyAPI (only where a TabbyAPI checkout is named by TABBY_TEST_DIR and its venv runs this test):
    load_model_gen with a fake container whose load fails part way -- the failed part is unloaded, a REAL spawned
    worker process standing in for a tensor-parallel shard is terminated even when exllamav3's teardown raises,
    the original error still reaches the caller, and a load that succeeds is left alone. No GPU is touched:
    CUDA_VISIBLE_DEVICES is emptied before torch is imported, so empty_cache() is a no-op.

    python -m pytest tests/patches/test_failed_load.py -q
    # on a TabbyAPI host, against a scratch COPY of the checkout patched with the tool:
    CUDA_VISIBLE_DEVICES= TABBY_TEST_DIR=/tmp/tabbytest <venv>/bin/python -m pytest tests/patches/test_failed_load.py -q
"""
import asyncio
import importlib.util
import multiprocessing
import os
import pathlib
import sys
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "patches"))


def _load(rel):
    spec = importlib.util.spec_from_file_location(pathlib.Path(rel).stem, os.path.join(ROOT, "patches", *rel.split("/")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ptf = _load("tabbyapi/failed_load.py")

# The shape of upstream common/model.py's load_model_gen around the patched region (2fd6cc7), trimmed.
UPSTREAM_SHAPE = '''import pathlib

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


def test_patch_wraps_the_load_and_compiles():
    out = ptf.patch_source(UPSTREAM_SHAPE)
    compile(out, "model.py", "exec")
    assert out.index("def _discard_failed_load") < out.index("async def load_model_gen")
    assert "        new_container = None\n        try:\n            new_container = await ExllamaV3Container" in out
    assert ("                    container = new_container\n                finally:\n"
            "                    progress.stop()\n        except Exception as exc:") in out
    assert "await _discard_failed_load(new_container, exc)\n            raise\n" in out
    # everything after the block is untouched and still outside the try
    assert out.endswith("        sampling.clear_model_overrides()\n")


def test_patch_refuses_twice_and_on_a_changed_tabbyapi():
    once = ptf.patch_source(UPSTREAM_SHAPE)
    with pytest.raises(ValueError, match="already patched"):
        ptf.patch_source(once)
    with pytest.raises(ValueError, match="container assign"):
        ptf.patch_source(UPSTREAM_SHAPE.replace("                container = new_container\n", ""))


def test_tool_keeps_crlf_and_reports_check(tmp_path, capsys):
    common = tmp_path / "common"
    common.mkdir()
    (common / "model.py").write_bytes(UPSTREAM_SHAPE.replace("\n", "\r\n").encode())
    assert ptf.main(["--target", str(tmp_path)]) == 1                       # --check is the default: not applied
    assert "NOT patched" in capsys.readouterr().out
    assert ptf.main(["--target", str(tmp_path), "--apply"]) == 0
    raw = (common / "model.py").read_bytes()
    assert b"_discard_failed_load" in raw and b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert ptf.main(["--target", str(tmp_path), "--check"]) == 0
    assert "PATCHED" in capsys.readouterr().out
    assert (common / "model.py.bb-orig-failed_load").exists(), "a backup is written before the change"
    assert (tmp_path / ".bb-patch-failed_load.json").exists(), "and so is the state record --revert uses"


def test_revert_puts_the_original_back_and_a_second_revert_has_nothing_to_do(tmp_path, capsys):
    common = tmp_path / "common"
    common.mkdir()
    original = UPSTREAM_SHAPE.replace("\n", "\r\n").encode()
    (common / "model.py").write_bytes(original)
    assert ptf.main(["--target", str(tmp_path), "--apply"]) == 0
    assert (common / "model.py").read_bytes() != original
    assert ptf.main(["--target", str(tmp_path), "--revert"]) == 0
    assert (common / "model.py").read_bytes() == original
    assert not list(common.glob("model.py.bb-orig-*")) and not (tmp_path / ".bb-patch-failed_load.json").exists()
    assert ptf.main(["--target", str(tmp_path), "--revert"]) == 1            # nothing to revert
    assert ptf.main(["--target", str(tmp_path), "--check"]) == 1


def test_revert_refuses_when_the_file_changed_since_the_apply(tmp_path, capsys):
    common = tmp_path / "common"
    common.mkdir()
    (common / "model.py").write_bytes(UPSTREAM_SHAPE.encode())
    assert ptf.main(["--target", str(tmp_path), "--apply"]) == 0
    patched = (common / "model.py").read_bytes()
    (common / "model.py").write_bytes(patched + b"# someone edited this\n")
    capsys.readouterr()
    assert ptf.main(["--target", str(tmp_path), "--revert"]) == 2
    assert "changed since the apply" in capsys.readouterr().out
    assert (common / "model.py").read_bytes() == patched + b"# someone edited this\n", "a refused revert changes nothing"


def test_check_reports_a_drifted_upstream_with_exit_2(tmp_path, capsys):
    common = tmp_path / "common"
    common.mkdir()
    (common / "model.py").write_bytes(UPSTREAM_SHAPE.replace("                container = new_container\n", "").encode())
    assert ptf.main(["--target", str(tmp_path)]) == 2
    assert "DRIFTED" in capsys.readouterr().out
    assert ptf.main(["--target", str(tmp_path), "--apply"]) == 2             # refuses, writes nothing
    assert not list(common.glob("model.py.bb-orig-*"))


# ── the patched TabbyAPI itself ──────────────────────────────────────────────────────────────────────────────────

TABBY = os.environ.get("TABBY_TEST_DIR")
# HARDWARE/ENVIRONMENT ONLY: these four cases import a real, patched TabbyAPI checkout and its venv (exllamav3, torch, a GPU
# or at least its CPU fallbacks). They are skipped everywhere else; set TABBY_TEST_DIR to the checkout to run them.
needs_tabby = pytest.mark.skipif(not TABBY, reason="SKIPPED (needs a patched TabbyAPI checkout): set TABBY_TEST_DIR to run")


def _sleeper():
    time.sleep(600)


class FakePart:
    """A model, draft or vision part: unload() and, for tensor parallel, spawned workers."""

    def __init__(self, workers=0, destroy_raises=False, unload_raises=False):
        self.loaded_tp = False
        self.mp_children = []
        self.destroyed = self.unloaded = False
        self.destroy_raises, self.unload_raises = destroy_raises, unload_raises
        self.module_unloads = 0
        self.modules = [types.SimpleNamespace(unload=self._module_unload) for _ in range(3)]
        if workers:
            ctx = multiprocessing.get_context("spawn")
            for _ in range(workers):
                p = ctx.Process(target=_sleeper, daemon=True)
                p.start()
                self.mp_children.append(p)
            self.mp_children.append(None)     # exllamav3 keeps None for devices it does not use

    def _module_unload(self):
        self.module_unloads += 1

    def destroy_tp_context(self):
        if self.destroy_raises:
            raise RuntimeError("TP teardown raised (fake)")
        self.destroyed = True
        for c in self.mp_children:
            if c is not None:
                c.terminate()
                c.join(5)
        self.mp_children = []

    def unload(self):
        if self.unload_raises:
            raise RuntimeError("unload raised (fake)")
        self.unloaded = True


class FakeContainer:
    use_draft_model = False
    use_vision = False
    warmup_enabled = False

    def __init__(self, part, fail=True, draft=None):
        self.model, self.draft_model, self.vision_model = part, draft, None
        self.generator, self.cache, self.loaded, self.fail = None, object(), False, fail

    async def load_gen(self, progress_callback=None, **kwargs):
        yield 0, 4
        yield 1, 4
        if self.fail:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (fake)")
        for i in range(2, 5):
            yield i, 4
        self.loaded = True


@pytest.fixture
def cm(monkeypatch):
    sys.path.insert(0, TABBY)
    monkeypatch.chdir(TABBY)
    import common.model as cm  # noqa: E402

    assert ptf.MARKER in open(cm.__file__, encoding="utf-8").read(), "TABBY_TEST_DIR is not patched"

    async def defaults(model_path, **kw):
        return kw

    async def hf(path):
        return types.SimpleNamespace(hf_config=types.SimpleNamespace(get_max_position_embeddings=lambda: 4096))

    monkeypatch.setattr(cm, "apply_load_defaults", defaults)
    monkeypatch.setattr(cm.HFModel, "from_directory", staticmethod(hf))
    monkeypatch.setattr(cm, "validate_backend", lambda *a, **k: None)
    monkeypatch.setattr(cm.sampling, "split_sampling_section", lambda *a, **k: (None, None))
    monkeypatch.setattr(cm, "container", None)
    return cm


def _run(cm, fake):
    async def create(*a, **k):
        return fake

    cm.ExllamaV3Container.create = staticmethod(create)

    async def drive():
        return [p async for p in cm.load_model_gen(pathlib.Path("fake-model"), max_seq_len=4096)]

    return asyncio.run(drive())


@needs_tabby
def test_a_failed_tp_load_kills_its_workers_even_when_the_teardown_raises(cm):
    part = FakePart(workers=2, destroy_raises=True)
    workers = [c for c in part.mp_children if c is not None]
    assert all(w.is_alive() for w in workers)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _run(cm, FakeContainer(part))
    for w in workers:
        w.join(10)
    assert not any(w.is_alive() for w in workers), "the failed load's TP workers must not outlive it"
    assert part.unloaded, "the failed part is unloaded"
    assert cm.container is None, "a failed load leaves no container claiming to be loaded"


@needs_tabby
def test_a_failed_tp_load_uses_exllamavs_own_teardown_when_it_works(cm):
    part = FakePart(workers=1)
    workers = [c for c in part.mp_children if c is not None]
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _run(cm, FakeContainer(part))
    assert part.destroyed and part.unloaded
    assert not any(w.is_alive() for w in workers)


@needs_tabby
def test_a_failed_load_whose_unload_raises_is_freed_module_by_module_and_the_draft_too(cm):
    part, draft = FakePart(unload_raises=True), FakePart()
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _run(cm, FakeContainer(part, draft=draft))
    assert part.module_unloads == 3, "every module is unloaded when the whole-part unload raises"
    assert draft.unloaded


@needs_tabby
def test_a_load_that_succeeds_is_not_touched(cm):
    part = FakePart(workers=1)
    workers = [c for c in part.mp_children if c is not None]
    fake = FakeContainer(part, fail=False)
    progress = _run(cm, fake)
    assert progress[-1][:2] == (4, 4)
    assert cm.container is fake and not part.unloaded and not part.destroyed
    assert all(w.is_alive() for w in workers)
    for w in workers:
        w.terminate()
        w.join(5)
