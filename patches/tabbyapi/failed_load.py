# SPDX-License-Identifier: AGPL-3.0-or-later
# This file exists only to modify TabbyAPI (https://github.com/theroyallab/tabbyAPI, AGPL-3.0) and embeds excerpts of it;
# it is offered under the same licence. Full text: patches/tabbyapi/LICENSE. Notes: patches/NOTICE.md.
"""A FAILED model load gives back everything it took.

THE CAUSE, in TabbyAPI's common/model.py load_model_gen (upstream 2fd6cc7):
  `container` is reset to None, a new container is built, and its load runs. If the load raises part way -- most
  often CUDA out of memory while the cache, the draft model or a tensor-parallel shard is allocated -- the
  exception leaves before `container = new_container`, and NOTHING unloads the half-built container. Its weights
  and cache stay on the GPUs until the process restarts, and every later load on that card fails for lack of
  memory: the "ghost load" (observed: 21.5 GB held on a 24 GB card while GET /v1/model answered "no model").
THE SECOND CAUSE, in exllamav3 (1.6.x model_tp.py): a tensor-parallel load spawns one worker process per GPU in
  create_tp_context(), but unload_tp() destroys them only when `loaded_tp` is set -- and that is set at the END of
  a SUCCESSFUL load. So a failed pair load leaves its spawned workers alive holding their shards (the orphan
  `spawn_main` processes, observed holding up to 22.6 GB), even if something does call unload().

THE FIX (this patch, to common/model.py only): load_model_gen catches a failed load, logs the full traceback,
clears the traceback's frames (their locals can pin tensors), destroys the tensor-parallel workers directly
(terminate, then kill, if exllamav3's own teardown fails), unloads every part (model, draft, vision; module by
module if a whole-part unload fails), closes a generator if one was built, collects garbage, empties the CUDA
cache -- and re-raises the original error, so the caller still reads why the load failed. A server shutdown
(CancelledError) is left alone: the process ends.

    python patches/tabbyapi/failed_load.py --target ~/tabbyAPI          # --check is the default
    ... --apply     patches it         ... --revert   puts the backup back (refuses if the file changed since)
    ... --diff      with --check: shows the change

Exit codes (--check): 0 applied, 1 not applied, 2 anchors drifted. Idempotent (the marker below), refuses unless every
anchor matches exactly once, backs up first, keeps the file's line endings, compiles the result. The running TabbyAPI
picks it up at its next start: patch, then restart the server only while its card is idle (or at a planned restart).
Upstream: TabbyAPI is AGPL-3.0 (see patches/NOTICE.md and patches/tabbyapi/LICENSE); pinned in patches/PINS.toml.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # patches/ (the shared kit)
import _patchkit  # noqa: E402

MARKER = "_discard_failed_load"

# ── the helpers, inserted right before load_model_gen ────────────────────────────────────────────────────────────
HELPERS_ANCHOR = "async def load_model_gen(model_path: pathlib.Path, **kwargs):\n"
HELPERS = '''def _free_failed_part(part, label: str):
    """Free one part (model, draft or vision model) of a load that failed. BigBrain patch failed_load
    (patches/tabbyapi/failed_load.py): see _discard_failed_load."""

    # Tensor-parallel workers first. exllamav3 destroys them in unload_tp() only once `loaded_tp` is set, which
    # happens when the load SUCCEEDED, so after a failed TP load they would live on holding their shards.
    children = getattr(part, "mp_children", None) or []
    if children and not getattr(part, "loaded_tp", False):
        try:
            part.destroy_tp_context()
        except Exception as exc:
            xlogger.warning(f"Failed-load cleanup: {label} TP teardown raised ({exc}); terminating its workers")
            for child in children:
                try:
                    if child is not None and hasattr(child, "pid") and child.is_alive():
                        child.terminate()
                        child.join(timeout=5)
                        if child.is_alive():
                            child.kill()
                            child.join(timeout=5)
                except Exception as exc2:
                    xlogger.warning(f"Failed-load cleanup: {label} worker: {exc2}")
            try:
                part.mp_children, part.mp_parent_conn, part.mp_child_conn = [], [], []
            except Exception:
                pass

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
    """Give back everything a FAILED load took. BigBrain patch failed_load
    (patches/tabbyapi/failed_load.py).

    Without this, a load that fails part way (most often CUDA out of memory while the cache, the draft model or a
    tensor-parallel shard is allocated) leaves its half-built container on the GPUs: `container` was reset to None
    before the load and nothing unloads the new one, so the memory stays held until the process restarts and every
    later load on the card fails for lack of memory."""

    import gc
    import traceback

    import torch

    xlogger.error(
        f"Model load failed ({type(exc).__name__}: {str(exc)[:300]}); freeing what it allocated.\\n"
        + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )
    # The traceback's frames hold the loader's locals, and those can pin GPU tensors
    traceback.clear_frames(exc.__traceback__)

    generator = getattr(failed, "generator", None)
    if generator is not None:
        try:
            await generator.close()
        except Exception as gen_exc:
            xlogger.warning(f"Failed-load cleanup: generator close: {gen_exc}")

    for label in ("vision_model", "draft_model", "model"):
        part = getattr(failed, label, None)
        if part is not None:
            _free_failed_part(part, label)

    for attr in ("generator", "cache", "draft_cache", "vision_model", "draft_model", "model"):
        if hasattr(failed, attr):
            try:
                setattr(failed, attr, None)
            except Exception:
                pass
    try:
        failed.loaded = False
    except Exception:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    xlogger.info("Failed-load cleanup finished: the failed load's memory is released.")


'''

# ── the wrap around the load in load_model_gen ───────────────────────────────────────────────────────────────────
BLOCK_START = "        new_container = await ExllamaV3Container.create(model_path.resolve(), hf_model, **kwargs)\n"
BLOCK_END = ("                container = new_container\n"
             "            finally:\n"
             "                progress.stop()\n")
WRAP_HEAD = ("        new_container = None\n"
             "        try:\n")
WRAP_TAIL = ("        except Exception as exc:\n"
             "            # A FAILED LOAD GIVES BACK WHAT IT TOOK (BigBrain patch failed_load): without this the\n"
             "            # half-built container kept its VRAM until the process restarted. The error is re-raised\n"
             "            # unchanged, so the caller still sees why the load failed.\n"
             "            if new_container is not None and container is not new_container:\n"
             "                await _discard_failed_load(new_container, exc)\n"
             "            raise\n")


def patch_source(src: str) -> str:
    """The patched text of common/model.py (LF line endings in and out). Raises ValueError when an anchor does not
    match exactly once, or the file is already patched."""
    if MARKER in src:
        raise ValueError("already patched")
    for label, anchor in (("load_model_gen", HELPERS_ANCHOR), ("container create", BLOCK_START),
                          ("container assign", BLOCK_END)):
        n = src.count(anchor)
        if n != 1:
            raise ValueError(f"anchor '{label}' matched {n} times, expected 1 -- TabbyAPI changed; re-read it")
    start = src.index(BLOCK_START)
    end = src.index(BLOCK_END, start) + len(BLOCK_END)
    if src.index(HELPERS_ANCHOR) > start:
        raise ValueError("load_model_gen comes after the container create -- TabbyAPI changed; re-read it")
    block = src[start:end]
    indented = "".join(("    " + ln) if ln.strip() else ln for ln in block.splitlines(keepends=True))
    src = src[:start] + WRAP_HEAD + indented + WRAP_TAIL + src[end:]
    return src.replace(HELPERS_ANCHOR, HELPERS + HELPERS_ANCHOR, 1)


PATCH = _patchkit.Patch(
    name="failed_load",
    summary=__doc__.split("\n")[0],
    edits=[_patchkit.Edit("common/model.py", MARKER, patch_source)],
)


def main(argv=None) -> int:
    return _patchkit.main(PATCH, argv)


if __name__ == "__main__":
    sys.exit(main())
