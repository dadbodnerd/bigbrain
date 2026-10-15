# SPDX-License-Identifier: AGPL-3.0-or-later
# This file exists only to modify TabbyAPI (https://github.com/theroyallab/tabbyAPI, AGPL-3.0) and embeds excerpts of it;
# it is offered under the same licence. Full text: patches/tabbyapi/LICENSE. Notes: patches/NOTICE.md.
"""A FAILED model load closes the weight files it opened.

THE SYMPTOM. A TabbyAPI on Windows (exllamav3 1.6.0) answered 273 model loads in a row with "RuntimeError: ## Error
opening file '...safetensors': Too many open files (errno=24)" until it was restarted.

THE CAUSE (read in the source and reproduced in a throwaway process):
  * exllamav3's stloader (exllamav3_ext/stloader.cpp, stloader_open_file) opens STLOADER_THREADS = 8 fopen()
    streams per safetensors file on Windows (one shared stream on Linux), and closes them in
    SafetensorsCollection.close().
  * model/model_ls.py _load_autosplit calls `config.stc.close()` ONLY at its end, i.e. when the load SUCCEEDED.
    A load that raises first ("Insufficient VRAM in split for model and cache", an out-of-memory error, a
    cancelled request) never closes, and SafetensorsCollection has no __del__: the streams stay open for the
    life of the process. A 2-file model leaks 16 streams per failed load.
  * The Windows C runtime allows 512 open stdio streams (_getmaxstdio() == 512), so the 33rd load failed,
    and so did every one after it. Each stream is also one process handle.
  * Upstream (turboderp-org/exllamav3 master and v1.6.1, TabbyAPI main) has the same shape; no issue or fix was found.

THE FIX, in the file failed_load.py already patches (common/model.py, `_free_failed_part`): after the failed part is
unloaded, close its weight files:
    part.config.stc.close()
close() is idempotent (a closed handle is set to None), so it is safe after a TP load that closed already.
Nothing in exllamav3 itself is changed. Needs failed_load.py applied first.

OPTIONAL SECOND LAYER (--stdio-limit, main.py, Windows only): raise the C runtime's stream ceiling from 512 to
8192 at start-up (`ctypes.cdll.ucrtbase._setmaxstdio(8192)`). The extension shares the process's UCRT, so it
applies to stloader_open_file (measured: 400 files = 3,200 streams opened fine with it, 63 files failed without
it). It does not stop the leak, it only makes it 16 times slower; restarting the server after a failed load stays
the safety net either way.

    python patches/tabbyapi/close_stc.py --target C:\\tabbyAPI --diff             # status + the diff, changes nothing
    python patches/tabbyapi/close_stc.py --target C:\\tabbyAPI --apply            # apply the close hook
    python patches/tabbyapi/close_stc.py --target C:\\tabbyAPI --apply --stdio-limit   # also the 8192 ceiling
    ... --revert    puts the backup back (refuses if the file changed since; revert this BEFORE failed_load)

Exit codes (--check): 0 applied, 1 not applied (also when patches/tabbyapi/failed_load.py has not been applied first:
--check says "cannot be applied yet" and --apply refuses with 2), 2 anchors drifted. Idempotent (the marker below), backs up
first, keeps the file's line endings, compiles the result. The running TabbyAPI picks it up at its next start (restart it
only while it is idle).
Upstream: TabbyAPI is AGPL-3.0 (see patches/NOTICE.md and patches/tabbyapi/LICENSE); pinned in patches/PINS.toml.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # patches/ (the shared kit)
import _patchkit  # noqa: E402

MARKER = "bigbrain_close_stc"

# ── common/model.py: close the failed part's weight files, at the end of _free_failed_part ───────────────────────
CLOSE_ANCHOR = "\n\nasync def _discard_failed_load(failed, exc: BaseException):\n"
CLOSE_HOOK = '''
    # And the weight files. exllamav3 closes its safetensors streams (config.stc.close()) only when a load
    # SUCCEEDS; a failed one leaks 8 fopen() streams per file on Windows (1 on Linux) for the life of the process,
    # and the Windows C runtime allows 512: 2026-10-13, 273 loads in a row failed with "Too many open files".
    # BigBrain patch bigbrain_close_stc (patches/tabbyapi/close_stc.py). close() is idempotent.
    try:
        part.config.stc.close()
    except Exception as exc:
        xlogger.warning(f"Failed-load cleanup: {label} weight files: {exc}")
'''

# ── main.py: raise the C runtime's open-stream ceiling (Windows), before anything loads ─────────────────────────
STDIO_ANCHOR = "\n\nasync def entrypoint_async():\n"
STDIO_BLOCK = '''

# BigBrain patch bigbrain_close_stc (patches/tabbyapi/close_stc.py --stdio-limit): the Windows C runtime
# allows 512 open stdio streams and exllamav3 opens 8 per safetensors file, so a leak of a few dozen failed loads
# exhausts it ("Too many open files", errno 24). Raise the ceiling to 8192; the leak itself is closed in
# common/model.py.
if platform.system() == "Windows":
    try:
        import ctypes

        ctypes.cdll.ucrtbase._setmaxstdio(8192)
    except Exception:
        pass
'''


def patch_model_source(src: str) -> str:
    """common/model.py with the close hook added (LF line endings in and out). ValueError when it cannot."""
    if MARKER in src:
        raise ValueError("already patched")
    if "def _free_failed_part(" not in src:
        raise ValueError("common/model.py has no _free_failed_part -- apply patches/tabbyapi/failed_load.py first")
    n = src.count(CLOSE_ANCHOR)
    if n != 1:
        raise ValueError(f"anchor '_discard_failed_load' matched {n} times, expected 1 -- TabbyAPI changed; re-read it")
    head, tail = src.split(CLOSE_ANCHOR, 1)
    if not head.rstrip("\n").endswith(("pass", "unload()")):
        raise ValueError("_free_failed_part does not end where expected -- TabbyAPI changed; re-read it")
    return head.rstrip("\n") + "\n" + CLOSE_HOOK.rstrip("\n") + "\n" + CLOSE_ANCHOR + tail


def patch_main_source(src: str) -> str:
    """main.py with the stdio ceiling added (LF in and out). ValueError when it cannot."""
    if MARKER in src:
        raise ValueError("already patched")
    n = src.count(STDIO_ANCHOR)
    if n != 1:
        raise ValueError(f"anchor 'entrypoint_async' matched {n} times, expected 1 -- TabbyAPI changed; re-read it")
    if "import platform" not in src:
        raise ValueError("main.py does not import platform -- TabbyAPI changed; re-read it")
    head, tail = src.split(STDIO_ANCHOR, 1)
    return head + STDIO_BLOCK.rstrip("\n") + "\n" + STDIO_ANCHOR + tail


def needs_failed_load(src: str) -> None:
    """--check reads 'not applied (cannot be applied yet)' instead of 'drifted' when only the prerequisite is missing."""
    if "def _free_failed_part(" not in src:
        raise ValueError("common/model.py has no _free_failed_part -- apply patches/tabbyapi/failed_load.py first")


PATCH = _patchkit.Patch(
    name="close_stc",
    summary=__doc__.split("\n")[0] + " " + __doc__.split("\n")[1],
    edits=[
        _patchkit.Edit("common/model.py", MARKER, patch_model_source, needs=needs_failed_load),
        _patchkit.Edit("main.py", MARKER, patch_main_source, opt="stdio-limit"),
    ],
    options={"stdio-limit": "also raise the C runtime's open-stream ceiling to 8192 in main.py (Windows)"},
)


def main(argv=None) -> int:
    return _patchkit.main(PATCH, argv)


if __name__ == "__main__":
    sys.exit(main())
