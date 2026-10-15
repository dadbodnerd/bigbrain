# patches/

Optional changes to upstream **TabbyAPI** and **exllamav3**, as scripts you can check, apply and undo. **None of this is needed to run
BigBrain.** The pipeline-parallel engine is `tools/bigbrain/`; it only borrows an interpreter that has exllamav3. These are reliability fixes
the author applies to the TabbyAPI servers that share the same machines, kept here with one version pin and one way to tell whether a host has
them. They edit upstream source, so read the licence note at the end first.

| patch | against | what it does | file |
|---|---|---|---|
| `failed_load` | TabbyAPI `common/model.py` | a failed model load gives back everything it took (the "ghost loads") | `tabbyapi/failed_load.py` |
| `close_stc` | TabbyAPI `common/model.py` (+ `main.py` with `--stdio-limit`) | a failed load closes its weight files (Windows "Too many open files"); needs `failed_load` first | `tabbyapi/close_stc.py` |
| `lora` | TabbyAPI `backends/exllamav3/model.py` | wires the ExLlamaV3 backend to exllamav3's LoRA support (not usable with tensor parallel) | `tabbyapi/lora.py` |
| `auth_redact` | TabbyAPI `common/auth.py` | the API and admin keys are not printed into the start-up log | `tabbyapi/auth_redact.patch` |
| `windows_launcher` | TabbyAPI `main.py`, `start.bat` | Windows: no winloop, and `start.bat` runs `main.py` | `tabbyapi/windows_launcher.patch` |
| `dflash2_tp` | exllamav3 **1.6.0 and 1.6.1 only** (6 files) | DFlash2 speculative decoding on a tensor-parallel target. **exllamav3 1.6.2 has this built in: never patch it** (see below) | `exllamav3/dflash2_tp.py` |

`PINS.toml` records the upstream versions the patches were written against: TabbyAPI commit `2fd6cc76203a` and exllamav3 1.6.0 / 1.6.1. A patch
whose anchors no longer match (upstream moved) refuses to apply and says so; it never guesses.

## One command line

```
python patches/tabbyapi/failed_load.py --target ~/tabbyAPI                 # --check is the default
python patches/tabbyapi/failed_load.py --target ~/tabbyAPI --diff          # also show what --apply would change
python patches/tabbyapi/failed_load.py --target ~/tabbyAPI --apply
python patches/tabbyapi/failed_load.py --target ~/tabbyAPI --revert
python patches/patchfile.py patches/tabbyapi/auth_redact.patch --target ~/tabbyAPI [--apply|--revert]
python patches/exllamav3/dflash2_tp.py --target ~/tabbyAPI/venv            # a venv root, site-packages, or the package dir
python patches/tabbyapi/failed_load.py --target DIR --adopt                # a tree an EARLIER copy of the script patched: record it so --revert works
```

| | `--check` | `--apply` | `--revert` | `--adopt` |
|---|---|---|---|---|
| 0 | applied | done, or already applied | done | recorded (or already) |
| 1 | not applied (or only part of it) | | nothing to revert | |
| 2 | anchors drifted: upstream changed, re-read it | refused, nothing written | refused: the file changed since the apply | refused: not patched, or no old backup reproduces it |

Applying backs up first (`<file>.bb-orig-<patch>`), keeps each file's line endings, compiles Python before writing, and is all-or-nothing across a
patch's files, also when a write fails part way (what was written is put back). `--revert` restores that backup and refuses if the file is not exactly
what the apply left, so it can never undo someone else's edit. If two patches edit one file (`failed_load` then `close_stc`), revert in the opposite
order. Details and the state record: `_patchkit.py`.

`--adopt` is for a tree that an earlier copy of these scripts patched (so there is no state record and `--revert` would refuse): it looks through the
`<file>.bak-*` / `<file>.orig-*` backups for the one that reproduces the current file when this patch is applied to it (checked, not assumed), copies it
to `<file>.bb-orig-<patch>`, and records both hashes. It changes nothing else. When one patch sits on another (`close_stc` on `failed_load`), adopt the
top one first. A backup that does not reproduce the file is refused, which also catches a stale backup.

## The exllamav3 version gate

`exllamav3` 1.6.2 carries DFlash2 tensor parallel itself, so `dflash2_tp.py` must never be applied to it. The script reads `<package>/version.py`:
`--apply` refuses anything but `1.6.0` and `1.6.1` (a `+local` suffix is fine, a dev or rc build is not), `--check` on 1.6.2 or later prints
`built in, never patch` and exits 1 (it exits 2 if our patch is on such a version anyway), and `verify.py` shows it as `not applied (built in)`,
which is what a host entry with `absent = ["dflash2_tp"]` expects. The two version numbers are in the script and, as documentation that a test
keeps equal, in `PINS.toml`.

## Is a host as it should be?

```
cp patches/hosts.example.toml patches/hosts.toml        # your machines, which trees and venvs they have, which patches each should carry
python patches/verify.py --hosts patches/hosts.toml --all
python patches/verify.py --hosts patches/hosts.toml --host alpha
```

(The hosts file can also come from `$BIGBRAIN_PATCH_HOSTS` or `[patches] hosts` in `site.toml`.) Read-only. For each tree and venv in the file it prints
the upstream version found and every patch's state, and reports each difference from what the file says. It never fixes one. Linux hosts are checked
over ssh with the script sent on stdin (nothing is copied to them); a host marked `local = true` is checked in-process, which is how a Windows PC
without ssh is covered. Exit 0 = every host matches, 1 = a difference, 3 = a host could not be reached.

## Licences

`PINS.toml` pins the upstream versions. `NOTICE.md`: `tabbyapi/` is **AGPL-3.0-or-later** (`tabbyapi/LICENSE`) because those files exist only to modify
TabbyAPI; `exllamav3/` carries exllamav3's **MIT** notice. Every file there has an SPDX tag: a comment on the `.py` files, a `<name>.patch.license`
sidecar for each `.patch` (the diff itself stays byte-exact so it applies cleanly). No upstream file is copied into this repository.
