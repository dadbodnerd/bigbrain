# Third-party notices for `patches/`

BigBrain's own code and documentation (the engine in `tools/bigbrain/`, `deploy/`, `tests/`, `docs/`, the top-level `*.md` files, and the
scripts and tests under `patches/` outside the two upstream-derived subtrees below) are released under the **MIT License**,
Copyright (c) 2026 dadbodnerd (see `/LICENSE`).

`patches/` is different. Its files exist only to change two upstream programs, and they embed small excerpts of
those programs as anchors and as the lines they replace. Those excerpts keep their own licences. Nothing from
TabbyAPI or exllamav3 is vendored here: no upstream source file is in this repository, only scripts that edit a
checkout or venv you already have, and the pins in `PINS.toml` that say which version they were written for.

This is a plain-language note, not legal advice.

| path | licence | SPDX tag |
|---|---|---|
| everything outside `patches/tabbyapi/` and `patches/exllamav3/` | MIT (`/LICENSE`) | `MIT` (every `.py`, `.sh` and `.service` file carries it; a test enforces that) |
| `patches/tabbyapi/` (the scripts, the `.patch` files and their `.license` sidecars) | AGPL-3.0-or-later (`patches/tabbyapi/LICENSE`) | `AGPL-3.0-or-later` |
| `patches/exllamav3/` | MIT, with the exllamav3 notice below | `MIT` |
| `patches/_patchkit.py`, `patches/patchfile.py`, `patches/verify.py`, `patches/PINS.toml` | MIT (BigBrain's own tooling: they contain no upstream text) | `MIT` |

## TabbyAPI: GNU Affero General Public License v3.0

`patches/tabbyapi/*` (`failed_load.py`, `close_stc.py`, `lora.py`, `auth_redact.patch`, `windows_launcher.patch`)
modify TabbyAPI (https://github.com/theroyallab/tabbyAPI, AGPL-3.0, pinned in `PINS.toml`). They are derivative of
it, so this subtree is offered under the **AGPL-3.0-or-later**; the full text is `patches/tabbyapi/LICENSE`
(TabbyAPI's own `LICENSE`, git blob `0ad25db4` at the pinned commit, with the line endings made LF).

What that means in practice:
- Running a patched TabbyAPI on your own LAN asks nothing of anyone.
- If a patched TabbyAPI is reachable over a network by people outside your own household or organisation, AGPL section 13 requires
  offering them the modified source. The source of the modification is this subtree plus the pinned upstream
  commit. Settle that before exposing a patched TabbyAPI beyond your LAN.
- The AGPL applies to this subtree only. BigBrain's engine neither imports nor calls TabbyAPI.

## exllamav3: MIT License

`patches/exllamav3/dflash2_tp.py` modifies exllamav3 (https://github.com/turboderp-org/exllamav3, pinned in
`PINS.toml`). The notice below must stay with it:

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

(The text is the `LICENSE` file shipped in the exllamav3 1.6.0 wheel's `dist-info/licenses/`.)

The BigBrain engine itself imports exllamav3 at run time (`Config`, `Model`, `Cache`, `Tokenizer`, and the internals
`exllamav3.modules.attn.prepare_for_attn` and `exllamav3.cache.recurrent_util`). It does not copy or patch any of it,
which is why the version is pinned to the 1.6 line (`pyproject.toml`, extra `gpu`).
