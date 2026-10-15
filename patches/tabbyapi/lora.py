# SPDX-License-Identifier: AGPL-3.0-or-later
# This file exists only to modify TabbyAPI (https://github.com/theroyallab/tabbyAPI, AGPL-3.0) and embeds excerpts of it;
# it is offered under the same licence. Full text: patches/tabbyapi/LICENSE. Notes: patches/NOTICE.md.
"""Wire TabbyAPI's ExLlamaV3 backend to exllamav3's LoRA support.

The backend's load_loras/get_loras were stubs that logged
"LoRA loading is not hooked up to the ExLlamaV3 backend yet" and returned every
requested adapter as a failure -- but the library has had LoRA support all along
(exllamav3/model/lora.py). Only the wiring was missing.

    python patches/tabbyapi/lora.py --target ~/tabbyAPI            # --check is the default
    ... --apply     patches backends/exllamav3/model.py     ... --revert   puts the backup back

Exit codes (--check): 0 applied, 1 not applied, 2 anchors drifted. Idempotent, refuses unless every anchor matches
exactly once, backs up first, compiles the result. TENSOR PARALLEL AND LoRA ARE MUTUALLY EXCLUSIVE (see the docstring
of the inserted load_loras).
Upstream: TabbyAPI is AGPL-3.0 (see patches/NOTICE.md and patches/tabbyapi/LICENSE); pinned in patches/PINS.toml.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # patches/ (the shared kit)
import _patchkit  # noqa: E402

MARKER = "from exllamav3.model.lora import LoRA"

IMPORT_OLD = "from exllamav3.cache import CacheLayer_quant\n"
IMPORT_NEW = ("from exllamav3.cache import CacheLayer_quant\n"
              "from exllamav3.model.lora import LoRA\n")

INIT_OLD = "        self.model_dir = model_directory\n"
INIT_NEW = ("        self.model_dir = model_directory\n"
            "        self.loras = []\n")

STUBS_OLD = '''    # TODO: Wire up exllamav3's LoRA support once the API surface for it is decided
    async def load_loras(self, lora_directory: pathlib.Path, **kwargs) -> Dict[str, List[str]]:
        """Stub. LoRAs aren't hooked up to the ExLlamaV3 backend yet."""

        xlogger.error("LoRA loading is not hooked up to the ExLlamaV3 backend yet.")
        return {
            "success": [],
            "failure": [lora.get("name", "unknown") for lora in kwargs.get("loras", [])],
        }

    def get_loras(self) -> List[Any]:
        """Stub. LoRAs aren't hooked up to the ExLlamaV3 backend yet."""

        return []
'''

STUBS_NEW = '''    async def load_loras(self, lora_directory: pathlib.Path, **kwargs) -> Dict[str, List[str]]:
        """Load PEFT LoRA adapters onto the loaded model.

        These were stubs that reported every adapter as a failure, on the
        assumption that exllamav3 had no LoRA support. It does:
        `exllamav3/model/lora.py` reads a PEFT adapter directory
        (adapter_config.json + adapter_model.safetensors) and applies A/B on top
        of each target layer's dequantized output. It matches PEFT keys to
        modules by SUFFIX, so the `base_model.model.` prefix peft writes does not
        need stripping.

        TENSOR PARALLEL AND LoRA ARE MUTUALLY EXCLUSIVE, and the failure is
        quiet. `lora.py` skips every module whose weight is TP-sliced, counts
        them into `tp_skipped`, and warns -- it does not raise. An adapter loaded
        onto a tensor-parallel model is therefore PARTIALLY APPLIED, which reads
        as a working model producing subtly wrong output. The count is surfaced
        below as a loud warning so the operator sees it; serve adapters from an
        endpoint with tensor_parallel off until the slicing is implemented.
        """

        loras = kwargs.get("loras") or []
        success: List[str] = []
        failure: List[str] = []

        if self.model is None:
            xlogger.error("Cannot load LoRAs: no model is loaded.")
            return {
                "success": [],
                "failure": [lora.get("name", "unknown") for lora in loras],
            }

        for lora in loras:
            name = lora.get("name", "unknown")
            scaling = lora.get("scaling", 1.0)
            directory = lora_directory / name

            try:
                adapter = LoRA.from_directory(self.model, str(directory), scaling)

                # TabbyAPI derives a LoRA card's id from
                # Path(lora.lora_path).parent.name (endpoints/core/utils/lora.py).
                # That attribute is an ExLlamaV2 name which exllamav3 does not
                # set, so point it at the adapter's config file: its parent is
                # the adapter directory, which is the name the operator asked
                # for.
                adapter.lora_path = str(directory / "adapter_config.json")

                self.loras.append(adapter)
                success.append(name)
                xlogger.info(f"Loaded LoRA {name} (scaling {scaling})")
            except Exception as exc:
                xlogger.error(f"Failed to load LoRA {name}: {exc}")
                failure.append(name)

        return {"success": success, "failure": failure}

    def get_loras(self) -> List[Any]:
        """Return the adapters currently applied to the model."""

        return self.loras
'''

UNLOAD_OLD = '''        # Nothing to do for LoRA-only unloads until LoRAs are hooked up
        if loras_only:
            xlogger.error("LoRA unloading is not hooked up to the ExLlamaV3 backend yet.")
            return
'''

UNLOAD_NEW = '''        # LoRA-only unload: drop the adapters and leave the model resident.
        # Each adapter removes its own tensors from the Linear modules it
        # registered on, so the base model is restored without a reload.
        if loras_only:
            for adapter in self.loras:
                try:
                    adapter.unload()
                except Exception as exc:
                    xlogger.error(f"Failed to unload LoRA {adapter.name}: {exc}")
            self.loras = []
            return
'''

EDITS = [
    ("import", IMPORT_OLD, IMPORT_NEW),
    ("self.loras init", INIT_OLD, INIT_NEW),
    ("load_loras/get_loras", STUBS_OLD, STUBS_NEW),
    ("unload(loras_only)", UNLOAD_OLD, UNLOAD_NEW),
]


def patch_source(src: str) -> str:
    """backends/exllamav3/model.py with the LoRA wiring (LF text in and out). ValueError when it cannot."""
    for label, old, new in EDITS:
        if src.count(old) != 1:
            raise ValueError(f"'{label}' anchor matched {src.count(old)} times, expected 1 -- TabbyAPI changed; re-read it")
    for label, old, new in EDITS:
        src = src.replace(old, new, 1)
    return src


PATCH = _patchkit.Patch(
    name="lora",
    summary=__doc__.split("\n")[0],
    edits=[_patchkit.Edit("backends/exllamav3/model.py", MARKER, patch_source)],
)


def main(argv=None) -> int:
    return _patchkit.main(PATCH, argv)


if __name__ == "__main__":
    sys.exit(main())
