# SPDX-License-Identifier: MIT
# This file embeds excerpts of exllamav3 (Copyright (c) 2025 Turboderp, MIT License). The full notice is in
# patches/NOTICE.md and must stay with this file; the additions are offered under the same MIT terms.
"""dflash2_tp.py -- let exllamav3 1.6.0 / 1.6.1 run a DFlash2 draft against a TENSOR-PARALLEL target.

WHY. exllamav3 1.6.0 refuses it in DFlash2Model.attach_to(): "DFlash2 does not support tensor-parallel targets because
the selector needs top-k logits". MTP and DFlash v1 work under TP because they only need the target head's ARGMAX per
position, and exllamav3 already computes that across the vocabulary shards (Model.lm_head_argmax ->
tp_dispatch_lm_head_argmax: each GPU takes the max of its shard, the output device picks the winner). DFlash2's selector
needs the TOP-K (k = selector_top_k, 16 for the Qwen3.8-27B draft) per position instead. Top-k merges across shards
exactly the same way: the global top-k is always inside the union of each shard's local top-k. So this adds the top-k
twin of the argmax path and feeds the selector's existing walk kernel from it. Nothing else in DFlash2 touches the
target's modules directly: its input layer already embeds through the TP embedding dispatch (DFlashInputLayer, shared
with DFlash v1, which supports TP), and the target hidden states it conditions on already come back from forward_tp in
params["export_states"] (the output device's pseudo-worker runs in this process).

Output stays lossless whatever the draft proposes: the target verifies every drafted token.

Touches six files under the given site-packages/exllamav3 (a backup <file>.bb-orig-dflash2_tp is made first):
  model/model_tp_fn.py      + mp_model_forward_lm_head_topk (per-shard top-k, gathered to the output device)
  model/model_tp.py         + ExLlamaTP.tp_dispatch_lm_head_topk (merge; batch chunked to fit the 16 KB small buffer)
  model/model.py            + Model.lm_head_topk (both modes)
  modules/arch_specific/dflash2.py  + DFlash2Selector.walk_topk (walk_block's second half, callable on its own)
  architecture/dflash2.py   attach_to accepts a TP target; sample_from_state uses lm_head_topk under TP
  generator/generator.py    context-aware verify window: 4 drafted tokens past 100K of context (env-tunable)

    python patches/exllamav3/dflash2_tp.py --target <venv>/lib/python3.12/site-packages/exllamav3        # --check is the default
    ... --apply     patches the 6 files (refuses if any anchor is missing; all or nothing)
    ... --revert    puts the backups back (refuses if a file changed since the apply)
--target also accepts the venv root or its site-packages dir; the package is found under it.

Exit codes (--check): 0 applied, 1 not applied or partly, 2 anchors drifted. Built against exllamav3 1.6.0; the anchors
also match 1.6.1. exllamav3 is MIT, Copyright (c) 2025 Turboderp (patches/NOTICE.md).

VERSION GATE. exllamav3 1.6.2 HAS DFlash2 TENSOR PARALLEL BUILT IN, so this patch must never go on it. The version is read from
`<package>/version.py` (`__version__`): --apply refuses anything but 1.6.0 and 1.6.1; --check on 1.6.2 or later says "built in,
never patch" and exits 1 (exit 2 if our patch is on such a version anyway: revert it). An unreadable or older version is
"unsupported": --check reports the files as usual plus a note, --apply refuses. patches/PINS.toml repeats the two numbers and a
test keeps the copies equal.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # patches/ (the shared kit)
import _patchkit  # noqa: E402

MARK = "DFLASH2-TP-PATCH"

#: the only exllamav3 versions this patch may be applied to
ALLOWED_VERSIONS = ("1.6.0", "1.6.1")
#: from this version on exllamav3 carries DFlash2 tensor parallel itself (verified on 1.6.2): never patch
BUILT_IN_FROM = "1.6.2"


def read_version(package_dir):
    """exllamav3's `__version__` from <package_dir>/version.py (e.g. "1.6.1"), or None."""
    import re
    try:
        with open(os.path.join(package_dir, "version.py"), encoding="utf-8") as fh:
            m = re.search(r"""__version__\s*=\s*['"]([^'"]+)['"]""", fh.read())
    except OSError:
        return None
    return m.group(1) if m else None


def _triple(version):
    import re
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", version or "")
    return tuple(int(x) for x in m.groups()) if m else None


def version_gate(package_dir):
    """None when the patch may be applied here; else ("builtin" | "unsupported", message)."""
    version = read_version(package_dir)
    triple = _triple(version)
    if triple is None:
        return ("unsupported", f"cannot read a version from {os.path.join(package_dir, 'version.py')} "
                               f"(found {version!r}); --apply only goes on exllamav3 {' or '.join(ALLOWED_VERSIONS)}")
    if triple >= _triple(BUILT_IN_FROM):
        return ("builtin", f"built in, never patch: exllamav3 {version} has DFlash2 tensor parallel built in "
                           f"({BUILT_IN_FROM} and later)")
    import re
    m = re.fullmatch(r"(\d+\.\d+\.\d+)(\+[0-9A-Za-z.]+)?", version)       # "1.6.1" or "1.6.1+local"; not a dev/rc build of it
    if not m or m.group(1) not in ALLOWED_VERSIONS:
        return ("unsupported", f"exllamav3 {version} is outside the verified versions ({', '.join(ALLOWED_VERSIONS)}); "
                               f"--apply refuses")
    return None


def sub(text, old, new, path):
    if text.count(old) != 1:
        raise ValueError(f"{path}: anchor found {text.count(old)} times (want 1):\n{old[:200]}")
    return text.replace(old, new)


# ---------------------------------------------------------------- model/model_tp_fn.py
TP_FN_ANCHOR = "# def mp_model_forward_lm_head_argmax_old("
TP_FN_ADD = '''def mp_model_forward_lm_head_topk(
    local_context: dict,
    shared_input: dict,
    params: dict,
    offset: int,
    gather_devices: list[int] | None,
    ldims: list[int] | None,
    vocab_size: int,
    k: int,
):
    """
    DFLASH2-TP-PATCH (2026-10-08): the top-k twin of mp_model_forward_lm_head_argmax. Each device holding a
    non-empty LM-head slice returns its shard's k best (value, global index) per position, padding columns
    past the vocabulary cropped exactly as the argmax does; the slices are gathered to the output device,
    which keeps the k best of the union (ExLlamaTP.tp_dispatch_lm_head_topk). Values travel as float32 from
    EVERY rank, participating or not: pg_gather_small computes each rank's offsets from its own element size,
    so all ranks must agree. scale/softcap are monotonic and applied by the caller after the merge.
    """
    consumer = local_context["inf_consumer"]
    device = local_context["device"]
    output_device = local_context["output_device"]
    backend = local_context["backend"]

    x = consumer.recv(shared_input)

    if offset >= 0:
        module = local_context["logits_module"]
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
        if vocab_size >= 0 and offset + x.shape[-1] > vocab_size:
            x = x[..., :max(0, vocab_size - offset)]
        kk = min(k, x.shape[-1])
        v = torch.full((*x.shape[:-1], k), -float("inf"), dtype = torch.float, device = x.device)
        i = torch.zeros((*x.shape[:-1], k), dtype = torch.long, device = x.device)
        if kk > 0:
            tv, ti = torch.topk(x, kk, dim = -1, sorted = False)
            v[..., :kk] = tv.float()
            i[..., :kk] = ti + offset
    else:
        v = torch.empty((*x.shape[:-1], 0), dtype = torch.float, device = x.device)
        i = torch.empty((*x.shape[:-1], 0), dtype = torch.long, device = x.device)

    if gather_devices is None:
        return v, i

    v = v.contiguous()
    i = i.contiguous()
    if device == output_device:
        out_v = torch.empty((*v.shape[:-1], sum(ldims)), dtype = torch.float, device = v.device)
        out_i = torch.empty((*i.shape[:-1], sum(ldims)), dtype = torch.long, device = i.device)
    else:
        out_v = None
        out_i = None

    backend.gather_small(v, out_v, gather_devices, output_device, ldims)
    backend.gather_small(i, out_i, gather_devices, output_device, ldims)

    return (out_v, out_i) if device == output_device else None


'''

# ---------------------------------------------------------------- model/model_tp.py
TP_ANCHOR = "    # def tp_dispatch_lm_head_argmax_old(self, args):"
TP_ADD = '''    def tp_dispatch_lm_head_topk(self, state, k: int):
        """
        DFLASH2-TP-PATCH (2026-10-08): top-k over the tensor-parallel sharded LM head, the twin of
        tp_dispatch_lm_head_argmax (same dispatch order, same shard bookkeeping). Returns (values, ids), each
        (..., k) on the output device; values are the raw head logits (float), ids global vocabulary indices.
        The batch is split so one gather of the int64 ids fits the native backend's 16 KB small buffer.
        """
        ad = {}
        for device in self.active_devices:
            a, b, _ = self.plan[device]["lm_head"]
            if b > a:
                ad[device] = a
        vocab = self.config.vocab_size

        lead = state.shape[:-1]
        flat = state.reshape(-1, 1, state.shape[-1])
        n_slices = max(1, len(ad))
        per_row = k * n_slices * 8
        chunk = max(1, 12 * 1024 // per_row)
        out_v, out_i = [], []
        for c0 in range(0, flat.shape[0], chunk):
            part = flat[c0:c0 + chunk].contiguous()
            x = self.tp_producer.send(part)
            args = (x, {})
            if len(ad) == 1 and self.tp_output_device in ad:
                v, i = self.tp_worker_dispatch_single(
                    self.tp_output_device,
                    mp_model_forward_lm_head_topk,
                    args + (ad[self.tp_output_device], None, None, vocab, k)
                )
                out_v.append(v.float())
                out_i.append(i)
                continue

            gd = sorted(set(ad.keys()) | {self.tp_output_device})
            ldims = [k if d in ad else 0 for d in gd]
            dispatched = []
            for device in self.active_devices:
                if device in gd:
                    self.tp_worker_dispatch(
                        device,
                        mp_model_forward_lm_head_topk,
                        args + (ad.get(device, -1), gd, ldims, vocab, k)
                    )
                    dispatched.append(device)
            results = []
            for device in dispatched:
                r = self.tp_worker_result(device)
                if r is not None:
                    results.append((device, r))
            assert len(results) == 1 and results[0][0] == self.tp_output_device, "TP logic error"
            all_v, all_i = results[0][1]
            all_v = all_v.float()
            # The union of every shard's k best holds the global k best; keep those
            top_v, sel = torch.topk(all_v, k, dim = -1, sorted = False)
            out_v.append(top_v)
            out_i.append(all_i.gather(-1, sel))
        values = torch.cat(out_v, dim = 0).reshape(*lead, k)
        ids = torch.cat(out_i, dim = 0).reshape(*lead, k)
        return values, ids


'''

# ---------------------------------------------------------------- model/model.py
MODEL_ANCHOR = '''    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):'''
MODEL_ADD = '''    @torch.inference_mode
    def lm_head_topk(self, state: torch.Tensor, k: int, scale: float = 1.0, softcap: float = 0.0):
        """
        DFLASH2-TP-PATCH (2026-10-08): the k best tokens per position from a hidden state through this
        model's LM head, as (scores float [..., k], ids long [..., k]), unsorted, the vocabulary cropped to
        the unpadded size. Scores are logit * scale, softcapped when softcap > 0 (both monotonic, so applying
        them after the selection is exact). Layer-split and tensor-parallel; DFlash2's selector consumes it.
        """
        if self.loaded_tp:
            values, ids = self.tp_dispatch_lm_head_topk(state, k)
        else:
            lm = self.modules[self.logit_layer_idx]
            logits = lm.forward(lm.prepare_for_device(state, {}), {})
            logits = logits[..., :self.config.vocab_size].float()
            values, ids = torch.topk(logits, k, dim = -1, sorted = False)
        values = values.float() * scale
        if softcap > 0.0:
            values = torch.tanh(values / softcap) * softcap
        return values, ids


'''

# ---------------------------------------------------------------- modules/arch_specific/dflash2.py
SEL_OLD = '''        if cuda:
            out = torch.empty((bsz, rows + 1), dtype = torch.long, device = hidden.device)
            conf = torch.empty((bsz, rows + 1), dtype = torch.float, device = hidden.device) if return_confidence else None
            ext.dflash2_selector_walk(
                unary.float().contiguous(), cands.long().contiguous(), gate.contiguous(),
                self.pred_codebook, self.succ_codebook, anchor_ids.to(hidden.device, non_blocking = anchor_ids.is_pinned()).contiguous(), out, conf,
            )
            return out, conf
        return self._walk_torch(unary.float(), cands.long(), gate.float(), anchor_ids, return_confidence)
'''
SEL_NEW = '''        return self._walk_from_topk(gate, unary, cands, anchor_ids, return_confidence, cuda)


    def walk_topk(
        self,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        cands: torch.Tensor,
        anchor_ids: torch.Tensor,
        return_confidence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """DFLASH2-TP-PATCH (2026-10-08): walk_block() from an already selected top-k -- scores [b, rows, k]
        (scaled/softcapped logits) and candidate ids [b, rows, k] -- for a target whose LM head is sharded
        across GPUs (Model.lm_head_topk merges the shards' top-k exactly). Same gate, same walk kernel."""
        gate = self.hidden_proj.forward(hidden.half().contiguous(), params = {})
        anchor_ids = anchor_ids.long()
        cuda = ext.HAS_DFLASH2 and hidden.is_cuda and self.pred_codebook.dtype in (torch.half, torch.bfloat16)
        unary = unary.to(hidden.device)
        cands = cands.to(hidden.device)
        return self._walk_from_topk(gate, unary, cands, anchor_ids, return_confidence, cuda)


    def _walk_from_topk(self, gate, unary, cands, anchor_ids, return_confidence, cuda):
        bsz, rows = unary.shape[:2]
        if cuda:
            out = torch.empty((bsz, rows + 1), dtype = torch.long, device = gate.device)
            conf = torch.empty((bsz, rows + 1), dtype = torch.float, device = gate.device) if return_confidence else None
            ext.dflash2_selector_walk(
                unary.float().contiguous(), cands.long().contiguous(), gate.contiguous(),
                self.pred_codebook, self.succ_codebook, anchor_ids.to(gate.device, non_blocking = anchor_ids.is_pinned()).contiguous(), out, conf,
            )
            return out, conf
        return self._walk_torch(unary.float(), cands.long(), gate.float(), anchor_ids, return_confidence)
'''

# ---------------------------------------------------------------- architecture/dflash2.py
ARCH_ATTACH_OLD = '''        if target.loaded_tp:
            raise NotImplementedError(
                "DFlash2 does not support tensor-parallel targets because the selector needs top-k logits"
            )
'''
ARCH_ATTACH_NEW = '''        # DFLASH2-TP-PATCH (2026-10-08): a tensor-parallel target is served through Model.lm_head_topk
        # (the shards' top-k merged exactly); 1.6.0 refused it here ("the selector needs top-k logits")
        if target.loaded_tp and not hasattr(target, "lm_head_topk"):
            raise NotImplementedError(
                "DFlash2 does not support tensor-parallel targets because the selector needs top-k logits"
            )
'''
ARCH_SAMPLE_OLD = '''        target = self.attached_model()
        lm = target.modules[target.logit_layer_idx]
'''
ARCH_SAMPLE_NEW = '''        target = self.attached_model()
        if target.loaded_tp:
            # DFLASH2-TP-PATCH (2026-10-08): the head is sharded across the TP workers; take the merged
            # top-k for rows 1.. (row 0 is the anchor) and walk from it, exactly as walk_block would
            dev = self.selector.device
            anchor = get_for_device(params, "dflash2_anchor_ids", dev)[:, -1]
            export_conf = params.get("export_draft_conf", False)
            rows = state[:, 1:].half()
            unary, cands = target.lm_head_topk(
                rows.contiguous(), self.selector.top_k,
                scale = self.config.output_multiplier,
                softcap = self.config.final_logit_softcapping,
            )
            out, confidence = self.selector.walk_topk(
                to_device(rows, dev), unary, cands, anchor, return_confidence = export_conf,
            )
            if export_conf:
                params["draft_conf"] = confidence
            return out
        lm = target.modules[target.logit_layer_idx]
'''

# ---------------------------------------------------------------- generator/generator.py
GEN_OLD = """        # The diffusion drafter always runs at its fixed block size, dynamic window truncates the drafted block
        window = self.num_draft_tokens
"""
GEN_NEW = """        # The diffusion drafter always runs at its fixed block size, dynamic window truncates the drafted block
        window = self.num_draft_tokens
        # DFLASH2-TP-PATCH: context-aware verify window. Verifying a drafted position costs
        # attention over the whole cache, and acceptance falls at long context, so past a context length
        # the full block costs more than it returns. Measured on a pair of 24 GB GPUs (Qwen3.8-27B 5.0bpw,
        # tensor parallel): 7 tokens beat 4 at an 88K prompt (112/119 vs 107/110 tok/s write/edit) and
        # 4 beat 7 at 131K (92 vs 72). EXL3_DFLASH_LONGCTX_TOKENS (default 100000; 0 = off) and
        # EXL3_DFLASH_LONGCTX_WINDOW (default 4) set the rule. The drafter itself still runs its full block.
        import os as _os
        _lt = int(_os.environ.get("EXL3_DFLASH_LONGCTX_TOKENS", "100000") or 0)
        _lw = int(_os.environ.get("EXL3_DFLASH_LONGCTX_WINDOW", "4") or 0)
        if _lt > 0 and 0 < _lw < window:
            _longest = max((seq.kv_position for job in self.active_jobs if job.is_prefill_done()
                            for seq in job.sequences), default = 0)
            if _longest >= _lt:
                window = _lw
"""


#: file (under the exllamav3 package dir) -> [(anchor, replacement), ...]
SUBSTITUTIONS = {
    "model/model_tp_fn.py": [(TP_FN_ANCHOR, TP_FN_ADD + TP_FN_ANCHOR)],
    "model/model_tp.py": [(TP_ANCHOR, TP_ADD + TP_ANCHOR)],
    "model/model.py": [(MODEL_ANCHOR, MODEL_ADD + MODEL_ANCHOR)],
    "modules/arch_specific/dflash2.py": [(SEL_OLD, SEL_NEW)],
    "architecture/dflash2.py": [(ARCH_ATTACH_OLD, ARCH_ATTACH_NEW), (ARCH_SAMPLE_OLD, ARCH_SAMPLE_NEW)],
    "generator/generator.py": [(GEN_OLD, GEN_NEW)],
}


def _editor(rel):
    def apply(text):
        for old, new in SUBSTITUTIONS[rel]:
            text = sub(text, old, new, rel)
        return text
    return apply


def resolve_package(path):
    """--target may be the exllamav3 package dir, the site-packages dir, or the venv root."""
    path = os.path.abspath(path)
    probes = ["", "exllamav3"]
    import glob
    probes += [os.path.relpath(p, path) for pat in ("lib/python3*/site-packages/exllamav3", "Lib/site-packages/exllamav3")
               for p in glob.glob(os.path.join(path, *pat.split("/")))]
    for rel in probes:
        cand = os.path.join(path, rel) if rel else path
        if os.path.isfile(os.path.join(cand, "model", "model_tp.py")):
            return cand
    return path


PATCH = _patchkit.Patch(
    name="dflash2_tp",
    summary=__doc__.split("\n")[0],
    edits=[_patchkit.Edit(rel, MARK, _editor(rel)) for rel in SUBSTITUTIONS],
    resolve=resolve_package,
    gate=version_gate,
)


def main(argv=None) -> int:
    return _patchkit.main(PATCH, argv)


if __name__ == "__main__":
    sys.exit(main())
