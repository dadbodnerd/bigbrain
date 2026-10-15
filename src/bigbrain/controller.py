# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/controller.py -- BigBrain cluster controller v1.

Launches and monitors one worker process per machine for a pipeline-parallel
model. Line topology: node 0 (tokenizer) -> node 1 -> ... -> node T-1
(samples); the last node's token id returns to node 0 over the token port.
One GPU per node; fixed in-ports (non-ephemeral -- port forwarding is a v2
concern). Dry-run by default: it prints the exact remote command per node and
touches nothing.

The controller runs on any machine with ssh keys to the GPU machines (it needs no GPU itself) and reaches them with
`ssh -o BatchMode=yes <user>@<host> <cmd>`. The ssh user comes from the node's registry entry, else site.toml's [ssh] user
(bigbrain/site.py), else the local user. The runner is injectable so tests exercise the whole plan/launch/wait path
with a fake.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import registry as bbreg
from . import site
from .paged import PagedSlots

#: Where the engine is staged on each PC, and where the nodes run from (v0.2.1, security review M4). A private per-user directory:
#: `python -m` puts the working directory FIRST on sys.path, so whoever can write here can run code as the node. `~/` is expanded
#: by the node's shell. (Not $XDG_RUNTIME_DIR: logind deletes it when the user's last session ends, i.e. mid-run.)
DEFAULT_STAGE_ROOT = "~/.cache/bigbrain/stage"


def rpath(p: str) -> str:
    """A path for the REMOTE shell, quoted; a leading ~/ becomes "$HOME"/ so it still expands there."""
    if p == "~":
        return '"$HOME"'
    if p.startswith("~/"):
        return '"$HOME"/' + shlex.quote(p[2:])
    return shlex.quote(p)


#: the remote shell variable that holds the stage dir in every command the controller sends
STAGE_VAR = "bb_stage"


def stage_dir_guard(d: str) -> str:
    """Shell: set $bb_stage to `d`, then carry on only when it is a directory owned by this user, not a symlink and not writable
    by group or others; else print why on stderr and exit 1. Run before anything is staged into `d` or run from it (security
    review M4). POSIX `find -prune -perm` is the portable way to read the mode bits."""
    v = f'"${STAGE_VAR}"'
    return (f'{STAGE_VAR}={rpath(d)} && {{ [ -d {v} ] && [ ! -L {v} ] && [ -O {v} ] && '
            f'[ -z "$(find {v} -prune \\( -perm -020 -o -perm -002 \\) -print)" ]; }} '
            f'|| {{ echo "bigbrain: refusing the stage dir ${STAGE_VAR}: it must be a directory of the ssh user, not a symlink, '
            f'not writable by group or others (chmod 700)" >&2; exit 1; }}')


@dataclass
class NodeSpec:
    """One pipeline node: a slice of the model on one GPU of one machine."""
    rank: int
    host: str
    gpu: int
    slice_start: int
    slice_end: int
    in_port: int  # 0 = ephemeral (rejected for non-first nodes in v1)
    user: str = field(default_factory=site.ssh_user)
    # Per-node overrides for mixed clusters: the model dir and the venv python
    # can differ per host (e.g. /opt/venvs/a/bin/python on one PC and
    # /opt/venvs/b/bin/python on another, or the model living at a
    # different path on each box). Empty = use the plan-level value.
    model_dir: str = ""
    python_bin: str = ""

    def __post_init__(self):
        # The host goes into the remote shell command of a neighbouring node and into an ssh argv; the user into an ssh argv.
        # Both may come from a registry or a site file someone handed you, so they are checked here (security review M1).
        site.check_host(self.host, f"node {self.rank} host")
        site.check_user(self.user, f"node {self.rank} ssh user")


@dataclass
class ClusterPlan:
    """The whole cluster: model, nodes (in rank order), and generate params."""
    model_dir: str
    nodes: List[NodeSpec]
    prompt: str = "Hello"
    max_new: int = 32
    max_num_tokens: int = 8192
    token_port: int = 0
    total_blocks: int = 0
    stage_root: str = DEFAULT_STAGE_ROOT  # remote dir the worker package is staged into (private: see DEFAULT_STAGE_ROOT)
    python_bin: str = "python"  # the venv python (exllamav3 + torch) on the nodes
    mem_fraction: Optional[float] = None  # shared-GPU cap, passed to each worker
    execute: bool = False  # actually launch over ssh (else dry-run)
    # SERVE mode (a long-running session instead of a one-shot run): node 0 serves
    # an OpenAI-compatible endpoint on http_port behind a bearer token instead
    # of running one --prompt. Node 0 gets --http-port and --api-key-file api_key
    # (worker.main hands them to bigbrain/server.py); api_token is the key itself,
    # which the controller stages as that file and never puts on a command line.
    http_port: int = 0
    api_token: str = ""
    # SPEED (several sequences in flight + speculative decoding; engine.py / spec.py).
    # slots > 1 or spec != "none" switches every node to the batched cache layout and
    # node 0 to the batched engine (serving mode only: a one-shot --prompt run is one sequence).
    slots: int = 1
    spec: str = "none"            # none | ngram | model (a draft model on node 0)
    spec_k: int = 4
    spec_model: str = ""          # the draft model folder on node 0's host
    draft_device: str = ""        # e.g. cuda:1; default: node 0's own card
    inflight: int = 0             # decode steps in flight (0 = the number of nodes)
    mb_size: int = 0              # most sequences per decode step (0 = auto)
    max_draft: int = 0            # 0 = spec_k when speculating
    watchdog_s: float = 0.0       # seconds of link silence before a node declares its peer dead; 0 = the nodes' default (5)

    @property
    def batched(self) -> bool:
        return self.slots > 1 or self.spec != "none"

    def validate(self) -> List[str]:
        """Problems with the plan (empty = ok)."""
        problems: List[str] = []
        n = len(self.nodes)
        if n == 0:
            return ["no nodes defined"]

        # Ranks must be exactly 0..T-1 in list order.
        for i, node in enumerate(self.nodes):
            if node.rank != i:
                problems.append(f"rank {node.rank} at position {i}: ranks must be 0..{n - 1} in order")
                break

        # Host and user values (checked at construction too; this catches a value changed afterwards).
        for i, node in enumerate(self.nodes):
            if not site.valid_host(node.host):
                problems.append(f"rank {i}: host {node.host!r} is not an IP address or a host name")
            if not site.valid_user(node.user):
                problems.append(f"rank {i}: ssh user {node.user!r} is not a user name")

        # Slice coverage: contiguous, no gaps or overlaps, starting at 0.
        prev_end = 0
        for i, node in enumerate(self.nodes):
            if node.slice_start != prev_end:
                kind = "gap" if node.slice_start > prev_end else "overlap"
                problems.append(f"{kind} at rank {i}: slice starts at {node.slice_start}, "
                                f"expected {prev_end}")
            if node.slice_end <= node.slice_start:
                problems.append(f"empty slice at rank {i}: [{node.slice_start},{node.slice_end})")
            prev_end = node.slice_end

        if self.nodes[0].slice_start != 0:
            problems.append(f"node 0 must start at block 0, got {self.nodes[0].slice_start}")
        if self.total_blocks and self.nodes[-1].slice_end != self.total_blocks:
            problems.append(f"last node ends at block {self.nodes[-1].slice_end}, "
                            f"model has {self.total_blocks}")

        # In-ports: unique PER HOST (different machines may reuse a port number --
        # they are separate sockets), and non-first nodes need a fixed one (v1).
        seen: Dict[Tuple[str, int], int] = {}
        for i, node in enumerate(self.nodes):
            if i > 0 and node.in_port == 0:
                problems.append(f"rank {i} needs a fixed in_port (ephemeral is a v2 concern)")
            else:
                key = (node.host, node.in_port)
                if key in seen:
                    problems.append(f"duplicate in_port {node.in_port} on {node.host} (ranks {seen[key]} and {i})")
                else:
                    seen[key] = i

        # Token port: required for a pipeline, and must not collide with an
        # in_port on node 0's host (it is bound there; a collision surfaces
        # only as a 600 s connect-retry stall).
        if n > 1:
            if self.token_port == 0:
                problems.append("token_port must be set for a multi-node plan")
            elif (self.nodes[0].host, self.token_port) in seen:
                problems.append(f"token_port {self.token_port} collides with an in_port on node 0's host ({self.nodes[0].host})")
        if not (self.stage_root.startswith("/") or self.stage_root.startswith("~/")) or "\n" in self.stage_root:
            problems.append(f"stage_root {self.stage_root!r} must be an absolute path or start with ~/ (a private directory)")
        if self.slots < 1:
            problems.append(f"slots must be >= 1, got {self.slots}")
        if self.spec not in ("none", "ngram", "model"):
            problems.append(f"spec must be none, ngram or model, got {self.spec!r}")
        elif self.spec != "none" and not 1 <= self.spec_k <= 15:
            problems.append(f"spec_k must be 1..15, got {self.spec_k}")
        if self.spec == "model" and not self.spec_model:
            problems.append("spec 'model' needs spec_model (the draft model folder on node 0's host)")
        if self.watchdog_s and self.watchdog_s < 2.0:
            problems.append(f"watchdog_s must be 0 (the default) or at least 2, got {self.watchdog_s:g}: the 0.5 s heartbeat needs "
                            f"room to arrive")
        if self.batched and not self.http_port:
            problems.append("several slots / speculation need the serving engine: give --serve-port "
                            "(a one-shot --prompt run is a single sequence)")
        if self.http_port:
            if (self.nodes[0].host, self.http_port) in seen or self.http_port == self.token_port:
                problems.append(f"http_port {self.http_port} collides with a pipe port on node 0's host "
                                f"({self.nodes[0].host})")
            if not self.api_token:
                problems.append("a serving plan (http_port set) needs an api_token")
            elif len(self.api_token) < MIN_API_KEY_LENGTH:
                problems.append(f"the API key is {len(self.api_token)} characters; at least {MIN_API_KEY_LENGTH} are required "
                                f"(python -c \"import secrets; print(secrets.token_hex(24))\")")
        return problems

    def build_commands(self, python_bin: Optional[str] = None,
                   redirect_logs: bool = False) -> List[str]:
        """One remote shell command per node, in rank order: the exact
        `cd <stage> && python -m bigbrain.worker ...` form. With
        redirect_logs, each node's stdout/stderr goes to <stage>/node_<i>.log."""
        pybin = python_bin or self.python_bin
        cmds = []
        n = len(self.nodes)
        for i, node in enumerate(self.nodes):
            node_pybin = node.python_bin or pybin
            node_model = node.model_dir or self.model_dir
            # Every value below is either an int/float formatted here or shell-quoted: the remote shell interprets this string.
            parts = [
                f"--model-dir {shlex.quote(node_model)}",
                f"--device cuda:{int(node.gpu)}",
                f"--slice {int(node.slice_start)} {int(node.slice_end)}",
                f"--rank {int(node.rank)}",
                f"--total {n}",
            ]
            if not self.http_port:
                # one-shot generation (the CLI); a serving session takes its
                # prompts over HTTP instead
                parts += [f"--prompt {shlex.quote(self.prompt)}", f"--max-new {int(self.max_new)}"]
            parts.append(f"--max-num-tokens {int(self.max_num_tokens)}")
            if self.mem_fraction is not None:
                parts.append(f"--mem-fraction {float(self.mem_fraction):g}")
            if self.watchdog_s:
                parts.append(f"--watchdog-s {float(self.watchdog_s):g}")
            if i > 0:
                parts.append(f"--in-port {int(node.in_port)}")
            if i < n - 1:
                nxt = self.nodes[i + 1]
                parts += [f"--next-host {shlex.quote(nxt.host)}", f"--next-port {int(nxt.in_port)}"]
                if i == 0:
                    # node 0 binds the token port (pre-load) for the last node
                    # to connect back to.
                    parts.append(f"--token-port {int(self.token_port)}")
            elif n > 1:
                parts += [f"--token-host {shlex.quote(self.nodes[0].host)}",
                          f"--token-port {int(self.token_port)}"]
            if i == 0 and self.http_port:
                # The key itself never goes on a command line (ps, /proc/PID/cmdline, the printed plan): the controller stages it
                # as <stage>/api_key, 0600, and node 0 reads it from its working directory (security review M5).
                parts += [f"--http-port {int(self.http_port)}",
                          f"--api-key-file {API_KEY_FILE}"]
            if self.batched:
                # EVERY node: the same slot count (the cache layout) and speculation headroom
                parts.append(f"--slots {int(self.slots)}")
                md = self.max_draft or (self.spec_k if self.spec != "none" else 0)
                if md:
                    parts.append(f"--max-draft {int(md)}")
                if i == 0:
                    if self.spec != "none":
                        parts += [f"--spec-draft {shlex.quote(self.spec)}", f"--spec-k {int(self.spec_k)}"]
                    if self.spec == "model":
                        parts.append(f"--spec-model {shlex.quote(self.spec_model)}")
                        if self.draft_device:
                            parts.append(f"--draft-device {shlex.quote(self.draft_device)}")
                    if self.inflight:
                        parts.append(f"--inflight {int(self.inflight)}")
                    if self.mb_size:
                        parts.append(f"--mb-size {int(self.mb_size)}")
            # cd first: -m bigbrain.worker resolves against the cwd, which
            # is where stage_package() puts the package. Operator-supplied paths
            # are shell-quoted (the remote shell interprets this string).
            # CUDA_DEVICE_ORDER=PCI_BUS_ID: torch's default order is FASTEST_FIRST,
            # so on a mixed box (e.g. a 5070 + 3090 + 3060) cuda:N need not be
            # nvidia-smi's index N -- and the planner sized each slice from
            # nvidia-smi's reading of index N. Bus order makes the two agree.
            # umask 077 and the guard first: the node runs (and logs) only from a private stage dir (security review M4).
            cmd = (f'umask 077 && {stage_dir_guard(self.stage_root)} && cd "${STAGE_VAR}" && {DEVICE_ORDER_ENV} '
                   f"{shlex.quote(node_pybin)} -m bigbrain.worker {' '.join(parts)}")
            if redirect_logs:
                cmd += f' > "${STAGE_VAR}"/node_{i}.log 2>&1'
            cmds.append(cmd)
        return cmds


# The worker package staged onto every node, relative to the directory that holds the `bigbrain` package (src/ in a checkout,
# site-packages in a pip install). On the node it lands as <stage-root>/bigbrain/*.py, which is what `cd <stage-root> && python -m
# bigbrain.worker` imports.
PACKAGE_FILES = (
    "bigbrain/__init__.py",
    "bigbrain/pipe.py",
    "bigbrain/worker.py",
    # node 0 in serving mode runs as the OpenAI endpoint (worker.main hands `--http-port` over to it)
    "bigbrain/server.py",
    # imported by worker.py / server.py: the batched engine, speculation, paged slots, the draft model
    "bigbrain/paged.py",
    "bigbrain/spec.py",
    "bigbrain/engine.py",
    "bigbrain/draft.py",
)

#: Prefixed to every worker command (see build_commands).
DEVICE_ORDER_ENV = "CUDA_DEVICE_ORDER=PCI_BUS_ID"

#: The file node 0 reads its bearer key from, relative to the stage dir (its working directory). The controller stages it, 0600.
API_KEY_FILE = "api_key"

#: Below this length a key is refused (security review S3): `python -c "import secrets; print(secrets.token_hex(24))"`.
MIN_API_KEY_LENGTH = 16

#: The env var the controller (and node 0) read the key from when no file is given.
API_KEY_ENV = "BIGBRAIN_API_KEY"


def redact(text: str, secret: str) -> str:
    """`text` with every occurrence of `secret` replaced by ***: for anything the controller prints. (A secret of a few characters,
    which validate() refuses anyway, is left alone rather than starring out every such letter in the output.)"""
    return text.replace(secret, "***") if secret and len(secret) >= 6 else text


#: `pgrep -f` (ERE) pattern for a BigBrain node's command line, in BOTH launch forms: `python -m bigbrain.worker ...` (the
#: controller, deploy/run_node.sh) and `python -c "... from bigbrain import worker ..."` (a wrapper script that imports the
#: module; its command line never contains the string "bigbrain.worker").
#: `server` is matched too (node 0 started as the endpoint). Written with [.] so the pattern's own text does not match it: the
#: shell that runs pgrep carries this string in its command line. The package is a top-level name, so the leading group keeps
#: `bigbrain` from matching inside a longer name or path (`mybigbrain.worker`, `/opt/bigbrain.server/`). Nodes started through the
#: `bigbrain-node` / `bigbrain-server` console scripts are NOT matched: their command line has no module name (stop those with
#: SIGTERM to node 0, or through their own service manager). Valid in Python's `re` as well as POSIX ERE (no [[:classes:]]).
NODE_PROCESS_PATTERN = "(^|[^A-Za-z0-9_./@-])bigbrain([.]| import )(worker|server)"


def teardown_command(stage_root: str, sig: str = "TERM") -> str:
    """The remote shell command that signals every BigBrain node (worker or server, started with `-m bigbrain.X` or with
    `-c "from bigbrain import X ..."`) whose working directory is `stage_root` -- this session's nodes and nothing else
    (another session, or a one-shot CLI run, has its own stage dir). The shell running this carries the pattern in its own
    command line, but the pattern does not match its own text (see NODE_PROCESS_PATTERN) and its cwd is the login dir.
    Always exits 0."""
    if sig not in ("TERM", "KILL", "INT"):
        raise ValueError(f"signal {sig!r} is not one teardown sends")
    d = rpath(stage_root.rstrip("/"))
    # the physical path (what /proc/PID/cwd reads) even when $HOME or the stage root goes through a symlink
    return (f'd=$(cd {d} 2>/dev/null && pwd -P) || d={d}; for p in $(pgrep -f "{NODE_PROCESS_PATTERN}" 2>/dev/null); do '
            f'[ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$d" ] && kill -{sig} "$p" 2>/dev/null; '
            f'done; true')


def ssh_argv(user: str, host: str, cmd: str, *options: str) -> List[str]:
    """`ssh -o BatchMode=yes [options] -- user@host cmd`, after checking the user and the host (ValueError otherwise). The `--` ends
    ssh's own options, so nothing in the destination is ever read as one (security review M1)."""
    site.check_user(user)
    site.check_host(host)
    return ["ssh", "-o", "BatchMode=yes", *options, "--", f"{user}@{host}", cmd]


def _stage_root_of(remote_path: str) -> str:
    """The stage root a staged file belongs to: <root>/bigbrain/x.py -> <root>; <root>/api_key -> <root>."""
    parent = remote_path.rsplit("/", 1)[0]
    return parent.rsplit("/", 1)[0] if parent.endswith("/bigbrain") else parent


def _default_stager(host: str, user: str, remote_path: str, content: str) -> None:
    """Ship one file to a node over ssh (stdin pipe -- no scp dependency). The stage root is created private (umask 077) and
    checked (stage_dir_guard) before anything is written into it; the file is created afresh, so it is 0600 whatever was there."""
    d = remote_path.rsplit("/", 1)[0]
    root = _stage_root_of(remote_path)
    subprocess.run(
        ssh_argv(user, host, f"umask 077 && mkdir -p {rpath(root)} && {stage_dir_guard(root)} && mkdir -p {rpath(d)} && "
                             f"rm -f {rpath(remote_path)} && cat > {rpath(remote_path)}"),
        input=content.encode("utf-8"), check=True)


def stage_package(host: str, user: str, pkg_dir: str, remote_root: str,
                  stager: Optional[Callable] = None) -> List[str]:
    """Copy the worker package to one node. stager(host, user, path, content)
    is injectable for tests. Returns the remote paths staged."""
    stager = stager or _default_stager
    root = Path(pkg_dir).resolve().parent  # .../src/bigbrain (or site-packages/bigbrain) -> the dir that holds `bigbrain`
    staged = []
    for rel in PACKAGE_FILES:
        # every file is read verbatim; a missing one is an error, never an empty file shipped to a node
        content = (root / rel).read_text(encoding="utf-8")
        remote = f"{remote_root}/{rel}"
        stager(host, user, remote, content)
        staged.append(remote)
    return staged


def _ssh_start(host: str, user: str, cmd: str) -> subprocess.Popen:
    """Start one node over ssh, detached from the controller's lifetime."""
    return subprocess.Popen(
        ssh_argv(user, host, cmd),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


class Controller:
    """Launch + wait for the cluster.

    runner(host, user, cmd) -> handle starts ONE node NON-BLOCKING and returns
    a handle with .poll() -> Optional[int] (None = still running). The default
    runner is ssh; tests inject a fake.
    """

    def __init__(self, plan: ClusterPlan,
                 runner: Optional[Callable[[str, str, str], object]] = None,
                 dry_run: bool = True,
                 stager: Optional[Callable] = None,
                 stage: bool = True,
                 pkg_dir: Optional[str] = None):
        self.plan = plan
        self.dry_run = dry_run
        self.runner = runner or (self._print_runner if dry_run else self._ssh_runner)
        self.stager = stager
        self.stage = stage
        self.pkg_dir = str(pkg_dir or Path(__file__).resolve().parent)
        self.handles: Dict[int, object] = {}

    def _redact(self, text: str) -> str:
        return redact(text, self.plan.api_token)

    def _print_runner(self, host: str, user: str, cmd: str) -> object:
        print(self._redact(f"[dry-run] ssh {user}@{host} {cmd}"))
        return None

    def _ssh_runner(self, host: str, user: str, cmd: str) -> subprocess.Popen:
        return _ssh_start(host, user, cmd)

    def launch(self) -> Dict[int, object]:
        """Stage the worker package on every node (and, serving, the key file on node 0), then start every node in
        REVERSE rank order (listeners up before connectors).
        Returns {rank: handle}."""
        if self.stage:
            targets = list(dict.fromkeys((n.host, n.user) for n in self.plan.nodes))
            for host, user in targets:
                if self.dry_run:
                    print(self._redact(f"[dry-run] stage {len(PACKAGE_FILES)} files -> "
                                       f"{user}@{host}:{self.plan.stage_root}"))
                else:
                    stage_package(host, user, self.pkg_dir,
                                  self.plan.stage_root, stager=self.stager)
            if self.plan.http_port and self.plan.api_token:
                n0 = self.plan.nodes[0]
                remote = f"{self.plan.stage_root.rstrip('/')}/{API_KEY_FILE}"
                if self.dry_run:
                    print(f"[dry-run] stage {API_KEY_FILE} (the key, mode 0600) -> {n0.user}@{n0.host}:{remote}")
                else:
                    # over ssh stdin, never argv; created afresh under umask 077 (the default stager)
                    (self.stager or _default_stager)(n0.host, n0.user, remote, self.plan.api_token + "\n")
        cmds = self.plan.build_commands(redirect_logs=not self.dry_run)
        for node in sorted(self.plan.nodes, key=lambda x: x.rank, reverse=True):
            self.handles[node.rank] = self.runner(node.host, node.user, cmds[node.rank])
        return self.handles

    def wait(self) -> int:
        """Poll every handle until all are done; return the worst exit code."""
        if self.dry_run:
            return 0
        worst = 0
        pending = dict(self.handles)
        while pending:
            for rank, h in list(pending.items()):
                code = h.poll()
                if code is not None:
                    if code < 0:
                        # killed by a signal (e.g. OOM SIGKILL): -9 must NOT
                        # read as success (max(0, -9) == 0).
                        code = 128 + abs(code)
                    worst = max(worst, code)
                    del pending[rank]
            if pending:
                import time
                time.sleep(0.5)
        return worst


def _num_layers(cfg: dict) -> int:
    """num_hidden_layers, top-level or nested under text_config (multimodal
    models like the Qwen3.8-27B keep the LM config in text_config)."""
    if cfg.get("num_hidden_layers"):
        return int(cfg["num_hidden_layers"])
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and tc.get("num_hidden_layers"):
        return int(tc["num_hidden_layers"])
    raise KeyError("num_hidden_layers not found (top-level or text_config)")


def _read_model_config(model_dir: str, host: str, ssh_user: Optional[str] = None) -> dict:
    """The model's config.json: the local path first (a controller machine without
    GPUs usually has no models, so it is usually absent), then ssh to the host that
    holds the model (read-only `cat`). Raises if neither works."""
    local = Path(model_dir) / "config.json"
    if local.exists():
        return json.loads(local.read_text())
    ssh_user = ssh_user or site.ssh_user()
    out = subprocess.run(
        ssh_argv(ssh_user, host, f"cat {shlex.quote(model_dir.rstrip('/') + '/config.json')}", "-o", "ConnectTimeout=10"),
        capture_output=True, text=True, check=True, timeout=60).stdout
    return json.loads(out)


def _read_total_blocks(model_dir: str, host: str,
                       ssh_user: Optional[str] = None) -> int:
    """num_hidden_layers from the model's config.json (see _read_model_config)."""
    return _num_layers(_read_model_config(model_dir, host, ssh_user))


def plan_auto(model_dir: str, gpus: List[Tuple[str, int]], prompt: str = "Hello",
              max_new: int = 32, max_num_tokens: int = 8192, total_blocks: int = 0,
              port_base: int = 16900, stage_root: str = DEFAULT_STAGE_ROOT,
              python_bin: str = "python", mem_fraction: Optional[float] = None,
              execute: bool = False, ssh_user: Optional[str] = None,
              annotate: Optional[Callable[[List[NodeSpec]], None]] = None
              ) -> ClusterPlan:
    """Auto-plan: split the model's layers EVENLY across the chosen host:gpu
    pairs. The operator picks WHICH gpus (any mix of machines); the planner
    computes slice boundaries, in-ports and the token port. gpus[0] is node 0
    (tokenizer + Embedding); the last is the sampler. The model dir is assumed
    to exist on gpus[0]'s host (for the config.json read)."""
    n = len(gpus)
    if n == 0:
        raise ValueError("need at least one gpu")
    total = total_blocks or _read_total_blocks(model_dir, gpus[0][0], ssh_user)
    base, rem = divmod(total, n)
    nodes, prev, port = [], 0, port_base
    for i, (host, gpu) in enumerate(gpus):
        cnt = base + (1 if i < rem else 0)
        if cnt <= 0:
            raise ValueError(f"more gpus ({n}) than layers ({total})")
        in_port = 0 if i == 0 else port
        if i > 0:
            port += 1
        nodes.append(NodeSpec(rank=i, host=host, gpu=gpu, slice_start=prev,
                              slice_end=prev + cnt, in_port=in_port))
        prev += cnt
    if annotate is not None:
        annotate(nodes)
    return ClusterPlan(model_dir=model_dir, nodes=nodes, prompt=prompt,
                       max_new=max_new, max_num_tokens=max_num_tokens,
                       token_port=port, total_blocks=total, stage_root=stage_root,
                       python_bin=python_bin, mem_fraction=mem_fraction,
                       execute=execute)


# ═══════════════════════════════════════════════════════════════════════════
# The CAPACITY planner
#
# plan_auto splits the layers EVENLY by count. A real cluster mixes a 24 GiB 3090
# with a 12 GiB 3060 and cards that already carry a ComfyUI, and real models
# are not uniform: node 0 carries the embedding, the last node the lm_head, and
# a hybrid model (Qwen3.5/3.8's GatedDeltaNet) caches per token only on its
# attention layers. plan_capacity sizes every node from MEASURED numbers:
#
#   usable_i  = free_i - headroom                    (free_i: the card reading,
#                                                     plus anything the caller
#                                                     will free first)
#   fixed_i   = overhead + [i = 0] embed + [i = last] head
#   budget_i  = usable_i - fixed_i                   (room for blocks)
#   cost_l    = weights_l + kv_bytes_per_token_l x ctx + recurrent_l
#
# and cuts the contiguous block list so each node's share of sum(cost) is
# proportional to its budget (cut k_i nearest to T x sum(budget_<=i) / sum(budget)).
# When rounding to whole layers overflows a card, a min-max fallback finds the
# contiguous split that minimises the fullest card's need/budget. A plan is
# refused, with the numbers, when sum(cost) > sum(budget), when a node cannot
# hold even its fixed part, or when no contiguous split fits.
#
# Units: MiB throughout. The KV cache is FP16 by default (exllamav3's Cache()
# default layer, which worker.py uses): 2 bytes per element.
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_HEADROOM_MIB = 1024          # prefill scratch + allocator slack per GPU (load_admission's 1 GiB)
DEFAULT_NODE_OVERHEAD_MIB = 1024     # CUDA context + torch + exllamav3 runtime per worker process
DEFAULT_CACHE_BYTES = 2.0            # FP16 KV (exllamav3 Cache default)
DEFAULT_HEAD_BITS = 8                # an unmeasured lm_head is assumed 8-bit (exl3 heads are 6-8)
RECURRENT_MIB_PER_LAYER = 8.0        # GatedDeltaNet state + conv state per linear layer (batch 1), generous
DEFAULT_LAN_MB_S = 110               # 1 GbE
DEFAULT_LOOPBACK_MB_S = 2000         # same-host hop: the pipe is TCP over loopback, not NVLink
MIB = 1024.0 * 1024.0


class PlanRefused(ValueError):
    """The model cannot be planned at all (missing facts)."""


@dataclass
class ModelShape:
    name: str
    layer_mib: List[float]
    embed_mib: float
    head_mib: float
    kv_bytes_per_token: List[float]
    recurrent_mib: List[float]
    hybrid: bool = False
    measured: bool = False

    @property
    def num_layers(self) -> int:
        return len(self.layer_mib)

    recurrent_state: bool = False   # hybrid (GatedDeltaNet) OR sliding-window state (gpt-oss)
    linear: List[bool] = field(default_factory=list)   # GatedDeltaNet layers: their state keeps per-token history

    @staticmethod
    def cache_tokens(ctx: int, slots: int = 1, batched: bool = False) -> int:
        """Cache(max_num_tokens=...) for `slots` sequences: the batched layout gives each slot
        its context plus one slack page (paged.PagedSlots); the single-sequence layout is ctx."""
        return PagedSlots.tokens_for(slots, ctx) if (batched or slots > 1) else int(ctx)

    def cache_mib(self, layer: int, ctx: int, slots: int = 1, batched: bool = False,
                  history: int = 0) -> float:
        """KV per token x cache tokens, plus the recurrent state of EVERY slot. `history` =
        speculative draft tokens: a GatedDeltaNet layer then keeps that many extra recurrent states
        per slot (Cache(max_history=k)); a sliding-window state rewinds in place and needs none."""
        lin = layer < len(self.linear) and self.linear[layer]
        return (self.kv_bytes_per_token[layer] * self.cache_tokens(ctx, slots, batched) / MIB
                + self.recurrent_mib[layer] * max(1, int(slots)) * (1 + (int(history) if lin else 0)))

    def costs(self, ctx: int, slots: int = 1, batched: bool = False, history: int = 0) -> List[float]:
        return [self.layer_mib[i] + self.cache_mib(i, ctx, slots, batched, history)
                for i in range(self.num_layers)]


def _is_attention(t) -> bool:
    t = str(t or "").lower()
    return "linear" not in t and ("attention" in t or t in ("full", "attn", "sliding"))


def model_shape(name: str, info: dict, cache_bytes: float = DEFAULT_CACHE_BYTES,
                head_bits: Optional[float] = None) -> ModelShape:
    """The planner's view of one registry model entry. Measured per-layer MiB
    (build_registry) win; without them the weights are spread evenly over the
    blocks after an embedding (vocab x hidden x 2 B) and a head (vocab x hidden
    x head_bits/8) estimated from the config facts. Raises PlanRefused when the
    entry lacks the layer count or the KV facts."""
    info = info or {}
    layers = info.get("layers")
    if not isinstance(layers, int) or isinstance(layers, bool) or layers <= 0:
        raise PlanRefused(f"the registry has no `layers` (layer count) for {name}, and its config.json "
                          f"was not read")
    kv = info.get("kv_elements_per_token")
    if not isinstance(kv, (int, float)) or isinstance(kv, bool) or kv < 0:
        raise PlanRefused(f"the registry has no kv_elements_per_token for {name} -- run build_registry")
    types = info.get("layer_types")
    if not (isinstance(types, list) and len(types) == layers):
        types = ["full_attention"] * layers
    # exllamav3 runs gpt-oss's sliding-window layers as SlidingAttention: a small per-sequence
    # RECURRENT STATE (window + 2 pages of K/V, whatever the context), not paged KV. The
    # registry's kv_elements_per_token counts the FULL layers only (footprint.kv_shape),
    # so spread it over those and price the sliding layers as state.
    swa = bbreg.has_swa_state(info)
    sliding = [swa and "sliding" in str(t).lower() for t in types]
    attn = [_is_attention(t) and not sliding[i] for i, t in enumerate(types)]
    n_attn = sum(attn)
    per_layer_elems = float(kv) / n_attn if n_attn else 0.0
    kvb = [per_layer_elems * float(cache_bytes) if a else 0.0 for a in attn]
    rec_each = float(info.get("recurrent_mib_per_layer") or RECURRENT_MIB_PER_LAYER)
    linear = ["linear" in str(t).lower() for t in types]
    swa_mib = bbreg.swa_state_mib(info, cache_bytes)
    rec = [rec_each if lin else (swa_mib if sliding[i] else 0.0) for i, lin in enumerate(linear)]
    hybrid = bool(info.get("hybrid")) or any(linear)
    recurrent_state = hybrid or any(sliding)

    lm = info.get("layer_mib")
    if isinstance(lm, list) and len(lm) == layers and all(isinstance(x, (int, float)) for x in lm):
        return ModelShape(name, [float(x) for x in lm], float(info.get("embed_mib") or 0.0),
                          float(info.get("head_mib") or 0.0), kvb, rec, hybrid, measured=True,
                          recurrent_state=recurrent_state, linear=linear)
    size = info.get("size_gib")
    if not isinstance(size, (int, float)) or size <= 0:
        raise PlanRefused(f"the registry has neither per-layer sizes nor size_gib for {name}")
    size_mib = float(size) * 1024.0
    hidden, vocab = info.get("hidden_size"), info.get("vocab_size")
    embed = head = 0.0
    if isinstance(hidden, int) and isinstance(vocab, int) and hidden > 0 and vocab > 0:
        bits = float(head_bits or info.get("head_bits") or DEFAULT_HEAD_BITS)
        embed = vocab * hidden * 2.0 / MIB
        head = vocab * hidden * bits / 8.0 / MIB
        if embed + head >= 0.5 * size_mib:       # implausible for a real checkpoint: do not guess
            embed = head = 0.0
    per = (size_mib - embed - head) / layers
    return ModelShape(name, [per] * layers, embed, head, kvb, rec, hybrid, measured=False,
                      recurrent_state=recurrent_state, linear=linear)


def draft_footprint_mib(info: dict, ctx: int, slots: int = 1, cache_bytes: float = DEFAULT_CACHE_BYTES,
                        separate_device: bool = False) -> float:
    """VRAM the draft model needs: its weights, its OWN paged cache for `slots` sequences of `ctx`
    tokens (the draft keeps every slot's context, like a node does) and, on a card of its own, a CUDA
    context. From the registry entry of the draft model."""
    shape = model_shape("draft", info, cache_bytes=cache_bytes)
    weights = sum(shape.layer_mib) + shape.embed_mib + shape.head_mib
    cache = sum(shape.cache_mib(i, int(ctx), max(1, int(slots)), True) for i in range(shape.num_layers))
    return weights + cache + (512.0 if separate_device else 0.0)


def _device_index(dev: str) -> Optional[int]:
    """'cuda:1' / '1' -> 1 (the CUDA ordinal, which CUDA_DEVICE_ORDER=PCI_BUS_ID makes nvidia-smi's index)."""
    m = re.search(r"(\d+)\s*$", str(dev or ""))
    return int(m.group(1)) if m else None


@dataclass
class GpuCapacity:
    """One candidate GPU, as the planner sees it. free_mib is what the node may
    use BEFORE headroom: the card reading's free memory plus whatever the
    caller will free first (a server it unloads)."""
    host: str
    gpu: int
    total_mib: int
    free_mib: int
    host_name: str = ""
    card_key: str = ""
    name: str = ""
    nvlink_peers: Tuple[int, ...] = ()
    model_dir: str = ""
    python_bin: str = ""
    user: str = field(default_factory=site.ssh_user)
    card_keys: Tuple[str, ...] = ()
    free_now_mib: Optional[int] = None
    reclaimable_mib: int = 0

    @property
    def label(self) -> str:
        return f"{self.host_name or self.host}:{self.gpu}"


@dataclass
class CapacityPlan:
    ok: bool
    reason: Optional[str]
    model: str
    ctx: int
    nodes: List[dict]
    total_need_mib: int
    links: List[dict]
    split: Optional[str] = None
    cluster: Optional[ClusterPlan] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """The wire shape of a plan, for a front end that shows one. Never the
        cluster plan: its commands carry the endpoint's bearer token."""
        return {"ok": self.ok, "reason": self.reason, "model": self.model, "ctx": self.ctx,
                "nodes": [dict(n) for n in self.nodes], "total_need_mib": self.total_need_mib,
                "links": [dict(x) for x in self.links], "split": self.split,
                "warnings": list(self.warnings)}


def _ceil(x: float) -> int:
    return int(math.ceil(x - 1e-9))


def _link_rate(rates: Union[int, Dict[str, int], None], host: str, default: int) -> int:
    if isinstance(rates, dict):
        return int(rates.get(host) or default)
    if isinstance(rates, (int, float)) and rates > 0:
        return int(rates)
    return default


def _links(gpus: Sequence[GpuCapacity], link_mb_s, loopback_mb_s) -> List[dict]:
    out = []
    for i in range(len(gpus) - 1):
        a, b = gpus[i], gpus[i + 1]
        same = a.host == b.host
        if same:
            est = _link_rate(loopback_mb_s, a.host, DEFAULT_LOOPBACK_MB_S)
        else:
            est = min(_link_rate(link_mb_s, a.host, DEFAULT_LAN_MB_S),
                      _link_rate(link_mb_s, b.host, DEFAULT_LAN_MB_S))
        out.append({"from": i, "to": i + 1, "kind": "same-host" if same else "lan",
                    "nvlink": bool(same and b.gpu in tuple(a.nvlink_peers)), "est_mb_s": est})
    return out


def _proportional_cuts(prefix: List[float], budgets: List[float]) -> List[int]:
    """Cut points (block index after each of the first N-1 nodes) nearest to
    each node's proportional share of the total cost."""
    n, layers = len(budgets), len(prefix) - 1
    total, whole = prefix[-1], sum(budgets)
    cuts, prev, acc = [], 0, 0.0
    for i in range(n - 1):
        acc += budgets[i]
        target = total * acc / whole
        lo, hi = prev + 1, layers - (n - 1 - i)
        k = min(range(lo, hi + 1), key=lambda j: (abs(prefix[j] - target), j))
        cuts.append(k)
        prev = k
    return cuts


def _bottleneck_cuts(prefix: List[float], budgets: List[float]) -> Tuple[float, List[int]]:
    """(the fullest node's load/budget, cuts) for the contiguous split that
    minimises it. O(N x L^2); L <= ~130 blocks, N <= 8 GPUs."""
    n, layers = len(budgets), len(prefix) - 1
    inf = float("inf")
    f = [[inf] * (layers + 1) for _ in range(n)]
    arg = [[0] * (layers + 1) for _ in range(n)]
    for k in range(1, layers - n + 2):
        f[0][k] = prefix[k] / budgets[0]
    for i in range(1, n):
        for k in range(i + 1, layers - (n - 1 - i) + 1):
            best, bj = inf, i
            for j in range(i, k):
                if f[i - 1][j] == inf:
                    continue
                v = max(f[i - 1][j], (prefix[k] - prefix[j]) / budgets[i])
                if v < best:
                    best, bj = v, j
            f[i][k], arg[i][k] = best, bj
    cuts, k = [], layers
    for i in range(n - 1, 0, -1):
        k = arg[i][k]
        cuts.append(k)
    return f[n - 1][layers], list(reversed(cuts))


def plan_capacity(shape: ModelShape, gpus: Sequence[GpuCapacity], ctx: int, *,
                  headroom_mib: int = DEFAULT_HEADROOM_MIB,
                  node_overhead_mib: int = DEFAULT_NODE_OVERHEAD_MIB,
                  port_base: int = 16900, stage_root: str = DEFAULT_STAGE_ROOT,
                  http_port: int = 0, api_token: str = "",
                  link_mb_s: Union[int, Dict[str, int], None] = None,
                  loopback_mb_s: Union[int, Dict[str, int], None] = None,
                  python_bin: str = "python", prompt: str = "Hello", max_new: int = 32,
                  mem_fraction: Optional[float] = None, execute: bool = False,
                  slots: int = 1, spec: str = "none", spec_k: int = 4, spec_model: str = "",
                  draft_device: str = "", draft_mib: float = 0.0, inflight: int = 0,
                  mb_size: int = 0) -> CapacityPlan:
    """Plan `shape` across `gpus` IN THE ORDER GIVEN (gpus[0] is node 0) at a
    context of `ctx` tokens for `slots` sequences at once (the cache and the
    recurrent state are per slot; `draft_mib` is the draft model's footprint on
    node 0 when spec == "model"). Never raises for a plan that does not fit: the
    answer is ok=False with a reason naming the numbers, and every GPU's row
    still carries its free/usable memory so a panel can show why."""
    ctx = int(ctx)
    slots = max(1, int(slots))
    batched = slots > 1 or spec != "none"
    history = int(spec_k) if spec != "none" else 0
    # where the draft model lives: node 0's own card, or another card of the SAME machine that is
    # already one of this plan's cards (so the reservation holds, parks and unloads it too)
    draft_rank = None
    if spec == "model":
        draft_rank = 0
        if draft_device:
            idx = _device_index(draft_device)
            draft_rank = next((i for i, g in enumerate(gpus)
                               if gpus and g.host == gpus[0].host and g.gpu == idx), None)
            if draft_rank is None:
                return CapacityPlan(False, f"the draft device {draft_device} is not one of the plan's cards "
                                           f"on node 0's machine -- add that GPU to the plan so the "
                                           f"reservation holds it", shape.name, ctx, [], 0, [])
    n, layers = len(gpus), shape.num_layers
    head, ovh = int(headroom_mib), int(node_overhead_mib)
    rows = []
    for i, g in enumerate(gpus):
        fixed_parts = [("the runtime overhead", float(ovh))]
        if i == 0:
            fixed_parts.append(("the embedding", shape.embed_mib))
        if i == draft_rank and draft_mib:
            fixed_parts.append(("the draft model", float(draft_mib)))
        if i == n - 1:
            fixed_parts.append(("the lm_head", shape.head_mib))
        rows.append({
            "rank": i, "host": g.host, "host_name": g.host_name or g.host, "gpu": g.gpu,
            "card_key": g.card_key, "card_keys": list(g.card_keys or ([g.card_key] if g.card_key else [])),
            "name": g.name, "layers": None, "need_mib": None, "free_mib": int(g.free_mib),
            "free_now_mib": g.free_now_mib if g.free_now_mib is not None else int(g.free_mib),
            "reclaimable_mib": int(g.reclaimable_mib), "total_mib": int(g.total_mib),
            "usable_mib": int(g.free_mib) - head, "headroom_mib": head, "overhead_mib": ovh,
            "weights_mib": None, "cache_mib": None, "spare_mib": None,
            "model_dir": g.model_dir, "_fixed": fixed_parts,
        })
    costs = shape.costs(ctx, slots, batched, history)
    total_cost = sum(costs)
    fixed_total = sum(sum(v for _, v in r["_fixed"]) for r in rows)
    links = _links(gpus, link_mb_s, loopback_mb_s)

    def refuse(reason: str, split: Optional[str] = None) -> CapacityPlan:
        for r in rows:
            r.pop("_fixed", None)
        return CapacityPlan(False, reason, shape.name, ctx, rows, _ceil(total_cost + fixed_total),
                            links, split=split)

    if n == 0:
        return refuse("no GPUs given")
    seen = set()
    for g in gpus:
        if (g.host, g.gpu) in seen:
            return refuse(f"GPU {g.label} is listed twice")
        seen.add((g.host, g.gpu))
    for g in gpus:
        if not g.model_dir:
            return refuse(f"{shape.name} is not on {g.host_name or g.host} ({g.host}) -- GPU {g.label} "
                          f"cannot hold a slice of it")
    if n > layers:
        return refuse(f"more GPUs ({n}) than layers ({layers}) in {shape.name}")
    budgets = []
    for r in rows:
        fixed = sum(v for _, v in r["_fixed"])
        budget = r["usable_mib"] - fixed
        if budget <= 0:
            parts = ", ".join(f"{name} {v:,.0f} MiB" for name, v in r["_fixed"] if v)
            return refuse(f"node {r['rank']} ({r['host_name']}:{r['gpu']}) has {r['usable_mib']:,} MiB usable "
                          f"after {head:,} MiB headroom but needs {fixed:,.0f} MiB before any layer ({parts})")
        budgets.append(float(budget))
    total_budget = sum(budgets)
    cache_total = sum(shape.cache_mib(i, ctx, slots, batched, history) for i in range(layers))
    if total_cost > total_budget:
        return refuse(
            f"the chosen GPUs cannot hold {shape.name} at {ctx:,} tokens: its {layers} layers need "
            f"{total_cost:,.0f} MiB ({total_cost - cache_total:,.0f} weights + {cache_total:,.0f} cache) and the "
            f"GPUs have {total_budget:,.0f} MiB left after headroom, runtime overhead, embedding and head "
            f"(short {total_cost - total_budget:,.0f} MiB)")

    prefix = [0.0]
    for c in costs:
        prefix.append(prefix[-1] + c)
    cuts = _proportional_cuts(prefix, budgets)
    split = "proportional"
    bounds = [0] + cuts + [layers]
    if any(prefix[bounds[i + 1]] - prefix[bounds[i]] > budgets[i] + 1e-6 for i in range(n)):
        worst, cuts = _bottleneck_cuts(prefix, budgets)
        split = "bottleneck"
        if worst > 1.0 + 1e-9:
            return refuse(f"no contiguous split of {shape.name}'s {layers} layers fits these GPUs at {ctx:,} "
                          f"tokens: the tightest card would need {worst:.0%} of its room", split=split)
        bounds = [0] + cuts + [layers]

    nodes_spec, out_rows = [], []
    for i, (g, r) in enumerate(zip(gpus, rows)):
        s, e = bounds[i], bounds[i + 1]
        w = sum(shape.layer_mib[s:e]) + sum(v for name, v in r["_fixed"] if name != "the runtime overhead")
        c = sum(shape.cache_mib(j, ctx, slots, batched, history) for j in range(s, e))
        need = w + c + ovh
        r.pop("_fixed", None)
        r.update({"layers": [s, e], "weights_mib": _ceil(w), "cache_mib": _ceil(c),
                  "need_mib": _ceil(need), "spare_mib": int(r["usable_mib"] - _ceil(need))})
        out_rows.append(r)
        nodes_spec.append(NodeSpec(rank=i, host=g.host, gpu=g.gpu, slice_start=s, slice_end=e,
                                   in_port=0 if i == 0 else port_base + i - 1, user=g.user or site.ssh_user(),
                                   model_dir=g.model_dir, python_bin=g.python_bin))
    cluster = ClusterPlan(model_dir=gpus[0].model_dir, nodes=nodes_spec, prompt=prompt, max_new=max_new,
                          max_num_tokens=ctx, token_port=(port_base + n - 1) if n > 1 else 0,
                          total_blocks=layers, stage_root=stage_root, python_bin=python_bin,
                          mem_fraction=mem_fraction, execute=execute, http_port=int(http_port or 0),
                          api_token=api_token or "", slots=slots, spec=spec, spec_k=int(spec_k),
                          spec_model=spec_model, draft_device=draft_device, inflight=int(inflight),
                          mb_size=int(mb_size))
    warnings = []
    hosts_in_order = [g.host for g in gpus]
    hops = sum(1 for a, b in zip(hosts_in_order, hosts_in_order[1:]) if a != b)
    if hops > len(set(hosts_in_order)) - 1:
        warnings.append(f"the GPU order crosses the LAN {hops} times; grouping each PC's GPUs together "
                        f"would cross it {len(set(hosts_in_order)) - 1} time(s)")
    if not shape.measured:
        warnings.append(f"{shape.name}'s per-layer sizes are estimated (uniform blocks); "
                        f"build_registry measures them")
    if spec == "model" and not draft_mib:
        warnings.append("the draft model's footprint was not given (draft_mib): the card that loads it "
                        "may not have the room")
    problems = cluster.validate()
    if problems:
        return CapacityPlan(False, "; ".join(problems), shape.name, ctx, out_rows,
                            sum(r["need_mib"] for r in out_rows), links, split=split, warnings=warnings)
    return CapacityPlan(True, None, shape.name, ctx, out_rows, sum(r["need_mib"] for r in out_rows),
                        links, split=split, cluster=cluster, warnings=warnings)


def _order_for_pipeline(subset: Sequence[GpuCapacity], host_rank: Dict[str, int]) -> List[GpuCapacity]:
    """Group each PC's GPUs together (one LAN hop per PC boundary), PCs in
    host_rank order, and inside a PC keep NVLink peers adjacent."""
    by_host: Dict[str, List[GpuCapacity]] = {}
    for g in subset:
        by_host.setdefault(g.host, []).append(g)
    out = []
    for host in sorted(by_host, key=lambda h: (host_rank.get(h, 99), h)):
        rest = sorted(by_host[host], key=lambda g: g.gpu)
        chain = [rest.pop(0)]
        while rest:
            peer = next((g for g in rest if g.gpu in tuple(chain[-1].nvlink_peers)), rest[0])
            rest.remove(peer)
            chain.append(peer)
        out += chain
    return out


def choose_gpus(shape: ModelShape, candidates: Sequence[GpuCapacity], ctx: int, **kw) -> CapacityPlan:
    """gpus="auto": the subset of `candidates` that holds the model at `ctx`,
    preferring (in order) the fewest machines, the fewest GPUs, the most
    same-machine NVLink hops, then the most spare room on the fullest card."""
    cands = [g for g in candidates if g.model_dir]
    if not cands:
        return CapacityPlan(False, f"no eligible GPU holds {shape.name} (none of the PCs that have it has a "
                                   f"free, unheld, launchable card)", shape.name, int(ctx), [], 0, [])
    head = int(kw.get("headroom_mib", DEFAULT_HEADROOM_MIB))
    ovh = int(kw.get("node_overhead_mib", DEFAULT_NODE_OVERHEAD_MIB))
    kslots = max(1, int(kw.get("slots", 1)))
    kspec = kw.get("spec", "none")
    need = sum(shape.costs(int(ctx), kslots, kslots > 1 or kspec != "none",
                           int(kw.get("spec_k", 4)) if kspec != "none" else 0)) \
        + shape.embed_mib + shape.head_mib + (float(kw.get("draft_mib") or 0) if kspec == "model" else 0.0)
    totals: Dict[str, int] = {}
    for g in cands:
        totals[g.host] = totals.get(g.host, 0) + g.free_mib
    host_rank = {h: i for i, h in enumerate(sorted(totals, key=lambda h: (-totals[h], h)))}
    subsets = []
    for r in range(1, len(cands) + 1):
        for combo in itertools.combinations(cands, r):
            room = sum(g.free_mib - head - ovh for g in combo)
            if room < need:
                continue
            subsets.append((len({g.host for g in combo}), len(combo), combo))
    subsets.sort(key=lambda t: (t[0], t[1]))
    best, best_key, tier = None, None, None
    for n_hosts, n_gpus, combo in subsets:
        if tier is not None and (n_hosts, n_gpus) != tier:
            break
        ordered = _order_for_pipeline(combo, host_rank)
        plan = plan_capacity(shape, ordered, ctx, **kw)
        if not plan.ok:
            continue
        tier = (n_hosts, n_gpus)
        nv = sum(1 for link in plan.links if link["nvlink"])
        min_spare = min(nd["spare_mib"] / max(1, nd["usable_mib"]) for nd in plan.nodes)
        key = (-nv, -min_spare, [(g.host, g.gpu) for g in ordered])
        if best_key is None or key < best_key:
            best, best_key = plan, key
    if best is not None:
        return best
    usable = sum(g.free_mib - head - ovh for g in cands)
    if usable < need:
        reason = (f"no combination of the {len(cands)} eligible GPU(s) holds {shape.name} at {int(ctx):,} tokens: "
                  f"it needs {need:,.0f} MiB plus {ovh:,} MiB runtime per GPU, and all of them together have "
                  f"{usable:,.0f} MiB usable after headroom and runtime (short {need - usable:,.0f} MiB)")
    else:
        reason = (f"the eligible GPUs have room for {shape.name} at {int(ctx):,} tokens in total, but no "
                  f"contiguous split of its layers fits them")
    return CapacityPlan(False, reason, shape.name, int(ctx), [], _ceil(need), [])


#: What the built-in reader runs on each PC (read-only).
FREE_MEM_QUERY = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"


def _ssh_capture(user: str, host: str, cmd: str, timeout: int = 30) -> str:
    return subprocess.run(ssh_argv(user, host, cmd, "-o", "ConnectTimeout=10"),
                          capture_output=True, text=True, check=True, timeout=timeout).stdout


def nvidia_smi_free_mib(reg: dict, cards: Sequence[Tuple[str, int]],
                        run: Optional[Callable[[str, str, str], str]] = None) -> Dict[Tuple[str, int], int]:
    """{(host, gpu): MiB free right now}: the built-in reader behind `--split capacity`. One read-only `nvidia-smi` per PC over
    ssh (`run(user, address, command)` is injectable for tests). The keys are the (host, gpu) pairs exactly as `cards` gave them;
    a card the PC did not report is simply absent, and the caller says so. Whatever already sits on a card (another server, a
    desktop) is not free, which keeps the plan conservative."""
    run = run or _ssh_capture
    by_host: Dict[str, List[int]] = {}
    for host, gpu in cards:
        by_host.setdefault(host, []).append(int(gpu))
    out: Dict[Tuple[str, int], int] = {}
    for host, gpus in by_host.items():
        meta = bbreg.node_meta(reg, host)
        try:
            text = run(meta.get("user") or site.ssh_user(), meta.get("host") or host, FREE_MEM_QUERY)
        except (subprocess.SubprocessError, OSError) as exc:
            raise PlanRefused(f"could not read the free GPU memory of {host} over ssh ({type(exc).__name__}: {exc}); fix the "
                              f"ssh access, or use --split even, which needs no reading") from None
        free: Dict[int, int] = {}
        for line in text.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                try:
                    free[int(parts[0])] = int(float(parts[1]))
                except ValueError:
                    continue
        for g in gpus:
            if g in free:
                out[(host, g)] = free[g]
    return out


def _cli_free_mib(reg: dict, cards: Sequence[Tuple[str, int]]) -> Dict[Tuple[str, int], int]:
    """{(host, gpu): MiB a worker may use} for the CLI's capacity split. By default the built-in nvidia-smi reader
    (nvidia_smi_free_mib). site.toml's [capacity] reader = "module:function" swaps in your own `function(reg, cards)`, for a
    cluster where something else knows better what a card may give up (a scheduler that will unload a server first). When that
    reader cannot be imported the answer is a PlanRefused that names the way out, not a bare ImportError."""
    spec = site.capacity_reader()
    if not spec:
        return nvidia_smi_free_mib(reg, cards)
    mod_name, _, fn_name = spec.partition(":")
    try:
        import importlib  # noqa: PLC0415
        fn = getattr(importlib.import_module(mod_name), fn_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise PlanRefused(
            f"site.toml [capacity] reader = {spec!r} cannot be loaded ({type(exc).__name__}: {exc}). Put its module on "
            f"PYTHONPATH, remove the setting to use the built-in nvidia-smi reader, or use --split even, which needs no "
            f"reading") from None
    return fn(reg, cards)


def _add_speed_args(p: argparse.ArgumentParser) -> None:
    """Serving mode + the speed features (read by apply_speed_args on the finished plan)."""
    p.add_argument("--serve-port", type=int, default=0,
                   help="SERVING mode: node 0 runs the OpenAI-compatible server on this port "
                        "(needs a key: --api-key-file or $BIGBRAIN_API_KEY) instead of generating one --prompt")
    p.add_argument("--api-key-file", default=None, metavar="PATH",
                   help="a file holding the bearer key node 0's server requires (keep it mode 0600). The controller stages it "
                        "to node 0 as <stage-root>/api_key (0600); it is never put on a command line. Default: "
                        "$BIGBRAIN_API_KEY")
    p.add_argument("--api-token", default="",
                   help="DEPRECATED: the key itself, which puts it on this command line (visible in ps). Use --api-key-file")
    p.add_argument("--slots", type=int, default=1,
                   help="sequences in flight at once (every node gets the same; serving mode)")
    p.add_argument("--spec-draft", "--spec", dest="spec_draft", choices=("none", "ngram", "model"),
                   default="none", help="speculative decoding for greedy requests")
    p.add_argument("--spec-k", type=int, default=4, help="most draft tokens per round (1..15)")
    p.add_argument("--spec-model", default="", help="the draft model folder on node 0's host")
    p.add_argument("--draft-device", default="", help="CUDA device for the draft model (default: node 0's)")
    p.add_argument("--draft-mib", type=float, default=0.0,
                   help="the draft model's VRAM on node 0 (capacity split reserves it)")
    p.add_argument("--inflight", type=int, default=0, help="decode steps in flight (0 = number of nodes)")
    p.add_argument("--mb-size", type=int, default=0, help="most sequences per decode step (0 = auto)")
    p.add_argument("--watchdog-s", type=float, default=0.0,
                   help="seconds of silence on a link before a node declares its peer dead, passed to EVERY node "
                        "(0 = the nodes' default, 5; at least 2). Raise it (30) for a dense model with a slow prefill "
                        "or a hop through an ssh tunnel")


def _controller_key(p: argparse.ArgumentParser, a) -> str:
    """The bearer key for a serving plan: --api-key-file, else the deprecated --api-token, else $BIGBRAIN_API_KEY; '' when there
    is none (validate() then says a serving plan needs one). Read here, on the controller; staged to node 0 as a 0600 file."""
    if a.api_key_file:
        try:
            with open(a.api_key_file, encoding="utf-8") as f:
                key = f.read().strip()
        except OSError as e:
            p.error(f"cannot read --api-key-file {a.api_key_file}: {e.strerror or e}")
        if not key:
            p.error(f"--api-key-file {a.api_key_file} is empty")
        return key
    if a.api_token:
        print("warning: --api-token is deprecated: it puts the key on the controller's command line, where any local user can "
              "read it with ps. Use --api-key-file PATH (or $BIGBRAIN_API_KEY). The key is still staged to node 0 as a 0600 "
              "file, never passed on node 0's command line.", file=sys.stderr)
        return a.api_token.strip()
    return os.environ.get(API_KEY_ENV, "").strip()


def apply_speed_args(plan: ClusterPlan, a) -> ClusterPlan:
    """Copy the serving + speed flags onto a plan the explicit / simple modes built."""
    if a.serve_port:
        plan.http_port, plan.api_token = int(a.serve_port), a.api_token
    plan.slots, plan.spec, plan.spec_k = int(a.slots), a.spec_draft, int(a.spec_k)
    plan.spec_model, plan.draft_device = a.spec_model, a.draft_device
    plan.inflight, plan.mb_size = int(a.inflight), int(a.mb_size)
    plan.watchdog_s = float(a.watchdog_s)
    return plan


def plan_from_args(argv: Optional[List[str]] = None) -> ClusterPlan:
    plan, a = _plan_from_args(argv)
    return apply_speed_args(plan, a)


def _plan_from_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="BigBrain cluster controller v1")
    p.add_argument("--model-dir", default=None,
                   help="explicit model path (node 0's host). Usually you give "
                        "--model NAME instead and the registry resolves each "
                        "node's path.")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="canonical model name from model_registry.json; the "
                        "registry resolves each node's local path + venv python")
    p.add_argument("--hosts", default=None, metavar="LIST",
                   help="comma list of PC names/IPs to draw cards from "
                        "(e.g. 'alpha,beta'); default = every PC that may be launched on")
    p.add_argument("--total-blocks", type=int, default=0,
                   help="override; default = read num_hidden_layers from the model")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--max-num-tokens", type=int, default=8192)
    p.add_argument("--token-port", type=int, default=0,
                   help="explicit --node mode only; auto mode assigns it")
    p.add_argument("--stage-root", default=DEFAULT_STAGE_ROOT,
                   help="the directory on each PC the engine is staged into and the nodes run from: an absolute path or ~/..., "
                        "private to the ssh user (a directory others can write is refused). Default %(default)s")
    p.add_argument("--python-bin", default="python",
                   help="the venv python (exllamav3 + torch) on the nodes")
    p.add_argument("--mem-fraction", type=float, default=None,
                   help="cap each worker's CUDA memory (shared GPUs)")
    p.add_argument("--execute", action="store_true",
                   help="actually stage + launch over ssh (default is dry-run)")
    p.add_argument("--gpus", action="append", dest="gpus",
                   metavar="HOST:GPU",
                   help="SIMPLE mode: which gpus to use, in rank order; the "
                        "layers are split evenly and ports auto-assigned")
    p.add_argument("--node", action="append", dest="nodes",
                   metavar="HOST:GPU:START:END:INPORT",
                   help="EXPLICIT mode: one node; repeat per node, in rank order")
    p.add_argument("--node-model", action="append", dest="node_models",
                   metavar="DIR",
                   help="per-node model-dir override, in rank order (mixed "
                        "clusters where the model lives at different paths)")
    p.add_argument("--node-python", action="append", dest="node_pythons",
                   metavar="BIN",
                   help="per-node venv-python override, in rank order (mixed "
                        "clusters where the venv differs per host)")
    p.add_argument("--all-cards", action="store_true",
                   help="use EVERY GPU on every selected PC (registry order); "
                        "pairs with --model NAME")
    p.add_argument("--split", choices=("auto", "even", "capacity"), default="auto",
                   help="how --gpus/--all-cards cut the layers: capacity = in proportion to each "
                        "GPU's usable memory (live reading, needs --model); even = by layer count; "
                        "auto (default) = capacity with --model, even with --model-dir")
    p.add_argument("--ctx", type=int, default=None,
                   help="context the plan is sized for (default: --max-num-tokens)")
    p.add_argument("--headroom-mib", type=int, default=DEFAULT_HEADROOM_MIB)
    p.add_argument("--node-overhead-mib", type=int, default=DEFAULT_NODE_OVERHEAD_MIB)
    p.add_argument("--cache-bytes", type=float, default=DEFAULT_CACHE_BYTES,
                   help="KV bytes per element (2 = the worker's FP16 cache)")
    _add_speed_args(p)
    a = p.parse_args(argv)
    a.api_token = _controller_key(p, a)
    # Explicit --node mode may supply per-node --node-model overrides instead of
    # a plan-level model (then the plan's model_dir is display-only).
    if not a.model and not a.model_dir \
            and not (a.nodes and a.node_models):
        p.error("give --model NAME (resolved via the registry) or --model-dir PATH")
    if a.gpus and a.nodes:
        p.error("use --gpus (simple) OR --node (explicit), not both")
    if a.all_cards and (a.gpus or a.nodes):
        p.error("--all-cards is mutually exclusive with --gpus/--node")

    # --model NAME: resolve per-node model paths + venv pythons from the
    # registry. Requires the registry and that the model is present on every
    # selected PC (a pipeline node must hold its own slice).
    reg = None
    if a.model:
        reg = bbreg.load()
        if a.model not in reg.get("models", {}):
            p.error(f"model {a.model!r} not in {bbreg.REGISTRY_PATH.name}; "
                    f"known: {', '.join(sorted(reg.get('models', {})))}")

    def _annotate(nodes: List[NodeSpec]) -> None:
        """Fill per-node model_dir/python_bin from the registry (or CLI
        overrides), leaving explicit --node-model/--node-python to win."""
        for i, nd in enumerate(nodes):
            if reg:
                host_model = bbreg.model_path(reg, a.model, nd.host)
                if host_model is None:
                    raise SystemExit(
                        f"model {a.model} is not on {nd.host} "
                        f"(present on: {', '.join(bbreg.model_hosts(reg, a.model))})")
                nd.model_dir = host_model
                meta = bbreg.node_meta(reg, nd.host)
                nd.python_bin = meta.get("python") or ""
                if meta.get("user"):         # the node's own ssh user (the capacity split always used it; v0.2.1: every mode)
                    nd.user = site.check_user(meta["user"], f"node {nd.rank} ssh user (registry)")
            if a.node_models and i < len(a.node_models):
                nd.model_dir = a.node_models[i]
            if a.node_pythons and i < len(a.node_pythons):
                nd.python_bin = a.node_pythons[i]

    split = a.split
    if split == "auto":
        split = "capacity" if reg else "even"
    if split == "capacity" and not reg:
        p.error("--split capacity needs --model NAME (the registry gives the model's shape)")
    ctx = int(a.ctx or a.max_num_tokens)

    def _capacity(cards: List[Tuple[str, int]]) -> ClusterPlan:
        """The capacity split for the CLI: the registry's model shape (its
        config.json read over ssh when the registry lacks the layer count) and
        a live free-memory reading of every card."""
        info = bbreg.model_info(reg, a.model) or {}
        if not info.get("layers"):
            host0 = bbreg.host_of(reg, cards[0][0]) or cards[0][0]
            path0 = bbreg.model_path(reg, a.model, host0) or ""
            cfg = _read_model_config(path0, host0, bbreg.node_meta(reg, host0).get("user") or site.ssh_user())
            info = bbreg.apply_config_facts(info, cfg)
        try:
            shape = model_shape(a.model, info, cache_bytes=a.cache_bytes)
        except PlanRefused as e:
            p.error(str(e))
        try:
            free = _cli_free_mib(reg, cards)
        except PlanRefused as e:
            p.error(str(e))
        caps = []
        for host, gpu in cards:
            meta = bbreg.node_meta(reg, host)
            g = bbreg.find_gpu(reg, host, gpu) or {}
            if (host, gpu) not in free:
                p.error(f"no free-memory reading for {host}:{gpu} (use --split even to plan blind)")
            addr = meta.get("host") or host
            caps.append(GpuCapacity(host=addr, gpu=int(gpu), total_mib=int(g.get("total_mib") or 0),
                                    free_mib=int(free[(host, gpu)]), host_name=g.get("host_name") or host,
                                    card_key=g.get("card_key") or "", name=g.get("name") or "",
                                    nvlink_peers=tuple(g.get("nvlink_peers") or ()),
                                    model_dir=bbreg.model_path(reg, a.model, host) or "",
                                    python_bin=meta.get("python") or "", user=meta.get("user") or site.ssh_user(),
                                    card_keys=tuple(g.get("card_keys") or ())))
        cp = plan_capacity(shape, caps, ctx, headroom_mib=a.headroom_mib,
                           node_overhead_mib=a.node_overhead_mib, stage_root=a.stage_root,
                           python_bin=a.python_bin, prompt=a.prompt, max_new=a.max_new,
                           mem_fraction=a.mem_fraction, execute=a.execute,
                           http_port=int(a.serve_port or 0), api_token=a.api_token,
                           slots=a.slots, spec=a.spec_draft, spec_k=a.spec_k, spec_model=a.spec_model,
                           draft_device=a.draft_device, draft_mib=a.draft_mib,
                           inflight=a.inflight, mb_size=a.mb_size,
                           link_mb_s={(n.get("host") or k): n.get("link_mb_s") or DEFAULT_LAN_MB_S
                                      for k, n in bbreg.nodes(reg).items()})
        for nd in cp.nodes:
            print(f"  node {nd['rank']} {nd['host_name']}:{nd['gpu']} layers {nd['layers']} "
                  f"need {nd['need_mib']} MiB of {nd['usable_mib']} usable")
        if not cp.ok:
            p.error(cp.reason)
        _annotate(cp.cluster.nodes)
        return cp.cluster, a

    if a.all_cards:
        if not reg:
            p.error("--all-cards needs --model (the registry lists the GPUs)")
        hosts = bbreg.resolve_hosts(reg, a.hosts)
        have = set(bbreg.model_hosts(reg, a.model))
        hosts = [h for h in hosts if h in have]
        cards = bbreg.all_cards(reg, hosts)
        if not cards:
            p.error(f"no usable cards: model {a.model} is not on any selected "
                    f"PC (it is on: {', '.join(sorted(have)) or 'none'})")
        if split == "capacity":
            return _capacity(cards)
        model_dir = bbreg.model_path(reg, a.model, cards[0][0])
        return plan_auto(model_dir, cards, prompt=a.prompt, max_new=a.max_new,
                         max_num_tokens=a.max_num_tokens,
                         total_blocks=a.total_blocks, stage_root=a.stage_root,
                         python_bin=a.python_bin, mem_fraction=a.mem_fraction,
                         execute=a.execute, annotate=_annotate), a
    if a.gpus:
        gpus = []
        for spec in a.gpus:
            host, gpu = spec.split(":")
            gpus.append((host, int(gpu)))
        if split == "capacity":
            return _capacity(gpus)
        model_dir = (bbreg.model_path(reg, a.model, gpus[0][0])
                     if a.model else a.model_dir)
        return plan_auto(model_dir, gpus, prompt=a.prompt, max_new=a.max_new,
                         max_num_tokens=a.max_num_tokens,
                         total_blocks=a.total_blocks, stage_root=a.stage_root,
                         python_bin=a.python_bin, mem_fraction=a.mem_fraction,
                         execute=a.execute, annotate=_annotate), a
    if not a.nodes:
        p.error("give --model NAME [--all-cards] OR --gpus HOST:GPU (repeat) "
                "OR --node HOST:GPU:START:END:INPORT")
    nodes = []
    for i, spec in enumerate(a.nodes):
        host, gpu, start, end, inport = spec.split(":")
        nodes.append(NodeSpec(rank=i, host=host, gpu=int(gpu),
                              slice_start=int(start), slice_end=int(end),
                              in_port=int(inport)))
    _annotate(nodes)
    return ClusterPlan(model_dir=a.model_dir, nodes=nodes, prompt=a.prompt,
                       max_new=a.max_new, max_num_tokens=a.max_num_tokens,
                       token_port=a.token_port, total_blocks=a.total_blocks,
                       stage_root=a.stage_root, python_bin=a.python_bin,
                       mem_fraction=a.mem_fraction, execute=a.execute), a


def main(argv: Optional[List[str]] = None) -> int:
    try:
        plan = plan_from_args(argv)
    except ValueError as e:              # a host or user name that is not one (NodeSpec), a malformed --node / --gpus
        print(f"bigbrain-controller: error: {e}", file=sys.stderr)
        return 2
    problems = plan.validate()
    print(f"model: {plan.model_dir}  blocks: {plan.total_blocks}  nodes: {len(plan.nodes)}")
    for i, cmd in enumerate(plan.build_commands()):
        print(redact(f"  node {i}: {cmd}", plan.api_token))
    if problems:
        print("PROBLEMS:")
        for pr in problems:
            print(redact(f"  - {pr}", plan.api_token))
        return 2
    if plan.execute:
        print("EXECUTING: staging + launching the cluster over ssh")
        ctl = Controller(plan, dry_run=False)
        ctl.launch()
        rc = ctl.wait()
        print(f"cluster finished, worst exit code = {rc}")
        return rc
    Controller(plan).launch()
    return 0


if __name__ == "__main__":
    sys.exit(main())