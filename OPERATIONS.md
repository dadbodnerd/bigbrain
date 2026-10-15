# Operating BigBrain

Day-to-day use: tests, starting and stopping, health and metrics, releases, supervised nodes, patches, and what to do when something breaks.
First-time setup is in INSTALL.md; every setting is in CONFIGURATION.md.

## Rules of thumb

- **One engine version per pipeline.** Every node must run the same `bigbrain/` files. The controller stages them from the package it
  runs from (the installed one, or a checkout's `src/bigbrain`), so launch from one version (`pip install ...@v0.2.1`, or
  `git checkout v0.2.1`) and check the nodes afterwards (below).
- **The key is a file, never a command-line argument.** Give the controller `--api-key-file api_key` (or `$BIGBRAIN_API_KEY`); it stages
  the key as `<stage-root>/api_key` (mode 0600) on node 0 and prints it nowhere, dry runs included. `--api-token` still works but is
  deprecated: it puts the key on the controller's own command line.
- **The stage directory is private.** The default is `~/.cache/bigbrain/stage` on each PC (mode 0700). A stage directory that is a symlink,
  belongs to someone else or is writable by group or others is refused, by the controller and by `deploy/run_node.sh`.
- **The package is `bigbrain`, one top-level name** (a src layout since v0.2.0; before that it was `tools.bigbrain`, which collided with
  any other project's `tools`). The controller stages `bigbrain/*.py` into each node's stage directory and starts `python -m
  bigbrain.worker` from there. `-m` puts the working directory first on `sys.path`, so the staged copy is the one that runs even when
  that node's venv also has `bigbrain` pip-installed.
- **Engine changes are tests first**, then the fix. The suite runs without a GPU.

## Run the tests

```bash
pip install -e ".[dev]"                  # once, in a checkout: the package (editable) plus pytest, build and twine
python -m pytest -q                       # everything
python -m pytest tests/patches -q         # the patch kit: scripts and the verifier
python -m pytest tests/deploy -q          # release and stage tooling (uses this repository's .git)
```

The tests import `bigbrain` from the environment (an editable install, or a wheel: run them from any directory with
`python -m pytest /path/to/checkout/tests` to test what pip installed), and fall back to the checkout's `src/` when it is not installed.
Needs Python 3.11+, numpy, jinja2 and pytest. At v0.2.1, on Windows, **with torch installed** the suite is 634 passed, 7 skipped from a
checkout (3 of the skips are POSIX permission tests that run on Linux); **against the installed wheel without torch**, run from a `git
archive` of the release (no `.git`), it is 517 passed, 124 skipped. The torch cases skip with the reason
`could not import 'torch'`. Of the 7 skips that remain with torch installed, 3 are the POSIX permission tests and 4 are the cases
that need a real, patched TabbyAPI checkout and its venv: `tests/patches/test_failed_load.py`, skipped with
`SKIPPED (needs a patched TabbyAPI checkout): set TABBY_TEST_DIR to run`. `tests/deploy` needs a git checkout (it hashes `git show` output),
and skips itself in a release directory that has no `.git`.

## Start

The normal path is the controller (INSTALL.md section 4): it stages the eight engine files to each PC and starts the nodes over ssh in
reverse rank order. It stays in the foreground while the cluster runs, so use `tmux` or `screen`. Node logs are
`<stage-root>/node_<rank>.log` on each PC.

Loading takes minutes and nodes wait for each other: a node retries connecting to its peer for up to 10 minutes while that peer is still loading.
`GET /health` on node 0 answers 503 until the nodes have shaken hands.

### Start a node by hand, or under systemd

For a supervised node, `deploy/run_node.sh <stage-dir>` starts one node from a stage directory, and `deploy/bigbrain-node@.service` is a systemd
**user** unit around it. You prepare the stage directory yourself; the controller does not write these files:

1. Stage the engine files into it (`stage_package(host, user, os.path.dirname(bigbrain.__file__), "~/.cache/bigbrain/stage")` from
   `bigbrain.controller` copies them over ssh into a private directory, or copy the files listed in `controller.PACKAGE_FILES`, all under
   `bigbrain/`, yourself and `chmod 700` the directory).
   If `BB_PYTHON`'s venv has `bigbrain` pip-installed (`pip install "git+https://github.com/dadbodnerd/bigbrain.git"`) there is nothing
   to stage: `python -m bigbrain.worker` finds the installed package. Then the stage dir holds only the files in steps 2-4.
2. `node.args`: the node's own command line on one line. Take it from the controller's dry-run output, the part after
   `-m bigbrain.worker` (node 0's already says `--api-key-file api_key`). It is split on blanks and never glob-expanded; a path with a
   space in it is not supported here.
3. `api_key`: a file with the key, mode 0600, in the stage directory (node 0 only).
4. `node.env`: at least `BB_PYTHON=/path/to/venv/bin/python`; anything else the node needs, exported (`export CUDA_VISIBLE_DEVICES=GPU-<uuid>`
   pins the node to one card by UUID; then give it `--device cuda:0`). **It is executed** (sourced as shell), so `run_node.sh` refuses a
   `node.env` that is a symlink, belongs to someone else or is writable by group or others, and does the same for the stage directory
   (exit 77).

Then `deploy/run_node.sh ~/.cache/bigbrain/stage`, or with the unit installed in `~/.config/systemd/user/`:

```bash
systemctl --user daemon-reload
systemctl --user start  "bigbrain-node@$(systemd-escape --path ~/.cache/bigbrain/stage).service"
journalctl --user -u    "bigbrain-node@$(systemd-escape --path ~/.cache/bigbrain/stage).service" -f
systemctl --user stop   "bigbrain-node@$(systemd-escape --path ~/.cache/bigbrain/stage).service"     # TERM, then KILL after 20 s; frees the VRAM
```

The unit runs with `NoNewPrivileges=yes`, `UMask=0077`, `RestrictSUIDSGID=yes` and `LockPersonality=yes`. A stronger sandbox (a private
`/tmp`, the system and your home read-only except the stage and cache directories) is the drop-in
`deploy/bigbrain-node@.service.d/sandbox.conf`. A user unit can apply it only where unprivileged user namespaces work (Ubuntu 24.04 and
later restrict them by default, and the node then fails with `status=226/NAMESPACE`), and inside it a model folder you can read only through
a group becomes unreadable: the file explains both and gives a one-line check before you copy it to
`~/.config/systemd/user/bigbrain-node@.service.d/`.

The script always starts the `worker` module (node 0 too: a command line containing `--http-port` is handed to `server.main`), with the stage directory as
its working directory, which is how `teardown_command` finds it. The unit runs `~/bigbrain/releases/current/deploy/run_node.sh`, where
`deploy/release.py` puts the code, and **has no `[Install]` section and `Restart=no` on purpose**: a node holds GPU memory, and one that
restarted by itself after a crash or reboot would take it back unannounced. Add them yourself if you want the opposite. Running after logout needs
`loginctl enable-linger <user>` (a one-time root action). This path is exercised by unit tests (`tests/deploy`) but not end to end in this repository.

## Health, stats, metrics

```bash
curl -s http://10.0.0.11:16980/health                                              # open; 200 ok / 503 degraded
curl -s -H "Authorization: Bearer $(cat api_key)" "http://10.0.0.11:16980/v1/bigbrain/stats?recent=20"
curl -s -H "Authorization: Bearer $(cat api_key)" http://10.0.0.11:16980/v1/bigbrain/metrics
```

`/health` has the queue depth, running requests, link state and, in `engine`, `ready`, `active`, `queued` and `dead` (the reason, when the
pipeline is gone). `/v1/bigbrain/stats` is cheap and good for a status page: decode and prefill tok/s, the aggregate over all slots, counts
and the last N requests. `/v1/bigbrain/metrics` asks every node for its numbers (busy / waiting-for-the-previous-node / sending
percentages, VRAM, speculation acceptance), so do not poll it quickly. All fields are in `docs/API.md`. A client can tag its own requests with the
`X-BigBrain-Job` header and the tag appears in the stats.

To measure a configuration, `bigbrain-bench grid ...` runs the concurrency x speculation grid and checks every output
against the plain run; `bench compare before.json after.json` compares two grids, for example before and after an exllamav3 upgrade.

## Stop

SIGTERM to node 0 ends the pipeline: it sends the terminator down the chain, the other nodes exit, and every process frees its VRAM. To stop
every BigBrain node of one stage directory on a PC:

```bash
python -c "from bigbrain.controller import teardown_command; print(teardown_command('~/.cache/bigbrain/stage'))"
ssh bigbrain@10.0.0.11 '<the printed line>'          # TERM; teardown_command(dir, "KILL") for a stuck node. Always exits 0.
nvidia-smi --query-gpu=index,memory.used --format=csv   # on the PC: confirm the VRAM came back
```

It matches only BigBrain processes (`python -m bigbrain.worker|server`, or a `-c "from bigbrain import worker ..."` wrapper)
whose working directory is that stage directory, so two clusters on one PC with different stage directories do not touch each other.
A node you started through the `bigbrain-node` / `bigbrain-server` command has no module name in its command line and is not found by
it: stop that one with SIGTERM to node 0.

## Releases and verification

The controller stages from the package it runs from (installed, or a checkout's `src/`), which is enough for most people. `deploy/` is not part of the pip package: run it from a checkout. `deploy/release.py` additionally makes an immutable
release directory on a host, for the systemd route above or to keep a known-good copy:

```bash
python deploy/release.py --commit v0.2.1 --host alpha --host beta --dry-run     # host names come from site.toml [hosts], or user@address
python deploy/release.py --commit v0.2.1 --host alpha --host beta               # ~/bigbrain/releases/<12 hex>/ , BUILD = the full hash, read-only
python deploy/release.py --commit v0.2.1 --to-dir /tmp/releases                 # a local directory instead of ssh
```

A release is `git archive` of a commit, extracted into `<dest>/<12 hex>/`, made read-only, with a `BUILD` file and a `current` symlink pointing at it.
It is built as `.<short>.partial` and renamed into place last, so a half-copied release is never mistaken for a finished one. A second run of the
same commit says `EXISTS` and touches nothing. It holds code only: never the registry, keys or `node.env`.

**Roll back** by repointing the link on the host and restarting the nodes: `ln -sfn <previous 12 hex> ~/bigbrain/releases/current`. To delete a
release you no longer need: `chmod -R u+w <dir> && rm -rf <dir>`.

**Prove what a host holds** (read-only: it hashes files and prints a verdict per file):

```bash
python deploy/verify_stage.py --commit v0.2.1 --host alpha --stage '~/.cache/bigbrain/stage'   # a node's stage dir (quoted: ~ is that PC's home)
python deploy/verify_stage.py --commit v0.2.1 --host alpha --release '~/bigbrain/releases/<12 hex>'
python deploy/verify_stage.py --commit v0.2.1 --stage ./stage_dir                         # a local directory (no --host)
```

Per file it says `IDENTICAL`, `IDENTICAL-LF` (same after CRLF -> LF: a stage that was copied from a Windows checkout; `--strict` refuses these),
`DIFFERENT` or `MISSING`. Exit 0 means every file matches. On Windows with Git Bash set `MSYS_NO_PATHCONV=1` (or use PowerShell), or Git Bash rewrites
`/tmp/...` into a Windows path before Python sees it.

## The optional patches

`patches/` holds scripts that change upstream TabbyAPI and exllamav3 (reliability fixes, a LoRA wire-up, DFlash2 under tensor parallel). BigBrain does not need
any of them. `patches/verify.py --hosts patches/hosts.toml --all` reports, per host, whether each tree and venv matches what you said it should carry
(the template is `patches/hosts.example.toml`). It never changes anything. Read `patches/README.md` before applying one: they edit upstream source.

## Troubleshooting

| you see | it means | do |
|---|---|---|
| `BigBrainPeerDead: no frame for 5.0s; declaring peer dead`, then `/health` 503 with `engine.dead` | a link went silent for the watchdog window: a node crashed (check its log and `nvidia-smi`), a network hiccup, or a node busy in one long compute step (a slow prefill on a dense model) | read `node_<rank>.log` on each PC for the first error. If every node is alive, relaunch with `--watchdog-s 30` on the controller command (it forwards the value to **every** node; a hand-started node needs it in its own `node.args`) |
| the controller prints `PlanRefused`-style text such as `no combination of the N eligible GPU(s) holds <model> at <ctx> tokens: it needs ... MiB ... (short ... MiB)` | the cards do not have the room at that context and slot count | lower `--slots` or `--max-num-tokens`, add a card, free memory on the PCs, or use a smaller quant |
| `no free-memory reading for HOST:GPU (use --split even to plan blind)` or `could not read the free GPU memory of HOST over ssh` | the capacity split could not run `nvidia-smi` over ssh (or your `[capacity] reader` skipped that card) | fix the ssh access, check `nvidia-smi` works for that user, or use `--split even` |
| `model X is not on HOST (present on: ...)` | the registry has no path for that model on that PC | copy the model there and re-run `build_registry`, or drop that PC from `--hosts` |
| `several slots / speculation need the serving engine: give --serve-port` | `--slots > 1` or `--spec-draft` without an endpoint | add `--serve-port N --api-key-file api_key` |
| `a serving plan (http_port set) needs an api_token`, or `the API key is N characters; at least 16 are required` | no key, or a short one | `python -c "import secrets; print(secrets.token_hex(24))" > api_key`, then `--api-key-file api_key` |
| `bigbrain: refusing the stage dir ...` (staging fails, or a node exits at once) | the stage directory on that PC is a symlink, belongs to another user, or is writable by group or others | use the default `~/.cache/bigbrain/stage`, or `chmod 700` a directory you own; never a shared one such as a `/tmp/bigbrain` someone else created |
| `run_node.sh: refusing node.env ...` (exit 77) | `node.env` is a symlink, someone else's, or writable by group or others | `chmod 600 node.env` and make it a plain file of yours |
| `jinja2 X is too old to render a model's chat template safely`, and node 0 exits | the node's environment has a jinja2 older than 3.1.6 (common in older TabbyAPI venvs) | `<that venv>/bin/python -m pip install -U "jinja2>=3.1.6"` |
| `a ... frame of N bytes is over this link's limit`, or another link error ending in `closing it`, then `/health` 503 | a peer sent a frame this node will not take: a different BigBrain build, or something that is not a BigBrain node | stage the same version everywhere (`deploy/verify_stage.py`); if a legitimate frame is that large (an unusual model), raise `--max-frame-mib` on the nodes. Check who else can reach the pipe ports |
| `... is not an IP address or a host name`, or `... is not a user name` | a host or user in the registry, the site file or on the command line has characters ssh or a shell would misread | fix the value; it is refused on purpose (SECURITY.md) |
| `bigbrain: ignored .../site.toml` on stderr | a site file in the current directory is not read on its own | `export BIGBRAIN_SITE=$PWD/site.toml`, or move it to `~/.config/bigbrain/site.toml` |
| HTTP 503 `too many open connections` | more than `--max-connections` (64) connections are open to node 0 | fewer parallel clients, or raise `--max-connections` on node 0 |
| the start-up handshake fails with `a node on an older build never answers: stage the same bigbrain/ package on every node` | nodes run different engine versions | re-launch from one checkout; `deploy/verify_stage.py` shows which node differs |
| `unsupported version N (this build speaks ...)` on a link | the same, at the frame level | the same |
| HTTP 503 with `Retry-After` on a request | the bounded queue is full (`--queue-size`, default 8) | back off and retry, or raise `--queue-size`; more throughput needs more `--slots` (and the memory for them) |
| HTTP 503 from `/health` for a long time after the launch | still loading, or `engine.dead` is set | `engine.dead` names the reason; otherwise read the node logs |
| CUDA out of memory while a node loads | the plan was sized from a free-memory reading and something else took memory since, or the cache for your slots and context is bigger than planned | stop the other process, re-plan, or lower `--slots` / `--max-num-tokens`; raise `--headroom-mib` if the margin is too thin |
| `Address already in use` on a node | an earlier run's node is still alive | run `teardown_command` for that stage directory |
| a node cannot connect to its peer for a long time | the peer is still loading (up to 10 minutes of retries), or a firewall blocks the port | check `ss -ltn` on the peer and the ports table in INSTALL.md |
| greedy output differs from a single-GPU run | exllamav3's kernels are not batch-invariant, so a near-tie can flip | run `bench grid` / `parity_check`; a divergence with a tiny logit margin is numerics, not a bug |
