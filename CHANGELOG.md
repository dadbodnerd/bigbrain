# Changelog

## v0.2.1 (2026-10-15): security fixes

The fixes of a security review of v0.2.0. SECURITY.md has the threat model and what is still not protected. Every fix has a test
(`tests/test_security.py`).

**Changed defaults and behaviour (read before you upgrade):**

- **The key is a file now.** The controller takes `--api-key-file PATH` (or `$BIGBRAIN_API_KEY`), stages the key on node 0 as
  `<stage-root>/api_key` (mode 0600) and starts node 0 with `--api-key-file api_key`. The key is on no command line and in no output, dry
  runs included. `--api-token` still works on the controller but is deprecated (it puts the key on the controller's own command line).
- **Keys must be at least 16 characters** (`python -c "import secrets; print(secrets.token_hex(24))"`), on node 0 and in the controller's plan.
- **The default stage directory is `~/.cache/bigbrain/stage`** (was `/tmp/bigbrain`), created with mode 0700. A stage directory that is a
  symlink, belongs to someone else or is writable by group or others is refused, by the controller and by `deploy/run_node.sh`, which
  also refuses such a `node.env`. `--stage-root` accepts `~/...` (expanded on the node).
- **jinja2 3.1.6 or newer** is required (`pyproject.toml`), and node 0 refuses to start with an older one. Older TabbyAPI venvs often have
  an older 3.1.x: `pip install -U "jinja2>=3.1.6"` in that venv. numpy must be 1.24 or newer.
- **A `site.toml` in the current directory is no longer read on its own.** The site file is `$BIGBRAIN_SITE`, else
  `~/.config/bigbrain/site.toml`; a `./site.toml` is named on stderr and ignored.
- **`--insecure-no-auth` needs a loopback `--host`.**
- **Stop strings are at most 256 characters** (still at most 16 of them).

**Fixed:**

- Shell injection through host names: node 0's remote command put `--next-host` / `--token-host` in unquoted. Every value in a remote
  command is quoted now; hosts (an IP address or a host name) and users (a POSIX-style name) are checked in the plan, in `site.toml`, in
  `build_registry` (a servers-file URL whose host is not a host name is skipped) and by `site.ssh_target`; every ssh argv, in the package
  and in `deploy/` and `patches/`, has `--` before the destination.
- Unauthenticated thread exhaustion on the HTTP endpoint: `--conn-timeout` (default 60 s) drops a silent connection, and
  `--max-connections` (default 64) answers any connection beyond it with a 503 at once.
- Wire frames: a frame's size is checked against a limit derived from the model before its payload is read (`--max-frame-mib`
  overrides), shapes are checked on Python integers (a huge shape could wrap to zero in numpy), and a malformed frame closes the link.
  Node 0's token-link reader and every node's inbound reader now treat any exception as a dead link, so the pipeline is marked dead at
  once (`/health` 503, new requests 503) instead of `/health` saying ok while requests hang for `--token-timeout`.
- `/health` without the key no longer shows the reason the pipeline died (it can name peers and paths); send the key to read it.
- `deploy/run_node.sh` no longer glob-expands `node.args`.
- `deploy/release.py`'s extraction fallback for Python before 3.11.4 refuses absolute paths, `..`, links and devices.
- The registry's per-node `user` is now used by `--split even` and `--node` plans too (only the capacity split used it before).
- `deploy/run_node.sh`, `release.py` and `verify_stage.py` are executable in git, so a release made by `release.py` can be started by
  the systemd unit (its `ExecStart` runs `run_node.sh` directly; before, it failed with `status=203/EXEC`).

**Added:**

- systemd unit hardening: `NoNewPrivileges`, `UMask=0077`, `RestrictSUIDSGID`, `LockPersonality`; and an optional sandbox drop-in
  (`deploy/bigbrain-node@.service.d/sandbox.conf`: `PrivateTmp`, `ProtectSystem=strict`, `ProtectHome=read-only`) for systems where a user
  unit can use one.
- Node 0 warns when its key file is readable by group or others.
- INSTALL.md: "How a BigBrain cluster is laid out" (what goes on which machine, one environment per GPU PC, per-rank interpreters, GPU
  addressing, the layer split, a worked example, platforms).

**Not changed (planned):** the links between the nodes are still unauthenticated and unencrypted; cluster authentication (an
HMAC-SHA256 challenge-response with a per-session secret) is planned. See SECURITY.md.

## v0.2.0 (2026-10-15)

A pip-installable package: the code moved to `src/bigbrain` (import name `bigbrain`; it was `tools.bigbrain`), seven console scripts, and
the `gpu` and `dev` extras.

## v0.1.1 (2026-10-14)

The first public release.
