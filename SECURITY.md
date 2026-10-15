# Security

**BigBrain is built for a trusted LAN: a home lab where every machine on the network is yours. Do not expose any of its ports to the
internet or to a network you do not control.** It protects its HTTP endpoint with a bearer key and reaches the GPU PCs with ssh keys;
the links between the nodes are not protected at all. This page says exactly what is covered and what is not. v0.2.1 fixed the
findings of a security review of v0.2.0 (listed at the end).

## Threat model

| | |
|---|---|
| **Trusted** | Every host on the network segment the nodes use. The people who can log in to the controller machine and to the GPU PCs as the BigBrain user (they can run code there anyway). The model folders you point BigBrain at. |
| **Not trusted, and defended against** | HTTP clients (they need the key for everything but `/health`). Other local users on a node PC or on the controller machine (the stage directory, the key and `node.env` are kept out of their reach). Files someone hands you: a model registry, a site file, a TabbyAPI servers file (host and user names are checked before they reach ssh or a shell). A model's chat template (rendered in jinja2's sandbox, jinja2 3.1.6 or newer). A misbehaving pipeline peer (frames are size-capped and validated). |
| **Not defended against** | Anyone who can reach the nodes' pipe ports or read the LAN traffic: see the first row of the table below. A user who can write to your home directory or your venv. |

## What is protected

- **The HTTP endpoint on node 0** requires `Authorization: Bearer <key>` (or `x-api-key: <key>`) on every route except `GET /health`. The
  key is compared in constant time (`secrets.compare_digest`) and must be at least 16 characters. The server **refuses to start without a
  key**; `--insecure-no-auth` is accepted only with a loopback `--host` (127.0.0.1, ::1, localhost).
- **Unauthenticated clients cannot pin the server's threads.** At most `--max-connections` (default 64) connections are held at once;
  more get a 503 at once. A connection that sends nothing (or stops reading) for `--conn-timeout` seconds (default 60) is dropped. Request
  bodies are capped at 8 MiB.
- **The key never appears on a command line or in the controller's output.** The controller reads it from `--api-key-file` (or
  `$BIGBRAIN_API_KEY`), sends it over ssh stdin into `<stage-root>/api_key` (mode 0600, in a 0700 directory) on node 0, and node 0 reads
  that file. Dry runs and logs never print it. Node 0 warns when its key file is readable by group or others.
- **The controller reaches the PCs over ssh with keys** (`ssh -o BatchMode=yes ... -- user@host`), never with a password. Host and user
  names are checked (an IP address or a host name of letters, digits, `.`, `-`, `_`; a POSIX-style user name) before they are used, the
  `--` stops a destination from being read as an ssh option, and every value in the remote command is shell-quoted.
- **The stage directory is private.** The default is `~/.cache/bigbrain/stage` on each PC, created with mode 0700. The controller (and
  `deploy/run_node.sh`) refuse a stage directory that is a symlink, is owned by someone else, or is writable by group or others, because
  `python -m` runs code from it. `run_node.sh` applies the same check to `node.env`, which it sources as shell. (The check looks at
  the directory itself: if you choose another `--stage-root`, its parent directories must not be writable by others either, or someone
  could swap the directory after the check. The default under your home is fine.)
- **Wire frames are bounded and validated.** A frame's size and shape are checked on its header before the payload is read; the limit
  comes from the model (hidden size x the widest step, fp32, with slack; at least 64 MiB, `--max-frame-mib` overrides). A malformed frame
  closes that link, the pipeline is marked dead (`/health` answers 503 and new requests get a 503 at once), and no reader thread dies
  silently.
- **No unpickling, no `eval`.** Frames are parsed with `struct` and numpy, control messages are JSON, debug dumps are loaded with
  `np.load(..., allow_pickle=False)`.
- **The chat template** of a model folder is rendered in jinja2's `ImmutableSandboxedEnvironment`. The package requires jinja2 3.1.6 or
  newer, and node 0 checks the version at start-up and refuses an older one (the GPU PCs run the staged files with whatever jinja2 their
  environment has, so pip's version floor does not reach them). Releases before 3.1.6 have published sandbox escapes (CVE-2024-56201,
  CVE-2024-56326, CVE-2025-27516).
- **A `site.toml` in the current directory is never read on its own.** The site file comes from `$BIGBRAIN_SITE` or
  `~/.config/bigbrain/site.toml`; a `./site.toml` is named on stderr and ignored, so running a command inside someone else's checkout does
  not pick up their hosts, users or `[capacity] reader`.

## What is not

| gap | consequence | what to do |
|---|---|---|
| **The node links have no authentication and no encryption** | The activation links (node *i* to node *i+1*) and the token link (last node to node 0) are raw TCP. **Prompts and answers cross them as plain token ids**: every batched `step` control frame carries each sequence's input token ids in JSON, single-sequence mode sends TOKEN frames, and the token link carries every sampled token. Anyone who can read the traffic (a compromised device on the LAN, a mirrored switch port) can decode every prompt and every answer with the model's public tokenizer; the hidden states on the activation links carry the content too. **At start-up each listener accepts whoever connects first**, and the ports are predictable (16900 + ...). A host on the LAN that connects to a node's port before the real peer receives everything that node is sent; one that connects first to node 0's token port can answer the handshake as if it were the last node and **choose every token every client receives** (with an agent downstream, that means injected tool calls). The real peer's connection then waits unanswered. Since v0.2.1 such a peer can no longer make a node allocate unbounded memory or hang node 0 silently, but it can read, forge or end the stream. | Run the nodes on a network segment you trust completely. Allow the pipe ports (16900.. by default) only from the other nodes' addresses (INSTALL.md has a `ufw` example). If a hop must cross a network you do not control, tunnel it (ssh `-L`, WireGuard) and bind the listeners to loopback with `--bind-host 127.0.0.1`. **Planned:** cluster authentication, an HMAC-SHA256 challenge-response on every link with a per-session secret the controller stages (mode 0600) beside the engine, before any frame is processed. |
| **One shared key, no TLS** | The key, the prompts and the answers cross the LAN in clear text between your clients and node 0. Anyone who can read the traffic can read and reuse the key. There are no per-user keys or scopes, and no limit on failed attempts. | Keep the endpoint on a trusted LAN, or put a TLS-terminating reverse proxy (or a VPN / ssh tunnel) in front of it and bind node 0 to loopback with `--host 127.0.0.1`. |
| **The key also opens the debug and metrics routes** | `/v1/bigbrain/generate` takes raw token ids and can return the logits; `/v1/bigbrain/metrics` and `/stats` describe the cluster. An error answer (HTTP 500) to a key holder can carry the exception text. | Treat the key as an admin key. |
| **`GET /health` is open** | It shows the model name, queue depth, link state and counters, `max_context`, where the chat template came from, and the job label and progress a client set with `X-BigBrain-Job`. Since v0.2.1 the reason the pipeline died is shown only to a caller that sends the key. | Firewall the port to your clients. Do not put anything secret in a job label. |
| **No per-client fairness** | The connection cap and idle timeout bound the server's threads, but one client can still take every connection or fill the request queue (`--queue-size`), and a key holder can keep the GPUs busy. | Trusted clients only, or a reverse proxy with rate limits in front. |
| **Fail-stop, no isolation between requests** | All requests share one cache and one pipeline; a dead link ends the service for everybody. | Do not use one cluster for mutually distrustful users. |
| **The stage directory, `node.env` and the venv are code** | Whoever can write to them runs code as the BigBrain user. v0.2.1 refuses a stage directory or `node.env` that someone else can write, but it cannot protect a venv or a home directory that is not yours alone. | Run the nodes as a user whose home and venv only that user can write. On a shared PC, keep the default stage directory or another private one. |
| **A model folder is trusted input** | exllamav3 loads its weights (safetensors) and configuration. The chat template is sandboxed (above), but a malformed or hostile model can still crash a node or produce whatever output it likes. | Use model folders from sources you trust. |
| **A site file can import code** | `[capacity] reader = "module:function"` imports and calls that module on the controller. | Use only site files you wrote. (A `./site.toml` is not read on its own any more.) |

## Secrets and the repository

- The files `api_key`, `*.env` (including `node.env`), `api_tokens.yml`, `config.yml` and a root-level `config.json` are git-ignored, and
  `tests/test_repo_hygiene.py` fails if one of them, or a string shaped like an API key, token or private key, is in the tree.
- `site.toml`, `model_registry.json` and `patches/hosts.toml` hold addresses and user names, not secrets, and are git-ignored too.
- Since v0.2.1 neither the controller's dry run nor the node command lines contain the key. Still, do not paste a real key, or a real
  registry with your addresses, into an issue.

## The optional patches

`patches/tabbyapi/auth_redact.patch` stops an upstream TabbyAPI printing its API and admin keys into its start-up log. It is unrelated to
BigBrain's own server, which never logs its key.

## Fixed in v0.2.1

The findings of the 2026-10-15 review of v0.2.0, all fixed with tests (`tests/test_security.py`):

| finding | fix |
|---|---|
| Host names went into node 0's remote shell command unquoted (a host such as `10.0.0.2;touch /tmp/pwned;#` ran a command on the node), and a user name starting with `-` became an ssh option | every value in a remote command is quoted; hosts and users are checked in the plan, the registry readers, `site.toml` and `build_registry`; every ssh argv has `--` before the destination |
| Any client could hold the HTTP server's threads with idle connections, before authentication | `--conn-timeout` (60 s) and `--max-connections` (64, then a 503) |
| A peer could announce a frame of any size (a 2^40-byte header raised `MemoryError`), and a malformed frame killed node 0's token-link reader without marking the pipeline dead, so `/health` said ok while every request hung for `--token-timeout` (600 s) | header-level size and shape checks, a model-derived frame limit (`--max-frame-mib`), malformed frame = closed link + dead pipeline + immediate 503s; no reader thread dies silently |
| The default stage directory `/tmp/bigbrain` was world-writable, put first on `sys.path`, and `run_node.sh` sourced `node.env` from it | private default `~/.cache/bigbrain/stage` (0700), refused when someone else can write it; the same check on `node.env` |
| The key was on node 0's command line and in every controller printout, dry runs included | the key is staged as a 0600 file and read from there; `--api-key-file` on the controller; nothing prints it |
| No jinja2 version floor for the chat-template sandbox | `jinja2>=3.1.6`, and node 0 refuses an older one at start-up |
| Also: `--insecure-no-auth` on any address; one-character keys; failure detail on the open `/health`; unbounded stop-string length; `run_node.sh` glob-expanded `node.args`; no systemd hardening; `./site.toml` read implicitly; an unfiltered tar extraction fallback in `deploy/release.py` | loopback only; at least 16 characters; the reason for key holders only; 256 characters; `set -f`; `NoNewPrivileges`, `UMask=0077`, `RestrictSUIDSGID`, `LockPersonality` in the unit and an optional sandbox drop-in; `$BIGBRAIN_SITE` or `~/.config/bigbrain/site.toml` only; members checked |

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting for this repository (Security tab ->
Report a vulnerability) if it is enabled, or contact the maintainer through the address on their GitHub profile. This is a
spare-time project: expect a reply in days, not hours.
