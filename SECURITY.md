# Security

**BigBrain is built for a trusted LAN: a home lab where every machine on the network is yours. Do not expose any of its ports to the
internet or to a network you do not control.** What protection it has is a bearer key on the HTTP endpoint and ssh keys for the
controller. This page says exactly what that covers and what it does not.

## What is protected

- **The HTTP endpoint on node 0** requires `Authorization: Bearer <key>` (or an `x-api-key: <key>` header) on every route except
  `GET /health`. The key is compared in constant time (`secrets.compare_digest`). The server **refuses to start without a key**, unless
  you pass `--insecure-no-auth` (which is for loopback tests only and logs a warning).
- **The controller reaches the PCs over ssh with keys** (`ssh -o BatchMode=yes`), never with a password. Whatever protects those keys
  protects the cluster: anyone who can ssh to a PC as the BigBrain user can run code there anyway.
- **Request bodies are bounded** (8 MiB), and BigBrain's own code never unpickles anything: wire frames are parsed with `struct` and numpy, control
  messages are JSON, and the debug dumps are loaded with `np.load(..., allow_pickle=False)`.
- **The chat template** of a model folder is rendered in a jinja2 sandbox. (Model folders are still trusted input: exllamav3 loads their weights and
  configuration. Do not point BigBrain at a model folder you did not get from somewhere you trust.)

## What is not

| gap | consequence | what to do |
|---|---|---|
| **One shared key, no TLS** | The key, the prompts and the answers cross the LAN in clear text. Anyone who can read the traffic can read and reuse the key. There are no per-user keys, scopes or rate limits. | Keep the endpoint on a trusted LAN, or put a TLS-terminating reverse proxy (or a VPN / ssh tunnel) in front of it and bind node 0 to loopback with `--host 127.0.0.1`. |
| **The key also opens the debug and metrics routes** | `/v1/bigbrain/generate` takes raw token ids and can return the logits; `/v1/bigbrain/metrics` and `/stats` describe the cluster. | Treat the key as an admin key. |
| **`GET /health` is open** | It reveals the model name, queue depth, link state and counters to anyone who can reach the port. | Firewall the port to your clients. |
| **The inter-node links have no authentication and no encryption** | The activation links and the token link are raw TCP with a magic number and a version, nothing more. Anyone who can reach a node's pipe port can read the hidden states passing through (which carry the content of the prompts), or inject frames and control messages and so corrupt or crash a running pipeline. There is no cap on the size of a frame a peer may send, so such a peer can also make a node buffer a lot of memory. | Allow those ports (16900.. by default) only from the other nodes' addresses (INSTALL.md has a `ufw` example). Do not route them through anything you do not trust. If you must cross an untrusted network, tunnel them (ssh `-L`, WireGuard) and bind the listeners to loopback with `--bind-host 127.0.0.1`. |
| **The key is on the node 0 command line** | The controller passes `--api-token KEY`, which any local user on that PC can read with `ps`, and the controller's dry run prints it. | On a shared PC start node 0 yourself with `--api-key-file` (a mode 0600 file; `deploy/run_node.sh` is built for this), and do not paste a dry run that contains a real key into an issue or a chat. |
| **The default stage directory is `/tmp/bigbrain`** | `/tmp` is writable by every local user: another user on a node PC could pre-create or replace the staged Python files, which the node then runs as the BigBrain user. | On any PC with other users, pass `--stage-root` pointing at a private directory (absolute path, e.g. under your home). |
| **No protection against slow or abusive HTTP clients** | The server is the Python standard library's threaded HTTP server with no socket timeouts, request-rate limits or connection caps; the bounded request queue protects the GPUs, not the threads. | Same answer: trusted clients only, or a reverse proxy in front. |
| **Fail-stop, no isolation between requests** | All requests share one cache and one pipeline; a dead link ends the service for everybody. | Do not use one cluster for mutually distrustful users. |

## Secrets and the repository

- The files `api_key`, `*.env` (including `node.env`), `api_tokens.yml`, `config.yml` and a root-level `config.json` are git-ignored, and
  `tests/test_repo_hygiene.py` fails if one of them, or a string shaped like an API key, token or private key, is in the tree.
- `site.toml`, `model_registry.json` and `patches/hosts.toml` hold addresses and user names, not secrets, and are git-ignored too.
- Do not paste real keys into issues. A dry run of the controller and the nodes' command lines contain the key.

## The optional patches

`patches/tabbyapi/auth_redact.patch` stops an upstream TabbyAPI printing its API and admin keys into its start-up log. It is unrelated to
BigBrain's own server, which never logs its key.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting for this repository (Security tab ->
Report a vulnerability) if it is enabled, or contact the maintainer through the address on their GitHub profile. This is a
spare-time project: expect a reply in days, not hours.
