# The HTTP endpoint (node 0, `src/bigbrain/server.py`)

What a client sees when it talks to node 0. The port is `--http-port` (the flag's default is 16890; the examples in these docs use 16980).
A field may be **added** to any answer in later versions; none is renamed or removed within v0.1.x.

## Auth

Every route except `GET /health` needs `Authorization: Bearer <key>` (or an `x-api-key: <key>` header), compared with
`secrets.compare_digest`. The key comes from `--api-key-file` (what the controller passes: it stages the key as `api_key`, mode 0600, in
node 0's stage directory), else the deprecated `--api-token` (visible in `ps`), else the environment variable named by `--api-key-env`
(default `BIGBRAIN_API_KEY`); it must be at least 16 characters. With none of them the server refuses to start, unless you pass
`--insecure-no-auth`, which is accepted only with a loopback `--host`. There is no TLS: see SECURITY.md.

Before auth: the server holds at most `--max-connections` (default 64) connections at once and answers any more with a 503 straight
away; a connection silent for `--conn-timeout` seconds (default 60) is dropped. When the pipeline is down, every request gets a 503
(`the pipeline is down: ...`) at once.

## Routes

| route | auth | what |
|---|---|---|
| `GET /health` | open | status, queue depth, link health; 503 until the nodes' handshake succeeds |
| `GET /v1/models` | key | the one model (`id` is `--model-name`, default the model folder's name) |
| `POST /v1/chat/completions` | key | messages -> the model folder's chat template; stream (SSE) or not |
| `POST /v1/completions` | key | a raw prompt (string, or token ids) |
| `POST /v1/bigbrain/generate` | key | debug and parity: `prompt_ids` in, token ids (+ every step's fp32 logits, base64) out; greedy by default |
| `GET /v1/bigbrain/metrics`, `POST` (`{"reset": true}`) | key | decode and prefill tok/s, first-token latency, speculation acceptance, per-node busy / wait / send % and VRAM; POST reads and zeroes the window. This one sends a `stats` op down the pipeline, so do not poll it fast |
| `GET /v1/bigbrain/stats[?recent=N]` | key | the per-request record (below). Node 0's memory only: free to poll |

Anything else answers 404, a bad or missing key 401, a malformed body 400, a body over 8 MiB 413. Errors have the OpenAI shape
`{"error": {"message", "type", "code"}}`.

### Chat and completion requests

`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `max_tokens` / `max_completion_tokens`, `stop`, `stream` and
`stream_options.include_usage`, `chat_template_kwargs`, and `tools` (passed to the chat template only: **tool calls are not parsed from the
output**). `n` must be 1 and `logprobs` are refused (400). `stop` takes at most 16 strings of at most 256 characters each (400 otherwise). Messages must be text (a list of text parts is joined).
When the request gives no sampling value the server uses the model folder's `generation_config.json` (`do_sample: false` means greedy).

The chat template is `chat_template.jinja` in the model folder, else `tokenizer_config.json`'s `chat_template`, rendered with jinja2 in a
sandbox the way transformers does. With no template or no jinja2 it falls back to ChatML, which is exact for the Qwen family only. A
jinja2 older than 3.1.6 is refused: node 0 does not start (its sandbox has published escapes).

Extensions on any route:

- `"spec": "none" | "ngram" | "draft"` and `"spec_k": 1..15`, **greedy requests only** (`temperature` 0 or `top_k` 1; a sampled request runs
  without drafts). `"draft"` needs a draft model (400 otherwise). With one slot and no speculation these are refused (single-sequence server).
- A client names its job with the headers `X-BigBrain-Job` (a label, printable text, at most 120 characters) and `X-BigBrain-Job-Progress`
  (`n/m`). The label shows up in `/v1/bigbrain/stats` as the job using the model. A malformed progress is ignored, never an error.

`POST /v1/bigbrain/generate` also takes `"diag": true` and answers an `info` object: `proposed`, `accepted`, `rounds`, `tokens`, `slot`,
`first_token_s`, `total_s`, `decode_tok_s`, `spec_mode`, and with `diag` the top-1 minus top-2 logit `margins` per token (a parity
divergence with a tiny margin is numerics, not a bug). `return_logits` is refused in batched mode (400).

### Scheduling and back-pressure

Single-sequence mode (`--slots 1`, no speculation): one engine thread, FIFO, from a bounded queue (`--queue-size`, default 8). Batched mode
(`--slots N > 1` and/or `--spec-draft ngram|model`): up to N requests run at once through `engine.PipelineEngine`; the rest wait in the same
bounded queue. When the queue is full the answer is **503** with `Retry-After` (`--retry-after`, default 5 s). A client that disconnects
cancels its request between tokens; queued requests are skipped.

### `GET /health`

Open. `200` with `"status": "ok"` when the engine is ready and the scheduler is alive, otherwise `503` with `"status": "degraded"`. Fields:
`status`, `model`, `busy`, `running`, `slots`, `queue_depth`, `queue_size`, `served` (also `requests_served`), `failed`, `max_context`,
`chat_template`, `stats` (the record of `/v1/bigbrain/stats` without `recent`) and `engine`. In batched mode `engine` is
`{ok, nodes, links {outbound, token_link: "ok" | "dead"}, recurrent, batched, ready, slots, active, queued, dead, spec, inflight_cap}`
(plus `tokens_per_s` and `active` at the top level); in single-sequence mode `{ok, nodes, links, recurrent, batched: false}`. It stays 503 until the
handshake succeeds, and reports `engine.dead` (503) when a link died, a peer sent a malformed frame, a step timed out, or the handshake
never completed. Without the key, `engine.dead` only says that the pipeline is down; send the key with the request to read the reason
(it can name peers, addresses and paths).

### `GET /v1/bigbrain/metrics`

`{"mode": "batched" | "single-sequence", "engine": {...}, "nodes": [...], "scheduler": {...}}`. `engine`: `decode_tok_s`, `per_sequence_tok_s`,
`prefill_tok_s`, `first_token_s {mean, p50, max}`, `steps`, `step_latency_s`, `inflight {mean, max}`, `rows_per_decode_step`,
`spec {mode, k, acceptance_rate, tokens_per_row, accepted_histogram, ...}`. Each node: `busy_pct`, `wait_pct` (blocked on an empty inbound
queue), `send_pct`, `steps`, `rollbacks`, `rolled_tokens`, `vram {allocated_mib, peak_allocated_mib, reserved_mib, free_mib, total_mib}`.
`scheduler`: `running`, `queued`, `served`, `failed`, `workers`.

### `GET /v1/bigbrain/stats[?recent=N]`

Cheap (node 0's memory only), so a status page may poll it every few seconds. It returns:

```
{ "since", "uptime_s",
  "requests":      {served, failed, cancelled, in_flight, queued},
  "tokens":        {prompt_total, completion_total},
  "decode_tok_s":  {last, avg, avg_n, aggregate, per_sequence},
  "prefill_tok_s": {last, avg},
  "first_token_s": {last, avg},
  "last_request":  {at, prompt_tokens, completion_tokens, total_s, finish},
  "job":           {label, unit, of, updated_at} | null,
  "recent":        [ {id, at, route, prompt_tokens, completion_tokens, queue_s, first_token_s, total_s,
                      decode_tok_s, prefill_tok_s, finish, spec, job}, ... ]      (only with ?recent=N, at most 200) }
```

Definitions: *decode tok/s* of a request is `(completion_tokens - 1) / (last token - first token)`; *prefill tok/s* is
`prompt_tokens / (first token - the moment a slot took the request)`; `avg` is token-weighted over the last 20 finished requests;
`aggregate` is generated tokens over the busy time of the last minute of activity across all slots; `per_sequence` is the plain mean of
the per-request decode rates. `finish` is `stop`, `length`, `cancelled` or `error`.

## Flags that shape the endpoint

`--http-port`, `--host`, `--model-name`, `--api-key-file` / `--api-key-env` / `--api-token` / `--insecure-no-auth`, `--max-connections`,
`--conn-timeout`, `--slots N` (alias
`--max-batch-size`, the larger wins), `--max-draft K`, `--spec-draft none|ngram|model`, `--spec-k`, `--spec-model DIR`, `--draft-device`,
`--ngram-min` / `--ngram-max`, `--inflight`, `--mb-size`, `--prefill-window`, `--hello-timeout`, `--token-timeout`, `--no-async-send`,
`--diag`, `--queue-size`, `--retry-after`, `--default-max-tokens`, and on every node `--watchdog-s` (default 5.0). CONFIGURATION.md lists
them with defaults. `server.py` accepts every node-placement flag the worker takes, and `worker.main` hands any command line that contains
`--http-port` over to `server.main`, so the controller starts node 0 with the same module as the others.
