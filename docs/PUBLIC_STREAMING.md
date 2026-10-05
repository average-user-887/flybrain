# Public streaming mode

`neurofly_daemon.py` was written for a trusted LAN. Its default surface:

| Endpoint | Method | Default behaviour |
| --- | --- | --- |
| `/api/status`, `/api/telemetry`, `/api/paradigms` | GET | open, CORS `*` |
| `/api/stream` | GET (SSE) | open, unlimited clients, 30 Hz per client |
| `/api/command` | POST | **open**: anyone can switch paradigm, inject stimuli, change speed, reset trials, write checkpoints |

Exposing that unchanged would let any visitor drive the fly. Public mode
(`stream_gateway.py`) changes only what is needed to stream safely.

## Enabling it

```bash
# read-only: nobody can send commands
NEUROFLY_PUBLIC=1 ./start_daemon.sh

# operators keep control through a bearer token (>= 16 characters)
export NEUROFLY_PUBLIC=1
export NEUROFLY_ADMIN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
./start_daemon.sh
```

Equivalent CLI flags: `--public`, `--max-stream-clients N`, `--stream-hz HZ`,
`--allowed-origin ORIGIN`. The token is read **only** from
`NEUROFLY_ADMIN_TOKEN`; there is no `--token` flag so it never appears in
`ps` output or shell history.

## What changes in public mode

| Aspect | Private (default) | Public |
| --- | --- | --- |
| `POST /api/command` | always dispatched | `403` unless `Authorization: Bearer <NEUROFLY_ADMIN_TOKEN>` matches (constant-time compare); `403 read_only` when no token is configured |
| Concurrent SSE clients | unlimited | capped (`NEUROFLY_MAX_STREAM_CLIENTS`, default 50); extra clients get `503` + `Retry-After: 5` |
| SSE rate | 30 Hz | throttled (`NEUROFLY_STREAM_HZ`, default 10) |
| Command body | any size | `413` above 64 KiB (this cap applies in both modes) |
| `Access-Control-Allow-Origin` | `*` | `*` unless `NEUROFLY_ALLOWED_ORIGIN` is set |
| `/api/status` | | gains a `stream` object: `public`, `read_only`, `commands_require_token`, `max_stream_clients`, `stream_hz`, `active_stream_clients`, `rejected_stream_clients`, `rejected_commands` |

Reads (`/api/status`, `/api/telemetry`, `/api/paradigms`, `/api/stream`) work
for everyone in both modes.

Sending a command with the token:

```bash
curl -X POST "$URL/api/command" \
     -H "Authorization: Bearer $NEUROFLY_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"action": "set_speed", "speed": 10}'
```

## What it does not do

- **No TLS, no rate limiting per IP, no request logging.** Put the daemon
  behind a reverse proxy or tunnel that terminates TLS and can throttle
  abusive clients. Bind the daemon to `--host 127.0.0.1` and let the proxy
  forward to it.
- **The viewer still POSTs.** `web/app.js` (`DaemonBridgeClient`) sends
  commands without a token. Its candidate URL list is now only the page origin,
  the page host and localhost (the LAN address was removed); a public viewer
  build must still handle `403` gracefully (see `docs/RELEASE_AUDIT.md`).
- **Telemetry is not redacted.** The stream contains only simulation state
  (fly position, sensory values, weights summary), never file paths or
  host details, so nothing needs redaction today. Keep it that way when
  adding fields.
- **Publishing a link is an outward-facing step** and needs the owner's
  confirmation (`docs/OPEN_SOURCE_PLAN.md`, Phase 4).

## Suggested deployment shape

```
browser ──HTTPS──> reverse proxy (TLS, per-IP limits, /api/stream & GET only)
                       │
                       └──http://127.0.0.1:8769──> neurofly_daemon.py --public --host 127.0.0.1
```

Allow only `GET /api/status`, `GET /api/telemetry`, `GET /api/paradigms` and
`GET /api/stream` through the proxy for anonymous visitors. Operators reach
`POST /api/command` over the LAN or an SSH tunnel with the bearer token.

## Tests

`tests/test_stream_gateway.py` covers policy parsing, token checks, the SSE
cap (including slot release after disconnect), CORS origin, oversized bodies
and that private mode is byte-for-byte the historical behaviour.
