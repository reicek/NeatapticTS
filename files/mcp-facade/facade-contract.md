# MCP Lazy-Load Facade — Step 02 Contract & Snapshots

> Research artifact for the `cortex` and `devtools` lazy-load facades.
> Updated for the router-tool contract; production facade code now lives under
> `scripts/agent-customization/mcp/`.

## Scope

Finalize the contract and lightweight tool snapshots for the two planned
stdio MCP facades:

| Facade key | Real server | Lazy-spawn command                                                                                   |
| ---------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `cortex`   | `cortex`    | `node scripts/mcp-semantic/repo-cortex-mcp.mjs`                                                      |
| `devtools` | `devtools`  | `npx -y chrome-devtools-mcp@1.4.0 --headless=true --usage-statistics=false --performance-crux=false` |

## Deliverables

All files live under `files/mcp-facade/` in this research session. The eventual
facade implementation should read the `*-tool-snapshot.json` files from
`scripts/agent-customization/mcp/` (per the plan), but those files are not
pre-positioned in source yet.

| File                          | Purpose                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `cortex-real-snapshot.json`   | JSON-RPC `initialize`, `tools/list`, and safe `tools/call` responses from the real Repo Cortex server. |
| `cortex-tool-snapshot.json`   | Single-router-tool snapshot the `cortex` facade exposes at session start.                              |
| `devtools-real-snapshot.json` | JSON-RPC `initialize` and `tools/list` from the real `devtools` package.                               |
| `devtools-tool-snapshot.json` | Single-router-tool snapshot the `devtools` facade exposes at session start.                            |
| `facade-contract.md`          | This document.                                                                                         |

## Key findings

### 1. Router-tool facade (one tool per facade)

- **Repo Cortex:** the facade now exposes exactly one router tool named
  `cortex`. The host calls it with `{ operation, args? }`; the facade extracts
  `operation` and forwards the real call to the heavy Repo Cortex server.
- **Chrome DevTools MCP v1.4.0:** the facade now exposes exactly one router
  tool named `devtools`. The host calls it with `{ operation, args? }` and the
  facade forwards to the real DevTools server.
- This reduces the per-session `tools/list` token cost from ~8.9 k tokens
  (18 Cortex + 29 DevTools tools) to a few hundred tokens (two router tools).

### 2. Stdio framing is different for each real server

- **Repo Cortex** uses the repo's custom `mcp-utils.mjs` framework. It
  supports both `Content-Length:` and newline-delimited JSON. The facade can
  reuse the same `mcp-utils.mjs` utilities to talk to it.
- **Chrome DevTools MCP** uses a bundled version of the SDK whose
  `StdioServerTransport` reads and writes **newline-delimited JSON**, not
  `Content-Length`. The `devtools` facade must use `JSON.stringify(msg) + '\n'`
  framing to the child process.

### 3. Chrome DevTools MCP startup behavior

- The server prints legal/telemetry disclaimers to **stderr**, not stdout.
- It calls `checkForUpdates()` before binding the transport, so the first
  `initialize` response can be delayed on cold installs or slow networks.
- With `--headless=true` and no browser URL, the server does **not** launch
  Chrome during startup; it only connects when a tool call actually needs the
  browser. This makes `tools/list` safe to capture and fast to proxy.
- Telemetry is disabled by setting `CI=1` and
  `CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS=1` and passing
  `--usage-statistics=false`. Performance CrUX lookups are disabled with
  `--performance-crux=false`.

### 4. Snapshot contract

The facade's `tools/list` must be a **static snapshot** because the host
queries it at session start, before any lazy spawn.

The snapshot contains a single router tool with an `operation` string argument
and an optional `args` object. The router tool's schema is small and stable,
while the real server's argument schemas are forwarded verbatim after the
facade spawns the child.

```json
{
  "name": "cortex",
  "description": "Router for the Repo Cortex MCP server. Call with { operation, args? }.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "operation": {
        "type": "string",
        "description": "Real Cortex tool name, e.g. search_corpus."
      },
      "args": {
        "type": "object",
        "description": "Arguments forwarded to the real tool."
      }
    },
    "required": ["operation"]
  }
}
```

The `devtools` snapshot uses the same shape with `name: "devtools"` and a
different description.

## Decision record: router-tool facade

- **Date:** 2026-07-05
- **Decision:** Replace the per-tool lazy-load facade snapshot with a single
  router tool per facade (`cortex` / `devtools`).
- **Rationale:** The original design mirrored every real tool in the static
  snapshot. That still cost ~8.9 k tokens per session because the host
  received 18 Cortex tools and 29 DevTools tools at startup. A router tool
  exposes only one stable schema and defers all operation-specific validation
  to the real server after lazy spawn, cutting the per-session tool-list cost
  to a few hundred tokens.
- **Contract:** Host calls `cortex({ operation: "<op>", args: {...} })` or
  `devtools({ operation: "<op>", args: {...} })`. The facade validates the router
  name, extracts `operation`, and forwards `tools/call { name: <op>,
arguments: args }` to the real server.
- **Trade-off:** The host no longer sees per-operation argument schemas at
  session start. Argument validation happens at call time by the real server,
  which is acceptable because both facades are lazy and the real server is the
  authoritative validator.
- **Callers updated:** All `.github/agents/*.agent.md`, `.github/skills/*/SKILL.md`,
  `.github/copilot-instructions.md`, and `CLAUDE.md` references now use the
  router envelope.
- **Obsolete artifacts removed:** `devtools-snapshot-generator.mjs`,
  `generate-cortex-tool-snapshot.mjs`, and `capture-devtools-snapshot.mjs` are
  deleted because the router snapshot is hand-written and stable.

## Facade contract details

### Operation mapping

Callers use the single router tool and pass `{ operation: "<tool-name>", args:
{ ... } }`. The facade extracts `operation` and `args`, then forwards a real
`tools/call { name, arguments }` to the heavy child server.

```text
caller:          cortex({ operation: "search_corpus", args: { query: "network" } })
facade forwards: tools/call { name: "search_corpus", arguments: { query: "network" } }
target server:   cortex
```

For tools that take no arguments, callers may omit `args` entirely:

```text
caller:          cortex({ operation: "freshness_check" })
facade forwards: tools/call { name: "freshness_check", arguments: {} }
```

### Lazy lifecycle

1. Host starts the facade server from `.vscode/mcp.json`.
2. Facade answers `initialize` and `tools/list` from its static snapshot
   without touching the real server.
3. First `tools/call` triggers:
   - spawn the real heavy server as a child process,
   - perform MCP `initialize` with the child,
   - cache the child stdio transport,
   - forward the call and return the result.
4. Subsequent calls reuse the cached transport.
5. On child crash or fatal transport error, clear the cache and re-lazy-init
   on the next call.

### Transport details

| Concern                  | `cortex` facade                                 | `devtools` facade                                                                                    |
| ------------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Spawn command            | `node scripts/mcp-semantic/repo-cortex-mcp.mjs` | `npx -y chrome-devtools-mcp@1.4.0 --headless=true --usage-statistics=false --performance-crux=false` |
| Framing to child         | Content-Length (via `mcp-utils.mjs`) or NDJSON  | NDJSON only                                                                                          |
| Env to disable telemetry | not applicable                                  | `CI=1`, `CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS=1`                                                  |
| Stderr handling          | log/ignore                                      | ignore (disclaimers); optional passthrough for debugging                                             |
| First-call timeout       | normal                                          | generous (slow `checkForUpdates` / install)                                                          |

### Error propagation

- **Unknown router tool name:** return `isError: true` immediately, no spawn.
- **Missing `operation` argument:** return `isError: true` immediately, no spawn.
- **Unknown `operation` value:** spawn the real server and let it return the
  tool-not-found error, so parity drift is visible.
- **Spawn or initialize failure:** return `isError: true` with
  `{ facade, target, error, available: false, fallbackHint }`.
- **Child crash after init:** mark transport stale, return error, and
  re-lazy-init next call.
- **Real server tool error:** forward the real response's `isError` and text,
  prefixing with `[cortex]` / `[devtools]` for observability.
- **Fatal facade misconfiguration:** throw a JSON-RPC error only if the
  facade server itself cannot start.

### JSON-RPC method routing and request-ID mapping

Methods are handled as follows:

| Method                                 | Handler                                                                                                                                                    |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `initialize`                           | Facade responds from snapshot metadata; child is not spawned. The returned capabilities must **not** advertise `tools.listChanged`.                        |
| `notifications/initialized`            | Facade acknowledges; no child interaction.                                                                                                                 |
| `ping`                                 | Facade responds locally.                                                                                                                                   |
| `tools/list`                           | Facade returns the static snapshot; no child interaction.                                                                                                  |
| `tools/call`                           | Facade validates the router tool name, extracts `{ operation, args? }`, lazily spawns the child if needed, then forwards `tools/call { name, arguments }`. |
| `resources/list`, `prompts/list`, etc. | Not supported by either target; return JSON-RPC error `-32601` (method not found).                                                                         |

Host request IDs are preserved in the host→facade→host response path. The
facade may use its own internal request IDs to the child, but the response
returned to the host must carry the original host ID.

### Initialize response shape

The facade's response to the host's `initialize` request must look like the
real server's response but with its own short server key and without dynamic
capabilities:

```json
{
  "protocolVersion": "2024-11-05",
  "capabilities": {
    "tools": {}
  },
  "serverInfo": {
    "name": "cortex",
    "title": "Neataptic Cortex facade",
    "version": "0.1.0"
  }
}
```

For `devtools`, `serverInfo.name` is `devtools` and `capabilities.tools` is
`{}` (no `listChanged`).

### Input schema passthrough

The router tool's schema only validates `operation` (required string) and
`args` (optional object). The facade does **not** validate the contents of
`args`; it relies on the host's schema validation against the full real tool
list (if available) and the real server's own validation. The facade forwards
`arguments` verbatim inside `tools/call { name, arguments }`.

### What is intentionally stripped from snapshots

The static snapshots intentionally expose **only** the router tool. The real
server's full argument schemas are not duplicated in the facade snapshot
because `args` is an opaque object from the host's perspective. The router
snapshot keeps only:

- `name` (`cortex` or `devtools`)
- One-line `description`
- Minimal `inputSchema` requiring `operation` and allowing optional `args`

This removes all per-operation schema bloat from the session-start tool list.

### Child lifecycle and cleanup

- The child is spawned once per facade process and cached.
- The facade must prevent concurrent first-call races with a lazy-init lock so
  only one child is created.
- On facade stdin close or `SIGTERM`/`SIGINT`, the facade must close the child
  transport and kill the child process (if it spawned it).
- If the child exits unexpectedly, the cached transport is cleared and the
  next `tools/call` attempts a fresh spawn.

### Self-check output contract

Both `cortex-facade.mjs --self-check --json` and
`devtools-facade.mjs --self-check --json` must return the same JSON shape
without connecting to the real servers:

```json
{
  "facade": "cortex",
  "target": "cortex",
  "snapshotPath": "scripts/agent-customization/mcp/cortex-tool-snapshot.json",
  "snapshotToolCount": 1,
  "snapshotValid": true,
  "canParseSnapshot": true,
  "spawnCommand": "node scripts/mcp-semantic/repo-cortex-mcp.mjs",
  "pass": true
}
```

For `devtools`, the same shape with `facade: "devtools"`, `target:
"devtools"`, and the appropriate snapshot/command. The self-check
verifies that the snapshot is readable, exposes exactly one router tool,
requires `operation`, and that the spawn command is known; it does **not**
spawn the real server.

## Snapshot parity gate recommendation

Add a validation gate (future work) that:

1. Starts the real Repo Cortex server and calls `tools/list`.
2. Starts the real Chrome DevTools MCP server and calls `tools/list`.
3. Confirms each real server still exposes all operations reachable through the
   router (i.e., the set of real tool names is a superset of the documented
   operations).
4. Fails if a documented operation no longer exists on the real server.

This catches drift after package updates or repo-cortex changes without
forcing the facade snapshot to mirror every tool schema.

## Risks and open questions

1. **Real server version pinning.** `npx -y devtools@latest` always
   resolves the latest version. A future release could change tool names,
   categories, or framing. Recommended mitigation: pin the spawn command to the
   captured version (`devtools@1.4.0`) and refresh the snapshot and
   parity gate when a newer version is adopted.
2. **Chrome availability.** The `devtools` facade only needs Chrome when a
   browser-scoped tool is called, but if those tools are invoked and Chrome is
   not installed, the proxy must surface the real server's error cleanly.
3. **`checkForUpdates` latency.** The first `devtools` call may be slow (use a
   ≥120 s first-call timeout). The facade should not warm the child on
   `tools/list`; it must remain lazy.
4. **Cortex DB path.** The Repo Cortex server defaults to
   `rag-index/data/turso-replica.sqlite`. The facade should pass `TURSO_DATABASE_URL`
   through from the host environment unchanged.
5. **Framing adapter risk.** The VS Code/Copilot host speaks
   `Content-Length` framing, while the `devtools` child speaks NDJSON. The
   facade must use `Content-Length` on the host side and NDJSON on the child
   side, with a robust incremental line parser for child stdout.
6. **`tools/listChanged` from devtools.** The real devtools server advertises
   `tools.listChanged: true`. Because the facade serves a static snapshot, it
   should initialize the child without exposing that capability, or ignore
   `notifications/tools/list_changed` from the child.
7. **Non-text content forwarding.** DevTools tools such as `take_screenshot`
   may return image/resource content arrays. The facade must forward such
   payloads without truncation or re-serialization loss.
8. **Concurrent first-call race.** Two simultaneous `tools/call` requests must
   spawn exactly one child process. Implement a lazy-init lock.

## Recommended first implementation slice

Start with `cortex-facade.mjs` because:

- The real server is repo-controlled and available without npm downloads.
- It can reuse the existing `scripts/agent-customization/mcp/mcp-utils.mjs`
  framework.
- The static snapshot is small and stable.

After red tests pass, implement `devtools-facade.mjs` using an NDJSON child
forwarder, then add the snapshot parity gate.
