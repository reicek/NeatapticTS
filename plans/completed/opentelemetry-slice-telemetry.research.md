# OpenTelemetry slice telemetry — the attribution research artifact

**Plan id:** `opentelemetry-slice-telemetry`
**Tracker:** `plans/completed/opentelemetry-slice-telemetry.plans.md`
**Companion log:** `plans/completed/opentelemetry-slice-telemetry.logs.md`
**Owning slice:** `OT-S1-3` — the executable attribution probe
**Probe:** `scripts/telemetry/probe-attribution-sources.mjs`
**Focused home:** `scripts/telemetry/probe-attribution-sources.test.mjs`

---

## What this artifact is

This artifact is the **published form of one probe run against the live repository**. It is not a
second source of truth — the probe is. Regenerate it with:

    node scripts/telemetry/probe-attribution-sources.mjs --slice-id=OT-S1-3

The probe prints **exactly one JSON document on stdout** (every diagnostic goes to stderr), persists a
copy of that same document under `logs/telemetry/probe-attribution-sources.json`, and **never
hardcodes a `file:line`**: every citation is discovered at run time from the bytes on disk, so the
report cannot drift from the repository it describes.

**The probe never rewrites this artifact.** It writes the stdout document and the git-ignored copy
under `logs/telemetry/` and nothing else, so the embedded block further down must be **re-pasted by
hand from the live stdout** of the command above. The block carries that stdout document unchanged
— the same characters in the same order, in this file's own line endings — so any change to the
repository changes the stdout, and therefore invalidates the pasted block until it is re-pasted.

The `slice_id` field is **optional and derived, never hardcoded**: the probe reads it from
`--slice-id=<ID>` or from the `NEATAPTIC_SLICE_ID` environment variable and **omits the field
entirely when neither is supplied**, so a dependent slice that reuses this probe never publishes
another slice's id. The embedded block carries `"slice_id": "OT-S1-3"` only because the command
above names this slice explicitly.

The focused home asserts that every cited line resolves to a real file and a real line, that a
*demonstrated* signal never cites a bare declaration of a field name, that the probe's verdicts agree
with the live oracles, and that **this artifact agrees with the probe on every signal**.

---

## The verdict table

Each row names one required signal, the verdict a dependent slice may read, the lane that owns the
measurement, and the live evidence the probe resolved.

| signal | verdict | owner (who computes it) | live evidence (`file:line`) |
| --- | --- | --- | --- |
| `slice_duration` | **demonstrated** | layer B, the repo-owned DONE path — the OpenTelemetry layer never computes a duration | `src/neat/evolve/runtime/evolve.runtime.utils.ts:154`, `src/architecture/network/gpu/network.gpu.profiling.ts:229`, `src/acceleration/acceleration.benchmark.ts:150` |
| `context_tokens_delivered` | **demonstrated** | the workflow MCP `get_slice_context` computes it; the dispatch-time producer persists it | `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:622`, `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:884`, `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:885`, `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:1167` |
| `llm_token_consumption` | **unavailable** | the host agent's model accounting, which this repository cannot reach | `scripts/telemetry/telemetry-contract.mjs:244`, `scripts/telemetry/telemetry-contract.mjs:245`, `scripts/telemetry/telemetry-contract.mjs:246` — declarations only, never a read |
| `tool_executions` | **demonstrated** | the host CLI post-tool surface; the repo-owned observer that counts executions is a later slice | `.github/hooks/cortex-refresh.json:8`, `scripts/agent-customization/hooks/post-write-reindex-hook.mjs:54`, `scripts/agent-customization/hooks/post-write-reindex-hook.mjs:44` |
| `session_to_slice_binding` | **unavailable** | the repository, at dispatch time | `scripts/agent-customization/enforcement/runtime-enforcement.mjs:28`, `scripts/agent-customization/enforcement/runtime-enforcement.mjs:29` (the payload-carried session id), `scripts/agent-customization/enforcement/runtime-enforcement.mjs:30`, `scripts/agent-customization/enforcement/runtime-enforcement.mjs:31`, `scripts/agent-customization/enforcement/runtime-enforcement.mjs:32` (the environment fallbacks), `scripts/telemetry/telemetry-contract.mjs:216` |

A **demonstrated** signal means a source that MEASURES the quantity exists on the live bytes today.
An **unavailable** signal means no such source exists — never that the quantity is zero.

---

## The measured attribution evidence, per signal

### `slice_duration` — demonstrated, owned by layer B

The live repository already measures a **monotonic** elapsed interval: a high-resolution clock read
with a wall-clock fallback at `src/neat/evolve/runtime/evolve.runtime.utils.ts:154`, a
`performance.now()` delta in the GPU profiler at
`src/architecture/network/gpu/network.gpu.profiling.ts:229`, and a clock read in the acceleration
benchmark at `src/acceleration/acceleration.benchmark.ts:150`.

What does **not** exist yet is the measurement *of the slice dispatch window itself*. The elapsed
interval of a slice is computed by the **layer-B DONE path** — the dispatch-time producer opens the
transient dispatch record and the DONE writer closes it and derives `duration_ms`. The OpenTelemetry
layer never computes it. The probe therefore reports the **measurement source that exists on the
live bytes**, not a slice measurement: a dependent slice that cannot find a slice-window duration in
the report writes `available: false` with the report's reason.

That limit is machine-readable and not only prose, so the report's `slice_duration` entry carries a
`scope` field: *"the elapsed-measurement SOURCE exists on the live bytes; the per-slice
dispatch-window duration is NOT measured yet, and the layer-B DONE path owns it"*. A dependent slice
that reads only the verdict therefore cannot mistake the demonstrated surface for a per-slice
measurement.

### `context_tokens_delivered` — demonstrated, owned by the workflow MCP

`get_slice_context` in `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs` computes the
per-slice context size: it defaults each response's count at
`scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:622`, accumulates it at
`scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:884` and
`scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:885`, and returns it in the delivered
context block at `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:1167`.

This is **context tokens DELIVERED** to the slice, never model consumption. The probe also confirms
the negative half of obligation (c): **nothing persists the count** anywhere in the repository today
— the value is computed, returned and discarded. The dispatch-time producer (`OT-S1-4`) is the owner
that will persist it.

### `llm_token_consumption` — unavailable

No live source **reads** a model-consumption field (`prompt_tokens`, `completion_tokens`,
`total_tokens`) from a host payload, so there is nothing to measure. The fields exist only as
**declarations** in the ledger schema —
`scripts/telemetry/telemetry-contract.mjs:244`, `scripts/telemetry/telemetry-contract.mjs:245` and
`scripts/telemetry/telemetry-contract.mjs:246`. A declaration is not a measurement.

The nearest total in the repository is its own **context-token estimate**, which is a different
quantity and is named `context_tokens_delivered` for exactly that reason. A dependent slice writes
`available: false` with this reason and `null` counters — **never `0`**.

### `tool_executions` — demonstrated (the surface exists; the observer does not)

The host tool-execution surface **exists and this repository already registers it**: the hook
manifest registers the post-tool event at `.github/hooks/cortex-refresh.json:8`, and the registered
hook code reads the host payload's tool-name field at
`scripts/agent-customization/hooks/post-write-reindex-hook.mjs:54`.

What does **not** exist yet is the repo-owned **observer** that counts executions per slice. The
write-tool hook it would sit beside is bounded in three exact ways, which is why it is evidence of
the host surface and **not** the counter:

- it matches **write tools only** — the pattern is declared at
  `scripts/agent-customization/hooks/post-write-reindex-hook.mjs:44`;
- it carries **no slice id**, so nothing it writes is attributable to a slice;
- every line it writes carries a **timestamp**, which the persisted-artifact contract forbids.

For the same reason the report's `tool_executions` entry carries a `scope` field: *"the host
post-tool SURFACE exists on the live bytes and is registered; the per-slice execution COUNTER is
not implemented yet, and the repo-owned observer owns it"*. A dependent slice that reads only the
verdict therefore cannot mistake the registered surface for a per-slice counter.

The observer is a later slice (`OT-S6-2`). **This is a forward requirement, not a description of the
live bytes:** the observer **must be** registered under **both** post event keys — `PostToolUse` (a
successful completion) and `PostToolUseFailure` (a failed completion) — because a failing tool call
skips the success key entirely. The live manifest today registers **only** `PostToolUse`
(`.github/hooks/cortex-refresh.json:8`); the failure key is **not** registered yet, so a failed
tool call is currently unobserved.

### `session_to_slice_binding` — unavailable

Two independent gaps, both measured:

1. **No documented session-id environment variable exists.** The probe found **two kinds** of
   session-id source on the live bytes, and neither one is a documented environment variable:
   - the **payload-carried** session id, which the runtime reads first, at
     `scripts/agent-customization/enforcement/runtime-enforcement.mjs:28` and
     `scripts/agent-customization/enforcement/runtime-enforcement.mjs:29`;
   - the repository-invented `COPILOT_*` **environment fallbacks** at
     `scripts/agent-customization/enforcement/runtime-enforcement.mjs:30`,
     `scripts/agent-customization/enforcement/runtime-enforcement.mjs:31` and
     `scripts/agent-customization/enforcement/runtime-enforcement.mjs:32`.

   The gap is therefore **not** that the session id is missing: the host payload already carries
   one and the runtime reads it. The gap is that **no documented environment variable** carries it,
   so a host that does not supply the payload field leaves only the repository-invented fallbacks.
   The design therefore never requires an environment variable to carry the session id.
2. **The dispatch-time producer does not exist yet.** The transient dispatch record has a declared
   shape and a writer — the record directory is declared at
   `scripts/telemetry/telemetry-contract.mjs:216` — but nothing calls that writer at dispatch time.

The binding is consequently owned by the repository at dispatch time: the observer reads the **open
dispatch record** and **stamps** its `slice_id` onto each observation. The binding is a direct
stamp, never an inference from a guessed id.

The verdict stays **unavailable** because gap 2 is independently true: the binding cannot be
published before the producer that stamps it exists, so a session half that is readable today does
not make the slice half available.

---

## The five probe obligations, and where each is answered

The ratified instrumentation contract requires the probe report to name, with a resolving
`file:line`, five things before any dependent slice runs.

| obligation | answered by | live evidence |
| --- | --- | --- |
| (a) the host hook events and payload fields the observer depends on, corroborated by the repo's own hook code | the `tool_executions` verdict | `.github/hooks/cortex-refresh.json:8`, `scripts/agent-customization/hooks/post-write-reindex-hook.mjs:54` |
| (b) the absence of any documented session-id environment variable | the `session_to_slice_binding` verdict | `scripts/agent-customization/enforcement/runtime-enforcement.mjs:28`, `scripts/agent-customization/enforcement/runtime-enforcement.mjs:29` (the payload-carried session id), `scripts/agent-customization/enforcement/runtime-enforcement.mjs:30`, `scripts/agent-customization/enforcement/runtime-enforcement.mjs:31`, `scripts/agent-customization/enforcement/runtime-enforcement.mjs:32` (the environment fallbacks) |
| (c) the per-slice `token_count` computation site, and the absence of any persistence of it before this contract | the `context_tokens_delivered` verdict | `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:884`, `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:885`, `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs:1167` |
| (d) the absence of any per-slice LLM token-consumption source | the `llm_token_consumption` verdict | `scripts/telemetry/telemetry-contract.mjs:244`, `scripts/telemetry/telemetry-contract.mjs:245`, `scripts/telemetry/telemetry-contract.mjs:246` (declarations only) |
| (e) the existence and the exact limits of the write-tool hook | the `tool_executions` verdict | `scripts/agent-customization/hooks/post-write-reindex-hook.mjs:44`, `scripts/agent-customization/hooks/post-write-reindex-hook.mjs:54` |

---

## What a dependent slice must write

A dependent slice reads the verdict and writes accordingly. A **demonstrated** signal may be
measured; an **unavailable** signal is written as `available: false` with the probe's reason and
**`null`** counters — **never `0`**, because a zero is a measurement and no measurement exists.

    usage:
      token_usage:
        available: false
        source: 'host-model-accounting'
        measures: 'the host model prompt, completion and total token consumption'
        context_tokens_delivered: null
        prompt_tokens: null
        completion_tokens: null
        total_tokens: null
        unavailable_reason: 'the repository has no access to the host agent model accounting'

The same rule governs the tool-execution counter while its observer does not exist, and the
session-to-slice binding: the binding is not a counter, so it is never written as a number at all.

---

## The probe report (verbatim)

The block below is the probe's stdout document, **unchanged**: the same characters in the same order
as the live stdout of `node scripts/telemetry/probe-attribution-sources.mjs --slice-id=OT-S1-3`, in
this file's own line endings. It is the machine-readable form of the verdict table above, and the
focused home asserts that the two agree. Re-paste it from the live stdout whenever the repository
changes, because the probe never rewrites this file.

```json
{
  "schema_version": 1,
  "report": "attribution-sources",
  "slice_id": "OT-S1-3",
  "signals": [
    {
      "signal": "slice_duration",
      "verdict": "demonstrated",
      "owner": "layer B (the repo-owned DONE path); the OpenTelemetry layer never computes it",
      "measures": "the monotonic elapsed interval of the slice dispatch window",
      "summary": "the repository measures a monotonic elapsed interval today: a high-resolution clock read with a wall-clock fallback, and a performance.now() delta in the GPU profiler. The elapsed interval of the SLICE dispatch window itself is computed by the layer-B DONE writer, which this tracker schedules in a later slice; this probe reports the measurement SOURCE that exists on the live bytes, not a slice measurement.",
      "scope": "the elapsed-measurement SOURCE exists on the live bytes; the per-slice dispatch-window duration is NOT measured yet, and the layer-B DONE path owns it",
      "observation": "elapsed-measurement CITED sites on the live bytes: src/acceleration/acceleration.benchmark.ts:150, src/architecture/network/gpu/network.gpu.profiling.ts:229, src/neat/evolve/runtime/evolve.runtime.utils.ts:154",
      "citations": [
        {
          "file": "src/acceleration/acceleration.benchmark.ts",
          "line": 150
        },
        {
          "file": "src/architecture/network/gpu/network.gpu.profiling.ts",
          "line": 229
        },
        {
          "file": "src/neat/evolve/runtime/evolve.runtime.utils.ts",
          "line": 154
        }
      ]
    },
    {
      "signal": "context_tokens_delivered",
      "verdict": "demonstrated",
      "owner": "the workflow MCP (get_slice_context); the persistence owner is the dispatch-time producer",
      "measures": "context tokens DELIVERED to the slice, not model consumption",
      "summary": "the workflow MCP computes a per-slice token_count for the RAG context it delivers: it accumulates the count of every search response, defaults it per response and returns it in the delivered context block. The count is DELIVERED context size, not model consumption, and on the live bytes NOTHING persists it - the value is computed, returned and discarded.",
      "observation": "the probe found no write of the delivered context-token count anywhere in the repository",
      "citations": [
        {
          "file": "scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs",
          "line": 622
        },
        {
          "file": "scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs",
          "line": 884
        },
        {
          "file": "scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs",
          "line": 885
        },
        {
          "file": "scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs",
          "line": 1167
        }
      ]
    },
    {
      "signal": "llm_token_consumption",
      "verdict": "unavailable",
      "owner": "the host agent model accounting, which this repository cannot reach",
      "measures": "the host model prompt, completion and total token consumption",
      "summary": "no per-slice LLM token-consumption source exists on the live bytes",
      "reason": "the repository has no access to the host agent model accounting: no live source READS a model-consumption field (prompt_tokens, completion_tokens, total_tokens) from a host payload, so there is nothing to measure. The fields exist only as DECLARATIONS in the ledger schema, and the nearest total in the repository is its own context-token estimate, which is a different quantity. A dependent slice must write available: false with this reason and null counters - never 0.",
      "observation": "model-consumption fields that are only DECLARED and never read from a host payload: scripts/telemetry/telemetry-contract.mjs:244, scripts/telemetry/telemetry-contract.mjs:245, scripts/telemetry/telemetry-contract.mjs:246; 17 further local token-estimator mentions declare the same field shape and are not host model accounting",
      "citations": [
        {
          "file": "scripts/telemetry/telemetry-contract.mjs",
          "line": 244
        },
        {
          "file": "scripts/telemetry/telemetry-contract.mjs",
          "line": 245
        },
        {
          "file": "scripts/telemetry/telemetry-contract.mjs",
          "line": 246
        }
      ]
    },
    {
      "signal": "tool_executions",
      "verdict": "demonstrated",
      "owner": "the host CLI post-tool surface; the repo-owned observer that counts executions is a later slice",
      "measures": "tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window",
      "summary": "the host tool-execution surface EXISTS and this repository already registers it: a hook manifest registers the post-tool event and the registered hook code already reads the host payload's tool-name field. What does NOT exist yet is the repo-owned OBSERVER that counts executions per slice - the write-tool hook it would sit beside matches write tools only, carries no slice id and writes a timestamped line - so the surface is demonstrated while the per-slice counter is not yet implemented.",
      "scope": "the host post-tool SURFACE exists on the live bytes and is registered; the per-slice execution COUNTER is not implemented yet, and the repo-owned observer owns it",
      "observation": "hook manifests found: .github/hooks/cortex-refresh.json; hook scripts they register: scripts/agent-customization/hooks/post-write-reindex-hook.mjs, scripts/agent-customization/hooks/pre-dispatch-freshness-hook.mjs",
      "citations": [
        {
          "file": ".github/hooks/cortex-refresh.json",
          "line": 8
        },
        {
          "file": "scripts/agent-customization/hooks/post-write-reindex-hook.mjs",
          "line": 54
        },
        {
          "file": "scripts/agent-customization/hooks/post-write-reindex-hook.mjs",
          "line": 44
        }
      ]
    },
    {
      "signal": "session_to_slice_binding",
      "verdict": "unavailable",
      "owner": "the repo, at dispatch time",
      "measures": "the binding of a host session to the slice whose dispatch record is open",
      "summary": "the session-to-slice binding does not exist on the live bytes yet",
      "reason": "no DOCUMENTED session-id environment variable exists - the environment fallbacks the probe found are repository-invented COPILOT_* names - and the dispatch-time producer that would stamp the slice id onto an observation does not exist yet: the transient dispatch record has a declared shape and a writer, but nothing calls that writer at dispatch time. The host payload ALREADY carries a session id, so the session half of the binding has a source; the slice half does not. The binding is therefore owned by the repository at dispatch time, and it is NOT available today.",
      "observation": "payload-carried session-id reads found: scripts/agent-customization/enforcement/runtime-enforcement.mjs:28, scripts/agent-customization/enforcement/runtime-enforcement.mjs:29, scripts/agent-customization/enforcement/runtime-enforcement.mjs:555, scripts/agent-customization/workflow-gap-audit.mjs:205; session-id environment fallbacks found: scripts/agent-customization/enforcement/runtime-enforcement.mjs:30, scripts/agent-customization/enforcement/runtime-enforcement.mjs:31, scripts/agent-customization/enforcement/runtime-enforcement.mjs:32",
      "citations": [
        {
          "file": "scripts/agent-customization/enforcement/runtime-enforcement.mjs",
          "line": 28
        },
        {
          "file": "scripts/agent-customization/enforcement/runtime-enforcement.mjs",
          "line": 29
        },
        {
          "file": "scripts/agent-customization/enforcement/runtime-enforcement.mjs",
          "line": 30
        },
        {
          "file": "scripts/agent-customization/enforcement/runtime-enforcement.mjs",
          "line": 31
        },
        {
          "file": "scripts/agent-customization/enforcement/runtime-enforcement.mjs",
          "line": 32
        },
        {
          "file": "scripts/telemetry/telemetry-contract.mjs",
          "line": 216
        }
      ]
    }
  ]
}
```

---

## Privacy of the published artifact

The probe's persisted copy carries only repo-relative paths, signal names, verdicts and reasons. It
contains **no** timestamp, **no** date, **no** clock time, **no** absolute path and **no** PII, so it
passes the persisted-artifact scan of `scripts/telemetry/telemetry-contract.mjs` unchanged. This
artifact repeats the same values and adds no absolute path and no date of its own.
