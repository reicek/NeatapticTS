# Execute Reference

This file contains extracted reference material for the `execute` skill.
The main `SKILL.md` is the operational playbook; this file holds JSON
schemas, tool name formats, and detailed reference tables that are consulted
on demand but do not need to be loaded into every dispatch decision.

## Dispatch Packet JSON Examples

### `list_dispatchable_agents` response

```json
{
  "agents": [
    {
      "name": "01-planning",
      "tier": 1,
      "model": "...",
      "skills": ["planning"],
      "agents": [],
      "tools": [],
      "userInvocable": true,
      "file": ".github/agents/01-planning.agent.md"
    }
  ],
  "total": 8,
  "tiers": { "1": 8, "2": 11, "3": 42, "4": 4 }
}
```

### `build_dispatch_packet` input

```json
{
  "target_agent": "plan-scout",
  "caller_tier": 1,
  "prompt": "Select the relevant plan. Load context via Cortex MCP / get_slice_context.",
  "context_tier": "default"
}
```

- `target_agent` — string, required. Agent name as declared in the
  frontmatter of a `.github/agents/*.agent.md` file.
- `caller_tier` — integer, required. Tier of the caller: `0` (Agent Zero),
  `1` (phase orchestrator), `2` (coordinator), `3` (specialist), or `4`
  (auxiliary).
- `prompt` — string, optional. Embedded verbatim into the dispatch packet;
  defaults to `""`.
- `context_tier` — string, optional. Either `"default"` or `"long_context"`;
  defaults to `"default"`.

### `build_dispatch_packet` success response

```json
{
  "ok": true,
  "dispatch_allowed": true,
  "reason": "Tier 0 may delegate to Tier 1",
  "agent": {
    "name": "01-planning",
    "tier": 1,
    "tier_label": "User-invocable phase orchestrators",
    "model": "...",
    "skills": ["planning"],
    "agents": [],
    "tools": [],
    "userInvocable": true,
    "file": ".github/agents/01-planning.agent.md"
  },
  "dispatch_packet": {
    "agent_type": "01-planning",
    "name": "01-planning",
    "description": "...",
    "model": "...",
    "prompt": "Select the relevant plan. Load context via Cortex MCP / get_slice_context.",
    "context_tier": "default",
    "skills": ["planning"]
  }
}
```

### `build_dispatch_packet` failure response

```json
{
  "ok": false,
  "dispatch_allowed": false,
  "reason": "Unknown agent 'foo'"
}
```

Common failure reasons:

- `Unknown agent '<name>'` — the target name does not exist in the inventory.
- `caller_tier must be 0, 1, 2, 3, or 4` — the caller tier is out of range.
- `Tier <caller> may not delegate to Tier <target> (target tier must be
greater than caller tier)` — upward or same-tier delegation.
- `userInvocable is only valid for Tier 1 agents; '<name>' is Tier <tier>` —
  the target has `userInvocable: true` but is not Tier 1.
- `Prompt length <n> exceeds the maximum allowed length of <max> characters`
  — the prompt exceeds the prompt-length guard. Use RAG-based dispatch.

### `get_dispatch_policy` response

```json
{
  "allowed_edges": [
    { "from": 0, "to": 1 },
    { "from": 1, "to": 2 },
    { "from": 1, "to": 3 },
    { "from": 1, "to": 4 },
    { "from": 2, "to": 3 },
    { "from": 2, "to": 4 },
    { "from": 3, "to": 4 }
  ],
  "user_invocable_rule": "Only Tier 1 agents may be userInvocable",
  "prompt_length_rule": "Prompts exceeding the maximum length are rejected to enforce RAG-based dispatch.",
  "prompt_length_max": 200,
  "notes": [
    "This server returns a dispatch packet only; it does not spawn subagents.",
    "Tier 0 (orchestrator) may only call Tier 1 agents."
  ]
}
```

### Example `build_dispatch_packet` calls for each Tier-1 agent

When Agent Zero (tier `0`) starts a phase:

```json
{ "target_agent": "00-helping",        "caller_tier": 0, "prompt": "Cross-tier escalation: concurrency limit exceeded. Load context via Cortex MCP / get_slice_context." }
{ "target_agent": "01-planning",       "caller_tier": 0, "prompt": "Plan the next implementation step. Load context via Cortex MCP / get_slice_context." }
{ "target_agent": "02-researching",    "caller_tier": 0, "prompt": "Map prior art for sparse activation functions. Load context via Cortex MCP / get_slice_context." }
{ "target_agent": "03-red-testing",    "caller_tier": 0, "prompt": "Write failing tests for checkpoint save/resume. Load context via Cortex MCP / get_slice_context." }
{ "target_agent": "04-implementing",   "caller_tier": 0, "prompt": "Implement slice 4.2: rolling snapshot. Load context via Cortex MCP / get_slice_context." }
{ "target_agent": "05-green-testing",  "caller_tier": 0, "prompt": "Validate slice 4.2. Load context via Cortex MCP / get_slice_context." }
{ "target_agent": "06-documenting",    "caller_tier": 0, "prompt": "Run docs-quality checks for the recent split. Load context via Cortex MCP / get_slice_context." }
{ "target_agent": "07-logging",        "caller_tier": 0, "prompt": "Compress completed phase 3. Load context via Cortex MCP / get_slice_context." }
```

A Tier-1 orchestrator that delegates downward uses its own tier:

```json
{ "target_agent": "plan-scout", "caller_tier": 1, "prompt": "Select the relevant plan for checkpointing. Load context via Cortex MCP / get_slice_context." }
{ "target_agent": "docs-scout",  "caller_tier": 1, "prompt": "Find README drift in src/neat/selection/ via Cortex MCP." }
```

### Correct vs incorrect dispatch

**Correct:**

```text
neataptic-dispatch-mcp / build_dispatch_packet
  { "target_agent": "04-implementing", "caller_tier": 0,
    "prompt": "Implement slice 4.2: rolling snapshot. Load context via Cortex MCP / get_slice_context." }

→ returns ok: true, dispatch_packet: { agent_type: "04-implementing", ... }

task tool
  agent_type: "04-implementing"
  name: "04-implementing-4-2-rolling-snapshot"
  prompt: "Implement slice 4.2: rolling snapshot. Load context via Cortex MCP / get_slice_context."
```

**Incorrect (workflow violation):**

```text
task tool
  agent_type: "general-purpose"   ← violates the agent definition
  name: "some-helper"
  prompt: "Implement slice 4.2: rolling snapshot. Load context via Cortex MCP / get_slice_context."
```

The second form bypasses the agent's `.agent.md` definition, its allowed
skills, tools, model, and subagent allow-list, and must not be used.

## MCP Tool Name Format (HYPHENS not underscores)

MCP tools are exposed using HYPHENS as separators between the server key
and the tool name. The format is: `<server-key>-<tool-name>`.

The `neataptic-workflow-mcp` server exposes tools as:

- `neataptic-workflow-mcp-get_slice_context`
- `neataptic-workflow-mcp-get_active_workflow_snapshot`
- `neataptic-workflow-mcp-get_customization_inventory`

The `neataptic-gate-mcp` server exposes:

- `neataptic-gate-mcp-run_gate_check`
- `neataptic-gate-mcp-query_tier_graph`
- `neataptic-gate-mcp-query_customization_routing_table`
- `neataptic-gate-mcp-list_gates`
- `neataptic-gate-mcp-get_slice_context`

The `neataptic-validation-mcp` server exposes:

- `neataptic-validation-mcp-get_active_validation_allowlist`
- `neataptic-validation-mcp-run_allowlisted_validation`

The `neataptic-dispatch-mcp` server exposes:

- `neataptic-dispatch-mcp-build_dispatch_packet`
- `neataptic-dispatch-mcp-list_dispatchable_agents`
- `neataptic-dispatch-mcp-get_dispatch_policy`

**NEVER use underscores in MCP tool names.** The separator between server
key and tool name is always a HYPHEN. Calling
`neataptic_workflow_mcp_get_slice_context` (all underscores) will fail — the
correct name is `neataptic-workflow-mcp-get_slice_context` (hyphens).

## Tier 3 Dispatch Capability

Not all Tier 3 agents can dispatch to Tier 4. The `agents` frontmatter field
controls which agents each agent may dispatch. Most Tier 3 agents are scouts
that gather evidence and return directly — they have `agents: []` and no
dispatch capability.

### Tier 3 Agents WITH T4 Dispatch

| Agent | Can dispatch to |
| ----- | --------------- |

### Design Rationale

Scouts are reconnaissance agents: they map boundaries, find patterns, and
return evidence. They do not orchestrate. Only Tier 3 specialists that
produce structured deliverables (quality audit reports, research synthesis
briefs) need T4 auxiliaries to finalize or summarize their output.

When a scout needs a T4 auxiliary, it should return its findings to the
parent orchestrator (Tier 1 or Tier 2), which dispatches the T4 agent
directly. This flat dispatch pattern works within any concurrent limit.

## Fix-Packet YAML Schema

```yaml
fix_packet:
  slice_id: '<slice_id>'
  iteration: <n>
  status: FAILED | REQUEST_CHANGES | OBSERVATIONS
  goal: '<short goal slug>'
  trigger: shared-validation-gate | specialist-review | green-testing
  shared_validation_artifact: '<path/to/artifact.json>' # optional
  observations:
    - source: '<agent or gate name>'
      type: '<classification>'
      detail: '<concise, actionable observation>'
  requested_changes: # only when status is REQUEST_CHANGES
    - '<specific requested change>'
```

### Where to place the block

Insert the block in the plan's `## Latest validation evidence` section,
after the `fix-loop: <slice-id> iteration <n> status=<failed|passed>`
marker that records the iteration. If the plan uses a dedicated
`## Fix packets` section, place the block there and reference the ID from
`## Latest validation evidence`.

### Relationship to RAG-Based Dispatch

Storing fix-loop observations in the active `.plans.md` file and dispatching
only the deterministic fix-packet ID is a **controlled deviation** from the
RAG-Based Dispatch rule that "the orchestrator MUST NOT embed observations
inline in a dispatch prompt." This deviation is permitted **only** when all
of the following are true:

- The observations are stored in the plan under a deterministic fix-packet ID
  (`fix-packet-<slice_id>-iteration-<n>`), where `slice_id` identifies the
  failing slice and the block follows the required YAML schema.
- The receiving agent loads those observations via Cortex RAG
  (`search_context`, `load_document`, or `load_chunk`) using the
  deterministic ID as the primary key, not via inline prompt text.
- The dispatch prompt contains only the fix-packet ID and a minimal
  instruction to load context via RAG.

If the observations cannot be stored in the plan or cannot be retrieved by
RAG, the orchestrator MUST fall back to the standard RAG-Based Dispatch
rule and MUST NOT embed the observations inline in the dispatch prompt.

## Gate Reliability (Consolidated Gate Graceful Degradation)

The consolidated `slice-advancement` gate calls multiple sub-gates
(plan-sync, step-packet, plan-slice-quality, plan-command-lint, and for FULL
slices shared-validation, code-coverage, specialist-review) in a single
invocation. Each sub-gate result is reported in the `sub_gates` array with
`{ name, pass, fixHint, gate_error }`.

There are two distinct failure modes, and the orchestrator MUST treat them
differently:

### Tooling failure — `gate_error: true`

A `gate_error: true` entry means the sub-gate script itself failed to run:
it crashed, timed out, threw an unhandled exception, or produced unparseable
output. This is an infrastructure/tooling problem, NOT a content problem.

When any sub-gate reports `gate_error: true`:

- **Log a warning** and proceed to the next orchestration step.
- **Do NOT retry** the gate or block dispatch.
- **Do NOT treat it as a content failure.** The consolidated gate's top-level
  `pass` boolean is computed only from sub-gates with `gate_error: false`.
- Record the errored gate name(s) in `VALIDATION_EVIDENCE` for awareness,
  but do not loop back or escalate solely because of a tooling error.

### Content failure — `pass: false`, `gate_error: false`

A `pass: false` entry with `gate_error: false` means the sub-gate ran
successfully but found a real issue: a test failure, a lint error, a coverage
gap, or a plan-format violation. This is a content problem that the
implementer or planner must fix.

When any sub-gate reports `pass: false` (and `gate_error: false`):

- **Follow the existing loop-back protocol.** The consolidated gate returns
  `pass: false` and the orchestrator routes the failure back to
  `04-implementing` (or `01-planning` for plan-format issues) with the
  aggregated `fixHint`.
- Record the failing gate(s) and their `fixHint` values in
  `VALIDATION_EVIDENCE`.

### Consolidated gate output shape

```json
{
  "pass": true,
  "sub_gates": [
    {
      "name": "plan-sync",
      "pass": true,
      "fixHint": "...",
      "gate_error": false
    },
    {
      "name": "step-packet",
      "pass": false,
      "fixHint": "Fix ...",
      "gate_error": false
    },
    {
      "name": "plan-slice-quality",
      "pass": true,
      "fixHint": "...",
      "gate_error": false
    }
  ],
  "fixHint": "Failed gates: step-packet. Fix the issues and re-run. ...",
  "evidence": {
    "failedGates": ["step-packet"],
    "erroredGates": []
  }
}
```

The orchestrator MUST use the `sub_gates` array (not just the top-level
`pass`) to determine the correct response: `gate_error: true` entries are
infrastructure warnings, while `pass: false` entries are actionable content
failures.

## Cortex Freshness and File-Lock Hooks

### Auto-Reindex (Post-Write Hook)

Agents do **NOT** need to manually trigger a Cortex reindex after writes.
The post-write reindex hook (`post-write-reindex-hook.mjs`, registered as a
PostToolUse hook in `.github/hooks/cortex-refresh.json`) fires after every
`edit`/`create`/`apply_patch` call, extracts the written file path, and
spawns a fire-and-forget background process that calls
`targeted-reindex.mjs → reindexFiles([filePath])`.

- Only eligible file types (`.md`, `.ts`, `.mjs`, `.js` under `plans/`,
  `.github/skills/`, `.github/agents/`, `src/`, `examples/`,
  `scripts/agent-customization/`, `rag-index/`, `scripts/mcp-semantic/`) are
  reindexed.
- The hook never blocks the host tool and never throws into the host.
- Reindex errors are logged to `artifacts/post-write-reindex.log`.

### Pre-Dispatch Freshness Hook

Before each dispatch (and at SessionStart), the pre-dispatch freshness hook
(`pre-dispatch-freshness-hook.mjs`) checks the index age against a
configurable grace window (default 300s, `CORTEX_GRACE_WINDOW_S`) and
staleness threshold (default 300s, `CORTEX_STALENESS_THRESHOLD_S`). If the
index is stale beyond grace + threshold, a background full reindex is
triggered. The hook **never blocks dispatch**; tooling errors degrade
gracefully. For complex slices, the orchestrator MAY set
`wait_for_reindex: true` to make the hook wait for the reindex to complete.

### File-Lock Tracker

The file-lock tracker (`file-lock-tracker.mjs`) provides `acquire(files)`,
`release(lockId)`, and `isConflict(files)` for serializing parallel slices
with overlapping file sets while parallelizing disjoint slices. Slices that
touch disjoint files run concurrently; slices with overlapping files
serialize via the lock tracker. The tracker is process-scoped (in-memory).

## Chrome DevTools MCP Specialist Quick Reference

Three Chrome DevTools MCP specialists are available for browser-based
measurement. Any agent that needs browser-based measurement, interaction,
or memory profiling should delegate to the appropriate specialist rather
than calling Chrome DevTools MCP tools directly (with the narrow exceptions
listed in the `chrome-devtools-mcp` skill decision tree).

| Specialist                     | When to Call                                                            |
| ------------------------------ | ----------------------------------------------------------------------- |
| `performance-trace-specialist` | Any performance trace capture or analysis                               |
| `browser-ui-specialist`        | Multi-step UI interaction, DOM verification, console/network inspection |
| `browser-memory-specialist`    | Heap snapshots, memory profiling, leak detection                        |

For the full decision trees covering when to use Chrome DevTools MCP
directly versus when to delegate to a specialist, reference the
`chrome-devtools-mcp` skill. That skill owns the tool catalog,
token-efficient strategies, and the trace capture, DOM interaction, and
memory profiling workflows.

## Cortex-First Search Policy (Pointer)

The full Cortex-First Search Policy — search order, enforcement rules, and
gap escalation protocol — lives in the **`research-methodology` skill**.
This file does not duplicate it. Reference `research-methodology` for:

- The required search order (`freshness_check` → `search_corpus` →
  `search_advanced` → `search_context` → `load_chunk` → `load_document` →
  `traverse_graph` → `expand_query` → native tools as last resort)
- Enforcement rules (which agents must include the skill)
- Gap escalation protocol (report → suggest → fallback → flag for
  `00-helping`)
