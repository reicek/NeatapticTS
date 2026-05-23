# Repo Cortex MCP Reliability (Repo Cortex — Layer 4)

**Status:** [PLANNED]

> Hardens the Repo Cortex MCP infrastructure before the advanced Embeddings layer (Layer 5).
> Adds a Repo Cortex Scout specialist agent, a durable Repo Cortex workflow skill, a unified
> Cortex lifecycle gate, MCP plan session redirect, docs TSConfig precheck,
> `validate-index.mjs` fixHint enrichment, `build-index.mjs --json-health` output,
> `neataptic-workflow-mcp` per-call `plan_path` override, and a Cortex Embeddings Scout
> specialist agent — with no compromises on scope.
> Sits between the archived Browser Snapshot baseline (Layer 3) and the Embeddings plan (Layer 5).
> Depends on the archived [Semantic_Knowledge_Browser_Snapshot.plans.md](completed/Semantic_Knowledge_Browser_Snapshot.plans.md).

## Purpose

Layers 1–3 of the Repo Cortex system are archived as [DONE]. Before the resource-intensive
Layer 5 (ONNX embeddings + hybrid ranking) begins, this layer closes five reliability gaps
that have emerged in daily use and pre-positions two specialist agents for future Cortex work:

1. **No Cortex-aware specialist agent** — agents that need to rebuild or validate the corpus
   index have no specialist to delegate to. A `Repo Cortex Scout` fills this gap and hands
   off to the `repo-cortex-workflow` skill for durable policy.
2. **No durable Cortex workflow skill** — freshness-check triggers, rebuild sequencing,
   snapshot regeneration policy, and MCP tool validation are scattered across plans and
   comments. A `repo-cortex-workflow` skill consolidates them as a reusable durable workflow.
3. **No unified Cortex lifecycle gate** — `validate-index.mjs` runs standalone but there is
   no gate-contract wrapper that downstream flows can query for `{ pass, evidence, fixHint, owner }`.
   The new `cortex-index.gate.mjs` aggregates index freshness, MCP health, and snapshot
   currency into a single gate that can be listed in the gate catalog and run by any flow.
4. **Weak MCP session binding** — `neataptic-workflow-mcp` binds to a fixed plan path at
   startup. There is no way to redirect the active plan mid-session or supply an override
   per-call without restarting the server. The `plan-session-redirect.mjs` script and a
   per-call `plan_path` override in the MCP server close this gap.
5. **Weak failure diagnostics** — `validate-index.mjs` emits a pass/fail boolean but does not
   emit structured `fixHint` fields, making failure diagnosis manual and slow. The enhancement
   adds explicit `fixHint` templates for stale, missing, and over-age index failures.

Additionally:

- `build-index.mjs --json-health` provides a fast post-build summary for CI and gate evidence.
- `validate-tsconfig-docs.mjs` prevents stale tsconfig drift from breaking the docs pipeline
  before any `npm run docs` run.
- `cortex-embeddings-scout.agent.md` pre-positions a Tier 3 scout for Layer 5 planning.
- Supporting evals and inventory updates keep the agent graph and skill registry current.

## Non-goals

- Changes to `src/` library code.
- ONNX embedding model or vector storage (see `Semantic_Knowledge_Embeddings.plans.md`).
- Browser snapshot changes — `docs/assets/semantic-snapshot.json` is a **generated artifact**
  and is treated as read-only. Never hand-edit it.
- Completed trackers (`plans/completed/**`) are **source-of-truth read-only baselines**.
  Do not modify archived plans or their matching logs.
- Browser-side semantic search changes (pre-ranked snapshot approach is closed for Layer 3).
- NeatChat local retrieval memory (separate plan: `NeatChat_Local_Retrieval_Memory.plans.md`).
- Changing the numbered SDLC orchestrator routing policy in `.github/copilot-instructions.md`
  beyond the additive agent/skill registrations this plan requires.
- Changing the 5-layer tier graph enforcement (see `Delegation_Tier_Enforcement.plans.md`).

## Dependencies

- [Semantic_Knowledge_Foundation.plans.md](completed/Semantic_Knowledge_Foundation.plans.md)
  [DONE] — `data/semantic-index.sqlite` and `scripts/semantic-index/` must be [DONE].
- [Semantic_Knowledge_MCP_Tools.plans.md](completed/Semantic_Knowledge_MCP_Tools.plans.md)
  [DONE] — `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs` and
  `neataptic-cortex-mcp` in `.vscode/mcp.json` must be active.
- [Semantic_Knowledge_Browser_Snapshot.plans.md](completed/Semantic_Knowledge_Browser_Snapshot.plans.md)
  [DONE] — Layer 3 archived; snapshot format stable and read-only.
- [completed/Agentic_Flows_and_Gates_Upgrade.plans.md](completed/Agentic_Flows_and_Gates_Upgrade.plans.md)
  [DONE] — Gate contract shape `{ pass, evidence, fixHint, owner }` is established.
- `scripts/agent-customization/gates/` directory must contain the existing gate pattern
  (e.g., `plan-sync.gate.mjs`) for `cortex-index.gate.mjs` to match.
- `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs` must be readable and its
  `plan_path` binding mechanism must be confirmed before enhancement.
- All existing `.github/agents/*.agent.md` files must pass frontmatter validation before new
  agents are added. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json`
  first.
- [Delegation_Tier_Enforcement.plans.md](Delegation_Tier_Enforcement.plans.md) [PLANNED] —
  Agentic Workflow Enforcement Prerequisite; the 5-layer tier graph must be formally defined
  and `validate-agent-graph.mjs` must pass before new hidden specialist agents are added
  by this plan.
- `validate-agent-graph.mjs` must exist (it does) so new agents can be tier-checked on add.

## Scope

### New artifacts

| Artifact                      | Path                                                      | Notes                                                         |
| ----------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- |
| Repo Cortex Scout agent       | `.github/agents/repo-cortex-scout.agent.md`               | Hidden Tier 3 specialist; `user-invocable: false`             |
| Cortex Embeddings Scout agent | `.github/agents/cortex-embeddings-scout.agent.md`         | Hidden Tier 3 specialist; `user-invocable: false`             |
| Repo Cortex workflow skill    | `.github/skills/repo-cortex-workflow/SKILL.md`            | Durable Cortex workflow: freshness, rebuild, MCP validation   |
| Cortex lifecycle gate         | `scripts/agent-customization/gates/cortex-index.gate.mjs` | Gate contract `{ pass, evidence, fixHint, owner }`            |
| Plan session redirect         | `scripts/agent-customization/plan-session-redirect.mjs`   | Per-session `plan_path` override for `neataptic-workflow-mcp` |
| Docs TSConfig precheck        | `scripts/agent-customization/validate-tsconfig-docs.mjs`  | Validates `tsconfig.docs.json` before docs pipeline runs      |

### Enhanced artifacts

| Artifact            | Path                                                         | Enhancement                                                    |
| ------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| Index validator     | `scripts/semantic-index/validate-index.mjs`                  | Add structured `fixHint` for stale, missing, over-age failures |
| Index builder       | `scripts/semantic-index/build-index.mjs`                     | Add `--json-health` flag: emit post-build summary JSON         |
| Workflow MCP server | `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs` | Add per-call `plan_path` override parameter                    |

### Supporting evals / inventory / gate registration

| Artifact            | Path                                                                     | Notes                                         |
| ------------------- | ------------------------------------------------------------------------ | --------------------------------------------- |
| Skill trigger evals | `.github/skills/repo-cortex-workflow/` (or evals folder)                 | should-trigger and should-not-trigger queries |
| Agent graph recheck | `validate-agent-graph.mjs` run post-add                                  | Confirm new agents pass tier graph            |
| Gate registration   | `scripts/agent-customization/gates/` catalog or `workflow-gap-audit.mjs` | `cortex-index.gate.mjs` visible in audit      |
| Inventory check     | `scripts/agent-customization/inventory-customizations.mjs --json`        | New skill and agents appear                   |

---

### Artifact detail: `cortex-index.gate.mjs`

The unified Cortex lifecycle gate aggregates three sub-checks into one gate contract:

1. **Index freshness** — `validate-index.mjs --json` must return `{ pass: true }`.
2. **MCP health** — `neataptic-workflow-mcp` responds to a lightweight probe (list resources
   or equivalent non-mutating call confirming the server is alive and bound to a plan path).
3. **Snapshot currency** — `docs/assets/semantic-snapshot.json` `generated_at` timestamp
   must not lag the SQLite corpus `mtime` by more than a configurable threshold (default 24h).

Gate output contract (matches existing gate pattern):

```jsonc
{
  "pass": true,
  "evidence": {
    "index_documents": 832,
    "index_fresh": true,
    "mcp_alive": true,
    "snapshot_age_seconds": 0,
  },
  "fixHint": null,
  "owner": "00-helping",
}
```

When any sub-check fails, `pass` is `false` and `fixHint` names the exact remediation:

- Index stale: `"Run: node scripts/semantic-index/build-index.mjs to rebuild stale index"`
- Snapshot stale: `"Run: npm run docs to regenerate snapshot"`
- MCP not reachable: `"Restart neataptic-workflow-mcp server to bind to active plan path"`

Script contract: supports `--json` and `--help`. Noninteractive. Idempotent. Exits 0 when
`pass: true`, exits 1 when `pass: false`.

---

### Artifact detail: `validate-index.mjs` fixHint enrichment

The existing `--json` output is extended to include a `fixHint` field on failure:

```jsonc
{
  "pass": false,
  "documents": 830,
  "chunks": 27000,
  "stale_paths": ["plans/completed/OldPlan.plans.md"],
  "missing_paths": [],
  "over_age_paths": [],
  "fixHint": "Stale paths detected. Run: node scripts/semantic-index/build-index.mjs",
}
```

Priority order for `fixHint`: `stale_paths` > `missing_paths` > `over_age_paths`. Only the
highest-priority template is emitted (the fix for stale resolves all). `fixHint` is `null`
when `pass: true` (no noise on clean runs).

---

### Artifact detail: `build-index.mjs --json-health`

```jsonc
{
  "status": "ok",
  "total_documents": 835,
  "new_documents": 3,
  "removed_documents": 0,
  "elapsed_ms": 4200,
  "index_path": "data/semantic-index.sqlite",
}
```

When the build fails, `status` is `"error"` and a `"message"` field is added. Compatible
with existing `--json` flag (both can coexist; `--json-health` suppresses the per-document
verbose output and emits only the summary).

---

### Artifact detail: `neataptic-workflow-mcp` per-call `plan_path` override

The `loadActivePlanContext` tool (and any other plan-dependent tool) in
`neataptic-workflow-mcp.mjs` accepts an optional `plan_path` input parameter. When provided,
the server loads that plan path for the duration of that call only (stateless override; does
not mutate session state). The startup-binding fixed path remains the default.

**Security constraint:** The `plan_path` parameter is validated against the repository root
before use. Only paths that resolve within the `plans/` directory are accepted. Path traversal
attempts (`../`, absolute paths outside the workspace) are rejected with a structured error.

Lower-priority default: before using the fixed startup binding, the server also checks for
`data/mcp-session-override.json` written by `plan-session-redirect.mjs`. If present and valid,
that path is used instead of the hardcoded startup binding (but still overridden by any
per-call `plan_path` parameter).

---

### Artifact detail: `plan-session-redirect.mjs`

Noninteractive CLI script. Writes a `plan_path` override to `data/mcp-session-override.json`
and reads it back to confirm. Allows redirecting the workflow MCP server's active plan without
a server restart.

```bash
node scripts/agent-customization/plan-session-redirect.mjs \
  --plan=plans/Repo_Cortex_MCP_Reliability.plans.md \
  --json
```

Output on success:

```jsonc
{
  "pass": true,
  "override_written": true,
  "plan_path": "plans/Repo_Cortex_MCP_Reliability.plans.md",
}
```

Supports `--clear` to remove the override file. Supports `--json` and `--help`.
`data/mcp-session-override.json` is gitignored (ephemeral).

---

### Artifact detail: `validate-tsconfig-docs.mjs`

Reads `tsconfig.docs.json`, extracts `include` paths, and checks each path exists on disk.
Emits `{ pass, checked_paths, missing_paths, fixHint }`. Supports `--json`, `--help`.
Intended to run before any `npm run docs` invocation.

```jsonc
{
  "pass": true,
  "checked_paths": ["src/**/*.ts", "examples/**/*.ts"],
  "missing_paths": [],
  "fixHint": null,
}
```

---

### Artifact detail: Repo Cortex Scout frontmatter intent

```yaml
name: Repo Cortex Scout
description: >
  Use when checking Repo Cortex index freshness, triggering a corpus rebuild,
  diagnosing validate-index failures, confirming MCP server binding, or deciding
  whether a semantic index issue belongs to repo-cortex-workflow. Hands off recon
  results to the repo-cortex-workflow skill. Keywords: repo cortex, index freshness,
  validate-index, build-index, cortex MCP, semantic snapshot, cortex lifecycle,
  cortex scout.
user-invocable: false
model: 'Claude Haiku 4.6'
tools:
  [
    'file_search',
    'grep_search',
    'read_file',
    'run_in_terminal',
    'semantic_search',
  ]
```

---

### Artifact detail: Cortex Embeddings Scout frontmatter intent

```yaml
name: Cortex Embeddings Scout
description: >
  Use when mapping ONNX embedding model cache state, embeddings index readiness,
  hybrid BM25+dense ranking gaps, or deciding whether an embeddings issue belongs
  to Semantic_Knowledge_Embeddings. Hands off to the embeddings skill when available.
  Keywords: cortex embeddings, ONNX embeddings, dense retrieval, embed-index,
  sqlite-vec, hybrid ranking, MRR, embeddings scout.
user-invocable: false
model: 'Claude Haiku 4.6'
tools: ['file_search', 'grep_search', 'read_file', 'semantic_search']
```

---

## Implementation phases

### Phase 1 — Planning [PLANNED]

#### Step 01 — Author and verify step packets (01-planning) [PLANNED]

```yaml
phase: 1
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Repo_Cortex_MCP_Reliability.plans.md'
skills: 'tracker-handoff, plan-sync-validation, agent-frontmatter-standards, model-routing-and-budget'
next_step: 'Step 02 — Research MCP/gate/agent infrastructure'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_MCP_Reliability.plans.md
  - node scripts/agent-customization/gates/plan-sync.gate.mjs --json
```

**Step objective:** Confirm all 9 artifact paths listed in the Scope section do not yet
exist. Read `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs` to confirm the
current `plan_path` binding mechanism and where the per-call override would be injected.
Read `scripts/semantic-index/validate-index.mjs` to confirm the existing `--json` output
shape before fixHint enrichment. Read one existing gate (e.g., `plan-sync.gate.mjs`) to
confirm the gate contract pattern matches the design above. Confirm that all step packets
01–07 are valid and their validation commands are reachable. Run plan sync check.

**Stop condition:** Plan sync passes. All step packets confirmed. Research brief notes the
current `validate-index.mjs` `--json` shape and the `neataptic-workflow-mcp.mjs` plan-load
pattern for use in Step 04b.

---

### Phase 2 — Research [PLANNED]

#### Step 02 — Inventory MCP/gate/agent infrastructure (02-researching) [PLANNED]

```yaml
phase: 2
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-researching.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Repo_Cortex_MCP_Reliability.plans.md'
skills: 'agent-frontmatter-standards, agent-script-tooling, mcp-local-server-workflow'
next_step: 'Step 03 — Red tests'
validation:
  - Read scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs (confirm plan_path load site)
  - Read scripts/semantic-index/validate-index.mjs (confirm current --json output shape)
  - Read scripts/semantic-index/build-index.mjs (confirm current CLI flag handling)
  - Read scripts/agent-customization/gates/cortex-mcp-smoke.mjs (existing smoke pattern)
  - Read scripts/agent-customization/gates/plan-sync.gate.mjs (gate contract reference)
  - Confirm .github/agents/repo-cortex-scout.agent.md is absent
  - Confirm .github/agents/cortex-embeddings-scout.agent.md is absent
  - Confirm scripts/agent-customization/gates/cortex-index.gate.mjs is absent
  - Confirm data/mcp-session-override.json is absent (no prior redirect)
```

**Step objective:** Produce a compact research brief confirming:

1. Exact line(s) where `plan_path` is loaded in `neataptic-workflow-mcp.mjs` and the
   proposed injection point for the per-call override.
2. Current `--json` output shape of `validate-index.mjs` — fields present, field `fixHint`
   confirmed absent.
3. How `build-index.mjs` parses CLI args today so `--json-health` matches the existing flag
   parsing pattern.
4. Gate contract shape from `plan-sync.gate.mjs` (field names, exit-code contract).
5. Whether `cortex-mcp-smoke.mjs` covers MCP health in a reusable way or must be wrapped.
6. Proposed `data/mcp-session-override.json` location confirmed free of prior use.

Delegate recon to `Repo Cortex Scout` if it already exists from a prior pass; otherwise
use `02-researching` directly.

**Stop condition:** Research brief authored with concrete findings for each of the 6 points.
Brief is appended as an HTML comment in this plan file for Step 04b reference.

---

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for gate, fixHint, health flag (03-red-testing) [PLANNED]

```yaml
phase: 3
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-testing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Repo_Cortex_MCP_Reliability.plans.md'
skills: 'red-test-contracts, agent-frontmatter-standards, skill-frontmatter-standards'
next_step: 'Step 04a — Implement agents and skill'
validation:
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/validate-index
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/build-index
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/agent-customization/gates/cortex-index
```

**Step objective:** Write failing (red) tests for:

1. `validate-index.mjs` — when a stale path is present, `--json` output includes a
   `fixHint` field (currently absent → red). Place in
   `scripts/semantic-index/__tests__/validate-index.fixhint.test.mjs` or the existing test
   file, matching the existing fixture/mock pattern.
2. `build-index.mjs` — `--json-health` flag emits an object with fields `status`,
   `total_documents`, `new_documents`, `removed_documents`, `elapsed_ms` (flag absent → red).
   Place in `scripts/semantic-index/__tests__/build-index.health.test.mjs`.
3. `cortex-index.gate.mjs` — when all sub-checks pass (mocked), gate returns
   `{ pass: true, evidence: {...}, fixHint: null, owner: "00-helping" }` (file absent → red).
   Place in `scripts/agent-customization/gates/__tests__/cortex-index.gate.test.mjs` (or
   nearest existing test location for gates).

**Conditional skip:** If the `validate-index` or `build-index` tests cannot be isolated from
live SQLite without significant fixture overhead and the existing test suite already has
adequate mock/fixture infrastructure, reuse those fixtures. Do not create a test that requires
a real database rebuild on CI.

**Stop condition:** Tests are written and failing (red). Zero implementation changes to
production code.

---

### Phase 4a — Implement agents and skill (04-implementing) [PLANNED]

#### Step 04a — Create Repo Cortex Scout, Cortex Embeddings Scout, and repo-cortex-workflow skill [PLANNED]

```yaml
phase: 4
step: '4a'
agent: '04-implementing'
agent_file: '.github/agents/04-implementing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Repo_Cortex_MCP_Reliability.plans.md'
skills: 'agent-frontmatter-standards, updating-agent-frontmatter, model-routing-and-budget, skill-frontmatter-standards, creating-specialist-agent'
next_step: 'Step 04b — Implement scripts, gates, and enhancements'
validation:
  - node scripts/agent-customization/validate-agent-frontmatter.mjs --json
  - node scripts/agent-customization/validate-agent-graph.mjs --json
  - node scripts/agent-customization/validate-skill-frontmatter.mjs --json
  - node scripts/agent-customization/inventory-customizations.mjs --json
```

**Step objective:** Create exactly 3 new files:

1. `.github/agents/repo-cortex-scout.agent.md` — Tier 3 hidden specialist.
   Frontmatter: `user-invocable: false`, description from the Scope section above,
   tools: `["file_search", "grep_search", "read_file", "run_in_terminal", "semantic_search"]`,
   model: `Claude Haiku 4.6`. Mode instructions: read Cortex index state, diagnose freshness,
   confirm MCP binding, and hand off a compact brief to the `repo-cortex-workflow` skill.
   Do not implement; reconnaissance only.

2. `.github/agents/cortex-embeddings-scout.agent.md` — Tier 3 hidden specialist.
   Frontmatter: `user-invocable: false`, description from the Scope section above,
   tools: `["file_search", "grep_search", "read_file", "semantic_search"]`,
   model: `Claude Haiku 4.6`. Mode instructions: map embedding model cache state, check
   `data/embeddings.sqlite` presence, and hand off to the embeddings skill.

3. `.github/skills/repo-cortex-workflow/SKILL.md` — Skill file.
   Frontmatter: `name: repo-cortex-workflow`, `user-invocable: false`, description under
   1024 characters (matching scope detail above), argument hint: "Describe the Cortex
   health issue, rebuild trigger, or MCP binding question". Body: concise durable workflow
   covering freshness check sequence, rebuild command sequence, snapshot regeneration
   sequence, gate invocation, and MCP override procedure.

Use `00-helping` (via `helping-agent-maintenance-coordinator`) to review frontmatter
compliance after creation — it owns agent/skill maintenance.

**Stop condition:** 3 new files exist. Frontmatter validation passes for all agents and skill.
Agent graph shows no violations. New agents and skill appear in inventory.

---

### Phase 4b — Implement scripts, gates, and enhancements (04-implementing) [PLANNED]

#### Step 04b — Create gate, scripts, and enhance existing CLI tools and MCP server [PLANNED]

```yaml
phase: 4
step: '4b'
agent: '04-implementing'
agent_file: '.github/agents/04-implementing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Repo_Cortex_MCP_Reliability.plans.md'
skills: 'agent-script-tooling, mcp-local-server-workflow, green-validation-gates'
next_step: 'Step 05 — Green validation'
validation:
  - node scripts/agent-customization/gates/cortex-index.gate.mjs --json
  - node scripts/agent-customization/plan-session-redirect.mjs --plan=plans/mcp-active-binding.plans.md --json
  - node scripts/agent-customization/validate-tsconfig-docs.mjs --json
  - node scripts/semantic-index/validate-index.mjs --json
  - node scripts/semantic-index/build-index.mjs --json-health
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/validate-index
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/build-index
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/agent-customization/gates/cortex-index
```

**Step objective:** Create and enhance exactly 6 artifacts:

1. **`scripts/agent-customization/gates/cortex-index.gate.mjs`** (new) — Unified Cortex
   lifecycle gate. Sub-checks: index freshness (calls `validate-index.mjs --json`), MCP
   health (probe `neataptic-workflow-mcp` via a non-mutating stdio call or PID check),
   snapshot currency (`docs/assets/semantic-snapshot.json` `generated_at` vs SQLite `mtime`).
   Gate output: `{ pass, evidence, fixHint, owner: "00-helping" }`. Supports `--json` and
   `--help`. Noninteractive. Idempotent. Exits 0 on pass, 1 on fail.

2. **`scripts/agent-customization/plan-session-redirect.mjs`** (new) — Writes `plan_path`
   override to `data/mcp-session-override.json`. Reads back to confirm. Supports `--plan=<path>`,
   `--clear`, `--json`, `--help`. Validates that the plan path resolves within `plans/`.
   Noninteractive. `data/mcp-session-override.json` must be gitignored.

3. **`scripts/agent-customization/validate-tsconfig-docs.mjs`** (new) — Reads
   `tsconfig.docs.json`, extracts `include` globs, and checks each resolved path exists on
   disk. Emits `{ pass, checked_paths, missing_paths, fixHint }`. Supports `--json`, `--help`.

4. **`scripts/semantic-index/validate-index.mjs`** (enhanced) — Add `fixHint` field to
   `--json` output on failure. Three fixHint templates per artifact detail above. `fixHint`
   is `null` when `pass: true`. Must not break existing callers of `--json` output (additive
   field only).

5. **`scripts/semantic-index/build-index.mjs`** (enhanced) — Add `--json-health` flag.
   When present, suppress verbose per-document output and emit the summary JSON per artifact
   detail above. Must coexist with existing `--json` flag without conflict.

6. **`scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs`** (enhanced) — Add
   optional `plan_path` parameter to tool calls. Validation: path must resolve within
   `plans/` (no path traversal). Stateless override (does not mutate session state). Also
   read `data/mcp-session-override.json` as lower-priority default before the startup
   binding. Apply the security constraint from the Scope section above.

**Stop condition:** All 6 artifacts created or enhanced. Gate command exits 0 with
`{ pass: true }` (or `pass: false` with valid `fixHint` if the index is stale). Plan session
redirect writes and reads back successfully. Docs TSConfig precheck passes. Build index
`--json-health` emits valid JSON. All three red tests from Step 03 now go green.

---

### Phase 5 — Green validation [PLANNED]

#### Step 05 — Validate all artifacts end-to-end (05-green-testing) [PLANNED]

```yaml
phase: 5
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Repo_Cortex_MCP_Reliability.plans.md'
skills: 'green-validation-gates, agent-frontmatter-standards, skill-frontmatter-standards'
next_step: 'Step 06 — Documentation'
validation:
  - node scripts/agent-customization/validate-agent-frontmatter.mjs --json
  - node scripts/agent-customization/validate-agent-graph.mjs --json
  - node scripts/agent-customization/validate-skill-frontmatter.mjs --json
  - node scripts/agent-customization/inventory-customizations.mjs --json
  - node scripts/agent-customization/gates/cortex-index.gate.mjs --json
  - node scripts/agent-customization/plan-session-redirect.mjs --plan=plans/mcp-active-binding.plans.md --json
  - node scripts/agent-customization/plan-session-redirect.mjs --clear --json
  - node scripts/agent-customization/validate-tsconfig-docs.mjs --json
  - node scripts/semantic-index/validate-index.mjs --json
  - node scripts/semantic-index/build-index.mjs --json-health
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_MCP_Reliability.plans.md
  - node scripts/agent-customization/gates/plan-sync.gate.mjs --json
  - node scripts/agent-customization/workflow-gap-audit.mjs --json
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/validate-index
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/build-index
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/agent-customization/gates/cortex-index
```

**Step objective:** Confirm all 13 acceptance criteria below pass. Specifically:

1. Frontmatter validation clean for all agents and skill.
2. Agent graph no tier violations.
3. Skill frontmatter valid.
4. New agents and skill appear in inventory.
5. `cortex-index.gate.mjs` returns `{ pass: true }` (or valid `{ pass: false, fixHint }` if
   corpus needs rebuild — rebuild it if so, then recheck).
6. Plan session redirect writes and clears without errors.
7. Docs TSConfig precheck passes on the current `tsconfig.docs.json`.
8. `validate-index.mjs --json` includes `fixHint: null` on clean run; verify `fixHint` field
   is present using a forced fixture test from Step 03.
9. `build-index.mjs --json-health` emits expected JSON shape.
10. Plan sync gate passes.
11. Workflow gap audit shows no new gaps introduced.
12. All three Jest test clusters from Step 03 are green.
13. No regressions in existing gate, frontmatter, or graph validators.

**Stop condition:** All 13 points confirmed. No regressions.

---

### Phase 6 — Documentation [PLANNED]

#### Step 06 — Document Cortex reliability infrastructure (06-documenting) [PLANNED]

```yaml
phase: 6
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Repo_Cortex_MCP_Reliability.plans.md'
skills: 'educational-docs, updating-js-docs'
next_step: 'Step 07 — Logging and plan closure'
validation:
  - node scripts/agent-customization/inventory-customizations.mjs --json (confirm skill + agents)
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_MCP_Reliability.plans.md
```

**Step objective:** Make exactly 5 additive documentation changes:

1. Add `Repo Cortex Scout` and `Cortex Embeddings Scout` to the agent list in
   `.github/copilot-instructions.md` (under the skill-first model companion agents section)
   with their routing boundaries: `Repo Cortex Scout` hands off to `repo-cortex-workflow`;
   `Cortex Embeddings Scout` hands off to `Semantic_Knowledge_Embeddings`.
2. Add `repo-cortex-workflow` to the skills section of `.github/copilot-instructions.md`
   with the canonical use-when description and a reference to `cortex-index.gate.mjs`.
3. Update `CLAUDE.md` companion agent table to include `Repo Cortex Scout` → `repo-cortex-workflow`
   and `Cortex Embeddings Scout` → (embeddings skill, planned).
4. Add `cortex-index.gate.mjs` to any gate catalog listing in `scripts/agent-customization/`
   or to the gate reference in `CLAUDE.md`.
5. Update `plans/README.md` trigger-phrase section to add:
   `cortex reliability, cortex scout, cortex lifecycle, plan session redirect, validate-tsconfig-docs,
cortex embeddings scout, embeddings scout, cortex index gate` → this plan.
   (Roadmap.md entry is already registered by this authoring step — no further change needed.)

**Stop condition:** 5 doc targets updated. Inventory confirms new agents and skill visible.
Plan sync passes after trigger phrase update.

---

### Phase 7 — Logging and plan closure (07-logging) [PLANNED]

#### Step 07 — Session log, compress, and archive (07-logging) [PLANNED]

```yaml
phase: 7
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-logging.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Repo_Cortex_MCP_Reliability.plans.md'
skills: 'tracker-handoff, summarizing-session-log, capturing-learning-event'
next_step: 'NONE — proceed to Semantic_Knowledge_Embeddings.plans.md'
validation:
  - Test-Path plans/completed/Repo_Cortex_MCP_Reliability.plans.md
  - Test-Path plans/completed/Repo_Cortex_MCP_Reliability.logs.md
  - node scripts/agent-customization/validate-plan-sync.mjs --json
```

**Step objective:**

1. Compress this plan into a short closed tracker (retain Purpose summary, artifact table,
   and final validation evidence; remove step-by-step detail).
2. Create `plans/completed/Repo_Cortex_MCP_Reliability.logs.md` as the audit record:
   files changed, gates passed, learning events, residual risks.
3. Move both files to `plans/completed/`.
4. Update `plans/README.md` active selection guide: change the link from
   `plans/Repo_Cortex_MCP_Reliability.plans.md` to
   `plans/completed/Repo_Cortex_MCP_Reliability.plans.md` and mark [DONE].
5. Update `plans/Roadmap.md` Repo Cortex Layer 4 entry: mark [DONE].
6. Fire a learning event in `.github/ai-learning/learning-log.jsonl` for any agent/skill/gate
   gaps discovered during execution.

**Stop condition:** Plan archived. Roadmap and README updated to [DONE]. Learning events filed.
`validate-plan-sync.mjs --json` passes for the overall plan surface.

---

## Acceptance criteria and validation gates

| #   | Criterion                                           | Validation command                                                                 | Expected                                               |
| --- | --------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------ | -------- |
| 1   | `repo-cortex-scout.agent.md` exists and valid       | `validate-agent-frontmatter.mjs --json`                                            | `user-invocable: false`, no violations                 |
| 2   | `cortex-embeddings-scout.agent.md` exists and valid | `validate-agent-frontmatter.mjs --json`                                            | `user-invocable: false`, no violations                 |
| 3   | `repo-cortex-workflow/SKILL.md` exists and valid    | `validate-skill-frontmatter.mjs --json`                                            | frontmatter valid, description ≤ 1024 chars            |
| 4   | `cortex-index.gate.mjs` passes                      | `node scripts/agent-customization/gates/cortex-index.gate.mjs --json`              | `{ pass: true }` (or valid `{ pass: false, fixHint }`) |
| 5   | `validate-index.mjs` fixHint on failure             | Jest test (forced-failure fixture)                                                 | `fixHint` field present and non-null                   |
| 6   | `build-index.mjs --json-health` emits summary       | `node scripts/semantic-index/build-index.mjs --json-health`                        | JSON with 5 expected fields                            |
| 7   | Plan session redirect writes and clears             | `plan-session-redirect.mjs --plan=... --json` then `--clear`                       | `{ pass: true, override_written: true }`               |
| 8   | Docs TSConfig precheck passes                       | `node scripts/agent-customization/validate-tsconfig-docs.mjs --json`               | `{ pass: true }`                                       |
| 9   | MCP per-call plan_path override accepted            | MCP tool call with `plan_path` param                                               | No error; loads named plan for call duration           |
| 10  | Agent graph no violations after additions           | `validate-agent-graph.mjs --json`                                                  | `{ violations: [] }`                                   |
| 11  | Plan sync gate passes                               | `node scripts/agent-customization/gates/plan-sync.gate.mjs --json`                 | `{ pass: true }`                                       |
| 12  | All Step 03 Jest tests green                        | `npx jest --testPathPattern=...cortex-index                                        | validate-index                                         | build-index` | All pass |
| 13  | No regressions in existing validators               | `validate-agent-frontmatter`, `validate-skill-frontmatter`, `validate-agent-graph` | All still pass                                         |

---

## Notes on [PLANNED] status and gate behavior

This plan is `[PLANNED]`, not `[WIP]`. The `step-packet.gate.mjs` scan may not flag this plan
during validation runs because the gate may only scan `[WIP]` plans for active step compliance.
This is expected and correct. Once the first session advances Step 01 to `[WIP]`, the gate
will begin scanning active step compliance.

`validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_MCP_Reliability.plans.md` should
pass immediately after registration in `plans/README.md` and `plans/Roadmap.md`.

`plan-sync.gate.mjs --json` verifies the aggregate plan surface and should also pass after
registration — it does not require `[WIP]` status.

---

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active plan: plans/Repo_Cortex_MCP_Reliability.plans.md [PLANNED]

Goal: Harden Repo Cortex MCP infrastructure (Layer 4) before the Embeddings layer (Layer 5).
All step packets are authored. Begin at Step 01 (01-planning) to verify artifact
pre-conditions, confirm step packet coverage, and run plan sync.

New artifacts to create (all absent as of plan authoring):
  .github/agents/repo-cortex-scout.agent.md           (Tier 3, Haiku 4.6, Cortex recon)
  .github/agents/cortex-embeddings-scout.agent.md     (Tier 3, Haiku 4.6, Embeddings recon)
  .github/skills/repo-cortex-workflow/SKILL.md        (durable Cortex workflow skill)
  scripts/agent-customization/gates/cortex-index.gate.mjs   (unified lifecycle gate)
  scripts/agent-customization/plan-session-redirect.mjs      (per-session plan_path redirect)
  scripts/agent-customization/validate-tsconfig-docs.mjs     (docs pipeline TSConfig precheck)

Artifacts to enhance (all exist):
  scripts/semantic-index/validate-index.mjs    add fixHint field on --json failure output
  scripts/semantic-index/build-index.mjs       add --json-health summary flag
  scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs  add per-call plan_path override

Security: plan_path param must validate within plans/ only (no path traversal).
Generated outputs (docs/assets/semantic-snapshot.json, plans/completed/**) are read-only.

Plan sync validators:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_MCP_Reliability.plans.md
  node scripts/agent-customization/gates/plan-sync.gate.mjs --json
```
