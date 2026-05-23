# Delegation Tier Enforcement (Agentic Workflow Enforcement Prerequisite)

**Status:** [WIP]

> Defines a formal 5-layer agent delegation tier graph for the NeatapticTS agentic workflow,
> inventories all agent frontmatter against that graph, extends `validate-agent-graph.mjs`,
> adds an MCP gate tool, and establishes an audit/escalation policy.
> Does not depend on the SQLite corpus index; can execute in parallel with Repo Cortex Layers 1–3,
> but **must complete before any plan that adds or reshapes custom agents or skills**,
> including Repo Cortex Layer 4 (`Repo_Cortex_MCP_Reliability.plans.md`).

## Purpose

The NeatapticTS repo uses a complex multi-level agent delegation system (orchestrators,
sub-orchestrators, hidden specialists, skill-owning coordinators, auxiliaries). Without a
formal tier contract and validation gate, delegation chains drift silently — an orchestrator
calls a specialist that calls another orchestrator, tool scopes widen unexpectedly, or
user-invocable flags leak into hidden agents.

This plan defines the **5-layer tier graph**, inventories every `.agent.md` file against it,
extends `scripts/agent-customization/validate-agent-graph.mjs` (or creates it if absent) to
enforce tier assignments, and exposes a live MCP gate tool so agents can query the tier graph
at runtime.

## Non-goals

- Changes to `src/` library code.
- SQLite corpus index or semantic search (separate Repo Cortex layers).
- Changing agent behavior — only frontmatter tier metadata and validation scripts.
- Removing any existing agents.
- Altering the numbered SDLC orchestrator routing policy (that lives in
  `.github/copilot-instructions.md`, which remains authoritative).

## Tier graph definition

| Tier | Label                                  | Examples                                                                      | user-invocable |
| ---- | -------------------------------------- | ----------------------------------------------------------------------------- | -------------- |
| 0    | Default / Main                         | Default VS Code Copilot agent                                                 | —              |
| 1    | Numbered SDLC Orchestrators            | `00-helping` through `07-logging`                                             | Yes            |
| 2    | Named coordinators / sub-orchestrators | `planning-context-coordinator`, `green-test-failure-triage-coordinator`, etc. | No             |
| 3    | Hidden scouts and specialists          | `Boundary Mapper`, `Coverage Scout`, `Plan Scout`, etc.                       | No             |
| 4    | Auxiliaries and one-shot helpers       | `acceptance-criteria-writer`, `docs-example-writer`, etc.                     | No             |

**Delegation rules enforced:**

- Tier 1 agents may call Tier 2, 3, 4.
- Tier 2 agents may call Tier 3, 4.
- Tier 3 agents may call Tier 4.
- Tier 4 agents may not call other agents.
- No tier may call a higher tier except via the `00.cross-tier-helper` escalation path.
- `user-invocable: true` is only valid for Tier 1 agents.
- All non-Tier-1 agents must have `user-invocable: false`.

## Dependencies

- All `.github/agents/*.agent.md` files must exist and be parseable YAML frontmatter.
- `scripts/agent-customization/` directory and existing scripts (reuse patterns).
- MCP server registration in `.vscode/mcp.json` for the gate tool (additive).
- Does not require SQLite corpus index.
- **Must complete before** [Repo_Cortex_MCP_Reliability.plans.md](Repo_Cortex_MCP_Reliability.plans.md)
  begins — that plan adds new hidden specialist agents which must pass tier graph validation.

## Scope

### Artifacts

| Artifact              | Path                                                          | Notes                                               |
| --------------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| Tier inventory script | `scripts/agent-customization/tier-inventory.mjs`              | Reads all .agent.md files; emits tier assignments   |
| Agent graph validator | `scripts/agent-customization/validate-agent-graph.mjs`        | Enforce tier rules; exit non-zero on violation      |
| Tier enforcement gate | `scripts/agent-customization/gates/tier-enforcement-gate.mjs` | Gate contract: `{ pass, evidence, fixHint, owner }` |
| MCP gate tool         | `scripts/agent-customization/mcp/cortex-tier-tool.mjs`        | Exposes `query_tier_graph` MCP tool                 |
| Audit report          | `scripts/agent-customization/tier-audit-report.mjs`           | Emits full tier inventory as JSON/markdown          |
| Frontmatter updates   | `.github/agents/*.agent.md`                                   | Add `tier:` field to frontmatter where missing      |

### `tier-inventory.mjs` output contract

```jsonc
{
  "generated_at": "<ISO timestamp>",
  "agents": [
    {
      "file": ".github/agents/01-planning.agent.md",
      "name": "01-planning",
      "tier": 1,
      "user_invocable": true,
      "delegates_to": ["02-researching", "03-red-testing", "..."],
      "violations": [],
    },
  ],
  "violations": [],
  "summary": { "total": 42, "by_tier": { "1": 8, "2": 12, "3": 15, "4": 7 } },
}
```

### `validate-agent-graph.mjs` contract

- Reads `tier-inventory.mjs` output.
- For each agent, checks that all `agents: [...]` allow-list entries are lower-tier.
- Checks that `user-invocable: true` only appears on Tier 1 agents.
- Exits 0 when no violations; exits 1 with violation report when violations exist.
- Supports `--json` and `--help`.

### MCP gate tool: `query_tier_graph`

Exposes a tool that returns the current tier inventory, violations, and summary. Registered
with an existing MCP server (e.g., `neataptic-validation-mcp`) or the new `neataptic-cortex-mcp`
depending on what Step 01 determines is the cleaner fit.

### Escalation policy

When `validate-agent-graph.mjs` detects a violation:

1. Report violation with `file`, `tier`, `rule`, and `fixHint`.
2. Three consecutive violations in a session escalate to `00-helping` via `00.cross-tier-helper`.
3. Record violations as learning events in `.github/ai-learning/learning-log.jsonl`.

## Implementation phases

### Phase 1 — Planning [PLANNED]

#### Step 01 — Author step packets (01-planning) [PLANNED]

```yaml
phase: 1
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Delegation_Tier_Enforcement.plans.md'
skills: 'tracker-handoff, agent-frontmatter-standards, agent-inventory-audit'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Delegation_Tier_Enforcement.plans.md
```

**Step objective:** Read all `.github/agents/*.agent.md` files to count agents and identify
whether `validate-agent-graph.mjs` already exists. Confirm that `tier:` frontmatter fields are
absent (requiring addition). Author Step 02 through Step 07 packets with concrete targets.

### Phase 2 — Research [DONE]

#### Step 02 — Inventory all agent frontmatter (02-researching) [DONE]

```yaml
phase: 2
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-researching.agent.md'
status: '[DONE]'
mode: 'sequential'
source_of_truth: 'plans/Delegation_Tier_Enforcement.plans.md'
skills: 'agent-inventory-audit, agent-frontmatter-standards'
validation:
  - Count .github/agents/*.agent.md files
  - Read each file for name, user-invocable, agents allow-list
  - Identify agents with user-invocable: true
  - Map current delegation chains against tier graph
```

**Step objective:** Produce a complete agent inventory with proposed tier assignments. Flag any
agents where `user-invocable: true` is set but the agent is not a Tier 1 SDLC orchestrator.
Hand off the inventory brief to Step 04 (implementing).

**Step 02 findings (completed 2026-05-23):**

- **Total agents:** 55 `.github/agents/*.agent.md` files confirmed.
- **`tier:` YAML frontmatter field:** ABSENT from all 55 agents. All `TIER:` occurrences are
  inside structured-v1 output-contract body text, NOT YAML frontmatter.
- **`user-invocable: true`:** Exactly 8 agents (00-helping through 07-logging). No non-Tier-1
  agent has `user-invocable: true`. Zero violations on this rule.
- **`validate-agent-graph.mjs`:** EXISTS. Currently checks: unknown subagent refs and cycles.
  Does NOT check tier constraints or user-invocable rules. Must be EXTENDED (not created).
- **`agent-graph.gate.mjs`:** EXISTS in `gates/`. Wraps `validate-agent-graph.mjs`.
- **`neataptic-gate-mcp.mjs`:** EXISTS. Already exposes `agent-graph` gate. `query_tier_graph`
  should be added here (not a new server). `cortex-tier-tool.mjs` = module imported by this
  server.
- **`.vscode/mcp.json`:** HAS 4 registered servers: `neataptic-workflow-mcp`,
  `neataptic-validation-mcp`, `neataptic-gate-mcp`, `neataptic-cortex-mcp`. No new server
  entry needed.
- **Jest coverage for scripts/agent-customization/:** NOT in `jest.config.mjs`. Step 03 must
  add a new `agent-customization-scripts` project (like `semantic-index-scripts` pattern).
  Test files → `scripts/agent-customization/*.test.ts`.
- **Plan contract conflict:** The plan `tier-inventory.mjs` output contract shows `total: 42`;
  correct count is **55**. Step 04 must use 55.
- **Proposed tier map:** See tier assignment table in the Step 02 output block below.

**Proposed tier summary:** Tier 1: 8, Tier 2: 10, Tier 3: 33, Tier 4: 4 (total 55).

**Tier 2 agents (10):** planning-context-coordinator, planning-risk-coordinator,
planning-test-strategy-coordinator, research-codebase-coordinator,
implementation-pattern-coordinator, green-test-failure-triage-coordinator,
helping-gap-resolution-coordinator, helping-agent-maintenance-coordinator,
solid-split, flappy-architecture-polish.

**Tier 4 agents (4):** acceptance-criteria-writer, docs-example-writer,
learning-event-capturer, file-change-summarizer.

**All remaining 33 agents:** Tier 3.

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for tier validator and gate (03-red-testing) [DONE]

```yaml
phase: 3
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-testing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Delegation_Tier_Enforcement.plans.md'
skills: 'red-test-contracts'
validation:
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/validate-agent-graph
```

**Step objective:** Write failing tests for:

- `validate-agent-graph.mjs`: given a mock agent with `user-invocable: true` at Tier 3, reports violation.
- `validate-agent-graph.mjs`: given a valid Tier 1 agent with correct `user-invocable: true`, reports no violation.
- `tier-enforcement-gate.mjs`: returns `{ pass: false }` when violations exist, `{ pass: true }` when clean.

**Step 03 findings (completed 2026-05-23):**

- Added Jest project `agent-customization-scripts` to make `scripts/agent-customization/**/*.test.ts` runnable without widening other script projects.
- Added `scripts/agent-customization/validate-agent-graph.red.test.ts` with four executable contracts:
  - non-Tier-1 `user-invocable: true` must be reported as a violation,
  - Tier 1 `user-invocable: true` must remain clean,
  - upward Tier 3 -> Tier 2 delegation must be reported as a violation,
  - `tier-enforcement-gate.mjs` must return a structured failing gate report for a violating fixture workspace.
- Exact red validation command:
  - `npx jest --config=jest.config.mjs --selectProjects=agent-customization-scripts --no-cache --testPathPatterns=scripts/agent-customization/validate-agent-graph`
- Red evidence:
  - `npx jest --config=jest.config.mjs --selectProjects=agent-customization-scripts --no-cache --testPathPatterns=scripts/agent-customization/validate-agent-graph` exited `1` with `3` failing tests and `1` passing control test,
  - `validate-agent-graph.mjs` returned `ok: true` with `issues: []` for both the Tier 3 `user-invocable: true` fixture and the Tier 3 -> Tier 2 delegation fixture because tier enforcement is not implemented yet,
  - `scripts/agent-customization/gates/tier-enforcement-gate.mjs` is still missing, and the red test captured the current `MODULE_NOT_FOUND` failure as `report: null` instead of structured `{ pass: false, ... }` output.
- Step 04 green condition:
  - extend `validate-agent-graph.mjs` to enforce tier and `user-invocable` rules while preserving unknown-ref and cycle checks,
  - add `scripts/agent-customization/gates/tier-enforcement-gate.mjs` with the standard gate contract,
  - rerun the exact Step 03 Jest command until all four contracts pass.

### Phase 4 — Implementation [DONE]

#### Step 04 — Implement tier inventory, validator, gate, and frontmatter updates (04-implementing) [DONE]

```yaml
phase: 4
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementing.agent.md'
status: '[DONE]'
mode: 'sequential'
source_of_truth: 'plans/Delegation_Tier_Enforcement.plans.md'
skills: 'agent-frontmatter-standards, updating-agent-frontmatter, agent-script-tooling'
validation:
  - node scripts/agent-customization/tier-inventory.mjs --json
  - node scripts/agent-customization/validate-agent-graph.mjs --json
  - node scripts/agent-customization/gates/tier-enforcement-gate.mjs --json
```

**Step objective:** Implement all artifacts:

1. `tier-inventory.mjs` — reads all `.agent.md` files, assigns tiers, emits JSON.
2. Add `tier: <N>` frontmatter field to all `.github/agents/*.agent.md` files per the tier graph.
3. `validate-agent-graph.mjs` — enforce tier rules, `user-invocable` constraints, delegation chain depth.
4. `tier-enforcement-gate.mjs` — gate contract `{ pass, evidence, fixHint, owner }`.
5. `tier-audit-report.mjs` — human-readable markdown report of all tiers + violations.
6. `cortex-tier-tool.mjs` — MCP tool handler for `query_tier_graph`.

**Step 04 findings (completed 2026-05-23):**

- Added `scripts/agent-customization/tier-graph-utils.mjs` as the shared tier-policy source for inventory collection, expected tier mapping, cycle detection, and legal delegation edges.
- Extended `scripts/agent-customization/validate-agent-graph.mjs` into an importable helper surface with CLI mode preserved. The validator now enforces `tier:` frontmatter, exact Tier 1 user-invocable policy, Tier 4 no-delegation policy, unknown refs, cycles, and legal tier edges.
- Added `scripts/agent-customization/tier-inventory.mjs` and `scripts/agent-customization/tier-audit-report.mjs` to expose JSON inventory plus a human-readable markdown audit report.
- Added `scripts/agent-customization/gates/tier-enforcement-gate.mjs` and upgraded `scripts/agent-customization/gates/agent-graph.gate.mjs` to use the importable validator instead of child-process path assumptions.
- Added `scripts/agent-customization/mcp/cortex-tier-tool.mjs` and wired `query_tier_graph` into `scripts/agent-customization/mcp/neataptic-gate-mcp.mjs`; no new MCP server registration was required because `.vscode/mcp.json` already registers `neataptic-gate-mcp`.
- Added `tier:` YAML frontmatter to all 55 `.github/agents/*.agent.md` files. Final enforced distribution matches the Step 02 proposal: Tier 1 = 8, Tier 2 = 10, Tier 3 = 33, Tier 4 = 4.
- Resolved the only repo-real tier conflict by removing the `implementation-pattern-coordinator -> solid-split` allow-list edge. This preserved the planned Tier 2 placement for `solid-split` while eliminating the same-tier delegation violation.
- Explicitly treated lowercase frontmatter `tier:` as the delegation-tier policy field and left uppercase structured-contract `TIER:` body fields untouched. The audit report now records that distinction to avoid semantic drift.

**Step 04 validation evidence:**

- `npx jest --config=jest.config.mjs --selectProjects=agent-customization-scripts --no-cache --testPathPatterns=scripts/agent-customization/validate-agent-graph` -> PASS (`4` tests, `1` suite)
- `node scripts/agent-customization/tier-inventory.mjs --json` -> PASS (`total=55`, `tier1=8`, `tier2=10`, `tier3=33`, `tier4=4`, `violations=0`)
- `node scripts/agent-customization/validate-agent-graph.mjs --json` -> PASS (`0` errors, `0` warnings)
- `node scripts/agent-customization/gates/tier-enforcement-gate.mjs --json` -> PASS (`pass: true`, `issueCount: 0`)
- `node scripts/agent-customization/tier-audit-report.mjs` -> PASS (markdown report emitted, `Violations: 0`)
- `node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json` -> PASS (`toolCount: 3`, `query_tier_graph` self-check clean)

### Phase 5 — Green validation [DONE]

#### Step 05 — Validate tier graph and gate (05-green-testing) [DONE]

```yaml
phase: 5
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Delegation_Tier_Enforcement.plans.md'
skills: 'green-validation-gates'
validation:
  - node scripts/agent-customization/tier-inventory.mjs --json
  - node scripts/agent-customization/validate-agent-graph.mjs --json
  - node scripts/agent-customization/gates/tier-enforcement-gate.mjs --json
  - node scripts/agent-customization/tier-audit-report.mjs
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/agent-customization/validate-agent-graph
```

**Step objective:** Confirm tier inventory runs cleanly, no violations reported for the full
agent set after frontmatter updates, gate passes, unit tests green.

**Step 05 findings (completed 2026-05-23):**

- Re-ran the full Step 05 validation slice from the post-Step-04 repo state. The tier inventory,
  validator, gate, and audit report all remained clean with the enforced 55-agent distribution of
  Tier 1 = 8, Tier 2 = 10, Tier 3 = 33, Tier 4 = 4 and zero violations.
- Re-ran the focused `agent-customization-scripts` Jest project for
  `validate-agent-graph.red.test.ts`; all four tier-enforcement contracts remained green.
- Re-ran the MCP server self-check as an additional focused validation beyond the Step 05 command
  list. The server still reports `query_tier_graph` on the in-process tool surface with zero
  errors or warnings.
- Coverage guard is not applicable for this step because the implemented change set is limited to
  agent frontmatter and customization scripts; no `src/` production files were changed.
- The only remaining operational caveat is external to the repo-local checks: VS Code's live MCP
  tool registry may still require restarting `neataptic-gate-mcp` or reloading the editor before
  `query_tier_graph` appears in the interactive tool picker.

**Step 05 validation evidence:**

- `node scripts/agent-customization/tier-inventory.mjs --json` -> PASS (`total=55`, `tier1=8`,
  `tier2=10`, `tier3=33`, `tier4=4`, `violations=0`)
- `node scripts/agent-customization/validate-agent-graph.mjs --json` -> PASS (`0` errors,
  `0` warnings)
- `node scripts/agent-customization/gates/tier-enforcement-gate.mjs --json` -> PASS
  (`pass: true`, `issueCount: 0`)
- `node scripts/agent-customization/tier-audit-report.mjs` -> PASS (`Violations: 0`)
- `npx jest --config=jest.config.mjs --selectProjects=agent-customization-scripts --no-cache --testPathPatterns=scripts/agent-customization/validate-agent-graph`
  -> PASS (`4` tests, `1` suite)
- `node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json` -> PASS
  (`toolsTested: run_gate_check, query_tier_graph`, `toolCount: 3`, `0` errors, `0` warnings)

### Phase 6 — Docs [DONE]

#### Step 06 — Document tier graph and enforcement policy (06-documenting) [DONE]

```yaml
phase: 6
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[DONE]'
mode: 'sequential'
source_of_truth: 'plans/Delegation_Tier_Enforcement.plans.md'
skills: 'educational-docs'
```

**Step objective:** Add a `## Tier graph` section to the relevant agent skill or
`.github/copilot-instructions.md` describing the 5-layer model, delegation rules,
and how to run the validator. Update `scripts/agent-customization/README.md` (if present)
to include the new scripts.

**Step 06 findings (completed 2026-05-23):**

- `scripts/agent-customization/README.md` did NOT exist. Created it as the operator-facing
  reference for all scripts in that directory. It covers: tier-inventory scripts, frontmatter
  and skill validators, inventory/reporting scripts, gate scripts with their contracts, MCP
  server registration and `query_tier_graph` tool, tier graph quick-summary table, per-tier
  delegation rules, validate-after-change commands, test file reference, and adding-a-new-agent
  checklist.
- Added `### Agent delegation tier graph` section to `.github/copilot-instructions.md` after
  "Flow, gate, and universal-helper routing policy". The new section defines the 5-layer tier
  model, delegation direction rules, the `user-invocable` constraint, enforced counts, and
  the four operator commands (`tier-inventory.mjs`, `validate-agent-graph.mjs`,
  `tier-enforcement-gate.mjs`, `tier-audit-report.mjs`). Also notes MCP `query_tier_graph`
  for live runtime queries.
- Source-of-truth decision: `.github/copilot-instructions.md` is the authoritative policy
  surface (always-on context); `scripts/agent-customization/README.md` is the operator
  reference for the script surface. The README cross-links to copilot-instructions.md for the
  authoritative tier model rather than duplicating it.
- No generated artifacts were edited. No `npm run docs` run was required (changes are in
  `.github/` and `scripts/`, not in `src/`).

**Step 06 validation evidence:**

- `grep "Agent delegation tier graph" .github/copilot-instructions.md` -> PASS (2 matches at
  lines 48 and 50)
- `scripts/agent-customization/README.md` created and content-verified in editor.

### Phase 7 — Logging [WIP]

#### Step 07 — Session log (07-logging) [WIP]

```yaml
phase: 7
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-logging.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Delegation_Tier_Enforcement.plans.md'
skills: 'tracker-handoff, summarizing-session-log'
```

## Acceptance criteria and validation gates

| Gate                                | Command                                                                       | Expected                                     |
| ----------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------- |
| Tier inventory emits JSON           | `node scripts/agent-customization/tier-inventory.mjs --json`                  | Valid JSON; all agents assigned tiers        |
| Validator clean                     | `node scripts/agent-customization/validate-agent-graph.mjs --json`            | `{ violations: [] }` or exit 0               |
| Gate passes                         | `node scripts/agent-customization/gates/tier-enforcement-gate.mjs --json`     | `{ pass: true }`                             |
| Unit tests green                    | `npx jest --testPathPattern=scripts/agent-customization/validate-agent-graph` | All pass                                     |
| No Tier 3/4 user-invocable          | Output of `tier-inventory.mjs`                                                | `user-invocable: true` only on Tier 1 agents |
| All 8 SDLC orchestrators are Tier 1 | Inventory                                                                     | `00-helping` through `07-logging` all Tier 1 |

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active plan: plans/Delegation_Tier_Enforcement.plans.md [WIP]

Goal: Define and enforce the 5-layer agent delegation tier graph for NeatapticTS.

Tier model:
  Tier 1: 00-helping through 07-logging (user-invocable: true)
  Tier 2: Named coordinators / sub-orchestrators (user-invocable: false)
  Tier 3: Hidden scouts and specialists (user-invocable: false)
  Tier 4: Auxiliaries / one-shot helpers (user-invocable: false)

Step 01 through Step 05 are complete and tracked in this plan. Start with Step 06 (06-documenting).

Current validated state:
  - 55 .github/agents/*.agent.md files all have frontmatter tier: N values
  - enforced tier counts are 8 / 10 / 33 / 4 across Tiers 1 / 2 / 3 / 4
  - implementation-pattern-coordinator no longer delegates to solid-split
  - query_tier_graph is implemented on the existing neataptic-gate-mcp server surface
  - Step 05 validation reran cleanly, including MCP self-check reporting query_tier_graph
  - live VS Code tool discovery for query_tier_graph may still require restarting neataptic-gate-mcp or reloading VS Code

Key artifacts:
  scripts/agent-customization/tier-graph-utils.mjs
  scripts/agent-customization/tier-inventory.mjs
  scripts/agent-customization/validate-agent-graph.mjs
  scripts/agent-customization/gates/tier-enforcement-gate.mjs
  scripts/agent-customization/tier-audit-report.mjs
  scripts/agent-customization/mcp/cortex-tier-tool.mjs
  scripts/agent-customization/mcp/neataptic-gate-mcp.mjs
  .github/agents/*.agent.md

Validator:
  node scripts/agent-customization/validate-agent-graph.mjs --json

Step 06 target:
  - document the 5-layer tier model, delegation rules, and validator usage in the authoritative agent workflow docs
  - update scripts/agent-customization/README.md only if that README exists and is the right source-of-truth for these scripts

Plan sync check:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Delegation_Tier_Enforcement.plans.md
```
