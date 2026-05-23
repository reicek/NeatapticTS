# Delegation Tier Enforcement (Agentic Workflow Enforcement Prerequisite)

**Status:** [PLANNED]

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

### Phase 2 — Research [PLANNED]

#### Step 02 — Inventory all agent frontmatter (02-researching) [PLANNED]

```yaml
phase: 2
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-researching.agent.md'
status: '[PLANNED]'
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

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for tier validator and gate (03-red-testing) [PLANNED]

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
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/agent-customization/validate-agent-graph
```

**Step objective:** Write failing tests for:

- `validate-agent-graph.mjs`: given a mock agent with `user-invocable: true` at Tier 3, reports violation.
- `validate-agent-graph.mjs`: given a valid Tier 1 agent with correct `user-invocable: true`, reports no violation.
- `tier-enforcement-gate.mjs`: returns `{ pass: false }` when violations exist, `{ pass: true }` when clean.

### Phase 4 — Implementation [PLANNED]

#### Step 04 — Implement tier inventory, validator, gate, and frontmatter updates (04-implementing) [PLANNED]

```yaml
phase: 4
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementing.agent.md'
status: '[PLANNED]'
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

### Phase 5 — Green validation [PLANNED]

#### Step 05 — Validate tier graph and gate (05-green-testing) [PLANNED]

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

### Phase 6 — Docs [PLANNED]

#### Step 06 — Document tier graph and enforcement policy (06-documenting) [PLANNED]

```yaml
phase: 6
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Delegation_Tier_Enforcement.plans.md'
skills: 'educational-docs'
```

**Step objective:** Add a `## Tier graph` section to the relevant agent skill or
`.github/copilot-instructions.md` describing the 5-layer model, delegation rules,
and how to run the validator. Update `scripts/agent-customization/README.md` (if present)
to include the new scripts.

### Phase 7 — Logging [PLANNED]

#### Step 07 — Session log (07-logging) [PLANNED]

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

Active plan: plans/Delegation_Tier_Enforcement.plans.md [PLANNED]

Goal: Define and enforce the 5-layer agent delegation tier graph for NeatapticTS.

Tier model:
  Tier 1: 00-helping through 07-logging (user-invocable: true)
  Tier 2: Named coordinators / sub-orchestrators (user-invocable: false)
  Tier 3: Hidden scouts and specialists (user-invocable: false)
  Tier 4: Auxiliaries / one-shot helpers (user-invocable: false)

Key artifacts:
  scripts/agent-customization/tier-inventory.mjs          (inventory all .agent.md files)
  scripts/agent-customization/validate-agent-graph.mjs    (enforce tier rules)
  scripts/agent-customization/gates/tier-enforcement-gate.mjs (gate contract)
  .github/agents/*.agent.md                               (add tier: N frontmatter)

Start with Step 01 (01-planning): count .github/agents/*.agent.md files and check
whether validate-agent-graph.mjs already exists before authoring remaining step packets.

Validator:
  node scripts/agent-customization/validate-agent-graph.mjs --json

Plan sync check:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Delegation_Tier_Enforcement.plans.md
```
