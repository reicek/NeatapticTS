# scripts/agent-customization

Operator and agent reference for the NeatapticTS agent customization scripts.
These scripts validate, inventory, and report on the `.github/agents/*.agent.md`
frontmatter surface. They are noninteractive, JSON-capable, and idempotent.

## Script reference

### Tier inventory and validation

| Script | Command | Output |
| --- | --- | --- |
| `tier-inventory.mjs` | `node scripts/agent-customization/tier-inventory.mjs [--json]` | JSON inventory of all agents with tier assignments, delegation edges, and violation list |
| `validate-agent-graph.mjs` | `node scripts/agent-customization/validate-agent-graph.mjs [--json]` | Validates tier rules, `user-invocable` policy, delegation edges, unknown refs, and cycles |
| `tier-audit-report.mjs` | `node scripts/agent-customization/tier-audit-report.mjs [--json\|--markdown]` | Human-readable or JSON audit report of all tier assignments and violations |
| `tier-graph-utils.mjs` | _(library, not a CLI)_ | Shared source for tier constants, agent collection, cycle detection, and delegation edge rules |

### Frontmatter and skill validation

| Script | Command | Output |
| --- | --- | --- |
| `validate-agent-quality.mjs` | `node scripts/agent-customization/validate-agent-quality.mjs [--json]` | Validates agent body structure and `structured-v1` output-contract compliance for all `.agent.md` files |
| `validate-agent-frontmatter.mjs` | `node scripts/agent-customization/validate-agent-frontmatter.mjs [--json]` | Validates required YAML frontmatter fields for all `.agent.md` files |
| `validate-skill-frontmatter.mjs` | `node scripts/agent-customization/validate-skill-frontmatter.mjs [--json]` | Validates required YAML frontmatter fields for all `SKILL.md` files |
| `validate-sdlc-skill-coverage.mjs` | `node scripts/agent-customization/validate-sdlc-skill-coverage.mjs [--json]` | Confirms numbered SDLC orchestrators reference the expected canonical skills |
| `validate-plan-phase-packets.mjs` | `node scripts/agent-customization/validate-plan-phase-packets.mjs [--json]` | Checks that active plan files have correctly structured phase step packets |
| `validate-plan-sync.mjs` | `node scripts/agent-customization/validate-plan-sync.mjs --plan=<path> [--json]` | Validates a single plan's registration in `plans/README.md` and `plans/Roadmap.md`, and emits the linked `downstreamTrackers` for cross-plan handoff visibility |
| `validate-numbered-agent-structured-v1-output.mjs` | `node scripts/agent-customization/validate-numbered-agent-structured-v1-output.mjs [--json]` | Validates that numbered SDLC agents output a correctly structured `structured-v1` block |

### Inventory and reporting

| Script | Command | Output |
| --- | --- | --- |
| `inventory-customizations.mjs` | `node scripts/agent-customization/inventory-customizations.mjs [--json]` | Inventories all customization files (agents, skills, flows, instructions) |
| `generate-agent-skill-routing-table.mjs` | `node scripts/agent-customization/generate-agent-skill-routing-table.mjs [--json]` | Regenerates `.github/agent-skill-routing-table.md` from current agent and skill frontmatter with a normalized source hash |
| `workflow-gap-audit.mjs` | `node scripts/agent-customization/workflow-gap-audit.mjs [--json]` | Audits workflow gaps: gate health, escalation evidence, runtime proof mismatches, and missing pre/post hook pairs |
| `enforcement/runtime-enforcement-context.mjs` | `node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --prepare ...` | Prepares, diagnoses, shows, or clears the repo-owned runtime proof carrier for strict write/execute actions |

### Eval runners

| Script | Command | Output |
| --- | --- | --- |
| `run-skill-trigger-evals.mjs` | `node scripts/agent-customization/run-skill-trigger-evals.mjs [--json]` | Runs should-trigger and should-not-trigger evals for skill descriptions |
| `run-skill-output-evals.mjs` | `node scripts/agent-customization/run-skill-output-evals.mjs [--json]` | Grades skill output quality with evidence-backed assertions |

## Gate scripts (`gates/`)

Gates return a standard contract: `{ pass: boolean, evidence: object, fixHint: string, owner: string }`.

| Gate | Command | Purpose |
| --- | --- | --- |
| `tier-enforcement-gate.mjs` | `node scripts/agent-customization/gates/tier-enforcement-gate.mjs [--json]` | Confirms all agents have valid `tier:` assignments and legal delegation edges |
| `routing-table-freshness.gate.mjs` | `node scripts/agent-customization/gates/routing-table-freshness.gate.mjs [--json]` | Confirms the generated canonical routing table matches current `.github/agents/**` and `.github/skills/**` sources |
| `agent-quality.gate.mjs` | `node scripts/agent-customization/gates/agent-quality.gate.mjs [--json]` | Wraps `validate-agent-quality.mjs`; confirms agent body structure and output-contract compliance |
| `agent-graph.gate.mjs` | `node scripts/agent-customization/gates/agent-graph.gate.mjs [--json]` | Wraps `validate-agent-graph.mjs`; reports unknown refs and cycle violations |
| `plan-sync.gate.mjs` | `node scripts/agent-customization/gates/plan-sync.gate.mjs [--json]` | Confirms active plans are registered in `plans/README.md` and `plans/Roadmap.md` |
| `planning-output-contract.gate.mjs` | `node scripts/agent-customization/gates/planning-output-contract.gate.mjs [--json]` | Validates that planning phase output meets the structured-v1 contract |
| `step-packet.gate.mjs` | `node scripts/agent-customization/gates/step-packet.gate.mjs [--json]` | Checks that authored step packets have all required fields |
| `red-test-confirmation.gate.mjs` | `node scripts/agent-customization/gates/red-test-confirmation.gate.mjs [--json]` | Confirms red-phase test contracts exist before implementation |
| `implementation-artifact-paths.gate.mjs` | `node scripts/agent-customization/gates/implementation-artifact-paths.gate.mjs [--json]` | Confirms all implementation artifact paths declared in a plan exist on disk |
| `green-validation-evidence.gate.mjs` | `node scripts/agent-customization/gates/green-validation-evidence.gate.mjs [--json]` | Confirms green-phase validation evidence was recorded |
| `docs-artifact-reference.gate.mjs` | `node scripts/agent-customization/gates/docs-artifact-reference.gate.mjs [--json]` | Confirms documentation phase artifact references are present |
| `output-resolution-evidence.gate.mjs` | `node scripts/agent-customization/gates/output-resolution-evidence.gate.mjs [--json]` | Confirms session output-resolution artifacts are present |
| `log-completion-marker.gate.mjs` | `node scripts/agent-customization/gates/log-completion-marker.gate.mjs [--json]` | Confirms session log includes a completion marker |
| `stale-wip-plans.gate.mjs` | `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs [--json]` | Detects plans left top-level `[WIP]` after all implementation phases and steps are `[DONE]` |
| `learning-event.gate.mjs` | `node scripts/agent-customization/gates/learning-event.gate.mjs [--json]` | Checks that a learning event was recorded when required |
| `research-findings-evidence.gate.mjs` | `node scripts/agent-customization/gates/research-findings-evidence.gate.mjs [--json]` | Confirms research phase findings evidence was captured |

### Gate exception logging

```sh
node scripts/agent-customization/gates/record-gate-exception.mjs \
  --gate=<gate-name> --failure="<description>"
```

Records a gate exception to `.github/ai-learning/learning-log.jsonl`.
Three consecutive gate failures in a session trigger automatic escalation to `00-helping`.

## Runtime enforcement

See `.github/runtime-enforcement-contract.md` for the canonical strict runtime
enforcement contract, including the repo-owned context carrier, pre/post hook
events, and escalation behavior.

## MCP server scripts (`mcp/`)

These scripts run as stdio MCP servers registered in `.vscode/mcp.json`.

| Server | Registration key | Registered tools |
| --- | --- | --- |
| `neataptic-gate-mcp.mjs` | `neataptic-gate-mcp` | `run_gate_check`, `run_tier_enforcement_gate`, `query_tier_graph`, `query_customization_routing_table` |
| `neataptic-validation-mcp.mjs` | `neataptic-validation-mcp` | Validation surface tools |
| `neataptic-workflow-mcp.mjs` | `neataptic-workflow-mcp` | Workflow surface tools |

The `query_tier_graph` tool (served by `neataptic-gate-mcp`) returns the current tier
inventory, violation list, and summary at runtime. It accepts optional boolean parameters:
- `includeAgents` (default `true`): include per-agent inventory in the response.
- `includeViolations` (default `true`): include the validation issue list in the response.

The `query_customization_routing_table` tool returns the generated canonical routing-table
rows plus freshness status for `.github/agent-skill-routing-table.md`. It accepts:
- `includeRows` (default `true`): include agent and skill row data.
- `includeMarkdown` (default `false`): include the full generated markdown body.

After adding or modifying agent files, restart the `neataptic-gate-mcp` MCP server (or
reload VS Code) for `query_tier_graph` to reflect the updated inventory.

## Agent delegation tier graph

See `.github/copilot-instructions.md` §"Agent delegation tier graph" for the authoritative
5-layer tier model, delegation rules, and operator command reference.

**Quick summary:**

| Tier | Label | `user-invocable` | Count |
| --- | --- | --- | --- |
| 1 | Numbered SDLC Orchestrators (`00-helping` – `07-logging`) | `true` | 8 |
| 2 | Named coordinators / sub-orchestrators | `false` | 10 |
| 3 | Hidden scouts and specialists | `false` | 33 |
| 4 | Auxiliaries and one-shot helpers | `false` | 4 |

**Key rules:**
- Tier 1 → may delegate to Tier 2, 3, or 4.
- Tier 2 → may delegate to Tier 3 or 4.
- Tier 3 → may delegate to Tier 4 only.
- Tier 4 → may not delegate.
- `user-invocable: true` is only valid for Tier 1 agents.

**Validate after any agent change:**

```sh
node scripts/agent-customization/validate-agent-graph.mjs --json
node scripts/agent-customization/gates/tier-enforcement-gate.mjs --json
```

## Test files

Unit tests for these scripts live alongside them and use the
`agent-customization-scripts` Jest project:

```sh
npx jest --config=jest.config.mjs --selectProjects=agent-customization-scripts --no-cache
```

Focused test for tier enforcement:

```sh
npx jest --config=jest.config.mjs --selectProjects=agent-customization-scripts \
  --no-cache --testPathPatterns=scripts/agent-customization/validate-agent-graph
```

## Adding a new agent

1. Create `.github/agents/<name>.agent.md` with valid YAML frontmatter including `tier: <N>`.
2. Choose the correct tier based on whether the agent is a user-invocable SDLC orchestrator
   (Tier 1), a named coordinator (Tier 2), a hidden scout or specialist (Tier 3), or a
   one-shot auxiliary (Tier 4).
3. Ensure `user-invocable: false` for Tiers 2–4.
4. Declare only downward or same-lateral delegation in `agents: [...]` (Tier 4 must have an
   empty `agents:` list or omit it entirely).
5. Declare `skills: [...]` explicitly on every agent, even when the list is empty.
6. Run `node scripts/agent-customization/validate-agent-graph.mjs --json` to confirm zero violations.
