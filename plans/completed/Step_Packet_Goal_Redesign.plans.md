# Step Packet Goal Redesign

**Status:** [DONE]

## Scope

Replace the step packet YAML `agent` and `agent_file` fields with a `goal` field that declares **what outcome the step needs** rather than **who does it**. Add an optional `tdd_sequence` field that signals multi-phase decomposition. Update the orchestrator routing table in `copilot-instructions.md` §3 to map `goal` + `tdd_sequence` combinations to dispatch behavior. Migrate all existing step packets, validation scripts, skills, and flow files.

### Problem

The current `agent: '04-implementing'` field conflates **what the step needs** with **who does it**. The Tier-0 orchestrator reads `agent` and monolithically delegates the entire step to a single numbered agent, even when the step includes a TDD cycle that should decompose across 03-red-testing → 04-implementing → 05-green-testing. This defeats the purpose of having specialist SDLC agents.

### Design

The new step packet YAML replaces `agent` and `agent_file` with `goal`:

```yaml
phase: 2
step: 18
goal: 'implementing' # What outcome this step needs
tdd_sequence: 'red-green' # How the orchestrator should decompose it (optional)
status: '[DONE]' # Example status — not a live WIP step
mode: 'fresh-session'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
next_step: 'Step 19 — Implement relevance feedback'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
specialists:
  - 'cortex-embeddings-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
```

**Stop conditions:** (Example only)

- **Done:** Example block is syntactically correct.
  **Required validation:** (Example only)
- `node scripts/agent-customization/gates/step-packet.gate.mjs --json`

#### Orchestrator Routing Table

When `tdd_sequence` is present, the orchestrator MUST decompose the step across the specified phases. When absent, it is a single-phase dispatch.

| `goal`          | `tdd_sequence` | Orchestrator behavior                                     |
| --------------- | -------------- | --------------------------------------------------------- |
| `planning`      | (none)         | Single dispatch → `01-planning`                           |
| `researching`   | (none)         | Single dispatch → `02-researching`                        |
| `red-testing`   | (none)         | Single dispatch → `03-red-testing`                        |
| `implementing`  | (none)         | Single dispatch → `04-implementing` (tests already exist) |
| `implementing`  | `red-green`    | Three-phase dispatch: 03-red → 04-impl → 05-green         |
| `green-testing` | (none)         | Single dispatch → `05-green-testing`                      |
| `documenting`   | (none)         | Single dispatch → `06-documenting`                        |
| `logging`       | (none)         | Single dispatch → `07-logging`                            |
| `helping`       | (none)         | Single dispatch → `00-helping`                            |

#### Field Definitions

| Field             | Required | Description                                                                                                                                                  |
| ----------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `phase`           | Yes      | Phase number (integer)                                                                                                                                       |
| `step`            | Yes      | Step number within the phase (integer)                                                                                                                       |
| `goal`            | Yes      | What outcome this step needs. Must be one of: `planning`, `researching`, `red-testing`, `implementing`, `green-testing`, `documenting`, `logging`, `helping` |
| `tdd_sequence`    | No       | How the orchestrator should decompose this step across phases. Must be one of: `red-green`, `green-only`. When absent, single-phase dispatch.                |
| `status`          | Yes      | Step status: `[PLANNED]`, `[WIP]`, or `[DONE]`                                                                                                               |
| `mode`            | Yes      | Session mode: `fresh-session` or `perpetual`                                                                                                                 |
| `source_of_truth` | Yes      | Path to the authoritative plan file                                                                                                                          |
| `copy_paste`      | Yes      | Whether the step packet is a paste-ready prompt (`true`/`false`)                                                                                             |
| `next_step`       | Yes      | Description of the next step, or `null` for terminal steps                                                                                                   |
| `skills`          | Yes      | List of skill names the agent should load                                                                                                                    |
| `specialists`     | No       | List of hidden specialist agent names for delegation                                                                                                         |
| `validation`      | Yes      | List of validation commands or evidence gates                                                                                                                |

#### Removed Fields

| Field        | Reason                                                                                                                                                                             |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent`      | Replaced by `goal`. The orchestrator derives the agent from `goal` using the routing table. Backward compatibility has been removed — `agent` is now a deprecated field violation. |
| `agent_file` | Replaced by the routing table. The orchestrator resolves the agent file path from the goal-derived agent name.                                                                     |

### Non-goals

- Do not change flow YAML files — they use `agent:` for flow ownership, not step packet dispatch, and that is a different concern.
- Do not change the runtime enforcement system's `currentAgent` CLI flag — it serves a different purpose (runtime proof carrier) than step packet dispatch.
- Do not change the phase ordering or the fundamental seven-step SDLC structure.
- Do not remove the `mcp-active-binding.plans.md` perpetual binding — it will be migrated in place.

## Current state

All plan files have been migrated from `agent`/`agent_file` to `goal` format. The step-packet gate requires `goal` and rejects `agent` as a deprecated field violation. The orchestrator routing table is in `copilot-instructions.md` §3. The `phase-handoff-workflow` SKILL.md has been updated with the new step packet template. Backward compatibility has been removed. All gates pass. Plan complete.

### Affected files inventory

**Plan files migrated (all `agent`/`agent_file` replaced with `goal`):**

- `plans/mcp-active-binding.plans.md` — 1 step packet ✓
- `plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md` — 26 step packets ✓
- `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md` — 8 active step packets + design notes ✓

**Scripts updated:**

- `scripts/agent-customization/gates/step-packet.gate.mjs` — `REQUIRED_YAML_FIELDS` now includes `goal`, `agent` is rejected as deprecated ✓
- `scripts/agent-customization/gates/step-packet.gate.test.ts` — 15 tests updated for goal-only validation ✓

**Skills updated:**

- `.github/skills/phase-handoff-workflow/SKILL.md` — Step packet shape template updated with `goal`/`tdd_sequence` ✓

**Orchestrator instructions updated:**

- `.github/copilot-instructions.md` — §3 routing table and goal-based dispatch rules added ✓

**Flow files (NOT changed — different concern):**

- `.github/flows/*.flow.yml` — Use `agent:` for flow ownership, not step packet dispatch.

## Coverage backlog

- [x] Update step-packet gate to validate `goal` field and reject `agent` as deprecated
- [x] Update copilot-instructions.md §3 with routing table
- [x] Update phase-handoff-workflow SKILL.md step packet template
- [x] Migrate all plan files from `agent`/`agent_file` to `goal`
- [x] Add backward-compatibility test cases (then removed compat in Phase 2 Step 05)
- [x] Remove backward compatibility from gate
- [x] Final validation and gate checks (Phase 2 Step 06)

## Immediate next steps

None — plan is complete.

## Deferred questions

- ~~Should the backward-compatibility period have a hard deadline?~~ — Resolved: backward compat removed in Phase 2 Step 05.
- Should `tdd_sequence` support any values beyond `red-green` and `green-only` in the initial release?

## Implementation phases

### Phase 1 — Design, scripts, and orchestrator update [DONE]

[DONE] Phase 1: Authored all step packets, updated step-packet gate to require `goal` and reject `agent` as deprecated, added routing table to `copilot-instructions.md` §3, updated `phase-handoff-workflow` SKILL.md step packet template, validated all gates.

### Phase 2 — Plan file migration [DONE]

[DONE] Steps 01-04: Migrated all 3 plan files (`mcp-active-binding`, `NEAT_Genesis_EvoDevo_Core_Readiness`, `Repo_Cortex_Advanced_RAG_Architecture`) from `agent`/`agent_file` to `goal` format.

[DONE] Step 05: Removed backward compatibility from `step-packet.gate.mjs`. All 15 tests pass. Gate correctly requires `goal` and rejects `agent`.

#### Step 06 — Final validation and gate checks [DONE]

```yaml
phase: 2
step: 6
goal: 'green-testing'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Step_Packet_Goal_Redesign.plans.md'
copy_paste: true
next_step: 'null'
skills:
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json'
  - 'node scripts/agent-customization/gates/plan-sync.gate.mjs --json'
  - 'node scripts/agent-customization/gates/agent-graph.gate.mjs --json'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Step_Packet_Goal_Redesign.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
```

**User instruction:** Paste this full step packet.

**Step objective:** Run final validation across all affected plan files and all gate checks to confirm the migration is complete and consistent.

**Stop conditions:**

- **Done:** All gates pass, all plan files validate correctly, no `agent` or `agent_file` fields remain in step packets.
- **Blocked:** A gate reveals a regression that needs a Phase 2 fix.

**Required validation:** All seven validation commands listed above.

## Validation gates

- step-packet gate: validates required YAML fields in WIP step packets
- plan-sync gate: validates plan registration in README and Roadmap
- agent-graph gate: validates agent delegation boundaries

### Latest validation evidence

- Step-packet gate: PASS — 0 violations across 2 WIP step YAML blocks, gate requires `goal`, rejects `agent` as deprecated.
- Plan-sync gate: PASS — all WIP plans registered, 0 missing references.
- Agent-graph gate: PASS — 61 agents, 0 issues, delegation boundaries valid.
- Plan sync (Step_Packet_Goal_Redesign): PASS — 0 errors, 0 warnings.
- Plan sync (mcp-active-binding): PASS — 0 errors, 0 warnings.
- Plan sync (NEAT_Genesis_EvoDevo_Core_Readiness): PASS — 0 errors, 0 warnings.
- Plan sync (Repo_Cortex_Advanced_RAG_Architecture): PASS — 0 errors, 0 warnings.
- No `agent:` or `agent_file:` fields remain in any plan file step YAML blocks.
- All seven validation commands executed and passed. Plan marked [DONE].

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
The Step Packet Goal Redesign plan is at plans/Step_Packet_Goal_Redesign.plans.md.
Phase 1 [DONE], Phase 2 [DONE]. Plan status: [DONE].
All gates PASS. No `agent` or `agent_file` fields remain. Migration complete.
```
