# Orchestration System Optimization (Mini-Agent Transition)

**Status:** [WIP]

## Scope

Optimize the NeatapticTS agent hierarchy to ensure that all substantive work is delegated to targeted "mini agents" (Tier 2-4). This involves eliminating "God-agent" behavior in Tier 1 orchestrators, extracting durable policies into skills, and introducing specialized executors.

**Key Requirements:**

- Tier 0 and 1 must delegate ALL substantive work.
- Every Tier 1 orchestrator must have ≥ 3 dedicated specialists.
- Maximize use of MCP, hooks, and flows.
- Ensure every agent has complementary skills for its instructions.

## MCP tracking plan

```yaml
workstream: orchestration_system_optimization
source_reference: plans/completed/Agentic_Workflow_Architecture.plans.md
active_tracker: plans/Orchestration_System_Optimization.plans.md
primary_boundary: tier_1_delegation_and_skill_extraction
reason:
  - 'Tier 1 orchestrators must delegate all substantive work to Tier 2-4 specialists per the mini-agent architecture.'
  - 'Durable policies currently reside in CLAUDE.md and copilot-instructions.md instead of skill files.'
  - '04-implementing lacks a Tier 2 executor for actual file edits, creating substantive work leakage.'
  - '02-researching and 01-planning lack dedicated skills for research methodology and synthesis.'
preserve_terms:
  - mini-agent
  - Tier 1 orchestrator
  - Tier 2 executor
  - Tier 3 specialist
  - skill-first
  - delegation tier
  - agent frontmatter
  - routing table
  - agent-graph gate
mcp_services:
  workflow:
    - neataptic-workflow-mcp.get_active_workflow_snapshot
    - neataptic-workflow-mcp.get_customization_inventory
  cortex:
    - neataptic-cortex-mcp.search_corpus
    - neataptic-cortex-mcp.freshness_check
  gates:
    - neataptic-gate-mcp.list_gates
    - neataptic-gate-mcp.run_gate_check
    - neataptic-gate-mcp.query_customization_routing_table
    - neataptic-gate-mcp.query_tier_graph
  validation:
    - neataptic-validation-mcp.get_active_validation_allowlist
    - neataptic-validation-mcp.run_allowlisted_validation
specialist_delegation:
  research:
    - Plan Scout
    - Boundary Mapper
    - Agent Inventory Auditor
    - Repo Cortex Scout
  implementation:
    - 04-implementing
    - Helping Agent Maintenance Coordinator
    - Creating Specialist Agent
  validation:
    - 05-green-testing
    - Green Validation Gates
    - Coverage Guard
  escalation:
    - '00-helping only when an MCP/tool/agent/flow gap blocks the active step.'
non_goals:
  - 'Do not change the number of Tier 1 orchestrators; this optimization works within the existing eight SDLC agents.'
  - 'Do not archive or modify plans/completed/Agentic_Workflow_Architecture.plans.md; it remains the upstream authority.'
  - 'Do not create new Tier 1 agents; all new agents must be Tier 2-4 specialists.'
  - 'Do not modify MCP server implementations; this workstream only updates agent frontmatter, skills, and flows.'
acceptance_criteria:
  - id: skill_extraction
    criterion: 'Given the critical gaps identified, when skill extraction completes, then implementation-standards, research-methodology, and routing-optimization-policy skills exist with proper frontmatter and durable policy content.'
    validation: 'skill files present with skills: frontmatter field'
  - id: specialist_creation
    criterion: 'Given the new skills are authored, when specialist authoring completes, then implementation-executor (Tier 2), research-synthesis-specialist (Tier 3), and code-quality-auditor (Tier 3) agent files exist with correct tier assignments.'
    validation: 'agent frontmatter tier field and agent-graph gate'
  - id: routing_sync
    criterion: 'Given new agents and skills are created, when routing sync completes, then all affected .agent.md files have updated skills: and agents: frontmatter fields.'
    validation: 'npm run agents:routing-table and routing-table-freshness gate'
  - id: delegation_enforcement
    criterion: 'Given the optimization is complete, when tier enforcement is validated, then no Tier 1 agent performs work that should be delegated to Tier 2/3.'
    validation: 'tier-enforcement gate and query_tier_graph MCP tool'
  - id: flow_integration
    criterion: 'Given agents and routing are updated, when flow integration completes, then 04.scoped-fix.flow.yml, 04.refactor.flow.yml, and 02.codebase-recon.flow.yml delegate to the new specialists.'
    validation: 'flow YAML files and agent-graph gate'
stop_conditions:
  done: 'All four phases complete with validation gates passing and tracker closed by 07-logging.'
  hold: 'The active step needs user prioritization or agent-architecture policy clarification.'
  blocked: 'An MCP/tool/agent/flow gap or upstream Agentic_Workflow_Architecture conflict prevents honest implementation.'
```

## Current State Audit

### Tier 1 Specialist Count

| Orchestrator     | Specialists | Status |
| :--------------- | :---------- | :----- |
| 00-helping       | 11          | PASS   |
| 01-planning      | 8           | PASS   |
| 02-researching   | 7           | PASS   |
| 03-red-testing   | 7           | PASS   |
| 04-implementing  | 19          | PASS   |
| 05-green-testing | 9           | PASS   |
| 06-documenting   | 6           | PASS   |
| 07-logging       | 5           | PASS   |

### Critical Gaps

1. **Substantive Work Leakage:** `04-implementing` currently performs the actual code synthesis (substantive work) directly. It lacks a Tier 2 "Executor" to handle the final write-phase.
2. **Skill Vacuum:** `04-implementing` has **0** listed skills. Durable standards (ES2023, Folder-based layout) are residing in `CLAUDE.md` and `copilot-instructions.md` instead of skill files.
3. **Research Method Gap:** `02-researching` is "skill-poor" (only 1 skill), acting as a thin wrapper over scouts without a durable research methodology.
4. **Synthesis Overload:** `01-planning` performs heavy synthesis of research data that could be delegated to a dedicated synthesis specialist.

## Optimization Proposals

### 1. New Durable Skills

- **`implementation-standards`**: Extract ES2023 policies, Module Architecture rules, and JSDoc requirements from `CLAUDE.md` into this skill.
- **`research-methodology`**: Formalize the "Discovery Order" and "Cortex-First" search patterns as a durable skill.
- **`routing-optimization-policy`**: Document the "mini-agent" delegation rules as a skill to prevent future drift.

### 2. New Specialized Agents

- **`implementation-executor` (Tier 2)**: A targeted agent responsible for the actual file edits, consuming the plan and coordination from `04-implementing`.
- **`research-synthesis-specialist` (Tier 3)**: Transforms raw scout data into the "Alignment Brief" requested by `01-planning`.
- **`code-quality-auditor` (Tier 3)**: A specialist for running and interpreting `npm run quality:folder` results before handing off to `05-green-testing`.

### 3. Routing Adjustments

- Move `04-implementing` from "Direct Coder" → "Implementation Coordinator".
- Bind `implementation-standards` to `04-implementing` and `implementation-executor`.
- Bind `research-methodology` to `02-researching`.

## Implementation Phases

### Phase 1: Skill Extraction [DONE]

**Outcome:** 3 durable skills extracted from `CLAUDE.md` and `copilot-instructions.md`

**Files created:**
- `.github/skills/implementation-standards/SKILL.md` — ES2023 policies, Module Architecture rules, JSDoc requirements
- `.github/skills/research-methodology/SKILL.md` — Discovery Order, Cortex-first search, certainty thresholds
- `.github/skills/routing-optimization-policy/SKILL.md` — Tier graph, delegation rules, mini-agent transition policy

**Validation:** All skills PASS `validate-skill-frontmatter` (0 errors, 0 warnings)

**Details:** See `plans/completed/Orchestration_System_Optimization.logs.md`

### Phase 2: Specialist Authoring [DONE]

**Outcome:** 7 new specialist agents created, Tier 1 allow-lists updated

**Agents created:**
- `implementation-executor` (Tier 2) — Execute scoped file edits from `04-implementing`
- `research-codebase-coordinator` (Tier 2) — Coordinate read-only reconnaissance
- `research-synthesis-specialist` (Tier 3) — Transform scout data into alignment briefs
- `code-quality-auditor` (Tier 3) — Run quality gates, classify violations, produce repair packets
- `test-coverage-analyst` (Tier 3) — Analyze lcov.info, map uncovered paths
- `acceptance-criteria-writer` (Tier 4) — Observable behavior boundaries
- `phase-handoff-designer` (Tier 3) — Sequential handoffs between phase agents

**Tier 1 allow-list updates:**
- `01-planning`: Added `research-synthesis-specialist`, `phase-handoff-designer`
- `05-green-testing`: Added `test-coverage-analyst`
- `07-logging`: Added `phase-handoff-designer`

**Validation:** `npm run agents:routing-table:gate` PASS (61 agents, 55 skills)

**Details:** See `plans/completed/Orchestration_System_Optimization.logs.md`

### Phase 3: Routing & Frontmatter Sync [DONE]

**Outcome:** All gates PASS, agent quality fixes applied, routing table validated. Tracker compressed — details archived to `plans/completed/Orchestration_System_Optimization.logs.md`.

**Gate results:**
- `tier-enforcement.gate`: **PASS** (8 Tier 1, 11 Tier 2, 38 Tier 3, 4 Tier 4, 0 violations)
- `agent-quality.gate`: **PASS** (0 errors, 0 warnings, all 61 agents compliant)
- `routing-table-freshness.gate`: **PASS** (hash match, 116 sources, 61 agents, 55 skills)

**Agent fixes applied:**
- `code-quality-auditor.agent.md`: Added `## Approach`, fixed structured-v1 field order (Tier 3)
- `implementation-executor.agent.md`: Fixed structured-v1 field order (Tier 2)
- `research-synthesis-specialist.agent.md`: Added `## Approach`, fixed structured-v1 field order (Tier 3)

**Validation:**
- `validate-plan-sync`: PASS (0 errors, 0 warnings)
- `validate-agent-graph`: PASS (0 errors, 0 violations)

**Details:** See `plans/completed/Orchestration_System_Optimization.logs.md`

---

## Step 02-07 Packets (Archived)

#### Step 02 — Create implementation-executor agent [DONE]

```yaml
phase: 2
step: 2
agent: 'creating-specialist-agent'
agent_file: '.github/agents/creating-specialist-agent.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 03 — Create research-codebase-coordinator agent'
skills: 'creating-specialist-agent, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent=implementation-executor
```

**Completion evidence:**
- Created `.github/agents/implementation-executor.agent.md` with valid Tier 2 frontmatter
- Model: `GPT-5.4 (copilot)` (qualified model string)
- Tools: `[read, search, edit, execute, todo, agent, neataptic-cortex-mcp/*, neataptic-gate-mcp/*, neataptic-validation-mcp/*, neataptic-workflow-mcp/*]`
- Skills: `['implementation-standards', 'coverage-guard']`
- Agents allow-list: `['boundary-mapper', 'docs-scout', 'browser-runtime-scout', 'worker-payload-scout', 'checkpoint-scout', 'determinism-scout', 'helping-gap-resolution-coordinator']`
- Handoff to `05-green-testing` configured
- Updated `04-implementing.agent.md` to include `implementation-executor` in agents allow-list
- Frontmatter validation: 0 errors for implementation-executor

**User instruction:** Start a fresh session, select `creating-specialist-agent`, and paste this full step packet.

**Step objective:** Create `.github/agents/implementation-executor.agent.md` with Tier 2 frontmatter, proper tool allow-list, and delegation contract with 04-implementing.

**Agent specification:**

```yaml
name: implementation-executor
tier: 2
model: 'qwen3.5:cloud (ollama)'
tools: [read, search, edit, execute, todo, agent, neataptic-cortex-mcp/*, neataptic-gate-mcp/*, neataptic-validation-mcp/*, neataptic-workflow-mcp/*]
user-invocable: false
agents: ['boundary-mapper', 'docs-scout', 'browser-runtime-scout', 'worker-payload-scout', 'checkpoint-scout', 'determinism-scout', 'helping-gap-resolution-coordinator']
skills: ['implementation-standards', 'coverage-guard']
handoffs:
  - label: 'Validate Green'
    agent: '05-green-testing'
    prompt: 'Continue from active plan and Step 02 implementation diff. Execute Step 05 validation for current phase.'
    send: false
    model: 'qwen3.5:cloud (ollama)'
```

**Responsibility boundary:** Executes scoped file edits delegated from 04-implementing. Does not plan, does not coordinate scouts, does not synthesize research. Pure execution of implementation packets.

**Stop conditions:**
- **Done:** Agent file created with valid frontmatter (0 errors, 0 warnings).
- **Hold:** A frontmatter policy decision is needed.
- **Blocked:** An MCP/tool/agent gap prevents honest agent creation.

---

#### Step 03 — Create research-codebase-coordinator agent [DONE]

```yaml
phase: 2
step: 3
agent: 'creating-specialist-agent'
agent_file: '.github/agents/creating-specialist-agent.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 04 — Create research-synthesis-specialist agent'
skills: 'creating-specialist-agent, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent=research-codebase-coordinator
```

**User instruction:** Start a fresh session, select `creating-specialist-agent`, and paste this full step packet.

**Step objective:** Create `.github/agents/research-codebase-coordinator.agent.md` with Tier 2 frontmatter, scout allow-list, and delegation contract with 02-researching.

**Agent specification:**

```yaml
name: research-codebase-coordinator
tier: 2
model: 'qwen3.5:cloud (ollama)'
tools: [read, search, edit, execute, todo, agent, neataptic-cortex-mcp/*, neataptic-gate-mcp/*, neataptic-validation-mcp/*, neataptic-workflow-mcp/*]
user-invocable: false
agents: ['plan-scout', 'docs-scout', 'repo-cortex-scout', 'boundary-mapper', 'research-synthesis-specialist', 'helping-gap-resolution-coordinator']
skills: ['research-methodology', 'plan-alignment']
handoffs:
  - label: 'Design Red Tests'
    agent: '03-red-testing'
    prompt: 'Continue from active plan and Step 03 research evidence. Execute Step 03 test design for current phase.'
    send: false
    model: 'qwen3.5:cloud (ollama)'
```

**Responsibility boundary:** Coordinates scout deployments for 02-researching, synthesizes results via research-synthesis-specialist, produces alignment briefs. Does not execute implementation, does not run tests.

**Stop conditions:**
- **Done:** Agent file created with valid frontmatter.
- **Hold:** A frontmatter policy decision is needed.
- **Blocked:** An MCP/tool/agent gap prevents honest agent creation.

---

#### Step 04 — Create research-synthesis-specialist agent [DONE]

```yaml
phase: 2
step: 4
agent: 'creating-specialist-agent'
agent_file: '.github/agents/creating-specialist-agent.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 05 — Create code-quality-auditor agent'
skills: 'creating-specialist-agent, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent=research-synthesis-specialist
```

**User instruction:** Start a fresh session, select `creating-specialist-agent`, and paste this full step packet.

**Step objective:** Create `.github/agents/research-synthesis-specialist.agent.md` with Tier 3 frontmatter, no delegation (Tier 4 only), and synthesis contract.

**Agent specification:**

```yaml
name: research-synthesis-specialist
tier: 3
model: 'qwen3.5:cloud (ollama)'
tools: [read, search, todo, neataptic-cortex-mcp/*, neataptic-gate-mcp/*, neataptic-validation-mcp/*, neataptic-workflow-mcp/*]
user-invocable: false
agents: ['acceptance-criteria-writer', 'file-change-summarizer']
skills: ['research-methodology', 'plan-alignment']
```

**Responsibility boundary:** Transforms raw scout data into structured alignment briefs. Read-only synthesis, no scouts, no implementation. May delegate to Tier 4 auxiliaries for criteria writing or summarization.

**Stop conditions:**
- **Done:** Agent file created with valid frontmatter.
- **Hold:** A frontmatter policy decision is needed.
- **Blocked:** An MCP/tool/agent gap prevents honest agent creation.

---

#### Step 05 — Create code-quality-auditor agent [DONE]

```yaml
phase: 2
step: 5
agent: 'creating-specialist-agent'
agent_file: '.github/agents/creating-specialist-agent.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 06 — Create test-coverage-analyst agent'
skills: 'creating-specialist-agent, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent=code-quality-auditor
```

**User instruction:** Start a fresh session, select `creating-specialist-agent`, and paste this full step packet.

**Step objective:** Create `.github/agents/code-quality-auditor.agent.md` with Tier 3 frontmatter, quality gate interpretation contract, and 05-green-testing delegation.

**Agent specification:**

```yaml
name: code-quality-auditor
tier: 3
model: 'qwen3.5:cloud (ollama)'
tools: [read, search, execute, todo, neataptic-cortex-mcp/*, neataptic-gate-mcp/*, neataptic-validation-mcp/*, neataptic-workflow-mcp/*]
user-invocable: false
agents: ['acceptance-criteria-writer']
skills: ['green-validation-gates', 'implementation-standards']
```

**Responsibility boundary:** Runs `npm run quality:folder`, interprets results, classifies violations, produces repair packets. Does not fix violations (delegates to 04-implementing or coverage-tranche).

**Stop conditions:**
- **Done:** Agent file created with valid frontmatter.
- **Hold:** A frontmatter policy decision is needed.
- **Blocked:** An MCP/tool/agent gap prevents honest agent creation.

---

#### Step 06 — Create test-coverage-analyst agent [DONE]

```yaml
phase: 2
step: 6
agent: 'creating-specialist-agent'
agent_file: '.github/agents/creating-specialist-agent.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 07 — Update Tier 1 agent allow-lists'
skills: 'creating-specialist-agent, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent=test-coverage-analyst
```

**User instruction:** Start a fresh session, select `creating-specialist-agent`, and paste this full step packet.

**Step objective:** Create `.github/agents/test-coverage-analyst.agent.md` with Tier 3 frontmatter, coverage analysis contract, and delegation from coverage-guard.

**Agent specification:**

```yaml
name: test-coverage-analyst
tier: 3
model: 'qwen3.5:cloud (ollama)'
tools: [read, search, execute, todo, neataptic-cortex-mcp/*, neataptic-gate-mcp/*, neataptic-validation-mcp/*, neataptic-workflow-mcp/*]
user-invocable: false
agents: []
skills: ['coverage-guard', 'coverage-tranche']
```

**Responsibility boundary:** Analyzes lcov.info, maps uncovered paths to source files, classifies dead vs reachable code, names owner-local test files. Read-only reconnaissance for coverage-tranche.

**Stop conditions:**
- **Done:** Agent file created with valid frontmatter.
- **Hold:** A frontmatter policy decision is needed.
- **Blocked:** An MCP/tool/agent gap prevents honest agent creation.

---

#### Step 07 — Update Tier 1 agent allow-lists [DONE]

```yaml
phase: 2
step: 7
agent: 'helping-agent-maintenance-coordinator'
agent_file: '.github/agents/helping-agent-maintenance-coordinator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Phase 3: Flow Integration'
skills: 'helping-agent-maintenance-coordinator, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - npm run agents:routing-table:gate
```

**Changes made:**
- Updated `01-planning.agent.md`: Added `research-synthesis-specialist`, `phase-handoff-designer` to agents: allow-list
- Updated `02-researching.agent.md`: `research-codebase-coordinator` already present (no change needed)
- Updated `04-implementing.agent.md`: `implementation-executor` already present (no change needed)
- Updated `05-green-testing.agent.md`: Added `test-coverage-analyst` to agents: allow-list (`code-quality-auditor` already present)
- Updated `07-logging.agent.md`: Added `phase-handoff-designer` to agents: allow-list (`file-change-summarizer` already present)
- Regenerated routing table with `npm run agents:routing-table`
- Validated with `npm run agents:routing-table:gate` - PASS (hash match, 61 agents, 55 skills)

**User instruction:** Start a fresh session, select `helping-agent-maintenance-coordinator`, and paste this full step packet.

**Step objective:** Update Tier 1 agent frontmatter to include new specialists in their `agents:` allow-lists, then regenerate and validate the routing table.

**Required updates:**

| Agent | Add to agents: |
|---|---|
| 04-implementing | `implementation-executor` |
| 02-researching | `research-codebase-coordinator` |
| 01-planning | `research-synthesis-specialist`, `acceptance-criteria-writer`, `phase-handoff-designer` |
| 05-green-testing | `code-quality-auditor`, `test-coverage-analyst` |
| 07-logging | `file-change-summarizer`, `phase-handoff-designer` |

**Stop conditions:**
- **Done:** All Tier 1 agents updated, routing table regenerated and validated.
- **Hold:** A frontmatter policy decision is needed.
- **Blocked:** An MCP/tool/agent gap prevents honest agent maintenance.

---

**Required validation:**

`npm run agents:routing-table:gate`

---

#### Step 01 — Audit routing table and specialist coverage [DONE]

**Completion evidence:**
- Routing table freshness gate: PASS (61 agents, 55 skills, hash match)
- All 7 new specialists properly indexed: `implementation-executor` (T2), `research-codebase-coordinator` (T2), `research-synthesis-specialist` (T3), `code-quality-auditor` (T3), `test-coverage-analyst` (T3), `acceptance-criteria-writer` (T4), `file-change-summarizer` (T4), `phase-handoff-designer` (T3)
- All 8 Tier 1 orchestrators have appropriate specialist delegations
- Tier violation fixed: Added `implementation-executor` to `TIER_2_AGENT_NAMES` in `tier-graph-utils.mjs`
- Agent graph validation: PASS (0 errors, 0 violations)

```yaml
phase: 3
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 02 — Update Tier 1 orchestrator delegations'
skills: 'plan-alignment, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - npm run agents:routing-table:gate
```

**Completion evidence:**
- Routing table freshness gate: PASS (61 agents, 55 skills, hash match)
- All 7 new specialists properly indexed in routing table
- All 8 Tier 1 orchestrators have appropriate specialist delegations
- Tier violation fixed: Added `implementation-executor` to `TIER_2_AGENT_NAMES` in `tier-graph-utils.mjs`
- Agent graph validation: PASS (0 errors, 0 violations)

#### Step 02 — Update Tier 1 orchestrator delegations [DONE]

```yaml
phase: 3
step: 2
agent: 'helping-agent-maintenance-coordinator'
agent_file: '.github/agents/helping-agent-maintenance-coordinator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 03 — Validate tier enforcement and agent quality'
skills: 'helping-agent-maintenance-coordinator, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - npm run agents:routing-table:gate
```

**Confirmation results:**

- Reviewed all 8 Tier 1 orchestrator `agents:` allow-lists
- Confirmed Phase 2 Step 07 already updated all required specialist delegations
- No gaps found — all specialists properly routed
- Routing table freshness gate: **PASS** (61 agents, 55 skills, hash match)

**Stop conditions:**

- **Done:** All Tier 1 orchestrators have complete specialist delegations, routing table validated. ✅
- **Hold:** A frontmatter policy decision is needed.
- **Blocked:** An MCP/tool/agent gap prevents honest agent maintenance.

#### Step 03 — Validate tier enforcement and agent quality [DONE]

```yaml
phase: 3
step: 3
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Phase 4: Flow Integration'
skills: 'green-validation-gates, coverage-guard, plan-sync-validation'
validation:
  - node scripts/agent-customization/gates/tier-enforcement.gate.mjs --json
  - node scripts/agent-customization/gates/agent-quality.gate.mjs --json
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
```

**Gate results:**

- `tier-enforcement.gate`: **PASS** (8 Tier 1, 11 Tier 2, 38 Tier 3, 4 Tier 4, 0 violations)
- `agent-quality.gate`: **PASS** (0 errors, 0 warnings, all 61 agents compliant)
- `validate-plan-sync`: **PASS** (0 errors, 0 warnings)

**Agent fixes applied:**

- `code-quality-auditor.agent.md`: Added `## Approach` section, fixed structured-v1 field order (Tier 3 scout contract)
- `implementation-executor.agent.md`: Fixed structured-v1 field order (Tier 2 coordinator contract: added `SPECIALISTS_USED`, `HANDOFF`)
- `research-synthesis-specialist.agent.md`: Added `## Approach` section, fixed structured-v1 field order (Tier 3 scout contract)

**Stop conditions:**

- **Done:** All gates pass, no tier violations, all specialists properly scoped. ✅
- **Hold:** Gate failures require policy decisions.
- **Blocked:** MCP/tool/agent gap prevents validation.

### Phase 4 — Flow Integration [WIP]

#### Step 01 — Map flows to Phase 4 objectives [WIP]

**Current Active Step:** Step 01 — Map flows to Phase 4 objectives [WIP]

```yaml
phase: 4
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[WIP]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 02 — Assign flow owners and validation gates'
skills: 'plan-alignment, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - node scripts/agent-customization/gates/flow-mention.gate.mjs --json
```

**User instruction:** Start a fresh session, select `01-planning`, and paste this full step packet.

**Step objective:** Map Phase 4 Flow Integration objectives to named agent flows from `.github/flows/`, assign owners, and define exit gates.

**Execution steps:**

1. Read Phase 4 objectives and existing flow definitions in `.github/flows/`.
2. Identify which flows map to Flow Integration work (e.g., flow-aware routing, gate protocol enforcement).
3. Define Step 02-07 packets for value-adding work or explicit skip records for non-value gates.
4. Update plan with flow assignments, gate contracts, and validation commands.
5. Run `flow-mention.gate` to confirm flows are properly referenced.

**Stop conditions:**

- **Done:** All Phase 4 objectives mapped to flows, owners assigned, gates defined.
- **Hold:** Flow definitions missing or ambiguous.
- **Blocked:** MCP/tool/agent gap prevents flow mapping.

## Validation Gates

- `agent-graph` gate: Confirm no Tier 1 agent is performing work that should be in Tier 2/3.
- `routing-table-freshness` gate: Confirm all new agents are indexed.
- `agent-quality` gate: Confirm new specialists follow the "mini-agent" (targeted context) pattern.
- `tier-enforcement` gate: Confirm all agents have correct tier assignments and delegation structure.


### Latest validation evidence

- 2026-06-08: Phase 3 [DONE] — All gates PASS, 3 agent quality fixes applied, routing table validated
- 2026-06-08: `npm run agents:routing-table:gate` PASS (hash match, 61 agents, 55 skills)
- 2026-06-08: `validate-plan-sync` PASS (0 errors, 0 warnings)
- 2026-06-08: Workflow sync: Phase 3 complete — Phase 4 Step 01 ready for Flow Integration

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Phase 3 is [DONE] — all gates PASS, agent quality fixed, routing table validated. Phase 4 Step 01 is the active frontier. Start a fresh session with `01-planning` and paste the Phase 4 Step 01 packet to begin flow integration work.
```
