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

## Implementation phases

### Phase 1: Skill Extraction [DONE]

#### Step 01 — Planning skill extraction strategy [DONE]

**Completion evidence:** Planning brief produced with policy inventories for all three skills. Step 02-04 packets prepared for 06-documenting.

```yaml
phase: 1
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 02 — Author implementation-standards skill'
skills: 'planning-acceptance-criteria, plan-alignment, skill-frontmatter-standards'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
```

**Step objective:** Plan the extraction of three durable skills (`implementation-standards`, `research-methodology`, `routing-optimization-policy`) by identifying which policies to extract from CLAUDE.md and copilot-instructions.md, defining skill boundaries, and preparing Step 02-04 packets.

**User instruction:** Start a fresh session, select `01-planning`, and paste this full step packet.

**Context the agent must know:**

- The upstream authority is `plans/completed/Agentic_Workflow_Architecture.plans.md` (archived baseline); preserve its terminology including skill-first customization, durable workflow knowledge, and tier graph.
- Skills must follow the standard SKILL.md frontmatter format with `skills:` field, description under 1024 characters, and proper visibility flags.
- Each skill file must be placed in `.github/skills/<skill-name>/SKILL.md` with proper folder structure.
- This step is planning-only; do not author the actual skill files (that happens in Steps 02-04).

**Execution steps:**

1. Use `neataptic-workflow-mcp.get_active_workflow_snapshot` to confirm this tracker is active and Step 01 is [WIP].
2. Read `CLAUDE.md` and `.github/copilot-instructions.md` to identify durable policies to extract.
3. Plan `implementation-standards` skill: list ES2023 policies, Module Architecture rules, and JSDoc requirements to extract.
4. Plan `research-methodology` skill: list "Discovery Order" and "Cortex-First" search patterns to formalize.
5. Plan `routing-optimization-policy` skill: list "mini-agent" delegation rules to document.
6. For each skill, define acceptance criteria and validation steps.
7. Prepare Step 02-04 packets (one per skill) with copy-paste instructions for `06-documenting`.
8. Record the planning output in this tracker before ending the step.

**Stop conditions:**

- **Done:** Planning brief complete with policy inventories for all three skills and Step 02-04 packets prepared.
- **Hold:** A policy decision is needed before planning can complete.
- **Blocked:** An MCP/tool/agent gap prevents honest planning; escalate to `00-helping`.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md`

#### Step 02 — Author implementation-standards skill [DONE]

**Completion evidence:** Skill file created at `.github/skills/implementation-standards/SKILL.md` with valid frontmatter (0 errors, 0 warnings).

```yaml
phase: 1
step: 2
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 03 — Author research-methodology skill'
skills: 'skill-frontmatter-standards, educational-docs'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - node scripts/agent-customization/validate-skill-frontmatter.mjs --json --skill=implementation-standards
```

**Step objective:** Create `.github/skills/implementation-standards/SKILL.md` with frontmatter and extracted policies from CLAUDE.md and copilot-instructions.md.

**User instruction:** Start a fresh session, select `06-documenting`, and paste this full step packet.

**Context the agent must know:**

- Use the planning brief from Step 01 as the policy inventory.
- Extract ES2023 policies, Module Architecture rules, and JSDoc requirements.
- Follow SKILL.md frontmatter format with `skills:` field and description under 1024 characters.

**Execution steps:**

1. Read the Step 01 planning brief and policy inventory for `implementation-standards`.
2. Read `CLAUDE.md` and `.github/copilot-instructions.md` to extract the identified policies.
3. Create `.github/skills/implementation-standards/SKILL.md` with proper frontmatter and content.
4. Validate with `node scripts/agent-customization/validate-skill-frontmatter.mjs --json --skill=implementation-standards`.
5. Record the skill authoring output in this tracker.

**Stop conditions:**

- **Done:** Skill file exists with valid frontmatter and complete policy content.
- **Hold:** A policy decision is needed before authoring can complete.
- **Blocked:** An MCP/tool/agent gap prevents honest skill authoring; escalate to `00-helping`.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md`

#### Step 03 — Author research-methodology skill [DONE]

```yaml
phase: 1
step: 3
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 04 — Author routing-optimization-policy skill'
skills: 'skill-frontmatter-standards, educational-docs'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - node scripts/agent-customization/validate-skill-frontmatter.mjs --json --skill=research-methodology
```

**Step objective:** Create `.github/skills/research-methodology/SKILL.md` with frontmatter and formalized discovery order patterns.

**Completed:**

- Created `.github/skills/research-methodology/SKILL.md` with proper frontmatter.
- Frontmatter validation: PASS (0 errors, 0 warnings).
- Plan sync validation: PASS (0 errors, 0 warnings).

**Skill content includes:**

- Discovery Order policy (README → parent README → plans → source files).
- Cortex-first search patterns with `use_dense: true` and prewarm guidance.
- Plan-aware execution workflow coordinating with `plan-alignment`.
- Certainty thresholds (< 90% stop, < 95% investigate, ≥ 95% proceed).
- Context window mitigation (plan updates, handoff prompts).
- Demo-first library gap policy for investigation from demo symptoms.

**User instruction:** Start a fresh session, select `06-documenting`, and paste this full step packet.

**Context the agent must know:**

- Use the Step 01 planning brief as the policy inventory.
- Formalize "Discovery Order" and "Cortex-First" search patterns from CLAUDE.md.

**Execution steps:**

1. Read the Step 01 planning brief and policy inventory for `research-methodology`.
2. Read `CLAUDE.md` to extract the identified discovery patterns.
3. Create `.github/skills/research-methodology/SKILL.md` with proper frontmatter and content.
4. Validate with `node scripts/agent-customization/validate-skill-frontmatter.mjs --json --skill=research-methodology`.
5. Record the skill authoring output in this tracker.

**Stop conditions:**

- **Done:** Skill file exists with valid frontmatter and complete discovery pattern content.
- **Hold:** A policy decision is needed before authoring can complete.
- **Blocked:** An MCP/tool/agent gap prevents honest skill authoring; escalate to `00-helping`.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md`

#### Step 04 — Author routing-optimization-policy skill [DONE]

**Completion evidence:** Skill file created at `.github/skills/routing-optimization-policy/SKILL.md` with valid frontmatter (0 errors, 0 warnings).

```yaml
phase: 1
step: 4
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Phase 2: Specialist Authoring'
skills: 'skill-frontmatter-standards, educational-docs'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - node scripts/agent-customization/validate-skill-frontmatter.mjs --json --skill=routing-optimization-policy
```

**Step objective:** Create `.github/skills/routing-optimization-policy/SKILL.md` with frontmatter and mini-agent delegation rules.

**User instruction:** Start a fresh session, select `06-documenting`, and paste this full step packet.

**Context the agent must know:**

- Use the Step 01 planning brief as the policy inventory.
- Document the "mini-agent" delegation rules to prevent future drift.

**Execution steps:**

1. Read the Step 01 planning brief and policy inventory for `routing-optimization-policy`.
2. Read `CLAUDE.md` and `plans/` to extract delegation rules.
3. Create `.github/skills/routing-optimization-policy/SKILL.md` with proper frontmatter and content.
4. Validate with `node scripts/agent-customization/validate-skill-frontmatter.mjs --json --skill=routing-optimization-policy`.
5. Record the skill authoring output in this tracker.

**Stop conditions:**

- **Done:** Skill file exists with valid frontmatter and complete delegation rule content.
- **Hold:** A policy decision is needed before authoring can complete.
- **Blocked:** An MCP/tool/agent gap prevents honest skill authoring; escalate to `00-helping`.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md`

### Phase 2: Specialist Authoring [DONE]

**Current Active Step:** Phase 3: Flow Integration [PLANNED]

**Latest validation evidence:**
- 2026-06-08: Step 02 complete — `implementation-executor.agent.md` created, frontmatter validated (0 errors)
- 2026-06-08: Step 03 complete — `research-codebase-coordinator.agent.md` created, frontmatter validated (0 errors)
- 2026-06-08: Step 04-06 complete — 4 additional specialists created with valid frontmatter
- 2026-06-08: Step 07 complete — Tier 1 agent allow-lists updated (01-planning, 05-green-testing, 07-logging)
- 2026-06-08: Routing table regenerated and validated (npm run agents:routing-table:gate PASS)
- 2026-06-08: `workflow-update-sync.mjs` advanced Phase 2→DONE, Phase 3→PLANNED

**Phase objective:** Create the specialist agent roster identified in Phase 1 gaps analysis. Author agent files with correct tier assignments, frontmatter, and bounded delegation contracts. Ensure each specialist has a clear responsibility boundary and integrates with the new skills from Phase 1.

**Phase progression rule:** Start with only Step 01. Step 01 must author the remaining numbered step packets, or explicit skipped-step packets, before the phase can advance.

#### Step 01: Plan specialist roster and boundaries [DONE]

**Status:** Complete

**Completion evidence:** Specialist roster defined with tier assignments, boundaries, and Step 02-07 packets prepared in tracker below. Plan sync validation: PASS (0 errors, 0 warnings).

```yaml
phase: 2
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 02 — Create implementation-executor agent'
skills: 'planning-acceptance-criteria, plan-alignment, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
```

**User instruction:** Start a fresh session, select `01-planning`, and paste this full step packet.

**Step objective:** Plan the specialist agent roster by defining the exact specialists to create, their tier assignments, responsibility boundaries, and which Tier 1 orchestrators they serve. Prepare Step 02-07 packets for execution.

**Context the agent must know:**

- Phase 1 created three skills: `implementation-standards`, `research-methodology`, `routing-optimization-policy`.
- Critical gaps from the audit: 04-implementing lacks a Tier 2 executor, 02-researching lacks synthesis support, research methodology was skill-poor.
- Specialists must follow the tier graph: Tier 2 may delegate to Tier 3/4, Tier 3 to Tier 4 only, Tier 4 cannot delegate.
- Only Tier 1 agents are user-invocable; all specialists must have `user-invocable: false`.
- Each specialist needs proper frontmatter with `tier`, `skills`, `agents` (allow-list), and `model` fields.

**Execution steps:**

1. Use `neataptic-workflow-mcp.get_active_workflow_snapshot` to confirm this tracker is active.
2. Review the Phase 1 completion evidence and the three new skills created.
3. Define the specialist roster:
   - `implementation-executor` (Tier 2): Handles actual file edits for 04-implementing.
   - `research-synthesis-specialist` (Tier 3): Transforms scout data into alignment briefs for 01-planning.
   - `code-quality-auditor` (Tier 3): Runs quality gates and interprets results for 05-green-testing.
   - Additional specialists as needed based on gap analysis.
4. For each specialist, define: tier, primary owner (which Tier 1 agent delegates to it), responsibility boundary, and required skills binding.
5. Prepare Step 02-07 packets with copy-paste instructions for the appropriate agents.
6. Record the specialist roster plan in this tracker before ending the step.

**Stop conditions:**

- **Done:** Specialist roster defined with tier assignments, boundaries, and Step 02-07 packets prepared.
- **Hold:** A tier or delegation policy decision is needed before planning can complete.
- **Blocked:** An MCP/tool/agent gap prevents honest planning; escalate to `00-helping`.

---

## Specialist Roster Plan

### Tier 2 Specialists (Coordinators / Sub-Orchestrators)

| Specialist | Primary Owner | Responsibility Boundary | Skills Binding | Model |
|---|---|---|---|---|
| `implementation-executor` | 04-implementing | Actual file edits, patch application, and write-phase synthesis. Consumes implementation packets from 04-implementing and executes scoped changes. | `implementation-standards`, `coverage-guard` | `qwen3.5:cloud` |
| `research-codebase-coordinator` | 02-researching | Coordinates scout deployments, synthesizes reconnaissance results, and produces alignment briefs for 01-planning. | `research-methodology`, `plan-alignment` | `qwen3.5:cloud` |

### Tier 3 Specialists (Hidden Scouts)

| Specialist | Primary Owner | Responsibility Boundary | Skills Binding | Model |
|---|---|---|---|---|
| `research-synthesis-specialist` | research-codebase-coordinator | Transforms raw scout data (from Plan Scout, Docs Scout, Boundary Mapper) into structured alignment briefs. Does not run scouts directly. | `research-methodology`, `plan-alignment` | `qwen3.5:cloud` |
| `code-quality-auditor` | 05-green-testing | Runs `npm run quality:folder`, interprets results, classifies violations, and produces repair packets for 04-implementing or coverage-tranche. | `green-validation-gates`, `implementation-standards` | `qwen3.5:cloud` |
| `test-coverage-analyst` | coverage-guard | Analyzes lcov.info, maps uncovered paths to source files, classifies dead vs reachable code, and names owner-local test files for coverage-tranche. | `coverage-guard`, `coverage-tranche` | `qwen3.5:cloud` |

### Tier 4 Specialists (Auxiliaries / One-Shot Helpers)

| Specialist | Primary Owner | Responsibility Boundary | Skills Binding | Model |
|---|---|---|---|---|
| `acceptance-criteria-writer` | 01-planning, 03-red-testing | Generates observable acceptance criteria from user intent before coding begins. | `planning-acceptance-criteria`, `red-test-contracts` | `qwen3.5:cloud` |
| `file-change-summarizer` | 07-logging | Summarizes changed files, affected customization surfaces, validation evidence, and residual risks for logging or handoff. | `summarizing-session-log` | `qwen3.5:cloud` |
| `phase-handoff-designer` | All Tier 1 agents | Designs sequential handoff prompts between seven phase agents within a plan. | `phase-handoff-workflow`, `tracker-handoff` | `qwen3.5:cloud` |

### Delegation Contracts

```
Tier 1 → Tier 2:
  04-implementing → implementation-executor (file edits)
  02-researching → research-codebase-coordinator (scout coordination)

Tier 2 → Tier 3:
  research-codebase-coordinator → research-synthesis-specialist (synthesis)
  05-green-testing → code-quality-auditor (quality gate interpretation)
  05-green-testing → test-coverage-analyst (coverage gap mapping)

Tier 3 → Tier 4:
  code-quality-auditor → acceptance-criteria-writer (if criteria missing)
  01-planning → acceptance-criteria-writer (direct, for planning phase)
  07-logging → file-change-summarizer (session summary)
  All Tier 1/2 → phase-handoff-designer (handoff prompt design)
```

### Frontmatter Requirements

All new specialists must declare:

```yaml
tier: <2|3|4>
model: 'qwen3.5:cloud (ollama)'
tools: [read, search, edit?, execute?, todo, agent?, neataptic-cortex-mcp/*, neataptic-gate-mcp/*, neataptic-validation-mcp/*, neataptic-workflow-mcp/*]
user-invocable: false
agents: [<allow-list of specialists this agent may delegate to>]
skills: [<skill names from Phase 1 or existing>]
```

**Tool restrictions by tier:**
- Tier 2: May use `edit`, `execute`, `agent` (delegate to Tier 3/4)
- Tier 3: May use `edit` only for plan files, `agent` (delegate to Tier 4)
- Tier 4: No `agent` tool (cannot delegate)

---

## Step 02-07 Packets

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

### Phase 3: Routing & Frontmatter Sync [WIP]

**Current Active Step:** Step 01 — Audit routing table and specialist coverage [WIP]

**Latest validation evidence:**
- 2026-06-08: Phase 2 [DONE] — All 7 specialist agents created with valid frontmatter, Tier 1 allow-lists updated, routing table regenerated and validated
- 2026-06-08: `npm run agents:routing-table:gate` PASS (hash match, 61 agents, 55 skills)
- 2026-06-08: `validate-plan-sync` PASS (0 errors, 0 warnings)
- 2026-06-08: Workflow sync: Phase 2 complete — all steps [DONE], Phase 3 Step 01 marked [WIP] for MCP tracking

#### Step 01 — Audit routing table and specialist coverage [WIP]

```yaml
phase: 3
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[WIP]'
mode: 'fresh-session'
source_of_truth: 'plans/Orchestration_System_Optimization.plans.md'
copy_paste: 'true'
next_step: 'Step 02 — Update Tier 1 orchestrator delegations'
skills: 'plan-alignment, agent-frontmatter-standards, routing-optimization-policy'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md
  - npm run agents:routing-table:gate
```

**User instruction:** Start a fresh session, select `01-planning`, and paste this full step packet.

**Step objective:** Audit the routing table to confirm all 7 new specialists are properly indexed, and identify any Tier 1 orchestrators that still lack specialist delegations.

**Context the agent must know:**

- Phase 2 created 7 specialists: `implementation-executor` (T2), `research-codebase-coordinator` (T2), `research-synthesis-specialist` (T3), `code-quality-auditor` (T3), `test-coverage-analyst` (T3), plus 2 Tier 4 auxiliaries.
- Step 07 already updated Tier 1 allow-lists, but this step confirms completeness.
- Use `neataptic-gate-mcp.query_customization_routing_table` to inspect the routing table.
- Use `neataptic-gate-mcp.query_tier_graph` to validate tier structure.

**Execution steps:**

1. Run `npm run agents:routing-table:gate` to confirm routing table freshness.
2. Query the routing table with `neataptic-gate-mcp.query_customization_routing_table --includeRows --includeMarkdown`.
3. Confirm all 7 new specialists appear in the routing table.
4. Identify any Tier 1 agents that still lack appropriate specialist delegations.
5. Record audit findings in this tracker.

**Stop conditions:**

- **Done:** Routing table audited, all specialists indexed, Tier 1 delegation gaps identified.
- **Hold:** A routing table or indexing gap needs user clarification.
- **Blocked:** A gate or MCP gap prevents honest audit; escalate to `00-helping`.

**Required validation:**

`npm run agents:routing-table:gate`

### Phase 4: Flow Integration [PLANNED]

## Validation Gates

- `agent-graph` gate: Confirm no Tier 1 agent is performing work that should be in Tier 2/3.
- `routing-table-freshness` gate: Confirm all new agents are indexed.
- `agent-quality` gate: Confirm new specialists follow the "mini-agent" (targeted context) pattern.
- `tier-enforcement` gate: Confirm all agents have correct tier assignments and delegation structure.


### Latest validation evidence

- 2026-06-08: Phase 2 [DONE] — All 7 specialist agents created with valid frontmatter, Tier 1 allow-lists updated, routing table regenerated and validated
- 2026-06-08: `npm run agents:routing-table:gate` PASS (hash match, 61 agents, 55 skills)
- 2026-06-08: `validate-plan-sync` PASS (0 errors, 0 warnings)
- 2026-06-08: Workflow sync: Phase 2 complete — all steps [DONE], Phase 3 ready for promotion to [WIP]

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Phase 2 is [DONE] — all 7 specialists created and Tier 1 allow-lists updated. Phase 3 Step 01 is the active frontier. Start a fresh session with `01-planning` and paste the Phase 3 Step 01 packet to begin routing table and frontmatter synchronization work.
```
