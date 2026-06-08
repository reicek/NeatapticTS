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

### Phase 2: Specialist Authoring [PLANNED]

### Phase 3: Routing & Frontmatter Sync [PLANNED]

### Phase 4: Flow Integration [PLANNED]

## Validation Gates

- `agent-graph` gate: Confirm no Tier 1 agent is performing work that should be in Tier 2/3.
- `routing-table-freshness` gate: Confirm all new agents are indexed.
- `agent-quality` gate: Confirm new specialists follow the "mini-agent" (targeted context) pattern.
- `tier-enforcement` gate: Confirm all agents have correct tier assignments and delegation structure.

## Handoff query

Phase 1 Step 01 is the active frontier. Start a fresh session with `01-planning` and paste the Step 01 packet to plan the skill extraction strategy. The planning agent will produce a brief with policy inventories and prepare Step 02-04 packets for `06-documenting`.
