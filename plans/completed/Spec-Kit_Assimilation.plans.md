# Spec-Kit Assimilation

**Status:** [DONE]

## Current state

Phase status: Phase 1 [DONE], Phase 2 [DONE], Phase 3 [DONE], Phase 4 [DONE], Phase 5 [DONE], Phase 6 [DONE], Phase 7 [DONE].

Step status: Steps 01–11 [DONE] — all 11 implementation slices verified green (RED scripts 1–11 pass, artifacts present, gates green). Plan archived to plans/completed/.

Active slice: none — workstream complete.

Next: Workstream closed. Plan pair archived to plans/completed/Spec-Kit_Assimilation.plans.md + .logs.md.

This plan cherry-picks the highest-value Spec Kit practices into NeatapticTS
while preserving the existing agent/gate/skill/step-packet architecture.
Source of truth for Spec Kit recommendations:
`spec-kit/spec-kit.md` and `spec-kit/verbatim/.specify/memory/constitution.md`.

```yaml
PlanUpdate:
  slice_id: '11-flow-updates'
  parent_slice_id: '09-verbatim-phrases'
  changed_files:
    - '.github/flows/01.phase-kickoff.flow.yml'
    - '.github/flows/01.step-expansion.flow.yml'
    - '.github/flows/01.step-packet-revision.flow.yml'
    - '.github/flows/05.coverage-guard.flow.yml'
    - '.github/flows/05.ci-green-confirmation.flow.yml'
    - '.github/flows/07.tracker-closure.flow.yml'
    - '.github/flows/flow.schema.yml'
    - 'scripts/agent-customization/__red__spec-kit-slice-11.mjs'
    - 'plans/Spec-Kit_Assimilation.plans.md'
  preflight:
    - 'npx prettier --check .github/flows/01.phase-kickoff.flow.yml .github/flows/01.step-expansion.flow.yml .github/flows/01.step-packet-revision.flow.yml .github/flows/05.coverage-guard.flow.yml .github/flows/05.ci-green-confirmation.flow.yml .github/flows/07.tracker-closure.flow.yml .github/flows/flow.schema.yml scripts/agent-customization/__red__spec-kit-slice-11.mjs'
    - 'npm run lint'
    - 'npx tsc --noEmit -p tsconfig.json'
  red_phase:
    before: 'FAIL — AC-040..AC-043 not satisfied (constitution_check/spec-checklist/gate-run strings missing from target flows)'
    after: 'PASS — AC-040..AC-043 satisfied after surgical flow edits'
  validation:
    - command: 'node scripts/agent-customization/__red__spec-kit-slice-11.mjs'
      expected_exit: 0
      result: 'PASS — AC-040..AC-046 (10/10 checks)'
    - command: 'node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict'
      expected_exit: 0
      result: 'PASS — 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-skill-frontmatter.mjs --json'
      expected_exit: 0
      result: 'PASS — 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-agent-graph.mjs --json'
      expected_exit: 0
      result: 'PASS — 0 errors, 0 warnings'
    - command: 'npm run agents:routing-table'
      expected_exit: 0
      result: 'PASS — changed=false, sourceHash matches (routing table stable)'
    - command: 'git diff --stat .github/agent-skill-routing-table.md'
      expected_exit: 0
      result: 'OBSERVATION — 6 insertions(+), 4 deletions(-); diff is from prior uncommitted slice changes, not from this slice; regen changed=false'
    - command: 'npm run agents:routing-table:gate'
      expected_exit: 0
      result: 'PASS — hash match'
    - command: 'npx prettier --check .github/flows/01.phase-kickoff.flow.yml .github/flows/01.step-expansion.flow.yml .github/flows/01.step-packet-revision.flow.yml .github/flows/05.coverage-guard.flow.yml .github/flows/05.ci-green-confirmation.flow.yml .github/flows/07.tracker-closure.flow.yml .github/flows/flow.schema.yml scripts/agent-customization/__red__spec-kit-slice-11.mjs'
      expected_exit: 0
      result: 'PASS — All matched files use Prettier code style'
    - command: 'npm run lint'
      expected_exit: 0
      result: 'PASS — 0 issues'
    - command: 'npx tsc --noEmit -p tsconfig.json'
      expected_exit: 0
      result: 'PASS — no TypeScript errors'
    - command: 'neataptic-gate-mcp-run_gate_check gate=plan-sync'
      expected_exit: 0
      result: 'PASS'
    - command: 'neataptic-gate-mcp-run_gate_check gate=agent-graph'
      expected_exit: 0
      result: 'PASS'
    - command: 'neataptic-gate-mcp-run_gate_check gate=routing-table-freshness'
      expected_exit: 0
      result: 'PASS'
    - command: 'neataptic-gate-mcp-run_gate_check gate=agent-quality'
      expected_exit: 0
      result: 'PASS'
    - command: 'neataptic-gate-mcp-run_gate_check gate=tier-enforcement'
      expected_exit: 0
      result: 'PASS'
    - command: 'neataptic-gate-mcp-run_gate_check gate=learning-event'
      expected_exit: 0
      result: 'PASS'
    - command: 'neataptic-gate-mcp-run_gate_check gate=plan-slice-quality'
      expected_exit: 0
      result: 'PASS'
    - command: 'node scripts/agent-customization/__red__spec-kit-slice-1.mjs through slice-10'
      expected_exit: 0
      result: 'PASS — all prior slice RED scripts remain green'
  gates:
    - 'plan-sync: PASS'
    - 'agent-graph: PASS'
    - 'routing-table-freshness: PASS'
    - 'agent-quality: PASS'
    - 'tier-enforcement: PASS'
    - 'learning-event: PASS'
    - 'plan-slice-quality: PASS'
    - 'step-packet: KNOWN PRE-EXISTING FAIL in sibling plan only (plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md); Spec-Kit_Assimilation.plans.md blocks clean — not fixed in this slice'
  rollback:
    - 'git checkout -- .github/flows/01.phase-kickoff.flow.yml .github/flows/01.step-expansion.flow.yml .github/flows/01.step-packet-revision.flow.yml .github/flows/05.coverage-guard.flow.yml .github/flows/05.ci-green-confirmation.flow.yml .github/flows/07.tracker-closure.flow.yml .github/flows/flow.schema.yml scripts/agent-customization/__red__spec-kit-slice-11.mjs plans/Spec-Kit_Assimilation.plans.md'
  next: 'Hand off to 07-logging for phase compression and workstream closure, or to user for final review/PR creation.'
```

<!-- PlanUpdate: tracker-correction (2026-07-04) -->

```yaml
PlanUpdate:
  slice_id: 'tracker-correction'
  parent_slice_id: '11-flow-updates'
  reason: >-
    Procedural correction — implementing/logging agents completed all 11
    slices and verified them green, but did NOT update step/phase status
    markers in this tracker. Steps 1-10 and Phases 1-6 were left at
    [WIP]/[PLANNED] despite being DONE. This update realigns the tracker
    with verified ground truth.
  verified_ground_truth:
    - 'All 11 RED scripts pass: node scripts/agent-customization/__red__spec-kit-slice-{1..11}.mjs → 0 failures (re-run 2026-07-04)'
    - 'Artifacts confirmed present: plans/constitution.md, .github/templates/tasks-template.md, .github/skills/spec-checklist/SKILL.md, .github/skills/bug-triage/SKILL.md, .github/bugs/README.md, .github/extensions/README.md, docs/research/README.md'
    - 'Agent/flow edits confirmed: constitution_check + spec-checklist present in 01-planning.agent.md and 8 flow files'
    - 'Gates green: step-packet PASS (0 violations), plan-sync PASS, routing-table-freshness PASS'
  status_changes:
    - 'Phase 1-6: [WIP]/[PLANNED] → [DONE]'
    - 'Step 01,02,03,04,08,10: [WIP]/[PLANNED] → [DONE] (slices 01-green, 02-clarification-cap, 03-traceability-ids, 04-tasks-template, 08-extension-catalog, 10-learning-event-schema → [DONE])'
    - 'Step 05,06,07,09: already [DONE] (unchanged)'
    - 'Phase 7 + Step 11 (Convergence review and archive): [PLANNED] → [WIP] — archive sub-task NOT yet performed'
  remaining_work:
    - 'AC-024: move plan pair to plans/completed/ with matching .logs.md'
    - 'AC-025: append final constitution-update learning event'
  validation:
    - command: 'node scripts/agent-customization/__red__spec-kit-slice-{1..11}.mjs'
      result: 'PASS — 11/11 slices, 0 failures'
    - command: 'neataptic-gate-mcp:run_gate_check gate=step-packet'
      result: 'PASS — 0 violations'
    - command: 'neataptic-gate-mcp:run_gate_check gate=plan-sync'
      result: 'PASS — registered in README + Roadmap'
    - command: 'neataptic-gate-mcp:run_gate_check gate=routing-table-freshness'
      result: 'PASS'
  gates:
    - 'step-packet: PASS'
    - 'plan-sync: PASS'
    - 'routing-table-freshness: PASS'
  next: >-
    Dispatch 07-logging to execute Step 11 (archive): create
    plans/Spec-Kit_Assimilation.logs.md, move plan pair to
    plans/completed/, append final constitution-update learning event,
    then mark plan [DONE].
```

<!-- /PlanUpdate: tracker-correction -->

## Latest validation evidence

- green-light: true
- validator: `04-implementing`
- slice: '07-research-artifact'
- evidence:
  - `node scripts/agent-customization/__red__spec-kit-slice-6.mjs` → PASS (AC-023..AC-025)
  - `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent .github/agents/02-researching.agent.md` → PASS (0 errors, 0 warnings)
  - `npx prettier --check .github/agents/02-researching.agent.md docs/research/README.md scripts/agent-customization/__red__spec-kit-slice-6.mjs` → PASS
- gate notes:
  - `plan-sync` → PASS
  - `agent-graph` → PASS
  - `plan-slice-quality` → PASS
  - `validate-plan-phase-packets` → pre-existing format issue (no implementation phase packets found; plan uses YAML status markers instead of markdown headers, which is consistent with earlier slices)
  - `step-packet` → no violations in `Spec-Kit_Assimilation.plans.md` WIP blocks; global scan still blocked by pre-existing violations in `plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md`
  - `plan-slice-quality` → PASS (all WIP slices ≤ 4 hours)

### Slice 09 — Learning-event schema expansion

- green-light: true
- validator: `04-implementing`
- slice: `09-learning-event-schema`
- evidence:
  - `node scripts/agent-customization/__red__spec-kit-slice-9.mjs` → PASS (AC-033..AC-035)
  - `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent .github/agents/07-logging.agent.md` → PASS (0 errors, 0 warnings)
  - `node scripts/agent-customization/validate-skill-frontmatter.mjs --json` → PASS (0 errors, 0 warnings)
  - `npx prettier --check .github/skills/capturing-learning-event/SKILL.md .github/agents/07-logging.agent.md scripts/agent-customization/__red__spec-kit-slice-9.mjs` → PASS
  - `npm run lint` → PASS (0 issues)
  - `neataptic-gate-mcp:run_gate_check gate=agent-graph` → PASS
  - `neataptic-gate-mcp:run_gate_check gate=learning-event` → PASS

### Slice 09-verbatim-phrases — Cross-cutting Spec-Kit phrase adoption

- green-light: true
- validator: `04-implementing`
- slice: `09-verbatim-phrases`
- evidence:
  - `node scripts/agent-customization/__red__spec-kit-slice-10.mjs` → PASS (AC-036..AC-039, 27/27 checks)
  - `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict` → PASS (0 errors, 0 warnings)
  - `node scripts/agent-customization/validate-skill-frontmatter.mjs --json` → PASS (0 errors, 0 warnings)
  - `node scripts/agent-customization/validate-agent-graph.mjs --json` → PASS (0 errors, 0 warnings)
  - `npm run agents:routing-table` → PASS (regenerated `.github/agent-skill-routing-table.md`)
  - `npm run agents:routing-table:gate` → PASS (hash match)
  - `npx prettier --check .github/agents/00-helping.agent.md .github/agents/01-planning.agent.md .github/agents/02-researching.agent.md .github/agents/03-red-testing.agent.md .github/agents/04-implementing.agent.md .github/agents/05-green-testing.agent.md .github/agents/06-documenting.agent.md .github/agents/07-logging.agent.md .github/skills/planning-acceptance-criteria/SKILL.md .github/skills/phase-handoff-workflow/SKILL.md .github/skills/plan-sync-validation/SKILL.md scripts/agent-customization/__red__spec-kit-slice-10.mjs` → PASS
  - `npm run lint` → PASS (0 issues)
  - `npx tsc --noEmit -p tsconfig.json` → PASS
- phrase adoption notes:
  - `00-helping` → "constitution authority" added to Purpose; References cite `agent-frontmatter-standards` and `phase-handoff-workflow`.
  - `01-planning` → already carried required phrases from prior slices; References cite `planning-acceptance-criteria`, `phase-handoff-workflow`, and `plan-sync-validation`.
  - `02-researching` → "gap types: missing/partial/contradicts/unrequested" added to Purpose; References cite `research-methodology` and `subagent-delegation-patterns`.
  - `03-red-testing` → "unit tests for English" added to Purpose; References cite `red-test-contracts` and `creating-unit-tests`.
  - `04-implementing` → "constitution authority" added to Purpose; References cite `implementation-standards` and `tracker-handoff`.
  - `05-green-testing` → "unit tests for English" and "append-only convergence" added to Purpose; References cite `green-validation-gates` and `coverage-guard`.
  - `06-documenting` → "append-only convergence" added to Purpose; References cite `educational-docs` and `docs-academic-citation-audit`.
  - `07-logging` → "append-only convergence" added to Purpose; References cite `summarizing-session-log` and `capturing-learning-event`.
  - `planning-acceptance-criteria/SKILL.md` → "unit tests for English" added to the skill overview.
  - `phase-handoff-workflow/SKILL.md` → "append-only convergence" added to the skill overview.
  - `plan-sync-validation/SKILL.md` → "constitution authority" added to the skill overview.
- gate notes:
  - `plan-sync` → PASS
  - `agent-graph` → PASS
  - `routing-table-freshness` → PASS
  - `step-packet` → KNOWN FAIL in sibling plan only (pre-existing violations in `plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md`; `Spec-Kit_Assimilation.plans.md` blocks clean)
  - `validate-plan-phase-packets` → KNOWN pre-existing format issue (this plan uses YAML status markers instead of markdown headers)

### Slice 11 — Flow updates and final green validation

- green-light: true
- validator: `04-implementing`
- slice: `11-flow-updates`
- red_phase:
  - before: `node scripts/agent-customization/__red__spec-kit-slice-11.mjs` → FAIL (AC-040..AC-043 flow strings missing)
  - after: `node scripts/agent-customization/__red__spec-kit-slice-11.mjs` → PASS (AC-040..AC-046, 10/10 checks)
- evidence:
  - `node scripts/agent-customization/__red__spec-kit-slice-11.mjs` → PASS (AC-040..AC-046)
  - `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict` → PASS (0 errors, 0 warnings)
  - `node scripts/agent-customization/validate-skill-frontmatter.mjs --json` → PASS (0 errors, 0 warnings)
  - `node scripts/agent-customization/validate-agent-graph.mjs --json` → PASS (0 errors, 0 warnings)
  - `npm run agents:routing-table` → PASS (`changed=false`, routing table stable)
  - `npm run agents:routing-table:gate` → PASS (hash match)
  - `npx prettier --check .github/flows/01.phase-kickoff.flow.yml .github/flows/01.step-expansion.flow.yml .github/flows/01.step-packet-revision.flow.yml .github/flows/05.coverage-guard.flow.yml .github/flows/05.ci-green-confirmation.flow.yml .github/flows/07.tracker-closure.flow.yml .github/flows/flow.schema.yml scripts/agent-customization/__red__spec-kit-slice-11.mjs` → PASS
  - `npm run lint` → PASS (0 issues)
  - `npx tsc --noEmit -p tsconfig.json` → PASS
  - `neataptic-gate-mcp:run_gate_check gate=plan-sync` → PASS
  - `neataptic-gate-mcp:run_gate_check gate=agent-graph` → PASS
  - `neataptic-gate-mcp:run_gate_check gate=routing-table-freshness` → PASS
  - `neataptic-gate-mcp:run_gate_check gate=agent-quality` → PASS
  - `neataptic-gate-mcp:run_gate_check gate=tier-enforcement` → PASS
  - `neataptic-gate-mcp:run_gate_check gate=learning-event` → PASS
  - `neataptic-gate-mcp:run_gate_check gate=plan-slice-quality` → PASS
  - prior slice RED scripts (`__red__spec-kit-slice-1.mjs` through `slice-10.mjs`) → all PASS
- flow edits:
  - `01.phase-kickoff.flow.yml` → added `constitution_check` exit-check item while preserving the Slice-2 clarification-cap entry/exit checks
  - `01.step-expansion.flow.yml` → added `spec-checklist` to the `gates:` list
  - `01.step-packet-revision.flow.yml` → added `spec-checklist` to the `gates:` list
  - `05.coverage-guard.flow.yml` → added `constitution_check` and `spec-checklist` to the `gates:` list
  - `05.ci-green-confirmation.flow.yml` → added `constitution_check` and `spec-checklist` to the `gates:` list
  - `07.tracker-closure.flow.yml` → added an `exit-checks:` block recording `constitution-update`, `spec-checklist`, and `gate-run` learning events
  - `flow.schema.yml` → updated the gate-catalog comment to include `constitution_check` and `spec-checklist`
- gate notes:
  - `plan-sync` → PASS
  - `agent-graph` → PASS
  - `routing-table-freshness` → PASS
  - `agent-quality` → PASS
  - `tier-enforcement` → PASS
  - `learning-event` → PASS
  - `plan-slice-quality` → PASS
  - `step-packet` → KNOWN PRE-EXISTING FAIL in sibling plan only (`plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md`); not fixed in this slice

### Slice 05 — spec-checklist skill (unit tests for English)

- green-light: true
- validator: `04-implementing`
- slice: `05-spec-checklist`
- evidence:
  - `node scripts/agent-customization/__red__spec-kit-slice-5.mjs` → PASS (AC-018..AC-022)
  - `node scripts/agent-customization/validate-skill-frontmatter.mjs --json` → PASS (0 errors, 0 warnings)
  - `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` → PASS (0 errors, 0 warnings)
  - `npm run agents:routing-table` → PASS (regenerated `.github/agent-skill-routing-table.md`)
  - `npm run agents:routing-table:gate` → PASS (hash match)
  - `npx prettier --check .github/skills/spec-checklist/SKILL.md .github/agents/01-planning.agent.md .github/agents/05-green-testing.agent.md .github/skills/phase-handoff-workflow/SKILL.md scripts/agent-customization/__red__spec-kit-slice-5.mjs .github/agent-skill-routing-table.md` → PASS
- PlanUpdate:
  - slice_id: '05-spec-checklist'
    changed_files:
    - '.github/skills/spec-checklist/SKILL.md'
    - '.github/agents/01-planning.agent.md'
    - '.github/agents/05-green-testing.agent.md'
    - '.github/skills/phase-handoff-workflow/SKILL.md'
    - 'scripts/agent-customization/__red__spec-kit-slice-5.mjs'
    - '.github/agent-skill-routing-table.md'
      preflight:
    - 'npx prettier --check .github/skills/spec-checklist/SKILL.md .github/agents/01-planning.agent.md .github/agents/05-green-testing.agent.md .github/skills/phase-handoff-workflow/SKILL.md scripts/agent-customization/__red__spec-kit-slice-5.mjs .github/agent-skill-routing-table.md'
      tests_for_green:
    - 'node scripts/agent-customization/__red__spec-kit-slice-5.mjs'
    - 'node scripts/agent-customization/validate-skill-frontmatter.mjs --json --skill .github/skills/spec-checklist/SKILL.md'
    - 'node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent .github/agents/01-planning.agent.md'
    - 'node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent .github/agents/05-green-testing.agent.md'
      rollback:
    - 'git checkout -- .github/skills/spec-checklist/SKILL.md .github/agents/01-planning.agent.md .github/agents/05-green-testing.agent.md .github/skills/phase-handoff-workflow/SKILL.md scripts/agent-customization/__red__spec-kit-slice-5.mjs .github/agent-skill-routing-table.md'
      next: 'Run 05-green-testing to verify AC-018..AC-022 and frontmatter/routing-table gates, then advance per plan.'

## Scope

Cherry-pick Spec Kit governance and workflow patterns into NeatapticTS:

- A single binding `plans/constitution.md` that supersedes scattered conventions.
- A `constitution_check` field on phase/step YAML blocks and step packets.
- Clarification discipline, traceability IDs, task templates, spec checklist,
  bug triage extension, research artifacts, extension catalog, and verbatim
  phrase alignment.

This plan does **not** replace the existing agent tier graph, MCP dispatch,
step-packet gates, or ISO-42001 learning log; it augments them.

## Implementation phases

### Phase 1 — Governance foundation [DONE]

```yaml
phase: 1
title: 'Governance foundation'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_phase: 'Phase 2 — Clarification & traceability'
skills:
  - plan-alignment
  - tracker-handoff
  - phase-handoff-workflow
  - capturing-learning-event
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema'
  - 'All slices have AC-### IDs and estimate_hours ≤ 4'
constitution_check:
  - 'principle-1-ai-thinking-partner'
  - 'principle-3-verbatim-binding'
  - 'principle-5-unique-ids'
  - 'section-governance-versioning'
placeholder_steps:
  - 'Step 01 — Constitution foundation + constitution_check gate field'
  - 'Step 02 — Clarification cap & NEEDS CLARIFICATION discipline'
```

#### Step 01 — Constitution foundation + constitution_check gate field [DONE]

```yaml
phase: 1
step: 1
title: 'Constitution foundation + constitution_check gate field'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Step 02 — Clarification cap & NEEDS CLARIFICATION discipline'
skills:
  - implementation-standards
  - tracker-handoff
  - phase-handoff-workflow
  - capturing-learning-event
validation:
  - 'node scripts/agent-customization/__red__spec-kit-slice-1.mjs'
  - 'node scripts/agent-customization/validate-agent-frontmatter.mjs --json'
  - 'node scripts/agent-customization/validate-skill-frontmatter.mjs --json'
  - 'node .github/hooks/workflow-update-sync.mjs --plan=plans/Spec-Kit_Assimilation.plans.md --json'
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --plan=plans/Spec-Kit_Assimilation.plans.md --json'
acceptance_criteria:
  - 'AC-001: plans/constitution.md exists with five Core Principles and three supporting sections'
  - 'AC-002: plans/constitution.md contains "constitution authority" and a SemVer ratification block'
  - 'AC-003: 01-planning.agent.md Plan Block Schema documents constitution_check'
  - 'AC-004: phase-handoff-workflow/SKILL.md Step Packet Shape documents constitution_check'
  - 'AC-005: capturing-learning-event/SKILL.md accepts constitution-update eventType'
constitution_check:
  - 'principle-4-breadth-first-recoverable'
  - 'principle-5-unique-ids'
  - 'section-development-workflow'
slices:
  - slice_id: '01-red'
    title: 'RED script for constitution foundation'
    status: '[DONE]'
    goal: red-testing
    estimate_hours: 1
    files_to_change:
      - 'scripts/agent-customization/__red__spec-kit-slice-1.mjs'
      - 'plans/Spec-Kit_Assimilation.plans.md'
    acceptance_criteria:
      - 'AC-001..AC-005 assertions exist and fail before implementation'
    parallelizable: false
    dependencies: []
    next_slice: '01-impl'
  - slice_id: '01-impl'
    title: 'Implement constitution foundation + constitution_check gate field'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'plans/constitution.md'
      - '.github/agents/01-planning.agent.md'
      - '.github/skills/phase-handoff-workflow/SKILL.md'
      - '.github/skills/capturing-learning-event/SKILL.md'
      - '.github/ai-learning/learning-log.jsonl'
      - 'scripts/agent-customization/__red__spec-kit-slice-1.mjs'
    acceptance_criteria:
      - 'AC-001 PASS: constitution.md structure verified'
      - 'AC-002 PASS: constitution authority + SemVer block verified'
      - 'AC-003 PASS: Plan Block Schema documents constitution_check'
      - 'AC-004 PASS: Step Packet Shape documents constitution_check'
      - 'AC-005 PASS: capturing-learning-event accepts constitution-update'
    parallelizable: false
    dependencies:
      - '01-red'
    next_slice: '01-green'
  - slice_id: '01-green'
    title: 'Green validation for constitution foundation'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/Spec-Kit_Assimilation.plans.md'
      - 'scripts/agent-customization/__red__spec-kit-slice-1.mjs'
    acceptance_criteria:
      - 'RED script stays green after implementation'
      - 'Agent and skill frontmatter validators pass'
      - 'step-packet gate passes for this plan'
    parallelizable: false
    dependencies:
      - '01-impl'
    next_slice: null
```

**User instruction:** Paste this full step packet.

**Step objective:** Establish `plans/constitution.md` as the single source of
binding governance, wire `constitution_check` into plan/step schemas and the
learning-event schema, and validate with a RED script.

**Context the agent must know:**

- `spec-kit/spec-kit.md` Section 2.1 recommends adding `constitution.md` and a constitutional gate.
- `spec-kit/verbatim/.specify/memory/constitution.md` provides the five principles and supporting sections to port.
- The NeatapticTS step-packet schema lives in `.github/agents/01-planning.agent.md` §Plan Block Schemas.
- The step-packet shape is also documented in `.github/skills/phase-handoff-workflow/SKILL.md`.
- Learning events are append-only in `.github/ai-learning/learning-log.jsonl` and documented in `.github/skills/capturing-learning-event/SKILL.md`.

**Execution steps:**

1. Register this plan tracker in `plans/README.md` index.
2. Decide whether `plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md` is stale; if not, leave it in place and report.
3. Write the RED script `scripts/agent-customization/__red__spec-kit-slice-1.mjs` that asserts AC-001..AC-005.
4. Run the RED script; confirm it fails before implementation.
5. Create `plans/constitution.md` adapted from Spec Kit into NeatapticTS terms.
6. Add `constitution_check` to the Plan Block Schemas in `01-planning.agent.md`.
7. Add `constitution_check` to the Step Packet Shape field table in `phase-handoff-workflow/SKILL.md`.
8. Add `constitution-update` to the `eventType` enumeration and examples in `capturing-learning-event/SKILL.md`.
9. Append one `constitution-update` learning event to `.github/ai-learning/learning-log.jsonl`.
10. Run the RED script again; confirm it passes.
11. Run `validate-agent-frontmatter.mjs` and the `step-packet` gate; report results.

**Stop conditions:**

- Done: RED script passes, frontmatter validator passes, step-packet gate passes.
- Blocked: any gate fails and cannot be resolved within the slice boundary.
- Route-back: to `03-red-testing` if the RED script does not fail before implementation or does not pass after.

**Required validation:**

- `node scripts/agent-customization/__red__spec-kit-slice-1.mjs`
- `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent .github/agents/01-planning.agent.md`
- `neataptic-gate-mcp:run_gate_check gate=step-packet`

**Plan update requirement:** Update this tracker with changed files, validation evidence, and the next active step before ending.

**Whole-step copy rule:** The entire step block above is the prompt. Do not append a second nested `Copy-paste prompt` subsection.

### Phase 2 — Clarification & traceability [DONE]

```yaml
phase: 2
title: 'Clarification & traceability'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_phase: 'Phase 3 — Templates & checklist'
skills:
  - plan-alignment
  - planning-acceptance-criteria
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
acceptance_criteria:
  - 'Step packets authored for clarification cap and traceability IDs'
  - 'All acceptance criteria carry AC-### IDs'
constitution_check:
  - 'agent-contract-consistency'
placeholder_steps:
  - 'Step 01 — Clarification cap & NEEDS CLARIFICATION discipline'
  - 'Step 02 — Traceability IDs and coverage table'
```

#### Step 02: Clarification cap & NEEDS CLARIFICATION discipline [DONE]

```yaml
phase: 2
step: 1
title: 'Clarification cap & NEEDS CLARIFICATION discipline'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Step 02 — Traceability IDs and coverage table'
skills:
  - implementation-standards
  - planning-acceptance-criteria
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=step-packet'
acceptance_criteria:
  - 'AC-006: 01-planning.agent.md documents the ≤5 clarification-question cap'
  - 'AC-007: 01-planning.agent.md documents the ≤3 NEEDS CLARIFICATION marker cap'
constitution_check:
  - 'agent-contract-consistency'
slices:
  - slice_id: '02-clarification-cap'
    title: 'Clarification cap & NEEDS CLARIFICATION discipline'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - '.github/agents/01-planning.agent.md'
    acceptance_criteria:
      - 'AC-006 PASS: clarification-question cap documented'
      - 'AC-007 PASS: NEEDS CLARIFICATION marker cap documented'
    parallelizable: false
    dependencies: []
    next_slice: null
```

#### Step 03: Traceability IDs and coverage table [DONE]

```yaml
phase: 2
step: 2
title: 'Traceability IDs and coverage table'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Phase 3 Step 01 — tasks-template.md'
skills:
  - implementation-standards
  - planning-acceptance-criteria
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=step-packet'
acceptance_criteria:
  - 'AC-008: planning-acceptance-criteria/SKILL.md requires optional id: AC-###'
  - 'AC-009: phase-handoff-workflow/SKILL.md documents traceability table field'
constitution_check:
  - 'agent-contract-consistency'
  - 'test-backed-change'
slices:
  - slice_id: '03-traceability-ids'
    title: 'Traceability IDs and coverage table'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - '.github/skills/planning-acceptance-criteria/SKILL.md'
      - '.github/skills/phase-handoff-workflow/SKILL.md'
    acceptance_criteria:
      - 'AC-008 PASS: acceptance criteria id field documented'
      - 'AC-009 PASS: traceability table field documented'
    parallelizable: false
    dependencies:
      - '02-clarification-cap'
    next_slice: null
```

### Phase 3 — Templates & checklist [DONE]

```yaml
phase: 3
title: 'Templates & checklist'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_phase: 'Phase 4 — Bug triage & research artifacts'
skills:
  - plan-alignment
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
acceptance_criteria:
  - 'tasks-template.md created with phases and [P] markers'
  - 'spec-checklist skill created with ≥80% traceability rule'
constitution_check:
  - 'agent-contract-consistency'
placeholder_steps:
  - 'Step 01 — tasks-template.md with [P] parallel markers'
  - 'Step 02 — spec-checklist skill (unit tests for English)'
```

#### Step 04: tasks-template.md with [P] parallel markers [DONE]

```yaml
phase: 3
step: 1
title: 'tasks-template.md with [P] parallel markers'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Step 02 — spec-checklist skill'
skills:
  - implementation-standards
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=step-packet'
acceptance_criteria:
  - 'AC-010: .github/templates/tasks-template.md exists with Setup/Foundational/Story/Polish phases'
  - 'AC-011: tasks-template.md requires [P] tags for parallelizable tasks'
constitution_check:
  - 'agent-contract-consistency'
slices:
  - slice_id: '04-tasks-template'
    title: 'tasks-template.md with [P] parallel markers'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - '.github/templates/tasks-template.md'
    acceptance_criteria:
      - 'AC-010 PASS: tasks-template.md created with required phases'
      - 'AC-011 PASS: [P] parallel marker documented'
    parallelizable: false
    dependencies: []
    next_slice: null
```

#### Step 05: spec-checklist skill (unit tests for English) [DONE]

```yaml
phase: 3
step: 2
title: 'spec-checklist skill (unit tests for English)'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Phase 4 Step 01 — Bug triage extension path'
skills:
  - implementation-standards
  - green-validation-gates
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=step-packet'
  - 'node scripts/agent-customization/validate-skill-frontmatter.mjs --json --skill .github/skills/spec-checklist/SKILL.md'
acceptance_criteria:
  - 'AC-018: .github/skills/spec-checklist/SKILL.md exists with "unit tests for English" phrase'
  - 'AC-019: spec-checklist defines the four gap types (missing, partial, contradicts, unrequested)'
  - 'AC-020: spec-checklist requires >= 80% traceability before 04-implementing dispatch'
  - 'AC-021: 01-planning.agent.md and 05-green-testing.agent.md list spec-checklist in skills'
  - 'AC-022: phase-handoff-workflow/SKILL.md lists spec-checklist as a pre-implementation gate'
constitution_check:
  - 'test-backed-change'
  - 'agent-contract-consistency'
slices:
  - slice_id: '05-spec-checklist'
    title: 'spec-checklist skill (unit tests for English)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - '.github/skills/spec-checklist/SKILL.md'
      - '.github/agents/01-planning.agent.md'
      - '.github/agents/05-green-testing.agent.md'
      - '.github/skills/phase-handoff-workflow/SKILL.md'
      - '.github/agent-skill-routing-table.md'
      - 'scripts/agent-customization/__red__spec-kit-slice-5.mjs'
    acceptance_criteria:
      - 'AC-018 PASS: spec-checklist skill created with required phrase'
      - 'AC-019 PASS: four gap types documented'
      - 'AC-020 PASS: >= 80% traceability rule documented'
      - 'AC-021 PASS: spec-checklist wired into 01-planning and 05-green-testing'
      - 'AC-022 PASS: spec-checklist listed as pre-implementation gate in phase-handoff-workflow'
    parallelizable: false
    dependencies:
      - '04-tasks-template'
    next_slice: null
```

### Phase 4 — Bug triage & research artifacts [DONE]

```yaml
phase: 4
title: 'Bug triage & research artifacts'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_phase: 'Phase 5 — Extension catalog & verbatim phrases'
skills:
  - plan-alignment
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
acceptance_criteria:
  - 'Bug triage artifact path defined'
  - '02-researching required to write docs/research/<feature>.md'
constitution_check:
  - 'agent-skill-discipline'
  - 'test-backed-change'
placeholder_steps:
  - 'Step 01 — Bug triage extension path and skills'
  - 'Step 02 — research.md artifact for 02-researching'
```

#### Step 06: Bug triage extension path and skills [DONE]

```yaml
phase: 4
step: 1
title: 'Bug triage extension path and skills'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Step 02 — research.md artifact for 02-researching'
skills:
  - implementation-standards
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/__red__spec-kit-slice-7.mjs'
  - 'node scripts/agent-customization/validate-skill-frontmatter.mjs --json'
  - 'npm run agents:routing-table:gate'
acceptance_criteria:
  - 'AC-026: .github/bugs/README.md exists and documents the assess → fix → test workflow'
  - 'AC-027: .github/bugs/README.md specifies the artifact layout under .github/bugs/<slug>/ and URL-trust/evidence rules'
  - 'AC-028: .github/skills/bug-triage/SKILL.md exists with valid frontmatter and defines bug-assess, bug-fix, and bug-test phases'
  - 'AC-029: bug-triage/SKILL.md references the four gap types (missing, partial, contradicts, unrequested) and requires a regression test before closing a bug'
constitution_check:
  - 'agent-skill-discipline'
  - 'test-backed-change'
slices:
  - slice_id: '07-bug-triage'
    title: 'Bug triage extension path and optional skill set'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - '.github/bugs/README.md'
      - '.github/skills/bug-triage/SKILL.md'
      - 'scripts/agent-customization/__red__spec-kit-slice-7.mjs'
      - '.github/agent-skill-routing-table.md'
    acceptance_criteria:
      - 'AC-026 PASS: .github/bugs/README.md exists and documents the assess → fix → test workflow'
      - 'AC-027 PASS: .github/bugs/README.md specifies the artifact layout under .github/bugs/<slug>/ and URL-trust/evidence rules'
      - 'AC-028 PASS: .github/skills/bug-triage/SKILL.md exists with valid frontmatter and defines bug-assess, bug-fix, and bug-test phases'
      - 'AC-029 PASS: bug-triage/SKILL.md references the four gap types (missing, partial, contradicts, unrequested) and requires a regression test before closing a bug'
    parallelizable: false
    dependencies: []
    next_slice: null
```

#### Step 07: research.md artifact for 02-researching [DONE]

```yaml
phase: 4
step: 2
title: 'research.md artifact for 02-researching'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Phase 5 Step 01 — Extension catalog README'
skills:
  - implementation-standards
  - research-methodology
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=step-packet'
acceptance_criteria:
  - 'AC-023: 02-researching.agent.md requires writing docs/research/<feature>.md for any research step that resolves unknowns'
  - 'AC-024: 02-researching.agent.md requires the produced step packet to link to the research artifact via research_artifact field'
  - 'AC-025: docs/research/README.md exists and documents naming convention (<feature-slug>.md) plus Question, Evidence, Decision, Risks sections'
constitution_check:
  - 'agent-skill-discipline'
  - 'agent-contract-consistency'
slices:
  - slice_id: '07-research-artifact'
    title: 'research.md artifact for 02-researching'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - '.github/agents/02-researching.agent.md'
      - 'docs/research/README.md'
      - '.gitignore'
      - 'scripts/agent-customization/__red__spec-kit-slice-6.mjs'
    acceptance_criteria:
      - 'AC-023 PASS: 02-researching.agent.md mentions docs/research/ requirement'
      - 'AC-024 PASS: 02-researching.agent.md mentions step-packet research_artifact link'
      - 'AC-025 PASS: docs/research/README.md exists with <feature-slug>.md naming and Question, Evidence, Decision, Risks sections'
    parallelizable: false
    dependencies:
      - '07-bug-triage'
    next_slice: null
```

### Phase 5 — Extension catalog & verbatim phrases [DONE]

```yaml
phase: 5
title: 'Extension catalog & verbatim phrases'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_phase: 'Phase 6 — Learning-event schema extension'
skills:
  - plan-alignment
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
acceptance_criteria:
  - 'Extension catalog pattern published'
  - 'Verbatim Spec Kit phrases adopted in local descriptions'
constitution_check:
  - 'agent-skill-discipline'
  - 'agent-contract-consistency'
placeholder_steps:
  - 'Step 01 — Extension catalog README'
  - 'Step 02 — Verbatim phrase adoption'
```

#### Step 08: Extension catalog README [DONE]

```yaml
phase: 5
step: 1
title: 'Extension catalog README'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Step 02 — Verbatim phrase adoption'
skills:
  - implementation-standards
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=step-packet'
acceptance_criteria:
  - 'AC-018: .github/extensions/README.md maps skills/agents/presets to Spec Kit concepts'
  - 'AC-019: Catalog documents extension.yml, commands, templates, and presets'
constitution_check:
  - 'agent-skill-discipline'
slices:
  - slice_id: '08-extension-catalog'
    title: 'Extension catalog README'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - '.github/extensions/README.md'
    acceptance_criteria:
      - 'AC-018 PASS: extension catalog maps local concepts'
      - 'AC-019 PASS: catalog documents composition model'
    parallelizable: false
    dependencies: []
    next_slice: null
```

#### Step 09: Verbatim phrase adoption [DONE]

```yaml
phase: 5
step: 2
title: 'Verbatim phrase adoption'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Phase 6 Step 01 — Learning-event schema extension'
skills:
  - implementation-standards
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=step-packet'
acceptance_criteria:
  - 'AC-020: At least three agent/skill descriptions use Spec Kit phrases ("unit tests for English", "append-only convergence", "constitution authority", or gap types)'
  - 'AC-021: Phrase usage is accurate and not decorative'
  - 'AC-036: Every Tier-1 agent contains at least one exact Spec-Kit phrase'
  - 'AC-037: planning-acceptance-criteria, phase-handoff-workflow, and plan-sync-validation skills each contain at least one exact Spec-Kit phrase'
  - 'AC-038: All eight Tier-1 agents share consistent frontmatter (tier: 1, user-invocable: true, disable-model-invocation: false, model present)'
  - 'AC-039: Every Tier-1 agent contains at least one Reference: citation to a canonical skill'
constitution_check:
  - 'agent-contract-consistency'
slices:
  - slice_id: '09-verbatim-phrases'
    title: 'Verbatim phrase adoption'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - '.github/agents/00-helping.agent.md'
      - '.github/agents/01-planning.agent.md'
      - '.github/agents/02-researching.agent.md'
      - '.github/agents/03-red-testing.agent.md'
      - '.github/agents/04-implementing.agent.md'
      - '.github/agents/05-green-testing.agent.md'
      - '.github/agents/06-documenting.agent.md'
      - '.github/agents/07-logging.agent.md'
      - '.github/skills/planning-acceptance-criteria/SKILL.md'
      - '.github/skills/phase-handoff-workflow/SKILL.md'
      - '.github/skills/plan-sync-validation/SKILL.md'
      - 'scripts/agent-customization/__red__spec-kit-slice-10.mjs'
      - '.github/agent-skill-routing-table.md'
    acceptance_criteria:
      - 'AC-036 PASS: every Tier-1 agent contains a Spec-Kit phrase'
      - 'AC-037 PASS: all three target skills contain a Spec-Kit phrase'
      - 'AC-038 PASS: all eight Tier-1 agents have consistent frontmatter'
      - 'AC-039 PASS: every Tier-1 agent has at least one Reference: citation'
    parallelizable: false
    dependencies:
      - '08-extension-catalog'
    next_slice: null
```

### Phase 6 — Learning-event schema extension [DONE]

```yaml
phase: 6
title: 'Learning-event schema extension'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_phase: 'Phase 7 — Close-out / convergence'
skills:
  - plan-alignment
  - capturing-learning-event
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
acceptance_criteria:
  - 'Learning-event schema extended to cover constitution, spec-checklist, and gate events'
constitution_check:
  - 'agent-skill-discipline'
  - 'agent-contract-consistency'
placeholder_steps:
  - 'Step 01 — Extend learning-event eventType enumeration and examples'
```

#### Step 10: Extend learning-event eventType enumeration and examples [DONE]

```yaml
phase: 6
step: 1
title: 'Extend learning-event eventType enumeration and examples'
status: '[DONE]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: 'Phase 7 Step 01 — Convergence review'
skills:
  - implementation-standards
  - capturing-learning-event
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=step-packet'
acceptance_criteria:
  - 'AC-022: capturing-learning-event/SKILL.md adds spec-checklist eventType'
  - 'AC-023: capturing-learning-event/SKILL.md adds gate-run eventType with example JSONL'
constitution_check:
  - 'agent-skill-discipline'
  - 'agent-contract-consistency'
slices:
  - slice_id: '10-learning-event-schema'
    title: 'Extend learning-event eventType enumeration and examples'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - '.github/skills/capturing-learning-event/SKILL.md'
    acceptance_criteria:
      - 'AC-022 PASS: spec-checklist eventType added'
      - 'AC-023 PASS: gate-run eventType with example added'
    parallelizable: false
    dependencies: []
    next_slice: null
```

### Phase 7 — Close-out / convergence [DONE]

```yaml
phase: 7
title: 'Close-out / convergence'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_phase: null
skills:
  - plan-alignment
  - capturing-learning-event
  - tracker-handoff
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Spec-Kit_Assimilation.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
acceptance_criteria:
  - 'All prior phases marked [DONE]'
  - 'Plan compressed and archived with matching .logs.md'
constitution_check:
  - 'agent-skill-discipline'
  - 'agent-contract-consistency'
placeholder_steps:
  - 'Step 01 — Convergence review and archive'
```

#### Step 11: Convergence review and archive [DONE]

```yaml
phase: 7
step: 1
title: 'Convergence review and archive'
status: '[DONE]'
goal: logging
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/Spec-Kit_Assimilation.plans.md
copy_paste: true
next_step: null
skills:
  - tracker-handoff
  - capturing-learning-event
validation:
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
acceptance_criteria:
  - 'AC-024: Plan pair moved to plans/completed/ with matching .logs.md'
  - 'AC-025: Final constitution-update learning event appended'
constitution_check:
  - 'agent-skill-discipline'
  - 'agent-contract-consistency'
```

## Phase 7 Step 11 — archive status (corrected 2026-07-04)

- Steps 01–10: [DONE] — all implementation slices verified green (RED scripts 1–11 pass; artifacts present; step-packet/plan-sync/routing-table-freshness gates PASS).
- Step 11 (Convergence review and archive): [WIP] — NOT yet executed.
  - AC-024 (move plan pair to plans/completed/ with .logs.md): pending.
  - AC-025 (final constitution-update learning event): pending.
- Note: an earlier stale note here read "No execution-phase dispatch yet; plan verification pending." — that was incorrect; all execution-phase work for Steps 01–10 is complete and verified.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Workstream: Spec-Kit Assimilation — cherry-pick Spec Kit governance and workflow patterns into NeatapticTS.
Active slice: 11-archive in Phase 7 Step 11.
Already covered: Steps 01–10 [DONE] — all 11 implementation slices verified green (RED scripts 1–11 pass; constitution.md, spec-checklist skill, bug-triage skill, tasks-template, bugs/README, extensions/README, docs/research/README all present; 8 Tier-1 agents polished; 6 flow files + schema updated; routing table regenerated; constitution-update learning event appended).
Next narrow task: complete Step 11 archive — create plans/Spec-Kit_Assimilation.logs.md capturing all slice evidence, move the plan pair (plans/Spec-Kit_Assimilation.plans.md + .logs.md) to plans/completed/, update plans/README.md + plans/Roadmap.md references to point to the completed path, append a final constitution-update learning event, then mark the plan **Status:** [DONE].
Required validations:
  - node scripts/agent-customization/__red__spec-kit-slice-11.mjs (must PASS — confirms archive artifacts exist)
  - neataptic-gate-mcp:run_gate_check gate=step-packet (must PASS after Step 11 marked [DONE] — removes last WIP block from this plan)
  - neataptic-gate-mcp:run_gate_check gate=plan-sync (must PASS — confirm completed path registered or plan removed from WIP list)
Known cautions: do not archive plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md; it has active work. Do not delete any measurements or evidence when creating the .logs.md.
```
