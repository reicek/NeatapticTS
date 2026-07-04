# Spec-Kit Assimilation Log

**Status:** [DONE]

## Audit scope

- Objective: cherry-pick Spec Kit governance and workflow patterns into
  NeatapticTS while preserving the existing agent tier graph, MCP dispatch,
  step-packet gates, skill-first customization, and ISO-42001 learning log.
- Bounded surface: `plans/constitution.md`, task templates, spec-checklist and
  bug-triage skills, bug/extension/research catalogs, verbatim phrase
  alignment across Tier-1 agents, flow YAML gate wiring, 11 RED scripts, and
  the `.github/ai-learning/learning-log.jsonl` constitution-update event.
- Closure pass surfaces: `plans/Spec-Kit_Assimilation.plans.md`,
  `plans/Spec-Kit_Assimilation.logs.md`, `plans/README.md`,
  `plans/Roadmap.md`, and `plans/completed/README.md`.

## Durable milestones

### [DONE] Phase 1 — Governance foundation

- Created `plans/constitution.md` with five Core Principles and three
  supporting sections, including a SemVer ratification block and explicit
  "constitution authority" language.
- Wired `constitution_check` into the plan/step YAML schema and the step-packet
  shape documented in `phase-handoff-workflow/SKILL.md`.
- Extended `capturing-learning-event/SKILL.md` to accept `constitution-update`
  as an eventType.
- Added `__red__spec-kit-slice-1.mjs` to assert AC-001..AC-005.

### [DONE] Phase 2 — Clarification & traceability

- Implemented the clarification cap and `NEEDS CLARIFICATION` discipline in
  `01-planning.agent.md` and the step-packet schema.
- Added traceability ID requirements (`AC-###`) to every acceptance criterion
  and slice, enforced by RED scripts and plan-phase-step validators.

### [DONE] Phase 3 — Task templates and research artifacts

- Created `.github/templates/tasks-template.md` as the canonical task packet
  template.
- Established `docs/research/README.md` as the durable research-artifact index.
- Validated template presence and research README structure via RED script.

### [DONE] Phase 4 — Bug triage and extension catalog

- Created `.github/skills/bug-triage/SKILL.md` with triage decision trees and
  ownership rules.
- Created `.github/bugs/README.md` and `.github/extensions/README.md` as
  boundary indexes for known bugs and planned extensions.
- Validated with `__red__spec-kit-slice-{4,7,8}.mjs`.

### [DONE] Phase 5 — Spec-checklist skill

- Created `.github/skills/spec-checklist/SKILL.md` defining the "unit tests for
  English" checklist and traceability-ID enforcement.
- Updated `01-planning.agent.md` and `05-green-testing.agent.md` to reference
  the spec-checklist skill.
- Regenerated `.github/agent-skill-routing-table.md`.

### [DONE] Phase 6 — Learning-event schema expansion

- Expanded `capturing-learning-event/SKILL.md` with `spec-checklist` and
  `gate-run` eventTypes and canonical examples.
- Polished `07-logging.agent.md` to reference the expanded schema and the
  `summarizing-session-log` skill.
- Validated frontmatter and skill metadata.

### [DONE] Phase 7 — Close-out / convergence

- Archived the plan pair into `plans/completed/`.
- Updated `plans/README.md`, `plans/Roadmap.md`, and
  `plans/completed/README.md` to point to the completed path.
- Appended the final `constitution-update` learning event.

## Controls and evidence

- `node scripts/agent-customization/__red__spec-kit-slice-1.mjs` → PASS (AC-001..AC-005)
- `node scripts/agent-customization/__red__spec-kit-slice-2.mjs` → PASS (AC-006..AC-010)
- `node scripts/agent-customization/__red__spec-kit-slice-3.mjs` → PASS (AC-011..AC-013)
- `node scripts/agent-customization/__red__spec-kit-slice-4.mjs` → PASS (AC-014..AC-016)
- `node scripts/agent-customization/__red__spec-kit-slice-5.mjs` → PASS (AC-018..AC-022)
- `node scripts/agent-customization/__red__spec-kit-slice-6.mjs` → PASS (AC-023..AC-025)
- `node scripts/agent-customization/__red__spec-kit-slice-7.mjs` → PASS
- `node scripts/agent-customization/__red__spec-kit-slice-8.mjs` → PASS
- `node scripts/agent-customization/__red__spec-kit-slice-9.mjs` → PASS (AC-033..AC-035)
- `node scripts/agent-customization/__red__spec-kit-slice-10.mjs` → PASS (AC-036..AC-039, 27/27 checks)
- `node scripts/agent-customization/__red__spec-kit-slice-11.mjs` → PASS (AC-040..AC-046, 10/10 checks)
- `node scripts/agent-customization/gates/step-packet.gate.mjs --json` → PASS
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json` → PASS
- `neataptic-gate-mcp:run_gate_check gate=plan-sync` → PASS
- `neataptic-gate-mcp:run_gate_check gate=agent-graph` → PASS
- `neataptic-gate-mcp:run_gate_check gate=routing-table-freshness` → PASS
- `neataptic-gate-mcp:run_gate_check gate=agent-quality` → PASS
- `neataptic-gate-mcp:run_gate_check gate=tier-enforcement` → PASS
- `neataptic-gate-mcp:run_gate_check gate=learning-event` → PASS
- `neataptic-gate-mcp:run_gate_check gate=plan-slice-quality` → PASS

Artifacts delivered:

- `plans/constitution.md`
- `.github/templates/tasks-template.md`
- `.github/skills/spec-checklist/SKILL.md`
- `.github/skills/bug-triage/SKILL.md`
- `.github/bugs/README.md`
- `.github/extensions/README.md`
- `docs/research/README.md`
- 11 RED scripts: `scripts/agent-customization/__red__spec-kit-slice-{1..11}.mjs`
- 8 polished Tier-1 agents (`00-helping` through `07-logging`)
- 6 flow YAML files + `flow.schema.yml` updated with
  `constitution_check`/`spec-checklist` gates
- `.github/agent-skill-routing-table.md` regenerated
- `constitution-update` learning event appended to
  `.github/ai-learning/learning-log.jsonl`

## Reopen triggers

- Spec Kit practices need amendment or additional phrase adoption.
- Constitution ratification is revised.
- A new plan needs to reuse the Spec-Kit Assimilation artifact baseline.
