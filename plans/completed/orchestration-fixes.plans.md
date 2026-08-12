# Orchestration System Fixes

**Status:** [DONE]

## Scope

This plan addresses twelve friction points discovered across fix-loop iterations (r5→r8) in the agentic workflow orchestration system. The work is confined to workflow skills, agent frontmatter, gate scripts, and plan-authoring conventions. It does **not** change core NEAT algorithms, public library APIs, or demo-specific logic.

The twelve issues are grouped into four phases:

- **Phase A — Policy & Gate Changes:** fix-severity classifier, single-expect relaxation, `04-implementing` targeted-test allowance, pre-specialist smoke gate, fix-loop convergence tracking.
- **Phase B — Infrastructure Fixes:** Cortex index auto-rebuild, slice context retention, test selection stored as file paths, plan-command-lint gate.
- **Phase C — Specialist Workflow Redesign:** shared validation phase, RAG-based fix-packet design, dispatch MCP prompt-length heuristic.
- **Phase D — Documentation & Concurrency:** concurrency clarification (real limit is 10, nested dispatch supported).

## Final state

Claim: Agent Zero @ 2026-07-29T01:38:00Z — plan fully complete and closed. All four phases (A–D) are [DONE] and compressed to `plans/completed/orchestration-fixes.logs.md`.

- Phase A: [DONE] — all 6 steps complete and compressed to logs.
- Phase B: [DONE] — all 5 steps (01–05) green-validated and compressed to `plans/completed/orchestration-fixes.logs.md`.
- Phase C: [DONE] — all 4 steps green-validated and compressed to logs. Specialist review rebalanced (1 max, 0 for TRIVIAL); master slice-advancement gate created; RAG fix-packet convention designed; dispatch MCP prompt-length guard implemented.
- Phase D: [DONE] — Step 01 [DONE] and Step 02 all slices [DONE]. Documentation-only work (TRIVIAL severity — specialist review skipped). Shared-validation gate pass; 05-green-testing GREEN: OK (AC-D2-001–AC-D2-009 verified, lint/prettier/plan-sync/workflow-sync all pass).

No active frontier remains; plan is ready for archive to `plans/completed/`.

## Latest validation evidence

- Phase D Step 01: [DONE] — Step 02 packet authored; step-packet gate pass (0 violations), plan-slice-quality gate pass (0 violations). AC-D1-001 and AC-D1-002 satisfied.
- Phase D Step 02: [DONE] — D2-concurrency-docs [DONE] (execute/SKILL.md §5.5 reframed). D2-rag-exception-docs [DONE] (execute/SKILL.md §5.8 controlled-deviation subsection added). D2-green [DONE] — 05-green-testing GREEN: OK.
- Phase D green evidence: `npm run lint` → 0 errors, 44 pre-existing warnings; `npx prettier --check` both skill files → pass; validate-skill-frontmatter → ok: true; validate-plan-sync → pass: true; workflow-update-sync → pass: true; plan-sync gate → pass: true. AC-D2-001–AC-D2-009 all verified.
- fix-loop: C3-design-rag-fix-packet iteration 1 status=passed — design edits applied to execute and tracker-handoff skills; preflight and plan sync gates all pass.
- Gate confirmation C3-design-rag-fix-packet: workflow-update-sync pass; validate-plan-sync pass (0 errors, 0 warnings); step-packet pass (0 violations); plan-slice-quality pass; plan-command-lint pass (0 issues).
- green-light: true — Phase C plan readiness re-verified after Step 04 insertion; step-packet, plan-slice-quality, plan-command-lint, and validate-plan-sync gates all pass.
- Phase A: [DONE] — all steps compressed to `plans/completed/orchestration-fixes.logs.md`.
- Phase B: [DONE] — all five steps green-validated and compressed to `plans/completed/orchestration-fixes.logs.md`.
- Phase C: [DONE] — all four steps green-validated; specialist review rebalanced (1 max, 0 for TRIVIAL); master slice-advancement gate created; RAG fix-packet convention designed; dispatch MCP prompt-length guard implemented. Compressed to logs.
- Phase D: [DONE] — all steps green-validated; plan fully complete.

## Source of truth

- This file: `plans/orchestration-fixes.plans.md`
- Constitution: `plans/constitution.md`
- Routing table: `.github/agent-skill-routing-table.md`
- Active workstream: `plans/Neon_Shooter_NGE_Demo.plans.md` (downstream beneficiary)

## Plan-wide acceptance criteria

- id: AC-PLAN-001
  text: 'Every phase has Step 01–N packets with valid YAML per the step-packet schema.'
  validation: 'plans/orchestration-fixes.plans.md'
- id: AC-PLAN-002
  text: 'No step contains more than 5 slices and no slice exceeds 4 hours.'
  validation: 'scripts/agent-customization/gates/plan-slice-quality.gate.mjs'
- id: AC-PLAN-003
  text: 'Validation entries in the plan reference file paths, not Jest CLI flags (Issue 5 fix applied to the plan itself).'
  validation: 'plans/orchestration-fixes.plans.md'
- id: AC-PLAN-004
  text: 'Plan is registered in plans/README.md and placed in plans/Roadmap.md under the Meta-Workflow lane.'
  validation: 'plans/README.md'
- id: AC-PLAN-005
  text: 'All issues have at least one slice, step, or explicit skipped-record mapping.'
  validation: 'plans/orchestration-fixes.plans.md'

## Non-goals

- Re-architecting the five-tier delegation graph.
- Changing the model-routing table or agent frontmatter models.
- Modifying core NEAT/network code.
- Removing the specialist review gate entirely; only tuning its application.

## Open assumptions

- The repo will continue using Jest as the primary test runner.
- Cortex RAG remains backed by the Turso/libSQL pipeline.
- The real Copilot CLI concurrent agent limit is 10 (per Issue 12 clarification).

## Implementation phases

### Phase A — Policy & Gate Changes [DONE]

**Phase objective:** Update workflow policy and agent skills so trivial fixes move faster, tests can encode related assertions pragmatically, `04-implementing` can self-check with targeted tests, and fix loops escalate before they oscillate.

Phase A is complete. Detailed YAML packets, slice records, PlanUpdate history, and validation evidence for all six steps moved to `plans/completed/orchestration-fixes.logs.md`.

#### Step 01: Plan Phase A step packets [DONE]

[DONE] Step 01: Plan Phase A step packets. Full YAML packet, acceptance criteria, and validation evidence moved to `plans/completed/orchestration-fixes.logs.md`.

#### Step 02: Specialist review severity classifier [DONE]

[DONE] Step 02: Specialist review severity classifier implemented and green validated. See `plans/completed/orchestration-fixes.logs.md` for the full YAML packet, acceptance criteria, slice details, and validation evidence.

#### Step 03: Relax single-expect rule [DONE]

[DONE] Step 03: Relax single-expect rule implemented and green validated. See `plans/completed/orchestration-fixes.logs.md` for the full YAML packet, acceptance criteria, slice details, PlanUpdate history, and validation evidence.

#### Step 04: Allow 04-implementing targeted test runs [DONE]

[DONE] Step 04: Allow 04-implementing targeted test runs. All acceptance criteria AC-A4-001 through AC-A4-010 pass. See `plans/completed/orchestration-fixes.logs.md` for the full YAML packet, acceptance criteria, slice details, PlanUpdate history, and validation evidence.

#### Step 05: Add pre-specialist smoke gate [DONE]

[DONE] Step 05: Add pre-specialist smoke gate implemented and green validated. See `plans/completed/orchestration-fixes.logs.md` for the full YAML packet, acceptance criteria, slice details, PlanUpdate history, and validation evidence.

#### Step 06: Add fix-loop convergence tracking [DONE]

[DONE] Step 06: Add fix-loop convergence tracking. 24/24 focused tests pass, 100% coverage on `scripts/agent-customization/gates/convergence-tracker.gate.mjs`. Full YAML packet, slice details, PlanUpdate history, and validation evidence moved to `plans/completed/orchestration-fixes.logs.md`.

---

### Phase B — Infrastructure Fixes [DONE]

**Phase objective:** Harden the RAG/infrastructure layer so Cortex stays fresh, completed slices remain retrievable, and plan-stored validation uses stable file paths instead of brittle CLI flags.

[DONE] Phase B complete. All five steps (01–05) validated and compressed. Full phase-level YAML packet, step packets, acceptance criteria, slices, traceability, PlanUpdate history, and validation evidence moved to `plans/completed/orchestration-fixes.logs.md`.

#### Step 01: Plan Phase B step packets [DONE]

[DONE] Step 01: Phase B step packets (Steps 02–05) authored and validated via `step-packet` and `plan-slice-quality` gates. Full YAML packet, acceptance criteria, and validation evidence moved to `plans/completed/orchestration-fixes.logs.md`.

#### Step 02: Cortex index auto-rebuild/freshness gate [DONE]

[DONE] Step 02: Cortex index auto-rebuild/freshness gate implemented and green validated. See plans/completed/orchestration-fixes.logs.md for the full YAML packet, acceptance criteria, slice details, validation evidence, and fix-loop history.

#### Step 03: Slice context retention [DONE]

[DONE] Step 03: Slice context retention implemented and green validated (B3-green-r5 via 00-helping convergence escalation). See plans/completed/orchestration-fixes.logs.md for the full YAML packet, acceptance criteria, traceability, slice details, validation evidence, and fix-loop history.

#### Step 04: Test selection as file paths [DONE]

[DONE] Step 04: Test selection stored as repo-relative file paths; validator updated, plan entries converted, green validation passed. Full YAML packet, slices, traceability, and evidence moved to plans/completed/orchestration-fixes.logs.md.

#### Step 05: Plan-command-lint gate [DONE]

[DONE] Step 05: Plan-command-lint gate implemented and green validated; 100% coverage on `scripts/agent-customization/gates/plan-command-lint.gate.mjs`. Full YAML packet, acceptance criteria, traceability, slices, and validation evidence moved to `plans/completed/orchestration-fixes.logs.md`.

---

### Phase C — Specialist Workflow Redesign [DONE]

**Phase objective:** Reduce specialist-review duplication and close the RAG gap for fix-loop packets by introducing a shared validation phase and a RAG-fix-packet convention.

[DONE] Phase C complete. All four steps (01–04) green-validated and compressed to `plans/completed/orchestration-fixes.logs.md`.

#### Step 01: Plan Phase C step packets [DONE]

[DONE] Step 01: Phase C Step 02–04 packets authored; step-packet and plan-slice-quality gates pass. Full packet moved to logs.

#### Step 02: Shared validation phase [DONE]

[DONE] Step 02: Shared validation phase implemented and green validated. `shared-validation.gate.mjs` passes 37/37 tests with 100% coverage; `specialist-review-workflow` and `execute` skills updated. Full packet moved to logs.

#### Step 03: RAG-based fix-packet design [DONE]

[DONE] Step 03: RAG-based fix-packet convention documented in `execute` and `tracker-handoff` skills; example fix-packet block added and validated. Full packet moved to logs.

#### Step 04: Dispatch MCP prompt-length heuristic [DONE]

[DONE] Step 04: Dispatch MCP `build_dispatch_packet` prompt-length guard (PROMPT_LENGTH_MAX=200) implemented; red tests 11/11 pass; self-check pass; `execute` skill updated. Full packet moved to logs.

---

### Phase D — Documentation & Concurrency [DONE]

**Phase objective:** Document that the real concurrent agent limit is 10 and that nested dispatch is supported; clarify RAG principle exceptions for fix packets.

[DONE] Phase D complete. Step 01 packet authored and Step 02 all slices green-validated. Full packets moved to `plans/completed/orchestration-fixes.logs.md`.

#### Step 01: Plan Phase D step packets [DONE]

[DONE] Step 01: Phase D Step 02 packet authored; step-packet and plan-slice-quality gates pass. Full packet moved to logs.

#### Step 02: Concurrency clarification and RAG exception docs [DONE]

[DONE] Step 02: `execute` skill concurrency section reframed (limit=10, nested dispatch supported); `subagent-delegation-patterns` skill concurrency note added; RAG fix-packet exception documented. Green validation passed. Full packet moved to logs.

---

## Validation gates

All phase work must pass the following gates before being marked [DONE]:

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/completed/orchestration-fixes.plans.md`
- `node scripts/agent-customization/gates/step-packet.gate.mjs --json`
- `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json`
- `node scripts/agent-customization/gates/plan-command-lint.gate.mjs --json --plan=plans/completed/orchestration-fixes.plans.md`

## Issue-to-slice mapping

| Issue | Suggested remediation                                       | Phase | Step | Slice(s)                                                                     |
| ----- | ----------------------------------------------------------- | ----- | ---- | ---------------------------------------------------------------------------- |
| 1     | Specialist review severity classifier (1 vs 3+ specialists) | A     | 02   | A2-red-tests, A2-impl, A2-green                                              |
| 4     | Relax single-expect rule                                    | A     | 03   | A3-relax-red-contracts, A3-relax-creating-unit-tests, A3-green               |
| 6     | Allow `04-implementing` targeted test runs                  | A     | 04   | A4-update-execute-skill, A4-update-impl-standards, A4-update-agent, A4-green |
| 7     | Pre-specialist smoke gate                                   | A     | 05   | A5-red-tests, A5-impl, A5-green                                              |
| 8     | Fix-loop convergence tracking                               | A     | 06   | A6-red-tests, A6-impl, A6-green                                              |
| 3     | Cortex index auto-rebuild/freshness gate                    | B     | 02   | B2-red-tests, B2-impl, B2-green                                              |
| 2     | Slice context retention                                     | B     | 03   | B3-red-tests, B3-impl, B3-green                                              |
| 5     | Test selection as file paths                                | B     | 04   | B4-impl-validator, B4-apply-to-plan, B4-green                                |
| 9     | Plan-command-lint gate                                      | B     | 05   | B5-red-tests, B5-impl, B5-green                                              |
| 11    | Shared validation phase for specialists                     | C     | 02   | C2-red-tests, C2-impl, C2-green                                              |
| 10    | RAG-based fix-packet design                                 | C     | 03   | C3-design-rag-fix-packet, C3-example-in-plan, C3-green                       |
| 13    | Prompt-length guard for dispatch MCP                        | C     | 04   | C4-red-tests, C4-impl, C4-green                                              |
| 12    | Concurrency=10 / nested dispatch docs                       | D     | 02   | D2-concurrency-docs, D2-rag-exception-docs, D2-green                         |
