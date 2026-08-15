---
name: spec-checklist
description: 'Lightweight "unit tests for English" gate: validates spec/plan prose quality, traceability >= 80%, and ID coverage before implementation dispatch.'
argument-hint: 'Name the plan or spec path and the target implementation slice.'
user-invocable: false
disable-model-invocation: false
skills:
  - research-methodology
  - plan-sync-validation
  - tracker-handoff
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Spec-Checklist Playbook

Use this skill when a plan, spec, or step packet needs a lightweight read-only
prose-quality and traceability gate before it is handed to implementation.
Think of it as **unit tests for English**: it checks whether the requirements
are written clearly and traceably, not whether the code already works.

This skill owns the durable contract for:

- prose-quality checks on plans and specs,
- traceability coverage measurement (≥ 80% required before implementation),
- ID coverage checks (`AC-###`, `T###`, `CHK-###`, `FR-###`, `SC-###`),
- the four canonical gap types: `missing`, `partial`, `contradicts`, `unrequested`.

When a tracker or handoff shape needs updating, defer to `tracker-handoff`. When
plan/README/Roadmap alignment is needed, defer to `plan-sync-validation`.

## When to Use

- A `.plans.md` or `spec.md` is about to move from planning to implementation.
- A step packet needs a pre-implementation quality gate.
- A green-validation pass must confirm that acceptance criteria were written, not
  only that the code passes tests.
- An audit asks "was the spec ready before code was changed?"

## When NOT to use

Do NOT use for implementation code review - use `implementation-standards`
skill instead. Do NOT use for runtime test execution - use
`green-validation-gates` or `05-green-testing` instead. Do NOT use as a
substitute for `plan-sync-validation`; it complements, but does not replace,
plan/README/Roadmap alignment checks.

## Workflow Diagram

```text
Flowchart summary: "Plan/spec ready" -> "Run spec-checklist"; "Run spec-checklist" -> "Coverage >= 80%?"; "Coverage >= 80%?" -> "Pass: advance to implementation" (Yes), "Fail: record gaps and block dispatch" (No); "Fail: record gaps and block dispatch" -> "Fix prose/traceability"; "Fix prose/traceability" -> "Run spec-checklist".
```

## Task Packet

Pass a compact packet that names the plan or spec path, the implementation slice
or step, and the IDs the checklist must verify.

```text
Use spec-checklist for plans/Worker_Friendly_Network_Serialization_Fastpath.md Step 03.
Target: step packet before 04-implementing dispatch.
Expected IDs: AC-018..AC-022, T001..T004.
Mode: pre-implementation blocking gate.
```

## Required Workflow

1. Start from the **constitution authority** `plans/constitution.md`, especially
   Principle 2: _Human owns the mission; AI owns the method inside the guardrails_.
   The spec-checklist gate is one of those guardrails.
2. Read the plan/spec and any linked step packet. Load only the sections relevant
   to the current handoff; do not dump the whole file.
3. Identify every declared identifier:
   - acceptance criteria (`AC-###`),
   - tasks (`T###`),
   - checklist items (`CHK-###`),
   - feature/scenario requirements (`FR-###`, `SC-###`) when present.
4. Run the checklist below. Record each finding under one of the four gap types.
5. Compute traceability coverage:
   - numerator: identifiers that map to at least one other traceable artifact,
   - denominator: all identifiers that should be traceable,
   - threshold: **≥ 80%**.
6. Produce the pass/fail report. If coverage is below ≥ 80%, block dispatch to
   `04-implementing` until the gaps are fixed.
7. Append the report to the plan's `VALIDATION_EVIDENCE` section and, if the
   finding is systemic, record it as a `capturing-learning-event`.

### Gap Type Definitions

These four gap types are the canonical vocabulary for spec-checklist findings:

- `missing`: an expected artifact or reference has no corresponding entry. For
  example, a file change with no `AC-###`, or a task with no mapped acceptance
  criterion.
- `partial`: an artifact exists but is incomplete. For example, an `AC-###` that
  lacks a validation command, or a traceability row that omits the files changed.
- `contradicts`: two artifacts disagree. For example, an acceptance criterion says
  "reject negative inputs" while the task says "accept signed values".
- `unrequested`: an artifact or validation command exists but has no supporting
  requirement or criterion. For example, a test command that covers behavior no
  `AC-###` asked for.

### Checklist

Run these checks read-only; do not edit code.

- Every acceptance criterion in scope has a stable `id: AC-###`.
- Every task in scope has a stable `id: T###` and maps to at least one `AC-###`.
- Every checklist item, if present, has a stable `id: CHK-###` and references a
  spec section or gap marker.
- No `NEEDS CLARIFICATION` markers remain above the active cap (≤ 3 per
  `01-planning` clarification discipline).
- The plan cites the constitution authority (`plans/constitution.md`) when a
  principle is exercised by the step.
- The step packet's `validation` list names focused commands that cover the files
  in `files_to_change`.
- The traceability table, if present, maps each `AC-###` to concrete files and a
  validation command.
- No gap of type `contradicts` exists between the acceptance criteria and the
  step instructions.

## Output Contract

Return a compact pass/fail report with these fields:

```json
{
  "pass": boolean,
  "coverage_percent": number,
  "gaps": {
    "missing": number,
    "partial": number,
    "contradicts": number,
    "unrequested": number
  },
  "findings": [
    { "id": "AC-018", "gap": "missing", "reason": "..." }
  ],
  "owner": "spec-checklist"
}
```

The gate passes only when `pass` is `true`, all gap-type counts are zero, and
`coverage_percent` is **≥ 80%**.

## Decision Tree

```text
Flowchart summary: "Spec/plan review" -> "Need prose + traceability check?"; "Need prose + traceability check?" -> "spec-checklist" (Yes), "skip to next gate" (No); "spec-checklist" -> "coverage >= 80%?"; "coverage >= 80%?" -> "allow implementation" (Yes), "block and report gaps" (No).
```

## Guardrails

- Do not treat spec-checklist as a code-quality gate; it tests English, not
  implementation.
- Do not permit dispatch to `04-implementing` when coverage is below **≥ 80%**.
- Do not silently downgrade a `contradicts` finding; escalate it to
  `00.cross-tier-helper` when it crosses a constitution principle.
- Do not ignore `unrequested` artifacts; either add a matching requirement or
  remove the orphan artifact.
- Do not cache old reports; rerun the checklist after every plan edit.
- Do not invent IDs for artifacts that the owner did not declare; report them as
  `missing` instead.

## Expected Final Output

A strong spec-checklist pass should report:

- the plan/spec path checked,
- the implementation step or slice under review,
- the traceability coverage percentage,
- the count of each gap type,
- the exact pass/fail verdict and any blocking gaps,
- the constitution principle exercised (typically Principle 2).
