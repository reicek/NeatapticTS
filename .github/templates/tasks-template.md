---
description: 'NeatapticTS user-story-driven task template for 04-implementing'
---

# NeatapticTS Tasks Template

> **Constitution authority:** This template is the canonical task-list scaffold
> for `04-implementing` work in `NeatapticTS`. Use it in place of ad-hoc task
> generation so every slice stays user-story-driven, traceable to acceptance
> criteria, and aligned with `plans/constitution.md`.
>
> **How to use:** copy this file into the active plan or slice tracker, replace
> the bracketed placeholders, and tick tasks off as they move from `[ ]` to
> `[x]`.

## Setup

**Purpose:** Prepare the repository, tooling, and environment for the slice.

- [ ] T001 [P] Create or claim the active plan tracker in `plans/<workstream>.plans.md`.
- [ ] T002 Install or verify dependencies and run `npm ci` if `package-lock.json` changed.
- [ ] T003 [P] Run focused preflight checks (`npx tsc --noEmit`, `npm run lint`) on the expected touch points.

---

## Foundational

**Purpose:** Lay the blocking groundwork that must be green before any story
work proceeds.

- [ ] T004 Define the public API surface and typed config defaults for the slice.
- [ ] T005 Add or update the smallest focused red test that proves the gap.
- [ ] T006 Capture the acceptance-criterion mapping in the plan's `PlanUpdate` block.

---

## Story

**Purpose:** Implement the user-facing behavior in thin, independently
completable slices.

- [ ] T007 [P] Implement the primary orchestration function for the slice.
- [ ] T008 [P] Extract helper modules below the orchestration file's fold.
- [ ] T009 Wire the new behavior into existing call sites without breaking prior contracts.
- [ ] T010 Run the focused red test and verify it turns green.

---

## Polish

**Purpose:** Harden documentation, coverage, and cross-cutting quality gates.

- [ ] T011 Add JSDoc with examples to every new exported symbol.
- [ ] T012 Run `npm run docs` and verify generated READMEs are clean.
- [ ] T013 Record validation evidence in the plan and run the relevant gate checks.

---

## Traceability

| Task ID | Phase        | Description                              | Acceptance Criterion | Constitution Principle                  | Parallel `[P]` |
| ------- | ------------ | ---------------------------------------- | -------------------- | --------------------------------------- | -------------- |
| T001    | Setup        | Create or claim the active plan tracker  | AC-014               | `principle-5-unique-ids`                | Yes            |
| T004    | Foundational | Define public API surface and defaults   | AC-014               | `principle-3-verbatim-binding`          | No             |
| T007    | Story        | Implement primary orchestration function | AC-015               | `principle-4-breadth-first-recoverable` | Yes            |
| T011    | Polish       | Add JSDoc with examples                  | AC-017               | `principle-3-verbatim-binding`          | No             |

> Replace the rows above with the actual slice tasks. Every task should map to
> one `AC-###` acceptance criterion and one principle from
> `plans/constitution.md`.

---

## Constitution Check

Before marking this slice complete, verify the following principles from
`plans/constitution.md`:

- [ ] `principle-1-ai-thinking-partner` — Trade-offs are surfaced and the human
      owner approves material decisions.
- [ ] `principle-2-human-mission-ai-method` — The slice implements the human
      "what" without violating documented guardrails.
- [ ] `principle-3-verbatim-binding` — Any verbatim step-packet, YAML, or
      schema block is honored exactly as written.
- [ ] `principle-4-breadth-first-recoverable` — The slice is small, independent,
      and has a cheap rollback path.
- [ ] `principle-5-unique-ids` — Task IDs (`T###`) and acceptance criteria
      (`AC-###`) are stable and traceable.

---

## Clarifications

> Append-only. Record any ambiguity, assumption, or change of interpretation
> discovered while using this template. Do not edit or delete prior entries.
