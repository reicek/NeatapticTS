# Spec Kit — Templates & Artifact Structures

Spec Kit’s templates are the delivery mechanism for its SDD discipline. Each template is a named Markdown file with placeholder sections, mandatory fields, and stable ID schemes (FR-###, SC-###, US#/AC#, T###, CHK###). This section reviews the five artifact templates and calls out the traceability patterns that make them effective.

## 1. `spec-template.md`

### Structure

```text
# Feature Specification: [FEATURE NAME]
**Feature Branch**: `[###-feature-name]`
**Created**: [DATE]
**Status**: Draft
**Input**: User description: "$ARGUMENTS"

## User Scenarios & Testing *(mandatory)*
### User Story 1 - [Brief Title] (Priority: P1)
### Edge Cases
## Requirements *(mandatory)*
### Functional Requirements
### Key Entities
## Success Criteria *(mandatory)*
### Measurable Outcomes
## Assumptions
```

### Mandatory Sections

The template marks the sections that must be filled:

- `## User Scenarios & Testing *(mandatory)*`
- `## Requirements *(mandatory)*`
- `## Success Criteria *(mandatory)*`

### User Story Discipline

The template comments are explicit about story independence:

> "User stories should be PRIORITIZED as user journeys ordered by importance.
> Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
> you should still have a viable MVP (Minimum Viable Product) that delivers value."

Each story has:

- A brief title and priority (`P1`, `P2`, `P3`...)
- A plain-language description of the user journey
- **Why this priority** — value justification
- **Independent Test** — how to verify this story on its own
- **Acceptance Scenarios** using **Given / When / Then**

### Functional Requirement IDs (FR-###)

Functional requirements are numbered and prefixed:

> `- **FR-001**: System MUST [specific capability, e.g., "allow users to create accounts"]`

The template also shows how to mark ambiguity without stalling:

> `- **FR-006**: System MUST authenticate users via [NEEDS CLARIFICATION: auth method not specified - email/password, SSO, OAuth?]`

### Success Criteria IDs (SC-###)

Success criteria are technology-agnostic and measurable:

> `- **SC-001**: [Measurable metric, e.g., "Users can complete account creation in under 2 minutes"]`
> `- **SC-002**: [Measurable metric, e.g., "System handles 1000 concurrent users without degradation"]`

### Key Entities (Conceptual)

The template asks for conceptual entities without implementation detail:

> `- **[Entity 1]**: [What it represents, key attributes without implementation]`

---

## 2. `plan-template.md`

### Structure

```text
# Implementation Plan: [FEATURE]
**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
## Summary
## Technical Context
## Constitution Check
## Project Structure
## Complexity Tracking
## Phase 0: Research & Foundation
## Phase 1: Design & Contracts
## Task Summary
```

### Constitution Check

The most important gate in the plan:

> ```text
> ## Constitution Check
> *GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*
> [Gates determined based on constitution file]
> ```

This is a fixed location where project values are enforced.

### Technical Context

A checklist of implementation-relevant facts, each of which may be `NEEDS CLARIFICATION`:

- Language/Version
- Primary Dependencies
- Storage
- Testing
- Target Platform
- Project Type
- Performance Goals
- Constraints
- Scale/Scope

### Project Structure

The template provides three structural options (single project, web app, mobile + API) but explicitly says:

> "The delivered plan must not include Option labels."

The plan command must collapse the options into the chosen concrete layout.

### Complexity Tracking

If the constitution limits complexity (e.g., a maximum number of projects or layers), the plan must justify violations:

> | Violation | Why Needed | Simpler Alternative Rejected Because |
> |-----------|------------|-------------------------------------|

This turns abstract principles into a visible decision log.

### Phase 0 / Phase 1

- **Phase 0 — Research & Foundation**: produces `research.md` and resolves all `NEEDS CLARIFICATION`.
- **Phase 1 — Design & Contracts**: produces `data-model.md`, `contracts/`, and `quickstart.md`.

---

## 3. `tasks-template.md`

### Structure

```text
# Tasks: [FEATURE NAME]
## Format: `[ID] [P?] [Story] Description`
## Phase 1: Setup (Shared Infrastructure)
## Phase 2: Foundational (Blocking Prerequisites)
## Phase 3: User Story 1 - [Title] (Priority: P1) 🎯 MVP
## Phase 4: User Story 2 - [Title] (Priority: P2)
...
## Phase N: Polish & Cross-Cutting Concerns
## Dependencies & Execution Order
## Implementation Strategy
```

### Format Rules

The format string is the entire contract for a task line:

> `## Format: [ID] [P?] [Story] Description`

Where:

- `[P]` = parallelizable
- `[Story]` = `US1`, `US2`, etc.
- Description must include exact file paths

### Story-Centric Phases

Every user story gets its own phase, with its own test and implementation subsections. This makes each story independently deliverable.

### Critical Foundation Gate

> "**⚠️ CRITICAL**: No user story work can begin until this phase is complete"

This is repeated in the dependency table:

> "**Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories"

### Optional Tests, But TDD If Included

When tests are requested, the template demands red-first behavior:

> "> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**"

### Within-Story Order

> "- Tests (if included) MUST be written and FAIL before implementation
> - Models before services
> - Services before endpoints
> - Core implementation before integration
> - Story complete before moving to next priority"

---

## 4. `constitution-template.md`

### Structure

```text
# [PROJECT_NAME] Constitution
## Core Principles
### [PRINCIPLE_1_NAME]
## [SECTION_2_NAME]
## [SECTION_3_NAME]
## Governance
```

### Purpose

The constitution is a durable governance document. The template examples show the kinds of rules a team might encode:

> "I. Library-First — Every feature starts as a standalone library; Libraries must be self-contained, independently testable, documented"
> "III. Test-First (NON-NEGOTIABLE) — TDD mandatory: Tests written → User approved → Tests fail → Then implement"
> "V. Observability — Text I/O ensures debuggability; Structured logging required"
> "VI. Versioning & Breaking Changes — MAJOR.MINOR.BUILD format"

### Governance Section

> "Constitution supersedes all other practices; Amendments require documentation, approval, migration plan"

This establishes the constitution as a higher-order norm than any individual command.

---

## 5. `checklist-template.md`

### Structure

```text
# [CHECKLIST TYPE] Checklist: [FEATURE NAME]
## [Category 1]
- [ ] CHK001 First checklist item with clear action
## [Category 2]
- [ ] CHK004 Another category item
## Notes
```

### Checklist ID Scheme

All items are prefixed `CHK###` and numbered sequentially. The template notes:

> "- Items are numbered sequentially for easy reference"

This is the artifact used by `/speckit.checklist` to create "unit tests for English".

---

## Traceability Summary

| ID Type | Meaning | Appears In |
|---------|---------|------------|
| `FR-###` | Functional requirement | `spec.md` |
| `SC-###` | Success criterion | `spec.md` |
| `US#` / `AC#` | User story / acceptance criterion | `spec.md`, `tasks.md` |
| `T###` | Implementation task | `tasks.md` |
| `CHK###` | Quality checklist item | `checklists/*.md` |

The templates do not enforce bi-directional linking in code, but the formats make it easy to trace from a failing task (`T014`) back to the story (`US1`) and the requirement (`FR-003`) it serves. This is the closest Spec Kit gets to NeatapticTS's gate evidence and tracker handoff structure.

## Comparison with NeatapticTS Artifacts

| Template | Spec Kit | NeatapticTS |
|----------|----------|-------------|
| Feature intent | `spec.md` (FR-###, SC-###, US#/AC#) | Often in `plans/*.plans.md` goal + acceptance criteria sections |
| Technical plan | `plan.md` + `research.md` + `data-model.md` + `contracts/` + `quickstart.md` | Embedded in `.plans.md` phase/step YAML + slices |
| Executable tasks | `tasks.md` checkbox format | Step-packet YAML with `files_to_change`, `validation`, `tdd_sequence` |
| Governance | `constitution.md` | `.github/copilot-instructions.md`, `.agent.md` frontmatter |
| Quality checklist | `checklists/*.md` | Gate output JSON, `VALIDATION_EVIDENCE` blocks |

The biggest template lesson for NeatapticTS is the **stable ID vocabulary**. Adopting `FR-###`, `SC-###`, `US#`, `AC#`, `T###`, and `CHK###` would make handoffs and gate evidence much more referential.
