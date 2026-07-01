# Spec Kit — Verbatim Phrases, Verbs, and Section Headings Worth Emulating

The Spec Kit command vocabulary is concise, consistent, and likely to become industry-familiar because it is published by GitHub. This document collects exact phrasing, command definitions, verbs, headings, and snippets that NeatapticTS should consider adopting or mirroring.

## Exact Command Names / Verbs

```text
/speckit.specify
/speckit.clarify
/speckit.plan
/speckit.tasks
/speckit.analyze
/speckit.checklist
/speckit.implement
/speckit.converge
/speckit.constitution
/speckit.bug.assess
/speckit.bug.fix
/speckit.bug.test
/speckit.git.feature
/speckit.git.initialize
/speckit.git.commit
/speckit.git.validate
/speckit.git.remote
/speckit.agent-context.update
/speckit.selftest.extension
```

The naming convention is `speckit.<domain>.<verb>` for extensions and `speckit.<verb>` for core commands. This is clean and predictable.

## Core Command Definitions (from `templates/commands/*.md`)

### `/speckit.specify`

> "Create a detailed feature specification from user description."

### `/speckit.clarify`

> "Resolve spec ambiguity through up to 5 targeted questions."

### `/speckit.plan`

> "Create an implementation plan from a feature specification."

### `/speckit.tasks`

> "Generate user-story-driven implementation tasks from the plan."

### `/speckit.analyze`

> "Perform read-only gap analysis against constitution and artifacts."

### `/speckit.checklist`

> "Generate requirements-quality checklists (unit tests for English)."

### `/speckit.implement`

> "Execute implementation tasks from tasks.md."

### `/speckit.converge`

> "Find gaps, contradictions, and omissions before closure."

### `/speckit.constitution`

> "Create or update the project constitution."

## High-Value Phrases

### Ambiguity Discipline

> "LIMIT: Maximum 3 [NEEDS CLARIFICATION] markers total"

> "Prioritize clarifications by impact: scope > security/privacy > user experience > technical details"

### Checklist Philosophy

> "Unit tests for English"

> "Requirements-quality validation"

### Convergence

> "append-only convergence tasks"

> Gap types:

- `missing`
- `partial`
- `contradicts`
- `unrequested`

### Task Format

> `- [ ] [TaskID] [P?] [Story] Description`

> "- **[P]**: Can run in parallel (different files, no dependencies)"

> "- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)"

### Independent Testability

> "Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them, you should still have a viable MVP (Minimum Viable Product) that delivers value."

> "- Developed independently
> - Tested independently
> - Deployed independently
> - Demonstrated to users independently"

### Constitution Gate

> "GATE: Must pass before Phase 0 research. Re-check after Phase 1 design."

### Mandatory Acceptance Format

> "**Given** [initial state], **When** [action], **Then** [expected outcome]"

### Severity Scale

> `critical` | `high` | `medium` | `low`

### Verdict Scale

> `valid` | `likely valid, needs reproduction` | `invalid`

### URL Trust Policy Branch Names

> "`allowlisted` / `confirmed-by-user` / `auto-refused: <reason>`"

### Preset Resolution Vocabulary

> "replace" | "prepend" | "append" | "wrap"

## Section Headings Worth Copying

From `spec-template.md`:

```text
## User Scenarios & Testing *(mandatory)*
### User Story 1 - [Brief Title] (Priority: P1)
#### Acceptance Scenarios
### Functional Requirements
### Key Entities
### Measurable Outcomes
## Assumptions
```

From `plan-template.md`:

```text
## Technical Context
## Constitution Check
## Complexity Tracking
## Phase 0: Research & Foundation
## Phase 1: Design & Contracts
```

From `tasks-template.md`:

```text
## Phase 1: Setup (Shared Infrastructure)
## Phase 2: Foundational (Blocking Prerequisites)
## Phase 3: User Story 1 - [Title] (Priority: P1) 🎯 MVP
## Phase N: Polish & Cross-Cutting Concerns
## Dependencies & Execution Order
## Implementation Strategy
```

From bug assessment:

```text
## Symptom
## Reproduction
## Suspected Code Paths
## Root Cause Hypothesis
## Proposed Remediation
## Risks & Considerations
## Open Questions
```

## ID Schemes

| Prefix | Use |
|--------|-----|
| `FR-###` | Functional requirement |
| `SC-###` | Success criterion |
| `US#` | User story |
| `AC#` | Acceptance criterion |
| `T###` | Task |
| `CHK###` | Checklist item |

These prefixes are short, sortable, and easy to reference in commits and comments.

## Agent Disclosure Verbatim

From `AGENTS.md`:

> "Disclosure is **continuous**, not a one-time event."

> "Every commit you author must carry an `Assisted-by:` trailer identifying the agent and whether it acted autonomously or under direct human supervision"

> "Assisted-by: GitHub Copilot (model: <name-if-known>, autonomous)"

## Preset / Extension CLI Verbs

```text
specify preset search|add|remove|list|info|resolve|enable|disable|set-priority
specify preset catalog list|add|remove
specify extension info|add
```

These CLI verbs are worth adopting for any future catalog system.

## NeatapticTS Adoption Priority

1. **Command names**: use `/speckit.specify` style verbs for user-facing slash commands (or our own namespace) — they are descriptive and short.
2. **ID prefixes**: adopt `FR-###`, `SC-###`, `US#/AC#`, `T###`, `CHK###`.
3. **Section headings**: borrow `User Scenarios & Testing`, `Constitution Check`, `Complexity Tracking`.
4. **Ambiguity marker limit**: hard cap on `[NEEDS CLARIFICATION]` markers.
5. **Assisted-by trailer**: add to commit policy.
6. **Severity/verdict vocabulary**: `critical/high/medium/low` and `valid/likely-valid/invalid` are simple and industry-adjacent.
