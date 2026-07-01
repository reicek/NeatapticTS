# Spec Kit — Planning & Triage Commands

The left-hand side of the SDD loop is where vague intent becomes a concrete, traceable implementation plan. Spec Kit exposes this through `/speckit.specify`, `/speckit.clarify`, `/speckit.plan`, and `/speckit.tasks`. This section compares those commands with NeatapticTS's numbered planning/research/red-testing orchestrators.

## `/speckit.specify` — From Feature Description to Spec

### Purpose

`/speckit.specify` turns a free-form feature description into a structured `spec.md` under `specs/<prefix>-<short-name>/`.

### What It Does

The command:
1. Generates a concise 2–4 word short name (e.g. `add-user-auth`).
2. Optionally triggers a `before_specify` extension hook for branch creation.
3. Creates the feature directory using sequential numbering (`003-user-auth`) or timestamp prefixes.
4. Copies the resolved `spec-template` into `spec.md`.
5. Writes `FEATURE_DIR` to `.specify/feature.json` so downstream commands can locate it without relying on git branch names.
6. Fills the template from the user's description, making **informed guesses** and marking only high-impact unknowns.

### The [NEEDS CLARIFICATION] Rule

The most important discipline in `specify` is the hard limit on ambiguity markers:

> "Only mark with `[NEEDS CLARIFICATION: specific question]` if:
> - The choice significantly impacts feature scope or user experience
> - Multiple reasonable interpretations exist with different implications
> - No reasonable default exists
> **LIMIT: Maximum 3 [NEEDS CLARIFICATION] markers total**"

The prioritization order is explicit:

> "Prioritize clarifications by impact: scope > security/privacy > user experience > technical details"

This prevents the specification phase from stalling on every open question.

### Built-In Spec Quality Checklist

After writing `spec.md`, `/speckit.specify` creates a self-checklist at `checklists/requirements.md` and re-validates the spec until all items pass (max three iterations). The checklist includes:

- No implementation details (languages, frameworks, APIs)
- Focused on user value and business needs
- No [NEEDS CLARIFICATION] markers remain
- Requirements are testable and unambiguous
- Success criteria are measurable and technology-agnostic
- All acceptance scenarios are defined
- Edge cases are identified

This is the first place Spec Kit's "unit tests for English" pattern appears in the core loop.

---

## `/speckit.clarify` — Up to Five Targeted Questions

### Purpose

`/speckit.clarify` runs **before** `/speckit.plan` to remove ambiguity from the spec. It is explicitly not a planning command:

> "This clarification workflow is expected to run (and be completed) BEFORE invoking `__SPECKIT_COMMAND_PLAN__`."

### Question Discipline

The command is constrained to a maximum of **five total questions** across the whole session. Each question must be:

> "answerable with EITHER:
> - A short multiple‑choice selection (2–5 distinct, mutually exclusive options), OR
> - A one-word / short‑phrase answer (explicitly constrain: 'Answer in <=5 words')."

Only questions whose answers materially impact architecture, data modeling, task decomposition, test design, UX behavior, operational readiness, or compliance validation are asked.

### Sequential, Recommended Questioning

The agent presents **exactly one question at a time**, with a recommended option at the top:

```text
**Recommended:** Option [X] - <reasoning>
| Option | Description |
|--------|-------------|
| A | ... |
```

After each accepted answer, the agent **immediately integrates** the clarification into the spec, adding a `## Clarifications` → `### Session YYYY-MM-DD` section if missing, and updating the relevant functional requirements, user stories, data model, success criteria, edge cases, or terminology.

### Coverage Taxonomy

Before generating questions, `/speckit.clarify` scans the spec across these dimensions:

- Functional Scope & Behavior
- Domain & Data Model
- Interaction & UX Flow
- Non-Functional Quality Attributes (performance, scalability, reliability, observability, security, privacy, compliance)
- Integration & External Dependencies
- Edge Cases & Failure Handling
- Constraints & Tradeoffs
- Terminology & Consistency
- Acceptance Criteria Testability
- TODO / Placeholder markers

This taxonomy is a useful checklist on its own and could be borrowed for any planning skill.

---

## `/speckit.plan` — Technical Plan with Constitution Gate

### Purpose

`/speckit.plan` consumes the spec and the user's chosen tech stack to produce `plan.md`, `research.md`, `data-model.md`, `contracts/`, and `quickstart.md`.

### Workflow

1. **Setup**: run `setup-plan.sh` to locate `FEATURE_SPEC`, `IMPL_PLAN`, etc.
2. **Load context**: read `spec.md` and `/memory/constitution.md`.
3. **Execute plan workflow**:
   - Fill Technical Context (mark unknowns as `NEEDS CLARIFICATION`).
   - Fill `## Constitution Check` from the constitution.
   - **ERROR on gate failures or unresolved clarifications**.
   - **Phase 0**: generate `research.md` and resolve all `NEEDS CLARIFICATION`.
   - **Phase 1**: generate `data-model.md`, `contracts/`, `quickstart.md`.
   - Re-evaluate `## Constitution Check` post-design.

### Constitution Check

The plan template contains a gate that must pass before research and be re-checked after design:

```text
## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*
[Gates determined based on constitution file]
```

The plan command's key rule:

> "Use absolute paths for filesystem operations; use project-relative paths for references in documentation.> ERROR on gate failures or unresolved clarifications"

### Output Artifacts

| Artifact | Content |
|----------|---------|
| `plan.md` | Architecture, stack, phases, constitution check, complexity tracking |
| `research.md` | Decision, rationale, alternatives considered |
| `data-model.md` | Entities, fields, relationships, validation rules, state transitions |
| `contracts/` | Public API / command / endpoint / UI contracts |
| `quickstart.md` | Runnable validation scenarios, setup/run commands, expected outcomes |

The `quickstart.md` is intentionally not a full implementation guide:

> "Do not include full implementation code, model/service/controller bodies, migrations, or complete test suites. Keep this artifact as a validation/run guide."

---

## `/speckit.tasks` — User-Story-Driven Task Breakdown

### Purpose

`/speckit.tasks` reads `plan.md` (required), plus optional `data-model.md`, `contracts/`, `research.md`, and `quickstart.md`, then writes an executable `tasks.md`.

### Phase Structure

The generated `tasks.md` is organized as:

- **Phase 1**: Setup (project initialization)
- **Phase 2**: Foundational (blocking prerequisites for all user stories)
- **Phase 3+**: One phase per user story in priority order (P1, P2, P3...)
- **Final Phase**: Polish & Cross-Cutting Concerns

Within each user story phase the typical order is: Tests (if requested) → Models → Services → Endpoints → Integration.

### Strict Checklist Format

Every task MUST follow this exact format:

```text
- [ ] [TaskID] [P?] [Story?] Description with file path
```

Examples:

- ✅ `- [ ] T001 Create project structure per implementation plan`
- ✅ `- [ ] T005 [P] Implement authentication middleware in src/middleware/auth.py`
- ✅ `- [ ] T012 [P] [US1] Create User model in src/models/user.py`
- ✅ `- [ ] T014 [US1] Implement UserService in src/services/user_service.py`

The `[P]` marker means parallelizable: different files, no dependencies on incomplete tasks. The `[US1]` label maps the task back to a user story from `spec.md`.

### Tests Are Optional

Unlike NeatapticTS, Spec Kit does not mandate a red phase:

> "**Tests are OPTIONAL**: Only generate test tasks if explicitly requested in the feature specification or if user requests TDD approach."

This is a major philosophical difference from NeatapticTS, where `03-red-testing` is a numbered orchestrator and red contracts are expected for behavior changes.

---

## Comparison with NeatapticTS Planning Architecture

| Aspect | Spec Kit | NeatapticTS |
|--------|----------|-------------|
| **Entry point** | `/speckit.specify` inside the coding agent | `01-planning` orchestrator agent |
| **Ambiguity handling** | ≤3 `[NEEDS CLARIFICATION]` markers in spec; ≤5 questions in `/speckit.clarify` | `planning-acceptance-criteria` surfaces open assumptions; no hard count |
| **Tech stack input** | Given in `/speckit.plan` prompt | Research first (`02-researching`), then plan |
| **Research artifact** | `research.md` generated during `/speckit.plan` | `02-researching` agent + scouts produce evidence; not a single mandated `research.md` |
| **Constitution gate** | `## Constitution Check` in every plan; MUST pass before Phase 0; re-check after Phase 1 | No project constitution file; plan alignment via `plan-alignment` and roadmap |
| **Data model / contracts** | `data-model.md` and `contracts/` produced by plan command | Produced during implementation or research depending on the active plan |
| **Task format** | Checkbox + T### IDs + `[P]` + `[US#]` labels | Step-packet YAML blocks with `slices`, `files_to_change`, `acceptance_criteria`, `validation` |
| **TDD / red phase** | Optional; test tasks only if requested | Mandatory for behavior changes (`03-red-testing` + `red-test-contracts`) |
| **Traceability IDs** | FR-###, SC-###, US#/AC#, T### | Less formal; uses step/slice IDs and gate IDs |
| **Plan verification** | Spec-quality checklist (≤3 iterations) | Fresh `01-planning` verification pass required before red/implementation; `plan-readiness` gate |
| **Phase compression** | Not explicit | `07-logging` compresses completed phases to `.logs.md` before advancing |
| **Roadmap alignment** | Team convention in constitution/onboarding | `plan-alignment` + `plan-sync-validation` enforce `plans/README.md` and `plans/Roadmap.md` alignment |

### What NeatapticTS Does Better

- **Mandatory red phase**: `03-red-testing` ensures failing tests exist before implementation, which is stronger than Spec Kit's optional test tasks.
- **Plan verification pass**: a fresh `01-planning` agent verifies the plan before execution, with a machine-readable gate.
- **Roadmap/index discipline**: `plan-sync-validation` keeps `plans/README.md`, `plans/Roadmap.md`, and the plan file aligned with `[PLANED]/[WIP]/[DONE]` markers.
- **Phase compression**: moving completed detail to `.logs.md` keeps active plans lean.
- **Structured step packets**: YAML blocks with `validation`, `acceptance_criteria`, `slices`, and `tdd_sequence` are more machine-actionable than Markdown task checkboxes.

### What Spec Kit Does Better

- **Hard ambiguity limits**: ≤3 `[NEEDS CLARIFICATION]` and ≤5 questions prevent planning paralysis.
- **Constitution gate**: a normative, versioned constitution evaluated in every plan is a clearer governance model than NeatapticTS's implicit conventions.
- **User-story-driven tasks**: mapping every task to `[US#]` makes traceability obvious.
- `[P]` parallel markers: simple, visible signal for what can be done in parallel.
- **Artifact templates**: spec/plan/tasks templates are stable, named files rather than ad-hoc plan formats.

## Recommended Assimilation

1. **Adopt the `[NEEDS CLARIFICATION]` cap** in `planning-acceptance-criteria` or `01-planning`: at most 3 open clarifications before a spec is considered ready.
2. **Add a constitution file** to NeatapticTS and evaluate it in the `plan-readiness` gate, making MUST conflicts block execution.
3. **Borrow the `[P]` and `[US#]` task format** for NeatapticTS tracker step packets while keeping YAML metadata.
4. **Keep mandatory red testing**; it is a stricter, more defensible practice than Spec Kit's optional tests.
5. **Preserve NeatapticTS plan/index/roadmap sync**; Spec Kit has no equivalent automated plan registration.
