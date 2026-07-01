# Spec Kit — SDD Workflow Analysis

## The Power Inversion

Spec Kit's SDD starts from a deliberate inversion of the traditional code-first model. The repo describes it bluntly:

> "For decades, code has been king. Specifications served code—they were the scaffolding we built and then discarded once the 'real work' of coding began."

SDD changes this so that **code serves specifications**. From `spec-driven.md`:

> "Specifications don't serve code—code serves specifications. The Product Requirements Document (PRD) isn't a guide for implementation; it's the source that generates implementation. Technical plans aren't documents that inform coding; they're precise definitions that produce code."

> "SDD eliminates the gap by making specifications and their concrete implementation plans born from the specification executable. When specifications and implementation plans generate code, there is no gap—only transformation."

This means the primary maintenance activity becomes **evolving specifications**, not hand-editing code. Refactoring is "restructuring for clarity"; debugging is "fixing specifications and their implementation plans that generate incorrect code."

## The Command-Phase Flow

The methodology is realized through a repeatable command sequence. The README lists the typical loop as:

1. `/speckit.constitution` — establish principles.
2. `/speckit.specify` — define what to build.
3. `/speckit.plan` — add the chosen tech stack and create a plan.
4. `/speckit.tasks` — derive tasks from the plan.
5. `/speckit.implement` — build the feature.

Optional but recommended validation commands slot in between:
- `/speckit.clarify` before `/speckit.plan`.
- `/speckit.checklist` and `/speckit.analyze` after `/speckit.tasks` and before `/speckit.implement`.
- `/speckit.converge` after `/speckit.implement` to generate a follow-up task set.

The reference puts the analyze gate in the standard flow more explicitly:

> "From the PRD, AI generates implementation plans that map requirements to technical decisions. Every technology choice has documented rationale. Every architectural decision traces back to specific requirements. Throughout this process, consistency validation continuously improves quality. AI analyzes specifications for ambiguity, contradictions, and gaps—not as a one-time gate, but as an ongoing refinement."

## Artifact Lifecycle

Each feature lives in `specs/[branch-name]/` and produces a set of linked Markdown artifacts:

| Artifact | Purpose | Created By |
|----------|---------|------------|
| `spec.md` | Product contract — user stories, FRs, NFRs, constraints, acceptance scenarios | `/speckit.specify` |
| `plan.md` | Technical implementation plan, architecture, rationale, constitution check | `/speckit.plan` |
| `research.md` | Technical context, library comparisons, performance/security findings | `/speckit.plan` |
| `data-model.md` | Domain entities and relationships | `/speckit.plan` |
| `contracts/` | API contracts, events, interfaces | `/speckit.plan` |
| `quickstart.md` | Key validation scenarios | `/speckit.plan` |
| `tasks.md` | Executable task list derived from the plan | `/speckit.tasks` |

The `spec-driven.md` example for a chat feature shows how these artifacts are generated in about 15 minutes:

> "Step 1: Create the feature specification (5 minutes) — `/speckit.specify Real-time chat system...`\nStep 2: Generate implementation plan (5 minutes) — `/speckit.plan WebSocket for real-time messaging...`\nStep 3: Generate executable tasks (5 minutes) — `/speckit.tasks`"

## Spec Persistence Models

Spec Kit does **not** force one artifact-maintenance strategy. `docs/concepts/spec-persistence.md` defines three models and lets the team choose:

### Flow-Back Spec

> "Use flow-back when `spec.md`, `plan.md`, `tasks.md`, and the implementation are all allowed to inform each other."

Edits can begin in any artifact; the team reconciles the set manually. Best for small teams where speed beats formal traceability, but risky because "silent divergence" can occur if lower-level changes are never reflected back into `spec.md`.

### Flow-Forward Spec

> "Use flow-forward when each feature directory should remain a historical record."

Completed artifacts are immutable. New requirements get a new feature directory. Good for auditability, but can fragment context across directories.

### Living Spec

> "Use living spec when `spec.md` is the contract and the other artifacts are derived from it."

The team updates `spec.md` first, then regenerates `plan.md` and `tasks.md`. This treats the plan and task list as disposable derivations and is the purest expression of "spec is source of truth."

The guide recommends documenting the chosen convention in the project constitution or team onboarding notes.

## Role of the Constitution

The constitution is the first artifact and a binding gate for everything that follows.

### Location and Form

- File: `.specify/memory/constitution.md`.
- Created/updated by: `/speckit.constitution`.
- Template: `.specify/templates/constitution-template.md`.

### Normative Force

The constitution is intentionally written in **MUST / SHOULD** language. Spec Kit's own ratified constitution says:

> "These principles are derived from the patterns the codebase already enforces. They are binding on all changes — including the `specify bundle` subcommand and any future command group, integration, extension, preset, or workflow."

Principle II is titled **"Test-Backed Change (NON-NEGOTIABLE)"** and states:

> "Every behavioral change MUST be accompanied by automated tests, and the suite is a hard gate."

### How the Constitution Enters Every Plan

The `plan-template.md` contains a `## Constitution Check` gate that must pass before research and be re-checked after design:

```text
## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*
[Gates determined based on constitution file]
```

The constitution command itself instructs:

> "Read `.specify/templates/plan-template.md` and ensure any 'Constitution Check' or rules align with updated principles."

### Governance Section

The constitution template requires a **Governance** section:

> "Amendments require a PR with rationale, maintainer approval, and a version bump... Any amendment MUST propagate to dependent templates and command guidance in the same change, recorded in the Sync Impact Report at the top of this file."

The constitution command adds a **Sync Impact Report** as an HTML comment at the top of the file, documenting version changes, modified principles, and affected templates.

### Constitutional Severity

From the ratified constitution:

> "Authority. Principles I–V are binding gates. The `## Constitution Check` section of the plan template MUST be evaluated against these principles, and `/speckit.analyze` treats conflicts with a MUST as CRITICAL. Violations are resolved by changing the spec, plan, or tasks — not by diluting a principle."

This is a key operational detail: constitution violations are not warnings; they are CRITICAL analysis findings that block convergence until the plan/spec changes.

## Extension Hooks in the SDD Loop

The constitution command is the first place the **extension hook system** appears. The command checks `.specify/extensions.yml` for `hooks.before_constitution` and `hooks.after_constitution` entries. If hooks are present, the agent must actually invoke them and wait for completion. This lets extensions inject behavior (for example, the git extension creating branches) at defined points in the workflow.

## Implications for NeatapticTS

Spec Kit's SDD workflow is simpler and more linear than NeatapticTS's tiered orchestrator model:
- It is **slash-command-driven** inside a generic coding agent rather than dispatched to numbered phase agents.
- It makes **spec.md the primary source of truth**, with explicit persistence models rather than a single mandated model.
- It elevates the **constitution** to a binding, versioned governance artifact evaluated as a gate in every plan and as a CRITICAL severity in analysis.
- It treats plans and tasks as **derived artifacts**, not co-equal sources of truth (unless the team chooses flow-back).

NeatapticTS already has plan/phase discipline (`phase-handoff-workflow`), but it does not have a first-class, normative constitution file that downstream agents MUST evaluate. Adopting this pattern could close a governance gap without losing the orchestrator structure.
