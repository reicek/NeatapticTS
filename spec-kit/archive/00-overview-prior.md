# Spec Kit — High-Level Overview

## What It Is

GitHub **Spec Kit** is an open-source toolkit for **Spec-Driven Development (SDD)**. It treats specifications as the primary artifact of software development and makes them executable: code is generated from specs and plans rather than written first.

From the repo:

> "Spec-Driven Development **flips the script** on traditional software development. For decades, code has been king — specifications were just scaffolding we built and discarded once the 'real work' of coding began. Spec-Driven Development changes this: **specifications become executable**, directly generating working implementations rather than just guiding them."

The toolkit ships as:
- **Specify CLI** — bootstraps a project and installs agent integrations.
- **Core slash commands** — `/speckit.*` prompts installed into the coding agent.
- **Templates** — `spec-template.md`, `plan-template.md`, `tasks-template.md`, `constitution-template.md`, `checklist-template.md`.
- **Extensions / presets / bundles** — layered customization system.

Spec Kit supports 30+ AI coding agents (both CLI and IDE-based). The goal is **technology independence**: the same specification can drive different stacks.

## SDD Philosophy (In Spec Kit's Own Words)

- **Intent-driven development**: specifications define the *what* before the *how*.
- **Multi-step refinement** rather than one-shot code generation from prompts.
- **Specifications as the lingua franca**: "Code becomes its expression in a particular language and framework."
- **Executable specifications**: precise enough to generate working systems.
- **Continuous refinement**: consistency validation is ongoing, not a one-time gate.
- **Research-driven context** and **bidirectional feedback** from production metrics, incidents, and operational learnings.
- **Branching for exploration**: generate multiple implementation approaches from the same specification.

## The Exact Core Command Loop

Spec Kit exposes the workflow through a small, stable set of slash commands. The typical loop is:

```text
/speckit.constitution   → establish governing principles
/speckit.specify        → define what to build (requirements, user stories)
/speckit.clarify        → resolve underspecified areas (up to 5 questions)
/speckit.plan           → create technical implementation plan
/speckit.checklist      → quality checklist for the spec/plan
/speckit.tasks          → generate actionable task list
/speckit.analyze        → read-only cross-artifact consistency & coverage check
/speckit.implement      → execute all tasks and build the feature
/speckit.converge       → assess the codebase, append remaining work as new tasks
```

The README command table lists these with their agent-skill names:

| Command                  | Agent Skill             | Description |
| ------------------------ | ----------------------- | ----------- |
| `/speckit.constitution`  | `speckit-constitution`  | Create or update project governing principles and development guidelines |
| `/speckit.specify`       | `speckit-specify`       | Define what you want to build (requirements and user stories) |
| `/speckit.plan`          | `speckit-plan`          | Create technical implementation plans with your chosen tech stack |
| `/speckit.tasks`         | `speckit-tasks`         | Generate actionable task lists for implementation |
| `/speckit.implement`     | `speckit-implement`     | Execute all tasks to build the feature according to the plan |
| `/speckit.converge`      | `speckit-converge`      | Assess the codebase against spec/plan/tasks and append remaining work as new tasks |
| `/speckit.clarify`       | `speckit-clarify`       | Clarify underspecified areas (recommended before `/speckit.plan`; formerly `/quizme`) |
| `/speckit.analyze`       | `speckit-analyze`       | Cross-artifact consistency & coverage analysis (run after `/speckit.tasks`, before `/speckit.implement`) |
| `/speckit.checklist`     | `speckit-checklist`     | Generate custom quality checklists that validate requirements completeness, clarity, and consistency (like "unit tests for English") |

There is also `/speckit.taskstoissues`, which converts a generated task list into GitHub issues for tracking and execution.

## Three Development Phases

Spec Kit explicitly distinguishes three phases:

| Phase | Focus |
| ----- | ----- |
| **0-to-1 Development** ("Greenfield") | Generate from scratch |
| **Creative Exploration** | Parallel implementations, diverse stacks, UX experiments |
| **Iterative Enhancement** ("Brownfield") | Add features, modernize legacy systems, adapt processes |

## Core Output Artifacts

Per feature, Spec Kit produces a directory under `specs/[feature-branch]/` containing:

- `spec.md` — the PRD: user stories, functional/non-functional requirements, constraints, acceptance scenarios.
- `plan.md` — technical implementation plan: architecture, technology choices, rationale, constitutional compliance.
- `tasks.md` — executable task list derived from the plan.
- `research.md` — technical comparisons and context.
- `data-model.md` — domain entities and relationships.
- `contracts/` — API contracts, events, interfaces.
- `quickstart.md` — key validation scenarios.
- `.specify/memory/constitution.md` — project-wide governing principles.

## Positioning vs. NeatapticTS (At-a-Glance)

Where Spec Kit is a **slash-command-driven SDD toolkit** that runs inside a generic coding agent, NeatapticTS is a **tiered, phase-orchestrated agent system** (01-planning → 07-logging) with explicit workflow gates, ISO 42001 learning-event capture, and a strong TDD/coverage discipline. The comparison is developed in the later sections of this study.

## Notable Design Choices

- **Template-driven quality**: templates constrain LLMs so they don't "prematurely optimize" into implementation details during specification.
- **Traceability as a first-class concern**: requirements carry IDs (FR-###, SC-###), tasks carry T001-style IDs, acceptance criteria carry US#/AC# references.
- **Constitution before specification**: project principles are established first and are referenced by every downstream command.
- **Layered customization stack**: project-local overrides → presets → extensions → core, resolved at runtime for templates and install time for commands.
