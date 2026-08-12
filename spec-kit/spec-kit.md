# Spec Kit Assimilation Report for NeatapticTS

> **Scope:** Compare the public GitHub [`github/spec-kit`](https://github.com/github/spec-kit) repository with the current NeatapticTS agent/skill/MCP/gate/routing/ISO-42001 implementation, and produce actionable cherry-pick recommendations.
> **Inputs:** 31 verbatim spec-kit source files cached under `spec-kit/verbatim/`.
> **Per-area detail:** See `00-overview.md` through `07-verbatim-phrases-and-verbs.md`. Prior exploratory drafts are archived under `spec-kit/archive/`.

---

## 1. Executive Summary

**Spec Kit** is a slash-command-driven, spec-driven development (SDD) framework. It turns natural-language intent into a chain of explicit artifacts: `constitution.md` → `spec.md` → `plan.md` → `tasks.md` → implementation → `converge.md`. It adds four optional validation commands (`clarify`, `analyze`, `checklist`, `converge`) and a runtime extension/preset/bundle system.

**NeatapticTS** already implements a functionally similar RED → IMPLEMENT → GREEN SDLC loop using eight Tier-1 phase agents (`01-planning` through `07-logging`, plus `00-helping`), step-packet YAML in `plans/`, gate contracts, MCP servers, and an ISO-42001-style `learning-log.jsonl`. The core ideas overlap heavily: specification-first intent, progressive refinement, research before design, traceability, and living documentation.

**The biggest differences are packaging, not philosophy:**

- Spec Kit uses **per-feature artifact files** in `specs/NNN-feature/`; NeatapticTS uses **per-step plan trackers** in `plans/<step>.md`.
- Spec Kit exposes its workflow as **slash commands** installed into an agent's command directory; NeatapticTS exposes its workflow as **agent IDs** invoked by the routing table and MCP.
- Spec Kit has a **project-level `constitution.md`**, a **template stack**, a **workflow engine**, and a **bug triage extension**; NeatapticTS has a **step-packet YAML schema**, **gate contracts**, and an **ISO-42001 learning loop**.

**Bottom line:** NeatapticTS should not wholesale adopt Spec Kit's directory layout or command names. Instead, it should cherry-pick the practices that fill real gaps in its current workflow, while preserving its existing plan file, agent naming, and ISO-42001 governance.

---

## 2. Top 10 Cherry-Pick Recommendations

These are ordered by expected value / implementation cost. Each recommendation names the area file where the full rationale lives.

### 2.1 Add a `constitution.md` and a constitutional gate (Area 01)

Spec Kit's `constitution.md` is loaded by `specify`, `plan`, `tasks`, `checklist`, and `converge`. NeatapticTS governance is currently spread across `STYLEGUIDE.md`, `CONTRIBUTING.md`, and agent frontmatter.

**Action:** Create `.github/ai-learning/constitution.md` (or `plans/constitution.md`) with MUST/SHOULD project principles. Add a `constitution_check` field to the step-packet schema and a gate in `01-planning`/`05-green-testing`.

**ISO 42001 note:** Record creation as `eventType: "governance-update"` in `learning-log.jsonl`.

### 2.2 Materialize resolved unknowns as `research.md` (Area 10)

Spec Kit's `plan` command produces `research.md` when the spec has `NEEDS CLARIFICATION` markers. NeatapticTS resolves unknowns in `02-researching`, but the evidence is usually embedded in the plan tracker.

**Action:** Require `02-researching` to write a short `docs/research/<feature>.md` artifact and link it from the step packet. This makes "what we learned before designing" auditable.

### 2.3 Introduce traceability IDs and a coverage table (Area 04, 07)

Spec Kit templates require `FR-###`/`SC-###` IDs, `US#`/`AC#` references, and `CHK-###` checklist items. NeatapticTS acceptance criteria exist but are prose, not machine-traceable.

**Action:** Add an optional `id` field to acceptance-criteria blocks and require `planning-acceptance-criteria` to label each criterion with `AC-###`. Update `phase-handoff-workflow` step packets to include a `traceability` table.

### 2.4 Adopt the clarify question cap (Area 03)

`/speckit.clarify` asks **up to 5 questions**, uses short-answer / multiple-choice, and increments a `## Clarifications` section. NeatapticTS planning has no explicit turn-cap.

**Action:** Add to `01-planning.agent.md`: "If the spec is ambiguous, ask at most 5 focused questions; do not proceed until ambiguity is below the `NEEDS CLARIFICATION` threshold."

### 2.5 Implement a lightweight `checklist` command / skill (Area 02)

Spec Kit's `/speckit.checklist` calls itself "unit tests for English": it validates spec quality, traceability ≥80%, and ID coverage. NeatapticTS has gates but no dedicated prose-quality gate.

**Action:** Create `.github/skills/spec-checklist/SKILL.md` (or extend `plan-sync-validation`) to run a read-only spec-quality checklist before any `04-implementing` dispatch.

### 2.6 Add a bug-triage extension (Area 02, 05)

Spec Kit's bug extension provides `assess` → `fix` → `test` with artifacts under `.specify/bugs/<slug>/`. NeatapticTS has `triaging-test-failures` and `test-fix-workflow` but no bug-specific artifact path.

**Action:** Add a `.github/bugs/` artifact layout and an optional bug skill set (`bug-assess`, `bug-fix`, `bug-test`) that mirrors spec-kit's URL-trust and evidence rules.

### 2.7 Preserve and extend ISO 42001 learning events (Area 06)

Spec Kit uses `Assisted-by:` and `Source:` trailers for disclosure but lacks a structured learning-log schema. NeatapticTS already records `capturing-learning-event`.

**Action:** Keep `capturing-learning-event` as a first-class skill. Add a YAML/JSONL template that records every constitution update, gate run, and spec checklist as a learning event.

### 2.8 Rationalize plan/task templates (Area 04)

Spec Kit's `tasks-template.md` is user-story-driven, with `[P]` parallel markers and a phase structure. NeatapticTS tasks are generated ad-hoc by `04-implementing`.

**Action:** Create a NeatapticTS `tasks-template.md` in `.github/templates/` with phases: Setup / Foundational / Story / Polish, and require `[P]` tags for parallelizable tasks.

### 2.9 Surface a public extension catalog pattern (Area 05)

Spec Kit defines extensions as `extension.yml` + command files and supports preset/bundle layering. NeatapticTS skills are close to this but not described as a catalog.

**Action:** Write `.github/extensions/README.md` documenting how `extension.yml`, commands, templates, and presets compose. This helps future contributors add skills/agents consistently.

### 2.10 Emulate command naming and phrasing where appropriate (Area 07)

Phrases like "unit tests for English", "append-only convergence", "gap types: missing/partial/contradicts/unrequested", and "constitution authority" are likely to become industry-familiar.

**Action:** Adopt these exact phrases in NeatapticTS agent descriptions, skill READMEs, and gate evidence. This improves discoverability and aligns with the public spec-kit vocabulary.

---

## 3. Area-by-Area Comparison Table

| Area | Spec Kit | NeatapticTS | Assessment |
|------|----------|-------------|------------|
| **SDD philosophy** | Spec-first, living artifacts, intent → spec → plan → tasks → converge | Phase-based RED→IMPLEMENT→GREEN loop, plan trackers, gates | Equivalent philosophy; spec-kit is file-centric, NeatapticTS is agent/gate-centric |
| **Command loop** | `/speckit.specify`, `/speckit.clarify`, `/speckit.plan`, `/speckit.tasks`, `/speckit.analyze`, `/speckit.checklist`, `/speckit.implement`, `/speckit.converge` | Tier-1 agents `01-planning` → `02-researching` → `03-red-testing` → `04-implementing` → `05-green-testing` → `06-documenting` → `07-logging` | Spec-kit exposes workflow as slash commands; NeatapticTS as agents. Both enforce a loop |
| **Constitution** | `.specify/memory/constitution.md`; loaded by most commands; conflicts are CRITICAL | No single constitution file; rules live in STYLEGUIDE, CONTRIBUTING, agent frontmatter | **Gap:** NeatapticTS needs a consolidated constitution and a constitutional gate |
| **Spec artifact** | `spec.md` with `FR-###`/`SC-###`, acceptance scenarios, ≤3 `NEEDS CLARIFICATION` markers | Acceptance criteria are prose in plan trackers / skill outputs | **Gap:** traceability IDs and a prose-quality gate |
| **Plan artifact** | `plan.md` + `research.md` + `contracts/` + `quickstart.md` | Step-packet YAML + `plans/<step>.md` trackers | Equivalent; spec-kit separates research evidence, NeatapticTS embeds it |
| **Tasks artifact** | `tasks.md` with user-story-driven phases, `[P]` parallel markers, `T###` IDs | Tasks generated by `04-implementing`, no standard template | **Gap:** standard task template with parallel markers |
| **Validation cycles** | `analyze` (read-only gap analysis), `checklist` ("unit tests for English"), `converge` (append-only gaps) | `green-validation-gates`, `plan-sync-validation`, `triaging-test-failures`, `test-fix-workflow` | Equivalent coverage; spec-kit names and artifact paths are clearer |
| **Bug triage** | `speckit.bug.assess` → `fix` → `test`; `.specify/bugs/<slug>/` | `test-fix-workflow`, `triaging-test-failures` | **Gap:** dedicated bug artifact path and URL-trust rules |
| **Extensions/presets** | `extension.yml`, commands, templates, presets, bundles | `.github/skills/*/SKILL.md`, `.github/agents/*.agent.md`, routing table | Same layering idea; spec-kit is more formally cataloged |
| **Disclosure / ISO 42001** | `Assisted-by:` / `Source:` trailers in prompts | `capturing-learning-event` skill, `learning-log.jsonl` | **NeatapticTS strength:** structured learning events; spec-kit only has prompt disclosure |

---

## 4. What Makes Spec Kit Great

1. **It turns opinionated workflow into reusable prompts.** Every command is a markdown file with embedded YAML frontmatter. This makes the workflow portable across agents.
2. **It treats ambiguity as a first-class artifact.** The `NEEDS CLARIFICATION` cap and `## Clarifications` section prevent infinite clarification loops.
3. **It makes quality gates explicit and quotable.** "Unit tests for English" and "coverage ≥ 80%" are simple, memorable rules.
4. **It uses traceability IDs everywhere.** `FR-###`, `SC-###`, `US#`, `AC#`, `CHK-###`, `T###` make cross-referencing cheap.
5. **It separates research, design, contracts, and quickstart** in the plan phase, so design does not happen before evidence.
6. **It has a community extension model.** `extension.yml`, presets, bundles, and a catalog concept let projects compose workflows.
7. **Its bug extension is a complete SDLC in miniature.** Assess → fix → test with URL-trust and evidence artifacts.

---

## 5. What NeatapticTS Should Preserve

1. **ISO 42001 learning-event logging.** Spec Kit has disclosure trailers but no structured event schema. NeatapticTS should keep `capturing-learning-event` as the authoritative evidence log.
2. **Agent tier graph and MCP dispatch.** The `neataptic-dispatch-mcp-*` tools and tier-graph validation are stronger governance than spec-kit's extension manifest alone.
3. **Step-packet YAML and gate contracts.** The explicit `active step packet`, `gate_id`, and `validation allowlist` provide reproducible checkpoints that spec-kit does not formalize.
4. **Phase-agent orchestration.** Eight numbered phase agents plus `00-helping` map cleanly onto spec-kit commands and enable parallel delegation.
5. **Cortex-first search policy.** NeatapticTS's RAG-first research methodology (`research-methodology` skill) is a richer research step than spec-kit's ad-hoc `research.md`.

---

## 6. Concrete Suggested Next Steps

1. **Create `plans/constitution.md`.** Port the five principles from `verbatim/.specify/memory/constitution.md` into NeatapticTS terms. Add a `constitution_check` gate.
2. **Update `01-planning.agent.md`.** Add the `NEEDS CLARIFICATION` cap (≤3 markers) and the "ask at most 5 questions" rule.
3. **Add traceability IDs to `planning-acceptance-criteria`.** Require optional `id: AC-###` on each criterion.
4. **Create `.github/templates/tasks-template.md`.** Use user-story phases and `[P]` parallel markers.
5. **Draft `.github/skills/spec-checklist/SKILL.md`.** Call it "unit tests for English" and require ≥80% traceability.
6. **Document extension catalog pattern.** Write `.github/extensions/README.md` mapping skills/agents/presets to spec-kit concepts.
7. **Keep `learning-log.jsonl` as the audit trail.** Add constitution, spec-checklist, and gate events to the event schema.

---

## 7. Reference Map

| NeatapticTS deliverable | Spec-kit source(s) | Local comparison file(s) |
|------------------------|--------------------|--------------------------|
| `00-overview.md` | `verbatim/README.md`, `verbatim/spec-driven.md` | NeatapticTS routing table, `01-planning.agent.md` |
| `01-sdd-workflow.md` | `verbatim/docs/concepts/sdd.md`, `verbatim/.specify/memory/constitution.md`, `verbatim/templates/commands/specify.md` | `phase-handoff-workflow/SKILL.md` |
| `02-validation-cycles.md` | `verbatim/templates/commands/analyze.md`, `checklist.md`, `converge.md`, `extensions/bug/commands/*.md` | `green-validation-gates/SKILL.md`, `red-test-contracts/SKILL.md`, `test-fix-workflow/SKILL.md`, `triaging-test-failures/SKILL.md` |
| `03-planning-triage.md` | `verbatim/templates/commands/specify.md`, `clarify.md`, `plan.md`, `tasks.md` | `01-planning.agent.md`, `02-researching.agent.md`, `03-red-testing.agent.md`, `planning-acceptance-criteria/SKILL.md`, `plan-sync-validation/SKILL.md`, `plan-alignment/SKILL.md` |
| `04-templates-artifacts.md` | `verbatim/templates/*-template.md` | Local `plans/*.md` and `.github/skills/phase-handoff-workflow/SKILL.md` |
| `05-extensions-ecosystem.md` | `verbatim/extensions/*`, `verbatim/presets/*`, `AGENTS.md` | `.github/skills/*`, `.github/agents/*`, routing table |
| `06-iso42001-learning-overlap.md` | `verbatim/AGENTS.md`, prompt disclosure trailers | `capturing-learning-event/SKILL.md` |
| `07-verbatim-phrases-and-verbs.md` | All core command prompts and templates | Local agent/skill descriptions |

---

## 8. Closing

Spec Kit is a polished, public expression of ideas that NeatapticTS already practices in a more agent-oriented, gate-heavy form. The highest-value assimilation is not to copy spec-kit's file layout, but to borrow its **constitution**, **clarification discipline**, **traceability IDs**, **checklist skill**, **bug extension pattern**, and **exact phrasing**, while continuing to lead on **ISO 42001 learning events**, **MCP dispatch**, and **step-packet gate contracts**.

The verbatim corpus and area analyses in `00-overview.md`–`07-verbatim-phrases-and-verbs.md` provide the detail needed to implement these cherry-picks incrementally.
