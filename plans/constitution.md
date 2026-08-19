# NeatapticTS Project Constitution

> This document is the **constitution authority** for all plan, skill, gate, and
> agent-customization work in the `NeatapticTS` repository. Every change that
> touches the architecture of how plans, skills, gates, or agent routing are
> authored, validated, or evolved must remain consistent with the principles
> below. When a principle and a convenience conflict, the principle wins.

---

## Core Principles

### 1. AI is a thinking partner, not a replacement

**ID:** `principle-1-ai-thinking-partner`

Humans and AI collaborate. AI can propose, expand, compress, refactor, and
validate, but the repository owner approves or rejects every material change.
AI agents must surface trade-offs, ask for clarification when requirements are
ambiguous, and never hide assumptions behind opaque defaults.

### 2. Human owns the mission; AI owns the method inside the guardrails

**ID:** `principle-2-human-mission-ai-method`

The human decides what should be built and why. The agent decides how to
implement it within the bounds of this constitution, the active plan, and the
relevant skills. If the "how" would violate a principle, the agent must stop,
record the conflict, and escalate rather than silently work around it.

### 3. Verbatim sections are binding; summaries are guidance

**ID:** `principle-3-verbatim-binding`

When a plan, skill, or gate document contains an explicit verbatim block (for
example a step packet shape, a YAML schema, or a structured-v1 contract), that
block is binding. Summaries, diagrams, and prose explain intent but do not
override the verbatim rules. If a verbatim block becomes wrong, change the
block; do not rely on a charitable reading of the summary.

### 4. Prefer breadth-first, shallow, and recoverable

**ID:** `principle-4-breadth-first-recoverable`

Favor many small, independently testable slices over one deep monolithic
change. Keep rollback paths cheap. Use RED-then-green workflows. Avoid
irreversible edits to durable records such as the learning-log, archived plans,
or published schemas. When in doubt, add a new file or a new version rather than
mutating the old one in place.

### 5. Every traceable artifact gets a unique ID

**ID:** `principle-5-unique-ids`

Plans, phases, steps, slices, acceptance criteria, skills, agents, gates,
learning events, and validation artifacts must carry stable, unique identifiers
where practical. IDs make handoffs, audits, regression analysis, and
phase-compression possible. Generated IDs must be deterministic or
human-readable, never opaque session-only tokens.

---

## Security & Cross-Platform Constraints

**ID:** `section-security-cross-platform`

- **No secrets in source.** Credentials, tokens, private keys, and API secrets
  must never be committed. If an agent detects a secret, it must stop and report
  it immediately.
- **Do not leak context outside the repository.** Project instructions, plan
  content, skill text, and learning events are internal workflow artifacts.
  Agents must not send them to third-party services unless explicitly authorized
  by the repository owner.
- **CI-compatible by default.** Scripts, gates, and tests must run on Linux and
  in headless/Chromium environments. Do not assume Windows-only paths, GUI
  tools, or local-only services.
- **Browser and worker parity.** Features that claim to work in Node must
  document what is expected to work in the browser and in Web Workers, and must
  not promise cross-environment determinism beyond the level the
  `reproducibility-contracts` skill defines.

---

## Development Workflow & Quality Gates

**ID:** `section-development-workflow`

- **Plan before implementation.** Active work must be represented in a `.plans.md`
  tracker before code is changed, unless the change is an emergency fix that is
  immediately recorded in the plan afterwards.
- **RED before green.** New behavior starts with a failing assertion, script, or
  test that proves the gap. Implementation makes the assertion pass. Validation
  confirms it stays passing.
- **One slice, one intent.** A slice should change at most a few related files
  for one behavioral purpose. Large cross-cutting changes must be sliced across
  multiple steps or phases.
- **Quality folder gate for `src/`.** Every change under `src/` must pass
  `npm run quality:folder -- --folder=<touched-folder>` or an equivalent
  focused validation. Full-suite runs happen only when explicitly required.
- **No deferred cleanup.** When a migration, refactor, or API replacement is
  performed, the old code, old imports, old tests, and old type exports must be
  removed in the same slice that introduces the replacement.

---

## Governance / Versioning / Compliance

**ID:** `section-governance-versioning`

- **SemVer ratification.** This constitution is versioned. A new version must
  ratify the previous version or explicitly document a breaking change and the
  migration path.
- **Plan authority.** A plan file in `plans/` is the source of truth for the
  workstream it names. The plan index in `plans/README.md` must list every
  active plan. Stale plans are archived to `plans/completed/` and marked
  `[DONE]`; never delete plan history.
- **Skill authority.** A skill document in `.github/skills/` or
  `.github/agents/` is the source of truth for the workflow it names. Skills are
  living documents and may be updated by the agent that owns them, but the
  update must be captured as a learning event.
- **Gate authority.** Gate scripts are the executable definition of workflow
  validity. A gate failure blocks handoff until it is resolved or a documented
  exception is recorded.
- **Learning events are append-only.** `.github/ai-learning/learning-log.jsonl`
  is an append-only audit trail. Events may not be edited, deleted, or
  back-dated. They must not contain timestamps, session identifiers, or other
  ephemeral data.

---

**Version:** 1.0.0  
**Ratified by:** `plans/Spec-Kit_Assimilation.plans.md` Phase 1 Step 01  
**Constitution authority:** This document.
