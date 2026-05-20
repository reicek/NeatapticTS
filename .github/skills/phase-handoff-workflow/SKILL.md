---
name: phase-handoff-workflow
description: 'Design and validate the seven-step NeatapticTS agent workflow inside each plan phase. Use when creating copy-pasteable step packets, phase-step handoffs, step stop conditions, sequential orchestration, or when deciding whether a phase can advance.'
argument-hint: 'Name the source phase, target phase, current plan state, and whether the handoff should be user-reviewed or auto-sent.'
user-invocable: false
disable-model-invocation: false
---

# Phase Handoff Workflow

Use this skill when adding or reviewing `handoffs` between numbered phase agents.

## Workflow

1. Keep phase order linear: Planning, Research, Red Testing, Implementation, Green Testing, Documentation, Session Logging.
2. Every handoff prompt must name the current plan file and the next narrow task.
3. Prefer `send: false` until the repository has evidence that automatic phase transitions are safe.
4. Use `handoffs.model` only with validated qualified model names.
5. Treat Red Testing and Green Testing as value gates, not ceremony. Keep them
   when they add independent evidence for behavior, runtime, or user-facing
   artifact changes; otherwise record an explicit skipped-step packet and move
   to the next value-adding step.
6. Do not skip Red Testing for behavior changes unless the active plan records
   why no honest failing test or eval can be created before implementation.
7. Do not add a standalone Green Testing step when the same focused validation
   is already required in the implementation or closure step and a separate
   session would only duplicate evidence.
8. Route failed Green Testing back to the smallest relevant prior phase instead
   of continuing forward.
9. Update `plans/Agentic_Workflow_Architecture.plans.md` after each phase step.

## Phase Step Standard

Every active or planned executable phase in an active `.plans.md` file should
be organized as a phase boundary plus numbered step packets. The user should be
able to start a fresh session, select the agent named in the current step, and
paste that step packet without relying on prior chat history.

Step 01 always belongs to `01-planning`. It is responsible for
planning the rest of that phase and authoring Step 02-07 packets, or explicit
skipped-step packets, before execution continues.

Use this shape:

````md
### Phase N — <Outcome> [PLANNED|WIP|DONE]

**Phase objective:** One concise paragraph describing the phase outcome.

**Phase progression rule:** Start with only Step 01. Step 01 must author the
remaining numbered step packets, or explicit skipped-step packets, before the
phase can advance.

#### Step 01: <Step Outcome> [WIP|PLANNED|DONE]

```yaml
phase: N
step: 1
agent: "01-planning"
agent_file: ".github/agents/01-planning-architect.agent.md"
status: "[WIP|PLANNED|DONE]"
mode: "fresh-session"
source_of_truth: "plans/<PlanName>.plans.md"
copy_paste: true
next_step: "Step 02 — Planner-defined by this step"
skills:
	- "<canonical-skill>"
specialists:
	- "<hidden specialist agent, if useful>"
validation:
	- "<focused command or manual evidence gate>"
```

**User instruction:** Start a fresh session, select `<agent>`, and paste this
full step packet.

**Step objective:** One concise paragraph describing what this step must
produce.

**Context the agent must know:** Bullets that assume current repo state only.

**Execution steps:** Numbered actions scoped to this step only.

**Stop conditions:** Exact done, blocked, and route-back conditions.

**Required validation:** Commands, scripts, manual checks, or evidence required
before advancing.

**Plan update requirement:** Update the source plan with changes, remaining
work, validation evidence, and the next active step before ending.

**Whole-step copy rule:** The entire step block above is the prompt. Do not
append a second nested `Copy-paste prompt` subsection.
````

Completed phases may keep compressed legacy coverage notes until reopened, and
completed steps inside active phases should also be compressed to concise done
notes. Active and newly planned phases should use step packets and keep only
the current step copy-pasteable.

## Output Contract

Return the source phase, current step, target step, step packet metadata, any
edits needed to keep the next step paste-ready, model choice, `send` decision,
and validation required before advancing.

## Sources

- VS Code custom agents documentation describes handoff buttons and `handoffs` frontmatter.
- The repo tracker defines the seven-phase sequence and central plan update rule.