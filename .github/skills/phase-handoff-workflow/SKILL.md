---
name: phase-handoff-workflow
description: 'Design and validate the seven-step NeatapticTS agent workflow inside each plan phase. Use when creating copy-pasteable step packets, phase-step handoffs, step stop conditions, sequential orchestration, or when deciding whether a phase can advance.'
argument-hint: 'Name the source phase, target phase, current plan state, and whether the handoff should be user-reviewed or auto-sent.'
user-invocable: false
disable-model-invocation: false
---

# Phase Handoff Workflow

Use this skill when adding or reviewing handoffs between numbered phase agents,
or when a plan file needs its phase structure brought into conformance with the
standard seven-step shape.

This skill owns the durable rules for phase ordering, step packet authoring,
gate contracts, tracker updates, and escalation behavior. It does not own the
content of any particular phase's work — that belongs to the relevant domain
skill.

## When to Use

- A plan file is missing phase boundaries or step packets and needs to be
  structured before work can begin.
- A phase agent has finished its work and the next step packet must be authored
  or reviewed before sending.
- A Green Testing step failed and the correct prior phase for the route-back
  must be identified.
- A Red Testing step is being evaluated for skip eligibility and the plan must
  record an explicit skipped-step packet.
- A numbered agent needs to confirm whether it can advance or must escalate to
  the `00-helping` cross-tier helper.
- The active tracker has drifted from the standard step-packet shape and needs
  to be brought back into conformance.

## Task Packet

Pass a compact packet naming the plan file, the current phase and step status,
and whether the output should be user-reviewed or auto-sent.

```text
Use phase-handoff-workflow for Phase 3 → Phase 4 handoff.
Plan: plans/Worker_Friendly_Network_Serialization_Fastpath.md.
Current state: Phase 3 Step 03 DONE.
Output: draft Step 04 packet for user review before send.
```

## Required Workflow

1. Keep phase order linear: Planning, Research, Red Testing, Implementation,
   Green Testing, Documentation, Session Logging.
2. Every handoff prompt must name the current plan file and the next narrow task.
3. Prefer `send: false` until the repository has evidence that automatic phase
   transitions are safe.
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
9. Update the active tracker after each phase step when one exists.

## Flow-Aware Handoff Contract

Each numbered agent selects a named flow from `.github/flows/` that matches
the current task shape. Flows declare exit gates; every gate must return
`{pass: true, evidence, fixHint, owner}` JSON before the flow is considered
complete. Post-phase fan-out from the flow definition runs after the flow body.

- Gate exceptions are recorded via
  `scripts/agent-customization/gates/record-gate-exception.mjs` and appended to
  `.github/ai-learning/learning-log.jsonl`.
- Three consecutive gate failures in a session escalate automatically to
  `00-helping` via the `00.cross-tier-helper` flow.
- Cross-tier helper calls from any numbered agent route to `00-helping`, which
  resolves the blocker and returns a resolution summary. All cross-tier calls
  are logged as learning events visible to `00.workflow-gap-audit`.
- When a phase step completes its declared flow, the structured-v1 output block
  must include gate evidence in `VALIDATION_EVIDENCE` before handing off.

## Step Packet Shape

Every active or planned executable phase in an active `.plans.md` file should
be organized as a phase boundary plus numbered step packets. The user should be
able to start a fresh session, select the agent named in the current step, and
paste that step packet without relying on prior chat history.

Step 01 always belongs to `01-planning`. It is responsible for planning the rest
of that phase and authoring Step 02-07 packets, or explicit skipped-step
packets, before execution continues.

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
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[WIP|PLANNED|DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/<PlanName>.plans.md'
copy_paste: true
next_step: 'Step 02 — Planner-defined by this step'
skills:
  - '<canonical-skill>'
specialists:
  - '<hidden specialist agent, if useful>'
validation:
  - '<focused command or manual evidence gate>'
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

Completed phases **must** have their history compressed to a concise coverage
note before the next phase is started or the workstream is closed. The
`phase-compression` gate (enforced in `01.phase-kickoff` and
`07.tracker-closure`) validates this requirement. Do not author the next
phase's step packets until the previous `[DONE]` phase has a compact summary
line in place of its verbose transcript.
Completed steps inside active phases should also be compressed to concise done
notes. Active and newly planned phases should use step packets and keep only
the current step copy-pasteable.

Tracker YAML step packets are workflow artifacts. They do not replace the
required Tier-0 or Tier-1 `structured-v1` chat envelope, which remains the
mandatory response shape when those agents answer in chat.

## Guardrails

- Do not author a step packet that requires prior session context to understand;
  every step must be self-contained for a fresh session.
- Do not set `send: true` on a handoff without explicit evidence that automatic
  phase transitions are safe in the current plan.
- Do not skip Red Testing for behavior changes without recording an explicit
  skipped-step packet and the reason in the plan.
- Do not continue forward from a failed Green Testing step; route back to the
  smallest relevant prior phase.
- Do not embed full workflow logic in step packets; packets reference skills,
  they do not duplicate them.

## Expected Final Output

A strong phase handoff pass should produce:

- the updated plan file with the next step packet in paste-ready form,
- a confirmed `send` decision and model assignment,
- any validation required before the receiving agent can start,
- the current tracker status reflecting the completed or in-progress step.
