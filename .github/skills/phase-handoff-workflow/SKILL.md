---
name: phase-handoff-workflow
description: 'Use when: designing phase gates, handoffs, or slice boundaries.'
argument-hint: 'Name the source phase, target phase, current plan state, and whether the handoff should be user-reviewed or auto-sent.'
user-invocable: false
disable-model-invocation: false
skills:
  - execute
  - tracker-handoff
  - plan-sync-validation
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Phase Handoff Workflow

Use this skill when adding or reviewing handoffs between numbered phase agents,
or when a plan file needs its phase structure brought into conformance with the
standard seven-step shape.

This skill owns the durable rules for phase ordering, step packet authoring,
gate contracts, tracker updates, and escalation behavior. Handoffs rely on
append-only convergence in the tracker so phase history remains reconstructible.
It does not own the content of any particular phase's work — that belongs to
the relevant domain skill.

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

## When NOT to use

Do NOT use for single-phase work without inter-phase handoffs. Do NOT use for tracker management - use `tracker-handoff` instead.

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
2. Every handoff prompt must state ONLY the next narrow task (slice ID or step
   ID) and instruct the receiver to load context via Cortex MCP and any
   declared pre_execute_hook/get_slice_context. Do not embed file lists, plan
   details, or step-by-step instructions in the prompt. All agents except
   `00-helping` and `01-planning` MUST receive their instructions via RAG.
   See `execute` skill Section 2.2 for the full RAG-Based Dispatch Policy.
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
10. After updating the tracker, invoke the workflow sync hook to advance the step
    markers automatically:

```bash
node .github/hooks/workflow-update-sync.mjs --plan=<active-plan-path> --json
```

This marks the current step `[DONE]` in the plan header AND the YAML
`status:` field, and advances the next `[PLANNED]` step to `[WIP]`.
**Exception:** if the step is explicitly awaiting a user response before the
next step can start, the hook may be skipped and the step left `[WIP]` until
the user replies. Record the hold reason in the plan.
If the hook returns `"actionTaken": "blocked"`, verify that the next step
exists and has `[PLANNED]` status before escalating. 11. Before any strict write/execute action inside the current step, prepare the
repo-owned runtime proof carrier with
`node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --prepare ...`
so pretool and posttool enforcement can validate the flow, delegator chain,
required skills, required specialists, and action class. Use
`.github/runtime-enforcement-contract.md` as the canonical contract for that
payload. 12. **Phase Compression (mandatory).** When all steps in a phase are marked
`[DONE]` and green validation has passed, the orchestrator MUST dispatch
`07-logging` to compress the completed phase before advancing to the next
phase or performing the phase-to-phase handoff. **Step-level compression**
should happen as each step completes — move the step's YAML packet and
details to `.logs.md` and keep a compact `[DONE]` reference in the plan.
Do not wait for the entire phase to finish before compressing individual
`[DONE]` steps. Compression means:

1. Move detailed step/slice/VALIDATION_EVIDENCE blocks from the plan file
   to the corresponding `.logs.md` file.
2. Replace the detailed content in the plan file with a compact `[DONE]`
   marker and a reference to the logs file.
3. Keep the phase header, goal, and status as `[DONE]` in the plan file.

This keeps plan files lean and focused on active work. Plan files should
never carry verbose `[DONE]` phase details — those belong in logs.

Skipping phase compression is a workflow violation. The orchestrator must
not advance to the next phase or send the phase-to-phase handoff until
compression is complete.

### Mandatory Plan Verification Loop

Before any phase step with `goal: red-testing` or `goal: implementing` is
dispatched, the plan must pass an independent verification pass:

1. **After Step 01 planning packets are authored**, the orchestrator dispatches
   a **fresh** `01-planning` agent in verification mode to read the active plan.
2. **The verifier checks:** completeness of phase/step YAML blocks, risk
   coverage, acceptance criteria quality, dependency ordering, and consistency
   with the phase objective.
3. **The verifier records the verdict in the plan's
   `## Latest validation evidence` section:**
   - `green-light: true` or `status: green-light` when the plan is ready for
     execution.
   - Blockers with `green-light: false` or `status: blocked` when the plan is
     not ready.
4. **If blockers are recorded**, the orchestrator routes back to `01-planning`
   for a patch cycle, then dispatches a new verification agent. This patch →
   verify loop repeats until green light is recorded.
5. **Only after a green-light verdict** may Agent Zero dispatch
   `03-red-testing`, `04-implementing`, or other execution-phase agents.

Every active `.plans.md` file must include a `## Latest validation evidence`
section containing the most recent verification verdict and timestamp. The
`plan-readiness` gate (`scripts/agent-customization/gates/plan-readiness.gate.mjs`)
verifies this marker before execution-phase dispatch.

## Flow-Aware Handoff Contract

Each numbered agent selects a named flow from `.github/flows/` that matches
the current task shape. Flows declare exit gates; every gate must return
`{pass: true, evidence, fixHint, owner}` JSON before the flow is considered
complete. Post-phase fan-out from the flow definition runs after the flow body.

- The `plan-readiness` gate must return `pass: true` before any handoff to a
  step with `goal: red-testing` or `goal: implementing`. The gate checks for a
  green-light marker recorded by a fresh `01-planning` verification pass in the
  plan's `## Latest validation evidence` section.
- The `spec-checklist` skill is a **pre-implementation quality gate**. Before
  any dispatch to `04-implementing`, run it read-only to validate prose quality,
  traceability coverage (≥ 80%), and ID coverage; block dispatch until any
  `missing`, `partial`, `contradicts`, or `unrequested` gaps are resolved.
- Gate exceptions are recorded via
  `scripts/agent-customization/gates/record-gate-exception.mjs` and appended to
  `.github/ai-learning/learning-log.jsonl`.
- Gate failures are recorded for audit but do not trigger automatic
  escalation. Continue retrying until resolved or a true technical limit is
  reached. No concessions.
- Cross-tier helper calls from any numbered agent route to `00-helping`, which
  resolves the blocker and returns a resolution summary. All cross-tier calls
  are logged as learning events visible to `00.workflow-gap-audit`.
- When a phase step completes its declared flow, the structured-v1 output block
  must include gate evidence in `VALIDATION_EVIDENCE` before handing off.

## Step Packet Shape

Every active or planned executable phase in an active `.plans.md` file should
be organized as a phase boundary plus numbered step packets. The user should be
able to paste the step packet without relying on prior chat history.

Step 01 always belongs to `01-planning`. It is responsible for planning the rest
of that phase and authoring Step 02-07 packets, or explicit skipped-step
packets, before execution continues.

### Field Definitions

| Field                | Required | Description                                                                                                                                                                                                                                                             |
| -------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `phase`              | Yes      | Phase number (integer)                                                                                                                                                                                                                                                  |
| `step`               | Yes      | Step number within the phase (integer)                                                                                                                                                                                                                                  |
| `goal`               | Yes      | What outcome this step needs. Must be one of: `planning`, `researching`, `red-testing`, `implementing`, `green-testing`, `documenting`, `logging`, `helping`                                                                                                            |
| `tdd_sequence`       | No       | How the orchestrator should decompose this step across phases. Must be one of: `red-green`, `green-only`. When absent, single-phase dispatch                                                                                                                            |
| `status`             | Yes      | Step status: `[PLANNED]`, `[WIP]`, or `[DONE]`                                                                                                                                                                                                                          |
| `mode`               | Yes      | Session mode: `fresh-session` or `perpetual`                                                                                                                                                                                                                            |
| `source_of_truth`    | Yes      | Path to the authoritative plan file                                                                                                                                                                                                                                     |
| `copy_paste`         | Yes      | Whether the step packet is a paste-ready prompt (`true`/`false`)                                                                                                                                                                                                        |
| `next_step`          | Yes      | Description of the next step, or `null` for terminal steps                                                                                                                                                                                                              |
| `skills`             | Yes      | List of skill names the agent should load                                                                                                                                                                                                                               |
| `specialists`        | No       | List of hidden specialist agent names for delegation                                                                                                                                                                                                                    |
| `validation`         | Yes      | List of validation commands or evidence gates                                                                                                                                                                                                                           |
| `pre_execute_hook`   | No       | Optional pre-execution hook declaring `{ tool, args }`. The named tool (e.g. `neataptic-workflow-mcp/get_slice_context`) is invoked with the provided args before the RED/IMPLEMENT phase begins, giving the dispatched specialist deterministic context from the plan. |
| `constitution_check` | No       | Stable principle identifiers from `plans/constitution.md` that this step exercises. Informational; preserved by plan-sync and reported in handoffs.                                                                                                                     |

The orchestrator resolves `goal` to the dispatched agent using the routing
table in `.github/copilot-instructions.md` §3. When `tdd_sequence` is present,
the orchestrator decomposes the step across the specified phases rather than
dispatching a single agent.

### Pre-execute hook

A step packet may declare a `pre_execute_hook` so that the specialist retrieves
plan-specific context before reading files or beginning implementation. When the
field is present, the orchestrator (or the receiving agent) MUST call the named
tool with the supplied `args` before the RED phase begins. The hook must have the
shape `{ tool: string, args: object }` and is validated by the `step-packet`
gate.

The canonical use is to fetch a slice context from the workflow MCP server:

```yaml
pre_execute_hook:
  tool: 'neataptic-workflow-mcp/get_slice_context'
  args:
    slice_id: 'C1-gate-and-skill'
```

This keeps the specialist's starting context deterministic and scoped to the
active slice, rather than relying on prior session memory. When no hook is
present, the receiving agent must follow the Cortex-First Search Policy from
`research-methodology` to load its own context.

### Backward Compatibility

During the migration transition, step packets may still contain the deprecated
`agent` and `agent_file` fields:

- If a step YAML contains `agent:` but not `goal:`, treat `agent` as an alias
  for `goal` using the mapping: `00-helping` → `helping`, `01-planning` →
  `planning`, `02-researching` → `researching`, `03-red-testing` →
  `red-testing`, `04-implementing` → `implementing`, `05-green-testing` →
  `green-testing`, `06-documenting` → `documenting`, `07-logging` → `logging`.
- If a step YAML contains both `agent:` and `goal:`, `goal` takes precedence;
  emit a deprecation warning.
- If a step YAML contains neither `agent:` nor `goal:`, the step-packet gate
  MUST fail with a fixHint explaining that one of these fields is required.
- `agent_file` is redundant because the orchestrator derives the file path from
  the goal-derived agent name.

This backward-compatibility period lasts until all plan files in `plans/` have
been migrated to the new format, at which point `agent` and `agent_file` become
invalid.

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
goal: 'planning'
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

**User instruction:** Paste this full step packet.

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

### Traceability Table (recommended)

For steps whose acceptance criteria map to concrete file changes and validation commands, include a `traceability` table in the step packet or in the plan's `VALIDATION_EVIDENCE` section. The table makes every `AC-###` criterion machine-traceable and supports the test-backed / gate-backed change principle in `plans/constitution.md`.

Use this shape:

```yaml
traceability:
  - id: AC-001
    criterion: 'buildMLP() default config produces 5-node network'
    files_changed:
      - 'src/architecture/network/builders/mlp/*.ts'
    validation_command: 'npx jest --testPathPattern=builders/mlp'
  - id: AC-002
    criterion: 'buildMLP() rejects empty hiddenLayers with actionable error'
    files_changed:
      - 'src/architecture/network/builders/mlp/*.ts'
      - 'src/architecture/network/builders/mlp/*.errors.ts'
    validation_command: 'npx jest --testPathPattern=builders/mlp'
```

Each row must reference one `AC-###` identifier, the concrete criterion text, the exact files changed, and the validation command that proves it. Gap types to flag while authoring the table: `missing` (no `AC-###` for a file change), `partial` (criterion lacks a validation command), `contradicts` (command does not cover the listed files), or `unrequested` (validation command present but no matching criterion).

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

## Workflow Diagram

```text
Flowchart summary: "Phase completes" → "Author step packet"; "Author step packet" → "Set send/model"; "Set send/model" → "Dispatch to next phase"; "Dispatch to next phase" → "Next phase executes"; "Next phase executes" → "Green pass?"; "Green pass?" → "Advance to next phase" (Yes), "Route back to prior phase" (No); "Advance to next phase"; "Route back to prior phase" → "Author step packet".
```

## Decision Tree

```text
Flowchart summary: "Phase step result" → "Green testing?"; "Green testing?" → "Advance to next phase" (Pass), "Loop back to prior phase" (Fail); "Advance to next phase" → "More steps?"; "Loop back to prior phase" → "3 failures? Escalate to 00-helping"; "More steps?" → "Author next step packet" (Yes), "Close phase" (No); "3 failures? Escalate to 00-helping"; "Author next step packet"; "Close phase".
```

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
