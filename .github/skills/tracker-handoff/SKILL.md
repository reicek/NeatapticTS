---
name: tracker-handoff
description: 'Use when: standardizing NeatapticTS `.plans.md` or `.logs.md` trackers, including creating or refreshing active trackers, compressing completed history, managing `[PLANNED]`/`[WIP]`/`[DONE]`, handling intentional parallel lanes, archiving closed tracker pairs in `plans/completed/`, or updating a reusable `Handoff query` for safe session continuation.'
argument-hint: 'Describe the tracker path, active vs closed intent, single-lane or parallel-lane state, history to compress, validations required, and what the next session must resume safely.'
user-invocable: true
disable-model-invocation: false
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Tracker And Handoff Playbook

Use this skill when work needs durable continuity in markdown tracker files.

This skill is the canonical workflow for `.plans.md` and `.logs.md` structure in
NeatapticTS. It owns tracker status markers, compression rules for old work, the
required `Handoff query` section that makes WIP sessions safe to resume with a
copy-paste prompt, explicit closure mechanics for finished plans, and the
parallel-lane rules needed when one workstream honestly has more than one active
frontier.

Other skills may decide **when** a tracker should change, but they should defer
the tracker shape itself to this skill instead of redefining status markers,
handoff layout, compression policy, or archive conventions ad hoc.

## When To Use

- A `.plans.md` file needs to be created, rewritten, compressed, or updated.
- A `.logs.md` file needs a concise done-state record.
- A long-running workstream needs a safe continuation prompt for the next
  session.
- Older tracker history has become too verbose and needs compression.
- A workflow skill or agent needs a standard tracker format instead of a custom
  local convention.
- A plan is closing and needs the correct `.plans.md` + `.logs.md` archive pair
  under `plans/completed/`.
- A workstream truly needs parallel active lanes and the tracker must represent
  them intentionally instead of drifting into multiple accidental `[WIP]`
  branches.

## Task Packet

Pass a compact packet that names the tracker, the intended tracker state, the
active frontier, and any validations or archive moves that must happen before
the pass can be called complete.

```text
Use tracker-handoff for plans/<Workstream>.plans.md.
Intent: <create | refresh active tracker | compress history | close and archive>.
Current state: <status marker, active frontier, known blocker if any>.
Lane model: <single active lane | named parallel lanes>.
History to compress: <completed sections or none>.
Validation: <manual only | command list>.
Next-session goal: <what the next session should continue safely>.
```

## Quick Start Checklist

1. Identify the file role: active `.plans.md`, accumulating `.logs.md`, or
   closing archive pair.
2. Confirm the tracker vocabulary uses only `[PLANNED]`, `[WIP]`, and `[DONE]`.
3. Preserve detail only at the active frontier; compress completed history into
   short coverage notes.
4. Add, refresh, or intentionally remove `## Handoff query` based on whether the
   workstream is still resumable in-context.
5. If the plan participates in flow-aware steps, run the relevant sync or gate
   command before declaring the pass finished.
6. Summarize what changed, what remains active, and what validation evidence now
   exists.

## Required Workflow

1. Confirm whether the tracker is active, accumulating, closing, or reopening.
2. Decide whether the workstream has one active frontier or a justified
   multi-lane frontier. Default to a single active lane.
3. Normalize the status markers before changing prose; drifted status vocabulary
   causes downstream confusion.
4. Keep the active frontier detailed, but compress completed work into concise
   coverage notes that prevent re-exploration.
5. Add or refresh `## Handoff query` for active work. Remove it from terminally
   closed plans unless the user explicitly wants reopen guidance.
6. If the tracker participates in flow-aware phase steps, update the tracker and
   then run the appropriate sync or gate command from the validation section
   below.
7. If the workstream is closing, create or refresh the same-boundary `.logs.md`
   file before moving the pair into `plans/completed/`.
8. Record only durable evidence in the tracker summary. Do not paste long raw
   command transcripts into the tracker.

## Canonical Tracker Rules

### File Roles

- Use `.plans.md` for active work, pending decisions, next steps, and the live
  `Handoff query`.
- Use `.logs.md` for compressed done-state records and completed work that no
  longer needs active session guidance.
- Keep active trackers in `plans/` and move terminally closed `.plans.md` plus
  `.logs.md` pairs into `plans/completed/`.

### Status Vocabulary And Active Frontier

Every durable tracker should use the same three states:

- `[PLANNED]` for queued work not yet in motion.
- `[WIP]` for the active workstream or active lane.
- `[DONE]` for completed durable coverage.

Apply them consistently:

- The document-level status line should use exactly one of these markers.
- Prefer exactly one active `[WIP]` section in a `.plans.md` file.
- Completed historical sections should become `[DONE]` coverage notes rather
  than verbose transcripts.
- Queued near-term work should be `[PLANNED]`.

### Parallel Workstream Policy

Some workstreams honestly require more than one active lane, but parallelism
must be explicit, named, and bounded.

- Default to one active lane. Multiple `[WIP]` sections are an exception, not a
  convenience.
- Only keep parallel lanes when the work is truly independent enough that one
  lane can pause without blocking the others.
- Name each active lane by responsibility, not by person or session.
- Keep one short coordinator section near the top that states how the lanes
  relate and which lane should move next if only one can progress.
- Give each lane its own next step and validation note so later sessions do not
  have to infer ownership or order.
- Refresh the `Handoff query` so it names the exact lane that the next session
  should continue. Do not make the next session rediscover which lane matters.

### Handoff Query Section

Active `.plans.md` files should include a `Handoff query` section whenever the
workstream may need to continue in a future session.

The section should:

- be easy to copy and paste,
- assume only current repo state, not prior chat history,
- name the current boundary or active work item,
- state what is already covered,
- state the next narrow task,
- include required validations,
- mention known worktree cautions when relevant.

Preferred shape:

````md
## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
<workstream-specific continuation prompt>
```
````

Closed `.plans.md` files should normally omit `Handoff query` because the plan
has no in-context follow-up step left to hand off.

### Compression Rules

When a tracker grows too long:

- keep the active frontier detailed,
- compress completed passes into minimal coverage notes,
- keep only the information needed to avoid re-exploring covered work,
- remove repetitive validation transcripts once the result is captured,
- prefer concise extent statements over narrative replay.

### Per-Phase Compression Rule (enforced)

When any phase is marked `[DONE]` — whether the workstream continues or closes —
the owning agent **must** compress that phase's history to a concise coverage
note before authoring the next phase's step packets or closing the workstream.

- Replace verbose step transcripts and raw validation output with a single
  compact summary line, e.g.:
  `[DONE] Phase 2: Implemented X, hardened Y, coverage gate passed.`
- The `phase-compression` gate (enforced in `01.phase-kickoff` and
  `07.tracker-closure`) validates this requirement.
- Do not advance to the next phase or close the workstream until the gate
  evidence is recorded in `VALIDATION_EVIDENCE`.

### Completion Closure Rule

When a workstream becomes fully complete, the final tracker step is always to
close it deliberately.

- Compress the completed `.plans.md` file into a short closed tracker that
  preserves only the durable reopen-point context.
- Add or update a same-boundary `.logs.md` file with the durable done-state
  record.
- Move the closed `.plans.md` file and its same-boundary `.logs.md` record into
  `plans/completed/` as the last tracker action before any next workstream
  begins.
- Remove active-session scaffolding that no longer applies, especially stale
  `Remaining gaps`, `Immediate next steps`, or `Handoff query` sections.
- Do not preserve or emit a next-session handoff prompt on a fully closed plan
  unless the user explicitly asks for reopen guidance.
- If there is still a real next step from the plan's own context, the plan is
  not closed yet and should stay active with a `Handoff query`.

### Flow-Aware Tracker Closure

Tracker closure runs through the `07.tracker-closure` flow. The flow requires
the `log-completion-marker` gate to pass before the workstream is considered
closed. The gate confirms that a compressed log entry is present and the target
phase is marked `[DONE]`. Closure must also clear the `stale-wip-plans` gate so
an all-`[DONE]` plan cannot remain top-level `[WIP]` after the archive handoff.

When closing a tracker that was run with flow-aware phase steps:

- The `VALIDATION_EVIDENCE` section of the final structured-v1 output block
  must include `log-completion-marker gate: pass` evidence.
- It must also include `stale-wip-plans gate: pass` evidence, confirming the
  archived plan no longer appears as a stale top-level `[WIP]` tracker.
- If the workstream produced gate exceptions, confirm they are recorded in
  `.github/ai-learning/learning-log.jsonl` before compressing.

### Heading And Date Rules

- Use stable undated headings.
- Do not prepend calendar dates to tracker sections.
- Keep section titles reusable across sessions.

## Automation And Validation

Prefer the narrowest validation that matches the tracker change. Record the
result as durable evidence, not as a raw transcript dump.

### Common commands

```bash
node .github/hooks/workflow-update-sync.mjs --plan=<active-plan-path> --json
node scripts/agent-customization/validate-plan-sync.mjs --json --plan=<active-plan-path>
node scripts/agent-customization/gates/phase-compression.gate.mjs --json
node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json
node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json
```

### Which command to use

- Run `workflow-update-sync.mjs` after editing an active phase-step plan when the
  next `[PLANNED]` step should advance automatically to `[WIP]`.
- Run `validate-plan-sync.mjs` when the plan's active status, README index
  entry, roadmap placement, or closure registration changed.
- Run `phase-compression.gate.mjs` after marking a phase `[DONE]` but before
  advancing to the next phase or closing the workstream.
- Run `log-completion-marker.gate.mjs` when the workstream is closing and the
  same-boundary `.logs.md` record was added or refreshed.
- Run `stale-wip-plans.gate.mjs` when a plan was closed, archived, or otherwise
  changed from active to terminal so stale top-level `[WIP]` markers do not
  survive.

### Validation reporting rule

- Report the command, whether it passed, and the one-line evidence or fix
  applied.
- Do not paste the entire JSON payload into the tracker unless the user asked
  for raw output preservation.
- If validation is intentionally deferred, say why and what exact next command
  remains.

## Edge Cases And Recovery Rules

### Reopening Archived Work

- If a closed workstream needs new work, prefer reopening intentionally rather
  than silently editing the archived pair in place.
- Either move the pair back into `plans/` or create a new active tracker that
  points to the archived record in `plans/completed/`.
- Add a fresh `Handoff query`; do not reuse a stale closure-era prompt.

### Interrupted Mid-Validation State

- If edits are made but validation did not complete, keep the tracker `[WIP]`.
- State the exact missing validation command and any known blocker.
- Do not compress the frontier so aggressively that the next session has to
  rediscover what remains unverified.

### Missing Same-Boundary Log On Closure

- A plan is not fully closed until the same-boundary `.logs.md` file exists and
  records the durable done state.
- If the log is missing, create or refresh it before moving anything into
  `plans/completed/`.

### All Phases Done But Top-Level Status Still WIP

- Treat this as a closure bug, not harmless drift.
- Fix the tracker status, refresh closure evidence, and run the
  `stale-wip-plans` gate before claiming the archive handoff is complete.

### Parallel Lanes Becoming Sequential Again

- When one lane finishes or becomes blocked permanently, collapse the remaining
  work back to a single explicit active frontier.
- Remove obsolete lane headings and refresh the `Handoff query` so it no longer
  implies parallel work that no longer exists.

## Feedback And Improvement Loop

When tracker work repeatedly needs the same manual cleanup, treat that as a
workflow-quality signal instead of normal operator burden.

- If tracker friction comes from missing tracker guidance, improve this skill.
- If it comes from step-packet drift, route to `phase-handoff-workflow`.
- If it comes from plan/README/Roadmap misalignment, route to
  `plan-sync-validation`.
- If a recurring tracker-system gap was fixed, record it with
  `capturing-learning-event` so future sessions inherit the lesson.
- Prefer fixing the durable tracker workflow once over repeating a local
  workaround in each new tracker.

## Recommended Active Plan Shape

For most `.plans.md` files, prefer this shape:

1. `# <Workstream name>`
2. `**Status:** [PLANNED|WIP|DONE]`
3. `## Scope`
4. `## Current state`
5. `## Coverage backlog`
6. `## Immediate next steps`
7. `## Deferred questions` when needed
8. `## Handoff query`

Not every plan needs every section, but active plans should remain compact,
forward-looking, and resumable.

## Recommended Closed Plan Shape

For terminally closed `.plans.md` files, prefer this shape:

1. `# <Workstream name>`
2. `**Status:** [DONE]`
3. `## Scope`
4. `## Final state`
5. `## Audit summary`
6. `## Reopen conditions`
7. `## Audit log`

## Recommended Log Shape

For `.logs.md` files, prefer:

1. `# <Workstream log>`
2. `**Status:** [DONE]` or `[WIP]` when still accumulating done-state records
3. short done-state entries grouped by durable milestone
4. no active TODO list unless the file is intentionally dual-purpose
5. the same boundary as the closed `.plans.md` file whenever the workstream is
   complete

## Guardrails

- Do not use `[]` and `[DONE]` as a mixed progress vocabulary. Use only
  `[PLANNED]`, `[WIP]`, and `[DONE]`.
- Do not keep multiple active `[WIP]` branches in one tracker unless the user
  explicitly wants parallel active lanes **and** each lane is named, scoped, and
  resumable on its own.
- Do not bury the next-session continuation prompt in prose.
- Do not leave a `.plans.md` file without a `Handoff query` section when the
  workstream is still active.
- Do not mark a workstream `[DONE]` and stop before compressing the plan,
  adding or updating the same-boundary `.logs.md` file, and moving both files
  into `plans/completed/`.
- Do not leave a stale `Handoff query` on a fully closed plan unless the user
  explicitly wants reopen guidance.
- Do not preserve detailed historical narration when compact coverage notes are
  enough to prevent re-exploration.
- Do not rewrite generated README files just to record progress; use trackers.
- Do not treat validation as optional ceremony when the tracker participates in a
  flow-aware plan or archive handoff.

## Expected Final Output

A strong tracker update should report:

- which tracker file was updated,
- whether it is now `[PLANNED]`, `[WIP]`, or `[DONE]`,
- whether the tracker uses a single active lane or named parallel lanes,
- what historical content was compressed,
- what the new active frontier is,
- whether a `Handoff query` section was added, refreshed, or intentionally
  removed because the plan was terminally closed,
- whether a same-boundary `.logs.md` file was added or refreshed,
- whether any workflow sync or gate command passed,
- whether the closed tracker pair now lives in `plans/completed/`.
