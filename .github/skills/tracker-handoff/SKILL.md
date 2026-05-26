---
name: tracker-handoff
description: 'Standardize durable tracker files in NeatapticTS. Use when creating or updating .plans.md or .logs.md files, compressing historical passes, marking [PLANNED]/[WIP]/[DONE], archiving terminally closed trackers in plans/completed, or adding a reusable Handoff query section for safe session continuation.'
argument-hint: 'Describe the tracker file, whether it is active or archival, the current workstream state, and what the next session should be able to continue safely.'
user-invocable: true
disable-model-invocation: false
---

# Tracker And Handoff Playbook

Use this skill when work needs durable continuity in markdown tracker files.

This skill is the canonical workflow for `.plans.md` and `.logs.md` structure in
NeatapticTS. It owns the tracker status markers, compression rules for old work,
the required `Handoff query` section that makes WIP sessions safe to resume
with a simple copy-paste prompt, and the terminal closure rule for finished
plans, including archival under `plans/completed/`.

Other skills may decide when a tracker should be updated, but they should defer
the tracker shape itself to this skill instead of redefining status markers,
handoff layout, or history-compression rules ad hoc.

## When To Use

- A `.plans.md` file needs to be created, rewritten, compressed, or updated.
- A `.logs.md` file needs a concise done-state record.
- A long-running workstream needs a safe continuation prompt for the next
  session.
- Older tracker history has become too verbose and needs compression.
- A workflow skill or agent needs a standard tracker format instead of a custom
  local convention.

## Canonical Tracker Rules

### File Roles

- Use `.plans.md` for active work, pending decisions, next steps, and the live
  `Handoff query`.
- Use `.logs.md` for compressed done-state records and completed work that no
  longer needs active session guidance.
- Keep active trackers in `plans/` and move terminally closed `.plans.md` plus
  `.logs.md` pairs into `plans/completed/`.

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
  `Remaining gaps`, `Next step`, or `Handoff query` sections.
- Do not preserve or emit a next-session handoff prompt on a fully closed plan
  unless the user explicitly asks for reopen guidance.
- If there is still a real next step from the plan's own context, the plan is
  not closed yet and should stay active with a `Handoff query`.

### Flow-Aware Tracker Closure

Tracker closure now runs through the `07.tracker-closure` flow. The flow
requires the `log-completion-marker` gate to pass before the workstream is
considered closed. The gate confirms that a compressed log entry is present and
the target phase is marked `[DONE]`.

When closing a tracker that was run with flow-aware phase steps:

- The `VALIDATION_EVIDENCE` section of the final structured-v1 output block
  must include `log-completion-marker gate: pass` evidence.
- If the workstream produced gate exceptions, confirm they are recorded in
  `.github/ai-learning/learning-log.jsonl` before compressing.

Every durable tracker should use the same three states:

- `[PLANNED]` for queued work not yet in motion.
- `[WIP]` for the single active workstream or active section.
- `[DONE]` for completed durable coverage.

Apply them consistently:

- The document-level status line should use exactly one of these markers.
- Prefer exactly one active `[WIP]` section in a `.plans.md` file.
- Completed historical sections should become `[DONE]` coverage notes rather
  than verbose transcripts.
- Queued near-term work should be `[PLANNED]`.

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

When a plan reaches terminal `[DONE]` state:

- compress the plan before considering the workstream finished,
- add or refresh the same-boundary `.logs.md` audit record,
- keep a short closed tracker focused on scope, final state, audit summary,
  reopen conditions, and the audit-log pointer.

Good completed note:

- `[DONE] Runtime diagnostics moved to runtime chapter; Network now delegates DropConnect, dropout reset, and training-health readers.`

Bad completed note:

- a session-style transcript of every read, search, and validation command.

### Heading And Date Rules

- Use stable undated headings.
- Do not prepend calendar dates to tracker sections.
- Keep section titles reusable across sessions.

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
  explicitly wants parallel active lanes.
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

## Expected Final Output

A strong tracker update should report:

- which tracker file was updated,
- whether it is now `[PLANNED]`, `[WIP]`, or `[DONE]`,
- what historical content was compressed,
- what the new active frontier is,
- whether a `Handoff query` section was added, refreshed, or intentionally
  removed because the plan was terminally closed,
- whether a same-boundary `.logs.md` file was added or refreshed,
- whether the closed tracker pair now lives in `plans/completed/`.

```

```
