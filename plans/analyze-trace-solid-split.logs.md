# Analyze Trace SOLID Split Log

**Status:** [DONE]

## Audit scope

- Objective: split the trace analyzer into a folder-owned subsystem while
  preserving the trace-analysis command path and its reporting behavior.

## Durable milestones

### [DONE] Analyzer boundary split

- Established stable ownership for constants, types, shared helpers, I/O,
  analysis, and reporting under the analyzer folder.

### [DONE] Orchestration clarification

- Made the root flow clearer as resolve, load, analyze, and print so future
  trace extensions land on a stable orchestration surface.

### [DONE] Determinism follow-through

- Added deterministic sort-tie handling so equal-duration sections produce
  stable ordering across runs.

## Controls and evidence

- Validation used clean file diagnostics,
  `npx tsc --noEmit -p tsconfig.docs.json`, and `npm run trace:analyze`.

## Reopen triggers

- Future analyzer features require deeper subfolder work.
- Deterministic ordering or report output regresses.
