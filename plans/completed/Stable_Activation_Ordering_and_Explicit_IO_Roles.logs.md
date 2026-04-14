# Stable Activation Ordering + Explicit I/O Roles Log

**Status:** [DONE]

## Audit scope

- Objective: make network execution deterministic and explainable through explicit ordered I/O role metadata, compiled activation schedules, schedule-aware traversal, and public scheduling diagnostics.
- Closed tracker: the final plan state and reopen conditions now live in [Stable_Activation_Ordering_and_Explicit_IO_Roles.md](Stable_Activation_Ordering_and_Explicit_IO_Roles.md).

## Durable milestones

### [DONE] Step 1 — Explicit I/O role plumbing

- Added ordered input and output role metadata to `Network` as stable node gene-id lists.
- Refreshed those roles through bootstrap, builders, restore paths, and evolutionary materialization seams so role metadata survives graph replacement.
- Added topology coverage proving constructed networks expose stable ordered role ids.

### [DONE] Step 2 — Deterministic acyclic scheduling

- Compiled deterministic Kahn-wave activation schedules with stable `geneId` tie-breaks.
- Preserved the legacy `_topoOrder` cache as a flattened compatibility view for existing acyclic consumers.
- Added topology coverage for deterministic wave grouping, stable ordering under node-array reordering, and cycle fallback behavior.

### [DONE] Step 3 — Deterministic recurrent scheduling contract

- Added SCC-condensation recurrent scheduling with explicit `recurrent-component` steps, default `iterations: 1`, and carried-state semantics.
- Kept the schedule contract distinct from ordinary activation-path adoption until the next step.
- Added topology coverage for recurrent schedule stability, singleton self-loop handling, and replacement of stale acyclic cache state.

### [DONE] Step 4 — Schedule-aware activation traversal

- Switched `activate()` and `noTraceActivate()` to resolve traversal from compiled schedules first and compatibility fallbacks second.
- Stabilized input injection and output readout through explicit role ids so raw node storage order no longer changes public vector semantics.
- Added activation regressions for reordered acyclic and recurrent node storage.

### [DONE] Step 5 — Diagnostics and docs closure

- Added `ActivationSchedulingDiagnostics` and `Network.getActivationSchedulingDiagnostics()` as the public scheduling reader.
- Recorded compiled-schedule, cycle-fallback, recurrent-carry, and stale-topology diagnostics during topology compilation and runtime reads.
- Synced source-first docs and generated README surfaces around activation ordering, carried recurrent state, and explicit role semantics.

## Controls and evidence

- Focused regression slice passed for `src/architecture/network/runtime/network.runtime.scheduling-diagnostics.test.ts`, `src/architecture/network/activate/network.activate.test.ts`, and `src/architecture/network/topology/network.topology.test.ts` (3 suites / 44 tests).
- Final closeout validation passed with `npm run build` and `npm run docs`.
- No full-suite rerun or lint pass was required for this tracker closure.

## Closure notes

- This closes the remaining Phase 1 activation-ordering lane in [../Roadmap.md](../Roadmap.md).
- Phase 2 builder work can now assume explicit I/O roles plus deterministic acyclic and recurrent scheduling as the baseline activation contract.
- Reopen only if future builder, serialization, or runtime changes expose missing role refresh, schedule invalidation, or diagnostics drift.