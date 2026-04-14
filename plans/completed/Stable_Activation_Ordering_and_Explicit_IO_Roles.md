# Stable Activation Ordering + Explicit I/O Roles Plan

**Status:** [DONE]

## Scope

This workstream covers the remaining Phase 1 critical-path activation-semantics item from [../Roadmap.md](../Roadmap.md). Its job was to make network execution deterministic and explainable through explicit ordered I/O role metadata, deterministic acyclic and recurrent scheduling, schedule-aware activation traversal, and user-facing scheduling diagnostics.

Not in scope for this closed lane:

- speculative performance work beyond cached scheduling
- reopening the Proper NEAT tracker
- pulling Phase 2 builder work forward before Phase 1 closure

## Final state

- `Network` now owns explicit ordered `inputNodeIds` and `outputNodeIds`, refreshed by bootstrap, builders, restore paths, and evolutionary materialization seams.
- Acyclic mode now compiles deterministic Kahn-wave schedules with stable node `geneId` tie-breaks while preserving the legacy flattened `_topoOrder` cache for compatibility.
- Recurrent mode now compiles deterministic SCC-condensation schedules with explicit `recurrent-component` steps, default `iterations: 1`, and `stateSemantics: 'carry'`.
- `activate()` and `noTraceActivate()` now resolve traversal from the compiled schedule first, then fall back only when required, while input injection and output readout follow the explicit role ids rather than raw node storage order.
- The runtime now exposes `getActivationSchedulingDiagnostics()` so callers can inspect compiled-schedule use, cycle fallback, stale-topology state, recurrent carry semantics, and suggested next actions.
- Source-first docs and generated README surfaces now explain activation ordering, explicit I/O role semantics, schedule-aware execution, and carried recurrent state.

## Audit summary

- Durable milestone history for Steps 1 through 5 now lives in [Stable_Activation_Ordering_and_Explicit_IO_Roles.logs.md](Stable_Activation_Ordering_and_Explicit_IO_Roles.logs.md).
- Focused regression coverage passed for the touched runtime boundaries: `src/architecture/network/runtime/network.runtime.scheduling-diagnostics.test.ts`, `src/architecture/network/activate/network.activate.test.ts`, and `src/architecture/network/topology/network.topology.test.ts` (3 suites / 44 tests).
- Final closure validation passed with `npm run build` and `npm run docs`.
- This closes the remaining Phase 1 activation-ordering lane, so Phase 2 builder work can assume explicit I/O roles and deterministic acyclic or recurrent scheduling as the baseline runtime contract.

## Reopen conditions

- Reopen this tracker only if future builder, serialization, or runtime work exposes missing explicit I/O role refresh or schedule invalidation seams.
- Reopen if deterministic activation ordering drifts for acyclic or recurrent graphs after structural edits or materialization.
- Reopen if public diagnostics or docs stop matching the actual compiled-schedule, fallback, or carried-state contracts.

## Audit log

See [Stable_Activation_Ordering_and_Explicit_IO_Roles.logs.md](Stable_Activation_Ordering_and_Explicit_IO_Roles.logs.md) for the durable milestone record and validation summary.
