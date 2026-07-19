# Acceleration UI + Parallelism

**Status:** [DONE]

## Scope

Finish the generic acceleration layer's weight-variant evaluation path and expose acceleration-mode telemetry in the racing-curriculum demo.

This workstream covers three concrete deliverables:

1. A new `parallelVariantCount` field on `AccelerationConfig` (default constant, resolved by `resolveAccelerationConfig`), using a name that does not collide with lifecycle variant counts or GPU batch sizing.
2. Concurrent dispatch in `src/acceleration/acceleration.variants.ts` so `evaluateWeightVariantsAsync` scores variants in parallel up to `parallelVariantCount`, preserves per-variant score ordering, and restores original connection weights exactly after each batch.
3. A color-coded acceleration status chip in the racing-curriculum demo (`examples/racing_curriculum/browser-entry/browser-entry.ts`) that shows "Acceleration: CPU/GPU/WORKER" in the stage-card metadata row and updates each frame without DOM churn.

Out of scope: rewriting WebGPU kernels, adding acceleration to `Network.train()`, changing racing physics or promotion rules, or polyfilling GPU/worker APIs.

## Final state

Phase 1 complete. parallelVariantCount config added, concurrent weight-variant evaluation implemented, racing-curriculum acceleration status chip shipped.

## Audit summary

- Files changed: src/acceleration/acceleration.{types,constants,config,variants}.ts, examples/racing_curriculum/controller/runtime.adaptation.ts, examples/racing_curriculum/browser-entry/browser-entry.ts, and their test files.
- Validation: 58/58 focused tests passed; src/acceleration/ touched files at 100% coverage; TypeScript and lint clean.
- Gates: phase-compression, log-completion-marker, stale-wip-plans, plan-sync, step-packet, agent-graph, agent-quality, tier-enforcement, routing-table-freshness, learning-event, cortex-index passed; code-coverage failed only on unrelated pre-existing GPU/nge-juvenile files (exception recorded).
- Research artifact: docs/research/racing-curriculum-acceleration-ui-and-parallelism.md updated with post-implementation notes.

## Reopen conditions

- New acceleration UI work, racing demo telemetry changes, or further parallel-dispatch policy evolution.

## Audit log

- Detailed evidence: [Acceleration_UI_Parallelism.logs.md](Acceleration_UI_Parallelism.logs.md)
