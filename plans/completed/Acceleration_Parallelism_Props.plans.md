# Acceleration Parallelism Props

**Status:** [DONE]

**Phase:** 1

**Workstream type:** focused source-level fix that follows [plans/completed/Acceleration_UI_Parallelism.plans.md](completed/Acceleration_UI_Parallelism.plans.md).

## Scope

Make acceleration parallelism configurable end-to-end in the racing-curriculum demo and its supporting library code.

1. Raise the generic acceleration default `parallelVariantCount` from `1` to `16` in `src/acceleration/acceleration.constants.ts`.
2. Expose NGE lifecycle stage variant counts as optional props/parameters to `evaluateNgeWeightVariants` in `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts` (no demo-specific identifiers in `src/`).
3. Forward an explicit `AccelerationConfig` from `evaluateNgeWeightVariants` into `evaluateWeightVariantsAsync` so `parallelVariantCount` is honored on the NGE variant path.
4. Thread `AccelerationConfig` through `RuntimeAdaptationEngineOptions` and the live racing adaptation engine so the resolved config actually reaches variant evaluation.
5. Set the racing demo default to `256x` and update the stage-card chip to display both the backend mode and the count (e.g., `CPU (256x)`, `GPU (256x)`, `WORKER (256x)`).
6. Remove the hardcoded `const backend = 'cpu'` in `src/acceleration/acceleration.variants.ts:130` and resolve the active backend through the existing acceleration policy/resolver so the evaluator defaults to GPU, falls back to WebWorker, and finally to CPU.
7. Preserve the boundary that `src/acceleration/` does **not** import `src/architecture/` and that all numeric acceleration defaults live in `src/acceleration/acceleration.constants.ts`.

## Final state

Phase 1 complete. All seven scope items delivered, 104/104 focused tests passed, touched `src/` files at 100% coverage, browser validation confirmed GPU `256g/256x` chip and console diagnostics.

## Audit summary

- Files changed: `src/acceleration/acceleration.{constants,config,variants,observer,orchestrator}.ts`, `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts`, `examples/racing_curriculum/controller/runtime.adaptation.ts`, `examples/racing_curriculum/browser-entry/browser-entry.ts`, and their test files.
- Validation: 104/104 focused tests passed; 100% coverage on touched `src/` files; TypeScript compile and lint clean; browser validation chip `GPU 256g/256x`, console log `[NeatapticTS Acceleration] Backend selected: gpu`, ~77 FPS over 30 s visible-foreground window.
- Gates: phase-compression, log-completion-marker, stale-wip-plans, plan-sync, step-packet passed.
- Research artifact: `docs/research/racing-demo-acceleration-cpu-root-cause.md` documents the sync/async backend-detection mismatch and the chip fix.

## Reopen conditions

- New acceleration parallelism configuration work, racing demo telemetry changes, or further backend resolution policy evolution.

## Audit log

- Detailed evidence: [Acceleration_Parallelism_Props.logs.md](Acceleration_Parallelism_Props.logs.md)
