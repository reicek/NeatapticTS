# Source No-Explicit-Any Cleanup

**Status:** [DONE]

## Scope

- Eliminate remaining `@typescript-eslint/no-explicit-any` debt in `src/`.
- Keep the cleanup aligned with the current folderized architecture instead of pre-split file lists.
- Treat unrelated ESLint failures as out of scope unless a touched file requires a local fix to validate safely.

## Current state

- The old root-level handoff note drifted after the architecture and NEAT refactors and is no longer a reliable file-by-file checklist.
- A current raw source scan now finds 0 `any` tokens across `src/**/*.ts`.
- The explicit-form follow-up scan for patterns such as `as any`, `Promise<any>`, `Array<any>`, `Map<any, ...>`, `: any`, and `{any[]}` also returns 0 matches across `src/**/*.ts`.
- `src/architecture/nodePool.ts`, previously marked as the next target in the legacy note, now lints clean and is no longer the correct starting point.
- A broad `npx eslint src` run now passes after the follow-up lint-hygiene cleanup removed stale unused imports and empty-interface wrappers that were outside the original explicit-any lane.

## Coverage backlog

### [DONE] Legacy pre-split cleanup pass

- The earlier cleanup covered `src/methods/`, `src/multithreading/`, and part of the pre-folderized architecture surface.
- That historical pass is kept only as a coarse extent note because the old path-by-path checklist no longer matches the current tree.

### [DONE] Root `src/neat.ts` explicit-any removal pass

- Removed the file-wide `@typescript-eslint/no-explicit-any` escape hatch from `src/neat.ts`.
- Replaced root-surface `any` usage with shared contracts for options, export/import payloads, RNG state, objective accessors, compatibility inputs, and helper `this` delegation.
- `npx eslint src/neat.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the root facade cleanup.

### [DONE] Root `src/neat.ts` polish follow-through

- The root strict-typing pass briefly exposed a small boundary-polish detour in the NEAT entry surface.
- That follow-through extracted the root compatibility alias shelf into `src/neat/neat.types.ts`, extracted the constructor-default shelf into `src/neat/neat.defaults.constants.ts`, exported constructor bootstrap contracts from `src/neat/init/neat.init.ts`, and centralized `buildEmptyDiversityStats()` in `src/neat/diversity/diversity.ts`.
- The result kept `src/neat.ts` orchestration-first without creating a second active roadmap lane.
- Documentation effects from that detour remain tracked in `plans/neat-docs.plans.md`; this tracker keeps only the compact structural note so the strict-typing lane stays self-contained.

### [DONE] `src/neat/evolve/objectives/evolve.objectives.utils.ts` cleanup pass

- Removed the file-wide `@typescript-eslint/no-explicit-any` escape hatch from `src/neat/evolve/objectives/evolve.objectives.utils.ts`.
- Replaced `any`-based objective, population, entropy-accessor, and cache-reset handling with the existing evolve contracts in `src/neat/evolve/evolve.types.ts`.
- Kept the runtime behavior narrow by centralizing the entropy accessor as a typed local helper instead of widening the shared controller contract.
- `npx eslint src/neat/evolve/objectives/evolve.objectives.utils.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the evolve-objectives cleanup.

### [DONE] `src/neat/evolve/population/evolve.population.utils.ts` cleanup pass

- Removed the file-wide `@typescript-eslint/no-explicit-any` escape hatch from `src/neat/evolve/population/evolve.population.utils.ts`.
- Replaced `any`-based species allocation, per-member score folding, remainder budgeting, and lineage bookkeeping with the existing evolve contracts in `src/neat/evolve/evolve.types.ts`.
- Kept the runtime behavior intact by preserving the existing lineage-id fallback semantics (`?? 0`) where parent ids remain optional on the shared genome contract.
- `npx eslint src/neat/evolve/population/evolve.population.utils.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the evolve-population cleanup.

### [DONE] `src/neat/evolve/adaptive/evolve.adaptive.utils.ts` cleanup pass

- Removed the file-wide `@typescript-eslint/no-explicit-any` escape hatch from `src/neat/evolve/adaptive/evolve.adaptive.utils.ts`.
- Replaced `any`-based option access, cache invalidation iteration, and re-enable counter resets with the existing evolve contracts in `src/neat/evolve/evolve.types.ts`.
- Added the narrow per-genome re-enable counters to `GenomeWithMetadata` so the adaptive bridge can reuse the shared evolve runtime metadata instead of reintroducing local casts.
- Normalized missing compatibility coefficients before auto-tuning updates so the helper no longer relies on `any` to hide optional-option gaps.
- `npx eslint src/neat/evolve/adaptive/evolve.adaptive.utils.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the evolve-adaptive cleanup.

### [DONE] `src/neat/multiobjective/category/multiobjective.category.ts` cleanup pass

- Removed the file-wide `@typescript-eslint/no-explicit-any` escape hatch from `src/neat/multiobjective/category/multiobjective.category.ts`.
- Replaced `any`-based Pareto snapshot shaping, objective-vector extraction, and objective-removal filtering with the existing evolve genome and objective descriptor contracts.
- Dropped the unnecessary `undefined as any` cache reset in favor of the native optional objective-list contract.
- `npx eslint src/neat/multiobjective/category/multiobjective.category.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the multi-objective category cleanup.

### [DONE] `src/neat/evolve/speciation/evolve.speciation.utils.ts` cleanup pass

- Removed the file-wide `@typescript-eslint/no-explicit-any` escape hatch from `src/neat/evolve/speciation/evolve.speciation.utils.ts`.
- Replaced `any`-based node filtering and species-history snapshot shaping with narrow local node typing plus the shared evolve species contracts.
- Aligned the minimal snapshot rows with `SpeciesHistoryRecord` by writing `bestScore` and `avgSharedFitness` instead of leaking ad hoc fields through `any`.
- `npx eslint src/neat/evolve/speciation/evolve.speciation.utils.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the evolve-speciation cleanup.

### [DONE] `src/neat/evolve/runtime/evolve.runtime.utils.ts` cleanup pass

- Removed the file-wide `@typescript-eslint/no-explicit-any` escape hatch from `src/neat/evolve/runtime/evolve.runtime.utils.ts`.
- Replaced `any`-based timer access with one typed high-resolution timer resolver shared by both start-time and elapsed-time helpers.
- Replaced the score-reset callback cast with direct `GenomeWithMetadata` inference across the live population.
- `npx eslint src/neat/evolve/runtime/evolve.runtime.utils.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the evolve-runtime cleanup.

### [DONE] `src/neat/adaptive/adaptive.ts` contract-doc cleanup pass

- Replaced the repeated JSDoc `@this {{ ... any ... }}` object literals in `src/neat/adaptive/adaptive.ts` with the existing `NeatLikeWithAdaptiveType` host contract.
- Kept the change documentation-only: behavior and implementation types were already routed through the adaptive core contracts.
- `npx eslint src/neat/adaptive/adaptive.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the adaptive-root cleanup.

### [DONE] `src/multithreading/multi.ts` worker-doc cleanup pass

- Replaced stale `Promise<any>` JSDoc return annotations on the worker helpers with `Promise<TestWorkerConstructor>` to match the existing TypeScript signatures.
- `npx eslint src/multithreading/multi.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the multithreading cleanup.

### [DONE] Root `src/neat.ts`, `src/neat/evolve/evolve.ts`, and network-chapter doc cleanup pass

- Removed stale any-focused commentary from `src/neat/evolve/evolve.ts` now that the helper split carries the narrow runtime contracts.
- Reworded the `_structuralEntropy` compatibility note in `src/neat.ts` so it no longer references loose `(controller as any)` test examples.
- Updated network chapter examples and JSDoc in `src/architecture/network/network.ts` and `src/architecture/network/slab/network.slab.utils.ts` to use the typed public slab accessors instead of loose-cast examples.
- `npx tsc --noEmit -p tsconfig.json` passes after the root/network documentation cleanup.

### [DONE] `src/architecture/network/stats/network.stats.utils.ts` doc cleanup pass

- Removed the stale note describing the `_lastStats` bridge as "typed as any" and replaced it with wording that points at the existing `NetworkStatsProps` bridge.
- `npx eslint src/architecture/network/stats/network.stats.utils.ts --rule "@typescript-eslint/no-explicit-any:error"` now passes.
- `npx tsc --noEmit -p tsconfig.json` passes after the network-stats cleanup.

### [DONE] Repo-wide prose-only cleanup pass

- Rewrote the remaining ordinary-English `any` phrases across the architecture, NEAT, mutation, telemetry, selection, objectives, and multithreading chapters so the raw `\bany\b` source scan is now clean.
- Kept this pass documentation-only: comments, examples, and JSDoc were tightened without changing runtime behavior or widening contracts.
- `grep_search` for raw `\bany\b` across `src/**/*.ts` now returns 0 matches.
- `grep_search` for explicit forms such as `as any`, `Promise<any>`, `Array<any>`, `Map<any>`, `: any`, and `{any[]}` across `src/**/*.ts` now returns 0 matches.
- `npx tsc --noEmit -p tsconfig.json` passes after the final prose-only sweep.

### [DONE] Broad `src` lint-hygiene follow-up

- Removed stale unused imports from `src/architecture/network/network.ts`, `src/architecture/network/onnx/export/network.onnx.export.types.ts`, and `src/architecture/network/onnx/network.onnx.utils.types.ts`.
- Replaced the empty facade interfaces in `src/neat/pruning/facade/pruning.facade.ts` and `src/neat/rng/facade/rng.facade.ts` with equivalent type aliases to satisfy `@typescript-eslint/no-empty-object-type`.
- `npx eslint src` now passes after the follow-up cleanup.
- `npx tsc --noEmit -p tsconfig.json` still passes after the lint-hygiene follow-up.

### [DONE] Current hotspots in the refactored tree

- No remaining hotspots in `src/**/*.ts` for raw `any` tokens.
- No remaining hotspots in `src/**/*.ts` for explicit-form `any` usage.
- No remaining broad ESLint errors in `src/` after the unused-import and empty-interface follow-up.

### [PLANNED] Validation discipline

- Validate touched files with targeted ESLint runs instead of treating the whole `src/` tree as a single pass/fail gate.
- Run `npx tsc --noEmit -p tsconfig.json` after each durable batch.
- Avoid broad test runs for this workstream unless a specific edit changes behavior rather than types.

## Immediate next steps

1. Treat this lane as complete unless future refactors reintroduce explicit-any debt into `src/`.
2. If later work reopens the lane, resume from current repo state and start with a fresh raw scan instead of replaying this completed tracker.

## Deferred questions

- Decide later whether some remaining `any` usage is genuinely architectural and should be documented with narrowly scoped eslint suppression instead of erased mechanically.
- If the broader ES2023 modernization pass introduces lint enforcement changes, keep this tracker aligned with the final lint gate instead of duplicating policy here.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Current repo state:
- The root `src/neat.ts` pass is complete.
- The root surface-polish follow-through from that pass has been compressed into this tracker; there is no separate active plan for it.
- `src/neat/evolve/objectives/evolve.objectives.utils.ts` is now clean for `@typescript-eslint/no-explicit-any`.
- `src/neat/evolve/population/evolve.population.utils.ts` is now clean for `@typescript-eslint/no-explicit-any`.
- `src/neat/evolve/adaptive/evolve.adaptive.utils.ts` is now clean for `@typescript-eslint/no-explicit-any`.
- `src/neat/multiobjective/category/multiobjective.category.ts` is now clean for `@typescript-eslint/no-explicit-any`.
- `src/neat/evolve/speciation/evolve.speciation.utils.ts` is now clean for `@typescript-eslint/no-explicit-any`.
- `src/neat/evolve/runtime/evolve.runtime.utils.ts` is now clean for `@typescript-eslint/no-explicit-any`.
- `src/neat/adaptive/adaptive.ts` is now clean for explicit-form `any` usage.
- `src/multithreading/multi.ts` is now clean for explicit-form `any` usage.
- `src/architecture/network/stats/network.stats.utils.ts` is now clean for explicit-form `any` usage.
- The raw `any` baseline is down to 0.
- The explicit-form scan for `as any`, `Promise<any>`, `Array<any>`, `Map<any>`, `: any`, and `{any[]}` is also at 0.
- `npx eslint src` passes.

Next target:
- No active target in this lane. Reopen only if future edits reintroduce explicit-any debt or new broad lint drift into `src/`.
- If reopened, begin with a fresh raw scan, `npx eslint src`, and `npx tsc --noEmit -p tsconfig.json` before selecting a new target.
```