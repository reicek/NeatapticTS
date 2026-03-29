# Source No-Explicit-Any Cleanup

**Status:** [DONE]

## Scope

- Remove `@typescript-eslint/no-explicit-any` debt across `src/` and close the
  remaining explicit-form `any` variants introduced by earlier refactors.
- Keep the cleanup behavior-preserving and validation-driven.

## Final state

- The tracked `src/` explicit-`any` debt was eliminated and the baseline closed
  at zero for the targeted forms in this lane.
- Root NEAT, evolve, adaptive, multiobjective, multithreading, and related
  hotspots were brought into stronger typed form without reopening unrelated
  behavior changes.
- No active backlog remains; this file is now a reopen point if explicit-`any`
  debt returns.

## Audit summary

- Validation used targeted ESLint runs and `npx tsc --noEmit -p tsconfig.json`
  after durable batches.
- The lane closed on a green `npx eslint src` final pass.

## Reopen conditions

- Future refactors reintroduce explicit `any` or equivalent typed debt in
  `src/`.
- Stronger typing regresses in the previously cleaned hotspot files.
- Lint or type baselines reopen because of source-surface drift.

## Audit log

- Durable completion notes now live in
  [src-no-explicit-any-cleanup.logs.md](src-no-explicit-any-cleanup.logs.md).
