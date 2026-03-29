# Source No-Explicit-Any Cleanup Log

**Status:** [DONE]

## Audit scope

- Objective: eliminate the explicit-`any` debt in `src/` and close the related
  typed-hygiene follow-through left by prior refactors.

## Durable milestones

### [DONE] Root baseline closure

- Closed the root explicit-`any` baseline at zero and removed the remaining
  explicit-form variants such as `as any` and `Promise<any>` from the tracked
  lane.

### [DONE] Hotspot cleanup passes

- Cleaned the root `neat.ts` surface and the key evolve, adaptive,
  multiobjective, and multithreading hotspots with behavior-preserving type
  strengthening.

### [DONE] Prose and lint-hygiene follow-through

- Completed the prose-only `any` sweep and the later lint-hygiene cleanup such
  as unused imports and empty-to-type alias follow-through.
- Closed the lane on a repository-local `src` lint baseline rather than leaving
  scattered reopen notes.

## Controls and evidence

- Validation used targeted ESLint checks, `npx tsc --noEmit -p tsconfig.json`,
  and a final green `npx eslint src` pass.

## Reopen triggers

- Explicit-`any` debt returns in `src/` after future refactors.
- Typing or lint baselines regress in the previously cleaned hotspots.
