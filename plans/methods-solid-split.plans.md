# methods SOLID Split

**Status:** [DONE]

## Purpose

Reduce the overloaded flat `src/methods/` surface into small chapter folders so
the generated methods README stops behaving like one monolithic page. Keep
`src/methods/methods.ts` as the orchestration-first public facade while moving
each concept into its own folder-based boundary.

## Root

- Split root: `src/methods`
- Nearest README reviewed: `src/methods/README.md`
- Parent README reviewed: `src/README.md`
- Relevant plan: `n/a`

## Durable Rules

- Keep exactly one active step at a time.
- Update this plan immediately after each completed step.
- Use direct-path migration for repo-local imports; do not leave flat shim files.
- Do not hand-edit generated README files; improve source JSDoc and run docs.
- Keep `src/methods/methods.ts` as the readable public facade.

## Target Shape

- `src/methods/methods.ts` remains the public facade.
- `src/methods/activation/` owns activation registry and utility math helpers.
- `src/methods/cost/` owns cost facade and cost utility helpers.
- `src/methods/rate/` owns schedule facade and rate utility helpers.
- `src/methods/selection/`, `src/methods/mutation/`, `src/methods/crossover/`,
  `src/methods/gating/`, and `src/methods/connection/` each become small chapter folders.
- Root docs ordering should describe chapter folders instead of a flat file list.

## Steps

- [x] Step 1: Folderize the methods categories, migrate repo-local imports to the new direct paths, and keep the root facade stable.
- [x] Step 2: Refresh generated docs ordering and educational-docs policy so oversized folder READMEs are split proactively.
- [x] Step 3: Validate the split surface and record the handoff state.

## Latest Pass

- Folderized the methods surface into chapter folders for `activation`, `cost`,
    `rate`, `selection`, `mutation`, `crossover`, `gating`, and `connection`.
- Migrated repo-local imports to direct folder paths and removed the stale flat
    `src/methods/mutation.ts` file that was still being compiled after the move.
- Updated `src/methods/methods.ts` to remain the root public facade and updated
    root ordering via `src/methods/docs.order.json`.
- Added chapter-level ordering configs for `activation`, `cost`, and `rate` so
    their nested READMEs open from the public facade files rather than utility
    helpers.
- Updated `.github/skills/educational-docs/SKILL.md` so oversized generated
    folder READMEs now point toward `solid-split` instead of continuing to grow
    monolithically.
- Regenerated docs and verified the generated openings for
    `src/methods/README.md`, `src/methods/activation/README.md`,
    `src/methods/cost/README.md`, and `src/methods/rate/README.md`.
- Final validation completed with `npm run docs` and
    `npx tsc --noEmit -p tsconfig.json`.

## Handoff State

- The split is complete and the root methods README now reads as a chapter map
    rather than a monolithic flat file dump.
- Nested method READMEs are generated from source-first JSDoc and should be
    maintained by editing the owning source files, not the generated READMEs.
- If a future methods family starts to feel oversized again, prefer another
    bounded folder split over expanding the root README.

## Done Criteria

- The flat `src/methods/*.ts` category files have been replaced by chapter folders.
- Repo-local imports point at the new direct folder paths.
- `src/methods/README.md` becomes smaller because chapter content moves into nested folder READMEs.
- The educational-docs skill explicitly points large monolithic README surfaces toward `solid-split`.
- The boundary can be resumed safely from this plan alone.