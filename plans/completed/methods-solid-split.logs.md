# methods SOLID Split Log

**Status:** [DONE]

## Audit scope

- Objective: move `src/methods/` from a flat collection of families into a
  folder-owned module structure with durable README chapter ownership.

## Durable milestones

### [DONE] Family folderization

- Folderized the activation, cost, rate, selection, mutation, crossover,
  gating, and connection families into stable chapter boundaries.
- Removed the need to keep the mutation family as a flat special case.

### [DONE] Import-path migration

- Retargeted repo-local imports to the new folder structure instead of relying
  on transitional flat-file ownership.
- Preserved public behavior while simplifying future maintenance in the methods
  area.

### [DONE] Documentation ownership follow-through

- Added root and local `docs.order.json` controls so the generated methods root
  reads as a chapter map rather than a flat symbol shelf.
- Closed the documentation follow-through exposed by the structural split.

## Controls and evidence

- Validation after split work used `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.
- The final methods structure is now the baseline for future reopen-only work.

## Reopen triggers

- A method family becomes oversized again.
- Import-path drift or README ownership regressions appear after later work.
- Additional family-level splits become necessary.
