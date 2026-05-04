# Generate Docs SOLID Split

**Status:** [DONE]

## Scope

- Split the docs generator into a stable folder-owned module boundary while
  keeping `npm run docs` behavior intact.
- Leave the generator ready for reopen-only follow-up instead of an active
  refactor frontier.

## Final state

- The docs generator now has a folder-owned structure with clear ownership for
  constants, types, state, targets, symbols, ordering, and output concerns.
- Repo callers were moved to the stable generated-script path instead of
  relying on the pre-split flat layout.
- No active backlog remains; this file is now a reopen point for later
  generator-boundary work.

## Audit summary

- Validation used `npm run docs:build-scripts` and `npm run docs`.
- Behavioral regressions discovered during the split were fixed before closure.

## Reopen conditions

- New generator features make the current folder ownership too coarse.
- Docs-output behavior regresses in intro-file control or generated links.
- Another nested split becomes necessary in the generator area.

## Audit log

- Durable completion notes now live in
  [generate-docs-solid-split.logs.md](generate-docs-solid-split.logs.md).
