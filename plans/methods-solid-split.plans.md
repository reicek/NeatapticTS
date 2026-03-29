# methods SOLID Split

**Status:** [DONE]

## Scope

- Replace the flat `src/methods/` layout with folder-owned chapter boundaries
  for the main method families.
- Keep the generated methods root readable and ownership-oriented after the
  split.

## Final state

- The major method families are now folderized and imported through stable
  chapter boundaries instead of the previous flat-file cluster.
- Root and local `docs.order.json` controls keep generated README openings tied
  to the correct public chapter maps.
- No active backlog remains; this plan is now the reopen point for later method
  family expansion or new oversized-boundary work.

## Audit summary

- Structural and documentation follow-through were validated with
  `npm run docs` and `npx tsc --noEmit -p tsconfig.json`.
- The methods root no longer acts as a monolithic blocker for modernization.

## Reopen conditions

- Another method family grows beyond the current folder structure.
- Import paths or generated chapter ownership drift after later refactors.
- The methods root README needs another chapter-map correction.

## Audit log

- Durable completion notes now live in
  [methods-solid-split.logs.md](methods-solid-split.logs.md).
