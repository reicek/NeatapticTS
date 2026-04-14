# Architecture Solid Split Plan

**Status:** [DONE]

## Scope

- Folderize flat `src/architecture/` boundaries and reduce the monolithic root
  documentation surface.
- Keep public behavior stable while moving ownership into chapter-aligned
  folders and documentation-order controls.

## Final state

- The architecture root, network families, and ONNX subareas now follow the
  folder-owned split structure needed for future work.
- `docs.order.json` controls now steer generated README openings toward chapter
  maps and public owners instead of facade-heavy or types-only surfaces.
- No active backlog remains; this plan is now the reopen point for future
  facade-removal, ownership-audit, or oversized-boundary follow-up.

## Audit summary

- Split and follow-through work was validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.
- The final state is stable enough to stop being a Phase 0 blocker.

## Reopen conditions

- Future architecture refactors reintroduce flat or monolithic ownership.
- Public-import cleanup becomes necessary after additional architecture work.
- ONNX or network documentation surfaces grow beyond the current split.

## Audit log

- Durable completion notes now live in
  [architecture-solid-split.logs.md](architecture-solid-split.logs.md).
