# Architecture Solid Split Log

**Status:** [DONE]

## Audit scope

- Objective: convert the flat `src/architecture/` area into stable folder-owned
  module boundaries and repair the generated chapter structure that depended on
  those boundaries.

## Durable milestones

### [DONE] Root and family split baseline

- Closed the split across the core architecture families, including the root
  architecture surface plus the main node, connection, group, architect, and
  network ownership boundaries.
- Eliminated the dependency on a facade-dominated root README opening.

### [DONE] Network and ONNX subchapter structure

- Split deeper network and ONNX concerns into stable subchapters so future work
  can extend them without returning to a flat-file architecture root.
- Preserved public behavior while moving implementation and doc ownership into
  the intended folders.

### [DONE] Documentation ownership follow-through

- Added and tuned `docs.order.json` controls so generated chapter openings come
  from the right public owner files and chapter maps.
- Closed the README-opening follow-through that the split exposed.

## Controls and evidence

- Validation after structural and doc-affecting edits used `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.
- The final architecture root is now a durable baseline rather than an active
  split frontier.

## Reopen triggers

- Future architecture changes reintroduce facade-heavy or oversized roots.
- Public-import cleanup is intentionally resumed.
- ONNX or network chapters need another structural split.
