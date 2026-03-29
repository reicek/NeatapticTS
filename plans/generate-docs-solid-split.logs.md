# Generate Docs SOLID Split Log

**Status:** [DONE]

## Audit scope

- Objective: split the docs generator into a folder-owned subsystem without
  breaking script build or generated docs output.

## Durable milestones

### [DONE] Generator root split

- Broke the flat generator into stable ownership areas for constants, types,
  state, targets, symbols, ordering, and output behavior.

### [DONE] Direct-path migration

- Retargeted repo callers to the stable generated script path instead of using
  a compatibility shim or old flat entry expectations.

### [DONE] Regression repair and nested follow-through

- Completed the symbols and output chapter split.
- Fixed the discovered regressions in intro-file control and folder-index link
  paths before closing the workstream.

## Controls and evidence

- Validation used `npm run docs:build-scripts` and `npm run docs`.
- Docs output remained stable at closeout.

## Reopen triggers

- Generator features outgrow the current folder ownership.
- Output regressions appear in link rewriting or intro selection.
