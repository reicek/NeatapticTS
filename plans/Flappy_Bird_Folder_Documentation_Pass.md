# Flappy Bird Folder Documentation Pass

## Purpose

Run a folder-focused documentation pass across every README-owning folder under
`test/examples/flappy_bird` so the generated documentation reads like a guided
tour of the example architecture rather than a sparse export inventory.

## Root

- Split root: `test/examples/flappy_bird`
- README inventory: 24 README-owning folders under the root
- Nearest README reviewed: `test/examples/flappy_bird/README.md`
- Parent README reviewed: `test/examples/README.md` if needed per step
- Relevant plan: `plans/Interactive_Examples_and_Learning_Path.md`

## Durable Rules

- Keep the README inventory explicit and visible in the todo list.
- Keep exactly one active README or README-owning folder at a time.
- Do not hand-edit generated subfolder README files; improve source JSDoc and
  regenerate docs.
- Keep compatibility facades documented as facades, not as implementation sinks.
- Prefer educational JSDoc that explains what, why, invariants, defaults, and
  performance tradeoffs when those details help a learner understand the example.

## Target Shape

- Each README-owning folder has source JSDoc rich enough to generate a useful
  README without manual rescue edits.
- Compatibility facades explain where deeper responsibilities live.
- Browser, evaluation, simulation, worker, and trainer boundaries read like a
  coherent architecture map when browsed top-down.

## Steps

- [x] Step 1: Tidy the root `test/examples/flappy_bird` surface and verify the
  README inventory-driven workflow.
- [x] Step 2: Tidy the worker and trainer-facing folders.
- [ ] Step 3: Tidy shared simulation, environment, constants, and evaluation
  folders.
- [ ] Step 4: Tidy browser-entry and its nested README-owning folders one by one.
- [ ] Step 5: Regenerate docs, run focused validation, and record any durable
  boundary notes that changed during the pass.

## Done Criteria

- Every README-owning folder in the inventory has been handled explicitly.
- Generated README content is educational and boundary-aware.
- The todo list reflects completion folder by folder.
- Docs regeneration succeeds after documentation-affecting changes.
- A later session can resume from this plan and the todo list without prior chat
  history.