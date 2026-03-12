# Flappy Evolution Worker Documentation Pass

## Purpose

Deepen the source JSDoc for the `test/examples/flappy_bird/flappy-evolution-worker`
boundary so the generated folder README explains not only what each exported
surface is called, but why the worker exists, how the protocol fits browser
rendering, which pieces own simulation versus transport versus warm-start
behavior, and what invariants matter when extending the example.

## Root

- Split root: `test/examples/flappy_bird/flappy-evolution-worker`
- Nearest README reviewed: `test/examples/flappy_bird/flappy-evolution-worker/README.md`
- Parent README reviewed: `test/examples/flappy_bird/README.md`
- Relevant plan: `plans/Interactive_Examples_and_Learning_Path.md`

## Durable Rules

- Keep exactly one active step at a time.
- Update this plan immediately after each completed step.
- Do not hand-edit generated README files; improve source JSDoc and run docs.
- Keep `flappy-evolution-worker.ts` orchestration-first and document helper
  ownership clearly.
- Treat the demo worker as a learning surface: documentation should teach the
  message flow, deterministic simulation model, and packed-snapshot transport.

## Target Shape

- `test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.ts`
  remains the orchestration-first worker entrypoint.
- Supporting files explain their responsibility in the generated README:
  runtime setup, protocol routing, evolution, playback, simulation,
  snapshots, warm-start, and transport contracts.
- Public and README-visible symbols include conceptual JSDoc, brief examples
  where behavior is non-obvious, and notes about defaults, invariants, or
  performance tradeoffs when that context improves learning value.

## Steps

- [x] Step 1: Review the nearest README files, current worker surface, and the
  relevant examples plan.
- [x] Step 2: Enrich foundational worker service docs (`constants`, `errors`,
  `runtime`, `protocol`, `evolution`, `playback`).
- [x] Step 3: Enrich simulation and transport docs (`simulation.*`,
  `snapshot.*`, `types`, `warm-start`, entrypoint helpers, and narrow tests).
- [x] Step 4: Run docs regeneration and focused validation, then record the
  finished boundary note.

## Done Criteria

- The generated worker README reads like a guided tour of the subsystem rather
  than a sparse symbol index.
- Each worker file has enough source JSDoc to explain its role in the folder.
- Non-obvious contracts mention the important defaults, invariants, or
  performance semantics.
- Docs regeneration completes after the source comments are updated.
- The next session can resume from this plan alone if another pass is needed.