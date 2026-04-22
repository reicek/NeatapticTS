# Flappy Recurrent Architecture Debug Pass Log

**Status:** [DONE]

## Audit scope

- Objective: diagnose the recurrent browser failures reported for GRU, LSTM,
  and NARX after the earlier Flappy Step 7 integration work, repair true
  runtime corruption, and leave only durable browser polish boundaries behind.

## Durable milestones

### [DONE] Runtime-init corruption root cause

- Confirmed the GRU and LSTM duplicate-innovation failure was caused by
  `ensureNoDeadEnds()` treating valid recurrent-module hidden nodes as generic
  dead ends during generation-zero repair.
- Added module-aware protection so hidden nodes owned by validated temporal
  descriptors are skipped by the generic repair path instead of being rewired
  with low-id innovations.

### [DONE] GRU and LSTM recurrent repair validation

- Verified the repaired recurrent seeds stay strict-valid at worker startup and
  through generation-zero warm-start and playback.
- Green validation covered the repair test surface, worker runtime coverage,
  the Flappy worker build, and the live browser probe used during the
  investigation.

### [DONE] NARX browser plateau classification and acceptance

- Treated NARX as a browser-polish question rather than a structural runtime
  bug once the population remained valid through startup, warm-start, playback,
  and winner-clone handoff.
- Reused the shared architecture progress probe plus focused test, build, and
  lint validation to confirm the accepted NARX browser envelope instead of
  keeping NARX marked as an open regression.

### [DONE] Recurrent debug instrumentation closure

- Removed the temporary recurrent debug service, its worker call sites, and the
  dead debug flag after GRU and NARX empirical reruns were green.
- Regenerated the Flappy docs so the generated README surfaces no longer expose
  the deleted debug shelf.

## Controls and evidence

- Structural repair validation used
  `src/neat/mutation/repair/mutation.repair.test.ts`,
  `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.runtime.service.test.ts`,
  `npm run build:flappy-worker`, and the headless browser probe against
  `examples/flappy_bird/index.html`.
- Recurrent browser acceptance reused `npm run flappy:architecture:progress`
  for GRU and NARX observational and strict reruns, plus focused Flappy worker
  Jest slices, `npm run build`, file-scoped ESLint, and `npm run docs`.
- The final cleanup pass validated the Flappy worker evolution, playback,
  runtime, and warm-start suites before the final build, lint, and docs pass.

## Reopen triggers

- Duplicate innovations reappear for a recurrent preset at worker startup or
  after generation-zero repair.
- The shared architecture progress probe regresses for GRU or NARX under the
  current browser defaults.
- A future recurrent pass needs temporary worker-side instrumentation to
  localize a new failure mode.