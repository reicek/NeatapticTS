# Flappy Startup Loading Preview Log

**Status:** [DONE]

## Audit scope

- Objective: remove the empty-feeling first-load canvas state in
  `examples/flappy_bird` without changing later generation playback behavior.

## Durable milestones

### [DONE] Startup preview runtime lane

- Added explicit startup preview timing and legend constants plus a pure visual
  state utility for opacity, frame clock, scroll progression, and responsive
  legend sizing.
- Added a browser runtime startup preview service that immediately paints the
  canvas, animates the existing background layers, draws the centered loading
  legend, and supports controlled completion or early stop.
- Extended the same service boundary so it can also run fixed-duration
  generation title cards with a fade-in, hold, and fade-out lifecycle.

### [DONE] First-generation playback gating

- Wired the first `generation-ready` wait path through the startup preview so
  the initial playback episode does not begin until the preview fade-out has
  finished.

### [DONE] Per-generation presentation before playback

- Added a reusable `GENERATION N` presentation step that runs when each ready
  generation payload arrives, including generation 1.
- Kept the broader between-generation waiting state unchanged outside the
  short presentation window that now appears immediately before playback.

### [DONE] Validation and docs synchronization

- Added targeted Jest coverage for the preview visual-state helper.
- Ran repo typecheck, rebuilt the Flappy browser bundle, and refreshed the
  targeted Flappy folder docs so the new runtime files were reflected in the
  generated documentation surface.

## Controls and evidence

- Validation used `npx jest runtime.startup-preview.utils.test.ts --runInBand`,
  `npx tsc --noEmit -p tsconfig.json`, `npm run build:flappy-bird`, and
  `npm run docs:folders:flappy-bird`.
- The workstream stayed browser-side and did not require worker protocol
  changes.

## Reopen triggers

- The loading preview needs to appear outside the first startup episode.
- The preview legend or background behavior drifts from the established Flappy
  neon visual language.
- Runtime refactors move or replace the current first-generation wait seam.
