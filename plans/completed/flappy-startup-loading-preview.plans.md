# Flappy Startup Loading Preview

**Status:** [DONE]

## Scope

- Add a browser-only startup preview for `examples/flappy_bird` so the main
  playback canvas does not appear empty while the first generation is still
  initializing.
- Animate the existing background layers during that startup window, but keep
  pipes and birds hidden until the first generation is ready.
- Show a centered canvas legend that reads `PREPARING NEURAL NETWORKS...`
  using the same neon-green color language and glow treatment as the pipes.
- Fade the legend in over `300 ms` at startup, fade it out over `300 ms` after
  the first generation is ready, and only then begin normal bird playback.
- Present every ready generation on the main canvas with the same neon title
  style using a `300 ms` fade-in, `300 ms` hold, and `300 ms` fade-out before
  each playback episode begins.

## Final state

- The Flappy browser runtime now renders a dedicated first-load startup preview
  before the first playback episode begins.
- The preview reuses the existing animated background renderer, keeps pipes and
  birds hidden, and draws a centered loading legend that matches the pipe neon
  palette and glow treatment.
- The first generation wait path now holds playback behind the original
  preparing screen and then the same canvas style is reused to present
  `GENERATION N` before every playback episode, including generation 1.
- Scope remained intentionally narrow to the initial startup window; later
  between-generation evolution waiting behavior was left unchanged outside the
  moment when each ready generation is presented.

## Audit summary

- Added shared preview timing constants, extended the preview service so it can
  drive both the first-load preparing screen and timed generation title cards,
  and wired the runtime loop so every ready generation is presented before
  playback starts.
- Validation used `npx jest runtime.startup-preview.utils.test.ts --runInBand`,
  `npx tsc --noEmit -p tsconfig.json`, `npm run build:flappy-bird`, and
  `npm run docs:folders:flappy-bird`.
- The targeted Flappy docs refresh completed after the runtime additions so the
  browser-entry documentation surface stayed synchronized with the new files.

## Reopen conditions

- The startup preview should also cover later between-generation `evolving`
  windows.
- The preview styling needs a different text treatment, timing curve, or a
  non-canvas overlay approach.
- Runtime startup flow changes invalidate the first-generation gating seam used
  by the current preview service.

## Audit log

- Durable completion notes now live in
  [flappy-startup-loading-preview.logs.md](flappy-startup-loading-preview.logs.md).
