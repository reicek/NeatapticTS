# Neatenstein Browser Entry

This folder contains the browser-facing half of the **Neatenstein** demo:
host-side orchestration, input routing, audio, a deterministic maze generator,
and the game simulation that feeds the worker-side renderer. The
[`harness/`](harness/) directory also owns the NGE enemy-population harness,
which performs per-death evolution: each kill or player death updates a
per-variant fitness ledger, proportional selection produces the next parent
generation, and an optional MAP-Elites archive preserves behavioral niches.

## Module layout

| Path                                   | Responsibility                                                                                             |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [`browser-entry.ts`](browser-entry.ts) | Demo entry point: wire the host loop, canvas, input, audio, and renderer bridge.                           |
| [`audio.ts`](audio.ts)                 | WebAudio audio engine stub and sound-effect trigger helpers.                                               |
| [`constants.ts`](constants.ts)         | Shared renderer/audio/frame constants.                                                                     |
| [`host/game/`](host/game/)             | Host-side game simulation: deterministic reset, player state, combat, movement, and episode lifecycle.     |
| [`host/waves.ts`](host/waves.ts)       | Wave transition: clear the arena, evolve the MLP enemy population one generation, and spawn the next wave. |
| [`host/`](host/)                       | Host shell, input routing, audio, and the renderer bridge.                                                 |
| [`harness/`](harness/)                 | NGE enemy-population harness: MLP topology, fitness, proportional per-death selection, MAP-Elites archive, barrier, and evolution runner. |
| [`renderer/`](renderer/)               | Grid DDA raycaster, CPU framebuffer with smoothstep distance fog, frame protocol, and the center-screen plasma-cannon overlay renderer.       |
| [`worker/`](worker/)                   | Display worker that owns simulation and rendering on offload tiers, with pooled typed-array buffers and a cached BFS distance map to keep the hot path allocation-free. B1 added a sim/render worker state split (`SimWorkerState` / `RenderWorkerState`), a shared map grid (`createSharedMapGrid`), and parallel enemy inference with a deterministic barrier (`runSimStepParallel`). |

## Deterministic reset

The game state is initialized through
[`host/game/state.ts`](host/game/state.ts). Calling `createGameState({ seed })`
with the same seed always returns the same canonical snapshot. The seed is
stored on the returned state as a plain value, and replay reconstructs the
seeded PRNG from it via `createGameRng(state.seed)`. This keeps the state safe
to clone or transfer while keeping episode replay and evolution benchmarking
reproducible.

## Quality boundaries

The browser-entry folder also owns the cross-cutting quality boundaries that
keep the demo maintainable as the raycaster, harness, and worker grow.

### Shared dependency floor

[`shared/`](shared/) is the dependency floor: `scripts/` and `browser-entry/`
both import downward into it, so guards and constants never leak upward. The
most important consolidation target is
[`shared/math-guards.utils.ts`](shared/math-guards.utils.ts), which supplies
`clamp`, `clamp01`, `isFiniteNumber`, and the dimension guards used across the
renderer and the harness.

On the worker side,
[`worker/display.worker.message-handler.ts`](worker/display.worker.message-handler.ts)
extracts the render compositing orchestrator from the display worker monolith.
[`worker/display.worker.test-hooks.ts`](worker/display.worker.test-hooks.ts)
holds the 22 `__testOnly*` helpers that let tests reach internal worker state
without widening the public API, and
[`worker/display.worker.test-helpers.ts`](worker/display.worker.test-helpers.ts)
replaces the old overloaded worker test file with focused utilities.

### Renderer module additions

[`renderer/`](renderer/) gained several quality-focused additions:

- `computeWallTexcoord` in [`renderer/walls.ts`](renderer/walls.ts) — wall-column
  texture coordinate derivation.
- `castNeatensteinFloorPerPixel` in [`renderer/floor.ts`](renderer/floor.ts) —
  unconditional per-pixel floor casting that draws a procedural integer grid at
  1-unit spacing via `fract(worldCoord)`.
- `resolveSideDistance` in [`renderer/raycast.ts`](renderer/raycast.ts) — NaN
  guard for grid-line-aligned rays.
- A unified `Infinity` z-buffer sentinel across walls, sprites, and effects.
- Shader scaffolds in [`renderer/shaders/`](renderer/shaders/) —
  `createNeatensteinCameraUniform`, `NEATENSTEIN_WALL_DDA_SHADER_SOURCE`, and
  `NEATENSTEIN_FLOOR_CASTER_SHADER_SOURCE` keep the GPU tier aligned with the
  CPU algorithms.

### Harness module additions

[`harness/`](harness/) now contains the algorithmic upgrade modules:

- [`harness/map-elites.ts`](harness/map-elites.ts) — MAP-Elites archive with
  10×10 behavioral grid.
- [`harness/cma-es.ts`](harness/cma-es.ts) — separable CMA-ES for 90-weight MLP
  optimization.
- [`harness/league.ts`](harness/league.ts) — unified league merging hall-of-fame
  and opponent-pool roles.
- [`harness/transition-replay.ts`](harness/transition-replay.ts) — per-enemy
  transition replay, shared replay, and death-surprise prioritization.
- [`harness/curriculum-difficulty.ts`](harness/curriculum-difficulty.ts) —
  player-telemetry difficulty scaling.
- [`harness/bounded-concurrency.ts`](harness/bounded-concurrency.ts) — bounded
  inference dispatch.

## Capabilities

- The renderer and audio modules run in the browser entry and worker tiers.
- The game simulation, wave transitions, and renderer bridge live under
  [`host/`](host/) and feed the worker through a versioned frame protocol.
