# Neatenstein Browser Entry

This folder contains the browser-facing half of the **Neatenstein** demo:
host-side orchestration, input routing, audio, and the deterministic game
simulation that feeds the worker-side renderer.

## Module layout

| Path                                   | Responsibility                                                                                             |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [`browser-entry.ts`](browser-entry.ts) | Demo entry point: wire the host loop, canvas, input, audio, and renderer bridge.                           |
| [`audio.ts`](audio.ts)                 | WebAudio audio engine stub and sound-effect trigger helpers.                                               |
| [`constants.ts`](constants.ts)         | Shared renderer/audio/frame constants.                                                                     |
| [`host/game/`](host/game/)             | Host-side game simulation: deterministic reset, player state, combat, movement, and episode lifecycle.     |
| [`host/waves.ts`](host/waves.ts)       | Wave transition: clear the arena, evolve the MLP enemy population one generation, and spawn the next wave. |
| [`host/`](host/)                       | Host shell, input routing, audio, and the renderer bridge.                                                 |
| [`harness/`](harness/)                 | NGE enemy-population harness: MLP topology, fitness, selection, barrier, and evolution runner.             |
| [`renderer/`](renderer/)               | Grid DDA raycaster, frame protocol, and the center-screen plasma-cannon overlay renderer.                  |
| [`worker/`](worker/)                   | Display worker that owns simulation and rendering on offload tiers.                                        |

## Deterministic reset

The game state is initialized through
[`host/game/state.ts`](host/game/state.ts). Calling `createGameState({ seed })`
with the same seed always returns the same canonical snapshot. The seed is
stored on the returned state as a plain value, and replay reconstructs the
seeded PRNG from it via `createGameRng(state.seed)`. This keeps the state safe
to clone or transfer while keeping episode replay and evolution benchmarking
reproducible.

## Capabilities

- The renderer and audio modules run in the browser entry and worker tiers.
- The game simulation, wave transitions, and renderer bridge live under
  [`host/`](host/) and feed the worker through a versioned frame protocol.
