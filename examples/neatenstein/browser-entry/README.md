# Neatenstein Browser Entry

This folder contains the browser-facing half of the **Neatenstein** demo:
host-side orchestration, input routing, audio, and the deterministic game
simulation that feeds the worker-side renderer.

## Module layout

| Path                           | Responsibility                                                                                                                |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| [`constants.ts`](constants.ts) | Shared renderer/audio/frame constants (Phase 1).                                                                              |
| [`host/game/`](host/game/)     | Host-side game simulation: deterministic reset, player state, enemy waves, combat, movement, and episode lifecycle (Phase 2). |
| [`host/`](host/)               | Host shell, input routing, and renderer bridge.                                                                               |
| [`renderer/`](renderer/)       | Grid DDA raycaster and frame protocol (Phase 1).                                                                              |
| [`worker/`](worker/)           | Display worker that owns simulation and rendering on offload tiers.                                                           |

## Deterministic reset

The game state is initialized through
[`host/game/state.ts`](host/game/state.ts). Calling `createGameState({ seed })`
with the same seed always returns the same canonical snapshot. The seed is
stored on the returned state as a plain value, and replay reconstructs the
seeded PRNG from it via `createGameRng(state.seed)`. This keeps the state safe
to clone or transfer while keeping episode replay and evolution benchmarking
reproducible.

## Status

- Phase 1 renderer and audio modules are implemented.
- Phase 2 game-logic modules are being added slice-by-slice under
  [`host/game/`](host/game/).
