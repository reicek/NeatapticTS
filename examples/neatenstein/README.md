# Neatenstein (NeatapticTS)

> **Thesis:** _Train your own killer — then survive it._

This example is a first-person neon raycasting demo built on top of NeatapticTS.
It pairs a human player against networks that evolve from the player's own
deaths, making the learning process visible, audible, and playable. The demo is
a teaching system, not a game product: the real subject is how neuroevolution,
reproducible simulation, and worker-backed rendering can be composed into one
browser runtime.

## What This Folder Is Trying To Teach

This example is organized around three reader questions:

1. How do you connect NGE lifecycle evolution to a real-time control task
   where the same agent both teaches and is attacked by what it taught?
2. How do you keep a 60fps renderer deterministic, tier-aware, and
   worker-offloadable without leaking rendering authority into the library core?
3. How do you reuse the repo's existing visualizer patterns (Flappy ground grid,
   worker frame snapshots, neon palette) instead of inventing a parallel demo stack?

## Choose Your Route

| If you want to...                                   | Start here                                                                                                                                                                                                                                                                                       |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Run the browser demo                                | Build with `node scripts/build-neatenstein.mjs`, then open `examples/neatenstein/index.html` in a browser.                                                                                                                                                                                       |
| Read the renderer constants and tier presets        | [browser-entry/constants.ts](browser-entry/constants.ts)                                                                                                                                                                                                                                         |
| Explore the coarse-grid recursive-backtracker maze  | [browser-entry/renderer/map.ts](browser-entry/renderer/map.ts)                                                                                                                                                                                                                                     |
| Explore the procedural enemy art pipeline           | [scripts/voxel-enemy.ts](scripts/voxel-enemy.ts), [scripts/snapshot-renderer.ts](scripts/snapshot-renderer.ts), [scripts/enemy-animator.ts](scripts/enemy-animator.ts), [scripts/generate-enemy-sprites.ts](scripts/generate-enemy-sprites.ts)                                                   |
| Wire enemies into the live renderer                 | [scripts/enemy-controller.ts](scripts/enemy-controller.ts), [scripts/enemy-sprite.ts](scripts/enemy-sprite.ts), [browser-entry/host/waves.ts](browser-entry/host/waves.ts), [browser-entry/host/renderer-bridge.ts](browser-entry/host/renderer-bridge.ts)                                       |
| Explore parallel enemy AI inference                 | [scripts/enemy-controller.parallel.utils.ts](scripts/enemy-controller.parallel.utils.ts) — SAB-backed weight pools, tiered inference strategy (SAB → channel → inline), deterministic barrier synchronization with `(simTick, enemyIndex)` tagging, and the sim/render worker split types in [browser-entry/worker/display.worker.types.ts](browser-entry/worker/display.worker.types.ts)                   |
| Understand the raycaster and frame protocol         | [browser-entry/renderer/](browser-entry/renderer/)                                                                                                                                                                                                                                               |
| Tune the plasma-cannon overlay and voxel projection | [browser-entry/renderer/gun.ts](browser-entry/renderer/gun.ts), [browser-entry/renderer/gun-sprite.ts](browser-entry/renderer/gun-sprite.ts)                                                                                                                                                     |
| Explore the NGE enemy-population harness            | [browser-entry/harness/enemy-population.ts](browser-entry/harness/enemy-population.ts), [browser-entry/harness/enemy-mlp.ts](browser-entry/harness/enemy-mlp.ts), [browser-entry/harness/enemy-evolution.ts](browser-entry/harness/enemy-evolution.ts), [browser-entry/harness/fitness.ts](browser-entry/harness/fitness.ts) |
| Tune the combat, return-fire, and death-effect loop | [browser-entry/host/game/combat.ts](browser-entry/host/game/combat.ts), [browser-entry/host/game/tick.ts](browser-entry/host/game/tick.ts), [browser-entry/renderer/derez.ts](browser-entry/renderer/derez.ts), [browser-entry/worker/display.worker.ts](browser-entry/worker/display.worker.ts) |
| Tune per-death selection, MAP-Elites archive, or arms-race evolution | [browser-entry/harness/select.ts](browser-entry/harness/select.ts), [browser-entry/harness/enemy-evolution.ts](browser-entry/harness/enemy-evolution.ts), and the NGA core in `src/` |
| Connect NGE lifecycle evolution to a runtime `Network` | [src/neat/nge-main-agent/nge-to-network.ts](../../src/neat/nge-main-agent/nge-to-network.ts) — the materialization bridge that turns NGE lifecycle state (embryo → juvenile → adult → reproducing) into a wired, activatable `Network` |

## What Exists in This Folder

- `browser-entry/` — host shell, worker entry, shared constants, the grid DDA
  raycaster, the audio stub, the deterministic game-state harness, the renderer
  bridge that forwards state to the worker, and the NGE enemy-population harness.
- `scripts/` — the procedural enemy art pipeline and runtime enemy behavior:
  voxel descriptor, orthographic snapshot renderer, deterministic frame animator,
  PNG sprite-sheet/reference-snapshot generator, enemy AI controller, CPU
  billboard sprite renderer, and parallel inference infrastructure
  (`enemy-controller.parallel.utils.ts`) for SAB-backed weight pools and
  deterministic barrier-synchronized enemy AI. The runtime uses the bundled
  `examples/neatenstein/robot-sprite-data.js`; reference snapshots are written
  to `examples/neatenstein/generated/` when you run the generator.
- `index.html` — the demo page that loads the built host bundle.

## Architecture boundaries

This demo accumulates several cross-cutting quality boundaries. Each boundary is
a teaching surface: it shows how to keep a real-time example maintainable while
adding visual polish and algorithmic depth.

### Code Quality

The `browser-entry/shared/` layer is the dependency floor for the demo. Both
`scripts/` and `browser-entry/` import downward into it, so constants and
guards never flow upward into unrelated modules. The layer owns:

- [`shared/math-guards.utils.ts`](browser-entry/shared/math-guards.utils.ts) —
  a single source of truth for `clamp`, `clamp01`, `isFiniteNumber`, and the
  dimension guards used by the renderer and the harness.
- [`worker/display.worker.message-handler.ts`](browser-entry/worker/display.worker.message-handler.ts) —
  the orchestrator that enforces the render compositing order
  (floor → ceiling → walls → sprites → pulses/sparks → bolts).
- [`worker/display.worker.test-hooks.ts`](browser-entry/worker/display.worker.test-hooks.ts) —
  test-only `__testOnly*` exports that let assertions reach internal helpers
  without widening the public API.
- [`worker/display.worker.message-handler.utils.ts`](browser-entry/worker/display.worker.message-handler.utils.ts),
  [`worker/display.worker.eval-delegation.utils.ts`](browser-entry/worker/display.worker.eval-delegation.utils.ts),
  and [`worker/display.worker.test-helpers.ts`](browser-entry/worker/display.worker.test-helpers.ts) —
  message-handler utilities, eval-delegation helpers, and shared worker test
  fixtures (mock globals, message senders, mock canvas/context) used by focused
  worker test files.

`@deprecated` markers were removed from the worker API surface, and dead
helper code was consolidated into the `shared/` layer so the same guard logic is
reused by the host, the worker, and the art scripts.

### Raycasting Quality

The renderer in [`browser-entry/renderer/`](browser-entry/renderer/) treats
the CPU raycaster as a tier, not a prototype. The raycasting quality boundary
contains:

- `computeWallTexcoord` in [`renderer/walls.ts`](browser-entry/renderer/walls.ts) —
  derives a wall-column texture coordinate from the perpendicular distance and
  side normal so flat-shaded walls can accept subtle texture accents without
  breaking the neon-dominant look.
- `castNeatensteinFloorPerPixel` in [`renderer/floor.ts`](browser-entry/renderer/floor.ts) —
  unconditional per-pixel floor casting that renders a procedural world-space
  integer grid via `fract(worldCoord)`. Grid spacing is exactly 1 world unit,
  keeping floor lines aligned with wall bases without sampled textures.
- `resolveSideDistance` in [`renderer/raycast.ts`](browser-entry/renderer/raycast.ts) —
  a NaN guard for rays aligned with grid lines, so DDA perpendicular distance
  never divides by zero.
- Unified z-buffer sentinel: `Infinity` is the single empty value across walls,
  sprites, and effects, removing the mixed sentinel bugs that caused depth races.
- Shader modules in [`renderer/shaders/`](browser-entry/renderer/shaders/) —
  `camera-uniform.ts`, `wall-dda.ts`, and `floor-caster.ts` export GLSL ES 3.00
  source strings that mirror the CPU algorithms. They are forward-looking
  scaffolds for a future GPU tier and serve as alignment tests for the current
  CPU implementation.

### Algorithm Upgrades

The NGE enemy harness gained several state-of-the-art operators:

- [`harness/map-elites.ts`](browser-entry/harness/map-elites.ts) — a 10×10
  MAP-Elites archive keyed by `aggression` and `positioning` behavior
  descriptors. `addToMapElitesArchive` admits a candidate only if it improves the
  cell score, and `computeNoveltyForArchive` measures behavioral distance so
  the archive can grow by diversity as well as fitness.
- [`harness/cma-es.ts`](browser-entry/harness/cma-es.ts) — separable CMA-ES
  (`createSepCmaEs`, `stepSepCmaEs`) with diagonal covariance, making
  90-weight MLP optimization feasible in the browser.
- [`harness/league.ts`](browser-entry/harness/league.ts) — a unified league that
  merges hall-of-fame and opponent-pool concerns: `addCurrentChampion`,
  `addDiverseSample`, `sampleOpponents`, and `getCurriculumOpponents` keep
  current champions, past main exploiters, and MAP-Elites diversity samples in
  one bounded structure.
- [`harness/transition-replay.ts`](browser-entry/harness/transition-replay.ts) —
  per-enemy bounded replay buffers (`createTransitionBuffer`), replay-driven
  weight updates (`runReplayUpdates`), death-surprise scoring
  (`computeDeathSurprise`), replay pressure normalization
  (`computeReplayPressureFromSurprise`), and CERL-style shared replay
  (`createSharedReplayBuffer`, `warmStartFromSharedReplay`).
- [`harness/curriculum-difficulty.ts`](browser-entry/harness/curriculum-difficulty.ts) —
  `computeCurriculumDifficulty` turns player performance telemetry into a
  `[0, 1]` difficulty signal, `scaleEnemyCapability` maps it to mutation sigma,
  and `computeWaveDifficulty` advances wave difficulty from survival rate.
- [`harness/bounded-concurrency.ts`](browser-entry/harness/bounded-concurrency.ts) —
  `runBoundedConcurrency` limits enemy inference to a fixed worker count instead
  of an unbounded `Promise.all`.
- CTRNN time constants in
  [`src/architecture/node/node.ts`](../../src/architecture/node/node.ts) and
  [`src/methods/mutation/mutation.ts`](../../src/methods/mutation/mutation.ts) —
  each `Node` now carries an evolvable `timeConstant` and an
  `applyCtrnnActivation(inputSum, dt)` method, while `MOD_TIME_CONSTANT` and
  `mutateTimeConstant` add per-node temporal-memory search moves to the mutation
  shelf.

## Run The Example

### Build the bundles

From the repo root:

```bash
node scripts/build-neatenstein.mjs
```

This produces `docs/assets/neatenstein.bundle.js` and
`docs/assets/neatenstein.worker.js` (a classic IIFE worker bundle). Open
`examples/neatenstein/index.html` in a browser to load the demo.

## Gameplay

The live demo is tuned around a short, readable combat loop:

- Render distance is capped at 30 cells so enemies emerge from fog with just
  enough warning time.
- Enemies spawn with 100 HP; each player bolt deals 20 damage, so a clean kill
  takes five hits.
- A non-lethal hit stuns an enemy for 200 ms and briefly pushes it back by one
  cell, giving the player breathing room.
- Enemies fire back with 10-damage bolts; the player has a 500 ms invincibility
  window after taking damage.
- Enemy hits leave a short neon impact mark (`NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS`).
- Dead enemies dissolve with a 700 ms Tron-style pixel-by-pixel de-rez animation
  (`ENEMY_CONTROLLER_DE_REZ_DURATION_MS`).

## The Core Idea In One Glance

```mermaid
flowchart LR
    subgraph Runtime["Browser runtime"]
        Host["browser-entry host"]
        Waves["host/waves.ts"]
        Worker["browser-entry worker"]
        Controller["scripts/enemy-controller.ts"]
        Sprite["scripts/enemy-sprite.ts"]
        Renderer["renderer/ DDA raycaster"]
        Frame["NeatensteinRenderFrame SoA"]
        Audio["WebAudio audio engine"]
    end

    subgraph ArtPipeline["Procedural enemy art pipeline"]
        Voxel["scripts/voxel-enemy.ts"]
        Snapshot["scripts/snapshot-renderer.ts"]
        Animator["scripts/enemy-animator.ts"]
        Generator["scripts/generate-enemy-sprites.ts"]
        Atlas["robot-sprite-data.js"]
    end

    Host --> Waves
    Waves --> Worker
    Worker --> Controller
    Worker --> Renderer
    Controller --> Sprite
    Renderer --> Frame
    Host --> Audio
    Voxel --> Snapshot
    Snapshot --> Generator
    Animator --> Generator
    Generator --> Atlas
    Atlas --> Sprite
    Sprite --> Renderer
```

The host owns audio, input, and wave transitions. The worker owns simulation,
NGE inference, enemy AI, and rendering. The two communicate through a versioned,
zero-copy render frame that uses typed arrays and a transfer list. Enemy
sprites are generated procedurally from a compact voxel descriptor and then
projected as camera-facing billboards inside the worker. The maze is generated
by a deterministic coarse-grid recursive backtracker and then rendered with a
custom grid DDA raycaster. Enemy genomes are managed by an NGE harness: each
death contributes to a per-variant fitness ledger, parents are selected
proportionally, and the best performers can be preserved in a MAP-Elites archive.
Enemy AI inference is parallelized via a tiered strategy (SharedArrayBuffer pool
→ InferenceChannel → inline) with a deterministic barrier that tags each result
with `(simTick, enemyIndex)` and applies them in index order, preserving
reproducibility regardless of inference latency.
