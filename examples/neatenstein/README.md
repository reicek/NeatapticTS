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
| Explore the procedural enemy art pipeline           | [scripts/voxel-enemy.ts](scripts/voxel-enemy.ts), [scripts/snapshot-renderer.ts](scripts/snapshot-renderer.ts), [scripts/enemy-animator.ts](scripts/enemy-animator.ts), [scripts/generate-enemy-sprites.ts](scripts/generate-enemy-sprites.ts)                                                   |
| Wire enemies into the live renderer                 | [scripts/enemy-controller.ts](scripts/enemy-controller.ts), [scripts/enemy-sprite.ts](scripts/enemy-sprite.ts), [browser-entry/host/waves.ts](browser-entry/host/waves.ts), [browser-entry/host/renderer-bridge.ts](browser-entry/host/renderer-bridge.ts)                                       |
| Understand the raycaster and frame protocol         | [browser-entry/renderer/](browser-entry/renderer/)                                                                                                                                                                                                                                               |
| Tune the plasma-cannon overlay and voxel projection | [browser-entry/renderer/gun.ts](browser-entry/renderer/gun.ts), [browser-entry/renderer/gun-sprite.ts](browser-entry/renderer/gun-sprite.ts)                                                                                                                                                     |
| Explore the NGE enemy-population harness            | [browser-entry/harness/enemy-population.ts](browser-entry/harness/enemy-population.ts), [browser-entry/harness/enemy-mlp.ts](browser-entry/harness/enemy-mlp.ts), [browser-entry/harness/fitness.ts](browser-entry/harness/fitness.ts)                                                           |
| Tune the combat, return-fire, and death-effect loop | [browser-entry/host/game/combat.ts](browser-entry/host/game/combat.ts), [browser-entry/host/game/tick.ts](browser-entry/host/game/tick.ts), [browser-entry/renderer/derez.ts](browser-entry/renderer/derez.ts), [browser-entry/worker/display.worker.ts](browser-entry/worker/display.worker.ts) |
| Tune NGE lifecycle or enemy co-evolution            | [browser-entry/harness/](browser-entry/harness/) and the NGA core in `src/`                                                                                                                                                                                                                      |

## What Exists in This Folder

- `browser-entry/` — host shell, worker entry, shared constants, the grid DDA
  raycaster, the audio stub, the deterministic game-state harness, and the
  renderer bridge that forwards state to the worker.
- `scripts/` — the procedural enemy art pipeline and runtime enemy behavior:
  voxel descriptor, orthographic snapshot renderer, deterministic frame animator,
  PNG sprite-sheet/reference-snapshot generator, enemy AI controller, and CPU
  billboard sprite renderer. The runtime uses the bundled
  `examples/neatenstein/robot-sprite-data.js`; reference snapshots are written
  to `examples/neatenstein/generated/` when you run the generator.
- `index.html` — the demo page that loads the built host bundle.

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
projected as camera-facing billboards inside the worker.
