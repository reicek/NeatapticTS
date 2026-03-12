# Flappy Bird (NeatapticTS)

This folder is the repo's most complete end-to-end neuroevolution example.

It uses a Flappy Bird-style control problem to exercise the full NeatapticTS pipeline:

- deterministic environment stepping,
- observation construction,
- rollout evaluation and fitness shaping,
- staged population-level selection,
- browser playback and visualization,
- worker-based simulation and snapshot transport.

If you want one place in the repository that shows how NEAT training, deterministic evaluation, browser rendering, and educational visualization fit together, start here.

## What This Example Is For

This example is not just a game clone. It is a compact systems demo for the library.

At a high level, it answers four practical questions:

1. How do you wire a NEAT population into a repeatable control problem?
2. How do you reduce luck so selection pressure reflects policy quality instead of lucky seeds?
3. How do you replay evolved behavior in the browser without moving the full simulation onto the main thread?
4. How do you expose the evolved network structure in a way that is useful for learning and debugging?

The current trainer emphasizes faster convergence and more stable generation quality by using shared-seed, multi-stage evaluation rather than one-rollout-per-genome scoring.

## Quick Start

Run training from the repo root:

```bash
npx ts-node test/examples/flappy_bird/trainFlappyBird.ts
```

You should see logs like:

- `gen=1 best=... pipes=... frames=...`

Training runs continuously until you stop it with `Ctrl+C`.

Run the browser demo locally from the repo root:

```bash
npm run start:local-server
```

Then open:

- `http://localhost:8080/test/examples/flappy_bird/index.html`

Notes:

- the page loads bundles from `docs/assets`, so run `npm run docs` after code changes,
- if port `8080` is busy, adjust the local server configuration.

## Folder Map

The folder is split by responsibility rather than by one giant "game" module.

- `browser-entry/`: main-thread browser runtime, HUD, playback renderer, network visualization, viewport/layout helpers, and host-side worker wiring.
- `constants/`: shared visual, training, environment, and playback constants used across multiple boundaries.
- `environment/`: deterministic Flappy world state, stepping, collision/progress accounting, and observation bridge helpers.
- `evaluation/`: rollout execution and fitness aggregation for one genome or a shared-seed batch.
- `flappy-evolution-worker/`: browser worker runtime that owns evolution, playback simulation, and packed snapshot transport.
- `simulation-shared/`: shared simulation-facing math and feature logic reused across environment, trainer, and browser paths.
- `trainer/`: Node-side training orchestration, staged evaluation plans, annealed mutation scheduling, and generation logging.
- `flappyEnvironment.ts`: compatibility-facing environment entrypoint.
- `flappyEvaluation.ts`: compatibility-facing evaluation entrypoint.
- `flappyEvolution.worker.ts`: worker entry used by browser bundling.
- `trainFlappyBird.ts`: simplest Node entrypoint for starting training.
- `index.html`: local browser shell for the interactive demo.
- `index.ts`: aggregated example-facing exports.
- `rng.ts`: deterministic RNG utilities shared by environment and evaluation flows.

## How The Split Works

The cleanest way to understand this example is to treat it as three layers plus two bridges.

### 1. Simulation layer

This is the deterministic game world.

- `environment/` owns the mutable episode state and how one frame advances.
- `simulation-shared/` owns logic that must stay consistent across training and playback, especially observation semantics.
- `constants/` supplies the fixed knobs that shape the world and visual presentation.

This layer should be understandable without knowing anything about the browser UI.

### 2. Evaluation and training layer

This is the evolutionary side.

- `evaluation/` answers: "How good was this network on one or many deterministic rollouts?"
- `trainer/` answers: "How should the population be evaluated, ranked, mutated, and logged across generations?"

The important architectural choice here is that evaluation is population-aware, not just genome-local. Genomes are compared on shared seeds in stages:

1. quick screen across the full population,
2. fuller pass on a narrowed subset,
3. reevaluation of top candidates.

That keeps runtime practical while reducing the amount of selection luck caused by random rollout variation.

### 3. Browser presentation layer

This is the educational and visual side.

- `browser-entry/host/` owns page setup and canvas/container lifecycle.
- `browser-entry/runtime/` owns browser-side orchestration.
- `browser-entry/worker-channel/` and `browser-entry/playback/worker-channel/` own message-level communication with the worker.
- `browser-entry/playback/` owns frame reconstruction, background/trail rendering, snapshot handling, and live playback pacing.
- `browser-entry/network-view/` and `browser-entry/visualization/` own the live network drawing and legend semantics.

This layer is intentionally thin on simulation authority. The browser renders and explains; it does not become the source of truth for evolution.

### 4. Worker bridge

The browser does not evolve or simulate the population on the main thread.

- `flappy-evolution-worker/` owns worker-local NEAT runtime setup, generation requests, playback simulation, and packed frame snapshots.

That split keeps the UI responsive while still allowing rich playback and HUD updates.

### 5. Compatibility bridge

The root files such as `flappyEnvironment.ts`, `flappyEvaluation.ts`, and `index.ts` act as stable shelves for consumers or older imports that should not need to know the current folder layout.

## Execution Paths

There are two main ways through this codebase.

### Training path

1. `trainFlappyBird.ts` starts the trainer.
2. `trainer/` resolves generation-level evaluation and mutation policy.
3. `evaluation/` runs deterministic rollouts against the environment.
4. `environment/` steps the world and returns episode outcomes.
5. The trainer logs robust generation statistics and evolves the population again.

### Browser path

1. `index.html` loads the browser bundle.
2. `browser-entry/` initializes the host UI and worker connection.
3. `flappy-evolution-worker/` evolves or simulates playback off-thread.
4. The worker posts packed snapshots and generation summaries back to the browser.
5. `browser-entry/playback/` reconstructs frames, renders the flock, and updates the HUD.
6. `browser-entry/network-view/` renders the active best network for inspection.

## Core Design Choices

Several choices are worth knowing before you read deeper.

- Determinism matters. The episode runner is deterministic when seeded, and the trainer uses shared seed batches per generation for fairer comparison.
- The observation vector is intentionally temporal. Each network receives 38 inputs composed from current, previous, and two-frames-ago feature slices, plus action-memory channels.
- The network output is simple: two competing action scores, `no flap` and `flap`, with flap chosen when `output[1] > output[0]`.
- Fitness is intentionally decomposed. It combines survival, progress, dense shaping, and terminal shaping so score inspection is more informative than a single opaque reward.
- Playback is population-level. The browser can render the whole generation population, not just one best bird.
- Visualization is educational, not decorative only. The network panel exposes node bias, connection magnitude/sign, disabled edges, and a legend so people can inspect what evolution is building.

## Observation And Fitness Summary

The policy sees a compact but structured 38-input view of the world:

1. current-frame core features,
2. previous-frame core features,
3. two-frames-ago core features,
4. last action,
5. recent flap-rate memory.

Those core features focus on corridor geometry and control urgency rather than raw scene detail:

- bird vertical position and velocity,
- distance and delta to the next gap,
- next gap bounds,
- distance and delta to the second gap,
- next-gap clearance,
- required vertical velocity toward the next-gap center,
- entry urgency,
- one-flap reachability at entry.

Fitness then combines normalized channels with caps so one lucky dimension does not dominate selection:

- survival,
- pipe progress,
- dense shaping,
- terminal shaping.

Dense shaping still rewards staying aligned to the next gap, approaching the next pipe productively, maintaining clearance, and preparing for the second upcoming gap.

## Browser Runtime At A Glance

In browser playback:

- heavy evolution, evaluation, and playback simulation run in a Web Worker,
- the main thread focuses on rendering and lightweight snapshot handoff,
- HUD counter updates are throttled,
- only living birds remain visible,
- each bird has a distinct color,
- the current leader is highlighted,
- the right-hand panel renders the active best network.

The background and playback renderer are also deliberately split into sub-boundaries such as `background/`, `frame-render/`, `snapshot/`, `trail/`, and `worker-channel/` so rendering concerns do not collapse into one monolithic canvas file.

## Recommended Reading Order

If you are new to this example, read in this order:

1. this README for the high-level map,
2. `trainer/README.md` to understand the evolutionary loop,
3. `evaluation/README.md` to understand rollout and scoring,
4. `environment/README.md` to understand the deterministic world,
5. `browser-entry/README.md` to understand the browser runtime,
6. `flappy-evolution-worker/README.md` to understand the worker protocol and playback transport.

If you only care about one use case:

- training and fitness tuning: start with `trainer/` and `evaluation/`,
- environment or control-problem changes: start with `environment/` and `simulation-shared/`,
- browser playback or UI work: start with `browser-entry/` and `flappy-evolution-worker/`.

## Why This Example Matters In The Repo

This example is one of the clearest demonstrations of the repository's broader direction from the interactive examples plan: educational, inspectable examples that show both how the library is used and why particular architectural boundaries exist.

It is not the smallest example in the repository, but it is likely the best one for understanding how NeatapticTS is intended to feel in a real application.
