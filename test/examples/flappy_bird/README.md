# Flappy Bird (NEAT demo)

This is a tiny Flappy Bird–style neuroevolution demo built **only** with this repo’s NeatapticTS implementation.

Goal: provide a quick, repeatable way to exercise the NEAT pipeline (population creation, evaluation, mutation/crossover, speciation) on a classic control problem.

The trainer now prioritizes **faster convergence** and **more stable generation quality** by using shared-seed, multi-stage evaluation.

## Run

From the repo root:

```bash
npx ts-node test/examples/flappy_bird/trainFlappyBird.ts
```

You should see generation logs like:

- `gen=1 best=... pipes=... frames=...`

The trainer does not stop on a fixed generation count. It keeps evolving until you close it (`Ctrl+C` in terminal).

## How it works

- Population is set to `100` to widen exploration while preserving strong winners (`elitism = 10`).
- Mutation now uses annealing (`0.70 -> 0.25` rate, `2 -> 1` amount over early generations):
  - early generations explore aggressively,
  - later generations refine and stabilize.

- Evaluation is now **population-level** and deterministic per generation:
  1. quick screen on all genomes (`3` shared seeds, shorter rollout, early-stop heuristic),
  2. full pass on top subset (`8` shared seeds),
  3. final reevaluation of top candidates (`32` shared seeds).

  This removes most per-genome luck while keeping runtime practical.

- The bird has a fixed `x` position.
- Pipes move left at a constant speed.
- Difficulty ramps adaptively with progress (gap narrows, pipes speed up, spawn interval shortens).
- A generation-level curriculum controls difficulty scale:
  - very easy fixed profile in early generations,
  - smooth ramp,
  - full adaptive difficulty in later generations.
- Pipe gaps are randomized but structured: each run starts around `40%` wider than the current hardest target, then each new spawned pipe shrinks a few pixels until reaching the current hardest gap.
- Each genome’s network receives 10 inputs:
  1. bird y position (normalized)
  2. bird vertical velocity (normalized)
  3. distance to next pipe (normalized)
  4. delta between bird y and next gap center (normalized)
  5. next gap top (normalized)
  6. next gap bottom (normalized)
  7. distance to second upcoming pipe (normalized)
  8. delta to second upcoming gap center (normalized)
  9. time-to-next-pipe closeness (normalized)
  10. signed next-gap clearance (inside vs outside corridor)
- The network outputs 2 values (`no flap`, `flap`); we flap when `output[1] > output[0]`.
- Fitness is now composed from normalized channels with caps to reduce domination by one term:
  - survival,
  - pipe progress,
  - dense shaping,
  - terminal shaping.

Dense shaping rewards still include:
- staying aligned with the next gap,
- reducing horizontal distance to the next pipe,
- improving centering toward the gap,
- keeping positive corridor clearance,
- pre-aligning toward the second upcoming gap,
- maintaining controllable vertical velocity.

The trainer logs robust distribution statistics each generation (`mean`, `median`, `p90`, `std`) in addition to best score.

In browser playback, the demo renders the full generation population (not only the top bird):
- only living birds are shown (eliminated birds are removed immediately),
- each bird has a distinct color,
- the current leader is highlighted.

The episode runner remains deterministic when a seed is provided. Training now uses **shared seed batches per generation** (instead of per-genome private seeds) for fairer comparisons.
