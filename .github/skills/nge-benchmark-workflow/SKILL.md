---
name: nge-benchmark-workflow
description: 'Design, implement, or validate NGE benchmark environments and demo harnesses in NeatapticTS. Use when working on predator/prey coevolution, ant-hive collective intelligence, racing curriculum tiers, rolling opponent snapshots, shared-field benchmark semantics, benchmark worker topology, observability charts, fairness contracts, or browser simulation methodology that consumes NGE core primitives.'
argument-hint: 'Describe the benchmark family, the active plan file, whether the pass is world design, harness implementation, metrics, or validation, which core primitives are assumed available, and what observable or acceptance criterion must be demonstrated.'
user-invocable: true
disable-model-invocation: false
---

# NGE Benchmark Workflow

Use this skill when the work is about benchmark environments, curriculum design,
evaluation methodology, or demo harnesses that exercise NGE.

This skill owns the durable workflow for:

- predator/prey coevolution,
- ant-hive collective intelligence,
- team-racing curriculum tiers,
- evaluation fairness,
- shared benchmark worker topology,
- observability and ablation methodology,
- browser simulation architecture for NGE demos.

It does not own core `NGE_DNA`, computation motif semantics, or lifecycle policy.
If a benchmark is blocked because a core primitive is missing or underspecified,
handoff to `nge-core-algorithm` instead of compensating in demo code.

See [NGE benchmark sources](./references/nge-benchmark-sources.md) for
paraphrased notes on curriculum learning, coevolution, and benchmark-plan
boundaries.

## Scope Boundary

- In scope: environment design, tier ladders, evaluation harnesses, deterministic
  seed packs, shared-field demo semantics, scripted adversary design,
  rolling-opponent snapshots, multi-agent worker coordination, observability
  charts, ablation harnesses, and benchmark acceptance criteria.
- Out of scope: core `computationType` meaning, `NGE_DNA` schema, deterministic
  development semantics, assimilation policy internals, and generic browser build
  tooling unless the benchmark specifically exposes a runtime constraint.

## Benchmark Families

### Predator or prey

- Owns two-population coevolution.
- Owns chem-trail plus voice environment rules.
- Owns rolling-opponent snapshots, arms-race observables, and generation
  synchronization barriers.

### Ant hive

- Owns colony-level fitness, GeoFront damage and repair, pheromone diffusion,
  scripted Angel pressure, and caste-emergence observability.
- Owns the benchmark decision to make pheromone deposition hardwired while
  evolving reactions to pheromone.

### Team racing

- Owns category ladder design, pit strategy, tire degradation budgets, team radio
  as stigmergy analog, and cross-team promotion fairness.
- Owns carry-state versus reset-state semantics across tiers.

## Shared Contracts

### Benchmark work must stay downstream of the core algorithm

- If a benchmark needs a new motif, memory primitive, or reproduction rule,
  define that missing requirement and hand it back to `nge-core-algorithm`.
- Do not patch around missing core semantics with benchmark-local hidden rules.

### Fairness comes before spectacle

- Compared genomes should see the same deterministic seed pack when the benchmark
  is claiming comparative improvement.
- In coevolutionary setups, evaluate against a frozen opponent snapshot when the
  plan requires stability against one-generation collapses.

### Group-level fitness must stay aligned with the task

- Team racing scores the team.
- Ant hive scores the colony.
- Predator or prey maintains separate population fitness with reciprocal pressure.

### Communication bootstraps may be hardwired only when the plan says so

- Predator or prey voice is engine-hardwired; evolution learns reactions.
- Ant pheromone deposition is hardwired; evolution learns how to read and exploit
  the field.
- Do not convert these into arbitrary learned emit-or-not toggles unless the plan
  changes.

## When to Use

- Implementing a benchmark world or simulation loop.
- Designing or updating a curriculum ladder.
- Adding rolling-opponent evaluation logic.
- Defining benchmark charts, ablation passes, or observables.
- Building episode-worker, simulation-worker, or coordinator logic for NGE demos.
- Validating whether role differentiation or collective behavior is actually
  observable.

## Task Packet

Pass a compact packet that includes:

- benchmark family,
- active plan file,
- smallest active acceptance criterion,
- assumed core prerequisites,
- focused validation target.

Compact example:

```text
Use nge-benchmark-workflow for predator/prey.
Plan: plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md.
Target: rolling opponent snapshot and generation barrier.
Core prerequisites: Phase E reproduction modes and two-population harness are already available.
Acceptance criterion: snapshot prevents single-generation fitness collapse while preserving deterministic seed-pack replay.
Validation: focused tests for snapshot sampling, barrier synchronization, and seed-stable episode aggregation.
```

## Required Workflow

1. Read the smallest relevant benchmark plan first.
2. Read `plans/completed/NEAT_Genesis_EvoDevo.md` only enough to confirm the core or
   benchmark boundary.
3. Name the exact observable or acceptance criterion the pass is trying to prove.
4. Add the smallest owner-local failing test or harness check.
5. Implement the minimal environment or orchestration change.
6. Immediately rerun the same focused validation after the first substantive
   edit.
7. If the issue is actually a missing core primitive, stop widening the benchmark
   and hand back to `nge-core-algorithm`.
8. Report the measured observable, the fairness contract used, and any remaining
   prerequisites.

## Curriculum Rules

- Difficulty must be explicit and stageable.
- Promotion should depend on deterministic packs or repeated reliability, not one
  lucky episode.
- Every tier should add one major complexity layer at a time.
- Carry-state and reset-state boundaries must be explicit whenever a curriculum
  promotes a phenotype upward.

## Coevolution Rules

- Keep populations genuinely independent when the plan says they are independent.
- Use rolling snapshots or hall-of-fame samples when live-opponent feedback would
  cause unstable one-generation collapse.
- Surface arms-race metrics that show more than raw fitness alone.
- Treat Red Queen behavior as a benchmark goal: the system should keep adapting,
  not converge immediately to a trivial fixed point.

## Shared-Field Rules

- Prefer typed-array field state over per-cell object graphs.
- Field updates should be deterministic and cheap enough for browser simulation.
- Shared fields are part of the benchmark semantics, not merely rendering effects.
- Use ablations to verify the field matters.

## Worker and Runtime Rules

- Prefer between-episode parallelism when an episode has strong shared-state
  coupling.
- Keep a single authoritative world owner for display episodes.
- Keep episode workers stateless where practical.
- Preserve training-only, display-only, and hybrid mode boundaries explicitly.

## Validation Cadence

- Seed-stable replay for environment and episode setup.
- Focused tests for sensors, worker protocols, snapshot logic, and tier
  transitions.
- Ablation checks for communication channels and memory surfaces.
- Browser performance check against the benchmark's stated display target.
- Focused checks that the claimed observable is actually measurable.

## Observable Design

Prefer benchmark observables that expose mechanism instead of only score:

- predator or prey: route diversity, coordination index, reproduction-mode shifts,
  wiring cost,
- ant hive: caste divergence, pheromone concentration, GeoFront integrity,
  tunnel-coverage response,
- racing: blocker emergence, pit-lap distribution, radio mutual information,
  role divergence.

## Guardrails

- Do not hardcode roles in identical-DNA benchmarks.
- Do not give planner-style full-map oracle inputs unless the plan explicitly says
  so.
- Do not replace rolling snapshots with live-opponent evaluation in coevolutionary
  tiers just to make code simpler.
- Do not let rendering polish block harness correctness; hand layout issues to
  `visualizer-workflow` when needed.
- Do not shift benchmark-local work into browser-bundle or runtime-tooling work;
  hand those boundaries to `browser-build` when they are the real blocker.

## Expected Final Output

A strong benchmark pass should report:

- benchmark family and active tier or environment,
- the exact observable or acceptance criterion targeted,
- the fairness contract used,
- validation and ablation results,
- any missing upstream primitive that belongs to `nge-core-algorithm`,
- any runtime or UI follow-up that belongs to another owner.
