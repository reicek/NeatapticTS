# NGE Benchmark Reference Notes

This file stores paraphrased reference notes for the
`nge-benchmark-workflow` skill.

These notes summarize upstream sources instead of copying them verbatim. Use the
linked sources for canonical wording and detail.

## Source Map

### 1. Team racing benchmark plan

- Source: `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- Why it matters:
  - Defines the category ladder, team-radio stigmergy analog, tire degradation,
    pit strategy, cross-team promotion rules, and racing acceptance criteria.

### 2. Predator or prey benchmark plan

- Source: `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md`
- Why it matters:
  - Defines two-population coevolution, rolling-opponent snapshots, chem trails,
    hardwired voice, arms-race observables, and worker synchronization needs.

### 3. Ant-hive benchmark plan

- Source: `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md`
- Why it matters:
  - Defines colony-level fitness, GeoFront mechanics, pheromone diffusion,
    scripted Angel pressure, caste emergence, and ant-hive acceptance criteria.

### 4. Wikipedia: curriculum learning

- URL: https://en.wikipedia.org/wiki/Curriculum_learning
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Curriculum learning is about exposing a learner to progressively harder tasks.
  - Difficulty must be defined, and the schedule must be explicit or self-paced.
  - This matches the racing category ladder and any future staged NGE benchmark.

### 5. Wikipedia: coevolution

- URL: https://en.wikipedia.org/wiki/Coevolution
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Coevolution is reciprocal selective pressure between populations or species.
  - Predator and prey dynamics often become arms races rather than stable,
    one-sided optimization.
  - Red Queen language is useful for describing why rolling-opponent evaluation
    should preserve gradual adaptation instead of single-generation collapse.

### 6. Wikipedia: stigmergy

- URL: https://en.wikipedia.org/wiki/Stigmergy
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Shared traces in the environment can coordinate many simple agents without
    direct addressed messaging.
  - This supports both ant pheromone fields and racing radio as shared-field,
    indirect coordination primitives.

### 7. Wikipedia: proportional control

- URL: https://en.wikipedia.org/wiki/Proportional_control
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Proportional feedback is a standard baseline for steering and throttle
    controllers in the racing benchmark.
  - The racing controller blends proportional line-following signals with
    evolved network outputs, so the concept is relevant to both scripted and
    evolved policies.

### 8. MDN: Transferable objects

- URL: https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferring_objects
- Authors: Mozilla contributors
- License note: MDN content is available under CC BY-SA 2.5; code samples are
  available under CC0 / MIT.
- Why it matters:
  - Racing benchmark worker messages use transfer lists to move packed
    typed-array snapshots zero-copy between the worker and the host.
  - Correct transfer-list ownership prevents detached-buffer bugs and keeps
    the streaming path fast enough for display cadence.

### 9. Wikipedia: forward compatibility

- URL: https://en.wikipedia.org/wiki/Forward_compatibility
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - The `RacingRenderFrame` schema version is a forward-compatibility gate:
    hosts reject frames whose `schemaVersion` they do not recognize.
  - This pattern lets the packed snapshot format evolve without silently
    corrupting older consumers.

## Practical Notes

### Benchmarks prove behavior, not just throughput

- A benchmark pass is incomplete if it only runs fast.
- It must show the intended observable: arms race, role divergence, pit strategy,
  collective defense, or field usage.

### Difficulty ladders need explicit semantics

- A curriculum is not just a list of levels.
- Each tier should add one meaningful constraint or strategic burden and define
  how advancement happens.

### Coevolution needs stabilization boundaries

- If both sides adapt directly against each other every generation with no frozen
  reference set, fitness can become noisy enough to hide real progress.
- Frozen snapshots and hall-of-fame sampling are benchmark methodology, not
  algorithm-core semantics.

### Shared fields are externalized memory

- Pheromone grids, chem trails, and team-radio fields should be treated as real
  environment state that agents exploit over time.
- This is why ablation matters: the field should change behavior measurably.

### Browser demos need typed-array discipline

- Shared benchmark state should stay flat and transferable.
- Per-cell objects and ad hoc allocation sabotage the very scale claims the
  benchmark is supposed to demonstrate.

## Working Heuristics For This Repo

- Keep benchmark work downstream of core NGE semantics.
- Use deterministic seed packs for claims about comparative improvement.
- Favor observable mechanism charts over cosmetic dashboards.
- Treat display mode and training mode as separate concerns even when they share
  world rules.
- Use ablations whenever a benchmark claims communication, memory, or coordination
  actually matters.
