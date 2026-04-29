# NEAT Genesis EvoDevo: Ant Hive Ecosystem Demo

**Status:** [PLANNED]

This plan defines the ant-hive ecosystem benchmark for [NEAT Genesis EvoDevo (NGE)](NEAT_Genesis_EvoDevo.md). It is the primary multi-agent stress test for collective intelligence, stigmergy, role differentiation from identical DNA, and neuromodulation-driven behavioral switching — all rendered as a live web canvas simulation.

This benchmark is downstream of [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) and [Memory_Optimization.md](Memory_Optimization.md). If this plan conflicts with either upstream plan, the upstream plan wins.

---

## Scope and Maturity

This is a benchmark-architecture plan, not an implementation-complete spec.

- **In scope:** world design, pheromone field mechanics, agent roles and sensory channels, caste differentiation model, NGE feature mapping, canvas simulation spec, and acceptance criteria.
- **Out of scope (for now):** exact physics constants, rendering library selection, final reward weights, and worker protocol shape.
- **Authority rule:** [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) and [Memory_Optimization.md](Memory_Optimization.md) remain authoritative.

---

## Why This Demo

The ant hive is the most direct embodiment of the two biological inspirations behind NGE:

1. **DNA as a program:** every ant in the colony shares a nearly identical genome. The vastly different behavior of foragers, workers, and soldiers emerges not from different DNA, but from different developmental experience under identical instructions. This is the "Experience gates where capacity grows" principle made observable.

2. **Metabolically optimal specialized intelligence:** ant brains are among the most efficiently wired nervous systems known. The mushroom bodies store associative memories; the central complex handles navigation; neuromodulation switches behavioral modes. NGE's `EpisodicSlot`, `GatedRecurrentCell`, and `ModulatorBroadcaster` archetypes are direct engineering analogs.

The ant hive also introduces stigmergy — the mechanism by which complex collective behavior emerges from agents leaving traces in a shared environment, with no direct agent-to-agent messaging required.

---

## NGE Capabilities Exercised

| NGE capability                           | How the ant hive exercises it                                                                                        |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Role differentiation from identical DNA  | Foragers, workers, and soldiers develop from the same DNA; caste emerges from experience-gated structural plasticity |
| Stigmergy (shared typed-array field)     | Pheromone grid is the sole inter-agent communication channel                                                         |
| `ModulatorBroadcaster`                   | Alarm pheromone triggers defense-mode gain shift in a single forward pass                                            |
| `EpisodicSlot` (medium-term memory)      | Food-source locations and trail quality stored by similarity; mushroom body analog                                   |
| `GatedRecurrentCell` (short-term memory) | Path integration state; recent trail quality; short-horizon navigation context                                       |
| `GatingRouter`                           | Task switching (forage vs. tend nest vs. defend) without structural change                                           |
| Wiring economy + pruning                 | Ants develop only the sensor processing relevant to their emerged role                                               |
| `reproductionPolicy` — polyandric        | Queen produces diverse worker cohorts from multiple drone genetic contributions                                      |
| Collective intelligence                  | Colony fitness is measured at the colony level, not per-agent                                                        |

---

## Execution Alignment

This benchmark belongs to **Phase G (Multi-Agent + Collective Intelligence)** in the NGE roadmap. It must not begin serious implementation before:

- NGE Phase A (DNA + deterministic development) is stable.
- NGE Phase B (Juvenile focus + local growth/prune) is implemented.
- All Phase 0 computation motif primitives (`ModulatorBroadcaster`, `GatedRecurrentCell`, `EpisodicSlot`, `GatingRouter`) are implemented and opt-in verified.
- Stigmergy field infrastructure (typed-array grid, diffusion/decay ops, agent read/write interface) is implemented.

---

## Design Pillars

- **Identical DNA, divergent structure:** all agents in a colony share the same DNA. Structural divergence between castes must emerge from experience-gated plasticity alone, never from DNA divergence within a colony.
- **Pheromone-only communication:** agents coordinate exclusively through chemical field signals. No direct agent-to-agent messaging.
- **Colony-level fitness:** individual ant performance is not the selection target. Colony survival, food throughput, and threat response are the fitness signals.
- **Visible specialization:** the benchmark should produce observable differences in module size distributions between forager, worker, and soldier agents by adult stage.
- **Metabolic honesty:** total colony wiring cost is a secondary selection pressure. Dense, unspecialized wiring should lose to compact, role-specialized wiring over generations.
- **Canvas-native:** the full simulation must run in a browser canvas at interactive rates with a colony of at least 50–100 agents.

---

## World Design

### Environment

A 2D grid world rendered on HTML canvas. Grid resolution: 128×128 to 256×256 cells, each cell ~4–6 px at standard canvas size. The world wraps or has hard boundaries (configurable).

**World zones:**

- **Nest zone:** central area where food is deposited, eggs hatch, and new agents spawn. Protected by walls. Has a nest-scent pheromone maintained by workers.
- **Foraging zone:** the majority of the map. Contains food sources placed procedurally (cluster distribution mimicking real food patches).
- **Threat zone:** one or more areas from which threats (predator agents or environmental hazards) may emerge. Triggers alarm pheromone.
- **Open terrain:** traversable by all agents; no cost modifier.
- **Obstacle patches:** impassable blocks or slow-terrain patches that agents must navigate around.

### Food Sources

Food sources are finite resource nodes placed procedurally at the start of each evaluation episode. Each source has:

- a position and a resource count (depletes as foragers harvest)
- a visual marker on canvas (brightness scaled to remaining resource)
- no pheromone emission of its own — trails are laid entirely by forager agents

### Threats

Threats are simple autonomous agents (not NGE-evolved, or optionally from the Predator/Prey demo) that patrol or emerge from threat zones. On contact with a non-soldier ant, the ant is removed from the colony. Soldiers intercept threats and can neutralize them.

---

## Pheromone Field System

The pheromone system is the primary inter-agent communication substrate. It is implemented as a set of typed-array grids (one per channel), all sized `gridWidth × gridHeight × Float32`.

### Channels

| Channel                | Laid by                               | Read by              | Meaning                                                         |
| ---------------------- | ------------------------------------- | -------------------- | --------------------------------------------------------------- |
| `foodTrail`            | Foragers (returning with food)        | Foragers (outbound)  | Guide toward known food sources                                 |
| `nestTrail`            | Foragers (outbound from nest)         | Foragers (returning) | Guide back toward nest                                          |
| `alarmPheromone`       | Any agent under threat                | All agents           | Signal danger; triggers defense-mode via `ModulatorBroadcaster` |
| `recruitmentPheromone` | Soldiers near threats                 | Workers, soldiers    | Rally defenders to a location                                   |
| `nestScent`            | Workers in nest zone                  | All agents           | Gradient toward home; used for path integration calibration     |
| `trailQuality`         | Foragers (proportional to food found) | Foragers             | Reinforcement signal for good trails                            |

### Diffusion and Decay

Each tick, the field applies:

```
field[x][y] = field[x][y] * decayRate + diffusion(neighbors) * diffusionRate
```

`decayRate` and `diffusionRate` are per-channel constants in the simulation config. The operation is a simple 5-point stencil (center + 4 neighbors), runnable as a typed-array sweep. This is the most performance-critical inner loop and should be the first candidate for WASM or SharedArrayBuffer optimization at scale.

### Agent Read/Write

Agents sample pheromone values at their current cell and the 8 surrounding cells (a 3×3 read window), giving 9 × 6 = 54 pheromone input channels. Agents write pheromone by incrementing the value at their current cell by a species-specific deposition rate.

---

## Agent Roles

Three emerged roles, all developing from the same DNA:

### Forager

**Primary task:** find food sources, harvest one unit, return to nest, deposit.

**Expected module specialization by adulthood:**

- Dense chemosensory processing zone (food trail + nest trail channels)
- `EpisodicSlot` module storing food-source locations by similarity to sensory context
- `GatedRecurrentCell` for path integration state
- Pruned or dormant threat-response zone (not their job)

### Worker

**Primary task:** maintain nest (repair walls, tend eggs, deposit food into storage, maintain nest scent).

**Expected module specialization by adulthood:**

- Nest-scent and nest-structure sensing zone
- `EpisodicSlot` for nest-state memory (where eggs are, where food storage is)
- Pruned or dormant foraging trail zone
- Pruned or dormant combat zone

### Soldier

**Primary task:** detect and intercept threats; lay recruitment pheromone to rally allies.

**Expected module specialization by adulthood:**

- Dense alarm + recruitment pheromone sensing zone
- `ModulatorBroadcaster` output calibrated for fast alarm-to-attack mode switch
- `GatedRecurrentCell` for threat tracking state
- Pruned or dormant food-trail zone

**Caste determination:** no caste is hardcoded. All three roles develop from identical DNA through experience-gated structural plasticity. An agent hatched near a food source develops forager-like structure; one hatched near the nest perimeter develops soldier-like structure. The development stage must be reactive to environmental context.

---

## Sensory Channels (per agent)

All channels are egocentric (relative to the agent's current position and heading).

### Pheromone senses (54 channels)

- 3×3 read window × 6 pheromone channels

### Local world senses (12 channels)

- obstacle presence in 8 surrounding cells
- food source presence and quantity at current cell
- nest zone indicator
- threat zone proximity
- current surface type

### Agent-state senses (10 channels)

- current heading (sin/cos)
- carrying food (binary)
- energy level (if metabolic budget is enabled)
- role-signal (soft signal from recent behavior history — not hardcoded)
- time since last food delivery
- time since last threat contact

### Neighbor senses (16 channels)

- count of nearby foragers, workers, soldiers in 3×3 window
- alarm pheromone rate-of-change (delta from last tick)
- nearest visible threat bearing and distance (if in sensor range)
- nearest nest-mate heading (swarm cohesion signal)

### Short-horizon memory senses (8 channels)

Raw history for `GatedRecurrentCell` integration:

- recent movement direction history (last 4 ticks, 2 channels each)

**Total sensory input width: ~100 channels**

---

## Colony Fitness Signal

Fitness is measured at the **colony level**, not per individual. This forces collective behavior to emerge — an individually high-performing forager in a colony that collapses under threat pressure produces low fitness.

**Primary fitness components:**

1. Food throughput (units deposited per episode tick)
2. Colony survival fraction at episode end (proportion of starting agent count surviving)
3. Threat neutralization rate (threats intercepted before nest contact)
4. Nest integrity (proportion of nest-zone cells undamaged)

**Secondary pressure:**

- Total colony wiring cost (sum of all active agent connection counts) — lower is better, all else equal

**Episode structure:**

- Fixed-length episodes (e.g., 2,000 ticks)
- Deterministic food placement and threat emergence per evaluation seed
- Multiple seeds evaluated per generation to reduce luck

---

## Reproduction and Colony Lifecycle

### Within-episode

Agents are not replaced within an episode. A colony starts with a fixed count (e.g., 60 agents: 30 foragers, 20 workers, 10 soldiers by role emergence). Agents that die from threats are not replaced mid-episode.

### Between generations

The colony's DNA (shared across all agents) is the unit of evolution. Fitness is the colony-level score.

**Reproduction policy for this demo:**

- Default: **polyandric** — queen DNA is the primary template; drone contributions diversify module archetype deltas across the worker cohort. This mirrors real ant colony genetics and is why real colonies produce diverse worker specializations from a single queen genome.
- Parthenogenesis allowed: for producing identical-DNA twin colonies to test role divergence from experience alone.
- Standard sexual: for cross-colony genetic mixing between high-fitness lineages.

---

## Canvas Simulation Spec

### Rendering

- Each agent: a colored dot (3–5 px radius), color-coded by emerged role (detected by module size distribution heuristic or explicit role signal).
- Pheromone fields: semi-transparent color overlays on the grid (food trail = green, alarm = red, nest scent = blue, recruitment = orange). Overlay opacity scales with field intensity. Toggle-able per channel via UI controls.
- Food sources: bright yellow patches, opacity scales with remaining resource.
- Threats: red triangles.
- Nest zone: outlined in white.

### Performance targets

- 50–100 agents at 30+ fps on a mid-range consumer device.
- Pheromone field diffusion/decay: single typed-array sweep per tick, no per-cell object allocation.
- Agent evaluation: each agent's network forward pass is a typed-array operation using slab-backed phenotype (memory plan infrastructure).
- Optional: Web Worker offload for pheromone field updates and agent network evaluations when main thread frame budget is tight.

### UI controls

- Play / pause / step
- Speed multiplier (1×, 2×, 5×, 10×)
- Pheromone channel toggles (show/hide per channel)
- Colony stats panel: food throughput, agent count, threat count, total wiring cost
- Generation counter and fitness history chart
- Module size distribution heatmap per agent (optional, debug mode)

---

## Acceptance Criteria

- Colony of 50–100 agents runs at 30+ fps on canvas.
- Pheromone fields visibly guide forager trails toward food sources.
- Alarm pheromone visibly triggers behavioral mode shifts (agent movement pattern changes within 1–2 ticks of alarm contact).
- By adult stage, forager agents have measurably larger chemosensory processing modules than soldier agents from the same colony DNA.
- By adult stage, soldier agents have measurably larger threat-response modules than forager agents from the same colony DNA.
- Colony fitness improves over generations (food throughput increases; survival fraction increases).
- Colonies with wiring economy pressure develop more compact agent networks than colonies without it, with no significant fitness loss.
- Polyandric reproduction produces measurably more diverse module size distributions across a worker cohort than parthenogenetic reproduction.
- Total pheromone field memory cost stays within slab budget (no per-cell JS object allocation).

---

## Readiness Checklist (for implementation start)

- [ ] NGE Phase G prerequisites met (stigmergy field infrastructure implemented).
- [ ] All Phase 0 computation motif primitives implemented and opt-in verified.
- [ ] Polyandric reproduction mode (Phase E) implemented.
- [ ] Multi-agent evaluation harness supports shared pheromone field state.
- [ ] Canvas rendering approach decided (raw 2D context vs. WebGL overlay).
- [ ] Pheromone diffusion/decay inner loop benchmarked (must not dominate frame budget).
- [ ] Colony-level fitness aggregation contract written.
- [ ] Deterministic episode seeding contract written (same seed → same food placement + threat emergence).
- [ ] Agent sensory channel normalization contract written.
- [ ] Role detection heuristic for canvas visualization defined (module size distribution or explicit signal).
