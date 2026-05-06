# NEAT Genesis EvoDevo: Ant Hive Ecosystem Demo

**Status:** [PLANNED]

This plan defines the ant-hive ecosystem benchmark for [NEAT Genesis EvoDevo (NGE)](NEAT_Genesis_EvoDevo.md). It is the primary stress test for collective intelligence, stigmergy, role differentiation from identical DNA, and neuromodulation-driven behavioral switching — all rendered as a live web canvas simulation.

This benchmark is downstream of [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md), [Memory_Optimization.md](Memory_Optimization.md), and [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md). If this plan conflicts with any upstream plan, the upstream plan wins.

---

## Scope and Maturity

This is a benchmark-architecture plan, not an implementation-complete spec.

- **In scope:** world design, GeoFront mechanics, food system, pheromone field, Angel patrol system, agent roles and sensory channels, caste differentiation model, NGE feature mapping, web worker architecture, canvas simulation spec, and acceptance criteria.
- **Out of scope (for now):** exact physics constants, final reward weights, and GeoFront repair animation details.
- **Authority rule:** [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) and [Memory_Optimization.md](Memory_Optimization.md) remain authoritative.

---

## Why This Demo

The ant hive is the most direct embodiment of the two biological inspirations behind NGE:

1. **DNA as a program:** every ant in the colony shares a nearly identical genome. The vastly different behavior of foragers, workers, and soldiers emerges not from different DNA, but from different developmental experience under identical instructions. This is the "Experience gates where capacity grows" principle made observable.

2. **Metabolically optimal specialized intelligence:** ant brains are among the most efficiently wired nervous systems known. The mushroom bodies store associative memories; the central complex handles navigation; neuromodulation switches behavioral modes. NGE's `EpisodicSlot`, `GatedRecurrentCell`, and `ModulatorBroadcaster` archetypes are direct engineering analogs.

The ant hive also introduces stigmergy — the mechanism by which complex collective behavior emerges from agents leaving traces in a shared environment, with no direct agent-to-agent messaging required.

**Why the maze layout** (vs. open world): the quarter-symmetric maze topology from the Predator/Prey demo is a natural fit for stigmergy. Corridor structure forces foragers into single-file routes where pheromone trails are unambiguous directional signals — the same reason real ant trails concentrate in corridors rather than diffusing across open terrain. Reusing the existing maze infrastructure reduces implementation risk while giving the ant hive a richer, more structured environment than a flat grid.

---

## Thematic Identity — The NGE Double Meaning

The project name **NEAT Genesis EvoDevo** abbreviates to **NGE** — which is also **Neon Genesis Evangelion**. The Predator/Prey demo introduces this pun; the ant hive makes it literal at the colony scale.

- **Colony (ants) → EVA Units** — NERV's defenders, each designated with an EVA unit number, protecting the GeoFront from invasion.
- **Threats → Angels** — the attacking entities, named from real Judeo-Christian angelology as used in the anime.
- **GeoFront → NERV HQ** — the center box is the GeoFront: the hidden underground fortress that must be defended at all costs. Its walls can be damaged by Angels; if the interior shrinks too far, the colony collapses.

The arms race the demo measures (caste specialization vs. Angel flanking via tunnels) maps onto the show's strategic tension: Angels probing defenses from every angle, NERV adapting its deployment of pilots and units.

**Constants shared with the Predator/Prey demo:**

The `EVA_UNITS` and `ANGELS` rosters in `constants/constants.theme.ts` are identical across both demos. The ant hive shares the same color palette, same angel name list, and the same GeoFront amber border (`#ff6600`). EVA designation cycles are assigned per ant at birth and are stable for the life of the episode.

Refer to the **Thematic Identity** section in [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) for the full EVA and Angel roster tables and `constants/constants.theme.ts` source.

---

## NGE Capabilities Exercised

| NGE capability                           | How the ant hive exercises it                                                                                        |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Role differentiation from identical DNA  | Foragers, workers, and soldiers develop from the same DNA; caste emerges from experience-gated structural plasticity |
| Stigmergy (shared typed-array field)     | 6-channel pheromone grid is the sole inter-agent coordination medium                                                 |
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
- Pheromone field infrastructure (typed-array grid, diffusion/decay ops, agent read/write interface) is implemented.
- The Predator/Prey demo maze infrastructure (`examples/predator_prey/maze/`) is implemented — the ant hive reuses it directly.

## Recommended agent + skill combo for this Phase G benchmark

- Benchmark architecture, colony systems, and rollout work — `NGE Benchmark Scout` + `nge-benchmark-workflow`
- Upstream NGE prerequisite drift — `NGE Core Scout` + `nge-core-algorithm`
- Canvas, layout, and interaction polish — `Visualizer Scout` + `visualizer-workflow`

---

## Design Pillars

- **Identical DNA, divergent structure:** all agents in a colony share the same DNA. Structural divergence between castes must emerge from experience-gated plasticity alone, never from DNA divergence within a colony.
- **Pheromone-only coordination:** agents coordinate exclusively through chemical field signals deposited in maze corridors. No direct agent-to-agent messaging, no voice channel (ants don't scream).
- **Colony-level fitness:** individual ant performance is not the selection target. Colony survival, food throughput, and threat response are the fitness signals.
- **Visible specialization:** the benchmark produces observable differences in module size distributions between forager, worker, and soldier agents by adult stage.
- **Metabolic honesty:** total colony wiring cost is a secondary selection pressure. Dense, unspecialized wiring should lose to compact, role-specialized wiring over generations.
- **GeoFront as existential pressure:** Angels can permanently shrink the GeoFront interior. Each breach reduces colony capacity. A colony that fails to defend faces a population death spiral.
- **Exploration pressure without starvation:** food pellets have a very slow respawn (PELLET_RESPAWN_TICKS ≈ 2000). Nearby corridors deplete first, forcing foragers further out over time — but food never runs out permanently.
- **Canvas-native:** the full simulation runs in a browser canvas at interactive rates with a colony of at least 50–100 agents.

---

## World Design

### Grid and Canvas

Identical to the Predator/Prey demo:

- **Full-screen canvas:** fills the real available viewport (`containerElement.getBoundingClientRect()`). Responsive — rescales when the window resizes.
- **Square world:** 1:1 aspect ratio. Cell size = `floor(min(viewportW, viewportH) / gridCells)`.
- **Grid size:** determined at runtime to fill the available square. Always odd dimension (e.g. 31×31).
- **Theme:** neon arcade — dark `#000` background, neon blue (`#00bfff`) double-line Unicode walls.

### Maze Generation (Quarter-Symmetric Procedural)

**Identical algorithm to the Predator/Prey demo** — quarter-generated, 3-fold mirrored, recursive backtracker with `p_extra ≈ 0.15` extra connections, 1-cell-wide corridors enforced. See [NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) for the full generation spec.

**Reuse:** `examples/predator_prey/maze/` in its entirety — generator, renderer, movement, vision, and tunnel modules are copy-pasted without modification.

**Fixed constants (identical to predator/prey, semantically re-labeled):**

- **Center GeoFront box:** filled rectangle at the exact center — this is NERV HQ. Interior is the ant colony's protected zone. Has one 2-cell-wide exit on its bottom edge. Rendered with amber border (`#ff6600`). Interior size determines colony population cap and food storage capacity.
- **Edge tunnels:** one tunnel opening at the center of each of the 4 edges (N/E/S/W). Tunnel transit costs 1 extra tick to prevent oscillation. Angels use tunnels to flank — see Angel System below.
- **Outer border:** all edge cells that are not tunnel openings are walls.

**Seeding:** maze is seeded per run. Same seed → same maze across all episodes of that run.

---

## GeoFront Mechanics

The GeoFront is not just a spawn point — it is a damageable asset whose degradation directly threatens colony survival.

### Interior Zone

The GeoFront interior is the set of open cells within the center box. At episode start, interior size = `GEOFRONT_INITIAL_CELLS` (configurable, e.g. 25 cells for a 5×5 box).

Interior cells provide:

- Spawn locations for new ants (up to colony population cap)
- Food storage (colony accumulates harvested pellets here)
- Nest scent anchor (workers maintain `nestScent` pheromone inside)
- Safety (Angels cannot enter the interior — only the outer wall ring)

### Damageable Walls (Option B)

The GeoFront has an outer wall ring (the fixed boundary cells of the center box). When an Angel reaches any outer wall cell and is not intercepted by a soldier within `ANGEL_ATTACK_DELAY_TICKS`, it damages the wall:

1. **Wall shrinks inward by one cell** along the breach axis. The interior loses one row or column of cells.
2. `GEOFRONT_INTERIOR_CELLS` decreases by the width of the lost row/column.
3. `MAX_COLONY_SIZE` is recalculated: `floor(GEOFRONT_INTERIOR_CELLS × ANTS_PER_CELL)`.
4. If the current colony population exceeds the new cap, excess ants are immediately removed (nearest-to-wall first — they get squeezed out).
5. Food storage capacity also shrinks proportionally: `floor(GEOFRONT_INTERIOR_CELLS × FOOD_PER_CELL)`.

**Spiral death condition:** damage → smaller colony → fewer soldiers → more Angels reach the wall → faster damage. The colony must maintain soldier coverage at the GeoFront perimeter to break this spiral.

**Worker repair:** workers inside the GeoFront can deposit repair actions. Each repair costs food from storage and restores one cell to the wall ring (up to the original boundary). Repair is slow — `REPAIR_TICKS_PER_CELL` ticks of continuous worker presence required. This gives workers a decisive role and creates a food-vs-repair trade-off (every food unit spent on repair is not available for colony growth).

**Collapse threshold:** if interior size drops below `MIN_VIABLE_CELLS` (e.g. 4 cells), the colony cannot sustain itself — the episode ends immediately with zero fitness.

### Visual Representation

- **Intact wall:** amber border (`#ff6600`), same as predator/prey ghost box.
- **Damaged section:** the wall segment is replaced by a dimmer amber (`#883300`) with a 2 px gap — visually indicates breach.
- **Repair in progress:** pulsing amber at the repair site.
- **Interior fill:** dark blue tint (`#00003380`) scales opacity with food storage level — fuller = brighter glow.

---

## Food System

### Pellets

- **Distribution:** on episode start, pellets fill every open non-tunnel, non-GeoFront-interior cell. Density is uniform across the maze.
- **Very slow respawn:** `PELLET_RESPAWN_TICKS ≈ 2000` (vs. 300 in the Predator/Prey demo). A depleted cell goes dark for ~66 seconds at 30 fps before replenishing.
- **Depletion gradient:** cells near the GeoFront exit are visited first. As nearby corridors deplete, foragers must venture further into the maze. This creates natural distance-based exploration pressure without requiring hardcoded role assignments.
- **No starvation:** the slow respawn ensures food never runs out permanently. The colony can always reach food, but must travel further to get it as nearby supplies deplete.

### Harvesting and Deposit

- Forager steps onto a cell with a pellet → carries it (one unit at a time; binary carrying state).
- Forager returns to GeoFront interior → deposits; colony food storage +1.
- Colony food storage is capped by `floor(GEOFRONT_INTERIOR_CELLS × FOOD_PER_CELL)`.
- Food storage drains at a slow maintenance rate (`MAINTENANCE_FOOD_RATE` per tick × colony size) — the colony must sustain a positive food throughput to survive long episodes.

---

## Angel System

### Scripted Patrol Routes

Angels are **scripted adversaries** — they follow deterministic patrol routes, not NEAT-evolved behavior. This is deliberate: the colony's challenge is not to out-evolve an adaptive opponent, but to develop effective collective defense against a known threat pattern.

Each Angel patrols a corridor route through the maze. Routes are computed at episode start from the maze structure (seeded — same seed → same Angel routes). Route planning uses depth-first corridor traversal from the Angel's starting position, biased toward the GeoFront.

**Angel patrol behaviors:**

- Move along assigned corridor route, reversing at dead ends.
- When corridor intersects a junction, choose the branch that minimizes distance to GeoFront.
- On entering a tunnel cell, transit to the opposite edge tunnel (same 1-tick cooldown as agents).
- If a soldier ant is adjacent, stop and engage (Angel is neutralized after `ANGEL_COMBAT_TICKS` of soldier contact).
- If no soldier intercepts within `ANGEL_ATTACK_DELAY_TICKS` of reaching a GeoFront wall cell, trigger wall damage.

### Tunnel Flanking

Angels that reach a tunnel entrance follow through to the opposite edge — this is not random; flanking Angels are assigned tunnel-transit routes by the route planner at episode start. This forces the colony to:

- Maintain soldier coverage at all 4 tunnel entrances simultaneously.
- Use `recruitmentPheromone` to rally soldiers to threatened tunnel zones.
- Develop perimeter-awareness soldiers (soldiers that monitor tunnel proximity) through experience-gated specialization.

A colony that only defends the GeoFront exits will be flanked repeatedly. A colony that only monitors tunnels will have Angels walk in through the main corridors. Effective defense requires role-specialized perimeter coverage.

### Angel Population

- `ANGEL_COUNT` Angels are active per episode (configurable, e.g. 5–10).
- Angels are assigned names from the ANGELS roster in order (same as predator/prey).
- Angel neon color and rhombus silhouette (hollow rotated square, 3 px stroke) identical to the Predator/Prey demo.
- Angels respawn at their episode-start positions after being neutralized, with a `ANGEL_RESPAWN_DELAY_TICKS` cooldown.

### Optional Upgrade: Evolved Angels

After the Predator/Prey demo is implemented, Angel genomes from the predator population's hall of fame can be imported and used in place of scripted patrol routes. This gives the ant hive demo adaptive adversaries and increases the evolutionary pressure on the colony.

This is a late feature — it requires the Predator/Prey demo to be complete and its rolling opponent snapshot format to be compatible with the ant hive episode runner. Mark as `[FUTURE]` on the readiness checklist.

---

## Pheromone Field System

The pheromone system is the primary inter-agent coordination substrate. It is implemented as 6 typed-array grids (one per channel), all sized `gridWidth × gridHeight × Float32`.

Unlike the simple decay-only chem trails in the Predator/Prey demo, pheromone fields here include **diffusion** — chemical signals spread into adjacent cells over time, creating gradient fields that can guide agents across multiple corridors.

### Channels

| Channel                | Laid by                               | Read by              | Meaning                                                         |
| ---------------------- | ------------------------------------- | -------------------- | --------------------------------------------------------------- |
| `foodTrail`            | Foragers (returning with food)        | Foragers (outbound)  | Guide toward known food sources                                 |
| `nestTrail`            | Foragers (outbound from nest)         | Foragers (returning) | Guide back toward GeoFront                                      |
| `alarmPheromone`       | Any ant under threat                  | All ants             | Signal danger; triggers defense-mode via `ModulatorBroadcaster` |
| `recruitmentPheromone` | Soldiers near Angels                  | Workers, soldiers    | Rally defenders to a location                                   |
| `nestScent`            | Workers in GeoFront interior          | All ants             | Gradient toward home; path integration calibration              |
| `trailQuality`         | Foragers (proportional to food found) | Foragers             | Reinforcement signal for productive trails                      |

### Diffusion and Decay

Each tick, the field applies a 5-point stencil (center + 4 neighbors) to open cells only — walls do not diffuse:

```
field[x][y] = field[x][y] * decayRate + mean(openNeighbors) * diffusionRate
```

`decayRate` and `diffusionRate` are per-channel constants. Diffusion is skipped for wall cells (wall cells are always zero). This creates corridor-confined gradients that naturally follow maze topology — exactly how real ant pheromone chemistry behaves.

The stencil sweep is a single typed-array pass — no per-cell object allocation.

### Hardwired Deposition Rules

Pheromone deposition is **instinctive** — like voice screaming in the Predator/Prey demo, ants cannot choose whether to lay pheromone. The engine applies deposition rules based on observable agent state:

| Condition                                 | Channel deposited                                                             |
| ----------------------------------------- | ----------------------------------------------------------------------------- |
| Carrying food                             | `foodTrail`                                                                   |
| Not carrying food, outside GeoFront       | `nestTrail`                                                                   |
| Angel within sensor range                 | `alarmPheromone`                                                              |
| Soldier role-signal active + Angel nearby | `recruitmentPheromone`                                                        |
| Inside GeoFront interior                  | `nestScent`                                                                   |
| Just deposited food at GeoFront           | `trailQuality` (proportional to distance traveled, i.e. trail was productive) |

NEAT does not evolve whether to deposit — it evolves how to react to the pheromone signals it reads. This eliminates the bootstrap problem: trails are meaningful from generation 1.

### Pheromone Sensor Reads

Each ant reads 5 cells per channel (current cell + N/E/S/W), ignoring wall cells (wall reads = 0):

```
5 reads/channel × 6 channels = 30 pheromone input channels
```

Values are normalized to [0, 1] using `tanh(value / PHEROMONE_NORM_SCALE)`.

---

## Agent Roles and Caste Emergence

Three emerged roles, all developing from the same DNA. **No caste is hardcoded.** The colony DNA encodes a single developmental program that responds differently depending on the environmental context the ant experiences during its juvenile phase.

### Forager

**Primary task:** find food sources, harvest one unit, return to GeoFront, deposit.

**Expected module specialization by adulthood:**

- Dense chemosensory zone (foodTrail + nestTrail channels dominate connection weight)
- `EpisodicSlot` storing food-source locations by corridor topology similarity
- `GatedRecurrentCell` for path integration state (tracking distance and direction from GeoFront)
- Pruned threat-response zone

**Developmental trigger:** hatching and spending early ticks in outer corridors (far from GeoFront). The pheromone gradient at hatching determines what the juvenile phase reinforces.

### Worker

**Primary task:** maintain GeoFront (deposit nestScent, repair wall damage, distribute food to storage).

**Expected module specialization by adulthood:**

- GeoFront proximity and integrity sensing zone
- `EpisodicSlot` for GeoFront-state memory (repair sites, food storage locations)
- Dense nestScent processing
- Pruned foraging trail zone and combat zone

**Developmental trigger:** hatching near or inside the GeoFront interior. Workers spend early ticks in high-nestScent zones.

### Soldier

**Primary task:** detect and intercept Angels; lay recruitmentPheromone to rally allies; patrol tunnel entrances.

**Expected module specialization by adulthood:**

- Dense alarmPheromone and recruitmentPheromone sensing zone
- `ModulatorBroadcaster` output calibrated for fast alarm-to-attack mode switch
- `GatedRecurrentCell` for Angel tracking state (recent movement direction, approach vector)
- Tunnel proximity processing zone (soldiers assigned perimeter patrol develop this)
- Pruned food-trail zone

**Developmental trigger:** hatching near the GeoFront perimeter or tunnel entrances where alarmPheromone and recruitmentPheromone are highest during early episodes.

---

## Sensory Channels (per agent)

All channels are corridor-relative (wall cells read as zero).

### Pheromone senses (30 channels)

- 5 reads per channel (current cell + N/E/S/W) × 6 channels

### Maze senses (9 channels)

- Wall presence in N/E/S/W: 4
- Corridor depth in N/E/S/W (normalized distance to nearest wall): 4
- Tunnel proximity (normalized distance to nearest tunnel entrance): 1

### GeoFront senses (4 channels)

- Inside GeoFront zone (binary): 1
- Distance to GeoFront exit (normalized corridor distance): 1
- GeoFront wall integrity (fraction of original boundary cells intact): 1
- Colony population headroom (normalized remaining cap): 1

### Food senses (3 channels)

- Food at current cell (binary): 1
- Food quantity at current cell (normalized): 1
- Nearest food source distance (normalized corridor BFS distance): 1

### Agent-state senses (7 channels)

- Current heading (sin/cos): 2
- Carrying food (binary): 1
- Energy level (if metabolic budget enabled): 1
- Role-signal (soft: recent-behavior-history mean — not hardcoded): 1
- Ticks since last food deposit (normalized): 1
- Ticks since last Angel contact (normalized): 1

### Neighbor senses (9 channels)

- Angel proximity in N/E/S/W corridors (distance to nearest Angel in each direction, 0=adjacent, 1=none): 4
- Nearest Angel overall (distance + direction sin/cos): 3 (distance + sin + cos)
- Alarm pheromone delta (rate of change this tick at current cell): 1
- Nestmate density in 3-cell corridor radius (normalized): 1

### Short-horizon memory senses (8 channels)

Raw history for `GatedRecurrentCell` integration:

- Recent movement direction (last 4 ticks, sin/cos each): 8

**Total sensory input width: 70 channels**

---

## Movement and Output

- **4 directions only:** North, East, South, West. No diagonals.
- **Network output:** 4 logits → softmax → argmax → direction index.
- Pheromone deposition is hardwired (not a network output) — see Hardwired Deposition Rules above.
- Tunnel traversal: same as predator/prey — step into tunnel cell → exit opposite edge tunnel after 1 tick.

---

## Colony Fitness Signal

Fitness is measured at the **colony level**, not per individual. This forces collective behavior to emerge — an individually high-performing forager in a colony that collapses under threat pressure produces low fitness.

**Primary fitness components:**

1. Food throughput (total units deposited per episode)
2. Colony survival fraction at episode end (proportion of starting agent count surviving)
3. Angel neutralization rate (Angels intercepted before GeoFront contact / total Angels that reached perimeter)
4. GeoFront integrity at episode end (fraction of original boundary cells intact)

**Secondary pressure:**

- Total colony wiring cost (sum of all active agent connection counts) — lower is better, all else equal
- Food delivery distance mean (average corridor distance foragers traveled per deposit) — higher distance rewards exploration under depletion

**Episode structure:**

- Fixed-length episodes (e.g. 3,000 ticks — longer than predator/prey to allow depletion gradients to develop)
- Deterministic food placement and Angel patrol routes per evaluation seed
- Multiple seeds evaluated per generation to reduce luck

---

## Reproduction and Colony Lifecycle

### Within-episode

Ants are not replaced mid-episode when killed. A colony starts with a fixed count (e.g. 60 agents), and deaths from Angels reduce the active count permanently. GeoFront wall damage can shrink the cap below the current population (excess ants are removed).

Workers can repair walls, restoring cap — the food-vs-repair trade-off is a key strategic tension.

### Between generations

The colony's DNA (shared across all ants) is the unit of evolution. Fitness is the colony-level score.

**Reproduction policy for this demo:**

- Default: **polyandric** — queen DNA is the primary template; drone contributions diversify module archetype deltas across the worker cohort. This mirrors real ant colony genetics and is why real colonies produce diverse worker specializations from a single queen genome.
- Parthenogenesis allowed: for producing identical-DNA twin colonies to test role divergence from experience alone.
- Standard sexual: for cross-colony genetic mixing between high-fitness lineages.

---

## Web Worker Architecture

### Why the architecture is simpler than Predator/Prey

The ant hive has a single NEAT population (no two-population generation barrier) and scripted Angels (no rolling opponent snapshot). The worker topology collapses to a simpler form:

```
COORDINATOR (main thread)
├── Colony NEAT Worker  (1) — owns colony gene pool, species, generation counter
├── Simulation Worker   (1) — owns live display episode (world, ants, pheromone, Angels)
└── Episode Workers     (N) — stateless; run isolated training episodes
```

Worker count formula (identical to predator/prey):

```ts
const cores = navigator.hardwareConcurrency ?? 4;
// Reserve: 1 main thread + 1 simulation + 1 NEAT = 3
const EPISODE_WORKER_COUNT = Math.max(2, cores - 3);
```

All workers are pre-spawned at demo start and kept alive.

### Message Protocol

**Coordinator → Colony NEAT Worker:**

```ts
type ColonyNeatWorkerRequest =
  | {
      type: 'init';
      payload: {
        populationSize: number;
        elitismCount: number;
        architectureProfileId: string;
        rngSeed: number;
      };
    }
  | {
      type: 'submit-fitness';
      payload: {
        genomeFitness: Array<{
          genomeId: string;
          fitness: number;
          stats: ColonyEpisodeStats;
        }>;
      };
    }
  | { type: 'evolve' }
  | { type: 'stop' };
```

**Colony NEAT Worker → Coordinator:**

```ts
type ColonyNeatWorkerEvent =
  | {
      type: 'population-ready';
      payload: {
        generation: number;
        population: SerializedGenome[];
        champion: SerializedGenome;
        generationStats: ColonyGenerationStats;
      };
    }
  | { type: 'error'; payload: { message: string } };
```

**Coordinator → Episode Worker:**

```ts
type AntEpisodeWorkerRequest =
  | {
      type: 'run-episode';
      payload: {
        taskId: string;
        colonyGenome: SerializedGenome;
        mazeConfig: MazeConfig;
        seed: number;
        rolloutIndex: number;
        angelCount: number;
      };
    }
  | { type: 'stop' };
```

**Episode Worker → Coordinator:**

```ts
type AntEpisodeWorkerEvent =
  | {
      type: 'episode-done';
      payload: { taskId: string; fitness: number; stats: ColonyEpisodeStats };
    }
  | { type: 'error'; payload: { taskId: string; message: string } };
```

**Coordinator → Simulation Worker:**

```ts
type AntSimulationWorkerRequest =
  | {
      type: 'start-display';
      payload: {
        colonyChampion: SerializedGenome;
        mazeConfig: MazeConfig;
        seed: number;
        angelCount: number;
      };
    }
  | {
      type: 'request-render-step';
      payload: { requestId: number; simulationSteps: number };
    }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'set-speed'; payload: { multiplier: number } }
  | { type: 'stop' };
```

**Simulation Worker → Coordinator:**

```ts
type AntSimulationWorkerEvent =
  | {
      type: 'render-step';
      payload: {
        requestId: number;
        frame: AntHiveRenderFrame;
        done: boolean;
        episodeStats?: ColonyEpisodeStats;
      };
    }
  | { type: 'error'; payload: { message: string } };
```

### Render Frame — Packed SoA Typed Arrays (Zero-Copy Transfer)

```ts
type AntHiveRenderFrame = {
  format: 'ah-packed-v1';
  tick: number;
  antCount: number;
  angelCount: number;
  gridW: number;
  gridH: number;

  // Ant arrays (fixed ordering throughout episode)
  antX: Int16Array; // [antCount]
  antY: Int16Array;
  antDir: Uint8Array; // 0=N 1=E 2=S 3=W
  antAlive: Uint8Array; // 1=alive 0=dead
  antCarrying: Uint8Array; // 1=carrying food 0=empty
  antEvaId: Uint8Array; // EVA roster index (stable for life of episode)

  // Angel arrays (fixed ordering throughout episode)
  angelX: Int16Array; // [angelCount]
  angelY: Int16Array;
  angelAlive: Uint8Array; // 1=active 0=neutralized (cooldown)

  // Pheromone channels — flat row-major [gridW × gridH], one per channel
  phFoodTrail: Float32Array;
  phNestTrail: Float32Array;
  phAlarmPheromone: Float32Array;
  phRecruitmentPheromone: Float32Array;
  phNestScent: Float32Array;
  phTrailQuality: Float32Array;

  // Pellet grid — bit-packed: byte (y*gridW+x)>>3, bit (y*gridW+x)&7
  pellets: Uint8Array;

  // GeoFront state
  geofrontIntegrity: number; // fraction of original boundary intact [0, 1]
  geofrontInteriorCells: number; // current interior cell count
  foodStorage: number; // units stored in GeoFront

  // Episode scalars
  antsAlive: number;
  angelsActive: number;
  foodDeposited: number; // total this episode
  pelletsRemaining: number;
};

// Transfer list (zero-copy):
const transferList = [
  frame.antX.buffer,
  frame.antY.buffer,
  frame.antDir.buffer,
  frame.antAlive.buffer,
  frame.antCarrying.buffer,
  frame.antEvaId.buffer,
  frame.angelX.buffer,
  frame.angelY.buffer,
  frame.angelAlive.buffer,
  frame.phFoodTrail.buffer,
  frame.phNestTrail.buffer,
  frame.phAlarmPheromone.buffer,
  frame.phRecruitmentPheromone.buffer,
  frame.phNestScent.buffer,
  frame.phTrailQuality.buffer,
  frame.pellets.buffer,
];
```

### Generation Lifecycle

No two-population barrier — single population, simpler loop:

```
Generation N lifecycle:

1. COORDINATOR receives 'population-ready' from Colony NEAT Worker.

2. COORDINATOR queues all episode tasks:
   for each colonyGenome × ROLLOUT_SEED_COUNT:
     tasks.push({ type: 'run-episode', genome: colonyGenome, seed: seeds[rolloutIndex], ... })

3. COORDINATOR dispatches tasks to episode worker pool (pre-spawned, idle tracking).

4. COORDINATOR collects EpisodeResults.
   → aggregates per genome: mean(fitness) − STABILITY_WEIGHT × stddev(fitness) across seeds

5. When ALL genomes have ROLLOUT_SEED_COUNT results:
   → COORDINATOR sends 'submit-fitness' + 'evolve' to Colony NEAT Worker
   → Coordinator sends new champion to Simulation Worker (hybrid mode)

6. Colony NEAT Worker replies 'population-ready' → Generation N+1 begins.
```

### Performance Optimizations

All 10 optimizations from the Predator/Prey plan apply here (offscreen maze canvas, ImageData trail render, slab forward passes, observation vector pool, typed-array decay sweep, single-flight render requests, generation-0 warm-start). Three ant-hive-specific additions:

**11. Pheromone diffusion as a masked typed-array sweep**
Pre-compute a `wallMask: Uint8Array` (1=open, 0=wall) at maze generation time. Apply as:

```ts
for (let i = 0; i < fieldSize; i++) {
  if (!wallMask[i]) continue;
  // 5-point stencil using precomputed neighbor index table
  field[i] =
    field[i] * decayRate + stencilMean(field, neighborTable[i]) * diffusionRate;
}
```

`neighborTable` is a `Int32Array[fieldSize × 4]` precomputed at episode start, mapping each open cell to its 4 neighbor indices (wall cells map to self, contributing zero gradient). One pass per channel per tick.

**12. Pheromone render via 6-channel ImageData blend**
Pre-allocate one `ImageData (RGBA, gridW×gridH)`. Each tick, iterate all 6 Float32Array channel grids and accumulate into RGBA using per-channel tint constants. One `ctx.putImageData()` call renders all overlays combined.

**13. Angel patrol pre-computation**
Angel patrol routes (including tunnel transits) are computed once at episode start as a `Int16Array[ANGEL_COUNT × MAX_ROUTE_LENGTH]` buffer of cell indices. Each tick, index into this buffer — no per-tick pathfinding needed.

---

## Canvas Simulation Spec

### Rendering

| Element                        | Visual                                                                                                                |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Walls                          | Neon blue (`#00bfff`) double-line Unicode box chars on black                                                          |
| Corridors                      | Dark background; floor dots at low opacity                                                                            |
| Pellets                        | Small cyan dots (`#00ffff`, 2 px radius)                                                                              |
| Depleted pellet site           | Faint grey dot (respawn timer visible as opacity)                                                                     |
| EVA Units (ants)               | Filled arc (Pac-Man mouth) in designated EVA neon color; mouth opens toward movement direction; carrying=filled mouth |
| Angels                         | Hollow rotated-square rhombus silhouette (3 px stroke) in designated angel neon color                                 |
| GeoFront box (intact)          | Amber border (`#ff6600`); interior dark blue tint scales with food storage                                            |
| GeoFront wall (damaged)        | Dimmer amber (`#883300`) with visual gap at breach site                                                               |
| GeoFront repair pulse          | Bright amber pulse at active repair site                                                                              |
| `foodTrail` overlay            | Green tint (`#00ff0020`), intensity proportional to concentration                                                     |
| `nestTrail` overlay            | Blue tint (`#0000ff20`)                                                                                               |
| `alarmPheromone` overlay       | Red tint (`#ff000040`) — highly visible; alarm is urgent                                                              |
| `recruitmentPheromone` overlay | Orange tint (`#ff880020`)                                                                                             |
| `nestScent` overlay            | Dim blue (`#0033ff10`) — background gradient                                                                          |
| `trailQuality` overlay         | Bright gold (`#ffdd0020`) — highlights productive trails                                                              |
| Tunnels                        | Pulsing cyan glow at 4 edge openings                                                                                  |
| Agent label                    | EVA ID or Angel name at 8 px above agent when hovered (toggle-able)                                                   |

### UI Controls

- Play / Pause / Step
- Speed multiplier (1×, 2×, 5×, 10×)
- **Colony stats panel:** food deposited this episode, ants alive, Angels neutralized, GeoFront integrity %, food storage level
- **Pheromone channel toggles:** show/hide each of the 6 channels independently
- **Generation counter** and fitness history chart (colony mean fitness over generations)
- **Caste distribution chart:** fraction of colony exhibiting forager / worker / soldier role-signal per generation
- **Wiring cost chart:** mean active connection count per agent over generations
- **Genetic diversity chart:** population species count and mean genomic distance over generations (polyandric diversity visible here)

### Performance Targets

- 50–100 ants + 5–10 Angels at 30+ fps in display mode.
- Pheromone diffusion: 6 masked typed-array sweeps per tick, no per-cell object allocation.
- Agent evaluation: slab-backed flat-array forward passes (`multi.activateSerializedNetwork`).
- Training throughput: episode worker pool saturates all available CPU cores.

---

## Implementation Files

### New demo folder: `examples/ant_hive/`

```
examples/ant_hive/
  browser-entry/
    browser-entry.spawn.utils.ts          ← adapted from predator_prey/browser-entry/
    host/
      host.ts                             ← canvas setup, responsive resize (shared pattern)
      host.types.ts
  maze/                                   ← COPY from examples/predator_prey/maze/ without modification
    maze.generator.ts
    maze.generator.types.ts
    maze.renderer.ts
    maze.movement.ts
    maze.vision.ts
    maze.tunnels.ts
    maze.pellets.ts                       ← adapted: PELLET_RESPAWN_TICKS override, depletion gradient seeding
  geofront/
    geofront.state.service.ts             ← interior cell tracking, integrity score, cap recalculation
    geofront.damage.service.ts            ← wall shrink logic, collapse detection, excess ant removal
    geofront.repair.service.ts            ← worker repair accumulation, wall restoration
    geofront.renderer.ts                  ← intact/damaged/pulse visual state
    geofront.types.ts
  signals/
    signals.pheromone.service.ts          ← 6 Float32Array grids, masked diffusion sweep, deposition rules
    signals.pheromone.renderer.ts         ← 6-channel ImageData blend, per-channel tint constants
    signals.types.ts
  angels/
    angels.patrol.service.ts              ← pre-computes patrol routes at episode start, tunnel transit
    angels.combat.service.ts              ← Angel vs soldier engagement, neutralization, respawn
    angels.renderer.ts                    ← rhombus silhouette, per-angel color, alive/neutralized state
    angels.types.ts
  environment/
    environment.state.service.ts          ← episode state: ants, angels, pheromone grids, geofront, tick
    environment.step.service.ts           ← tick: all ant forward passes, pheromone update, angel patrol, combat
    environment.types.ts
  agents/
    ant.sensor.service.ts                 ← builds Float32Array(70) ant observation vector per tick
    ant.deposition.service.ts             ← hardwired pheromone deposition rules by agent state
    ant.caste.detection.service.ts        ← heuristic caste label from module size distribution (debug/display)
    agents.types.ts
  workers/
    workers.types.ts                      ← all message type unions (colony neat + episode + simulation workers)
    workers.coordinator.ts                ← generation lifecycle, task queue, mode management
    workers.pool.ts                       ← episode worker pool (pre-spawned, idle tracking, task dispatch)
    workers.worker-count.utils.ts         ← hardwareConcurrency → worker count formula
    simulation-worker/
      simulation-worker.ts                ← entrypoint, mutable state bag, message routing
      simulation-worker.runtime.service.ts ← init, champion genome install, maze + geofront + angels setup
      simulation-worker.step.service.ts   ← per-tick: env step + forward passes + pheromone update
      simulation-worker.snapshot.utils.ts ← AntHiveRenderFrame packing + transfer list
      simulation-worker.offscreen.service.ts ← draw maze once to OffscreenCanvas; blit each tick
      simulation-worker.types.ts
    episode-worker/
      episode-worker.ts                   ← entrypoint, stateless
      episode-worker.episode.service.ts   ← runs one complete isolated episode (all ticks, all agents)
      episode-worker.fitness.service.ts   ← colony fitness aggregation across rollout seeds
      episode-worker.warm-start.service.ts ← gen-0 heuristic teacher: follow food trail / flee alarm pheromone
      episode-worker.types.ts
    neat-worker/
      neat-worker.colony.ts               ← entrypoint with colony config
      neat-worker.evolution.service.ts    ← submit fitness, evolve, emit population-ready
      neat-worker.types.ts
  constants/
    constants.maze.ts                     ← grid size, cell size, PELLET_RESPAWN_TICKS (≈2000)
    constants.geofront.ts                 ← initial cells, min viable, ants/cell, food/cell, repair rate, attack delay
    constants.angels.ts                   ← angel count, respawn delay, combat ticks, patrol bias constants
    constants.pheromone.ts                ← decayRate, diffusionRate, emitAmount per channel; normalization scale
    constants.agents.ts                   ← population size, episode length (≈3000 ticks)
    constants.fitness.ts                  ← fitness weights, stability weight, rollout seed count
    constants.workers.ts                  ← rollout count, worker count formula
    constants.theme.ts                    ← IMPORT from predator_prey/constants/constants.theme.ts (or copy-paste)
```

### Source Reuse

**From `examples/predator_prey/`** (copy-paste, adapt — self-contained):

| Source                                                     | Reuse level | Target                                                                                |
| ---------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------- |
| `maze/*` (all 7 files)                                     | ~100%       | `maze/*` — identical algorithm, `maze.pellets.ts` needs PELLET_RESPAWN_TICKS override |
| `browser-entry/host/host.ts`                               | ~95%        | `browser-entry/host/host.ts` — canvas setup is identical                              |
| `environment/environment.state.service.ts`                 | ~50%        | Extend: add geofront, pheromone grids, angel state                                    |
| `workers/workers.pool.ts`                                  | ~95%        | Identical pool pattern, smaller worker count                                          |
| `workers/workers.coordinator.ts`                           | ~60%        | Simplified: single NEAT worker, no generation barrier, no snapshot                    |
| `simulation-worker/simulation-worker.offscreen.service.ts` | ~95%        | Identical offscreen maze blit pattern                                                 |
| `episode-worker/episode-worker.warm-start.service.ts`      | ~40%        | New teachers: follow foodTrail if not carrying, flee alarmPheromone                   |
| `constants/constants.theme.ts`                             | ~100%       | Shared EVA + Angel roster — copy-paste                                                |

**From `examples/flappy_bird/`**:

| Source                                      | Reuse level | Target                                                                           |
| ------------------------------------------- | ----------- | -------------------------------------------------------------------------------- |
| `flappy-evolution-worker.types.ts`          | ~70%        | `workers/workers.types.ts` — simplify to 1 NEAT worker                           |
| `flappy-evolution-worker.snapshot.utils.ts` | ~40%        | `simulation-worker.snapshot.utils.ts` — extend for pheromone channels + geofront |

---

## Acceptance Criteria

- Colony of 50–100 ants + 5–10 Angels runs at 30+ fps in display mode.
- Pheromone field gradients are visibly concentrated in productive foraging corridors after 200+ episode ticks.
- Alarm pheromone visibly triggers behavioral mode shifts — agents in alarm zones exhibit measurably different movement patterns than baseline.
- GeoFront wall damage occurs when soldiers fail to intercept Angels. Damage is visually obvious (amber gap on the border).
- Worker repair restores damaged wall sections when food storage is sufficient.
- Colony fitness improves over generations (food throughput increases; GeoFront integrity at episode end increases).
- By adult stage, forager agents have measurably larger foodTrail/nestTrail processing zones than soldier agents from the same colony DNA.
- By adult stage, soldier agents have measurably larger alarmPheromone/recruitmentPheromone processing zones than forager agents from the same colony DNA.
- Tunnel-flanking Angels succeed more often in colonies without perimeter patrol than in colonies that develop tunnel-coverage soldiers.
- Colonies with wiring economy pressure develop more compact agent networks than colonies without it, at no significant fitness cost.
- Polyandric reproduction produces measurably more diverse module size distributions across a worker cohort than parthenogenetic reproduction.
- Total pheromone field memory cost stays within slab budget (no per-cell JS object allocation).
- Full-screen canvas fills available viewport and rescales correctly on window resize.

---

## Readiness Checklist (for implementation start)

**NGE prerequisites:**

- [ ] NGE Phase G prerequisites met (stigmergy field infrastructure implemented).
- [ ] All Phase 0 computation motif primitives implemented and opt-in verified.
- [ ] Polyandric reproduction mode (Phase E) implemented.
- [ ] Multi-agent evaluation harness supports shared pheromone field state.
- [ ] Predator/Prey demo maze infrastructure (`examples/predator_prey/maze/`) implemented.

**Environment:**

- [ ] `maze/` — copied from predator_prey, `maze.pellets.ts` updated with PELLET_RESPAWN_TICKS ≈ 2000.
- [ ] GeoFront state, damage, and repair services implemented and unit-tested.
- [ ] GeoFront collapse detection (interior below MIN_VIABLE_CELLS) working.
- [ ] Food system: harvest, deposit, storage cap, maintenance drain implemented.
- [ ] Pheromone field: 6-channel diffusion sweep (masked by wall) implemented and unit-tested.
- [ ] Hardwired deposition rules by agent state implemented.
- [ ] Ant sensor service (70 inputs) implemented and unit-tested.
- [ ] Angel patrol pre-computation (including tunnel transits) implemented and unit-tested.
- [ ] Angel combat (engagement timer, neutralization, respawn) implemented.
- [ ] Deterministic episode seeding (same seed → same maze + same Angel routes) verified.

**Worker architecture:**

- [ ] `workers.types.ts` — all message type unions written and reviewed.
- [ ] Episode worker pool — pre-spawned, idle tracking, task dispatch.
- [ ] Episode worker — isolated episode runner, slab forward passes, observation vector pool.
- [ ] Generation-0 warm-start — follow foodTrail teacher + flee alarmPheromone teacher verified.
- [ ] Colony fitness aggregation — mean − STABILITY_WEIGHT × stddev across ROLLOUT_SEED_COUNT seeds.
- [ ] Colony NEAT worker — emit `population-ready`, receive `submit-fitness`, evolve.
- [ ] Simulation worker — display episode, offscreen maze canvas, per-tick SoA render frame packing.
- [ ] Render frame transfer list — all typed array buffers transferred (zero-copy) each tick.
- [ ] Single-flight render request — at most 1 in-flight `request-render-step` at a time.
- [ ] Operation mode switching (training-only / display-only / hybrid) working correctly.

**Rendering and UI:**

- [ ] GeoFront renderer (intact/damaged/repair pulse/food storage glow) implemented.
- [ ] Pheromone 6-channel ImageData renderer (per-channel tint blend, single putImageData) implemented.
- [ ] Angel renderer (rhombus silhouette, per-angel color, neutralized state) implemented.
- [ ] Ant carrying-state visual (filled vs. open mouth arc) implemented.
- [ ] Pheromone channel toggles (each of 6 independently toggleable) working.
- [ ] Colony stats panel, fitness history chart, caste distribution chart, wiring cost chart implemented.
- [ ] Full-screen responsive canvas implemented.

**Verification:**

- [ ] Role emergence verified: forager module weights measurably different from soldier module weights by generation 50.
- [ ] Tunnel-flanking pressure verified: ablation (disable tunnel transit for Angels) shows higher colony survival.
- [ ] Pheromone effectiveness verified: ablation (disable pheromone input channels) shows lower food throughput.
- [ ] GeoFront death spiral verified: colony that loses wall segment shows accelerated fitness decline.
- [ ] 30+ fps confirmed in hybrid mode with 80+ ants + 8 Angels on display canvas.
- [ ] Episode worker throughput scales with core count.

**Future (post-implementation):**

- [ ] `[FUTURE]` Evolved Angel genome import from Predator/Prey demo hall-of-fame — requires compatible SerializedGenome format and predator episode runner interface.
