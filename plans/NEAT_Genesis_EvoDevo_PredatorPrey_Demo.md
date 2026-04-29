# NEAT Genesis EvoDevo: Predator/Prey Co-evolution Demo

**Status:** [PLANNED]

This plan defines the predator/prey co-evolutionary benchmark for [NEAT Genesis EvoDevo (NGE)](NEAT_Genesis_EvoDevo.md). It is the primary stress test for co-evolutionary dynamics, sensory arms race emergence, structural divergence under non-stationary selection pressure, and reproduction mode diversity — all rendered as a live web canvas simulation.

This benchmark is downstream of [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) and [Memory_Optimization.md](Memory_Optimization.md). If this plan conflicts with either upstream plan, the upstream plan wins.

---

## Scope and Maturity

This is a benchmark-architecture plan, not an implementation-complete spec.

- **In scope:** world design, two-population co-evolutionary structure, prey and predator sensory channels, arms race observables, NGE feature mapping, canvas simulation spec, and acceptance criteria.
- **Out of scope (for now):** exact physics constants, rendering library selection, final reward weights, and worker protocol shape.
- **Authority rule:** [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) and [Memory_Optimization.md](Memory_Optimization.md) remain authoritative.

---

## Why This Demo

The predator/prey arms race is one of the oldest and most productive experimental paradigms in evolutionary computation. It is uniquely valuable for NGE because:

- **Non-stationary fitness landscape:** neither population has a fixed target. As prey evolve better camouflage, predators must evolve better detection — and vice versa. This is the most direct test of whether NGE's assimilation mechanism can track a moving target without catastrophic forgetting.
- **Structural divergence under pressure:** prey and predators start from similar base DNA but should diverge dramatically in `computationType` composition over generations — prey growing larger perceptual integration and evasion modules; predators growing larger detection and pursuit modules.
- **Reproduction mode selection:** in stable pursuit-dominated phases, predators may converge toward parthenogenesis (preserve pursuit DNA). In phases where prey develop novel evasion, predators may shift toward polyandric or sexual reproduction for diversity.
- **Multi-population NEAT:** running two independent NEAT populations with shared evaluation harness is a prerequisite for the ant-hive multi-colony extension.

---

## NGE Capabilities Exercised

| NGE capability                          | How the predator/prey demo exercises it                                                                                     |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Co-evolutionary dynamics                | Two populations with independent NEAT + assimilation cycles; fitness computed against rolling opponent snapshot             |
| Non-stationary fitness landscape        | Structural arms race: predator detection vs. prey camouflage/evasion cycles                                                 |
| `computationType` structural divergence | Prey grow larger perceptual zones; predators grow larger pursuit-planning zones                                             |
| `ModulatorBroadcaster`                  | Threat detection triggers fast evasion-mode gain shift in prey; contact detection triggers pursuit-commit mode in predators |
| `GatedRecurrentCell`                    | Pursuit state (predators); evasion trajectory state (prey)                                                                  |
| `EpisodicSlot`                          | Prey store predator encounter patterns by similarity; predators store prey evasion patterns                                 |
| `reproductionPolicy`                    | Mode should shift as arms race phases change; tests `modeIsEvolvable: true`                                                 |
| Wiring economy                          | Both populations are under size pressure; compact specialists should outcompete bloated generalists                         |
| Sensory arms race                       | Observable structural metric: predator chemosensory/visual zone size grows as prey evolve camouflage                        |

---

## Execution Alignment

This benchmark belongs to **Phase G (Multi-Agent + Collective Intelligence)** in the NGE roadmap. It must not begin serious implementation before:

- NGE Phase A (DNA + deterministic development) and Phase B (Juvenile focus) are stable.
- All Phase 0 computation motif primitives are implemented and opt-in verified.
- Phase E (Evolution integration + reproduction modes) is implemented.
- Co-evolutionary evaluation harness (two-population with rolling opponent snapshot) is implemented.

---

## Design Pillars

- **Two genuinely independent populations:** prey and predators each have their own DNA gene pool, NEAT species tracking, and assimilation cycle. They are not variants of a single population.
- **Chemical and visual sensory systems:** prey emit a detectable chemical trail (like scent); predators detect it. Prey can suppress emission at a cost (metabolic budget) — this is the camouflage mechanism. Predators can evolve more sensitive detection — this is the arms race driver.
- **Evasion vs. pursuit as structural specialization:** the primary observable is that the two populations develop measurably different `computationType` module compositions.
- **No scripted behaviors:** all pursuit, evasion, camouflage, and alarm strategies emerge from the evolved network structure and `ModulatorBroadcaster` gain calibration.
- **Arms race phases are observable:** the simulation UI should show the structural arms race metric over generations (predator detection zone size vs. prey emission suppression level).
- **Canvas-native at scale:** at least 20–40 agents per population running at interactive rates in a browser canvas.

---

## World Design

### Environment

A 2D continuous or grid world rendered on HTML canvas. Recommended: 256×256 grid cells or a continuous 1024×1024 coordinate space with 4:1 scaling.

**World zones:**

- **Prey habitat:** dense vegetation patches (traversable by prey at full speed; predators slow down). Prey favor these zones for cover.
- **Open terrain:** faster movement for both; prey are more exposed here.
- **Predator territory:** border or specific zones where predators spawn / have home advantage.
- **Resource patches:** food nodes that sustain prey energy. Prey must forage to survive; predators must hunt to survive.

### Energy and Survival

Both populations operate under a simple energy budget:

- Prey gain energy by reaching food patches; lose energy by moving, especially at high speed.
- Predators gain energy by successfully catching prey; lose energy by moving and by running pursuit bursts.
- Agents that reach zero energy are removed from the episode. Episode ends when all prey or all predators are eliminated, or after a fixed tick limit.

---

## Sensory Arms Race Mechanics

The arms race is driven by two interacting traits, both encoded in DNA and evolvable:

### Prey: chemical emission and suppression

- Prey passively emit a chemical scent signal at a base rate encoded in DNA.
- Prey can actively suppress emission (reduce scent output to near zero) at an energy cost proportional to suppression intensity.
- Suppression is a network output, not a hardcoded behavior — the network decides when the energy cost is worth paying.
- DNA encodes: base emission rate (evolvable), maximum suppression capability (evolvable), suppression energy cost coefficient (evolvable).

### Predators: chemical detection sensitivity

- Predators sense a chemical field (analogous to the pheromone grid in the ant demo, but here it is prey scent).
- Detection sensitivity is encoded in DNA as a gain applied to the raw scent channel inputs.
- Higher sensitivity = better detection of suppressed prey, but also higher noise sensitivity (false positives from terrain residue).
- DNA encodes: detection gain (evolvable), noise-rejection threshold (evolvable).

### Arms race trajectory

Phase 1 (early): prey emit freely; predators detect easily; prey survival depends on evasion speed.
Phase 2: prey that suppress emission survive longer; suppression trait spreads.
Phase 3: predators that evolve higher detection sensitivity counter suppression; detection trait spreads.
Phase 4: prey may evolve motion-stillness switching (move fast when not detected; freeze when detected); predators evolve motion-detection sensitivity.
Phase N: continues indefinitely; the observable structural metric should show oscillating advantage.

---

## Prey Agent Design

### Primary behaviors (all emergent)

- Forage for food while minimizing predator contact
- Suppress scent emission when predator proximity is sensed
- Evasive movement when threatened
- Alarm signaling to nearby prey (chemical broadcast)
- Stillness / camouflage mode when predator is very close

### Expected module specialization by adulthood

- Large perceptual integration zone processing scent gradient + visual proximity signals
- `EpisodicSlot` storing predator encounter contexts by similarity (predator approach patterns)
- `GatedRecurrentCell` for evasion trajectory state (where am I going, how fast, when to change direction)
- `ModulatorBroadcaster` calibrated for fast transition: forage mode → suppress mode → full-evasion mode

### Prey sensory channels

**Chemical senses (18 channels):**

- Predator scent concentration at current cell and 8 surrounding cells (9 channels)
- Alarm pheromone from nearby prey at current cell and 8 surrounding cells (9 channels)

**Visual / proximity senses (14 channels):**

- Ray distances to nearest predator in 8 directions (8 channels)
- Nearest predator heading estimate (2 channels, sin/cos)
- Nearest predator relative speed
- Nearest predator closing rate
- Predator count in local radius

**Terrain senses (8 channels):**

- Vegetation density in 8 surrounding cells (cover quality)
- Food source presence and distance (nearest 3 sources: distance + bearing = 6 channels)
- Open-terrain exposure indicator

**Agent-state senses (8 channels):**

- Current speed
- Current heading (sin/cos)
- Energy level
- Scent emission level (current output)
- Suppression cost being paid (current energy drain rate)
- Time since last predator contact
- Time since last food

**Short-horizon memory senses (6 channels):**

- Recent movement direction history (last 3 ticks, 2 channels each)

**Total prey input width: ~54 channels**

---

## Predator Agent Design

### Primary behaviors (all emergent)

- Detect and pursue prey using chemical gradient and visual proximity
- Manage pursuit energy (burst speed is expensive; long chases deplete energy)
- Coordinate with other predators to cut off evasion paths (emergent, not scripted)
- Switch between patrol (energy-conserving) and pursuit (committed) modes

### Expected module specialization by adulthood

- Large chemical detection zone with high-gain scent gradient processing
- `EpisodicSlot` storing prey evasion pattern contexts (prey tend to repeat evasion trajectories)
- `GatedRecurrentCell` for pursuit state (target position extrapolation, closing rate tracking)
- `ModulatorBroadcaster` calibrated for patrol mode → commit-pursuit mode transition

### Predator sensory channels

**Chemical senses (9 channels):**

- Prey scent concentration at current cell and 8 surrounding cells (9 channels, gain-scaled by DNA)

**Visual / proximity senses (16 channels):**

- Ray distances to nearest prey in 8 directions (8 channels)
- Nearest prey heading estimate (2 channels, sin/cos)
- Nearest prey relative speed and closing rate
- Nearest prey distance
- Prey count in local radius
- Nearest co-predator position (coordination signal, 2 channels)

**Terrain senses (6 channels):**

- Vegetation density in 8 surrounding cells (aggregated to 4 quadrant averages)
- Open-terrain indicator
- Terrain speed modifier at current cell

**Agent-state senses (8 channels):**

- Current speed
- Current heading (sin/cos)
- Energy level
- Pursuit burst available (cooldown indicator)
- Time since last successful catch
- Time since last prey detection

**Short-horizon memory senses (6 channels):**

- Recent movement direction history (last 3 ticks, 2 channels each)

**Total predator input width: ~45 channels**

---

## Co-evolutionary Dynamics

### Two-population NEAT

Each population maintains its own:

- Gene pool with NEAT innovation tracking
- Species partitioning (compatibility distance threshold, species representatives)
- Fitness history and species stagnation counters
- Assimilation cycle and `reproductionPolicy`

The two populations do **not** share innovation numbers, species, or DNA. They are fully independent evolutionary systems that interact only through the shared evaluation environment.

### Fitness Computation

**Prey fitness:**

- Primary: episodes survived (ticks alive)
- Secondary: food gathered per episode tick
- Secondary: scent suppression efficiency (low emission when predator is close; normal emission otherwise)
- Penalty: energy wasted on unnecessary suppression (suppressing when no predator is near)

**Predator fitness:**

- Primary: prey caught per episode
- Secondary: pursuit efficiency (prey caught per unit of burst energy spent)
- Secondary: patrol coverage (exploration of territory while energy-conserving)
- Penalty: starvation events

### Rolling Opponent Snapshot

To prevent the Red Queen problem from collapsing into trivial oscillation (both populations evolving to a fixed point and then reversing), fitness is evaluated against a **rolling opponent snapshot** rather than the current live opponent population:

- Each generation, a fixed set of opponent representatives is frozen (hall-of-fame sample + recent-population sample).
- Evaluation runs against this frozen set.
- The snapshot is updated every N generations (configurable; typical: every 5–10 generations).

This prevents a single generation of prey from collapsing predator fitness by evolving a hard counter in one step, forcing co-adaptation to be gradual.

### Arms Race Observable

The UI should expose a real-time arms race metric chart:

- **Predator detection zone size** (sum of active scent-channel weight magnitudes) — rises as prey evolve suppression
- **Prey suppression level** (mean emission suppression output across prey population) — rises as predators evolve detection
- **Prey evasion module size** (size of visual/proximity processing zone) — rises as predators become faster/more aggressive
- **Predator pursuit module size** (size of pursuit planning zone) — rises as prey become harder to catch

These four metrics should show visible oscillation or ratcheting over generations as the arms race progresses.

---

## Reproduction Mode Dynamics

This demo is the primary test of `modeIsEvolvable: true` in the `reproductionPolicy`.

**Expected trajectory:**

- Early (unstable, exploring): standard sexual reproduction dominates; high variance favors rapid search.
- Stable pursuit phase (predators dominant): predator lineages converge toward parthenogenesis (preserve successful pursuit DNA); prey lineages shift toward polyandric (diverse evasion strategies from a stable core).
- Prey breakthrough (novel suppression or evasion): predator lineages shift back toward sexual reproduction for rapid adaptation; prey consolidate with parthenogenesis.

**Observable:** a chart of reproduction mode distribution per population per generation should show mode switching correlated with arms race phase transitions.

---

## Canvas Simulation Spec

### Rendering

- **Prey:** green circles (3–4 px radius). Opacity encodes suppression level — low emission = more transparent (camouflaged).
- **Predators:** red triangles (5–6 px, pointing in heading direction).
- **Prey scent field:** green semi-transparent overlay on the grid, intensity proportional to scent concentration.
- **Vegetation / cover zones:** dark green background patches.
- **Food patches:** yellow dots, scaled by resource remaining.
- **Pursuit rays:** faint red lines from predator to detected prey when in pursuit mode (optional, toggle-able).
- **Alarm pheromone:** orange overlay when active.

### UI panels

- Play / pause / step
- Speed multiplier (1×, 2×, 5×, 10×)
- **Arms race chart:** four-line real-time chart (predator detection zone, prey suppression, prey evasion zone, predator pursuit zone) over generations
- **Reproduction mode chart:** stacked bar per population per generation (parthenogenetic % / polyandric % / sexual %)
- Population fitness history (separate lines for prey and predator mean fitness)
- Episode stats: prey alive count, predators alive count, food remaining
- Layer toggles: scent field, vegetation overlay, pursuit rays, alarm overlay

### Performance targets

- 20–40 prey + 10–20 predators at 30+ fps on canvas.
- Scent field diffusion: typed-array sweep (same infrastructure as ant hive pheromone field).
- Agent network evaluation: slab-backed typed-array forward passes.
- Optional: Web Worker offload for field updates and network evaluations.

---

## Acceptance Criteria

- Both populations run at 30+ fps on canvas with 20–40 prey and 10–20 predators.
- Prey fitness improves over generations (longer survival, more food gathered).
- Predator fitness improves over generations (more prey caught per energy unit).
- Arms race metric chart shows meaningful non-trivial trajectory (not immediate fixed-point convergence, not pure random walk).
- Predator and prey populations develop measurably different `computationType` module compositions by generation 50+.
- Prey scent suppression behavior emerges without being hardcoded (network learns to suppress when predator is near, not always).
- Predator pursuit-commit mode switch is measurably faster via `ModulatorBroadcaster` than would be possible with structural change alone (i.e., the gain shift happens within 1–2 forward passes of prey detection).
- Reproduction mode distribution shifts are observable and correlated with arms race phase transitions when `modeIsEvolvable: true`.
- Rolling opponent snapshot prevents trivial one-generation fitness collapse in either population.
- Total agent network wiring cost declines over generations relative to task performance (compact specialists emerge).

---

## Readiness Checklist (for implementation start)

- [ ] NGE Phase G prerequisites met (co-evolutionary evaluation harness implemented).
- [ ] All Phase 0 computation motif primitives implemented.
- [ ] Phase E reproduction modes (parthenogenesis, polyandric, sexual) implemented with `modeIsEvolvable` support.
- [ ] Two-population NEAT harness implemented (independent gene pools, independent species tracking).
- [ ] Rolling opponent snapshot mechanism implemented (hall-of-fame + recent-population sampling).
- [ ] Scent field typed-array infrastructure implemented (reuse ant hive pheromone field infrastructure).
- [ ] Arms race observable metric definitions finalized (which module size proxy to use for each metric).
- [ ] Per-population fitness contract written (prey survival vs. predator catch efficiency).
- [ ] Canvas rendering approach decided (raw 2D context vs. WebGL).
- [ ] Deterministic episode seeding contract written (same seed → same terrain, food placement, starting positions).
- [ ] Energy budget constants balanced (prey should not trivially starve; predators should not trivially dominate).
