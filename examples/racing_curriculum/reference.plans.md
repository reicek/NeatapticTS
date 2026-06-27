# NEAT Genesis EvoDevo: Racing Curriculum

**Status:** [WIP]

This plan defines the team adversarial racing benchmark for [NEAT Genesis EvoDevo (NGE)](completed/NEAT_Genesis_EvoDevo.md). It is designed to be a genuine NGE showcase: two independently evolved teams of three cars each, all teammates sharing identical DNA, roles emerging from experience rather than from role-assignment code, team coordination via a stigmergy-analog radio field, tire degradation as a metabolic budget, and co-evolutionary pressure between the two teams.

This benchmark is downstream of [NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md) and [completed/Memory_Optimization.md](completed/Memory_Optimization.md). If this plan conflicts with either upstream plan, the upstream plan wins.

---

## Why Team Racing Fits NGE

The core NGE thesis is that complex, specialized, cooperative behaviors emerge from simple genetic programs — not from hand-coded role scripts. Team adversarial racing is structured to validate exactly that:

| NGE thesis                              | How team racing validates it                                                                                                                              |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Role differentiation from identical DNA | All three teammates start from the same genotype; queen/blocker/pacer roles emerge from driving experience alone                                          |
| Stigmergy via shared signal field       | Team radio is a typed-array shared signal, not discrete messages — same primitive as ant pheromone                                                        |
| Polyandric reproduction                 | Winning "queen car" is the primary genetic template; blocker and pacer drone contributions patch distinct DNA regions                                     |
| Co-evolution between populations        | Team A and Team B are fully independent NEAT populations; one team's improvements shift the other's fitness landscape                                     |
| Three-tier memory                       | Short-term: recurrent per-episode state; medium-term: rival pit patterns and teammate radio calibration; long-term: cornering priors assimilated into DNA |
| `ModulatorBroadcaster` neuromodulation  | Behavioral mode switches (sprint → block → hold pit → rejoin) must happen within one forward pass — structural change is too slow                         |
| Wiring economy under complexity         | Six networked agents per race; per-agent networks must compact or they cannot run at browser-frame rates                                                  |

Compare this to the original solo racing design, which exercised `GatedRecurrentCell` and `ModulatorBroadcaster` but not role differentiation, stigmergy, polyandric reproduction, or co-evolution. The team structure closes those gaps.

---

## Scope and Maturity

This is a benchmark-architecture plan, not an implementation-complete spec.

- **In scope:** team structure, team radio protocol, tire degradation system, pit stop design, category ladder, carry-state semantics, sensory families, behavioral-drive vocabulary, co-evolutionary dynamics, reproduction policy, canvas rendering spec, and acceptance criteria.
- **Out of scope (for now):** final vehicle-physics constants, rendering library selection, final reward weights, and worker protocol shape.
- **Authority rule:** [NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md) and [completed/Memory_Optimization.md](completed/Memory_Optimization.md) remain authoritative.

---

## NGE Capabilities Exercised

| NGE capability                           | How team racing exercises it                                                                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `computationType` specialization         | Braking, line-tracking, traffic, radio-reading, and pit-timing zones should produce structurally distinct modules per emerged role            |
| `ModulatorBroadcaster`                   | Behavioral drives (sprint, block, pit, rejoin) must switch within one forward pass; each team car must be capable of all modes                |
| `GatedRecurrentCell` (short-term memory) | Per-episode recurrent state: recent steering/slip/throttle traces, recent radio state, tire decay trajectory                                  |
| `EpisodicSlot` (medium-term memory)      | Opponent pit timing patterns, teammate radio calibration (what does a high signal on channel 4 actually mean?), track segment danger profiles |
| `GatingRouter`                           | Hard task switching: sprint vs. block vs. hold-pit vs. pit-entry vs. recovery — distinct policy heads, not soft interpolation                 |
| `ResidualTap`                            | Track geometry and race-position signals flow as a residual highway available to all processing zones without wiring cost                     |
| Identical-DNA role differentiation       | All team members share one genotype; queen/blocker/pacer specialization emerges from driving history and radio interaction alone              |
| Stigmergy via radio field                | 6–8 dimensional team radio written and read by all teammates — same typed-array primitive as ant pheromone; no addressed messages             |
| Polyandric reproduction                  | Winning queen car = primary template; blocker/pacer drone DNA patches non-overlapping gene regions                                            |
| Co-evolutionary dynamics                 | Team A and Team B are fully independent NEAT populations; fitness computed against rolling opponent team snapshot                             |
| `reproductionPolicy.modeIsEvolvable`     | Teams may shift reproduction strategy as co-evolutionary phase shifts (search phase vs. consolidation phase)                                  |
| Wiring economy                           | Six networked agents must run at browser-frame rates; per-agent compaction under team complexity is the key pressure test                     |

---

## Execution Alignment

This benchmark belongs to **Phase G (Multi-Agent + Collective Intelligence)** in the NGE roadmap. It must not begin serious implementation before:

- NGE Phase A (DNA + deterministic development) and Phase B (Juvenile focus) are stable.
- All Phase 0 computation motif primitives are implemented and opt-in verified.
- Phase E (Evolution integration + reproduction modes) is implemented with polyandric support and `modeIsEvolvable`.
- Two-population NEAT harness is implemented (independent gene pools, independent species tracking).
- Stigmergy typed-array field primitive is available — landed as `src/neat/nge-collective/neat.nge-collective.shared-field.ts` in Phase G Step 04 (shared with ant hive pheromone infrastructure).

## Recommended agent + skill combo for this Phase G benchmark

- Benchmark architecture, curriculum, and rollout work — `NGE Benchmark Scout` + `nge-benchmark-workflow`
- Upstream NGE prerequisite drift — `NGE Core Scout` + `nge-core-algorithm`
- Canvas, layout, and interaction polish — `Visualizer Scout` + `visualizer-workflow`

---

## Design Pillars

- **Two genuinely independent teams:** Team A and Team B each have their own DNA gene pool, NEAT species tracking, and assimilation cycle. They interact only through the shared evaluation environment.
- **Identical-DNA teams, divergent roles:** all three teammates start from one genotype. Role divergence (queen/pacer/blocker) emerges from driving experience and radio interaction — never from role-assignment code.
- **Team radio as stigmergy:** a typed-array shared signal written by all three cars and read by all three. No messages, no addressing. Same primitive as ant pheromone — teammates must learn to read and write it through evolution.
- **Team wins if any member wins:** this is the fitness pressure that drives cooperative strategy. A blocker that protects the fast car is rewarded even if it finishes last.
- **Tire degradation as metabolic budget:** tire state [1.0 → 0.0] decays with driving aggression. Visual feedback per wheel. Tire state is a sensory input channel. Team radio carries teammate tire state. Pit stops restore tire state.
- **One pit per team:** each team has a fixed dedicated pit position. Pit entrance is wide enough for legal blocking. Team radio coordination governs when to pit and whether to block the opponent from entering their own pit.
- **No scripted strategies:** all blocking, pacing, pit timing, and radio semantics emerge from the evolved network structure and `ModulatorBroadcaster` gain calibration.
- **Egocentric observations only:** each car's controller never receives a planner-style full-map oracle.
- **Visible specialization:** the benchmark should make it easy to observe sensor-family specialization, modular growth around radio-reading and pit-timing, and adult pruning of unused sensor families.

---

## Team Structure

### Two Teams, Three Cars Each

```
Team A:  A1 · A2 · A3    (all share Team A DNA genotype)
Team B:  B1 · B2 · B3    (all share Team B DNA genotype)
```

**Team A and Team B are fully independent NEAT populations.** They do not share innovation numbers, species, DNA, or assimilation cycles.

**Within a team, all three cars share one genotype.** There is no separate DNA for "the blocker car." Role divergence emerges from the order in which the network experiences:

- who gets into traffic first
- who reads high vs. low tire decay signals on the radio
- who happens to be ahead of the opponent's fast car at lap 1

The network reacts to its current sensory context — including the team radio field — and the `EpisodicSlot` accumulates a history that gradually differentiates the cars' behavioral patterns.

### Team Fitness

**A team's score in one race = the finishing position of its best-finishing car.**

This is the critical fitness pressure. A team that produces a fast queen car and two effective blockers beats a team that produces three mediocre individual performers. The three-way shared DNA means natural selection acts on the team's collective strategy, not three individual strategies.

---

## Team Radio (Stigmergy Analog)

The team radio is a **6–8 dimensional typed-array shared field**, written by all three cars and read by all three. It is not a message-passing system. There are no recipients, no channels reserved for specific cars, and no protocol defined in advance. Teams must evolve a shared semantic for the signal through natural selection.

### Radio Field Semantics (evolved, not prescribed)

The radio field has no prescribed semantics at initialization. However, the sensory inputs are designed so that the following semantics are structurally available for evolution to discover:

- tire state of teammates (high-dimensional enough to carry all three)
- pace signal (is a teammate currently in sprint mode?)
- threat signal (is a teammate being pressured?)
- pit-intent signal (is a teammate about to pit?)
- position context (rough lap position of teammates)
- coordination request (generic urgency signal)

None of these are labeled. The network learns to write and read them.

### Radio Input to Each Car

Each car reads **three independent radio vectors** (one per teammate) of 6–8 dimensions each.

```
Total radio input per car: 3 teammates × 7 dimensions = 21 channels
```

Each car also writes a radio output vector of 6–8 dimensions, which is placed into the shared field for teammates to read.

The shared field is a typed-array that is updated synchronously with the simulation tick, exactly as the ant hive pheromone field. There is no decay — the field reflects the most recent output from each car.

### Why This Fits NGE

The team radio is structurally isomorphic to the ant hive pheromone grid:

| Racing                              | Ant Hive                           |
| ----------------------------------- | ---------------------------------- |
| 7-dimensional radio per car         | 6-channel pheromone field per cell |
| Three cars each writing and reading | Many ants each writing and reading |
| Team fitness                        | Colony fitness                     |
| Role divergence from experience     | Role divergence from experience    |
| Pit timing coordination             | Recruitment coordination           |

---

## Tire Degradation System

### Tire State

Each car maintains four tire state values: front-left, front-right, rear-left, rear-right.

```
tire_state ∈ [0.0, 1.0]   (1.0 = fresh, 0.0 = destroyed)
```

Tire state decays as a function of:

- **lateral force** (cornering aggression)
- **longitudinal force** (hard braking and acceleration)
- **speed** (higher speed = faster base decay)
- **current tire state** (degraded tires decay faster — exponential degradation curve)

A fresh tire at nominal pace degrades slowly. Aggressive driving on degraded tires degrades very fast. This creates a genuine strategic tradeoff: aggressive pursuit burns tires faster; conservative driving preserves them but sacrifices pace.

### Visual Feedback

Each tire is drawn at its corner of the car icon on canvas. Color encodes tire state:

```
1.0 → 0.75  : green
0.75 → 0.50 : yellow
0.50 → 0.25 : orange
0.25 → 0.0  : red
```

The four-corner color display is always visible. This gives a human observer immediate insight into each car's strategic position without reading numbers.

### Tire State as Sensory Input

Each car receives its own four tire states as sensory inputs (4 channels). Each car also receives teammate tire states via the team radio field. Tire state directly affects:

- grip proxy (low tire state = reduced effective grip)
- braking urgency estimate (degraded tires need more distance)
- stability margin (degraded tires = narrower safe slip range)

The car must integrate tire state into its driving strategy. A network that ignores tire state will over-push on degraded tires and crash.

### Performance Effect

Tire degradation affects car performance:

- **grip multiplier:** scales with tire state (degraded tires = lower peak grip)
- **braking efficiency:** degraded tires extend minimum braking distance
- **slip onset:** degraded tires lose traction at lower lateral force

---

## Pit Stop System

### One Pit Per Team

Each team has exactly **one dedicated pit position**, fixed on the track layout. Teams do not share pits. The pit is:

- wide enough for one car at a time
- visible on canvas as a colored box (Team A: cyan; Team B: red/orange)
- entered by driving into the pit entrance corridor

### Pit Stop Mechanics

- A car entering the pit is removed from the active simulation for a **fixed stop duration** (3–5 simulation ticks, balancing realism with strategic weight).
- During the stop, the car's tire states are restored to [1.0, 1.0, 1.0, 1.0].
- No other repairs occur in the pit. The pit stop is purely a tire restoration event.
- Only one car may occupy the pit at a time. If a second car from the same team enters while the first is in the pit, it must wait.

### Pit Entrance Blocking

The pit entrance corridor is wide enough for **legal defensive positioning by an opponent car**. A car from the opposing team that is positioned in the pit entrance corridor can legally impede the other team's cars from entering their pit.

This creates a high-stakes coordination scenario:

- **Blocking team:** must sacrifice one car's pace to hold the corridor, coordinated via team radio (no scripted signal — must evolve radio semantics for "hold the pit entrance").
- **Pitting team:** must choose when the window is clear enough to attempt pit entry, or whether to extend the stint and hope the blocker runs out of energy (tire budget).

The pit entrance blocking behavior must emerge from network evolution, not from hardcoded logic.

### Pit Strategy as EpisodicSlot Target

The `EpisodicSlot` medium-term memory is ideally motivated by pit strategy:

- **Opponent pit timing patterns:** when did the opponent team pit last race? Are they pitting early or running long? The `EpisodicSlot` should store opponent pit timing by track configuration similarity.
- **Teammate radio calibration:** what radio signal pattern precedes a teammate pitting? The `EpisodicSlot` stores radio state → pit-intent associations.
- **Pit entrance blocking coordination history:** what signal from teammates reliably indicated "hold the corridor"?

---

## Co-evolutionary Dynamics

### Two Independent NEAT Populations

Each team maintains its own:

- Gene pool with NEAT innovation tracking
- Species partitioning (compatibility distance threshold, species representatives)
- Fitness history and species stagnation counters
- Assimilation cycle and `reproductionPolicy`

The two teams do **not** share innovation numbers, species, or DNA. They interact only through the shared evaluation environment.

### Rolling Opponent Snapshot

Fitness is evaluated against a **rolling opponent team snapshot** rather than the current live opponent team:

- Each generation, a fixed set of opponent team representatives is frozen (hall-of-fame sample + recent-population sample).
- Evaluation runs against this frozen set.
- The snapshot is updated every N generations (configurable; typical: every 5–10 generations).

This prevents a single generation breakthrough from collapsing opponent fitness in one step, forcing co-adaptation to be gradual.

### Co-evolutionary Observable

The simulation UI should expose:

- **Team A mean fitness vs. Team B mean fitness** over generations (separate lines)
- **Strategy divergence metric:** how different are the two teams' network `computationType` compositions? Rising divergence = arms race. Converging = one team copying the other's strategy.
- **Pit timing distribution:** histogram of pit lap for each team per generation — shows whether pit strategies are converging or diverging.

---

## Polyandric Reproduction

The canonical reproduction mode for team racing is **polyandric**.

**Why polyandric fits:** within a team, the car that wins (or finishes best) is the "queen." The blocker and pacer cars are "drones" whose DNA contributed to the team's cooperative strategy. All three contributed to the team's success, but in structurally different gene regions.

**Polyandric policy for team racing:**

```ts
reproductionPolicy: {
  mode: "polyandric",
  polyandricDroneCount: 2,               // two drone contributors per child
  polyandricDroneContributionFraction: 0.25, // each drone patches 25% of DNA
  queenBias: 0.85,                       // queen DNA dominates
  assignedRegionStrategy: "non-overlapping",
  modeIsEvolvable: true,                 // teams may shift to sexual when exploring
  seedPolicy: "queen-weighted"
}
```

**Evolutionary trajectory:**

- Early (unstable, new category): sexual reproduction may dominate — high variance favors rapid search.
- Stable co-evolutionary phase (one team consistently winning): winning team lineages may converge toward parthenogenesis (preserve winning DNA); losing team may shift toward polyandric or sexual for diversity.
- After a strategic breakthrough by the losing team: winner team shifts back toward sexual or polyandric for rapid adaptation.

The `modeIsEvolvable: true` flag allows the reproduction policy itself to be subject to selection pressure.

---

## Category Ladder

The six-tier category ladder is designed so that each tier unlocks one additional layer of strategic complexity. Early tiers verify that the NGE lifecycle can handle basic racecraft before exposing it to team coordination and co-evolutionary arms races.

### Tier 1 — 1v1, No Radio

```
2 cars (one per team), no team radio, no pits, fresh tires only.
Track: simple oval or wide flowing circuit.
```

**Purpose:** baseline verification that single-car NGE can learn to drive at all. Eliminates team mechanics from the first learning problem. Both teams evolve a single-car policy.

### Tier 2 — 1v1 with Radio

```
2 cars (one per team), team radio active (only one car per team so self-communication).
No pits, no tire degradation.
Track: simple circuit with one tight corner requiring overtaking.
```

**Purpose:** verify that network can write and read radio without teammates present. The car learns to use radio as a self-monitoring signal (e.g., pace intent, threat level). This is a degenerate but structurally valid use of the radio field.

### Tier 3 — 2v2, No Pits

```
2 cars per team (4 total), team radio active between teammates.
No pit stops, no tire degradation.
Track: intermediate circuit with genuine overtaking zones.
```

**Purpose:** first appearance of role differentiation. Two identical-DNA teammates must develop different behavioral specializations through experience. No pit timing complexity.

### Tier 4 — 2v2, Tires and Pits

```
2 cars per team (4 total), team radio active.
Tire degradation active, one pit per team.
Track: intermediate circuit with clear pit window tradeoffs.
```

**Purpose:** introduce tire degradation as metabolic budget. Teams must evolve pit timing and pit-entrance blocking coordination. First appearance of `EpisodicSlot` motivation (opponent pit timing patterns).

### Tier 5 — 3v3 Full

```
3 cars per team (6 total), full team radio.
Tire degradation active, one pit per team, pit-entrance blocking legal.
Track: full competition circuit.
```

**Purpose:** full NGE team racing. Queen/blocker/pacer roles must emerge from experience. Co-evolutionary arms race between Team A and Team B. Polyandric reproduction active.

### Tier 6 — 3v3 Advanced Strategy

```
3 cars per team (6 total), full team radio.
Tire degradation active, pit-entrance blocking active.
Track: full circuit with multi-window strategic decisions.
Multi-generation hall-of-fame opponent snapshots.
```

**Purpose:** sustained co-evolutionary arms race. Hall-of-fame opponent evaluation. `reproductionPolicy.modeIsEvolvable` fully engaged. The benchmark for demonstrating that NGE can produce stable, sophisticated, cooperative strategies under non-stationary opponent pressure.

---

## Promotion and Refill Rules

The ladder advances only when a team completes the current tier reliably over a small deterministic pack of race variants (not a single lucky race).

**Within-team refill policy (after promotion):**

- The car with the best performance becomes the "queen" for the next generation's polyandric reproduction.
- Newborns receive queen DNA as primary template, with non-overlapping drone patches from the other cars.
- Newborns may receive a short driving-school warm-start (see Newborn Nursery).

**Cross-team promotion:** both teams must reach promotion-threshold performance to advance to the next tier together. A team that is far ahead holds at the current tier until the opponent catches up within a threshold, or until a maximum wait generation is reached. This prevents co-evolutionary dynamics from collapsing when one team has a runaway advantage.

---

## Carry-State and Reset-State Semantics

Promoted cars carry their phenotype state upward rather than being flattened into fresh random starts.

**State that carries across tier promotion:**

- current weights and biases
- `GatedRecurrentCell` hidden state (slow lifetime adaptation)
- `EpisodicSlot` contents (opponent pit timing, teammate radio calibration, track danger profiles)
- developmental stage and module focus history
- `ModulatorBroadcaster` gain calibration
- team radio read/write calibration (learned radio semantic)
- other category-independent policy state

**State that resets at every new race start:**

- world position, heading, and speed
- tire states (restored to [1.0, 1.0, 1.0, 1.0])
- collision cooldowns and off-track timers
- short-horizon observation buffers (episode-scoped recurrent state)
- recent action-history buffers
- current team radio field (cleared at race start)
- other race-local episode state

---

## Rich Sensorium

The benchmark intentionally overprovisions sensory channels to give NGE developmental structure enough raw material to specialize around braking, line tracking, traffic handling, tire management, radio reading, and pit coordination.

### Vehicle-State Senses (16 channels)

- current speed
- longitudinal acceleration, lateral acceleration, yaw rate
- heading error relative to track tangent
- steering angle, steering change rate
- throttle level, brake level
- slip angle, traction reserve / grip proxy
- estimated braking distance, stability margin
- recent control smoothness / oscillation score

### Tire-State Senses (4 channels)

- front-left tire state, front-right tire state
- rear-left tire state, rear-right tire state

### Track-Geometry Senses (12 channels)

- lateral offset from centerline
- lateral offset from the optimal line (when guidance is active)
- heading error relative to local path tangent
- curvature ahead at several lookahead distances
- upcoming corner severity, next apex side, distance to apex
- local track width, exit width after next corner
- safe-speed envelope for the next segment

### Boundary and Hazard Senses (10 channels)

- ray distances to asphalt edge, sand boundary, and wall
- nearest wall angle
- sand-entry risk, wall-impact urgency
- rejoin corridor quality, off-track recovery angle
- surface type under car
- pit entrance corridor: is it blocked? (binary + blocking car identity: same team / opponent)

### Opponent and Race-Context Senses (18 channels)

- front-left / front / front-right occupancy (opponent cars)
- left / right overlap
- rear-left / rear / rear-right pressure
- relative speed to nearest rival ahead and behind
- time-to-contact estimate
- inside lane blocked, outside lane blocked
- nearest opponent tire state estimate (inferrable from opponent behavior)
- signed forward progress, current place
- gap to car ahead and behind
- time since last clean overtake, time since last collision

### Team Radio Senses (21 channels)

- 3 teammates × 7-dimensional radio vector = 21 channels
- No semantic labeling at initialization; teams evolve the shared protocol

### Pit and Strategy Senses (8 channels)

- distance to own pit entrance
- own pit: occupied / clear / blocked by opponent
- laps since last pit
- teammate pit status (from radio: is any teammate currently in pit?)
- current tire degradation rate (derivative of mean tire state)
- estimated laps remaining before tire failure

### Short-Horizon Memory Senses (10 channels)

These channels are the raw input that `GatedRecurrentCell` modules integrate:

- recent steering, throttle, and brake history (last 3 ticks)
- recent slip history
- recent team radio state history (last 2 ticks)
- recent tire decay rate history

### Initial Scale Guidance

- raw sensory surface: ~99 channels before temporal expansion
- effective policy input width: ~110–130 once short-horizon buffers included
- team radio output head: 7 auxiliary output channels (stigmergy write)
- initial scaffold: ~110–160 nodes
- initial sparse connectivity: ~1,000–2,000 connections

---

## Behavioral Drives and Neuromodulation

Each car's controller must be capable of all behavioral modes — the network switches between them via `ModulatorBroadcaster` gain shifts, not by structural change.

**Drive families → `ModulatorBroadcaster` mapping:**

| Drive                     | Modulates                                | Mode switch speed |
| ------------------------- | ---------------------------------------- | ----------------- |
| Sprint drive              | Forward-progress zone gain               | Fast              |
| Line-adherence drive      | Tracking zone gain                       | Fast              |
| Safety-margin drive       | Boundary/hazard zone gain                | Fast              |
| Grip-preservation drive   | Slip/traction zone gain                  | Fast              |
| Collision-avoidance drive | Opponent zone gain                       | Fast              |
| Blocking drive            | Opponent zone + inside-line zone gain    | Fast              |
| Pacing drive              | Throttle constraint zone gain            | Fast              |
| Pit-preparation drive     | Braking zone gain + tire management gain | Fast              |
| Recovery drive            | Rejoin/correction zone gain              | Fast              |
| Radio-write drive         | Radio output head gain                   | Fast              |
| Anti-degeneracy drive     | Global oscillation suppression           | Slow              |

**`GatingRouter` control modes** (hard task switches, not soft interpolations):

- `SPRINT` — maximize pace, accept tire burn
- `BLOCK` — hold inside line, sacrifice own pace to impede opponent
- `PACE` — conservative speed, preserve tire budget
- `PIT_ENTRY` — low speed, pit-corridor alignment
- `RECOVERY` — off-track or post-contact stabilization
- `HOLD_PIT_ENTRANCE` — park in opponent's pit corridor at minimum speed

The `GatingRouter` selects the dominant policy head based on the current sensory context, including team radio signals. A car commanded by radio to block does not gradually interpolate toward blocking — it hard-routes to the `BLOCK` policy head.

---

## Competitive Racecraft

The benchmark should reward racecraft and penalize contact-heavy strategies.

**Desired behaviors (all emergent):**

- clean overtake
- slipstream capture and pass
- inside-line protection
- outside pressure without contact
- forcing a rival onto a slower but safe line
- aborting a pass when the move becomes unsafe
- tire-aware pace modulation (back off when tires are degraded)
- radio-coordinated pit timing with teammates
- pit-entrance blocking via `HOLD_PIT_ENTRANCE` mode
- pacing a teammate's race (leading a blocker/pacer role from sprint to pace)

**Undesired exploit patterns:**

- intentional wall-push behavior
- contact farming
- freezing in place to avoid risk
- wrong-direction blocking
- repeated unsafe rejoins
- pitting every lap to avoid tire degradation (must be penalized by stop-time cost)

---

## Newborn Nursery Warm-start

Newborns may receive a short driving-school warm-start before entering scored tier races. This is a one-time bootstrap pass that moves newborns out of pure-random territory without replacing the real evolutionary search.

**Nursery goals:**

- stay on track
- follow the optimal line when available
- brake before high curvature
- recover from simple slides
- suppress wrong-direction behavior
- write and read team radio (any signal, not a prescribed semantic)

**Warm-start guardrails:**

- same observation surface as the scored task
- short and budgeted (not a full training run)
- topology fixed during warm-start; only parameters adjusted inside the existing scaffold
- post-nursery noise applied so the prior is shared but non-rigid
- radio semantic must remain undetermined — nursery does not teach any specific radio protocol
- does not cover blocking, pit timing, or advanced racecraft

Promoted cars do not return to the nursery. It is a newborn-only scaffold.

---

## Optimal-Line Guidance Fade Policy

Tier 1–2 provide explicit optimal-line guidance channels as curriculum scaffold. These fade through the tier ladder:

- **Tier 1–2:** explicit optimal line, lateral error, heading error, target speed envelope
- **Tier 3–4:** guidance available but less reliable; traffic and tire pressure dominate
- **Tier 5–6:** guidance reduced or removed; network must have internalized cornering priors into `EpisodicSlot` and DNA-assimilated weights by this point

---

## Canvas Rendering Spec

### Agent Rendering

- **Team A cars:** cyan triangles (5–6 px, pointing in heading direction)
- **Team B cars:** red/orange triangles (5–6 px, pointing in heading direction)
- **Tire state:** four colored dots at the car icon corners (front-left, front-right, rear-left, rear-right), each colored green → yellow → orange → red based on tire state value
- **Radio activity:** faint halo pulsing when radio write output is above threshold (optional toggle)
- **Mode indicator:** small letter overlay (S=Sprint, B=Block, P=Pace, I=PitEntry, H=HoldPit, R=Recovery) for debug/inspection

### Track Rendering

- **Pit boxes:** colored rectangle at each team's pit position (Team A: cyan; Team B: red/orange)
- **Pit entrance corridor:** faintly marked as a dashed line from track edge to pit box
- **Optimal-line overlay:** faint white dashed line (visible in early tiers, fades in later tiers)
- **Sand trap zones:** tan colored sections off asphalt
- **Wall boundaries:** solid grey lines

### UI Panels

- Play / pause / step
- Speed multiplier (1×, 2×, 5×, 10×)
- **Team fitness chart:** Team A mean fitness vs. Team B mean fitness over generations (two-line chart)
- **Tire degradation overlay:** per-car tire state bar for all 6 cars visible during live race
- **Team radio monitor:** 6×7 heatmap of current radio field values (Team A cars on top row, Team B on bottom row) — purely diagnostic
- **Reproduction mode chart:** stacked bar per team per generation (parthenogenetic / polyandric / sexual %)
- **Co-evolutionary strategy divergence:** metric showing how different the two teams' `computationType` compositions are over generations
- **Category tier indicator:** current tier label, promotion threshold progress
- Episode stats: cars alive per team, current lap, race timer
- Layer toggles: optimal-line overlay, tire colors, radio halo, mode overlay

### Performance Targets

- 6 agents (3 per team) at 30+ fps on canvas
- Tire state update: 4 floats per car, 24 total — trivial cost
- Radio field: 6 cars × 7 floats = 42 floats per tick — typed-array, trivial
- Agent network evaluation: slab-backed typed-array forward passes
- Optional: Web Worker offload for network evaluations

---

## Acceptance Criteria

- Both teams run at 30+ fps with 3 cars each on canvas.
- Tier 1 is reliably solvable by at least one lineage in each team under deterministic race packs.
- Tier 5 produces measurable role differentiation within teams: at least one car consistently scores lower individual position but improves team score (blocker behavior).
- Tire degradation creates measurable strategy divergence between teams: different mean pit laps, different driving aggression profiles.
- Team radio field shows non-random structure by Tier 3 (mutual information between radio output and next-tick behavior is above baseline).
- `ModulatorBroadcaster` mode switches are measurably fast: sprint → block mode switch completes within 1–2 forward passes of trigger signal.
- Pit-entrance blocking behavior emerges without scripting: cars from the opposing team hold the pit corridor in at least 10% of late-tier races.
- `EpisodicSlot` contents show opponent pit timing patterns: networks that recall opponent pit timing should outperform those that do not (measurable by ablation in evaluation).
- Polyandric reproduction produces viable offspring by Tier 5: teams using polyandric mode should not stagnate relative to teams using sexual reproduction alone.
- `modeIsEvolvable: true` produces observable reproduction mode shifts correlated with co-evolutionary phase transitions.
- Total agent network wiring cost declines over generations relative to task performance (compact specialists emerge; bloated generalists are outcompeted).
- Tier 6 co-evolutionary arms race shows non-trivial trajectory: not immediate fixed-point convergence, not pure random walk — alternating advantage between the two teams is the target observable.

---

## Readiness Checklist (for implementation start)

- [ ] NGE Phase A (DNA + deterministic development) stable enough for phenotype lifecycle and deterministic replay.
- [ ] NGE Phase B (Juvenile focus + local growth/prune) implemented.
- [ ] `ModulatorBroadcaster` archetype implemented (Phase 0).
- [ ] `GatedRecurrentCell` archetype implemented (Phase 0).
- [ ] `EpisodicSlot` archetype implemented (Phase 0).
- [ ] `GatingRouter` archetype implemented (Phase 0).
- [ ] `ResidualTap` archetype implemented (Phase 0).
- [ ] Phase E reproduction modes (parthenogenesis, polyandric, sexual) implemented with `modeIsEvolvable` support.
- [ ] Two-population NEAT harness implemented (independent gene pools, independent species tracking).
- [x] Rolling opponent snapshot mechanism implemented (hall-of-fame + recent-population sampling). <!-- Phase G Step 04: `createOpponentSnapshotPool` + `addOpponentSnapshot` in `src/neat/nge-collective/` provide the fixed-capacity rolling buffer. Benchmark-specific sampling policy (hall-of-fame weighting, population sweep) is a benchmark-local concern. -->
- [x] Stigmergy typed-array field primitive available (shared with ant hive pheromone infrastructure). <!-- Phase G Step 04: `src/neat/nge-collective/neat.nge-collective.shared-field.ts` is implemented with `createSharedField`, `writeCell`, `readCell`, `applyDecay`, `applyDiffusion`, `clearField`. -->
- [x] Tire degradation model (state field, decay function, grip multiplier) implemented. <!-- Phase 5: `decayTireState` in `environment.step.service.ts`, `FRESH_TIRE_HEALTH` and `PIT_STOP_TICKS` in race-pack service, grip multiplier wired into race-pack physics. -->
- [x] Tier schema drafted with deterministic race pack semantics and cross-team promotion rules. <!-- Phases 2-6: per-tier team layouts, car counts, and observation dimensions in `simulation-worker.coevolution.service.ts`; deterministic race pack via `createDeterministicRacePack`. -->
- [ ] Carry-state and reset-state boundary agreed.
- [x] Team radio protocol: typed-array size fixed; no semantic prescribed. <!-- Phases 3-6: `RADIO_CHANNELS_PER_CAR = 7` in race-pack service; per-tier radio row population in tier3/tier4/tier5 modules. -->
- [ ] Optimal-line guidance fade policy fixed per tier.
- [ ] Sensor-family normalization contract written.
- [ ] Behavioral-drive vocabulary and `GatingRouter` policy-head count agreed.
- [ ] Newborn nursery warm-start contract fixed.
- [x] Reward and penalty contract written (anti-contact-exploit guardrails, pit-spam penalty, wrong-direction penalty). <!-- Phase 3: `OFF_TRACK_PENALTY`, `OFF_TRACK_GRACE_TICKS`, wrong-direction detection, and car-vs-car pushing in race-pack service. -->
- [x] Canvas rendering approach decided (raw 2D context vs. WebGL). <!-- Phases 1-3: raw 2D canvas context in `racing.renderer.ts`. -->
- [x] Deterministic race seeding contract written (same seed → same track layout, starting positions, opponent snapshot). <!-- Phases 2-6: `createDeterministicRacePack(seed, opponentSnapshot)` in race-pack service; identical seed + snapshot → identical frame. -->
- [ ] Tire degradation balance constants verified (fresh tires should last 1–2 full laps at aggressive pace; conservative pace extends tire life meaningfully).

---

## Implementation Phases

### Phase 1 — Racing Curriculum Planning [WIP]

**Phase objective:** Author the seven-step implementation workflow for the Team Racing Curriculum
benchmark and advance to the first active step.

#### Step 01 — Planning packet [WIP]

```yaml
phase: '1'
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[WIP]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: 'true'
next_step: 'Step 02 — Research boundary mapping'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md
```

**User instruction:** Paste this full step packet.

**Step objective:** Packetize the full Team Racing Curriculum implementation workflow into SDLC-numbered
step packets (Steps 02-07) and confirm the smallest honest first implementation boundary.

**Context the agent must know:**

- This plan is downstream of `plans/completed/NEAT_Genesis_EvoDevo.md` (now fully closed through Phase G)
  and `plans/completed/Memory_Optimization.md`. If this plan conflicts with either upstream, the
  upstream plan wins.
- The `src/neat/nge-collective/` core is available as a shared primitive providing `SharedField`,
  `OpponentSnapshotPool`, `addOpponentSnapshot`, and `runCollectiveEvaluationTick`.
- This is a benchmark-architecture plan; serious implementation must not begin before the Readiness
  Checklist NGE prerequisites are met or an explicit honest narrow boundary is agreed.
- `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` and `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md`
  are peer plans (neither is a prerequisite for this one).

**Execution steps:**

1. Re-read this plan's Scope, Category Ladder, Readiness Checklist, and Acceptance Criteria sections
   to understand the current design intent and which NGE prerequisites are already met.
2. Assess which Readiness Checklist items are currently met, which are unmet, and whether an
   honest narrow starting boundary (e.g., single-car Tier 1 scaffolding, sensor contract) can be
   selected before all NGE prerequisites exist.
3. Author step packets for Steps 02-07 (Research, Red Testing, Implementation, Green Testing,
   Documentation, Logging) scoped to the selected honest boundary.
4. Record the boundary decision, non-goals, and any deferred capability gaps in this plan.
5. Advance to Step 02 only when the plan is self-contained for a fresh research session.

**Stop conditions:**

- **Done:** the plan records the selected honest boundary, step packets for Steps 02-07, and
  the next active step (Step 02).
- **Hold:** keep Step 01 [WIP] if the honest boundary cannot be determined without additional
  research.
- **Blocked:** a missing upstream prerequisite prevents any honest boundary selection; stop and
  escalate to `00-helping`.
- **Route-back:** no earlier step (this is the first step).

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`

---

**Plan update requirement:** Update this plan with the selected boundary, authored step packets,
non-goals, and the next active step before ending.

**Whole-step copy rule:** The entire step block above is the prompt. Do not append a second nested
`Copy-paste prompt` subsection.

---

## Validation gates

Plan-sync gate for this plan's active step.

### Latest validation evidence

- 2026-05-29: Status advanced from [PLANNED] to [WIP]; `## Implementation Phases` section added to satisfy MCP `IMPLEMENTATION_SECTION_PATTERN` lookahead requirement for workflow-MCP binding.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Current NGE workstream state:
- plans/completed/NEAT_Genesis_EvoDevo.md Phases 0 through G are fully closed.
- Active frontier: plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md Step 01 — Planning packet [WIP].
- The src/neat/nge-collective/ shared-field and multi-agent evaluation core is implemented and at
  100% owner-local runtime coverage. Primitives available: SharedField, OpponentSnapshotPool,
  addOpponentSnapshot, runCollectiveEvaluationTick.
- plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md is a peer plan, currently [PLANNED].
- plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md is a peer plan, currently [PLANNED].

Begin with 01-planning on Step 01. Keep the plan self-contained and fresh-session safe.
```