# NEAT Genesis EvoDevo: Racing Curriculum

**Status:** [WIP]

This plan defines the team adversarial racing benchmark for [NEAT Genesis EvoDevo
(NGE)](completed/NEAT_Genesis_EvoDevo.md). It is designed to be a genuine NGE showcase: two
independently evolved teams of three cars each, all teammates sharing identical DNA, roles
emerging from experience rather than from role-assignment code, team coordination via a
stigmergy-analog radio field, tire degradation as a metabolic budget, and co-evolutionary
pressure between the two teams.

This benchmark is downstream of [NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md) and
[completed/Memory_Optimization.md](completed/Memory_Optimization.md). If this plan conflicts with
either upstream plan, the upstream plan wins.

---

## Why Team Racing Fits NGE

The core NGE thesis is that complex, specialized, cooperative behaviors emerge from simple
genetic programs â€” not from hand-coded role scripts. Team adversarial racing is structured to
validate exactly that:

| NGE thesis                              | How team racing validates it                                                                                                                              |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Role differentiation from identical DNA | All three teammates start from the same genotype; queen/blocker/pacer roles emerge from driving experience alone                                          |
| Stigmergy via shared signal field       | Team radio is a typed-array shared signal, not discrete messages â€” same primitive as ant pheromone                                                        |
| Polyandric reproduction                 | Winning "queen car" is the primary genetic template; blocker and pacer drone contributions patch distinct DNA regions                                     |
| Co-evolution between populations        | Team A and Team B are fully independent NEAT populations; one team's improvements shift the other's fitness landscape                                     |
| Three-tier memory                       | Short-term: recurrent per-episode state; medium-term: rival pit patterns and teammate radio calibration; long-term: cornering priors assimilated into DNA |
| `ModulatorBroadcaster` neuromodulation  | Behavioral mode switches (sprint â†’ block â†’ hold pit â†’ rejoin) must happen within one forward pass â€” structural change is too slow                         |
| Wiring economy under complexity         | Six networked agents per race; per-agent networks must compact or they cannot run at browser-frame rates                                                  |

Compare this to the original solo racing design, which exercised `GatedRecurrentCell` and
`ModulatorBroadcaster` but not role differentiation, stigmergy, polyandric reproduction, or
co-evolution. The team structure closes those gaps.

---

## Scope and Maturity

This is a benchmark-architecture plan, not an implementation-complete spec.

- **In scope:** team structure, team radio protocol, tire degradation system, pit stop design,
  category ladder, carry-state semantics, sensory families, behavioral-drive vocabulary,
  co-evolutionary dynamics, reproduction policy, visual design and browser-runtime architecture,
  procedural track generation, the worker protocol and packed render-frame contract, the honest
  first implementation boundary, and acceptance criteria.
- **Out of scope (for now):** final vehicle-physics constants, final reward weights, and the
  _full_ co-evolution analytics dashboards (their data schemas are defined here; their
  implementation is deferred until the NGE prerequisites that feed them land).
- **Rendering decision (made):** raw Canvas 2D context. With at most six visible cars and an
  explicit no-3D constraint, WebGL is unnecessary upfront and is deferred unless profiling proves
  Canvas 2D insufficient.
- **Authority rule:** [NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md) and
  [completed/Memory_Optimization.md](completed/Memory_Optimization.md) remain authoritative.

---

## NGE Capabilities Exercised

| NGE capability                           | How team racing exercises it                                                                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `computationType` specialization         | Braking, line-tracking, traffic, radio-reading, and pit-timing zones should produce structurally distinct modules per emerged role            |
| `ModulatorBroadcaster`                   | Behavioral drives (sprint, block, pit, rejoin) must switch within one forward pass; each team car must be capable of all modes                |
| `GatedRecurrentCell` (short-term memory) | Per-episode recurrent state: recent steering/slip/throttle traces, recent radio state, tire decay trajectory                                  |
| `EpisodicSlot` (medium-term memory)      | Opponent pit timing patterns, teammate radio calibration (what does a high signal on channel 4 actually mean?), track segment danger profiles |
| `GatingRouter`                           | Hard task switching: sprint vs. block vs. hold-pit vs. pit-entry vs. recovery â€” distinct policy heads, not soft interpolation                 |
| `ResidualTap`                            | Track geometry and race-position signals flow as a residual highway available to all processing zones without wiring cost                     |
| Identical-DNA role differentiation       | All team members share one genotype; queen/blocker/pacer specialization emerges from driving history and radio interaction alone              |
| Stigmergy via radio field                | 6â€“8 dimensional team radio written and read by all teammates â€” same typed-array primitive as ant pheromone; no addressed messages             |
| Polyandric reproduction                  | Winning queen car = primary template; blocker/pacer drone DNA patches non-overlapping gene regions                                            |
| Co-evolutionary dynamics                 | Team A and Team B are fully independent NEAT populations; fitness computed against rolling opponent team snapshot                             |
| `reproductionPolicy.modeIsEvolvable`     | Teams may shift reproduction strategy as co-evolutionary phase shifts (search phase vs. consolidation phase)                                  |
| Wiring economy                           | Six networked agents must run at browser-frame rates; per-agent compaction under team complexity is the key pressure test                     |

---

## Execution Alignment

This benchmark belongs to **Phase G (Multi-Agent + Collective Intelligence)** in the NGE roadmap.
It must not begin serious implementation before:

- NGE Phase A (DNA + deterministic development) and Phase B (Juvenile focus) are stable.
- All Phase 0 computation motif primitives are implemented and opt-in verified.
- Phase E (Evolution integration + reproduction modes) is implemented with polyandric support and
  `modeIsEvolvable`.
- Two-population NEAT harness is implemented (independent gene pools, independent species
  tracking).
- Stigmergy typed-array field primitive is available â€” landed as
  `src/neat/nge-collective/neat.nge-collective.shared-field.ts` in Phase G Step 04 (shared with
  ant hive pheromone infrastructure).

## Recommended agent + skill combo for this Phase G benchmark

- Benchmark architecture, curriculum, and rollout work â€” `NGE Benchmark Scout` +
  `nge-benchmark-workflow`
- Upstream NGE prerequisite drift â€” `NGE Core Scout` + `nge-core-algorithm`
- Canvas, layout, and interaction polish â€” `Visualizer Scout` + `visualizer-workflow`

---

## Design Pillars

- **Two genuinely independent teams:** Team A and Team B each have their own DNA gene pool, NEAT
  species tracking, and assimilation cycle. They interact only through the shared evaluation
  environment.
- **Identical-DNA teams, divergent roles:** all three teammates start from one genotype. Role
  divergence (queen/pacer/blocker) emerges from driving experience and radio interaction â€” never
  from role-assignment code.
- **Team radio as stigmergy:** a typed-array shared signal written by all three cars and read by
  all three. No messages, no addressing. Same primitive as ant pheromone â€” teammates must learn
  to read and write it through evolution.
- **Team wins if any member wins:** this is the fitness pressure that drives cooperative strategy.
  A blocker that protects the fast car is rewarded even if it finishes last.
- **Tire degradation as metabolic budget:** tire state [1.0 â†’ 0.0] decays with driving
  aggression. Visual feedback per wheel. Tire state is a sensory input channel. Team radio carries
  teammate tire state. Pit stops restore tire state.
- **One pit per team:** each team has a fixed dedicated pit position. Pit entrance is wide enough
  for legal blocking. Team radio coordination governs when to pit and whether to block the
  opponent from entering their own pit.
- **No scripted strategies:** all blocking, pacing, pit timing, and radio semantics emerge from
  the evolved network structure and `ModulatorBroadcaster` gain calibration.
- **Egocentric observations only:** each car's controller never receives a planner-style full-map
  oracle.
- **Visible specialization:** the benchmark should make it easy to observe sensor-family
  specialization, modular growth around radio-reading and pit-timing, and adult pruning of unused
  sensor families.

---

## Team Structure

### Two Teams, Three Cars Each

```
Team A:  A1 Â· A2 Â· A3    (all share Team A DNA genotype)
Team B:  B1 Â· B2 Â· B3    (all share Team B DNA genotype)
```

**Team A and Team B are fully independent NEAT populations.** They do not share innovation
numbers, species, DNA, or assimilation cycles.

**Within a team, all three cars share one genotype.** There is no separate DNA for "the blocker
car." Role divergence emerges from the order in which the network experiences:

- who gets into traffic first
- who reads high vs. low tire decay signals on the radio
- who happens to be ahead of the opponent's fast car at lap 1

The network reacts to its current sensory context â€” including the team radio field â€” and the
`EpisodicSlot` accumulates a history that gradually differentiates the cars' behavioral patterns.

### Team Fitness

**A team's score in one race = the finishing position of its best-finishing car.**

This is the critical fitness pressure. A team that produces a fast queen car and two effective
blockers beats a team that produces three mediocre individual performers. The three-way shared DNA
means natural selection acts on the team's collective strategy, not three individual strategies.

---

## Team Radio (Stigmergy Analog)

The team radio is a **6â€“8 dimensional typed-array shared field**, written by all three cars and
read by all three. It is not a message-passing system. There are no recipients, no channels
reserved for specific cars, and no protocol defined in advance. Teams must evolve a shared
semantic for the signal through natural selection.

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

Each car reads **three independent radio vectors** (one per teammate) of 6â€“8 dimensions each.

```
Total radio input per car: 3 teammates Ã— 7 dimensions = 21 channels
```

Each car also writes a radio output vector of 6â€“8 dimensions, which is placed into the shared field for teammates to read.

The shared field is a typed-array that is updated synchronously with the simulation tick, exactly as the ant hive pheromone field. There is no decay â€” the field reflects the most recent output from each car.

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
tire_state âˆˆ [0.0, 1.0]   (1.0 = fresh, 0.0 = destroyed)
```

Tire state decays as a function of:

- **lateral force** (cornering aggression)
- **longitudinal force** (hard braking and acceleration)
- **speed** (higher speed = faster base decay)
- **current tire state** (degraded tires decay faster â€” exponential degradation curve)

A fresh tire at nominal pace degrades slowly. Aggressive driving on degraded tires degrades very fast. This creates a genuine strategic tradeoff: aggressive pursuit burns tires faster; conservative driving preserves them but sacrifices pace.

### Visual Feedback

Each car is drawn as a minimalistic **square outline** in its team color, with a short heading indicator. The four tires are drawn as small marks at the four corners of that square (front-left, front-right, rear-left, rear-right). Tires render in a **distinct base color** from the car body so they read as separate elements.

When tire degradation is **enabled**, each corner mark shifts color with its tire state:

```
1.0 â†’ 0.75  : green
0.75 â†’ 0.50 : yellow
0.50 â†’ 0.25 : orange
0.25 â†’ 0.0  : red
```

When tire degradation is **disabled** (early tiers / Tier 0 scaffold), the four marks stay in their neutral base color and do not shift.

The four-corner display is always visible. This gives a human observer immediate insight into each car's strategic position without reading numbers.

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

- A car entering the pit is removed from the active simulation for a **fixed stop duration** (3â€“5 simulation ticks, balancing realism with strategic weight).
- During the stop, the car's tire states are restored to [1.0, 1.0, 1.0, 1.0].
- No other repairs occur in the pit. The pit stop is purely a tire restoration event.
- Only one car may occupy the pit at a time. If a second car from the same team enters while the first is in the pit, it must wait.

### Pit Entrance Blocking

The pit entrance corridor is wide enough for **legal defensive positioning by an opponent car**. A car from the opposing team that is positioned in the pit entrance corridor can legally impede the other team's cars from entering their pit.

This creates a high-stakes coordination scenario:

- **Blocking team:** must sacrifice one car's pace to hold the corridor, coordinated via team radio (no scripted signal â€” must evolve radio semantics for "hold the pit entrance").
- **Pitting team:** must choose when the window is clear enough to attempt pit entry, or whether to extend the stint and hope the blocker runs out of energy (tire budget).

The pit entrance blocking behavior must emerge from network evolution, not from hardcoded logic.

### Pit Strategy as EpisodicSlot Target

The `EpisodicSlot` medium-term memory is ideally motivated by pit strategy:

- **Opponent pit timing patterns:** when did the opponent team pit last race? Are they pitting early or running long? The `EpisodicSlot` should store opponent pit timing by track configuration similarity.
- **Teammate radio calibration:** what radio signal pattern precedes a teammate pitting? The `EpisodicSlot` stores radio state â†’ pit-intent associations.
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
- The snapshot is updated every N generations (configurable; typical: every 5â€“10 generations).

This prevents a single generation breakthrough from collapsing opponent fitness in one step, forcing co-adaptation to be gradual.

### Co-evolutionary Observable

The simulation UI should expose:

- **Team A mean fitness vs. Team B mean fitness** over generations (separate lines)
- **Strategy divergence metric:** how different are the two teams' network `computationType` compositions? Rising divergence = arms race. Converging = one team copying the other's strategy.
- **Pit timing distribution:** histogram of pit lap for each team per generation â€” shows whether pit strategies are converging or diverging.

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

- Early (unstable, new category): sexual reproduction may dominate â€” high variance favors rapid search.
- Stable co-evolutionary phase (one team consistently winning): winning team lineages may converge toward parthenogenesis (preserve winning DNA); losing team may shift toward polyandric or sexual for diversity.
- After a strategic breakthrough by the losing team: winner team shifts back toward sexual or polyandric for rapid adaptation.

The `modeIsEvolvable: true` flag allows the reproduction policy itself to be subject to selection pressure.

---

## Category Ladder

The six-tier category ladder is designed so that each tier unlocks one additional layer of strategic complexity. Early tiers verify that the NGE lifecycle can handle basic racecraft before exposing it to team coordination and co-evolutionary arms races.

### Tier 1 â€” 1v1, No Radio

```
2 cars (one per team), no team radio, no pits, fresh tires only.
Track: simple oval or wide flowing circuit.
```

**Purpose:** baseline verification that single-car NGE can learn to drive at all. Eliminates team mechanics from the first learning problem. Both teams evolve a single-car policy.

### Tier 2 â€” 1v1 with Radio

```
2 cars (one per team), team radio active (only one car per team so self-communication).
No pits, no tire degradation.
Track: simple circuit with one tight corner requiring overtaking.
```

**Purpose:** verify that network can write and read radio without teammates present. The car learns to use radio as a self-monitoring signal (e.g., pace intent, threat level). This is a degenerate but structurally valid use of the radio field.

### Tier 3 â€” 2v2, No Pits

```
2 cars per team (4 total), team radio active between teammates.
No pit stops, no tire degradation.
Track: intermediate circuit with genuine overtaking zones.
```

**Purpose:** first appearance of role differentiation. Two identical-DNA teammates must develop different behavioral specializations through experience. No pit timing complexity.

### Tier 4 â€” 2v2, Tires and Pits

```
2 cars per team (4 total), team radio active.
Tire degradation active, one pit per team.
Track: intermediate circuit with clear pit window tradeoffs.
```

**Purpose:** introduce tire degradation as metabolic budget. Teams must evolve pit timing and pit-entrance blocking coordination. First appearance of `EpisodicSlot` motivation (opponent pit timing patterns).

### Tier 5 â€” 3v3 Full

```
3 cars per team (6 total), full team radio.
Tire degradation active, one pit per team, pit-entrance blocking legal.
Track: full competition circuit.
```

**Purpose:** full NGE team racing. Queen/blocker/pacer roles must emerge from experience. Co-evolutionary arms race between Team A and Team B. Polyandric reproduction active.

### Tier 6 â€” 3v3 Advanced Strategy

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

- 3 teammates Ã— 7-dimensional radio vector = 21 channels
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
- effective policy input width: ~110â€“130 once short-horizon buffers included
- team radio output head: 7 auxiliary output channels (stigmergy write)
- initial scaffold: ~110â€“160 nodes
- initial sparse connectivity: ~1,000â€“2,000 connections

---

## Behavioral Drives and Neuromodulation

Each car's controller must be capable of all behavioral modes â€” the network switches between them via `ModulatorBroadcaster` gain shifts, not by structural change.

**Drive families â†’ `ModulatorBroadcaster` mapping:**

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

- `SPRINT` â€” maximize pace, accept tire burn
- `BLOCK` â€” hold inside line, sacrifice own pace to impede opponent
- `PACE` â€” conservative speed, preserve tire budget
- `PIT_ENTRY` â€” low speed, pit-corridor alignment
- `RECOVERY` â€” off-track or post-contact stabilization
- `HOLD_PIT_ENTRANCE` â€” park in opponent's pit corridor at minimum speed

The `GatingRouter` selects the dominant policy head based on the current sensory context, including team radio signals. A car commanded by radio to block does not gradually interpolate toward blocking â€” it hard-routes to the `BLOCK` policy head.

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
- radio semantic must remain undetermined â€” nursery does not teach any specific radio protocol
- does not cover blocking, pit timing, or advanced racecraft

Promoted cars do not return to the nursery. It is a newborn-only scaffold.

---

## Optimal-Line Guidance Fade Policy

Tier 1â€“2 provide explicit optimal-line guidance channels as curriculum scaffold. These fade through the tier ladder:

- **Tier 1â€“2:** explicit optimal line, lateral error, heading error, target speed envelope
- **Tier 3â€“4:** guidance available but less reliable; traffic and tire pressure dominate
- **Tier 5â€“6:** guidance reduced or removed; network must have internalized cornering priors into `EpisodicSlot` and DNA-assimilated weights by this point

---

## Visual Design and Browser Runtime

The benchmark ships as a browser demo in the same family as the existing Flappy Bird example. It must look great on its own terms â€” a neon-retro-arcade racing scene â€” while staying a faithful, inspectable window onto the NGE controllers driving it.

### Aesthetic â€” Neon-Retro-Arcade (Flappy parity)

The demo matches the repo's established Astro-Bird / neon-retro-arcade direction used by the Flappy Bird example.

- **Background:** deep near-black blue (`#060b14`). Flat. **No 3D scene and no starfield** â€” the racing surface is the focus.
- **Structural lines:** blue and cyan (`#00bfff`, `#00ffff`) for walls, track edges, and panel chrome.
- **Labels and text:** high-contrast cyan-white (`#9fdcff`) on the dark field.
- **Accents:** restrained warm-neon (amber/orange) reserved for the single most important highlight in a view (e.g., the leading car, an active pit, an alarm state). Contrast and consistency over decorative intensity.
- **Team identity:** Team A = cyan family; Team B = warm red/orange family. Team color is the car-body stroke color.
- Final color, glow, and motion-polish details are deliberately deferred to a closing polish pass; the tokens above are the contract everything else builds on.

> Style tokens live in a `visualization.colors` / style-constants module mirroring the Flappy example, so the demo and any generated docs share one palette source.

### Screen Layout â€” Canvas Left, Network Right, Visualizer Last

The page is a responsive three-region layout, mirroring the Flappy `browser-entry` SOLID split (`host/`, `playback/`, `network-view/`, `visualization/`):

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  SIMULATION CANVAS                  â”‚  NETWORK VIEW         â”‚   top row
â”‚  (procedural track + cars)          â”‚  (live controller     â”‚
â”‚  left, dominant width               â”‚   graph of the        â”‚
â”‚                                     â”‚   focused car)        â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  VISUALIZER  (charts + diagnostics, full width)            â”‚   last / bottom
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

- **Left â€” simulation canvas:** the dominant region; the procedural track and cars (see below).
- **Right â€” network view:** the live forward-pass graph of the currently focused car, with the same neon node/edge styling and hover tooltips as the Flappy network view.
- **Bottom (last) â€” visualizer:** the diagnostic charts and panels, full width.
- **Responsiveness:** below a width threshold the layout collapses to a single column (canvas â†’ network â†’ visualizer) so it remains legible on narrow viewports.
- **Host ownership:** a `host/` module owns DOM assembly and resize, exactly as the Flappy host does.

### Publication Pipeline

The example follows the repo's standard browser-demo path:

- Source entrypoint at `examples/<racing-demo>/index.html` loads a built bundle from `../../docs/assets/<racing>.bundle.js` (repo path) or `../../assets/...` (published path), exactly as `examples/flappy_bird/index.html` does.
- `scripts/copy-examples.ts` republishes the page to `docs/examples/<racing-demo>/index.html` during `npm run docs`. **Do not hand-edit the `docs/examples/**` copy.\*\*

### Procedural Track Generation

The track fills the canvas and is procedurally generated. A larger viewport yields a **longer circuit with more alternate lines**, not a scaled-up small oval.

**Determinism boundary (critical for replay and fairness):**

- The track is generated from `seed + layoutVersion + quantizedSizeBucket`, where `quantizedSizeBucket` is the CSS layout size snapped to a coarse bucket â€” **not** raw physical pixels or device-pixel-ratio.
- The generated `TrackSpec` is frozen into the race pack at episode/race reset. It is **never** regenerated during a live resize; a resize only re-fits the camera/scale onto the existing `TrackSpec`. Crossing into a new size bucket only changes the track at the next race reset, intentionally.
- This guarantees `same seed + same size bucket â†’ identical TrackSpec`, so evaluation and replay stay reproducible across machines and DPI settings.

**Generation algorithm (graph/spline validated, not freeform):**

1. Seed a deterministic PRNG from the determinism tuple above.
2. Choose a quantized logical world size from the size bucket (logical units, decoupled from pixels).
3. Place angularly ordered control points around an ellipse / convex hull to define a **closed centerline**.
4. Smooth the centerline with a closed Catmull-Rom (or cubic) spline.
5. Enforce geometry constraints: minimum segment length, minimum corner radius, maximum curvature, and minimum wall clearance.
6. Add **alternate lines** as branch segments that reconnect between two progress markers `s0 < s1`: shorter/riskier inside cuts, wider/safer outside lines, and pit-adjacent tactical lanes. Larger worlds admit more branches.
7. Place each team's pit box and pit-entrance corridor adjacent to the main loop.
8. **Validate** the sampled geometry with a spatial index: no self-intersection, no dead ends, minimum track width respected, start line and both pit boxes reachable, and unambiguous lap progression. Reject-and-reseed on failure.

**Avoid:** random oval stretching, pixel-space generation, branches that make lap counting ambiguous, and alternate lines that are strictly always better (they must be genuine risk/reward tradeoffs).

### Car and Tire Rendering

- **Car body:** a minimalistic **square outline** stroked in the team color, with a short heading indicator (a tick or notch on the leading edge). No fill, no sprite â€” clean neon linework.
- **Tires:** four small marks at the square's corners (FL, FR, RL, RR) drawn in a **distinct base color** from the body so they read as separate elements. With degradation enabled they shift `green â†’ yellow â†’ orange â†’ red` with tire state; with degradation disabled they hold the neutral base color (see the Tire Degradation System Â§ Visual Feedback).
- **Focused car:** the car shown in the network-view panel gets the warm-neon highlight accent and an outline emphasis.
- **Radio activity (toggle):** a faint halo pulse when a car's radio write output exceeds a threshold.
- **Mode indicator (toggle, debug):** a small letter overlay (S=Sprint, B=Block, P=Pace, I=PitEntry, H=HoldPit, R=Recovery).

### Track Rendering

- **Asphalt:** dark drivable corridors over the background; the racing surface.
- **Walls / boundaries:** neon blue/cyan lines (`#00bfff`).
- **Sand traps:** muted tan off-asphalt zones (low-grip penalty regions).
- **Pit boxes:** a rectangle at each team's pit position (Team A cyan, Team B red/orange).
- **Pit entrance corridor:** a faint dashed lane from the track edge to the pit box.
- **Optimal-line overlay:** a faint dashed guidance line, strongest in early tiers and fading through the ladder (see Optimal-Line Guidance Fade Policy).
- **Alternate lines:** rendered subtly so a viewer can see the route choices the cars trade off.
- **Start/finish:** a clear neon start/finish marker; lap progression is read from the centerline parameter.

### UI Panels (Visualizer â€” last region)

Controls and diagnostics. Charts whose data depends on not-yet-implemented NGE capabilities are defined here as **schemas / placeholders**; they render once their feeding subsystem lands rather than blocking the first implementation.

- Play / pause / step
- Speed multiplier (1Ã—, 2Ã—, 5Ã—, 10Ã—)
- **Episode stats:** cars alive per team, current lap, race timer, current tier label + promotion progress
- **Tire degradation overlay:** per-car tire-state bars for all live cars _(active once tire degradation lands)_
- **Team fitness chart:** Team A vs. Team B mean fitness over generations _(active once two-population evaluation lands)_
- **Team radio monitor:** heatmap of the current radio field, one row per car _(placeholder until radio lands)_
- **Reproduction mode chart:** stacked bar per team per generation (parthenogenetic / polyandric / sexual %) _(placeholder until reproduction modes land)_
- **Co-evolutionary strategy divergence:** divergence of the two teams' `computationType` compositions over generations _(placeholder until co-evolution lands)_
- **Layer toggles:** optimal-line overlay, tire colors, radio halo, mode overlay, alternate lines

### Network View (right region)

- Renders the focused car's controller as a live graph using the Flappy network-view conventions (neon node/edge color scale encoding weight/bias sign and magnitude, hover tooltips).
- A focus selector cycles which car feeds the panel.
- Because all teammates share one genotype, the network view also makes role-divergence inspectable: the same topology, different live activations per car.

### Performance Targets and Optimizations

- **Target:** 6 cars at 30+ fps in display mode; training throughput saturates available cores.
- **Canvas 2D** rendering (decision above); offscreen static track layer cached once per `TrackSpec` and only the cars/overlays redrawn per frame.
- **Slab-backed forward passes** for controllers (`multi.activateSerializedNetwork`), reusing the library's typed-array fast path.
- **Single-flight render requests:** the display loop requests at most one in-flight frame from the worker at a time (no backlog).
- **Trivial per-tick state:** tire state = 4 floats/car; radio field = cars Ã— 7 floats â€” typed-array, negligible.
- Optional **Web Worker offload** for network evaluation and simulation stepping (see Worker Protocol below).

---

## Worker Protocol and Render Frame

Simulation/evaluation runs off the main thread; the display loop consumes packed snapshots. This mirrors the peer NGE demo plans (`NEAT_Genesis_EvoDevo_AntHive_Demo.md`, `NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md`).

### Headless / Display / Training Split

The authoritative benchmark logic is **headless-safe** and browser-independent:

- A shared environment module owns physics, sensors, procedural track generation, lap timing, penalties, and reward hooks.
- The browser layer only **renders snapshots** of that environment; it never owns benchmark truth.
- The same environment module is driven by headless tests (generator invariants, physics invariants, protocol round-trips) and by the worker.

### Simulation Timing

- **Fixed-timestep** simulation with **render interpolation**: the environment advances in fixed dt steps for determinism; rendering interpolates between the two most recent states.
- Determinism must hold under variable frame rate â€” replay from a seed reproduces the same trajectory regardless of display fps.

### Render Frame â€” Packed SoA Typed Arrays (zero-copy transfer)

```ts
type RacingRenderFrame = {
  schemaVersion: 'racing-packed-v1';
  tick: number;
  seed: number;
  trackId: number; // identifies the frozen TrackSpec this frame belongs to
  agentCount: number; // fixed for the episode; unused slots are flagged inactive
  featureFlags: number; // bitfield: tiresEnabled, radioEnabled, pitsEnabled, ...

  // Car arrays (fixed ordering for the whole episode)
  carX: Float32Array; // [agentCount] logical world coords
  carY: Float32Array;
  carHeading: Float32Array; // radians
  carActive: Uint8Array; // 1=in race, 0=inactive/disabled slot (e.g. solo Tier 0)
  carTeam: Uint8Array; // 0=Team A, 1=Team B
  carMode: Uint8Array; // GatingRouter mode index (0 when routing not yet present)

  // Tire state â€” 4 corners per car, row-major [agentCount Ã— 4]
  tireState: Float32Array; // [agentCount*4] in [0,1]; all 1.0 when degradation disabled

  // Radio field â€” [agentCount Ã— radioDim]; empty when radio disabled
  radioField: Float32Array;

  // Episode scalars
  lap: Uint16Array; // [agentCount]
  place: Uint8Array; // [agentCount]
  raceTimeMs: number;
  done: boolean;
};
```

### Buffer Ownership

Zero-copy transfer **detaches** the underlying buffers, so the producer must not reuse a transferred array. Ownership rule:

- The worker writes into a **double- or triple-buffered** pool of `RacingRenderFrame` arrays (a frame allocator); it transfers buffer `N` and immediately begins filling buffer `N+1`.
- The display side reads the received frame, then returns its buffers to the pool (or lets them be GC'd if not pooled). No frame is read after its buffers have been transferred onward.
- Every message carries `schemaVersion`; consumers reject unknown versions rather than misreading bytes.
- A transfer-list completeness check ensures every typed-array buffer in the frame is listed exactly once.

### Generation Lifecycle (when NGE evaluation is active)

```
Generation N:
1. Coordinator receives 'population-ready' from the team NEAT worker(s).
2. Coordinator queues episode tasks: each team genome Ã— deterministic race-pack seeds,
   paired against the rolling opponent snapshot.
3. Episode worker pool runs episodes headless; results aggregate per genome
   (mean fitness âˆ’ stability_weight Ã— stddev across seeds).
4. When all genomes have their seed results, coordinator submits fitness + 'evolve'.
5. Coordinator forwards the current champion(s) to the display worker for live rendering.
6. NEAT worker replies 'population-ready' â†’ Generation N+1.
```

---

## Acceptance Criteria

These are split so that early scaffold work is judged against what it can actually satisfy, and the NGE-level criteria are not applied to the harness prematurely.

### Tier 0 â€” Visual Driving Harness (scaffold) criteria

- `same seed + same size bucket â†’ byte-identical TrackSpec`; generated geometry passes all validation invariants (closed loop, no self-intersection, no dead ends, min width, reachable start + pit boxes, unambiguous lap progression).
- Fixed-timestep replay from a seed is stable and frame-rate independent (same trajectory regardless of display fps).
- A solo car driven by a baseline/scripted controller can complete laps on generated tracks.
- The `RacingRenderFrame` packs and unpacks deterministically; transfer-list completeness holds and `schemaVersion` mismatches are rejected.
- The network-view panel renders the active controller graph; the canvas-left / network-right / visualizer-last layout is correct and collapses to one column on narrow viewports.
- The neon-retro-arcade style tokens are applied; no 3D and no starfield are present.

### NGE benchmark criteria (gated on the relevant prerequisites landing)

- Both teams run at 30+ fps with 3 cars each on canvas.
- Tier 1 is reliably solvable by at least one lineage in each team under deterministic race packs.
- Tier 5 produces measurable role differentiation within teams: at least one car consistently scores lower individual position but improves team score (blocker behavior).
- Tire degradation creates measurable strategy divergence between teams: different mean pit laps, different driving aggression profiles.
- Team radio field shows non-random structure by Tier 3 (mutual information between radio output and next-tick behavior is above baseline).
- `ModulatorBroadcaster` mode switches are measurably fast: sprint â†’ block mode switch completes within 1â€“2 forward passes of trigger signal.
- Pit-entrance blocking behavior emerges without scripting: cars from the opposing team hold the pit corridor in at least 10% of late-tier races.
- `EpisodicSlot` contents show opponent pit timing patterns: networks that recall opponent pit timing should outperform those that do not (measurable by ablation in evaluation).
- Polyandric reproduction produces viable offspring by Tier 5: teams using polyandric mode should not stagnate relative to teams using sexual reproduction alone.
- `modeIsEvolvable: true` produces observable reproduction mode shifts correlated with co-evolutionary phase transitions.
- Total agent network wiring cost declines over generations relative to task performance (compact specialists emerge; bloated generalists are outcompeted).
- Tier 6 co-evolutionary arms race shows non-trivial trajectory: not immediate fixed-point convergence, not pure random walk â€” alternating advantage between the two teams is the target observable.

---

## Readiness Checklist (for implementation start)

The checklist is split into the **honest first implementation boundary** (Tier 0 visual driving harness â€” startable now) and the **NGE prerequisites** that gate the team/co-evolution tiers. Tier 0 is allowed to begin before the NGE prerequisites are met, on the condition that it implements the shared, future-proof contracts listed below so it does not become throwaway work.

### Tier 0 â€” Visual Driving Harness boundary (startable now)

- [ ] Honest first boundary agreed and scoped: browser visual harness + procedural track + vehicle physics + solo (single-car) driving with the live network-view. Non-goals: no team fitness, no radio, no role emergence, no co-evolution, no reproduction modes.
- [ ] Headless-safe environment module owns physics, sensors, track generation, lap timing, penalties, and reward hooks; browser only renders snapshots.
- [ ] Fixed-timestep simulation with render interpolation; frame-rate-independent deterministic replay.
- [ ] Procedural track generator written with the `seed + layoutVersion + quantizedSizeBucket` determinism boundary, spline-based generation, and validation invariants; `TrackSpec` frozen into the race pack at reset.
- [ ] Screen-responsive layout contract fixed (canvas left, network-view right, visualizer last; single-column collapse).
- [ ] Neon-retro-arcade style tokens module defined (shared palette; no 3D, no starfield).
- [ ] Car + tire render contract fixed (square outline, heading indicator, distinct-color corner tires with degradation coloring when enabled).
- [ ] `RacingRenderFrame` packed-SoA worker/render-frame protocol fixed, including buffer-ownership (double/triple-buffer), `schemaVersion`, and transfer-list completeness rules.
- [ ] Controller interface compatible with future NGE slab forward passes (so the harness controller seam survives the multi-agent evaluation loop).
- [ ] Network-view integration with the focused-car selector.
- [x] Rendering approach decided: raw Canvas 2D (WebGL deferred).

### NGE prerequisites (gate the team and co-evolution tiers)

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
- [ ] Tire degradation model (state field, decay function, grip multiplier) implemented.
- [ ] Tier schema drafted with deterministic race pack semantics and cross-team promotion rules.
- [ ] Deterministic race seeding contract extended to multi-car tiers (starting grid positions and rolling opponent-snapshot selection are reproducible from the race-pack seed).
- [ ] Carry-state and reset-state boundary agreed.
- [ ] Team radio protocol: typed-array size fixed; no semantic prescribed.
- [ ] Optimal-line guidance fade policy fixed per tier.
- [ ] Sensor-family normalization contract written.
- [ ] Behavioral-drive vocabulary and `GatingRouter` policy-head count agreed.
- [ ] Newborn nursery warm-start contract fixed.
- [ ] Reward and penalty contract written (anti-contact-exploit guardrails, pit-spam penalty, wrong-direction penalty).
- [ ] Tire degradation balance constants verified (fresh tires should last 1â€“2 full laps at aggressive pace; conservative pace extends tire life meaningfully).

---

## Implementation Phases

### Implementation Roadmap (phase overview)

The honest first implementation boundary is **Tier 0 â€” the Visual Driving Harness**. It is buildable now because it does not require the unmet NGE collective stack (radio, role differentiation, polyandric reproduction, co-evolution). Each later phase layers on capability and unlocks only when its NGE prerequisite lands. Every phase shares the same headless-safe environment, controller seam, worker protocol, and `RacingRenderFrame` contract so nothing built early is throwaway.

| Roadmap phase                       | Delivers                                                                                                                                                                                                                                                                                         | Gating prerequisite                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| **Tier 0 â€” Visual Driving Harness** | Neon-arcade browser shell (canvas left / network right / visualizer last), procedural track generator + validation, fixed-timestep physics, solo car + baseline controller, square-outline car + degradation-colored tires, `RacingRenderFrame` worker protocol, network-view of the focused car | None â€” startable now                                               |
| **Tier 1â€“2 â€” Solo NGE driving**     | Replace baseline controller with an evolved NGE genome; single-car racecraft; optimal-line fade; radio present but self-only                                                                                                                                                                     | NGE Phase A/B + Phase 0 archetypes                                 |
| **Tier 3 â€” 2v2 roles**              | Two identical-DNA teammates; role differentiation; team radio between teammates                                                                                                                                                                                                                  | Two-population harness + stigmergy field (field already available) |
| **Tier 4 â€” Tires + pits**           | Tire degradation budget, pit stops, pit-entrance blocking; tire/pit panels activate                                                                                                                                                                                                              | Tire degradation model                                             |
| **Tier 5â€“6 â€” Full co-evolution**    | 3v3, polyandric reproduction, rolling opponent snapshots, `modeIsEvolvable`; co-evolution dashboards activate                                                                                                                                                                                    | Reproduction modes + co-evolution loop                             |

Phase 1 below (planning) packetizes the SDLC steps that build the **Tier 0** boundary first; later roadmap phases are packetized as their prerequisites land.

### Phase 1 â€” Racing Curriculum Planning [DONE]

**Phase outcome:** The Tier 0 visual driving harness boundary is now packetized, implemented,
validated, documented, and compressed into a reusable closure record for this workstream.

**Next frontier:** No legitimate Phase 2 step packet exists yet in this plan. A future
`01-planning` pass must author and activate Phase 2 Step 01 before MCP can advance beyond this
closed Phase 1 boundary.

#### Step 01 â€” Planning packet [DONE]

Packetized the Tier 0 â€” Visual Driving Harness boundary into self-contained Steps 02-07, preserved
the fixed racing-circuit visual/runtime contracts, and advanced the plan to Step 02 for
fresh-session research. Validation evidence is recorded below.

#### Step 02 â€” Research boundary mapping [DONE]

Mapped the honest Tier 0 owner boundary to `examples/racing_curriculum/` only, using
`examples/flappy_bird/`, peer NGE demo plans, and `src/neat/nge-collective/` as the nearest
README-first anchors. The pass fixed the intended worker/runtime shape, identified the Flappy
browser-entry surfaces worth reusing, and constrained Step 03 to four honest seams: deterministic
track generation, fixed-timestep environment replay, packed `RacingRenderFrame`
transfer/schema validation, and deferred browser-host DOM layout checks. Validation passed.

#### Step 03 â€” Red Testing tier-0 contracts [DONE]

Authored 23 focused Tier 0 tests across three owner-local files for the track generator,
environment stepping, and simulation-worker snapshot seams. Twenty-two cases were intentionally
red and one baseline-green scaffold check (`createInitialState` at tick zero) remained green by
design. `npx tsc --noEmit -p tsconfig.test.json` and plan-sync both passed. Honest caveat
retained: browser-host DOM layout automation stayed deferred until a real host scaffold existed
for Step 04/05.

#### Step 04 â€” Implementing tier-0 vertical slice [DONE]

Implemented the Tier 0 slice entirely inside `examples/racing_curriculum/`: deterministic
`TrackSpec` generation and validation, frozen race-pack reset semantics, fixed-timestep
environment stepping, packed `RacingRenderFrame` transfer/schema helpers, a Canvas 2D renderer,
and the thin browser host/runtime seam with stable `canvas-left`, `network-right`, and
`visualizer-last` markers. The initial focused rerun turned the Step 03 contracts green
(`PASS 3/3 suites, 23/23 tests`) and `npx tsc --noEmit -p tsconfig.json` passed.

The same step then handled the narrow route-back needed to make the demo honestly usable without
widening scope: the stale example publication path was refreshed so the demo rendered, the track
renderer moved to a continuous Catmull-Rom ribbon, the UI made the scripted Tier 0 controller
explicit, Flappy-parity tooltip/theme/split-alignment passes landed, and the user explicitly
approved the final visual baseline. `npm run build:racing-curriculum` and plan-sync stayed green
across the follow-up passes.

#### Step 05 â€” Green Testing focused validation [DONE]

Re-ran honest green validation against the post-polish codebase rather than inheriting the older
pre-polish result. The exact Tier 0 Jest slice stayed green (`PASS 3/3 suites, 23/23 tests`),
`npm run build` exited 0, `npm run test:silent` stayed green (`PASS 442/442 suites, 5056/5056
tests`), and plan-sync returned ok. Coverage guard was N/A because no `src/` files changed. Known
caveat retained: no honest jsdom `host.layout.test.ts` exists yet, so responsive browser layout
collapse and live network-view DOM behavior remain manually approved rather than automation-proven.

#### Step 06 â€” Documenting direct deltas [DONE]

Audited only the direct documentation surfaces created or changed by Tier 0. Existing JSDoc on
the owner-local racing sources was already complete, so the step was mostly publication hygiene:
`npm run docs:examples` re-published the racing example, full `npm run docs` passed, no Mermaid or
`src/**/README.md` drift was introduced, and the earlier unrelated docs failure was confirmed
stale.

#### Step 07 â€” Logging and compression [DONE]

Compressed the finished Phase 1 history into concise done notes, refreshed the final Tier 0
validation record, cleared stale Phase 1 WIP markers, and left the plan fresh-session safe.
Phase 1 is now closed. No legitimate Phase 2 packet exists yet in this plan, so the next required
orchestrator is `01-planning` to author and activate Phase 2 Step 01 when the user wants to open
the next benchmark boundary.

### Phase 2 â€” Tier 1â€“2: Solo NGE Driving [DONE]

**Phase objective:** Replace the scripted Tier 0 baseline controller with a live NGE genome,
validate single-car racecraft through the owner-local controller seam, fade optimal-line guidance
across Tiers 0â€“2, and bring the seven-channel degenerate single-car radio path online for Tier 2.

**Phase outcome:** The solo NGE boundary is now packetized, implemented, validated, documented,
and compressed into a durable closure record. The Step 04 seam stayed entirely inside
`examples/racing_curriculum/`; no `src/` files changed, coverage guard remained N/A, and the
public `Network.activate(...)` path was sufficient for the controller integration.

**Next frontier:** No Phase 3 packet exists yet in this plan. The next legitimate orchestrator is
`01-planning` to author and activate Phase 3 â€” Tier 3: 2v2 Roles, starting with the still-blocking
two-population NEAT harness seam at `src/neat/nge-collective/`, likely
`neat.nge-collective.two-population.ts`.

#### Step 01 â€” Prerequisite audit and packetization [DONE]

Confirmed the Tier 1â€“2 gate was honestly met: NGE Phases A/B were already closed, the required
Phase 0 archetypes were exportable, the ResidualTap nuance was documented as a non-blocking
edge/property implementation detail, and the missing two-population harness was recorded as a
Phase 3 blocker only. The pass then authored the bounded Step 02â€“07 packets for the solo slice.

#### Step 02 â€” Research boundary mapping [DONE]

Mapped the implementation boundary to `examples/racing_curriculum/` only, fixed the
authoritative guidance contract to Tier 0 full > Tier 1 faded > Tier 2 none, and narrowed the
production seam to the controller factory, observation assembler, renderer guidance fade, and
seven-channel self-monitoring radio tail.

#### Step 03 â€” Red testing NGE controller seam [DONE]

Authored five owner-local red seams across the controller and observation assembler. Routed the
tracker contradictions through `00-helping` before red validation, then confirmed the focused
controller Jest slice failed as intended while plan-sync stayed green.

#### Step 04 â€” Implementing NGE controller integration [DONE]

Shipped the owner-local controller seam in `examples/racing_curriculum/controller/nge.controller.ts`,
`examples/racing_curriculum/controller/observation.assembler.ts`,
`examples/racing_curriculum/renderer/racing.renderer.ts`, and
`examples/racing_curriculum/browser-entry/browser-entry.ts`. Focused Jest validation turned green,
the `RacingRenderFrame` schema stayed intact, and no `src/` changes were needed.

#### Step 05 â€” Green validation Tier 1â€“2 slice [DONE]

Workflow MCP confirmed the intended starting state, `00-helping` repaired a Step 05
allowlist/prose mismatch before validation continued, and validation MCP then passed `npm run
build`, `npm run test:silent`, and
`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`.
Coverage guard remained N/A because the implementation stayed owner-local to
`examples/racing_curriculum/`.

#### Step 06 â€” Documentation seam deltas [DONE]

Kept the work doc-only and owner-local. JSDoc was tightened for the NGE controller, observation
assembler, renderer, and browser handle surfaces; `tsc --noEmit` and plan-sync passed; no
`src/**/README.md` regeneration was required.

#### Step 07 â€” Logging and compression Phase 2 closure [DONE]

Workflow MCP confirmed Phase 2 Step 07 was the active frontier before closure. `00-helping`
repaired the missing `Required validation:` prose so validation MCP stayed honest, then Phase 2
history was compressed, the handoff was refreshed to Phase 3 packetization, and the tracker was
left ready for a future `01-planning` pass. Phase 2 is now closed.

---

### Phase 3 â€” Tier 3: 2v2 Roles [DONE]

**Phase objective:** Implement the Tier 3 two-vs-two racing tier: two identical-DNA teammates per team (four cars total), team radio active between teammates, role differentiation emerging from driving experience alone, no pit stops, no tire degradation. At Phase 3 kickoff, the primary gating prerequisite was the missing two-population NEAT harness at `src/neat/nge-collective/`, and Step 01 audited that boundary before authoring the bounded Steps 02â€“07 packet set.

**Phase progression rule:** Step 01 is audit-only and must not touch production code or tests. Steps 02â€“07 stay bounded to the Tier 3 roadmap row: two-population harness, 2v2 controller/assembler seams, focused validation, documentation, and closure only.

**Phase outcome:** Landed the team-isolated two-population harness, Tier 3 91-channel teammate-radio observation flow, Tier 3 2v2 worker support, focused green validation, and refreshed public docs, leaving the plan ready for Phase 4 packetization.

#### Step 01 â€” Prerequisite audit and Phase 3 packetization [DONE]

Done note (2026-05-30): confirmed that `src/neat/nge-collective/neat.nge-collective.two-population.ts` does not exist. The current `src/neat/nge-collective/` boundary implements only the shared-field primitive (`createSharedField`, `writeCell`, `readCell`, `applyDecay`, `applyDiffusion`, `clearField`), sequential collective evaluation (`createCollectiveEvaluationContext`, `runCollectiveEvaluationTick`, `resetCollectiveEvaluationState`), and observability helpers (`computeRoleDivergenceMetric`, `createOpponentSnapshotPool`, `addOpponentSnapshot`) plus constants, errors, tests, and generated README coverage. Missing for Phase 3 is any team-scoped two-population harness surface: no Team A / Team B `Neat` pair boundary, no team-isolated species bookkeeping, no independent innovation-tracker ownership, no independent assimilation-cycle orchestration, and no shared-race evaluation facade that advances two isolated populations without shared mutable state.

Audit note: the Phase 1â€“2 seams are present. `examples/racing_curriculum/controller/nge.controller.ts` exists as the controller seam, `examples/racing_curriculum/controller/observation.assembler.ts` exists as the observation assembler, and the racing worker protocol already exists through `examples/racing_curriculum/workers/simulation-worker/simulation-worker.types.ts` plus `simulation-worker.snapshot.utils.ts`. The controller and assembler are still Tier 1â€“2 scoped (`ObservationTier = 1 | 2`, seven-channel self-radio tail only), so Phase 3 still needs the 2v2 / 21-channel extension even though the seam itself is in place.

Recorded minimum Phase 3 harness contract for Step 04:

- One new boundary file at `src/neat/nge-collective/neat.nge-collective.two-population.ts`.
- One harness-creation API that accepts or creates two distinct `Neat` controllers (`teamA`, `teamB`) and rejects any shared controller/state aliasing.
- One team-scoped state shape where each team owns its own population, innovation tracker/history, species state/history, assimilation or reproduction cycle state, and opponent snapshot pool.
- One headless-safe 2v2 evaluation surface that uses the existing shared-field and collective-evaluation primitives to evaluate both teams in the same race without merging controller-owned state.
- One post-evaluation advance surface that evolves or assimilates Team A and Team B independently, then registers frozen opponent-team snapshots without cross-team mutation leakage.

#### Step 02 â€” Research boundary mapping [DONE]

Done note (2026-05-30): mapped the Phase 3 contract for `src/neat/nge-collective/neat.nge-collective.two-population.ts` against the current `Neat`, NGE adult/assimilation, racing controller, observation, and worker seams. `TeamScopedState` should own one distinct `Neat` controller plus team-local runtime shelves only: the controller remains the owner of the team's population array, innovation tracker, and species bookkeeping; the harness adds one team-local opponent snapshot pool, one team-local adult/assimilation shelf, and one team-local reproduction-policy snapshot so no mutable evolutionary state aliases across teams.

Confirmed API surface for the missing boundary:

- `createTwoPopulationHarness(configA, configB): TwoPopulationHarnessState` should reject aliased controllers/state and return `{ teamA, teamB, sharedEvaluationContext, radioChannelCount, fieldSize }`.
- `TwoPopulationHarnessState` should own one shared `CollectiveEvaluationContext` for the live 2v2 race (`agentCount = 4`) plus `teamA: TeamScopedState` and `teamB: TeamScopedState`.
- `runTwoTeamEvaluationTick(harness, raceState)` should compose four stable per-car evaluators (`A0, A1, B0, B1`), call existing `runCollectiveEvaluationTick(...)`, then partition the returned `CollectiveTickResult` back into team-local result slices for later advance.
- `advanceTwoPopulations(harness, resultsA, resultsB)` should evolve Team A and Team B independently, then register frozen opponent-team snapshots with `addOpponentSnapshot(...)` after each team's advance completes.
- Isolation contract: Team A and Team B must not share controller instances, population arrays, innovation numbering/history, species representatives/history, adult or assimilation state, reproduction policy state, or opponent snapshot pools.

Worker, observation, and shared-field findings:

- `RacingRenderFrame` is already Phase 3-safe: `agentCount`, `carTeam`, and `radioField` already encode four-car 2v2 frames, and the snapshot transport helpers allocate and transfer typed arrays from `agentCount`; Phase 3 should keep schema `'racing-packed-v1'` unchanged and only populate `agentCount = 4`, `carTeam = [0, 0, 1, 1]`, and `radioField.length = 28`.
- Tier 3 observation should widen `ObservationTier` to `1 | 2 | 3`; reuse the current seven-channel Tier 2 payload as one teammate slot (`forwardSpeedWorld`, `lateralSpeedWorld`, `speedWorld`, `yawRateRadiansPerSecond`, `slipAngleRadians`, `progress01`, `optimalLineLateralOffsetWorld`), and append three ordered teammate slots (21 channels total) for a 91-channel Tier 3 vector. In 2v2, only slot 0 is live; slots 1-2 are zero-padded (14 zeros) for honest forward compatibility.
- `createSharedField`, `writeCell`, and `readCell` already fit 2v2 radio if the field is treated as row-major `[agentCount Ã— channelCount]`: `agentCount = 4`, `channelCount = 7`, `fieldSize = 28`, `width = 7`, `height = 4`. Additional seam discovered: the evaluation wrapper must keep opponent rows out of controller-visible radio input even though the underlying shared field stores all four cars in one array.

#### Step 03 â€” Red testing Phase 3 contracts [DONE]

Done note (2026-05-29 21:44:58 UTC): authored owner-local red tests at `src/neat/nge-collective/neat.nge-collective.two-population.test.ts`, `examples/racing_curriculum/controller/observation.assembler.tier3.test.ts`, and `examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier3.test.ts`. The two-population harness seam is now locked to two distinct team states, aliased-controller rejection, `sharedEvaluationContext.agentCount = 4`, partitioned `teamA` / `teamB` evaluation slices, isolated Team A / Team B advance semantics, and opponent-snapshot registration after advance. The Tier 3 observation seam is now locked to a 91-channel vector with one live teammate slot plus 14 zero-padded channels in 2v2, and the worker seam is now locked to `carTeam = [0, 0, 1, 1]`, `radioField.length = 28`, and team-local readable radio rows.

Validation note: focused Jest red validation failed as intended (`FAIL 3/3 suites, 18/18 tests`) when run with `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="src/neat/nge-collective/neat.nge-collective.two-population|examples/racing_curriculum/controller/observation.assembler.tier3|examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier3"`; failures were for the expected missing-seam reasons: `./neat.nge-collective.two-population` not found, `./simulation-worker.tier3` not found, and Tier 3 observation exports (`assembleTier3Observation`, `createTier3ObservationOptions`) not present yet. `npx tsc --noEmit -p tsconfig.test.json` exited 0. Plan-sync validation recorded below.

#### Step 04 â€” Implementing Phase 3 two-population slice [DONE]

```yaml
phase: 3
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementing.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: false
next_step: 'Step 05 â€” Green validation Phase 3 slice'
skills:
  - 'nge-benchmark-workflow'
  - 'nge-core-algorithm'
validation:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="src/neat/nge-collective|examples/racing_curriculum/controller|examples/racing_curriculum/workers/simulation-worker"'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

**Step objective:** Implement the smallest honest Phase 3 vertical slice: isolated two-population orchestration plus the 2v2 controller/assembler integration needed to evaluate four cars headlessly.

**Step implementation targets:**

- Implement `src/neat/nge-collective/neat.nge-collective.two-population.ts` with the team-isolated harness contract recorded in Step 01 and refined by Step 02.
- Extend `examples/racing_curriculum/controller/observation.assembler.ts` from the Tier 2 seven-channel self-radio tail to the forward-compatible 21-channel teammate-radio contract.
- Extend `examples/racing_curriculum/controller/nge.controller.ts` from the single-car radio seam to a two-car team seam that can consume teammate radio without introducing shared mutable controller state across teams.
- Validate the Tier 3 four-car / two-team evaluation loop headlessly, reusing the existing racing worker protocol unless Step 02 proved a narrow schema change is unavoidable.

**Required validation:**

```text
npx tsc --noEmit -p tsconfig.json
npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="src/neat/nge-collective|examples/racing_curriculum/controller|examples/racing_curriculum/workers/simulation-worker"
node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md
```

Done note (2026-05-30): implemented the Phase 3 two-population vertical slice with a new team-isolated harness at `src/neat/nge-collective/neat.nge-collective.two-population.ts`, Tier 3 91-channel teammate-radio assembly in `examples/racing_curriculum/controller/observation.assembler.ts`, and a new Tier 3 2v2 worker race pack at `examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier3.ts`, plus barrel re-exports from `src/neat/nge-collective/neat.nge-collective.ts`. Focused Jest for the three red seams is now green (`18/18 tests`), `npx tsc --noEmit -p tsconfig.json` exited 0, `neat.nge-collective.two-population.ts` is at 100% statements/branches/functions/lines in focused coverage, and `npm run test:silent` finished green (`463 suites / 5104 tests`).

#### Step 05 â€” Green validation Phase 3 slice [DONE]

```yaml
phase: 3
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: false
next_step: 'Step 06 â€” Documenting Phase 3 deltas'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
validation:
  - 'npm run quality:folder -- --folder=src/neat/nge-collective'
  - 'npm run quality:folder -- --folder=examples/racing_curriculum'
  - 'npm run build'
  - 'npm run test:silent'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

**Step objective:** Prove the Phase 3 slice is green without widening scope beyond the touched two-population and 2v2 racing boundaries.

**Step audit targets:**

- Run focused Jest coverage on the two-population harness and Tier 3 loop before broader validation.
- Confirm `npm run build` and `npm run test:silent` remain green after the Phase 3 slice lands.
- Invoke `coverage-guard` for every touched `src/` file and hold the line at 100% statements, branches, functions, and lines.
- Re-run plan-sync so the tracker remains authoritative before documentation or closure begins.

**Required validation:**

```text
npm run quality:folder -- --folder=src/neat/nge-collective
npm run quality:folder -- --folder=examples/racing_curriculum
npm run build
npm run test:silent
node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md
```

#### Step 06 â€” Documenting Phase 3 deltas [DONE]

```yaml
phase: 3
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: false
next_step: 'Step 07 â€” Logging and compression Phase 3 closure'
skills:
  - 'educational-docs'
  - 'nge-benchmark-workflow'
validation:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npm run docs'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

**Step objective:** Document only the public Phase 3 deltas introduced by the two-population and 2v2 radio surfaces.

**Step implementation targets:**

- Add or refine JSDoc for the public surface in `src/neat/nge-collective/neat.nge-collective.two-population.ts` and any newly public 2v2 controller/assembler seam.
- Confirm generated `src/**/README.md` output stays aligned with the new public API surface; regenerate with `npm run docs` only when doc-affecting source changed.
- Keep documentation atemporal and Tier 3-scoped: no tire, pit, or 3v3 forward leakage.
- Reconfirm the plan frontier and evidence trail before closure starts.

**Required validation:**

```text
npx tsc --noEmit -p tsconfig.json
npm run docs
node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md
```

#### Step 07 â€” Logging and compression Phase 3 closure [DONE]

```yaml
phase: 3
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-logging.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: false
next_step: 'Phase 4 Step 01 â€” Tier 4 packetization when the Tier 3 slice is closed'
skills:
  - 'tracker-handoff'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

**Step objective:** Compress Phase 3 into a concise closure record and leave an honest handoff toward Tier 4 only after the Tier 3 slice is fully green.

**Step implementation targets:**

- Compress completed Phase 3 history into concise done notes and remove any stale `[WIP]` step markers.
- Refresh the final validation evidence and handoff so the next legitimate frontier is explicit.
- Confirm no open Tier 3 blockers remain hidden in chat-only context.
- Handoff to Phase 4 only after the two-population Tier 3 boundary is closed, validated, and documented.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`

Done note (2026-05-30): advanced Step 07 to [WIP], compressed Phase 3 closure history, marked Phase 3 [DONE], refreshed the handoff to Phase 4 packetization by `01-planning`, and kept the tracker active in `plans/` pending manual testing before any archive move.

### Phase 4 — Tier 4: Tires + Pits [DONE]

**Phase objective:** Implement the Tier 4 2v2 racing tier: tire degradation as a metabolic budget, fixed-duration pit stops, pit-entrance blocking, and the smallest honest tire/pit render plus observation seams without widening into Tier 5 3v3 or co-evolution work.

**Phase outcome:** Phase 4 is complete and compressed.

Deliverables:
- Tire degradation: `tireState: [fl, fr, rl, rr]` per car (`1.0 = fresh`, `0.0 = destroyed`). Decay: `delta = (lateralForce * 0.002 + longitudinalForce * 0.001 + speed * 0.0001) * (1 + (1 - current) * 0.5)`. Grip multiplier: `mean(tireState)^0.5` on lateral + braking.
- Pit occupancy: two-slot per-team `pitOccupancy` in `EnvironmentState`. `NO_CAR_INDEX = 255` sentinel. `PIT_STOP_TICKS = 4` fixed duration. On exit, tires reset to `[1, 1, 1, 1]`.
- `TrackSpec.pitBoxes`: one AABB corridor per team at about 25% / 75% track progress, validated for non-overlap and reachability.
- Tier 4 observation: 95 channels = Tier 3 (91) + own-car tire `[fl, fr, rl, rr]`. `ObservationTier = 1 | 2 | 3 | 4`.
- `RacingRenderFrame`: `tireState: Float32Array(16)` + `pitStatus: Int16Array(4)`.
- Renderer: live tire color corner marks (`green ≥ 0.75` / `yellow ≥ 0.50` / `orange ≥ 0.25` / `red < 0.25`) + pit box overlays.
- New worker: `simulation-worker.tier4.ts` with `createTier4RacePack()`.
- Tests: 4 suites / 19 tests green. Full suite at closure: 470 suites / 5,147 tests green.

**Files touched:**
- `examples/racing_curriculum/environment/environment.types.ts`
- `examples/racing_curriculum/environment/environment.step.service.ts`
- `examples/racing_curriculum/track/track.generator.types.ts`
- `examples/racing_curriculum/track/track.generator.ts`
- `examples/racing_curriculum/track/track.validation.ts`
- `examples/racing_curriculum/controller/observation.assembler.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.types.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.snapshot.utils.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier4.ts`
- `examples/racing_curriculum/renderer/racing.renderer.ts`
- Tests: 4 new test files

**Next: Phase 5 — Tier 5–6: Full Co-Evolution [WIP]**
Prerequisite audit completed: upstream NGE DNA / reproduction primitives exist, but the usable benchmark-facing loop is still blocked at the harness-integration layer.
Route to `02-researching` for the examples-only Tier 5 boundary map, then keep Phase 5 honest by shipping only the six-car simulation and basic two-population co-evolution seams that do not require new `src/` reproduction wiring.

#### Step 01 — Prerequisite audit and Phase 4 packetization [DONE]

Audited the missing tire-degradation prerequisite honestly, kept the boundary examples-only, and packetized the bounded Tier 4 tire/pit steps.

#### Step 02 — Research tire and pit boundary mapping [DONE]

Mapped the owner boundary to `environment/`, `track/`, `controller/`, `workers/simulation-worker/`, and `renderer/`; kept `src/` out of scope; fixed the Tier 4 contract for tire state, pit geometry, pit occupancy, worker transport, and a new 95-channel observation tier.

#### Step 03 — Red testing tire and pit contracts [DONE]

Added four owner-local red suites / 19 tests covering tire decay, pit occupancy/reset, pit-box validation, Tier 4 observation expansion, and the new Tier 4 worker pack; the focused slice failed red as intended before implementation.

#### Step 04 — Implementing the Phase 4 tire/pit slice [DONE]

Landed the examples-only vertical slice: per-car tire decay + grip scaling, two-slot per-team pit occupancy, deterministic `pitBoxes`, Tier 4 observation assembly, `pitStatus` worker transport, `simulation-worker.tier4.ts`, and renderer tire-health / pit overlays.

#### Step 05 — Green validation Phase 4 slice [DONE]

Focused Tier 4 validation finished green at `PASS 4/4 suites, 19/19 tests`; the folder-quality gate, TypeScript validation, plan-sync, and full-suite validation stayed green, and coverage guard remained N/A because no `src/` files changed.

#### Step 06 — Documenting Phase 4 deltas [DONE]

Improved JSDoc/comments across the 10 owner-local Tier 4 files, documented the decay formula, pit sentinel / stop semantics, observation layout, packed worker lengths, and renderer contract, and intentionally skipped `npm run docs` because no generated README surface changed.

#### Step 07 — Logging and compression Phase 4 closure [DONE]

Done note (2026-05-30): compressed Phase 4 history to the closure summary above, marked Phase 4 [DONE], refreshed the handoff to the Phase 5 prerequisite-audit frontier, and kept the plan active at `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md` pending user manual test confirmation.

### Phase 5 — Tier 5–6: Full Co-Evolution (Honest Partial: Tier 5 Simulation) [DONE]

**Phase summary (2026-05-30):**
- Six-car 3v3 roster: `agentCount = 6`, `carTeam = [0, 0, 0, 1, 1, 1]`, `radioField.length = 42` (`6 × 7`)
- Packed worker tire buffer: `Float32Array(24)` (`6 × 4`); `pitStatus` stays `Int16Array(4)`
- Pit occupancy: unchanged 2-slot per-team, first-car-wins under 3v3 traffic
- `ObservationTier = 1|2|3|4|5`; Tier 5 stays **95 channels**, byte-stable with Tier 4 (3×7 radio slots now fully populated for 3v3 teammates vs zero-padded in 2v2)
- `simulation-worker.tier5.ts`: `createTier5RacePack()`, readable teammate-row helper
- `stepEnvironment` confirmed car-count-generic (no 4-car hardcoding)
- Tests: 4 suites / 22 tests green. Full suite at closure: 474 suites / 5,169 tests green
- Upstream blockers explicitly recorded: full polyandric queen/drone race-loop and `modeIsEvolvable` strategy switching remain gated on `src/neat.ts` exposure of Phase E reproduction surface

**Files touched:**
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.ts` (new)
- `examples/racing_curriculum/environment/environment.step.service.ts`
- `examples/racing_curriculum/environment/environment.types.ts`
- `examples/racing_curriculum/controller/observation.assembler.ts`
- Tests: 4 new test files

#### Step 01 — Prerequisite audit and packetization [DONE]

Audited the Tier 5–6 prerequisites, chose the honest partial Tier 5 simulation boundary, and kept full Tier 6 co-evolution explicitly upstream-gated.

#### Step 02 — Research boundary mapping [DONE]

Fixed the six-car examples-only contract: `agentCount = 6`, `carTeam = [0, 0, 0, 1, 1, 1]`, `radioField.length = 42`, packed tires = 24, `pitStatus` = 4, byte-stable 95-channel Tier 5 observations, and unchanged first-car-wins team pit semantics.

#### Step 03 — Red testing [DONE]

Added four owner-local red suites / 22 tests covering the Tier 5 worker pack, observation, environment, and six-car pit behavior while leaving the blocked polyandric / `modeIsEvolvable` assertions out of scope.

#### Step 04 — Implementation [DONE]

Shipped the six-car examples-only Tier 5 slice with `simulation-worker.tier5.ts`, full 3v3 race-pack plumbing, readable teammate rows, and car-count-generic environment stepping without widening the upstream 2v2 `src/neat/nge-collective/` harness.

#### Step 05 — Green validation [DONE]

Validated the Tier 5 slice green at 4 suites / 22 tests and retained the broader closure baseline at 474 suites / 5,169 tests with no `src/` coverage impact.

#### Step 06 — Documentation [DONE]

Documented the stable six-car roster, field widths, readable teammate rows, byte-stable 95-channel observation contract, and generic pit semantics while keeping the upstream Tier 6 reproduction gap explicit.

#### Step 07 — Logging and compression [DONE]

Compressed Phase 5 to the closure summary above, marked Phase 5 [DONE], refreshed the held-for-manual-test handoff, and kept the plan active at `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md` pending user confirmation.
---

## Validation gates

Plan-sync gate for this plan's current state and most recently closed frontier.

### Latest validation evidence

- 2026-05-30: Phase 5 Step 07 closure completed by `07-logging`; Phase 5 was compressed to its durable closure summary, marked [DONE], the plan remained top-level [WIP] and unarchived at `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md` pending user manual browser confirmation, and plan-sync passed (`PASS 0 errors, 0 warnings`).
- 2026-05-30: Phase 5 Step 01 completed by `01-planning`. Audited all Tier 5–6 prerequisites, confirmed that upstream NGE DNA / reproduction primitives exist but are not yet consumable from the benchmark-facing two-population loop, chose **Option B — honest partial Tier 5 simulation**, authored Phase 5 Steps 02–07 packets, marked `### Phase 5 — Tier 5–6: Full Co-Evolution [WIP]`, and re-ran plan-sync after packetization.
- 2026-05-30: Phase 4 Step 06 completed. JSDoc/comments were improved across the 10 owner-local
  tire/pit files in `examples/racing_curriculum/`; the pass documented the tire decay formula,
  `[0, 1]` clamp, pit `255 = no car` sentinel, fixed stop + tire reset semantics, Tier 4
  observation tail layout, one-pit-per-team corridor AABB contract, and the packed Tier 4 worker
  lengths. `npm run docs` was skipped intentionally because `examples/racing_curriculum/` has no
  generated README surface (the published example path is `examples/racing_curriculum/index.html`
  copied to `docs/examples/racing_curriculum/index.html`). `npx tsc --noEmit -p tsconfig.json`
  exited 0 and plan-sync passed with `0 errors / 0 warnings`.
- 2026-05-29: Status advanced from [PLANNED] to [WIP]; `## Implementation Phases` was added so
  workflow MCP could bind this plan through the required implementation-section pattern.
- 2026-05-29: Visual/runtime design pass fixed the honest first boundary: racing-circuit visual
  model, Canvas 2D rendering, deterministic `seed + layoutVersion + quantizedSizeBucket` track
  contract, packed `RacingRenderFrame` ownership rules, and the Tier 0 Visual Driving Harness
  acceptance scope.
- 2026-05-29: Phase 1 Step 01 packetization completed. Step 01 was compressed to a done note,
  authored the seven-step Tier 0 packet set, and plan-sync passed.
- 2026-05-29: Step 02 completed. README-first reconnaissance narrowed Tier 0 to
  `examples/racing_curriculum/`, identified the Flappy reuse anchors and worker topology, and
  fixed four honest Step 03 seams. Plan-sync passed.
- 2026-05-29: Step 03 completed. Twenty-three focused tests were authored across the track,
  environment, and simulation-worker snapshot seams; the intended red state was confirmed; test
  typecheck and plan-sync both passed.
- 2026-05-29: Step 04 completed after a controlled route-back inside the same owner boundary.
  The Tier 0 slice shipped deterministic track generation/validation, fixed-timestep replay,
  packed render-frame helpers, and the browser host seam. The follow-up passes then refreshed the
  published example so the demo rendered, resolved the five user-reported baseline issues
  (layout fill, continuous rounded track, scripted-controller disclosure, tooltip/theme parity,
  split alignment), and recorded final user visual approval. Focused Jest rerun stayed green
  (`PASS 3/3 suites, 23/23 tests`), `npx tsc --noEmit -p tsconfig.json` exited 0, and
  `npm run build:racing-curriculum` plus plan-sync passed across the follow-ups.
- 2026-05-29: Step 05 fresh rerun completed on the post-polish codebase. The focused Tier 0 Jest
  slice stayed green (`PASS 3/3 suites, 23/23 tests`), `npm run build` exited 0, `npm run
test:silent` stayed green (`PASS 442/442 suites, 5056/5056 tests`), and plan-sync returned ok.
  Coverage guard was N/A because no `src/` files changed. Honest caveat retained: no automated
  jsdom `host.layout.test.ts` exists yet.
- 2026-05-29: Step 06 completed. Tier 0 JSDoc was already sufficient, `npm run docs:examples`
  re-published the racing example, full `npm run docs` exited 0, no `src/**/README.md`
  regeneration was required, and plan-sync passed.
- 2026-05-29: Step 07 closure completed. Phase 1 history was compressed, stale Phase 1 WIP
  markers were removed, the handoff was refreshed, and plan-sync passed after closure. Phase 1 is
  [DONE]. No Phase 2 step packet currently exists in this plan; the next legitimate frontier is
  `01-planning` to author and activate Phase 2 Step 01.
- 2026-05-29: Phase 2 â€” Tier 1â€“2: Solo NGE Driving packetized by `01-planning`. Phase 2 set to
  [WIP]; Step 01 â€” Prerequisite audit and Phase 2 packetization set to [WIP]. Steps 02â€“07
  authored as [PLANNED] placeholders aligned to the roadmap table. Handoff updated to Phase 2
  Step 01 as the active frontier. Plan-sync validation run after authoring (see below).
- 2026-05-29: Phase 2 Step 01 audit completed by `01-planning`. All four Phase 2 prerequisite
  targets were investigated:
  - NGE Phase A [DONE] (`src/neat/nge-dna/`), Phase B [DONE] (`src/neat/nge-juvenile/`): both
    confirmed via archived plan + source README recon.
  - Phase 0 archetypes GatedRecurrentCell, EpisodicSlot, ModulatorBroadcaster, GatingRouter:
    all confirmed via `src/neat/genome/genome.types.ts` (descriptor types) and
    `src/neat/genome/genome.utils.ts` (materialized runtime with `activate()`).
  - ResidualTap nuance documented: implemented as `isResidualTap: boolean` on `NgeRealizedEdge`
    (edge property, not standalone computation type). Not blocking for Tier 1â€“2.
  - Two-population NEAT harness: NOT implemented; Phase 3 blocker only; seam boundary recorded.
  - Gate verdict: **MET** for Tier 1â€“2 scope.
  - Steps 02â€“07 replaced with bounded honest packets (YAML metadata blocks + detailed targets).
  - MCP workflow snapshot: Step 01 was missing a YAML block (expected fresh-planning-step state);
    probe confirmed this; YAML block added in this pass; MCP will bind successfully post-update.
  - Plan-sync validation: run below.
- 2026-05-29: Phase 2 Step 03 completed by `03-red-testing`. Owner-local red tests were added in
  `examples/racing_curriculum/controller/nge.controller.test.ts` and
  `examples/racing_curriculum/controller/observation.assembler.test.ts`. Workflow MCP confirmed
  the starting state was Phase 2 [WIP], Step 03 [WIP]. Two tracker contradictions were fixed via
  `00-helping` before test authoring: guidanceAlpha now follows Tier 0 full > Tier 1 faded >
  Tier 2 none, and the Tier 2 radio seam now requires 7-channel self-monitoring round-trip
  fidelity plus assembler inclusion. Validation MCP allowlist matched. Focused controller Jest
  validation exited red as intended (`FAIL 2/2 suites, 5/5 tests`) because `./nge.controller`
  and `./observation.assembler` are not implemented yet. Plan-sync passed and Step 04 was opened
  as the next frontier.
- 2026-05-29: Phase 2 Step 04 completed by `04-implementing`. The owner-local NGE controller
  seam shipped in `examples/racing_curriculum/controller/nge.controller.ts`,
  `examples/racing_curriculum/controller/observation.assembler.ts`,
  `examples/racing_curriculum/renderer/racing.renderer.ts`, and
  `examples/racing_curriculum/browser-entry/browser-entry.ts`. Focused MCP Jest validation passed
  (`PASS 2/2 suites, 5/5 tests`), the `RacingRenderFrame` schema stayed intact, and no `src/`
  files were touched.
- 2026-05-29: Phase 2 Step 05 completed by `05-green-testing`. Workflow MCP confirmed the
  starting state was Phase 2 [WIP], Step 05 [WIP]. A Step 05 allowlist packet mismatch was
  routed to `00-helping` and repaired before validation continued. Validation MCP then passed
  `npm run build` (exit 0 in 27,993 ms), `npm run test:silent` (exit 0 in 516,719 ms), and
  `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
  (exit 0 in 72 ms; PASS 0 errors / 0 warnings). Coverage guard remained N/A because Step 04
  touched only `examples/racing_curriculum/`.
- 2026-05-29: Phase 2 Step 06 completed by `06-documenting`. JSDoc stayed owner-local to
  `examples/racing_curriculum/`; `npx tsc --noEmit -p tsconfig.json` exited 0, plan-sync passed,
  and no `src/**/README.md` drift was introduced.
- 2026-05-29: Phase 2 Step 07 closure completed by `07-logging`. Workflow MCP confirmed Step 07
  was active; a Step 07 validation-prose mismatch was routed to `00-helping` and repaired before
  closure; validation MCP self-check and plan-sync passed; Phase 2 was compressed to done notes,
  and the next legitimate frontier is Phase 3 packetization by `01-planning`, starting with the
  two-population harness seam at `src/neat/nge-collective/`, likely
  `neat.nge-collective.two-population.ts`.
- 2026-05-30: Phase 3 packetized by `01-planning`; prerequisite status from earlier phases was
  reconfirmed, the missing two-population harness blocker was anchored at
  `src/neat/nge-collective/neat.nge-collective.two-population.ts`, Steps 01â€“07 were authored, and
  plan-sync passed (`PASS 0 errors, 0 warnings`).
- 2026-05-30: Phase 3 Step 01 completed by `01-planning`; `src/neat/nge-collective/` was audited,
  the absence of any Team A / Team B harness was confirmed, the Phase 1â€“2 controller/observation/
  worker seams were reconfirmed, and the bounded Step 02â€“07 packets remained aligned to Tier 3.
  Plan-sync passed (`PASS 0 errors, 0 warnings`).
- 2026-05-30 01:37:27 UTC: Phase 3 Step 02 completed by `02-researching`; the two-population
  contract was fixed to `createTwoPopulationHarness`, `runTwoTeamEvaluationTick`, and
  `advanceTwoPopulations`, while the Phase 3 transport stayed stable at `agentCount = 4`,
  `carTeam = [0, 0, 1, 1]`, `radioField.length = 28`, and a 91-channel Tier 3 observation.
  Plan-sync passed (`PASS 0 errors, 0 warnings`).
- 2026-05-29 21:44:58 UTC: Phase 3 Step 03 completed by `03-red-testing`; owner-local red tests
  locked the missing two-population, Tier 3 observation, and Tier 3 worker seams, focused Jest
  failed as intended (`FAIL 3/3 suites, 18/18 tests`), and TypeScript plus plan-sync both passed.
- 2026-05-30: Phase 3 Step 04 completed by `04-implementing`; the new two-population harness,
  Tier 3 observation assembly, and Tier 3 worker race pack landed, focused Jest turned green
  (`PASS 3/3 suites, 18/18 tests`), the new `src/` boundary held 100% coverage, TypeScript passed,
  and the full suite remained green.
- 2026-05-30: Phase 3 Step 05 completed by `05-green-testing`; focused green validation, folder
  quality gates, coverage guard, and broader build/test validation all passed, with the
  `evolveXor` flake re-run as a non-regression.
- 2026-05-30: Phase 3 Step 06 completed by `06-documenting`; educational JSDoc was upgraded across
  the two-population, barrel, observation, and Tier 3 worker seams, `npm run docs` refreshed
  `src/neat/nge-collective/README.md`, focused Jest passed (`31/31 tests`), `tsc --noEmit` exited
  0, the folder quality gate passed, and plan-sync passed (`PASS 0 errors, 0 warnings`).
- 2026-05-30: Phase 3 Step 07 closure completed by `07-logging`; Step 07 was advanced to [WIP],
  Phase 3 history was compressed, Phase 3 was marked [DONE], the handoff was refreshed to Phase 4
  packetization by `01-planning`, and plan-sync passed (`PASS 0 errors, 0 warnings`).
- 2026-05-30: Phase 4 closure record compressed the Tier 4 history to its durable outcome: per-car `tireState` decay/grip scaling, two-slot per-team `pitOccupancy`, `TrackSpec.pitBoxes` AABB corridors, Tier 4 95-channel observations, `RacingRenderFrame` `tireState` + `pitStatus` transport, live tire-health / pit overlays, new `simulation-worker.tier4.ts`, and four new green owner-local suites (`19/19 tests`). Full-suite validation remained green at `PASS 470/470 suites, 5147/5147 tests`.
- 2026-05-30: Phase 4 Step 07 closure completed by `07-logging`; Phase 4 was marked [DONE], the Phase 5 prerequisite-audit handoff was refreshed to `01-planning`, the plan stayed active at `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md` pending user manual test confirmation, and plan-sync passed (`PASS 0 errors, 0 warnings`).

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Plan status: All 5 phases DONE — HELD for user manual test confirmation

Phases 1–5 are complete. The plan stays at `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md` until the user manually tests the racing curriculum browser demo and confirms.

What was built:
- Phase 1: Planning and design (Tier 0–6 architecture)
- Phase 2: Tier 0 Visual Driving Harness (solo car, canvas renderer, baseline controller, procedural track)
- Phase 3: Tier 3 2v2 Roles (two-population NEAT harness, team radio, role differentiation seam, `nge-collective`)
- Phase 4: Tier 4 Tires + Pits (tire degradation, pit stops, pit-entrance blocking, 95-channel Tier 4 observation)
- Phase 5: Tier 5 six-car simulation (3v3 pack, 42-float radio field, byte-stable 95-channel observation, first-car-wins 3-teammate pit)

Remaining upstream prerequisites (not in scope of this plan):
- Full polyandric reproduction loop: `src/neat.ts` must expose Phase E reproduction surface
- `modeIsEvolvable` strategy switching: requires `src/neat/nge-collective/` wiring
- Tier 6 hall-of-fame opponent evaluation: blocked on both above

To reopen this plan: run manual browser test → confirm → archive to `plans/completed/`
```
