# NEAT Genesis EvoDevo: Predator/Prey Co-evolution Demo

**Status:** [PLANNED]

This plan defines the predator/prey co-evolutionary benchmark for [NEAT Genesis EvoDevo (NGE)](NEAT_Genesis_EvoDevo.md). It is the primary stress test for co-evolutionary dynamics, sensory arms race emergence, structural divergence under non-stationary selection pressure, and reproduction mode diversity — all rendered as a live web canvas simulation in the neon arcade aesthetic used across this project.

This benchmark is downstream of [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) and [completed/Memory_Optimization.md](completed/Memory_Optimization.md). If this plan conflicts with either upstream plan, the upstream plan wins.

---

## Scope and Maturity

This is a benchmark-architecture plan, not an implementation-complete spec.

- **In scope:** world design, two-population co-evolutionary structure, prey and predator sensory channels, coordination signal design, arms race observables, NGE feature mapping, canvas simulation spec, and acceptance criteria.
- **Out of scope (for now):** exact physics constants and final reward weights.
- **Authority rule:** [NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md) and [completed/Memory_Optimization.md](completed/Memory_Optimization.md) remain authoritative.

---

## Why This Demo

The predator/prey arms race is one of the oldest and most productive experimental paradigms in evolutionary computation. It is uniquely valuable for NGE because:

- **Non-stationary fitness landscape:** neither population has a fixed target. As prey evolve better evasion routing, predators must evolve better pursuit coordination — and vice versa. This is the most direct test of whether NGE's assimilation mechanism can track a moving target without catastrophic forgetting.
- **Structural divergence under pressure:** prey and predators start from similar base DNA but should diverge dramatically in `computationType` composition over generations — prey growing larger perceptual integration and evasion modules; predators growing larger detection and pursuit-coordination modules.
- **Reproduction mode selection:** in stable pursuit-dominated phases, predators may converge toward parthenogenesis (preserve pursuit DNA). In phases where prey develop novel evasion, predators may shift toward polyandric or sexual reproduction for diversity.
- **Multi-population NEAT:** running two independent NEAT populations with shared evaluation harness is a prerequisite for the ant-hive multi-colony extension.

The Pac-Man ghost/prey framing is intentional: the game is _literally_ the canonical predator/prey scenario, instantly legible to observers, and maps directly onto the NGE co-evolutionary machinery. The EVA/Angel theming makes the project's own NGE acronym pun explicit — see the **Thematic Identity** section below.

---

## Thematic Identity — The NGE Double Meaning

The project name **NEAT Genesis EvoDevo** abbreviates to **NGE** — which is also **Neon Genesis Evangelion**. This demo makes the pun literal and intentional. The predator/prey co-evolution is framed as the canonical EVA vs. Angel conflict:

- **Prey → EVA Units** — the pilots and their Evangelions, fighting for survival
- **Predators → Angels** — the attacking entities, each with a name from real Judeo-Christian angelology as used in the anime

The arms race the demo measures (prey evasion diversity vs. predator coordination) maps directly onto the show's central drama: Angels evolving new attack strategies, NERV responding with new pilot tactics, neither side ever resolving the conflict into a fixed winner.

---

### EVA Unit Roster (Prey Color Palette)

Each prey agent in the population is permanently assigned an EVA designation for the run. The designation cycles through the unit roster. Multiple agents sharing a unit number are visually indistinguishable (same color) — just as multiple Rei clones share the same EVA-00 appearance.

| Unit   | Neon Color    | Hex       | Notes                           |
| ------ | ------------- | --------- | ------------------------------- |
| EVA-00 | Neon Orange   | `#ff8c00` | Prototype; Rei Ayanami          |
| EVA-01 | Neon Blue     | `#0099ff` | Test Type; Shinji Ikari         |
| EVA-02 | Neon Red      | `#ff0033` | Production Model; Asuka Langley |
| EVA-03 | Neon Lime     | `#99ff00` | Bardiel's host — dark irony     |
| EVA-04 | Neon Violet   | `#cc00ff` | Destroyed in S² experiment      |
| EVA-05 | Neon Gold     | `#ffcc00` | Mark.05; Rei clone variant      |
| EVA-06 | Neon Hot Pink | `#ff3399` | Kaworu's unit in Rebuild        |
| EVA-07 | Neon Spring   | `#00ff88` | Twins unit                      |

For population sizes > 8: cycle through the palette with a double-border outline (2 px inner + 4 px outer) to distinguish overflow units visually from primary designations.

**Shape:** filled arc (Pac-Man style) with mouth opening in the agent's current movement direction. Color fills the arc body; a thin `#ffffff` outline ring at 10% opacity adds depth.

---

### Angel Roster (Predator Names + Colors)

Each predator genome is permanently assigned an Angel name for the run. The primary roster is the 17 canonical NGE Angels. The supplementary roster extends into real Judeo-Christian angelology (from which NGE itself drew its names) for larger populations.

#### Primary — NGE Canonical (17 Angels)

| #    | Name       | Neon Color    | Hex       | NGE Form / Power                        |
| ---- | ---------- | ------------- | --------- | --------------------------------------- |
| 1st  | Adam       | Radiant White | `#ffffff` | Progenitor of Angels; Second Impact     |
| 2nd  | Lilith     | Pale Violet   | `#dd88ff` | Progenitor of humanity / Lilin          |
| 3rd  | Sachiel    | Electric Blue | `#4499ff` | First attacker; humanoid; energy shield |
| 4th  | Shamshel   | Deep Purple   | `#bb44ff` | Cephalopod; energy whips                |
| 5th  | Ramiel     | Ice Blue      | `#00aaff` | Octahedron; particle cannon             |
| 6th  | Gaghiel    | Sea Cyan      | `#00ffcc` | Aquatic; attacked Pacific fleet         |
| 7th  | Israfel    | Twin Yellow   | `#ffff00` | Splits into two bodies                  |
| 8th  | Sandalphon | Lava Red      | `#ff4400` | Lava-dwelling; volcano embryo           |
| 9th  | Matarael   | Acid Green    | `#88ff00` | Spider-like; drops acid                 |
| 10th | Sahaquiel  | Cosmic Pink   | `#ff00aa` | Orbital bomber; dropped from space      |
| 11th | Ireul      | Circuit Green | `#00ff44` | Microscopic; invaded MAGI computers     |
| 12th | Leliel     | Inverse White | `#f0f0f0` | Shadow sphere; internal universe        |
| 13th | Bardiel    | Dark Violet   | `#551188` | Infected EVA-03; corruption             |
| 14th | Zeruel     | Blood Red     | `#ff2222` | Most powerful; destroyed Nerv HQ        |
| 15th | Arael      | Solar Gold    | `#ffdd00` | Orbiting; psychological attack          |
| 16th | Armisael   | Merge Pink    | `#ff66cc` | Double helix; merges with EVA-00        |
| 17th | Tabris     | Silver Cyan   | `#aaffee` | Kaworu Nagisa; Angel of Free Will       |

#### Supplementary — Real Angelology (overflow for populations > 17)

Drawn from the Hebrew Bible, Book of Enoch, Talmud, and Kabbalah — the same sources NGE mined:

| Name      | Tradition       | Neon Color       | Hex       | Meaning                      |
| --------- | --------------- | ---------------- | --------- | ---------------------------- |
| Metatron  | Kabbalah        | Radiant Gold     | `#ffee88` | Scribe of God; highest angel |
| Michael   | Hebrew Bible    | Warrior Blue     | `#2255ff` | Archangel; divine warrior    |
| Gabriel   | Hebrew Bible    | Herald Silver    | `#ccccff` | Archangel; divine messenger  |
| Raphael   | Book of Tobit   | Healer Green     | `#44ff88` | Archangel; divine healer     |
| Uriel     | Book of Enoch   | Flame Orange     | `#ff7700` | Archangel; light and wisdom  |
| Azrael    | Jewish/Islamic  | Ash Grey         | `#aaaaaa` | Angel of death               |
| Samael    | Talmud/Kabbalah | Venom Red        | `#cc0022` | "Venom of God"; adversary    |
| Jophiel   | Talmudic        | Amber            | `#ffaa00` | "Beauty of God"              |
| Zadkiel   | Tradition       | Indigo           | `#4400cc` | "Righteousness of God"       |
| Cassiel   | Tradition       | Storm Grey       | `#8899aa` | "Speed of God"; solitude     |
| Raguel    | Book of Enoch   | Ocean Blue       | `#0077cc` | "Friend of God"              |
| Sariel    | Book of Enoch   | Pale Gold        | `#ddcc77` | Prince of the presence       |
| Remiel    | Book of Enoch   | Thunder Blue     | `#3366ff` | "Thunder of God"             |
| Phanuel   | Book of Enoch   | Penitence Violet | `#9933cc` | "Face of God"                |
| Anael     | Tradition       | Rose             | `#ff8899` | Venus; love and harmony      |
| Nathanael | Tradition       | Emerald          | `#00cc66` | "Gift of God"                |

**Shape:** angular rhombus silhouette (a rotated square, 45°) with a hollow centre — a simple geometric abstraction of the A.T. Field cross shape. Each angel's assigned neon color fills the outline stroke (no fill, stroke only, 3 px). This keeps the visual footprint smaller than the EVA arcs and makes species immediately distinguishable by silhouette shape, not just color.

---

### Naming Implementation

```ts
// constants/constants.theme.ts

export const EVA_UNITS = [
  { id: 'EVA-00', color: '#ff8c00' },
  { id: 'EVA-01', color: '#0099ff' },
  { id: 'EVA-02', color: '#ff0033' },
  { id: 'EVA-03', color: '#99ff00' },
  { id: 'EVA-04', color: '#cc00ff' },
  { id: 'EVA-05', color: '#ffcc00' },
  { id: 'EVA-06', color: '#ff3399' },
  { id: 'EVA-07', color: '#00ff88' },
] as const;

export const ANGELS = [
  { name: 'Adam', color: '#ffffff' },
  { name: 'Lilith', color: '#dd88ff' },
  { name: 'Sachiel', color: '#4499ff' },
  { name: 'Shamshel', color: '#bb44ff' },
  { name: 'Ramiel', color: '#00aaff' },
  { name: 'Gaghiel', color: '#00ffcc' },
  { name: 'Israfel', color: '#ffff00' },
  { name: 'Sandalphon', color: '#ff4400' },
  { name: 'Matarael', color: '#88ff00' },
  { name: 'Sahaquiel', color: '#ff00aa' },
  { name: 'Ireul', color: '#00ff44' },
  { name: 'Leliel', color: '#f0f0f0' },
  { name: 'Bardiel', color: '#551188' },
  { name: 'Zeruel', color: '#ff2222' },
  { name: 'Arael', color: '#ffdd00' },
  { name: 'Armisael', color: '#ff66cc' },
  { name: 'Tabris', color: '#aaffee' },
  // Supplementary (angelology) — used when predator population > 17
  { name: 'Metatron', color: '#ffee88' },
  { name: 'Michael', color: '#2255ff' },
  { name: 'Gabriel', color: '#ccccff' },
  { name: 'Raphael', color: '#44ff88' },
  { name: 'Uriel', color: '#ff7700' },
  { name: 'Azrael', color: '#aaaaaa' },
  { name: 'Samael', color: '#cc0022' },
  { name: 'Jophiel', color: '#ffaa00' },
  { name: 'Zadkiel', color: '#4400cc' },
  { name: 'Cassiel', color: '#8899aa' },
  { name: 'Raguel', color: '#0077cc' },
  { name: 'Sariel', color: '#ddcc77' },
  { name: 'Remiel', color: '#3366ff' },
  { name: 'Phanuel', color: '#9933cc' },
  { name: 'Anael', color: '#ff8899' },
  { name: 'Nathanael', color: '#00cc66' },
] as const;

// Assignment: genome index % roster.length → designation
// Assignments are stable for the life of the run (not re-shuffled per generation)
```

---

## NGE Capabilities Exercised

| NGE capability                          | How the predator/prey demo exercises it                                                                         |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Co-evolutionary dynamics                | Two populations with independent NEAT + assimilation cycles; fitness computed against rolling opponent snapshot |
| Non-stationary fitness landscape        | Structural arms race: predator coordination vs. prey evasion routing diversity                                  |
| `computationType` structural divergence | Prey grow larger perceptual/evasion zones; predators grow larger pursuit-planning/coordination zones            |
| `ModulatorBroadcaster`                  | Voice scream signal triggers fast mode shift in both species (flee vs. pursue)                                  |
| `GatedRecurrentCell`                    | Pursuit state (predators); evasion trajectory state (prey)                                                      |
| `EpisodicSlot`                          | Prey store predator encounter patterns by corridor topology; predators store prey evasion patterns              |
| `reproductionPolicy`                    | Mode should shift as arms race phases change; tests `modeIsEvolvable: true`                                     |
| Wiring economy                          | Both populations under size pressure; compact specialists should outcompete bloated generalists                 |
| Sensory arms race                       | Observable: prey evasion route diversity vs. predator coordination index over generations                       |

---

## Execution Alignment

This benchmark belongs to **Phase G (Multi-Agent + Collective Intelligence)** in the NGE roadmap. It must not begin serious implementation before:

- NGE Phase A (DNA + deterministic development) and Phase B (Juvenile focus) are stable.
- All Phase 0 computation motif primitives are implemented and opt-in verified.
- Phase E (Evolution integration + reproduction modes) is implemented.
- Co-evolutionary evaluation harness (two-population with rolling opponent snapshot) is implemented.

## Recommended agent + skill combo for this Phase G benchmark

- Benchmark architecture, co-evolution harness, and rollout work — `NGE Benchmark Scout` + `nge-benchmark-workflow`
- Upstream NGE prerequisite drift — `NGE Core Scout` + `nge-core-algorithm`
- Canvas, layout, and interaction polish — `Visualizer Scout` + `visualizer-workflow`

---

## Design Pillars

- **Two genuinely independent populations:** prey and predators each have their own DNA gene pool, NEAT species tracking, and assimilation cycle. They are not variants of a single population.
- **Grid-based maze world:** 4-directional movement only (N/E/S/W). The maze topology provides natural cover, evasion structure, and corridor-based line-of-sight — replacing the need for a continuous scent field.
- **Two coordination signal layers:** chem trails (spatial, persistent, both species readable) and voice (instinctive screams, hardwired by engine, only reactions are evolved). The interaction between these two channels is where the richest emergent behaviors live.
- **No scripted behaviors:** all pursuit, evasion, coordination, and alarm strategies emerge from the evolved network structure.
- **Arms race phases are observable:** the simulation UI exposes a real-time chart of prey evasion diversity vs. predator coordination index over generations.
- **Canvas-native at scale:** at least 20–40 prey + 10–20 predators running at interactive rates.

---

## World Design

### Grid and Canvas

- **Full-screen canvas:** fills the real available viewport (`containerElement.getBoundingClientRect()`). Responsive — maze rescales when the window resizes.
- **Square world:** 1:1 aspect ratio. Cell size = `floor(min(viewportW, viewportH) / gridCells)`.
- **Grid size:** determined at runtime to fill the available square. Always odd dimension (e.g. 31×31) so quarter-generation works cleanly with a true center cell.
- **Theme:** neon arcade — dark `#000` background, neon blue (`#00bfff`) double-line Unicode walls (`╔═╗║╚╝╠╣╦╩╬`), matching the Astro Bird / Flappy Bird aesthetic.

### Maze Generation (Quarter-Symmetric Procedural)

Only the **top-left quarter** is procedurally generated. The other three quadrants are mirrors:

```
[ Q1  | mirror(Q1, H)  ]    Q1 = generated quarter
[ mirror(Q1, V) | mirror(Q1, HV) ]
```

This produces a 4-fold symmetric maze on every run, like the classic Pac-Man layout. Different seeds → different mazes across runs.

**Fixed constants (never procedurally altered):**

- **Center ghost box:** filled rectangle at the exact center, sized to fit the initial predator population. Has one 2-cell-wide exit on its bottom edge — the only way out. Rendered with amber border (`#ff6600`).
- **Edge tunnels:** one tunnel opening at the center of each of the 4 edges (left, right, top, bottom). Tunnel cells are always open. Agents walking into a tunnel exit from the opposite edge (left↔right, top↔bottom). Transit costs 1 extra tick to prevent oscillation.
- **Outer border:** all edge cells that are not tunnel openings are walls.

**Generation algorithm (top-left quarter):**

1. Start with all cells as walls.
2. Place required connection points: center of the top edge (→ top tunnel), center of the left edge (→ left tunnel), center where Q1 meets the ghost box.
3. Run a **recursive backtracker** (depth-first maze carving), stepping 2 cells at a time. Produces a spanning tree with no loops.
4. **Add extra connections:** randomly remove internal walls at `p_extra ≈ 0.15` to create multiple paths. Goal is many paths, not a single-solution maze.
5. **Enforce 1-cell-wide corridors:** scan and re-wall any 2×2 open regions.
6. Mirror to the other three quadrants. At the axis seam, the connecting corridor is shared — each side sees its own wall; the open cell at the axis is common.

**Seeding:** maze is seeded per run. Same seed → same maze across all episodes of that run.

### Pellets

- On episode start: a pellet fills every open non-tunnel cell except the ghost box interior.
- **Respawn:** each pellet respawns `PELLET_RESPAWN_TICKS` ticks after being eaten (default ~300).
- No power pills.

---

## Agent Spawn

- **Predators (Ghosts):** all spawn inside the center ghost box. Released through the box exit one by one, staggered by `GHOST_RELEASE_INTERVAL` ticks — mirrors classic Pac-Man ghost release pacing.
- **Prey (Pac-Men):** spawn at random open cells (not inside ghost box, not tunnel cells) at episode start. Positions seeded for reproducibility.

---

## Movement

- **4 directions only:** North, East, South, West. No diagonals.
- **Direction constants** (reuse from `asciiMaze/mazeMovement/mazeMovement.constants.ts`):
  ```ts
  DIRECTION_DELTAS: [
    [0, -1],
    [1, 0],
    [0, 1],
    [-1, 0],
  ]; // N, E, S, W
  OPPOSITE_DIR: [2, 3, 0, 1];
  ```
- **Tick-based:** one step per tick. Both populations step simultaneously.
- **Collision:** agents cannot move into wall cells. Walled direction → agent stays in place that tick.
- **Network output:** 4 logits → softmax → argmax → direction index.

---

## Coordination Signals

Two independent coordination channels operate in parallel. The emergent richness comes from both being active simultaneously — neither alone tells the full story.

### 1. Chem Trail (Spatial, Persistent)

Each species leaves a **persistent decaying chemical trace** on every cell it visits. Two typed-array grids (one per species), both readable by both species.

- **Emission:** `trail[x][y] += EMIT_AMOUNT` each tick an agent occupies the cell.
- **Decay:** `trail[x][y] *= DECAY_FACTOR` each tick (e.g. 0.95 → ~13 tick half-life).
- **Both teams read both grids:** prey can sense where ghosts have been; ghosts can follow prey trails.

Emergent behaviors enabled:

- Prey routing away from corridors with hot ghost trails.
- Ghost following a fresh prey trail around a corner to close in silently before triggering voice.
- Prey that vary their routes (avoid retracing) confuse ghost pursuit — the maze equivalent of scent suppression.
- Prey crossing each other's trails to create false targets for ghosts.
- **Trail + voice interaction:** ghost silently follows prey trail → closes to trigger radius → both start screaming → other ghosts converge on the hot trail zone while other prey flee the scream.

**Sensor channels (+5 per agent):**

| Channel                   | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| Ghost trail N/E/S/W       | Max ghost trail intensity in each directional corridor   |
| Own-trail at current cell | Own species trail density here (self-awareness of scent) |

### 2. Voice (Instinctive, Engine-Hardwired, Symmetric)

Screaming is **not a NEAT output** — it is hardwired by the engine. NEAT only evolves _reactions_ to incoming screams. This eliminates the communication bootstrap problem: from generation 1, every encounter generates a real signal that selection can act on immediately.

**Trigger:** both agents begin screaming simultaneously the moment either has a clear **4-directional raycast** to an enemy within `SCREAM_TRIGGER_RADIUS` cells. Vision rays travel along open corridors in N/E/S/W until hitting a wall. Both agents see each other at the same moment and both trigger simultaneously.

**Duration:** `SCREAM_DURATION_TICKS` (fixed). Resets if visual contact is re-established — a chase keeps the scream alive for its full duration. Agents nearby hear a lingering scream even after both parties round a corner.

**Propagation:** wall-attenuated Manhattan distance.

```
heard = DISTANCE_DECAY^manhattanDistance × WALL_ATTENUATION^wallCount
```

Agents in the same open corridor hear each other clearly. Agents separated by walls hear only a fraction.

**Aggregation:** `max` per species — receiver hears the loudest nearby scream from each team. Gives proximity information without inflating with population density.

**Dilemma (no choice, but consequences):** a ghost cannot stop screaming when it spots prey — it alerts all nearby prey while calling nearby ghosts. A prey cannot stop screaming when spotted — it warns allied pac-men while broadcasting its location to every nearby ghost. The strategic question becomes _how do I route and act knowing I cannot suppress this signal?_

**Sensor channels (+2 per agent):**

- `heardGhostScream`: wall-attenuated max ghost scream intensity
- `heardPreyScream`: wall-attenuated max prey scream intensity

**No output neuron** — screaming is not evolvable.

---

## Sensory Channels — Final Design

8-directional rays, chemical scent fields, vegetation, and alarm pheromones from the original open-world design are **removed** — the maze topology, chem trails, and voice replace them.

### Prey (Pac-Man) — 31 inputs

| Group                        | Ch  | Description                                                                            |
| ---------------------------- | --- | -------------------------------------------------------------------------------------- |
| Wall presence                | 4   | Binary: wall adjacent in N/E/S/W                                                       |
| Corridor depth               | 4   | Normalized distance to nearest wall in each direction                                  |
| Ghost proximity by direction | 4   | Distance to nearest ghost in each direction (0=adjacent, 1=far/none; blocked by walls) |
| Nearest ghost overall        | 2   | Distance (normalized) + direction (sin/cos of cardinal bearing)                        |
| Pellet density by direction  | 4   | Count of pellets in each direction up to N cells (normalized)                          |
| Nearest pellet               | 1   | Distance to nearest pellet                                                             |
| Tunnel proximity             | 2   | Distance + direction to nearest tunnel entrance                                        |
| Self-state                   | 3   | Current heading (sin/cos) + energy level                                               |
| Ghost trail by direction     | 4   | Max ghost chem trail intensity in N/E/S/W corridors                                    |
| Own-trail at current cell    | 1   | Own species trail density here                                                         |
| Heard ghost scream           | 1   | Wall-attenuated max ghost scream intensity                                             |
| Heard prey scream            | 1   | Wall-attenuated max prey scream intensity                                              |

### Predator (Ghost) — 28 inputs

| Group                       | Ch  | Description                                           |
| --------------------------- | --- | ----------------------------------------------------- |
| Wall presence               | 4   | Binary: wall adjacent in N/E/S/W                      |
| Corridor depth              | 4   | Normalized distance to nearest wall in each direction |
| Prey proximity by direction | 4   | Distance to nearest prey in each direction            |
| Nearest prey overall        | 2   | Distance + direction (sin/cos)                        |
| Prey count in radius        | 1   | Normalized count of prey within R cells               |
| Nearest co-predator         | 2   | Distance + direction (coordination signal)            |
| Tunnel proximity            | 1   | Distance to nearest tunnel entrance                   |
| Self-state                  | 3   | Current heading (sin/cos) + energy level              |
| Prey trail by direction     | 4   | Max prey chem trail intensity in N/E/S/W corridors    |
| Own-trail at current cell   | 1   | Own species trail density here                        |
| Heard ghost scream          | 1   | Wall-attenuated max ghost scream intensity            |
| Heard prey scream           | 1   | Wall-attenuated max prey scream intensity             |

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

- Primary: ticks survived
- Secondary: pellets collected
- Secondary: escape events (successfully evading a ghost within 3 cells)
- Penalty: captures

**Predator fitness:**

- Primary: prey caught
- Secondary: territory coverage (distinct cells visited per episode)
- Secondary: pursuit efficiency (prey caught per ticks spent within 5 cells of any prey)
- Penalty: stagnation (ticks not moving toward any prey)

### Rolling Opponent Snapshot

Fitness is evaluated against a **rolling opponent snapshot** rather than the current live opponent population:

- Each generation, a fixed set of opponent representatives is frozen (hall-of-fame sample + recent-population sample).
- Evaluation runs against this frozen set.
- The snapshot is updated every N generations (configurable; typical: every 5–10 generations).

This prevents a single generation from collapsing the opponent's fitness by evolving a hard counter in one step, forcing co-adaptation to be gradual.

### Arms Race Observable

The UI exposes a real-time arms race metric chart:

- **Prey evasion routing diversity:** entropy of path choices at corridor junctions — rises when prey evolve varied routes that resist ghost trail-following
- **Predator pursuit coordination index:** frequency of two predators approaching prey from different directions simultaneously — rises when ghosts learn to coordinate via voice + trail
- **Mean prey survival ticks** over generations
- **Mean predator catch rate** over generations

These four metrics should show visible oscillation or ratcheting as the arms race progresses.

---

## Reproduction Mode Dynamics

This demo is the primary test of `modeIsEvolvable: true` in the `reproductionPolicy`.

**Expected trajectory:**

- Early (unstable, exploring): standard sexual reproduction dominates; high variance favors rapid search.
- Stable pursuit phase (predators dominant): predator lineages converge toward parthenogenesis; prey lineages shift toward polyandric (diverse evasion strategies from a stable core).
- Prey breakthrough (novel routing or trail evasion): predator lineages shift back toward sexual reproduction for rapid adaptation; prey consolidate with parthenogenesis.

**Observable:** a chart of reproduction mode distribution per population per generation should show mode switching correlated with arms race phase transitions.

---

## Canvas Simulation Spec

### Rendering

| Element                  | Visual                                                                                                                                 |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| Walls                    | Neon blue (`#00bfff`) double-line Unicode box chars on black                                                                           |
| Pellets                  | Small cyan dots (`#00ffff`, 2 px radius) — the Lilin's food supply                                                                     |
| Empty floor              | Dark grid dots at low opacity                                                                                                          |
| EVA Units (prey)         | Filled arc (Pac-Man mouth) in the unit's designated neon color; thin `#ffffff` 10%-opacity ring; mouth opens toward movement direction |
| Angels (predators)       | Hollow rotated-square rhombus silhouette (3 px stroke), no fill, in the angel's designated neon color                                  |
| Angel trail              | Semi-transparent overlay in `#ff006620` (blood red tint), intensity proportional to trail concentration                                |
| EVA trail                | Semi-transparent overlay in `#0099ff20` (blue tint), intensity proportional to trail concentration                                     |
| A.T. Field (scream ring) | Brief pulsing hexagonal ring around screaming agents, colored to species; toggle-able                                                  |
| Tunnels                  | Pulsing cyan glow at the 4 edge openings — the sea of Dirac                                                                            |
| Ghost box (Angel spawn)  | Neon rectangle with amber border (`#ff6600`) — the GeoFront                                                                            |
| Agent label              | Unit ID or Angel name rendered at 8 px above agent when hovered (toggle-able)                                                          |

### UI Panels

- Play / Pause / Step
- Speed multiplier (1×, 2×, 5×, 10×)
- **Arms race chart:** 4-line real-time (prey survival, predator catch rate, prey route diversity, predator coordination index) over generations
- **Reproduction mode chart:** stacked bar per population per generation (parthenogenetic % / polyandric % / sexual %)
- Population fitness history (separate lines for prey and predator mean fitness)
- Episode stats: prey alive, predators alive, pellets remaining
- Layer toggles: ghost trail overlay, prey trail overlay, scream rings

### Performance Targets

- 20–40 prey + 10–20 predators at 30+ fps on canvas in display mode.
- Chem trail diffusion: single typed-array multiply sweep per tick (JIT-vectorizable).
- Agent network evaluation: slab-backed flat-array forward passes (`multi.activateSerializedNetwork`).
- Training throughput: episode worker pool saturates all available CPU cores.

---

## Web Worker Architecture

### Why the architecture differs from Flappy Bird

Flappy Bird uses one worker that owns everything (NEAT + playback) because each genome runs an isolated episode with no inter-agent coupling. That design breaks here for three reasons:

1. **Inter-agent coupling within episodes.** Chem trail grids, voice propagation, and proximity sensors are shared state every tick. All agents in one episode must step through a single authoritative world — parallelising within an episode costs more in synchronisation than it saves in compute.
2. **Two independent populations must synchronise.** Prey and predator populations evolve independently but must cross a generation barrier together before the rolling opponent snapshot can update.
3. **Between-episode parallelism is the real win.** Episodes are fully isolated from each other. Saturating all CPU cores with concurrent episodes gives linear throughput scaling with core count.

### Thread topology

```
COORDINATOR (main thread)
├── Prey NEAT Worker     (1) — owns prey gene pool, species, generation counter
├── Predator NEAT Worker (1) — owns predator gene pool, species, generation counter
├── Simulation Worker   (1) — owns live display episode (world, agents, trails, voice)
└── Episode Workers     (N) — stateless; run isolated training episodes
```

Worker count formula:

```ts
const cores = navigator.hardwareConcurrency ?? 4;
// Reserve: 1 main thread + 1 simulation + 2 NEAT = 4
const EPISODE_WORKER_COUNT = Math.max(2, cores - 4);
```

All workers are pre-spawned at demo start and kept alive. Spawning cost is ~50–100 ms; posting to a live worker is ~0.1 ms.

### Complete message protocol

**Coordinator → Prey NEAT Worker** (same shape for Predator NEAT Worker):

```ts
type PreyNeatWorkerRequest =
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
          stats: PreyArmsRaceStats;
        }>;
      };
    }
  | { type: 'evolve' }
  | { type: 'stop' };
```

**Prey NEAT Worker → Coordinator**:

```ts
type PreyNeatWorkerEvent =
  | {
      type: 'population-ready';
      payload: {
        generation: number;
        population: SerializedGenome[];
        champion: SerializedGenome;
        generationStats: PreyGenerationStats;
      };
    }
  | { type: 'error'; payload: { message: string } };
```

**Coordinator → Episode Worker**:

```ts
type EpisodeWorkerRequest =
  | {
      type: 'run-prey-episode';
      payload: {
        taskId: string;
        preyGenome: SerializedGenome;
        opponentGhosts: SerializedGenome[];
        mazeConfig: MazeConfig;
        seed: number;
        rolloutIndex: number;
      };
    }
  | {
      type: 'run-predator-episode';
      payload: {
        taskId: string;
        predatorGenome: SerializedGenome;
        opponentPrey: SerializedGenome[];
        mazeConfig: MazeConfig;
        seed: number;
        rolloutIndex: number;
      };
    }
  | { type: 'stop' };
```

**Episode Worker → Coordinator**:

```ts
type EpisodeWorkerEvent =
  | {
      type: 'episode-done';
      payload: {
        taskId: string;
        fitness: number;
        armsRaceStats: ArmsRaceStats;
        episodeStats: EpisodeStats;
      };
    }
  | { type: 'error'; payload: { taskId: string; message: string } };
```

**Coordinator → Simulation Worker**:

```ts
type SimulationWorkerRequest =
  | {
      type: 'start-display';
      payload: {
        preyChampions: SerializedGenome[];
        ghostChampions: SerializedGenome[];
        mazeConfig: MazeConfig;
        seed: number;
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

**Simulation Worker → Coordinator**:

```ts
type SimulationWorkerEvent =
  | {
      type: 'render-step';
      payload: {
        requestId: number;
        frame: PredatorPreyRenderFrame;
        done: boolean;
        episodeStats?: EpisodeStats;
      };
    }
  | { type: 'error'; payload: { message: string } };
```

### Render frame — packed SoA typed arrays (zero-copy transfer)

The render frame follows the same structure-of-arrays pattern as Flappy Bird's `WorkerPlaybackFrameSnapshot`, extended for 60 agents and two trail grids:

```ts
type PredatorPreyRenderFrame = {
  format: 'pp-packed-v1';
  tick: number;
  preyCount: number;
  predatorCount: number;
  gridW: number;
  gridH: number;

  // Agent arrays — prey first, then predators (fixed ordering throughout episode)
  agentX: Int16Array; // [preyCount + predatorCount]
  agentY: Int16Array;
  agentDir: Uint8Array; // 0=N 1=E 2=S 3=W
  agentAlive: Uint8Array; // 1=alive 0=dead
  agentScreaming: Uint8Array; // 1=screaming 0=silent

  // Trail grids — flat row-major [gridW × gridH]
  preyTrail: Float32Array;
  ghostTrail: Float32Array;

  // Pellet grid — bit-packed: byte (y*gridW+x)>>3, bit (y*gridW+x)&7
  pellets: Uint8Array;

  // Episode scalars (not transferred — included in message body)
  preyAlive: number;
  predatorsAlive: number;
  pelletsRemaining: number;
};

// Transfer list — buffers move to main thread (zero copy):
const transferList = [
  frame.agentX.buffer,
  frame.agentY.buffer,
  frame.agentDir.buffer,
  frame.agentAlive.buffer,
  frame.agentScreaming.buffer,
  frame.preyTrail.buffer,
  frame.ghostTrail.buffer,
  frame.pellets.buffer,
];
```

Simulation worker re-allocates typed arrays each frame (not double-buffered). At 30 fps × ~10 KB per frame = ~300 KB/s allocation rate — negligible GC pressure.

### Generation synchronisation barrier

The coordination problem unique to this demo: both populations must finish fitness evaluation before the snapshot updates and the next generation begins. The prey and predator sides are **fully independent within a generation** — whichever finishes first evolves immediately without waiting.

```
Generation N lifecycle:

1. COORDINATOR receives 'population-ready' from both NEAT workers
   → checks SNAPSHOT_UPDATE_INTERVAL; if due, updates rolling snapshot:
       hallOfFame ← append(currentChampions), evict oldest if full
       recentSample ← random sample of size RECENT_SAMPLE_SIZE from each population
   → freezes currentPreySnapshot, currentPredatorSnapshot for this generation

2. COORDINATOR queues all episode tasks (prey + predator interleaved):
   for each preyGenome × ROLLOUT_SEED_COUNT:
     tasks.push({ type: 'prey',     genome: preyGenome,     opponents: currentPredatorSnapshot, ... })
   for each predatorGenome × ROLLOUT_SEED_COUNT:
     tasks.push({ type: 'predator', genome: predatorGenome, opponents: currentPreySnapshot, ... })
   shuffle(tasks)  // interleave prey + predator to prevent worker idle at generation boundary

3. COORDINATOR dispatches tasks to episode worker pool
   → pool feeds idle workers; capacity = EPISODE_WORKER_COUNT

4. COORDINATOR collects EpisodeResults
   → aggregates per genome: mean(fitness) − STABILITY_WEIGHT × stddev(fitness) across seeds
   → accumulates armsRaceStats per genome

5a. When ALL prey genomes have ROLLOUT_SEED_COUNT results:
    → COORDINATOR sends 'submit-fitness' + 'evolve' to Prey NEAT Worker immediately
    (does NOT wait for predator side)

5b. When ALL predator genomes have ROLLOUT_SEED_COUNT results:
    → COORDINATOR sends 'submit-fitness' + 'evolve' to Predator NEAT Worker immediately
    (does NOT wait for prey side)

6. When BOTH NEAT workers send 'population-ready':
   → Generation N+1 begins (go to step 1)
   → Coordinator sends new champions to Simulation Worker (hybrid mode)
```

### Rolling opponent snapshot

```ts
const HALL_OF_FAME_SIZE = 10; // best genomes from all past generations
const RECENT_SAMPLE_SIZE = 10; // random sample from last completed generation
const SNAPSHOT_UPDATE_INTERVAL = 5; // generations between snapshot refreshes
const ROLLOUT_SEED_COUNT = 3; // seeds per genome per generation
const OPPONENTS_PER_ROLLOUT = 8; // opponents sampled from snapshot per episode
```

Snapshot is owned by the coordinator. Episode workers receive their subset of opponents as part of each task payload (JSON, copied — not transferred). No cross-worker snapshot state needed.

### Operation modes

| Mode             | Episode workers | Simulation worker      | Main thread         |
| ---------------- | --------------- | ---------------------- | ------------------- |
| Training-only    | Full throughput | Idle                   | Fitness charts only |
| Display-only     | Idle            | Live episode at 30 fps | Full rendering      |
| Hybrid (default) | Full throughput | Live episode at 30 fps | Rendering + charts  |

Mode switching: UI button handled by coordinator. Switching to display-only pauses episode task queue but does not terminate workers.

### Performance optimisations

**1. Offscreen maze canvas (simulation worker)**
Maze walls never change during an episode. Draw them once to an `OffscreenCanvas` at episode start; each tick calls `ctx.drawImage(offscreenMaze, 0, 0)` — a single GPU blit. ~50× faster than re-drawing Unicode box-drawing characters each frame.

**2. Trail grids rendered via ImageData (simulation worker)**
Pre-allocate one `ImageData` (RGBA, gridW×gridH). Each tick: iterate the two Float32Array trail grids → write RGBA pixels (magenta for ghost trail, yellow for prey trail, alpha = intensity). One `ctx.putImageData()` call per tick for both overlays combined.

**3. Slab forward passes in all workers**
Use `multi.activateSerializedNetwork(inputs, activationValues, stateValues, serializedNetwork, activationFns)` from `src/multithreading/multi.utils.ts`. This is the flat-array path used by Flappy Bird workers — no object-graph traversal, no per-call GC. Pre-allocate `activationValues` and `stateValues` per agent at episode start; reuse across all ticks.

**4. Observation vector pool**
Pre-allocate one `Float32Array(31)` per prey and one `Float32Array(28)` per predator at episode start. Overwrite in place each tick. No per-tick allocation.

**5. Trail decay as a single typed-array sweep**

```ts
for (let i = 0; i < preyTrail.length; i++) preyTrail[i] *= DECAY_FACTOR;
for (let i = 0; i < ghostTrail.length; i++) ghostTrail[i] *= DECAY_FACTOR;
```

Two passes over flat arrays — JIT-vectorisable, no branching.

**6. Voice propagation via sparse event-driven update**
On scream trigger (agent A has clear raycast to agent B within SCREAM_TRIGGER_RADIUS): compute attenuated signal at all other agents' positions using wall-attenuated Manhattan. Only agents within SCREAM_HEAR_RADIUS are updated. O(agents) per trigger event, not O(grid cells).

**7. Single-flight render requests (simulation worker)**
Coordinator maintains at most 1 in-flight `request-render-step` at a time (same pattern as Flappy Bird's `WorkerRequestPlaybackStepMessage`). Request ID echoed in response for ordering verification.

**8. Interleaved task queue (episode worker pool)**
Prey and predator tasks are shuffled together before dispatch. This prevents a situation where all prey tasks complete first, leaving predator tasks to run with half the workers idle at the end.

**9. Fitness normalisation across rollout seeds**
Each genome is evaluated on ROLLOUT_SEED_COUNT=3 different maze seeds per generation. Seeds are shared across all genomes in a generation (same seeds, different genomes). Composite fitness:

```
aggregateFitness = mean(rolloutFitnesses) − STABILITY_WEIGHT × stddev(rolloutFitnesses)
```

Rewards consistent performers over lucky outliers. Same pattern as Flappy Bird's multi-seed evaluation for recurrent profiles.

**10. Generation-0 warm-start (both species)**
Without warm-start, generation-0 networks produce random movement — fitness signal is near-zero, selection pressure is noise. Heuristic teacher bootstraps both populations:

- **Prey teacher**: at each junction, move away from strongest adjacent ghost trail; if no ghost trail, move toward nearest pellet.
- **Predator teacher**: at each junction, move toward strongest adjacent prey trail; if no trail, move toward loudest heard prey scream.
  Run ~50 supervised episodes per genome. Same architectural pattern as `flappy-evolution-worker.warm-start.service.ts`. Applied once at generation 0 only.

### Arms race stats contract

Episode workers return these alongside fitness. The coordinator aggregates per-genome (mean across seeds) and per-generation (population mean) for the arms race chart.

```ts
type PreyArmsRaceStats = {
  junctionChoiceEntropy: number; // bits — diversity of routing decisions at 3/4-way junctions
  screamReactionRate: number; // fraction of ticks where heard-scream > threshold and direction changed
  trailAvoidanceRate: number; // fraction of moves away from ghost trail vs toward
  pelletsCollected: number;
  ticksSurvived: number;
  escapeEvents: number; // entered within 3 cells of ghost and exited alive
};

type PredatorArmsRaceStats = {
  coordinationEvents: number; // ticks where 2+ predators approached same prey from different quadrants
  territoryFraction: number; // fraction of open cells visited
  pursuitEfficiency: number; // prey caught / ticks within 5 cells of any prey
  trailFollowingRate: number; // fraction of moves toward hotter prey trail
  screamReactionRate: number; // fraction of ticks where heard-prey-scream > threshold and moved toward source
};
```

---

## Implementation Files

### New demo folder: `examples/predator_prey/`

```
examples/predator_prey/
  browser-entry/
    browser-entry.spawn.utils.ts        ← adapted from flappy_bird/browser-entry/
    host/
      host.ts                           ← canvas setup, responsive resize (adapted from flappy_bird)
      host.types.ts
  maze/
    maze.generator.ts                   ← quarter-gen + 4-fold mirror (new)
    maze.generator.types.ts
    maze.renderer.ts                    ← wall chars + neon color (adapted from asciiMaze)
    maze.movement.ts                    ← 4-dir deltas, collision (copied from asciiMaze)
    maze.vision.ts                      ← 4-dir raycasts, pooled buffers (adapted from asciiMaze)
    maze.tunnels.ts                     ← tunnel wrap + transit cooldown (new)
    maze.pellets.ts                     ← pellet grid + respawn timer (new)
  signals/
    signals.chem-trail.service.ts       ← two Float32Array grids, decay sweep, directional reads
    signals.voice.service.ts            ← 4-dir raycast trigger, scream timer, wall-attenuated propagation
    signals.types.ts
  environment/
    environment.state.service.ts        ← episode state (agents, pellets, trails, scream timers, tick)
    environment.step.service.ts         ← tick: move all agents, capture, pellet, trail decay, voice
    environment.types.ts
  agents/
    prey.sensor.service.ts              ← builds Float32Array(31) prey observation vector per tick
    predator.sensor.service.ts          ← builds Float32Array(28) predator observation vector per tick
    agents.types.ts
  workers/
    workers.types.ts                    ← all message type unions (all 4 worker types, both directions)
    workers.coordinator.ts              ← generation barrier, task queue, mode management, snapshot updates
    workers.pool.ts                     ← episode worker pool (pre-spawned, idle tracking, task dispatch)
    workers.snapshot.service.ts         ← rolling snapshot: hall-of-fame + recent sample management
    workers.worker-count.utils.ts       ← hardwareConcurrency → worker count formula
    simulation-worker/
      simulation-worker.ts              ← entrypoint, mutable state bag, message routing
      simulation-worker.runtime.service.ts   ← init, champion genome install, maze setup
      simulation-worker.step.service.ts      ← per-tick: env step + all forward passes + signals update
      simulation-worker.snapshot.utils.ts    ← render frame packing + transfer list resolution
      simulation-worker.offscreen.service.ts ← draw maze once to OffscreenCanvas; blit each tick
      simulation-worker.types.ts
    episode-worker/
      episode-worker.ts                 ← entrypoint, stateless (no persistent NEAT state)
      episode-worker.episode.service.ts ← runs one complete isolated episode (all ticks, all agents)
      episode-worker.fitness.service.ts ← fitness aggregation across rollout seeds
      episode-worker.arms-race.service.ts    ← junction entropy, coordination event counting
      episode-worker.warm-start.service.ts   ← gen-0 heuristic teacher for prey and predator
      episode-worker.types.ts
    neat-worker/
      neat-worker.prey.ts               ← entrypoint with prey-specific config
      neat-worker.predator.ts           ← entrypoint with predator-specific config
      neat-worker.evolution.service.ts  ← shared: submit fitness, evolve, emit population-ready
      neat-worker.types.ts
  constants/
    constants.maze.ts                   ← grid size, cell size, pellet respawn ticks
    constants.agents.ts                 ← population sizes, angel release interval
    constants.signals.ts                ← trail + voice constants (see values below)
    constants.fitness.ts                ← fitness weights, stability weight, rollout seed count
    constants.workers.ts                ← snapshot sizes, update interval, rollout count
    constants.theme.ts                  ← EVA_UNITS roster + ANGELS roster (name + hex color)
```

### Source reuse (copy-paste, adapt — each demo self-contained, no shared service):

**Flappy Bird worker patterns** (direct structural model):

| Source                                              | Reuse          | Target                                                                               |
| --------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------ |
| `flappy-evolution-worker.types.ts`                  | ~80% structure | `workers/workers.types.ts` — extend for two populations                              |
| `flappy-evolution-worker.snapshot.utils.ts`         | ~70%           | `simulation-worker/simulation-worker.snapshot.utils.ts` — extend for trails + agents |
| `flappy-evolution-worker.warm-start.service.ts`     | ~60%           | `episode-worker/episode-worker.warm-start.service.ts` — two species teachers         |
| `runtime.evolution-loop.service.ts`                 | ~50%           | `workers/workers.coordinator.ts` — extend for generation barrier                     |
| `worker-channel/worker-channel.playback.service.ts` | ~80%           | integrated into `workers.coordinator.ts`                                             |

**Maze + environment patterns**:

| Source                                                 | Reuse | Target                                     |
| ------------------------------------------------------ | ----- | ------------------------------------------ |
| `asciiMaze/mazeMovement/mazeMovement.constants.ts`     | ~95%  | `maze/maze.movement.ts`                    |
| `asciiMaze/mazeVisualization.ts`                       | ~70%  | `maze/maze.renderer.ts`                    |
| `asciiMaze/mazeVision.ts`                              | ~50%  | `maze/maze.vision.ts`                      |
| `flappy_bird/browser-entry/host/host.ts`               | ~80%  | `browser-entry/host/host.ts`               |
| `flappy_bird/environment/environment.state.service.ts` | ~60%  | `environment/environment.state.service.ts` |
| `examples/architectureProfiles.ts`                     | 100%  | Reuse directly                             |

### Signal constants (`constants/constants.signals.ts`)

```ts
EMIT_AMOUNT = 1.0; // trail deposited per tick per agent
DECAY_FACTOR = 0.95; // trail multiplied per tick (~13 tick half-life)
WALL_ATTENUATION = 0.5; // voice signal multiplied per wall cell in Manhattan path
DISTANCE_DECAY = 0.85; // voice signal multiplied per Manhattan step
SCREAM_TRIGGER_RADIUS = 8; // cells — visual contact range (4-directional raycast)
SCREAM_HEAR_RADIUS = 16; // cells — max hearing range (= 2 × trigger radius)
SCREAM_DURATION_TICKS = 30; // ticks scream persists after last visual contact
```

---

## Acceptance Criteria

- Both populations run at 30+ fps on canvas with 20–40 prey and 10–20 predators.
- Prey fitness improves over generations (longer survival, more pellets gathered).
- Predator fitness improves over generations (more prey caught per episode).
- Arms race metric chart shows meaningful non-trivial trajectory (not immediate fixed-point convergence, not pure random walk).
- Predator and prey populations develop measurably different `computationType` module compositions by generation 50+.
- Voice scream signals are measurably used by receivers — mean reaction to heard screams differs statistically from baseline behavior (verifiable by ablation: disable voice input channels and compare fitness trajectory).
- Chem trail following is measurably used by predators — ghost path correlation with prey trail concentration is above chance (verifiable by ablation: disable trail input channels).
- Predator pursuit coordination emerges (two predators approach a prey from different directions more often than random by generation 30+).
- Reproduction mode distribution shifts are observable and correlated with arms race phase transitions when `modeIsEvolvable: true`.
- Rolling opponent snapshot prevents trivial one-generation fitness collapse in either population.
- Total agent network wiring cost declines over generations relative to task performance (compact specialists emerge).
- Full-screen canvas fills the available viewport and rescales correctly on window resize.

---

## Readiness Checklist (for implementation start)

**NGE prerequisites:**

- [ ] NGE Phase G prerequisites met (co-evolutionary evaluation harness implemented).
- [ ] All Phase 0 computation motif primitives implemented.
- [ ] Phase E reproduction modes (parthenogenesis, polyandric, sexual) implemented with `modeIsEvolvable` support.
- [ ] Two-population NEAT harness implemented (independent gene pools, independent species tracking).

**Environment:**

- [ ] Quarter-symmetric maze generator implemented and visually verified.
- [ ] 4-directional movement + tunnel wrap logic implemented and unit-tested.
- [ ] Chem trail typed-array grids (emit, decay, directional reads) implemented and unit-tested.
- [ ] Voice trigger (4-dir raycast) + wall-attenuated propagation implemented and unit-tested.
- [ ] Prey sensor service (31 inputs) implemented and unit-tested.
- [ ] Predator sensor service (28 inputs) implemented and unit-tested.
- [ ] Ghost release stagger logic implemented.
- [ ] Pellet respawn timer implemented.
- [ ] Deterministic episode seeding contract written (same seed → same maze, same spawn positions).
- [ ] Energy budget constants balanced.

**Worker architecture:**

- [ ] `workers.types.ts` — all message type unions written and reviewed.
- [ ] Episode worker pool (`workers.pool.ts`) — pre-spawned, idle tracking, task dispatch.
- [ ] Episode worker (`episode-worker/`) — isolated episode runner with slab forward passes and observation vector pool.
- [ ] Generation-0 warm-start (`episode-worker.warm-start.service.ts`) — both species teachers verified.
- [ ] Rolling opponent snapshot (`workers.snapshot.service.ts`) — hall-of-fame + recent sample logic.
- [ ] Generation synchronisation barrier (`workers.coordinator.ts`) — prey and predator sides independent within generation; both must complete before snapshot update.
- [ ] Fitness aggregation — mean − STABILITY_WEIGHT × stddev across ROLLOUT_SEED_COUNT seeds.
- [ ] Arms race stats collection (`episode-worker.arms-race.service.ts`) — junction entropy, coordination events.
- [ ] Prey NEAT worker + Predator NEAT worker — independent gene pools, emit `population-ready`.
- [ ] Simulation worker — display episode, offscreen maze canvas, per-tick SoA render frame packing.
- [ ] Render frame transfer list — all typed array buffers transferred (zero-copy) each tick.
- [ ] Single-flight render request — at most 1 in-flight `request-render-step` at a time.
- [ ] Operation mode switching (training-only / display-only / hybrid) working correctly.
- [ ] Worker count formula verified across 4-core, 8-core, and 16-core machines.

**Rendering and UI:**

- [ ] Neon arcade renderer (walls, pellets, trail ImageData overlays, agents, tunnels, ghost box) implemented.
- [ ] Offscreen maze canvas (draw once, blit per tick) verified.
- [ ] Full-screen responsive canvas implemented; rescales on window resize.
- [ ] Arms race observable chart (4-line real-time) implemented.
- [ ] Reproduction mode chart implemented.
- [ ] Layer toggles (ghost trail, prey trail, scream rings) working.

**Verification:**

- [ ] Ablation harness implemented (disable trail / voice input channels independently).
- [ ] Rolling opponent snapshot verified — single-generation spike does not collapse opponent fitness.
- [ ] 30+ fps confirmed in hybrid mode with 20+ prey + 10+ predators on display canvas.
- [ ] Episode worker throughput scales with core count (benchmark on 4-core vs 8-core).
