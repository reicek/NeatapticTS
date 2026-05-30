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
| Stigmergy via shared signal field       | Team radio is a typed-array shared signal, not discrete messages â€” same primitive as ant pheromone                                                      |
| Polyandric reproduction                 | Winning "queen car" is the primary genetic template; blocker and pacer drone contributions patch distinct DNA regions                                     |
| Co-evolution between populations        | Team A and Team B are fully independent NEAT populations; one team's improvements shift the other's fitness landscape                                     |
| Three-tier memory                       | Short-term: recurrent per-episode state; medium-term: rival pit patterns and teammate radio calibration; long-term: cornering priors assimilated into DNA |
| `ModulatorBroadcaster` neuromodulation  | Behavioral mode switches (sprint â†’ block â†’ hold pit â†’ rejoin) must happen within one forward pass â€” structural change is too slow                 |
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
| `GatingRouter`                           | Hard task switching: sprint vs. block vs. hold-pit vs. pit-entry vs. recovery â€” distinct policy heads, not soft interpolation               |
| `ResidualTap`                            | Track geometry and race-position signals flow as a residual highway available to all processing zones without wiring cost                     |
| Identical-DNA role differentiation       | All team members share one genotype; queen/blocker/pacer specialization emerges from driving history and radio interaction alone              |
| Stigmergy via radio field                | 6â€“8 dimensional team radio written and read by all teammates â€” same typed-array primitive as ant pheromone; no addressed messages         |
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

## Completed phases snapshot (compressed)

- Phase 1 [DONE]: Tier 0 harness packetized, implemented, validated, documented, and closed in `examples/racing_curriculum/`.
- Phase 2 [DONE]: Tier 1-2 solo NGE controller seam shipped (guidance fade + Tier 2 self-radio), validated, and closed.
- Phase 3 [DONE]: Tier 3 2v2 roles boundary shipped (two-population harness + 91-channel observation + Tier 3 worker), validated, and closed.
- Phase 4 [DONE]: Tier 4 tires+pits seams shipped (degradation, pit occupancy, 95-channel observation, renderer overlays), validated, and closed.
- Phase 5 [DONE]: Tier 5 six-car 3v3 simulation shipped (`agentCount=6`, `radioField=42`, byte-stable 95-channel observation), validated, and closed.
- Plan-sync status for closed history: no recorded failures.

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

| Roadmap phase                         | Delivers                                                                                                                                                                                                                                                                                         | Gating prerequisite                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| **Tier 0 â€” Visual Driving Harness** | Neon-arcade browser shell (canvas left / network right / visualizer last), procedural track generator + validation, fixed-timestep physics, solo car + baseline controller, square-outline car + degradation-colored tires, `RacingRenderFrame` worker protocol, network-view of the focused car | None â€” startable now                                             |
| **Tier 1â€“2 â€” Solo NGE driving**   | Replace baseline controller with an evolved NGE genome; single-car racecraft; optimal-line fade; radio present but self-only                                                                                                                                                                     | NGE Phase A/B + Phase 0 archetypes                                 |
| **Tier 3 â€” 2v2 roles**              | Two identical-DNA teammates; role differentiation; team radio between teammates                                                                                                                                                                                                                  | Two-population harness + stigmergy field (field already available) |
| **Tier 4 â€” Tires + pits**           | Tire degradation budget, pit stops, pit-entrance blocking; tire/pit panels activate                                                                                                                                                                                                              | Tire degradation model                                             |
| **Tier 5â€“6 â€” Full co-evolution**  | 3v3, polyandric reproduction, rolling opponent snapshots, `modeIsEvolvable`; co-evolution dashboards activate                                                                                                                                                                                    | Reproduction modes + co-evolution loop                             |

Phase 1 below (planning) packetizes the SDLC steps that build the **Tier 0** boundary first; later roadmap phases are packetized as their prerequisites land.

two-population NEAT harness seam at `src/neat/nge-collective/`, likely

### Tier Completion Matrix

This matrix is the working table for turning the curriculum into a fully complete tier ladder.
It separates what each tier changes, how many cars are on track, what ends the tier, and what
still needs to exist before that tier can be considered complete. The table is intentionally
redundant with the roadmap so it can be used later as a direct expansion checklist.

| Tier   | On-track roster / spawn         | What changes on this tier                                                                                                                                | Finish condition                                                                                                         | Current completion gate                                                                       | Notes for later expansion                                                                                          |
| ------ | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Tier 0 | 1 car, solo spawn               | Browser visual driving harness, deterministic track, fixed-timestep physics, baseline controller, focused-car network view, packed render frame contract | Tier 0 scaffold criteria pass: deterministic replay, solo laps, stable frame packing, layout, and style contract         | Still bounded to the visual harness boundary                                                  | This is the first honest implementation boundary and should stay future-proof for later multi-car expansion        |
| Tier 1 | 1 car, evolved lane             | Live NGE controller replaces the scripted baseline; optimal-line guidance is strongest here                                                              | Tier 1 lane is reproducible and a lineage can complete laps under the deterministic race pack                            | Requires NGE Phase A/B and Phase 0 motifs                                                     | This is the first real learning tier, but still single-car only                                                    |
| Tier 2 | 1 car, widened observation lane | Same single-car race, but the observation seam widens and self-radio appears                                                                             | Tier 2 finish is the same lane completed with the widened policy inputs and stable self-radio behavior                   | Requires the same single-car controller seam plus the Tier 2 observation tail                 | Use this tier to validate the controller seam before team behavior exists                                          |
| Tier 3 | 2 cars per team, 2v2 total      | First team phase: identical-DNA teammates, role emergence, and team radio between teammates                                                              | Tier 3 completes when two-population harness, role differentiation, and teammate radio are stable without scripting      | Requires two-population harness and the stigmergy field primitive                             | This is the first true multi-car spawn step; it is where the roster begins to expand                               |
| Tier 4 | 2 cars per team, 4 total        | Tire degradation, pit stops, pit-entrance blocking, and pit/tire UI become active                                                                        | Tier 4 completes when tire state, pit occupancy, and pit-stop flow are all working and measurable                        | Requires the tire degradation model and pit semantics                                         | This tier adds endurance strategy instead of increasing roster size                                                |
| Tier 5 | 3 cars per team, 6 total        | Full six-car 3v3 pack, polyandric reproduction hooks, rolling opponent snapshots, and the expanded radio/pit transport contract                          | Tier 5 completes when six-car simulation is stable, byte-stable, and the Tier 5 benchmark criteria pass                  | Requires reproduction modes plus the co-evolution loop                                        | This is the second roster expansion point and the current highest implemented car count                            |
| Tier 6 | 3 cars per team, 6 total        | Full co-evolution behavior, reproduction-mode evolution, and the arms-race observability surfaces                                                        | Tier 6 completes when the co-evolutionary arms race shows non-trivial alternating advantage instead of collapse or noise | Requires the upstream reproduction and mode-evolution surface that is still out of scope here | Tier 6 is the current end-state target, but it is intentionally left blocked until the upstream prerequisites land |

#### Tier completion dimensions

Use the following dimensions when expanding the plan into a more detailed completion map for each tier:

| Dimension               | What to capture                                                                                    | Why it matters                                                                             |
| ----------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Roster / spawn          | Car count, team split, whether the tier changes the active spawn shape                             | This is the most visible user-facing expansion from tier to tier                           |
| Controller seam         | Which controller surface is active, and whether the tier changes observation width or policy heads | Prevents later tiers from silently changing the contract the browser demo depends on       |
| Environment surface     | Track, physics, tire, pit, and radio capabilities that are active on the tier                      | Separates what the environment owns from what the browser only renders                     |
| Worker / frame contract | Packed frame schema, transfer rules, and any tier-specific buffers                                 | Keeps the browser, worker, and headless paths aligned                                      |
| Completion gate         | Observable pass condition for the tier                                                             | Makes each tier finishable instead of just “more complete”                                 |
| Upstream blocker        | The prerequisite that must land before the tier can be considered complete                         | Prevents the plan from overpromising capabilities that the core stack does not yet support |

#### Tier completion expansion ledger (2026-05-30)

This ledger converts the matrix into concrete completion packets and stop points. It is the source of truth for what is already closed versus what still must run before this workstream can be declared fully complete.

| Tier   | Completion state        | Already complete evidence                                                                      | Remaining implementation / validation / docs / logging                                                           |
| ------ | ----------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Tier 0 | Complete                | Solo harness, deterministic replay, frame packing, and browser shell shipped in earlier phases | None beyond preserving regression coverage in ongoing follow-up validation                                       |
| Tier 1 | Complete                | Live NGE controller replaced scripted baseline; deterministic lap completion evidence recorded | None; keep deterministic controller behavior green in follow-up slices                                           |
| Tier 2 | Complete                | Widened observation lane and self-radio semantics validated and logged                         | None; keep Tier 2 observation/radio contracts regression-covered                                                 |
| Tier 3 | Complete                | 2v2 role seams, teammate radio, and two-population harness shipped and validated               | None; retain parity in Tier 5+ regression matrix checks                                                          |
| Tier 4 | Complete                | Tires, pits, occupancy, and Tier 4 transport semantics shipped and validated                   | None; ensure pit/tire semantics stay stable under six-car soak                                                   |
| Tier 5 | In follow-up completion | Six-car 3v3 simulation, 95-channel byte-stable observation, and Tier 5 worker seam shipped     | Close Phase 6 Step 05-07 with full green matrix + browser soak + docs/log compression evidence                   |
| Tier 6 | Blocked upstream        | Tier 6 target and blocker contract documented in roadmap and matrix                            | Author blocked-step packet with explicit upstream ownership and no local overreach; keep closure criteria honest |

#### Executable tier completion packets (sequential MCP queue)

These packets replace pre-planning notes with executable, one-step-at-a-time handoff blocks.
Active `[WIP]` ownership remains in the phase step packets; this queue mirrors that order.

##### Packet 1 — Tier 5 closure validation matrix (maps to Phase 6 Step 05)

```yaml
phase: 6
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: true
next_step: 'Phase 6 Step 06 — Tier 5 closure docs refresh (manual confirmation gate first)'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
validation:
  - 'npm run quality:folder -- --folder=examples/racing_curriculum'
  - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/racing_curriculum/browser-entry --testPathPatterns=examples/racing_curriculum/renderer --testPathPatterns=examples/racing_curriculum/workers/simulation-worker'
  - 'npm run build:racing-curriculum'
  - 'npm run test:silent'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

**Step objective:** Prove Tier 5 closure-readiness by running the full follow-up green matrix plus manual browser soak evidence without widening scope outside `examples/racing_curriculum/`.

**Context the agent must know:**

- Tier 5 implementation seams are already shipped; this packet is validation-first closure proof.
- Tier 6 remains blocked upstream; this packet must not implement `src/` co-evolution features.
- Current risk seams: promotion continuity, stage text sync, panel-readiness honesty, six-car parity.

**Execution steps:**

1. Run the required automation matrix in listed order.
2. Run manual browser soak for repeated progression/restart loops and capture concise pass/fail notes.
3. If regressions appear, stop and route back to the smallest prior phase packet instead of continuing.

**Stop conditions:**

- **Done:** all matrix gates pass and manual soak evidence is recorded.
- **Blocked:** any gate fails with no bounded examples-local fix path.
- **Route-back:** any failure requiring code change returns to `04-implementing` with a narrow defect packet.

**Required validation:** Run every command in the YAML `validation` block and include manual soak evidence summary.

**Plan update requirement:** Record command outcomes, soak notes, and set only Phase 6 Step 06 to `[WIP]` after explicit manual confirmation.

##### Packet 2 — Tier 5 closure documentation refresh (maps to Phase 6 Step 06) [PLANNED]

```yaml
phase: 6
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: true
next_step: 'Phase 6 Step 07 — Tier 5 closure logging and compression'
skills:
  - 'educational-docs'
  - 'tracker-handoff'
validation:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npm run docs'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

**Step objective:** Refresh owner-local docs/comments for the closed Tier 5 follow-up behavior and explicit non-goals.

**Context the agent must know:**

- Execute only after Packet 1 evidence is complete and manually confirmed.
- Keep language honest: Tier 5 closure in this plan is examples-local hardening, not Tier 6 completion.
- Preserve prior completed-phase history exactly; append only new closure deltas.

**Execution steps:**

1. Update only docs/comments touched by the validated Tier 5+ follow-up seams.
2. Keep blocked-upstream Tier 6 language explicit and unchanged in meaning.
3. Prepare concise closure-ready notes for Packet 3 logging.

**Stop conditions:**

- **Done:** docs/comments updated and validation commands pass.
- **Blocked:** required docs regeneration fails or surfaces unresolved drift.
- **Route-back:** if documentation reveals behavior mismatch, route to the smallest correcting phase.

**Required validation:** Run YAML validation commands and capture evidence in plan validation section.

**Plan update requirement:** Mark Step 06 `[DONE]`, advance Step 07 to `[WIP]`, and retain single active step.

##### Packet 3 — Tier 5 closure logging and closure decision (maps to Phase 6 Step 07) [PLANNED]

```yaml
phase: 6
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-logging.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: true
next_step: 'Phase 7 Step 01 — Tier 6 blocked-state governance packet'
skills:
  - 'tracker-handoff'
  - 'green-validation-gates'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

**Step objective:** Compress Phase 6 to concise done notes and make an explicit closure decision with evidence.

**Context the agent must know:**

- Phase 6 may close only when Packet 1 matrix + soak evidence and Packet 2 docs evidence are present.
- If any closure criterion is missing, keep Phase 6 open with a narrow unresolved-defect packet.
- No Tier 6 capability claims are allowed in closure text.

**Execution steps:**

1. Verify Packet 1 and Packet 2 evidence exists and is internally consistent.
2. Compress verbose Phase 6 logs into concise completion notes.
3. Record closure decision and queue Phase 7 blocked-state governance step.

**Stop conditions:**

- **Done:** closure decision is explicit and evidence-backed.
- **Blocked:** evidence is incomplete or contradictory.
- **Route-back:** reopen the smallest prior packet needed to fix evidence gaps.

**Required validation:** Run plan-sync validation and include output in validation evidence.

**Plan update requirement:** Mark Step 07 `[DONE]` only when criteria are met; then move to Phase 7 Step 01 `[WIP]`.

##### Packet 4 — Tier 6 blocked-state governance and escalation checks (maps to Phase 7 Step 01) [PLANNED]

```yaml
phase: 7
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: true
next_step: 'Planner-defined by this step (blocked-state maintenance or upstream-unblocked execution queue)'
skills:
  - 'phase-handoff-workflow'
  - 'plan-alignment'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

**Step objective:** Keep Tier 6 explicitly blocked with current upstream ownership, escalation path, and precondition checks while preventing scope overreach.

**Context the agent must know:**

- Tier 6 is blocked on upstream `src/` prerequisites outside this bounded examples-local plan closure.
- Required blocker owners: `src/neat.ts` reproduction exposure, `src/neat/nge-collective/` mode-switching wiring, hall-of-fame opponent evaluation seam.
- This packet is governance-only unless blockers are objectively removed.

**Execution steps:**

1. Re-validate blocker list and ownership against current repo state.
2. Record escalation/precondition checks and explicitly state no local Tier 6 implementation is authorized.
3. Author next packet set: either continued blocked maintenance or, if unblocked, a new bounded execution queue.

**Stop conditions:**

- **Done:** blocker status is current, evidence-backed, and escalation path is explicit.
- **Blocked:** blocker ownership is ambiguous or contradictory.
- **Route-back:** if contradictions appear, escalate via `00-helping` cross-tier helper before planning further execution.

**Required validation:** run plan-sync and capture blocker-state evidence references.

**Plan update requirement:** Keep Tier 6 marked blocked unless upstream prerequisites are demonstrably satisfied; never claim Tier 6 completion from this packet.

### Phase 1 - Racing Curriculum Planning [DONE]

Compressed outcome: Tier 0 harness boundary was packetized, implemented, validated, documented, and closed inside `examples/racing_curriculum/`.

### Phase 2 - Tier 1-2: Solo NGE Driving [DONE]

Compressed outcome: scripted baseline was replaced with the live NGE controller seam, guidance fade and Tier 2 self-radio landed, validations stayed green, and the phase closed without `src/` edits.

### Phase 3 - Tier 3: 2v2 Roles [DONE]

Compressed outcome: team-isolated two-population harness + Tier 3 observation/worker seams shipped and validated; documentation refreshed; phase closed.

### Phase 4 - Tier 4: Tires + Pits [DONE]

Compressed outcome: tire degradation, pit occupancy/stop flow, Tier 4 observation/worker transport, and renderer overlays shipped and validated; phase closed.

### Phase 5 - Tier 5-6: Full Co-Evolution (Honest Partial: Tier 5 Simulation) [DONE]

Compressed outcome: six-car 3v3 simulation seam shipped (`agentCount=6`, `radioField=42`, byte-stable 95-channel observation), validated, documented, and closed. Upstream reproduction/mode-evolution work remains explicitly out of scope here.

### Phase 6 — Post-Closure Follow-up: UI Completion + Tier 5+ Stability [WIP]

**Phase objective:** Close remaining placeholder UI/visualizer surfaces and harden Tier 5+ browser-demo behavior with a validation-first pass, including promotion continuity and stage text synchronization under repeated progression.

**Honest scope (bounded):**

- In scope:
  - Remaining deferred/placeholder UI surfaces in `examples/racing_curriculum/browser-entry/` and `examples/racing_curriculum/renderer/`.
  - Tier 5+ validation matrix (automated + manual) for six-car progression behavior.
  - Browser regression soak for tier promotion transitions, stage subtitle/footer/tooltip sync, and panel-readiness accuracy.
  - Examples-local bug fixes required for demo correctness and operator trust.
- Out of scope / non-goals:
  - Upstream blocked reproduction work (`src/neat.ts` polyandric race-loop exposure and `modeIsEvolvable` strategy plumbing).
  - New `src/` architecture expansion unrelated to observed racing-demo defects.
  - New co-evolution dashboard features beyond regression containment.

#### Step 01 — Follow-up packetization and risk framing [DONE]

Done note (2026-05-30): locked the Phase 6 follow-up boundary to examples-local UI completion and Tier 5+ stability hardening, authored bounded Steps 02-07, fixed explicit non-goals to prevent upstream reproduction scope creep, and refreshed handoff sequencing so the next active frontier is Step 02 research mapping.

#### Step 02 — UI and Tier 5+ boundary research map [DONE]

Done note (2026-05-30): completed read-only reconnaissance across `examples/racing_curriculum/browser-entry/`, `renderer/`, and `workers/simulation-worker/` plus plan contracts. Confirmed concrete follow-up seams:

- Deferred/future-facing UI copy still present in active panels (Network view / Race Pack readiness language) and placeholder chart expectations remain plan-visible.
- Browser runtime path still behaves as a mostly single-car playback seam while late-tier six-car worker/radio/pit contracts are only partially surfaced in UI.
- Tier progression still uses a fallback lap-count rule that does not yet model the plan's cross-team promotion fairness contract.
- Tier 5+ risk seams to gate in Step 03: promotion continuity, stage subtitle/footer/tooltip sync, six-car render parity, pit occupancy contention under 3v3, teammate-radio visibility consistency, and Tier 6 observation-cap ambiguity.

Step 03 red-test contract focus:

- Promotion fairness/continuity regression tests (repeated promotion cycles).
- Stage-readiness and narrative-sync DOM contract tests.
- Six-car renderer and packed-frame parity tests.
- Tier 5+ pit and radio guardrail tests for contention/visibility semantics.

#### Step 03 — Red testing for promotion and UI contracts [DONE]

Done note (2026-05-30): authored focused red contracts in owner-local boundaries only:

- `examples/racing_curriculum/browser-entry/browser-entry.progression.test.ts`: promotion fairness seam now expects Tier 5 hold behavior when cross-team fairness evidence is unavailable.
- `examples/racing_curriculum/browser-entry/browser-entry.test.ts`: panel-readiness parity seam now rejects future-facing/deferred placeholder copy in active UI.
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5-renderer.test.ts`: six-car renderer parity seam now requires explicit focused-car metadata on the Tier 5 race pack.
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.test.ts`: Tier 5+ guardrails now assert pit-contention wait-slot metadata and Tier 3-parity radio readability semantics.

Focused red evidence (intentional):

- `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="examples/racing_curriculum/browser-entry/browser-entry.progression.test.ts|examples/racing_curriculum/browser-entry/browser-entry.test.ts|examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.test.ts|examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5-renderer.test.ts"`
- Result: `FAIL 4/4 suites, 5 failed / 20 total tests`.
- Failure seams: Tier 5 promotion fairness hold (`didAdvance` true vs expected false), active panel readiness copy still includes `Future-facing`/`Deferred`, Tier 5 race pack missing `focusCarIndex` metadata, pit contention tuple still width 4 (expected 6), Tier 5 radio readability omits self row (`[1, 2]` vs `[0, 1, 2]`).

Step 04 green condition:

- Implement owner-local fixes so the focused command above turns green without widening scope beyond `examples/racing_curriculum/browser-entry/**`, `examples/racing_curriculum/renderer/**`, and `examples/racing_curriculum/workers/simulation-worker/**`.

#### Step 04 — Implement UI completion and Tier 5+ fixes [DONE]

Done note (2026-05-30): implemented owner-local fixes for the Phase 6 red seams:

- Promotion fallback now blocks auto-promotion once the curriculum reaches Tier 4+ (`MAX_FALLBACK_AUTOPROMOTION_TIER`) so late-tier progression does not overrun fairness checks.
- Active panel readiness copy removed `Future-facing` / `Deferred` labels in browser-entry status rows.
- Tier 5 worker race pack now includes explicit focused-car metadata (`focusCarIndex`) and expanded pit contention metadata (`pitStatus.length = 6`) with wait-slot sentinels.
- Tier 5 radio readability now includes self-row parity (`[0, 1, 2]` for Team A and `[3, 4, 5]` for Team B).
- Focused Step 03 red-test slice turned green: `PASS 4/4 suites, 20/20 tests`.

#### Step 05 — Green validation and regression soak [WIP]

```yaml
phase: 6
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[WIP]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: true
next_step: 'Step 06 — Documentation deltas and known-limits refresh'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
validation:
  - 'npm run quality:folder -- --folder=examples/racing_curriculum'
  - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/racing_curriculum/browser-entry --testPathPatterns=examples/racing_curriculum/renderer --testPathPatterns=examples/racing_curriculum/workers/simulation-worker'
  - 'npm run build:racing-curriculum'
  - 'npm run test:silent'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

Run focused automation plus manual browser soak matrix across repeated tier promotions and six-car traffic scenarios.

- Current browser fallback promotion rule advances by 3 laps per tier, but it hard-stops automatic promotion at Tier 4; reaching Tier 5 requires the future fairness/co-evolution promotion path rather than extra laps alone.
- Confirm the right-side network panel renders the live NGE controller graph instead of the placeholder seam and stays in sync when the controller tier is rebuilt.
- If a true specific-network selector is still needed after the live graph lands, capture that as a bounded follow-up note rather than widening this phase in place.

#### Step 06 — Documentation deltas and known-limits refresh [PLANNED]

```yaml
phase: 6
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: true
next_step: 'Step 07 — Logging, compression, and closure decision'
skills:
  - 'educational-docs'
  - 'tracker-handoff'
validation:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npm run docs'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

Update owner-local docs/comments to reflect completed UI surfaces, validated Tier 5+ behavior, and remaining explicit non-goals.

#### Step 07 — Logging, compression, and closure decision [PLANNED]

```yaml
phase: 6
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-logging.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: true
next_step: 'Phase 7 Step 01 — Tier 6 blocked-state governance and escalation checks'
skills:
  - 'tracker-handoff'
  - 'green-validation-gates'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

Compress Phase 6 history and either close the follow-up pass (if matrix gates pass) or leave a narrow unresolved-defect packet.

**Phase 6 validation baseline:**

```text
npm run quality:folder -- --folder=examples/racing_curriculum
npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/racing_curriculum/browser-entry --testPathPatterns=examples/racing_curriculum/renderer --testPathPatterns=examples/racing_curriculum/workers/simulation-worker
npm run build:racing-curriculum
npm run test:silent
node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md
```

### Phase 7 — Tier 6 blocked-state governance [PLANNED]

**Phase objective:** Keep Tier 6 explicitly blocked until upstream reproduction/mode-evolution seams are available, with clear escalation ownership and no `src/` overreach from this bounded racing follow-up plan.

**Phase progression rule:** Start at Step 01 after Phase 6 closure decision is complete. Step 01 must either author continued blocked-maintenance packets or, if prerequisites are unblocked, author a fresh execution queue before any implementation work begins.

#### Step 01 — Revalidate blockers and author blocked-state queue [PLANNED]

```yaml
phase: 7
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: true
next_step: 'Step 02 — Planner-defined by Step 01 (blocked maintenance or unblocked execution path)'
skills:
  - 'phase-handoff-workflow'
  - 'plan-alignment'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
```

**Step objective:** Produce a current blocker-ownership verdict and a self-contained next packet set without claiming Tier 6 completion.

**Context the agent must know:**

- Tier 6 is still upstream-blocked on reproduction exposure, mode-evolution wiring, and hall-of-fame evaluation seams.
- This phase is governance and sequencing only unless blockers are proven resolved.
- Any blocker ambiguity routes to `00-helping` escalation instead of speculative local implementation.

**Execution steps:**

1. Verify each Tier 6 blocker and its owner against current repository state.
2. Record escalation and precondition checks with explicit pass/fail outcomes.
3. Author next sequential packet(s) for either blocked maintenance or unblocked execution.

**Stop conditions:**

- **Done:** blocker verdict and next packet queue are explicit and validated.
- **Blocked:** blocker ownership or readiness remains ambiguous.
- **Route-back:** escalate via cross-tier helper before any further phase expansion.

**Required validation:** run plan-sync and include blocker-verdict evidence references.

## **Plan update requirement:** keep exactly one active `[WIP]` step in the full plan and never mark Tier 6 complete while upstream blockers remain.

## Validation gates

Plan-sync gate for this plan's current state and most recently closed frontier.

### Latest validation evidence

- 2026-05-30: validation-MCP blocker was resolved by updating the Step 05 Jest command to an MCP-safe format without shell metacharacters (`--testPathPatterns=` repeated per boundary). Allowlisted focused regression then passed cleanly (`PASS 10/10 suites, 42/42 tests`).
- 2026-05-30: Phase 6 Step 05 automation run stayed green after debt cleanup: `npm run quality:folder -- --folder=examples/racing_curriculum`, `npm run build:racing-curriculum`, `npm run test:silent` (`PASS 475/475 suites, 5178/5178 tests`), and plan-sync validation all passed.
- 2026-05-30: racing-demo tech-debt cleanup pass completed before Phase 6 Step 05 execution: quality metrics stayed green (`npm run quality:folder -- --folder=examples/racing_curriculum`) and focused browser-entry/controller regression slice stayed green (`PASS 10/10 suites, 33/33 tests`).
- 2026-05-30: plan-sync re-run after tier packetization updates passed cleanly — `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md` returned `PASS plan sync: 0 errors, 0 warnings` (`ok: true`).
- 2026-05-30: Phase 6 Step 04 completed by `04-implementing`; owner-local fixes landed in `examples/racing_curriculum/browser-entry/browser-entry.ts` and `examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.ts` plus aligned Tier 5 tests. The focused red-test command turned green (`PASS 4/4 suites, 20/20 tests`). Broader follow-up gates also passed in-session: `npm run quality:folder -- --folder=examples/racing_curriculum`, `npm run build:racing-curriculum`, and `npm run test:silent`.
- 2026-05-30: Phase 6 Step 02 completed by `02-researching`; read-only boundary mapping confirmed deferred/future-facing UI surfaces, single-car-vs-six-car runtime parity gaps, promotion fallback-vs-plan fairness mismatch, and Tier 5+ risk seams (promotion continuity, stage narrative sync, six-car render parity, pit contention, radio visibility, Tier 6 cap ambiguity). Phase 6 Step 03 was advanced to [WIP] for red-test authoring, and plan-sync passed (`PASS 0 errors, 0 warnings`).
- 2026-05-30: User-requested follow-up orchestration added `### Phase 6 — Post-Closure Follow-up: UI Completion + Tier 5+ Stability [WIP]` with bounded Steps 01-07, explicit non-goals, and a Tier 5+ regression matrix baseline. Plan handoff was updated to make Phase 6 Step 01 the active frontier, and plan-sync passed (`PASS 0 errors, 0 warnings`).
- Historical closure tail (Phases 1-5, detailed step logs, and prior validation granularity) intentionally compressed in this tracker. Durable outcomes remain captured in the completed-phase summaries above and the current Phase 6 handoff below.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Plan status: Phase 6 follow-up is [WIP] (UI completion + Tier 5+ stability)

Phases 1–5 are complete; Phase 6 is the active follow-up frontier. Keep work bounded to `examples/racing_curriculum/` unless a proven blocker requires escalation.

What was built:
- Phase 1: Planning and design (Tier 0–6 architecture)
- Phase 2: Tier 0 Visual Driving Harness (solo car, canvas renderer, baseline controller, procedural track)
- Phase 3: Tier 3 2v2 Roles (two-population NEAT harness, team radio, role differentiation seam, `nge-collective`)
- Phase 4: Tier 4 Tires + Pits (tire degradation, pit stops, pit-entrance blocking, 95-channel Tier 4 observation)
- Phase 5: Tier 5 six-car simulation (3v3 pack, 42-float radio field, byte-stable 95-channel observation, first-car-wins 3-teammate pit)

Active follow-up targets (Phase 6):
- Replace remaining deferred/placeholder UI surfaces.
- Render the right-side NGE network panel with the live deterministic controller graph so the browser host no longer shows an empty network slot.
- Keep the network panel honest about what it is inspecting today: one focused NGE controller, with a future-specific-network picker captured only if this pass reveals a real need for it.
- Execute Tier 5+ validation matrix (automated + manual).
- Run promotion + stage-text regression soak under repeated progression.
- Keep panel-readiness language honest (no overclaiming unfinished runtime surfaces).

Remaining upstream prerequisites (not in scope of this plan):
- Full polyandric reproduction loop: `src/neat.ts` must expose Phase E reproduction surface
- `modeIsEvolvable` strategy switching: requires `src/neat/nge-collective/` wiring
- Tier 6 hall-of-fame opponent evaluation: blocked on both above

Current entry step: Phase 6 Step 05 [WIP] (green validation and regression soak) following completed Step 04 implementation.

Sequenced next steps after Step 05:
- Phase 6 Step 06 (docs refresh) — start only after explicit manual confirmation on Step 05 evidence.
- Phase 6 Step 07 (logging/compression + closure decision) — start only after explicit manual confirmation on Step 06.
- Phase 7 Step 01 (Tier 6 blocked-state governance) — opens only after Phase 6 closure decision is recorded.

Closure rule: do not archive this plan until Phase 6 validation matrix and browser soak pass with explicit evidence.
```
