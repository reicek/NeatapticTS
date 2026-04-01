# HyperEvoDevo Racing Curriculum

**Status:** [PLANNED]

This plan is a follow-on racing benchmark for [HyperEvoDevo MorphoNEAT](HyperEvoDevoMorphoNEAT.md). It pressure-tests the evo-devo lifecycle on a three-car competitive racing task with a rich local sensorium, early-category optimal-line guidance, and behavioral drives.

This benchmark is intentionally downstream of [HyperEvoDevoMorphoNEAT.md](HyperEvoDevoMorphoNEAT.md) and [Memory_Optimization.md](Memory_Optimization.md). If this plan conflicts with either upstream plan, the upstream plan wins.

## Scope and maturity

This is a benchmark-architecture plan, not an implementation-complete spec.

- **In scope:** category structure, promotion rules, carry-state semantics, sensory families, optimal-line guidance policy, behavioral-drive vocabulary, newborn warm-start guardrails, evaluation priorities, and acceptance criteria.
- **Out of scope (for now):** final vehicle-physics implementation, rendering/UI details, worker protocol shape, exact network API, and exact reward constants.
- **Authority rule:** if this plan conflicts with [HyperEvoDevoMorphoNEAT.md](HyperEvoDevoMorphoNEAT.md) or [Memory_Optimization.md](Memory_Optimization.md), those plans remain authoritative.

## Execution alignment

This plan belongs to the same advanced-research lane as [HyperEvoDevoMorphoNEAT.md](HyperEvoDevoMorphoNEAT.md), but it is sequenced after the core Hyper contracts are stable enough to support meaningful benchmark work.

- Treat this plan as a **follow-on validation and benchmark plan**, not as a prerequisite for the core Hyper DNA/lifecycle design.
- Do not treat the racing benchmark as authority over HyperDNA, assimilation, or memory-management boundaries.
- Only begin serious implementation once the upstream Hyper plan has enough stable contracts for phenotype lifecycle, deterministic replay, and bounded growth to make benchmark results interpretable.

## Design pillars

- **Rich sensorium + sparse start:** overprovision local sensory structure, but begin with a sparse scaffold so the benchmark still rewards selective growth and later pruning.
- **Early guidance, later autonomy:** only early categories expose explicit optimal-line guidance; later categories reduce or remove that scaffold so racecraft and internalized cornering matter.
- **Competitive self-play without contact farming:** overtakes and defensive pressure are encouraged, but collision-heavy policies must be unprofitable.
- **Continuing adults + newborn refill:** promoted winners continue as the same phenotype, while empty slots are refilled from winner DNA.
- **Bounded nursery, no weight inheritance:** newborns may receive a short driving-school warm-start, but runtime weights are not written back into genotype state.
- **Egocentric observations only:** the controller should never receive a planner-style full-map oracle.
- **Visible specialization and compaction:** the benchmark should make it easy to observe sensor-family specialization, modular growth around corners and traffic, and later pruning of unused circuitry.

## Core benchmark structure

### Three-car category ladder

Each category contains exactly three cars that start in parallel on the same grid. The cars should be able to perceive one another through local egocentric signals so collision avoidance, overtaking, and defensive line choice emerge from the task instead of being scripted.

Each category defines:

- a car class with bounded speed, grip, and braking behavior,
- a deterministic circuit pack or deterministic variant set,
- a curvature and corner-severity band,
- an off-track sand profile,
- a wall-severity profile,
- a promotion threshold.

Early categories use slower cars, wider tracks, gentle corners, and forgiving recovery margins. Later categories increase speed, reduce braking margin, introduce sharper corners, enlarge dangerous sand traps, and punish poor line choice more aggressively.

### Promotion and refill rules

The ladder advances only when at least one car finishes the current category. A category should be judged over a small deterministic pack of starts or track variants so promotion is not dominated by one lucky race.

Refill policy:

- **One winner:** the winner advances unchanged, and two newborns are created from that winner DNA.
- **Two winners:** both winners advance unchanged, and one newborn child is created from aligned winner DNA.
- **Three winners:** all three winners advance unchanged.

The benchmark should support both conservative and exploratory newborns so the next category mixes stability with continued search.

## Carry-state and reset-state semantics

Promoted winners move upward as the same phenotype rather than being flattened into fresh random starts.

State that should carry across category promotion:

- current weights and biases,
- plastic or adaptive traces,
- developmental stage,
- module focus history,
- slow lifetime adaptation state,
- other category-independent policy state that belongs to the continuing individual.

State that should reset at every new race start:

- world position and heading,
- instantaneous speed and slip state,
- collision cooldowns,
- off-track timers,
- short-horizon observation-memory buffers,
- recent action-history buffers,
- other race-local episode state.

This boundary preserves the idea that the same driver survives upward through the ladder while preventing race-local residue from contaminating the next category.

## Optimal-line guidance

Tracks may expose an optional optimal-path signal, but **only early categories should provide this guidance**. The optimal line is part of the curriculum scaffold, not a permanent oracle.

Early-category optimal-line guidance may include:

- lateral error to the optimal line,
- heading error relative to the optimal-line tangent,
- target speed envelope for the next segment,
- braking urgency relative to the ideal line,
- distance to the next apex target,
- rejoin target after a disturbance.

Guidance fade policy:

- **Early categories:** optimal line is explicit and locally available.
- **Middle categories:** guidance remains available, but traffic pressure and sharper corners matter more.
- **Later categories:** optimal-line guidance is reduced, partially hidden, or removed so the controller must balance pace, traffic, slip, and collision risk without relying on an explicit ideal path channel.

## Rich sensorium

The racing benchmark should intentionally overprovide local sensory channels relative to the minimum needed for basic driving. The purpose is to give developmental structure enough raw material to specialize around braking, line tracking, traffic handling, recovery, and defensive racecraft.

### Vehicle-state senses

- current speed,
- longitudinal acceleration,
- lateral acceleration,
- yaw rate,
- heading error relative to track tangent,
- steering angle,
- steering change rate,
- throttle level,
- brake level,
- slip angle,
- traction reserve or grip proxy,
- estimated braking distance,
- stability margin,
- recent control smoothness or oscillation score.

### Track-geometry senses

- lateral offset from centerline,
- lateral offset from the optimal line when available,
- heading error relative to local path tangent,
- curvature ahead at several lookahead distances,
- upcoming corner severity,
- next apex side,
- distance to apex,
- local track width,
- exit width after the next corner,
- safe-speed envelope for the next segment,
- entry-line quality,
- exit-line quality.

### Boundary and hazard senses

- ray distances to the asphalt edge,
- ray distances to the sand boundary,
- ray distances to the wall,
- nearest wall angle,
- sand-entry risk,
- wall-impact urgency,
- rejoin corridor quality,
- off-track recovery angle,
- corner-trap severity,
- surface type under the car.

### Opponent and race-context senses

- front-left occupancy,
- front occupancy,
- front-right occupancy,
- left overlap,
- right overlap,
- rear-left pressure,
- rear pressure,
- rear-right pressure,
- relative speed to the nearest rival ahead,
- relative speed to the nearest rival behind,
- time-to-contact estimate,
- inside lane blocked or open,
- outside lane blocked or open,
- opponent stability or slide cue,
- signed forward progress,
- current place,
- gap to the car ahead,
- gap to the car behind,
- time since last clean overtake,
- time since last collision,
- wrong-direction severity.

### Short-horizon memory senses

- recent steering history,
- recent throttle history,
- recent brake history,
- recent slip history,
- recent collision impulse,
- recent off-track duration,
- recent behavioral-mode history,
- recent opponent-avoidance state.

### Initial scale guidance

The benchmark should start larger than a minimal control demo, but it should still be structurally honest.

Reasonable initial ranges:

- raw sensory surface around **72-96 scalar channels** before temporal expansion,
- effective policy input width around **96-128** once short-horizon memory and control traces are included,
- optional drive/modulation head around **6-12 channels** if explicit drive outputs are used,
- initial scaffold around **96-160 nodes**,
- initial sparse connectivity around **900-1800 connections**.

The exact numbers are secondary. The important constraint is that the benchmark begins rich enough to support specialization, while remaining sparse enough for wiring economy and adult pruning to matter.

## Behavioral drives

Behavioral drives are low-dimensional modulatory biases. Control modes are the short-lived realized states that appear under current local conditions.

Behavioral drives should bias behavior without becoming hardcoded action scripts.

Recommended drive families:

- **Pace drive:** maintain forward progress and avoid unnecessary time loss.
- **Line-adherence drive:** follow the fastest clean line when traffic permits.
- **Safety-margin drive:** preserve room to avoid walls, sand traps, and unstable entries.
- **Grip-preservation drive:** reduce sliding, over-rotation, and traction collapse.
- **Collision-avoidance drive:** avoid overlap that is likely to turn into contact.
- **Overtake-commitment drive:** convert a real passing window into a decisive move.
- **Defensive-pressure drive:** protect the inside line or deny easy overlap without initiating contact.
- **Recovery drive:** rejoin the track and restore stability after mistakes.
- **Traffic-patience drive:** back out of bad overtakes instead of forcing low-percentage moves.
- **Anti-degeneracy drive:** suppress oscillation, freezing, panic braking, and other reward-hacking behaviors.

Representative control modes:

- clear-track pace mode,
- corner-entry caution mode,
- corner-exit commit mode,
- attack or overtake mode,
- defend-inside mode,
- contact-avoidance mode,
- recovery mode,
- rejoin mode.

Implementation note: behavioral drives may be represented as explicit modulatory channels, as inspectable auxiliary outputs, or as equivalent internal submodule organization, but they should not be treated as direct actuator outputs.

## Competitive racecraft

The benchmark should encourage racecraft rather than contact-heavy “sabotage”.

Desired racecraft behaviors:

- clean overtake,
- slipstream capture and pass,
- inside-line protection,
- outside pressure without contact,
- forcing a rival onto a slower but still safe line,
- aborting a pass when the move becomes unsafe,
- holding position through braking zones.

Undesired exploit patterns:

- intentional wall-push behavior,
- contact farming,
- freezing in place to avoid risk,
- wrong-direction blocking,
- repeated unsafe rejoins.

The reward and acceptance criteria should make the undesired patterns unprofitable.

## Newborn nursery warm-start

Newborns may receive a short pre-race driving-school warm-start before entering scored category races. This is a racing benchmark equivalent of a one-time bootstrap pass: enough to move newborns out of pure-random territory, but not enough to replace the real evolutionary search.

Nursery goals:

- stay on the track,
- follow the optimal line when it is available,
- brake before high curvature,
- recover from simple slides,
- avoid obvious overlap,
- suppress wrong-direction behavior.

Warm-start guardrails:

- use the same observation surface as the scored task,
- keep the warm-start short and budgeted,
- keep topology fixed by default and adjust parameters inside the existing scaffold,
- apply post-bootstrap noise so the prior remains shared but non-rigid,
- do not let nursery training fully solve overtaking or defensive racecraft.

Promoted winners do not return to the nursery. It is a newborn-only scaffold.

## Evolution and inheritance boundary

This benchmark may look somewhat Lamarckian in practice because promoted winners continue as the same phenotype and newborns may receive bounded birth-time guidance. Even so, it must stay aligned with the no-weight-inheritance rule of [HyperEvoDevoMorphoNEAT.md](HyperEvoDevoMorphoNEAT.md).

Required boundary:

- promoted winners keep their current phenotype state,
- newborns are created from winner DNA plus optional weak birth-time priors,
- newborn warm-start does not become genotype state,
- structural assimilation remains the only slow write-back path into DNA,
- runtime weights are not stored as inherited genotype state.

This preserves the core Hyper story: structure and policies are inherited slowly; lifetime parameter state is not directly copied into DNA.

## Evaluation priorities

The racing benchmark should reduce luck the same way robust control demos do: compare lineages on small deterministic category packs rather than one-off runs.

Category ranking priorities:

1. valid finish status,
2. signed forward progress,
3. clean pace and line quality,
4. overtakes and sustained position gain,
5. compactness and wiring economy as a standing secondary pressure.

Primary penalties:

- wrong-direction travel,
- off-track time scaled by speed,
- wall impact,
- collision severity,
- repeated unsafe rejoins,
- unstable control oscillation,
- degenerate “bully” strategies that gain by repeated contact.

## Acceptance criteria

- Early categories can be completed reliably by at least one lineage under deterministic category packs.
- Later categories remain solvable as explicit optimal-line guidance fades.
- Sensor families that matter produce visible specialization around braking, line tracking, traffic handling, and recovery.
- Adult pruning reduces unused sensor wiring and overbuilt structure rather than only growing the controller indefinitely.
- Promotion is not dominated by one lucky race outcome.
- Winners carry phenotype state across categories without leaking race-local buffers into new race starts.
- Collision-heavy policies do not dominate selection.
- The benchmark produces interpretable evidence that richer sensory structure can still be compacted by wiring-economy pressure.

## Readiness checklist (for implementation start)

- [ ] Category schema drafted with deterministic pack semantics.
- [ ] Carry-state and reset-state boundary agreed.
- [ ] Early-only optimal-line guidance fade policy fixed.
- [ ] Sensor-family draft and normalization contract written.
- [ ] Behavioral-drive vocabulary and optional control-mode diagnostics agreed.
- [ ] Newborn nursery warm-start contract fixed.
- [ ] Reward and penalty contract written, including anti-contact-exploit guardrails.
- [ ] Benchmark success metrics aligned with upstream Hyper and memory-plan constraints.
