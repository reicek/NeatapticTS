# Racing Curriculum Tier 1 Browser Demo Follow-Up Defects

Research artifact for `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` —
Phase 8, Step 18 — follow-up browser demo defects.

## Question

After Step 18 implementation in the racing-curriculum browser demo
(`examples/racing_curriculum/browser-entry/browser-entry.ts`), why does (1) only the
blue car appear to evolve via NGE while the red car behaves deterministically, and
(2) the blue NGE car fail to learn productive driving, sometimes starting well and
then regressing?

## Evidence

### 1. Only car 0's adaptation engine is ticked

- `buildPerCarControllers` creates one deterministic network and one
  `createNgeController` wrapper per car (`browser-entry.ts:436-452`).
- `buildPerCarAdaptationEngines` creates an independent `RuntimeAdaptationEngine`
  for every car (`browser-entry.ts:457-470`).
- The fixed-timestep loop computes controls for every car via
  `resolvePerCarControls` (`browser-entry.ts:533-537`), but only gathers evidence
  and calls `adaptOnTick` for car 0 (`browser-entry.ts:538-556`).
- Therefore the red car (team 1, car 1) has an engine but it is never advanced;
  its network remains the deterministic seed.

### 2. Tier promotion resets cars 1..N

- On tier promotion, car 0 keeps its evolved substrate:
  `remapControllerNetworkForObservationTier(focusedControllerNetwork,
  activeObservationTier)` (`browser-entry.ts:587-590`) and is reused
  (`browser-entry.ts:605-610`).
- Cars 1+ receive brand-new deterministic networks:
  `createDeterministicRacingControllerNetwork(activeObservationTier)`
  (`browser-entry.ts:613-620`).
- Adaptation engines are then rebuilt for the new roster
  (`browser-entry.ts:623`).
- So even if the loop were fixed to tick every car, any growth accumulated by
  cars 1+ would be discarded at the next promotion.

### 3. Both teams start with equivalent substrates

- Tier 1 layout is `[0, 1]` (`browser-entry.ts:179`), so car 0 is blue
  (`TEAM_BLUE_INDEX = 0`, `browser-entry.ts:189`) and car 1 is red
  (`TEAM_RED_INDEX = 1`, `browser-entry.ts:195`).
- `createDeterministicRacingControllerNetwork` builds the same MLP shape and
  applies identical activations/biases for every car
  (`browser-entry.ts:1536-1548`, `browser-entry.ts:1557-1599`).
- The substrate asymmetry is therefore not in the seed but in which engines are
  ticked.

### 4. Score signal is a narrow heading-alignment proxy

- The loop pushes only `focusedTickResult.evidence.headingAlignment01` into the
  rolling score window (`browser-entry.ts:548`) and passes it to `adaptOnTick`
  (`browser-entry.ts:552-556`).
- `evaluateRacingTrendScore` returns `scoreMean + scoreTrend * 0.5` and
  intentionally omits a size penalty
  (`runtime.adaptation.ts:466-482`).
- It never rewards track progress, speed, lap completion, or staying on track.
  A car that is well-aligned but slow or stationary can score highly; a car that
  picks up speed but temporarily misaligns receives a negative trend and may roll
  back useful structure.

### 5. Adaptation gating is too permissive/aggressive

- Defaults: `improvementThreshold = 0`, `mutationCooldownTicks = 0`,
  `rollbackCooldownTicks = 0`, cadence `every_tick`
  (`runtime.adaptation.ts:135-145`, `runtime.adaptation.ts:147`).
- `improvementThreshold = 0` means any morph that does not strictly lower the
  score is committed, including neutral changes that add useless structure.
- No cooldowns and `every_tick` cadence cause the network to be mutated every
  simulation tick before previous changes have time to produce a stable trend.

### 6. Rollback restores topology but leaves global side effects

- `adaptOnTick` snapshots with `tickInput.network.toJSON()`
  (`runtime.adaptation.ts:302`) and, if the candidate does not improve, calls
  `restoreNetworkSnapshot` (`runtime.adaptation.ts:359-360`), which rehydrates the
  network via `Network.fromJSON` and overwrites the target object's own
  properties (`runtime.adaptation.ts:673-685`).
- `runNgeLifecycle` mutates the passed network in place via
  `applyMorphDeltas` (`src/neat/neat.nge-lifecycle.ts:197-201`) and also
  advances the global connection innovation counter via
  `syncInnovationCounterToNetwork`
  (`src/neat/neat.nge-lifecycle.ts:191`, `src/neat/neat.nge-lifecycle.ts:252-260`).
  That counter is not restored by the JSON rollback, so the live network can carry
  stale innovation IDs into the next adaptation tick.

## Decision

The two demo defects are caused by a combination of wiring and policy bugs in the
browser harness, not by a fundamental failure of the NGE growth engine:

1. **Asymmetric adaptation wiring:** the animation loop must tick every car's
   adaptation engine, not only car 0. On tier promotion, every car's remapped
   network must be carried forward, or at a minimum the same per-car carry policy
   must apply so red/blue remain symmetric.
2. **Inadequate score signal:** the rolling window should be fed a composite
   driving-quality signal that includes track progress (distance along spline /
   lap count), forward speed, and an on-track/heading-alignment component, rather
   than heading alignment alone.
3. **Loose adaptation gating:** raise `improvementThreshold` above zero, add
   non-zero mutation/rollback cooldowns, and consider moving cadence from
   `every_tick` to `every_n_ticks` so the score window can stabilize.
4. **Rollback hygiene:** ensure the innovation counter (and any other global/
   process-level state touched by `runNgeLifecycle`) is captured and restored
   alongside the network JSON, or refactor the engine to apply planned morphs to
   a cloned network and commit only after the candidate passes.

## Risks

| Risk | Owner | Mitigation |
| --- | --- | --- |
| Switching the harness to worker-authoritative evolution may make the local `adaptOnTick` loop obsolete; ensure the two paths are not both running and double-mutating. | 04-implementing | Boundary map the worker→host message flow before wiring the new loop. |
| A composite driving score needs careful normalization; a badly scaled reward can still mislead the engine. | 03-red-testing / 04-implementing | Add red tests for score behavior on good vs bad driving traces. |
| Raising `improvementThreshold` and adding cooldowns may slow visible growth; browser-ui validation must confirm growth still occurs. | 05-green-testing | Measure node count over a fixed number of ticks in the browser smoke test. |
| The global innovation counter drift may affect future morph determinism and ID uniqueness even after rollback. | nge-core-scout / 04-implementing | Add a red test that asserts innovation counter resets after rollback. |

## Research artifact link

- `docs/research/racing-curriculum-tier1-demo-defects.md`
