# Racing Curriculum Tier 4 cars stuck at start line

## Question

Why do Tier 4 cars remain motionless at the start line in the browser demo at
`http://localhost:8080/docs/examples/racing_curriculum/index.html`, and which
files need red tests before a fix is implemented?

Originating plan step:
`plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` — Phase 8 — Racing
Curriculum v2, Step 08 — Fix Tier 4 cars stuck at start line in browser demo.

## Evidence

### Static code review

- `examples/racing_curriculum/controller/observation.assembler.ts:15-35` defines
  the Tier 4/5 observation vector as 103 channels:
  - `TIER_ONE_CHANNEL_COUNT = 70`
  - `TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT = 21`
  - `TIRE_CHANNEL_COUNT = 4`
  - `PIT_STRATEGY_CHANNEL_COUNT = 8`
  - `TOTAL_TIER4_INPUT_SIZE = 70 + 21 + 4 + 8 = 103`
- `observation.assembler.ts:181-213` dispatches observation tier 4 to
  `assembleTier4Observation` and tier 5 to `assembleTier5Observation`; both
  produce 103-channel `Float32Array`s.
- `examples/racing_curriculum/browser-entry/browser-entry.ts:2348-2364`
  `resolveControllerInputCountForObservationTier` returns **95** for every tier
  that is not 1, 2, or 3 (i.e., tiers 4 and 5).
- `browser-entry.ts:1489-1499` `createDeterministicRacingControllerNetwork`
  builds `Network.createMLP(resolvedInputCount, [4], resolvedOutputCount)`, so
  Tier 4/5 networks are created with **95** input nodes.
- `controller/nge.controller.ts:141-149` `createNgeController` assembles the
  observation vector and calls `network.activate(observationVector)` without
  truncating or validating length.
- `src/architecture/network/activate/network.activate.core.utils.ts:132-142`
  `validateInputVector` throws `NetworkActivateInputSizeMismatchError` when the
  input length does not equal `network.input`.

### Browser runtime confirmation (browser-runtime-scout)

- A visible Chrome browser was launched against
  `http://localhost:8080/docs/examples/racing_curriculum/index.html`.
- The default Tier 1 run was stopped and Tier 4 was started via
  `window.racingCurriculumStart('racing-curriculum-output', { tier: 4 })`.
- The first control tick threw an unhandled rejection:
  ```
  Input size mismatch: expected 95, got 103
  NetworkActivateInputSizeMismatchError: Input size mismatch: expected 95, got 103
  ```
- Source-mapped stack:
  - `src/architecture/network/activate/network.activate.core.utils.ts:72`
    (`validateInputVector`)
  - `src/architecture/network/network.ts:1108` (`Network.activate`)
  - `examples/racing_curriculum/controller/nge.controller.ts:149` (`computeControl`)
  - `examples/racing_curriculum/browser-entry/browser-entry.ts:503`
    (`resolvePerCarControls`)
- Screenshot `tmp/tier4-screenshot.png` shows the four-car Tier 4 grid still
  stationary after ~3.5 s.

### Boundary mapping (boundary-mapper)

- The browser demo control path is a hub-and-spoke around
  `browser-entry/browser-entry.ts`.
- `resolvePerCarControls` calls one `NgeController.computeControl` per car,
  passing a per-car observation state from `derivePerCarObservationState`.
- The worker step bridge (`requestRacingWorkerStep`) posts full
  `EnvironmentState` objects via structured clone; the worker side just calls
  `stepEnvironment`.
- The worker race-pack service (`simulation-worker.race-pack.service.ts`) is not
  used by the browser demo path; it builds its own 103-channel observations and
  runs a centerline-follower physics model.

## Decision

The root cause is a contract mismatch in `browser-entry.ts`:
`resolveControllerInputCountForObservationTier` sizes the Tier 4/5 controller
network to **95** inputs, while `observation.assembler.ts` emits a **103**
-channel observation vector. The resulting
`NetworkActivateInputSizeMismatchError` is unhandled on the first animation-loop
control tick, so no per-car throttle/steer values are ever produced and the
cars stay at the start line.

Recommended fix location:
`examples/racing_curriculum/browser-entry/browser-entry.ts` — update
`resolveControllerInputCountForObservationTier` to return 103 for tiers 4 and 5
(matching `TOTAL_TIER4_INPUT_SIZE`), then rebuild
`docs/assets/racing-curriculum.bundle.js` with `npm run build:racing-curriculum`.

## Risks

- Tier 5 is affected by the same 95-vs-103 mismatch; any fix must cover both
  tiers.
- `Network.activate` throws synchronously inside an async `animationStep` with no
  try/catch, so one bad tick silently stops the entire loop. A future hardening
  slice should add loop-level error telemetry.
- The browser demo path and the worker race-pack path use different physics
  models (`environment.step.service.ts` vs. the race-pack centerline-follower),
  so green validation must include both Jest suites and real browser visual
  confirmation.
- No existing unit test wires a Tier 4/5 deterministic controller network to a
  Tier 4/5 observation vector, which is why the bug reached the browser demo.

## Recommended red tests

Add failing tests in these files before implementation:

1. `examples/racing_curriculum/browser-entry/browser-entry.test.ts`
   - `createDeterministicRacingControllerNetwork(4).input` equals 103.
   - `createDeterministicRacingControllerNetwork(5).input` equals 103.
   - `resolvePerCarControls` produces one non-zeroed control per car when the
     controller map is fully populated.

2. `examples/racing_curriculum/controller/nge.controller.test.ts`
   - A Tier 4/5 `createNgeController` can call `computeControl` with a Tier 4/5
     observation state without throwing a size-mismatch error.

3. `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.test.ts`
   / `.tier5.test.ts`
   - `resolvePerCarObservation` output length equals every runner network's
     input length.
