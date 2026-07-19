# Pit/Strategy Sensory Channel Boundary Map

Research artifact for `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` — Phase 8 Step 02.

## Question

Where does the current pit lifecycle live, how is the Tier 4/5 observation vector assembled, and what is the exact insertion point for the 8 new pit/strategy sensory channels planned for Racing Curriculum v2? The answer must name every file that will change in Step 03-04 and confirm that no GPU execution paths are involved.

## Evidence

### Plan / documentation authority

- The active plan states the first Phase 8 slice extends the Tier 4+ observation vector from **95 to 103 channels** with 8 pit/strategy channels: distance to own pit entrance, own pit occupancy/blocking status, laps since last pit, teammate pit status, current tire degradation rate, estimated laps before tire failure, plus two reserved context channels (`plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`, lines 1123-1131 and 1213-1220).
- The generated READMEs document the current layout:
  - Tier 1 base = 70 channels `[0..69]` (`examples/racing_curriculum/README.md`, lines 437-441).
  - Tier 3 adds teammate-radio `[70..90]` = 21 channels.
  - Tier 4/5 append own-car tire health at `[91..94]` = 4 channels, producing the current 95-channel vector (`examples/racing_curriculum/README.md`, lines 802-822, 1076-1085; `examples/racing_curriculum/controller/README.md`, lines 480-498).
- The reference design describes the pit/strategy family semantics but does **not** assign numeric offsets (`examples/racing_curriculum/reference.plans.md`, lines 419-500).

### Static source authority

- **Pit lifecycle owner:** `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`.
  - `tickPitLifecycle()` lines 776-800 — decrements stop counters and restores tires when the stop completes.
  - `resolvePitEntries()` lines 818-854 — claims the team's compact pit slot when a car enters its own `entranceCorridor` AABB.
  - `buildRaceFrame()` lines 1014-1026 — initializes the compact `pitStatus` Uint8Array:
    - 4-car packs: `[teamA_car, teamA_ticks, teamB_car, teamB_ticks]`.
    - 6-car packs: `[teamA_car, teamA_ticks, teamA_waiting, teamB_car, teamB_ticks, teamB_waiting]`.
  - Constants at lines 131-137: `NO_CAR_INDEX = 255`, `PIT_STOP_TICKS = 4`, `FRESH_TIRE_HEALTH = 1.0`.
  - Tire decay is delegated to `decayTireState()` in `examples/racing_curriculum/environment/environment.step.service.ts` (verified via imports and call sites).
- **Observation assembler owner:** `examples/racing_curriculum/controller/observation.assembler.ts`.
  - Tier 4/5 vector is built by appending own tire health to the Tier 3 vector via `appendOwnTireState()` at lines 593-604.
  - `assembleTier4Observation()` lines 492-500 and `assembleTier5Observation()` lines 555-562 both call `appendOwnTireState()`.
  - The tire tail is written at offset `baseVector.length`, which is currently `91`, producing indices `[91..94]`.
- **Per-car observation context:** `resolvePerCarObservation()` in `simulation-worker.race-pack.service.ts` lines 709-735 builds a `RacingObservationState` and calls `derivePerCarObservationState()` (in `observation.assembler.ts`, lines 243-273). The 8 pit/strategy values must be attached to that state before `derivePerCarObservationState()` is invoked.
- **Type seam:** `ObservationExtensions` in `observation.assembler.ts` lines 67-86 is the natural place to add the new optional fields. `RacingObservationState` is defined as `EnvironmentState & ObservationExtensions` at line 89.
- **GPU files:** `Select-String` for `gpu|webgpu|GPU|WebGPU` across `simulation-worker.race-pack.service.ts` and `observation.assembler.ts` returned **zero matches**. The GPU entry point `simulation-worker.gpu.ts` is not imported by either file.

### Runtime / validation authority

- `node scripts/agent-customization/gates/plan-sync.gate.mjs --json` — PASS.
- `node scripts/agent-customization/gates/step-packet.gate.mjs --json` — PASS.
- `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json` — PASS.

## Decision

The 8 pit/strategy channels will be **appended after the existing tire tail**, keeping every pre-existing offset byte-stable:

| Indices | Current / new content | Notes |
| ------- | --------------------- | ----- |
| `[0..69]` | Tier 1 base | unchanged |
| `[70..90]` | 21 teammate-radio channels | unchanged |
| `[91..94]` | Own-car tire health `[FL, FR, RL, RR]` | unchanged |
| `[95..102]` | **8 new pit/strategy channels** | total vector grows from 95 → 103 |

Proposed channel assignment (to be pinned by Step 03 red tests):

| Offset | Channel |
| ------ | ------- |
| 95 | distance to own pit entrance |
| 96 | own pit occupancy / blocking status |
| 97 | laps since last pit |
| 98 | teammate pit status |
| 99 | current tire degradation rate |
| 100 | estimated laps before tire failure |
| 101 | reserved context channel 1 |
| 102 | reserved context channel 2 |

**Exact insertion coordinates in source code:**

- In `observation.assembler.ts`, add a new helper `appendPitStrategyState()` and call it immediately after `appendOwnTireState()` inside `assembleTier4Observation()` (line 496-499) and `assembleTier5Observation()` (line 559-562). The new helper writes into a 103-element `Float32Array` at offset `baseVector.length + TIRE_CHANNEL_COUNT` (i.e. 95), copying the existing base and tire data first.
- In `simulation-worker.race-pack.service.ts`, compute the 8 values in `resolvePerCarObservation()` before line 729 (`derivePerCarObservationState`) and pass them through `ObservationExtensions` on the `RacingObservationState` object built at lines 714-728.

## Files that will change in Step 03-04

### Step 03 (red tests)

- `examples/racing_curriculum/controller/observation.assembler.pit-strategy.test.ts` (new)

### Step 04, slice `04-obs-constants`

- `examples/racing_curriculum/controller/observation.assembler.ts`
- `examples/racing_curriculum/controller/observation.assembler.tier4.test.ts`
- `examples/racing_curriculum/controller/observation.assembler.tier5.test.ts`

### Step 04, slice `04-pit-fields`

- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`
- `examples/racing_curriculum/controller/observation.assembler.ts`
- `examples/racing_curriculum/environment/environment.types.ts`

### Step 04, slice `04-wire-vector`

- `examples/racing_curriculum/controller/observation.assembler.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`
- `examples/racing_curriculum/controller/observation.assembler.tier4.test.ts`
- `examples/racing_curriculum/controller/observation.assembler.tier5.test.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts`

### Additional dependency discovered outside the authored slices

- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts` — hard-codes `TIER_FOUR_CONTROLLER_INPUT_SIZE = 95` at line 163. It must be updated to `103` or the coevolution tests will fail when the Tier 4+ vector grows.
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.test.ts` — contains 95-input assertions around lines 334-364 that must be updated to match the new width.

No GPU files are involved.

## Risks

| Risk | Owner | Mitigation |
| ---- | ----- | ---------- |
| Exact semantics and normalization of the two reserved context channels (offsets 101-102) are still open. | Step 03 red-test writer | Pin reserved-channel expectations in the new red tests; treat them as zero-filled until Step 04 assigns semantics. |
| The coevolution service/test 95-input width is outside the authored Step 04 slices. | Step 04 implementer / Step 05 green testing | Update `TIER_FOUR_CONTROLLER_INPUT_SIZE` and related assertions in the same change set so existing coevolution tests do not regress. |
| Inserting the 8 channels anywhere other than the tire tail would shift indices `[91..94]` and break existing Tier 4/5 tests. | Step 04 implementer | Strictly append at the tail; never interleave. |
| Tire decay is computed in `environment.step.service.ts`, but the degradation-rate channel needs a per-car derivative that the race-pack service must compute from consecutive ticks. | Step 04 implementer | Store previous mean tire health per car in `runnerState` or derive from `tireState` and the decay formula; ensure the value is normalized consistently with existing tire channels. |

## Research artifact link

- `docs/research/pit-strategy-sensory-channels.md`
