# Research: Opponent / Rival-Team Perception for the Racing Curriculum

**Question:** How should the `examples/racing_curriculum` observation pipeline be extended to give each car usable opponent (rival-team) perception, including semantic relative-position channels, without breaking the existing tier architecture and coevolution plumbing?

**Scope:** Read-only research and design. No production files were edited. This note covers the observation pipeline, a concrete channel layout, the physics/environment data that must be exposed, downstream impact on workers/GPU/coevolution, the `buildTeammateSlot` speed bug, and a tier-strategy recommendation.

---

## 1. Current observation pipeline (stable as of DF17)

| Tier | Input width | What changed                              |
| ---- | ----------- | ----------------------------------------- |
| 1    | 70          | 20 scalar + 40 look-ahead + 10 memory     |
| 2    | 77          | +7 self-radio channels                    |
| 3    | 91          | +3 × 7 teammate radio slots               |
| 4/5  | **103**     | +4 own-car tire + 8 pit/strategy channels |

The canonical 103-channel layout is exported from `examples/racing_curriculum/controller/observation.assembler.ts` as `TOTAL_TIER4_INPUT_SIZE`. It is consumed by:

- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.ts`
- `examples/racing_curriculum/controller/runtime.adaptation.ts` (indirectly, via `network.input`)

The packed `RacingRenderFrame` carries per-car `carX`, `carY`, `carHeading`, `carTeam`, `tireState`, `radioField`, and pit state, but **no per-car velocity buffers**.

The worker-authoritative episode runner (`simulation-worker.race-pack.service.ts:717-750`) builds a per-car `RacingObservationState`, derives `perCarState`, and calls `assembleTier4Observation` (2v2) or `assembleTier5Observation` (3v3). Both assemblers append the same 4 + 8 tail and therefore share the 103-channel byte contract.

---

## 2. Proposed opponent channel layout

The reference design (`examples/racing_curriculum/reference.plans.md:458-469`) budgets **18 channels** for "Opponent and Race-Context Senses," but those channels are mostly _global_ race-context features (front-zone occupancy, nearest rival ahead/behind, time-to-contact, etc.). The request here is more specific: **per-opponent semantic framing** (in-front/behind, left/right, proximity category). That requires more than 18 channels if encoded per rival car.

### 2.1 Per-opponent slot (11 channels)

Each opponent slot is encoded in the focal car's local frame. The layout mirrors the existing teammate slot conventions (positions use `TRACK_POSITION_WORLD_SCALE = 96`, distances use `DISTANCE_WORLD_SCALE = 64`, speeds use `SPEED_WORLD_SCALE = 108`, headings are encoded as `sin(relativeHeading)`).

| Channel | Name                 | Source / formula                                                                                            | Range   |
| ------- | -------------------- | ----------------------------------------------------------------------------------------------------------- | ------- |
| 0       | `relForwardWorld`    | Longitudinal offset to opponent, positive = in front. `normalizeSignedValue(dx * cos(h) + dy * sin(h), 96)` | [-1, 1] |
| 1       | `relLeftWorld`       | Lateral offset, positive = left. `normalizeSignedValue(-dx * sin(h) + dy * cos(h), 96)`                     | [-1, 1] |
| 2       | `distanceWorld`      | Euclidean distance. `normalizeUnsignedValue(hypot(dx, dy), 64)`                                             | [0, 1]  |
| 3       | `opponentSpeedWorld` | Opponent speed. `normalizeSignedValue(opponent.speedWorld, 108)`                                            | [-1, 1] |
| 4       | `relHeadingSin`      | `sin(wrapAngleToMinusPiPi(opponent.heading - focal.heading))`                                               | [-1, 1] |
| 5       | `inFrontBehind`      | `+1` if `relForwardWorld > 0`, else `-1`                                                                    | [-1, 1] |
| 6       | `leftRight`          | `+1` if `relLeftWorld > 0`, else `-1`                                                                       | [-1, 1] |
| 7       | `proximityTouching`  | `1` if raw distance < 5 world units, else `0`                                                               | {0, 1}  |
| 8       | `proximityClose`     | `1` if 5 ≤ distance < 20, else `0`                                                                          | {0, 1}  |
| 9       | `proximityNear`      | `1` if 20 ≤ distance < 50, else `0`                                                                         | {0, 1}  |
| 10      | `proximityFar`       | `1` if distance ≥ 50, else `0`                                                                              | {0, 1}  |

_Note:_ the teammate slot today emits only 7 channels (`examples/racing_curriculum/controller/observation.assembler.ts:366-385`). Because opponent perception needs raw geometry _plus_ semantic framing, the opponent slot is wider.

### 2.2 Fixed-width allocation by roster size

To keep the vector shape stable across 1v1/2v2/3v3, always allocate **3 opponent slots** and zero-pad unused ones, exactly as the teammate-radio block does today.

| Roster | Live opponent slots | Opponent channels | Total Tier 6 width |
| ------ | ------------------- | ----------------- | ------------------ |
| 1v1    | 1                   | 3 × 11 = **33**   | 103 + 33 = **136** |
| 2v2    | 2                   | 3 × 11 = **33**   | **136**            |
| 3v3    | 3                   | 3 × 11 = **33**   | **136**            |

### 2.3 New byte-stable Tier 6 layout

```text
[0..69]    Tier 1 base (70)
[70..90]   teammate radio (3 × 7 = 21)
[91..94]   own-car tire health (4)
[95..102]  pit/strategy (8)
[103..135] opponent radio (3 × 11 = 33)
```

This layout keeps the existing Tier 4/5 vector unchanged; opponent data is appended as a new tail.

### 2.4 Recommended constants

Add to `examples/racing_curriculum/controller/observation.assembler.ts`:

```ts
const TIER_SIX_OPPONENT_SLOT_COUNT = 3;
const TIER_SIX_CHANNELS_PER_OPPONENT_SLOT = 11;
const TIER_SIX_OPPONENT_RADIO_CHANNEL_COUNT =
  TIER_SIX_OPPONENT_SLOT_COUNT * TIER_SIX_CHANNELS_PER_OPPONENT_SLOT; // 33
export const TOTAL_TIER6_INPUT_SIZE =
  TOTAL_TIER4_INPUT_SIZE + TIER_SIX_OPPONENT_RADIO_CHANNEL_COUNT; // 136

const OPPONENT_PROXIMITY_TOUCHING_THRESHOLD_WORLD = 5;
const OPPONENT_PROXIMITY_CLOSE_THRESHOLD_WORLD = 20;
const OPPONENT_PROXIMITY_NEAR_THRESHOLD_WORLD = 50;
```

Helper names should follow the existing pattern: `buildOpponentRadioSlots`, `buildOpponentSlot`, and `assembleTier6Observation`.

---

## 3. Physics / environment data needed

### 3.1 Per-car velocity fields (speed bug root cause)

`CarState` / `RacingCarState` (`examples/racing_curriculum/environment/environment.types.ts:88-110`) currently stores only pose, team, tire state, and an optional reward. There is no velocity field, so:

1. `buildTeammateSlot` hard-codes the speed channel to `0` at `observation.assembler.ts:380`.
2. `stepCarKinematics` (`environment.step.service.ts:788-819`) computes a local `speed` variable but does not return it.
3. The worker episode runner builds `RacingCarState[]` from `carX/carY/carHeading/tireState` only, so even after fixing the assembler there is no source value to read.

**Fix for the speed bug:**

1. Add `forwardSpeedWorld`, `lateralSpeedWorld`, and `speedWorld` (or at least `speedWorld`) to `CarState`/`RacingCarState`.
2. Populate them in `stepCarKinematics`.
3. Initialize them to `0` in `resolveCars` and the browser's `createCurriculumEnvironmentState`.
4. In the worker runner, either:
   - derive speed from the local `distanceAlongTrack` delta in `buildCarsFromFrame`, **or**
   - add a `carSpeed` typed array to `RacingRenderFrame` and include it in `resolveRaceStepTransferList`.
5. Change `buildTeammateSlot` line 380 from `0` to `teammate.speedWorld / SPEED_WORLD_SCALE` (or `normalizeSignedValue`).

### 3.2 Opponent geometry source

The environment already computes pairwise distances in `separateCars` (`environment.step.service.ts`). The same relative vectors can be reused for opponent slots. The assembler only needs:

- the focal car's pose (already in `perCarState`),
- the full `cars` roster (already passed to `buildTeammateRadioSlots`),
- each car's `speedWorld` (added above).

No new physics model is required; the existing AABB/separation math is sufficient.

---

## 4. Downstream impact assessment

### 4.1 Coevolution service

`simulation-worker.coevolution.service.ts:159-168` derives controller dimensions:

```ts
const isTier4 = config.tier >= 4;
const inputSize = isTier4 ? TIER_FOUR_CONTROLLER_INPUT_SIZE : ...
const outputSize = isTier3 ? TIER_THREE_CONTROLLER_OUTPUT_SIZE : ...
```

A new Tier 6 must introduce `TIER_SIX_CONTROLLER_INPUT_SIZE = TOTAL_TIER6_INPUT_SIZE` (136) and branch on `config.tier >= 6`. The output dimension stays 9 (2 control + 7 radio write); opponent channels are read-only.

### 4.2 Race-pack service and workers

- `resolvePerCarObservation` (`simulation-worker.race-pack.service.ts:717-750`) must use `assembleTier6Observation` for the new tier.
- `buildCarsFromFrame` must supply the new velocity fields.
- `simulation-worker.tier5.ts` already allocates a 6-car frame; if speed buffers are added to the frame, they must be initialized there as well.
- The worker message protocol already carries a `tier` field (`simulation-worker.race-pack.tier6.test.ts` assumes `tier: 6`), so the wire shape itself does not change.

### 4.3 Browser entry

- `SupportedObservationTier` (`browser-entry.ts:322`) is `1 | 2 | 3 | 4 | 5`; extend to `6`.
- `MAX_SUPPORTED_OBSERVATION_TIER` (`browser-entry.ts:399`) must become `6`.
- `resolveControllerInputCountForObservationTier` (`browser-entry.ts:3452-3468`) must return 136 for tier 6.
- `createDeterministicRacingControllerNetwork` must build a Tier-6-width MLP when the curriculum tier is 6.
- `createCurriculumEnvironmentState` must initialize new velocity fields.

### 4.4 GPU path

`simulation-worker.gpu.ts:48-87` only checks batch size and structural eligibility (gating, self-connections, activation support). It is **transparent to input dimension**; the batched CPU/GPU paths will receive the correct-length vector automatically as long as the network was created with the right `inputSize`. No GPU-kernel change is needed.

### 4.5 Runtime adaptation

`controller/runtime.adaptation.ts` builds training vectors from `network.input`. Because it does not hard-code observation widths, it will adapt automatically once the network is created with 136 inputs.

### 4.6 NGE carry-forward

Changing the input dimension of an _existing_ tier would invalidate saved genomes and serialized network envelopes. Treating this as a **new Tier 6** lets NGE's growth mechanism add the 33 new input modules during Tier 5→6 promotion while preserving Tier 5 weights and recurrent state.

---

## 5. Speed=0 bug: exact fix

**Location:** `examples/racing_curriculum/controller/observation.assembler.ts:380`

```ts
// today:
0,
// should be (after adding speedWorld to RacingCarState):
normalizeSignedValue(teammate.speedWorld, SPEED_WORLD_SCALE),
```

**Data source:** `teammate.speedWorld`, populated by `stepCarKinematics` and threaded through `buildCarsFromFrame` / `derivePerCarObservationState`.

This fix is independent of the opponent-perception work and should be done first. It does **not** change the observation vector width, so it is safe for existing Tier 3/4/5 networks.

---

## 6. Tier strategy recommendation

**Recommendation: implement opponent/rival-team perception as a new Tier 6 observation tier, not as an extension of Tier 4/5.**

Reasoning:

1. **Byte-contract stability.** Tier 4/5 share the exported `TOTAL_TIER4_INPUT_SIZE = 103`. A large test fleet and all saved genomes rely on that exact width. Extending Tier 4/5 would force destructive migration.
2. **Reference-plan alignment.** `reference.plans.md:366-375` already defines Tier 6 as "3v3 Advanced Strategy" with hall-of-fame opponent snapshots. The active plan's Phase 7 Tier 6 work (`NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md:1000-1002`) completed the strategy-divergence plumbing, leaving the 18-channel (here, 33-channel) opponent observation surface as the natural remaining deliverable.
3. **Genome compatibility.** NGE promotion carries weights and recurrent state upward. A fresh Tier 6 input dimension allows new modules to be grown rather than resizing existing envelopes.
4. **Curriculum wiring is ready.** `browser-entry.ts` already knows curriculum tier 6 but clamps the observation tier to 5. Lifting that clamp and adding a Tier 6 input-size branch is a small, localized change.

**Trade-off:** the per-opponent semantic layout costs 33 channels (total 136), which is slightly above the reference design's "effective policy input width: ~110–130" guidance. If staying inside that 18-channel budget is mandatory, the design must switch to aggregated race-context channels (front-zone occupancy + nearest rival ahead/behind) and drop per-opponent semantic flags. Given the explicit request for per-opponent semantic channels, 136 is the honest cost.

---

## 7. Files that would change (implementation inventory)

| File                                                                                            | Change                                                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `examples/racing_curriculum/environment/environment.types.ts`                                   | Add `forwardSpeedWorld`/`lateralSpeedWorld`/`speedWorld` to `CarState` / `RacingCarState`.                                                                                                           |
| `examples/racing_curriculum/environment/environment.step.service.ts`                            | Populate new velocity fields in `stepCarKinematics`; initialize to `0` in `resolveCars`.                                                                                                             |
| `examples/racing_curriculum/controller/observation.assembler.ts`                                | Fix `buildTeammateSlot` speed channel; add `buildOpponentRadioSlots`/`buildOpponentSlot`; add Tier 6 constants and `assembleTier6Observation`; widen `ObservationTier`.                              |
| `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`   | Derive / store per-car speed in `buildCarsFromFrame`; branch to `assembleTier6Observation`; update frame transfer list if speed buffers added.                                                       |
| `examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.ts`               | Initialize new frame speed fields if frame schema changes; add `resolveReadableOpponentRows` helper if needed.                                                                                       |
| `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts` | Add `TIER_SIX_CONTROLLER_INPUT_SIZE = TOTAL_TIER6_INPUT_SIZE`; branch on `config.tier >= 6`.                                                                                                         |
| `examples/racing_curriculum/browser-entry/browser-entry.ts`                                     | Extend `SupportedObservationTier`/`MAX_SUPPORTED_OBSERVATION_TIER`; return 136 in `resolveControllerInputCountForObservationTier`; initialize velocity fields in `createCurriculumEnvironmentState`. |
| `examples/racing_curriculum/controller/nge.controller.ts`                                       | No functional change; only widen accepted observation tier type.                                                                                                                                     |
| `examples/racing_curriculum/controller/runtime.adaptation.ts`                                   | No hard-coded width change needed.                                                                                                                                                                   |
| New tests                                                                                       | `observation.assembler.tier6.test.ts`, coevolution Tier 6 input-size tests, update `race-pack.tier6.test.ts` expectations.                                                                           |
| Generated docs                                                                                  | Run `npm run docs` after source JSDoc updates.                                                                                                                                                       |

---

## 8. Decision summary

- **Fix the teammate speed bug first.** Add velocity fields to `RacingCarState`, populate them in `stepCarKinematics`, and replace the literal `0` in `buildTeammateSlot` with `normalizeSignedValue(teammate.speedWorld, SPEED_WORLD_SCALE)`.
- **Add opponent perception as a new Tier 6 observation.** Use a stable 136-channel layout: existing 103 channels + 3 opponent slots × 11 channels.
- **Encode both raw geometry and semantic framing per opponent slot:** relative forward/left position, distance, speed, heading, signed in-front/behind and left/right flags, and four one-hot proximity categories.
- **Update coevolution and browser wiring** to recognize Tier 6 and allocate networks with 136 inputs/9 outputs, while leaving the existing Tier 4/5 path untouched.

---

## 9. Risks and residual gaps

| Risk                                                             | Mitigation                                                                                                                                                                                       |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 136 inputs is 32% larger than the current 103-channel vector.    | Growth-stabilize thresholds may need retuning for the larger search space; run browser smoke and NGE growth benchmarks after implementation.                                                     |
| Per-car speed is not currently persisted in `RacingRenderFrame`. | Derive from `distanceAlongTrack` delta, or extend the frame schema and include the new buffer in the transfer list.                                                                              |
| Existing Tier 6 curriculum code may assume Tier 6 == 103 inputs. | Add coevolution/browser branches and tests before enabling Tier 6 in the UI.                                                                                                                     |
| Cortex semantic index was stale during this research.            | Rebuild the index (`node rag-index/build-index.mjs`) before further Cortex-first searches.                                                                                                       |
| `reference.plans.md` budgets 18 opponent/race-context channels.  | Document that the per-opponent semantic layout exceeds the original 18-channel estimate; offer the 18-channel global-race-context alternative if the team wants to stay closer to the reference. |

---

## 10. Suggested next agent / step

Route implementation to the racing-curriculum feature implementer or `04-implementing`. The first slice should be the standalone speed-bug fix; the second slice should add the Tier 6 opponent observation surface and update tests. After code lands, run an `educational-docs` pass to regenerate example READMEs/JSDoc.
