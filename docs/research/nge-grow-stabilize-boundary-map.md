# NGE Grow-Stabilize Cycle Boundary Map

**Date:** 2026-07-12
**Phase:** 9, Step 02
**Source file:** `examples/racing_curriculum/controller/runtime.adaptation.ts` (1,390 lines, 49KB)
**Target directory:** `src/neat/nge-juvenile/`

## Question

Confirm the boundary map for extracting NGE grow-stabilize cycle logic from the racing demo's app layer (`runtime.adaptation.ts`) into the NGE core library (`src/neat/nge-juvenile/`). Verify which functions, constants, and logic blocks move; which stay; what new files and types are needed; how the app layer calls the extracted core; and investigate the "all cars" and "driving improvement" defects.

## Evidence

### 1. Functions/Constants/Objects That Move to `src/neat/nge-juvenile/`

All of the following are NGE-core algorithmic logic currently embedded in the app layer. They have no dependency on racing domain concepts (no `RacingQualitySignal`, no `toDrivingQuality`, no track/physics types).

#### Exported Functions

| Function | Line | Current Export | Target |
|---|---|---|---|
| `resolveAdaptiveHysteresis(nodeCount: number): number` | 225 | `export` | `neat.nge-juvenile.grow-stabilize.ts` |

#### Non-Exported Functions (Internal)

| Function | Line | Current Export | Target |
|---|---|---|---|
| `isPlateauReached(scoreWindow, hasGrownBefore, stabilizationTicksSinceGrowth): boolean` | 1322 | internal | `neat.nge-juvenile.grow-stabilize.ts` |
| `applyWeightMutations(network: Network, random: () => number): number` | 1379 | internal | `neat.nge-juvenile.grow-stabilize.ts` |
| `computeGrowthThrottle(network: Network, tick: number): {shouldThrottle, interval}` | 1260 | internal | `neat.nge-juvenile.grow-stabilize.ts` |
| `buildCandidateScoreWindow(network, evidenceWindow): number[]` | 850 | internal | `neat.nge-juvenile.grow-stabilize.ts` |
| `collectForwardPassOutputs(network, scoreHistory): number[][]` | 808 | internal | `neat.nge-juvenile.grow-stabilize.ts` |
| `resolveSampleIndices(historyLength, maxSamples): number[]` | 874 | internal | `neat.nge-juvenile.grow-stabilize.ts` |
| `resolveBehavioralComplexity(outputs: number[][]): number` | 929 | internal | `neat.nge-juvenile.grow-stabilize.ts` |

**Note on `buildCandidateScoreWindow`/`collectForwardPassOutputs`/`resolveObservationVector`:** These functions currently accept `number | RacingQualitySignal` as the score history entry type. The core extraction must abstract this: the generic quality signal type `NgeQualitySignal` should carry numeric fields, or the core module should accept a generic score-to-observation mapper function from the caller. The racing-specific `resolveObservationVector` (which tiles `RacingQualitySignal` fields) stays in the app layer as a mapper callback.

#### Constants (All Non-Exported)

| Constant | Line | Value | Target |
|---|---|---|---|
| `PLATEAU_WINDOW_SIZE` | 240 | 5 | `neat.nge-juvenile.constants.ts` |
| `PLATEAU_VARIANCE_THRESHOLD` | 249 | 0.1 | `neat.nge-juvenile.constants.ts` |
| `WEIGHT_MUTATION_RATE` | 256 | 0.3 | `neat.nge-juvenile.constants.ts` |
| `WEIGHT_MUTATION_MAGNITUDE` | 263 | 0.1 | `neat.nge-juvenile.constants.ts` |
| `MIN_STABILIZATION_TICKS` | 270 | 5 | `neat.nge-juvenile.constants.ts` |
| `MAX_STABILIZATION_TICKS` | 278 | 25 | `neat.nge-juvenile.constants.ts` |
| `MAX_EPISODIC_SLOTS` | 200 | 15 | `neat.nge-juvenile.constants.ts` |
| `LARGE_NETWORK_NODE_THRESHOLD` | 207 | 1,000 | `neat.nge-juvenile.constants.ts` |
| `GROWTH_THROTTLE_BASE_INTERVAL_TICKS` | 213 | 3 | `neat.nge-juvenile.constants.ts` |
| `MAX_FORWARD_PASS_SAMPLES` | 796 | 5 | `neat.nge-juvenile.constants.ts` |

#### Embedded Logic Blocks (in `createRuntimeAdaptationEngine.adaptOnTick`)

| Logic Block | Lines | Description |
|---|---|---|
| Two-phase adaptation (growth → stabilization) | 458-534 | Phase transition: if plateau not reached → stabilization; else → growth |
| Weight mutation commit/rollback | 464-533 | Apply weight mutations, evaluate improvement, commit or rollback |
| First-growth bypass | 577, 646-648 | `isFirstGrowth = !hasGrownBefore` → unconditional commit if safety checks pass |
| `preMutationBaselineScore` logic | 557-564 | Forward-pass baseline score before lifecycle mutation |
| `qualityScoreHistory` management | 310, 375-378 | Rolling window push/shift for plateau detection |
| Growth throttle gate | 421-442 | Size-based back-off for large networks |

#### State Variables That Move to `NgeGrowStabilizeState`

| Variable | Line | Type |
|---|---|---|
| `qualityScoreHistory` | 310 | `number[]` |
| `hasGrownBefore` | 311 | `boolean` |
| `stabilizationTicksSinceGrowth` | 312 | `number` |
| `currentPhase` | 313 | `'growth' \| 'stabilization'` |

### 2. What Stays in `examples/racing_curriculum/`

#### Stays in `runtime.adaptation.ts` (App Layer)

| Item | Line | Reason |
|---|---|---|
| `toDrivingQuality(entry: number \| RacingQualitySignal): number` | 1026 | Domain-specific: maps composite racing signals to scalar |
| `evaluateRacingTrendScore(network, scoreHistory): number` | 982 (exported) | Domain-specific: racing trend evaluator with complexity bonus |
| `evaluateRollingScoreWindow(network, scoreHistory): number` | 760 (exported) | Domain-specific: default rolling window evaluator |
| `RACING_COMPLEXITY_WEIGHT` | 788 | Domain-specific: complexity weight constant |
| `resolveObservationVector(entry, inputSize): number[]` | 897 | Domain-specific: tiles RacingQualitySignal fields into observation vector |
| `RuntimeAdaptationCadenceMode` type | 16 | App layer: cadence mode enum |
| `RuntimeAdaptationCadenceOptions` interface | 90 | App layer: cadence policy |
| `RuntimeAdaptationLimits` interface | 100 | App layer: hard bounds (maxNodes, maxConnections, cooldowns) |
| `RuntimeAdaptationEngineOptions` interface | 114 | App layer: engine options |
| `RuntimeAdaptationTickInput` interface | 139 | App layer: per-tick input |
| `RuntimeAdaptationEngine` interface | 153 | App layer: engine surface |
| `RuntimeAdaptationTelemetry` interface | 32 | App layer: telemetry output |
| `RuntimeAdaptationOperation` type | 20 | App layer: operation type |
| `RuntimeNetworkSizeSnapshot` interface | 24 | App layer: size snapshot |
| `RacingQualitySignal` interface | 76 | Domain-specific: composite driving-quality signal |
| `createRuntimeAdaptationEngine(options)` | 286 (exported) | App layer: engine factory (calls core cycle) |
| `createPerCarAdaptationEngines(carCount, options)` | 739 (exported) | App layer: per-car factory |
| `resolveCadenceOptions`, `resolveLimitOptions`, `resolveEvidenceWindow` | 1040-1109 | App layer: option resolvers |
| `isCadenceReady` | 1111 | App layer: cadence gating |
| `buildModuleMetricsSnapshot` | 1155 | App layer: maps racing scores to NGE metrics |
| `buildGrowthBudget`, `buildPruneBudget` | 1187, 1207 | App layer: builds budgets from limits + network |
| `mapOutcomesToOperations` | 1224 | App layer: maps lifecycle outcomes to telemetry ops |
| `passesSafetyChecks` | 1238 | App layer: safety check |
| `resolveNetworkSizeSnapshot` | 1277 | App layer: network size snapshot |
| `createTelemetry` | 1286 | App layer: telemetry factory |
| `DEFAULT_CADENCE`, `DEFAULT_LIMITS`, `DEFAULT_IMPROVEMENT_THRESHOLD`, `DEFAULT_MINIMUM_EVIDENCE_WINDOW`, `RUNTIME_MODULE_ID` | 178-193 | App layer: defaults |

#### Stays in `browser-entry.ts` (Browser Layer)

| Item | Line | Reason |
|---|---|---|
| `TIER_N_FLOOR` record | 372-379 | Domain-specific: tier promotion node floors |
| `TelemetryPanelNodes` interface | 293 | Domain-specific: UI panel |
| Lap time display logic | 695-703 | Domain-specific: lap-time tracking |
| `resolvePerCarTrackProgress` | 3094 | Domain-specific: per-car quality signal |
| `resolvePerCarForwardSpeed` | 3123 | Domain-specific: per-car quality signal |
| `resolvePerCarHeadingAlignment` | 3152 | Domain-specific: per-car quality signal |
| `resolvePerCarOffTrackPenalty` | 3188 | Domain-specific: per-car quality signal |
| `resolveTierPromotion` | 2938 | Domain-specific: tier promotion logic |

#### Stays in `environment.step.service.ts` (Physics Layer)

| Item | Line | Reason |
|---|---|---|
| `OFF_TRACK_CLAMP_REWARD = -5` | 43 | Domain-specific: border penalty |
| `WRONG_DIRECTION_REWARD = -5` | 45 | Domain-specific: wrong direction penalty |
| `GUIDE_FOLLOW_REWARD_CLOSE = 3` | 47 | Domain-specific: guide reward |
| `GUIDE_FOLLOW_REWARD_MODERATE = 1` | 49 | Domain-specific: guide reward |
| `GUIDE_DIVERGENCE_PENALTY = -2` | 51 | Domain-specific: guide penalty |
| `MAX_ESCALATING_BORDER_TICKS = 10` | 59 | Domain-specific: escalating penalty cap |
| `computeGuideFollowReward` | 1044 | Domain-specific: guide reward function |
| `computeGuideDivergencePenalty` | 1068 | Domain-specific: guide penalty function |
| Escalating border-contact penalty logic | 244-254 | Domain-specific: per-tick escalating penalty |
| `RewardShapingStateExtensions` type | 90 | Domain-specific: reward shaping state |

### 3. New Files in `src/neat/nge-juvenile/`

#### `neat.nge-juvenile.grow-stabilize.ts` (New Core Module)

Contains the extracted NGE grow-stabilize cycle:

```typescript
// Main entry point — the cycle orchestrator
export function runNgeGrowStabilizeCycle(input: NgeGrowStabilizeInput): NgeGrowStabilizeResult;

// Extracted helper functions
export function resolveAdaptiveHysteresis(nodeCount: number): number;
export function isPlateauReached(scoreWindow, hasGrownBefore, stabilizationTicksSinceGrowth): boolean;
export function applyWeightMutations(network: Network, random: () => number): number;
export function computeGrowthThrottle(network: Network, tick: number): {shouldThrottle, interval};
export function buildCandidateScoreWindow(network, evidenceWindow, observationMapper?): number[];
export function collectForwardPassOutputs(network, scoreHistory, observationMapper?): number[][];
```

The `observationMapper` callback pattern lets the core stay domain-agnostic: the app layer passes a function that converts its domain-specific quality signal into a numeric observation vector.

#### Types to Add to `neat.nge-juvenile.types.ts`

```typescript
export interface NgeGrowStabilizeConfig {
  readonly plateauWindowSize: number;
  readonly plateauVarianceThreshold: number;
  readonly minStabilizationTicks: number;
  readonly maxStabilizationTicks: number;
  readonly weightMutationRate: number;
  readonly weightMutationMagnitude: number;
  readonly maxEpisodicSlots: number;
  readonly largeNetworkNodeThreshold: number;
  readonly growthThrottleBaseIntervalTicks: number;
  readonly improvementThreshold: number;
  readonly maxStructuralEditsPerStep: number; // dead knob to be wired
  readonly maxForwardPassSamples: number;
}

export interface NgeGrowStabilizeState {
  qualityScoreHistory: number[];
  hasGrownBefore: boolean;
  stabilizationTicksSinceGrowth: number;
  currentPhase: NgeGrowStabilizePhase;
}

export type NgeGrowStabilizePhase = 'growth' | 'stabilization';

export interface NgeGrowStabilizeInput {
  readonly tick: number;
  readonly network: Network;
  readonly evidenceWindow: readonly (number | NgeQualitySignal)[];
  readonly config: Partial<NgeGrowStabilizeConfig>;
  readonly state: NgeGrowStabilizeState;
  readonly evaluateScore: (network: Network, scoreWindow: number[]) => number;
  readonly observationMapper?: (entry: number | NgeQualitySignal, inputSize: number) => number[];
  readonly lifecycleRunner?: (input: NgeJuvenileLifecycleInput) => NgeLifecycleResult;
  readonly hysteresis: NgeHysteresisState;
  readonly limits: { maxNodes: number; maxConnections: number; maxStructuralEditsPerStep: number };
}

export interface NgeGrowStabilizeResult {
  readonly committed: boolean;
  readonly reason: string;
  readonly scoreBefore: number;
  readonly scoreAfter: number;
  readonly operations: readonly string[];
  readonly phase: NgeGrowStabilizePhase;
  readonly stabilizationTicksSinceGrowth: number;
  readonly networkSizeBefore: { nodes: number; connections: number };
  readonly networkSizeAfter: { nodes: number; connections: number };
  readonly updatedState: NgeGrowStabilizeState;
  readonly updatedHysteresis: NgeHysteresisState;
}

export interface NgeQualitySignal {
  readonly [key: string]: number;
}
```

#### Constants to Add to `neat.nge-juvenile.constants.ts`

```typescript
export const NGE_GROW_STABILIZE_DEFAULT_PLATEAU_WINDOW_SIZE = 5;
export const NGE_GROW_STABILIZE_DEFAULT_PLATEAU_VARIANCE_THRESHOLD = 0.1;
export const NGE_GROW_STABILIZE_DEFAULT_WEIGHT_MUTATION_RATE = 0.3;
export const NGE_GROW_STABILIZE_DEFAULT_WEIGHT_MUTATION_MAGNITUDE = 0.1;
export const NGE_GROW_STABILIZE_DEFAULT_MIN_STABILIZATION_TICKS = 5;
export const NGE_GROW_STABILIZE_DEFAULT_MAX_STABILIZATION_TICKS = 25;
export const NGE_GROW_STABILIZE_DEFAULT_MAX_EPISODIC_SLOTS = 15;
export const NGE_GROW_STABILIZE_DEFAULT_LARGE_NETWORK_NODE_THRESHOLD = 1_000;
export const NGE_GROW_STABILIZE_DEFAULT_GROWTH_THROTTLE_BASE_INTERVAL_TICKS = 3;
export const NGE_GROW_STABILIZE_DEFAULT_MAX_FORWARD_PASS_SAMPLES = 5;
export const NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP = 5; // raised from 1
```

#### Barrel Re-Export in `neat.nge-juvenile.ts`

Add:
```typescript
export * from './neat.nge-juvenile.grow-stabilize';
```

#### Cycle-Break in `neat.nge-lifecycle.ts`

**Current (circular risk):**
```
neat.nge-lifecycle.ts → imports from ./nge-juvenile/neat.nge-juvenile (barrel)
barrel → re-exports ./neat.nge-juvenile.grow-stabilize
grow-stabilize.ts → imports runNgeLifecycle from ../../neat.nge-lifecycle ← CIRCULAR
```

**Fix — refactor `neat.nge-lifecycle.ts` to import from specific source files:**

```typescript
// Instead of:
import { applyMorphDeltas, commitGrowth, computeFocusScores, planGrowthMorphs, resolveFocusConfig } from './nge-juvenile/neat.nge-juvenile';

// Change to:
import { applyMorphDeltas } from './nge-juvenile/neat.nge-juvenile.apply';
import { commitGrowth } from './nge-juvenile/neat.nge-juvenile.grow';
import { computeFocusScores } from './nge-juvenile/neat.nge-juvenile.focus';
import { planGrowthMorphs } from './nge-juvenile/neat.nge-juvenile.grow';
import { resolveFocusConfig } from './nge-juvenile/neat.nge-juvenile.focus';
```

This breaks the cycle: `neat.nge-lifecycle.ts` no longer depends on the barrel, so `grow-stabilize.ts` can safely import `runNgeLifecycle` without creating a circular dependency.

**Alternative (preferred): dependency injection.** The `runNgeGrowStabilizeCycle` function accepts a `lifecycleRunner` callback parameter instead of importing `runNgeLifecycle` directly. This eliminates the circular dependency entirely and makes the core module fully testable in isolation. The app layer passes `runNgeLifecycle` as the callback.

### 4. App Layer Invocation Interface

After extraction, `runtime.adaptation.ts` will:

1. **Import** `runNgeGrowStabilizeCycle`, types, and constants from `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts`
2. **Create** a `NgeGrowStabilizeState` in the `createRuntimeAdaptationEngine` closure
3. **On each `adaptOnTick` call:**
   - Check cadence (stays in app layer)
   - Check evidence window sufficiency (stays in app layer)
   - Advance hysteresis (stays in app layer)
   - Check cooldowns (stays in app layer)
   - Check growth throttle (can delegate to core or stay)
   - Call `runNgeGrowStabilizeCycle` with:
     - `tick`, `network`, `evidenceWindow`
     - `config`: resolved from `RuntimeAdaptationEngineOptions` (thresholds, plateau params, weight mutation params, maxStructuralEditsPerStep)
     - `state`: the mutable `NgeGrowStabilizeState`
     - `evaluateScore`: the racing-specific evaluator (`evaluateRacingTrendScore` or custom)
     - `observationMapper`: `resolveObservationVector` (racing-specific)
     - `lifecycleRunner`: `runNgeLifecycle` (injected)
     - `hysteresis`: the current hysteresis state
     - `limits`: from `RuntimeAdaptationLimits`
   - Map the `NgeGrowStabilizeResult` to `RuntimeAdaptationTelemetry`
4. **`createPerCarAdaptationEngines`** stays in app layer — it creates independent engines per car, each with its own state

### 5. "All Cars" Issue Analysis

**Finding: Per-car wiring is ALREADY correct.** The code at `browser-entry.ts:614` iterates all car indices:
```typescript
for (let carIndex = 0; carIndex < carCountThisStep; carIndex++) {
  // ... builds per-car quality signal, pushes to scoreHistory, calls adaptOnTick
}
```

Each car has:
- Its own `controllerNetworkByCarIndex.get(carIndex)` (line 652)
- Its own `scoreHistoryByCarIndex.get(carIndex)` (line 615)
- Its own `adaptationEngineByCarIndex.get(carIndex)` (line 648)
- Independent closure state (confirmed by `createPerCarAdaptationEngines` at line 739-750, which creates separate `createRuntimeAdaptationEngine` calls per car)

**Root cause of the "all cars" perception issue:**
1. **Visualizer only shows car 0**: `focusedControllerNetwork = controllerNetworkByCarIndex.get(0)!` (line 525, 780). The UI only renders car 0's network, making it appear that only one car is evolving.
2. **Tier 1 starts with 2 cars (1v1)**: `TIER_1_PACK_CAR_COUNT = 2` (line 180). Both cars get adaptation ticks, but only car 0 is visualized.
3. **Cars may not grow if quality signal is weak**: The `evaluateRacingTrendScore` requires a positive quality trend (`scoreTrend >= 0`) for the complexity bonus. If a car drives poorly, its score trend is negative, and growth is not rewarded — it gets stuck in a negative feedback loop.

**Slice 04c fix plan:**
- Ensure the adaptation loop covers all cars even after tier transitions (verify `carCountThisStep` is correct)
- Consider visualizing multiple cars' growth in the telemetry panel
- Verify no edge case where `envState.cars?.length` is shorter than expected

### 6. Driving Improvement Issue Analysis

**Finding: Reward signal chain is too weak for learning.**

The reward chain:
1. `environment.step.service.ts` produces `carState.reward` from:
   - Off-track clamp: `-5 * consecutiveBorderContactTicks` (escalating, capped at 10 ticks → max -50)
   - Wrong direction: `-5` (flat, not escalating)
   - Guide-following close: `+3`
   - Guide-following moderate: `+1`
   - Guide divergence: `-2`
2. `browser-entry.ts` reads `carState?.reward ?? 0` as `physicsReward` in `RacingQualitySignal`
3. `toDrivingQuality()` converts: `trackProgress * 0.35 + forwardSpeed * 0.25 + headingAlignment * 0.3 - offTrackPenalty * 0.3 + physicsReward * 0.3`

**Problems identified:**

1. **`physicsReward` weight is only 0.3**: The guide-following reward (+3) becomes +0.9 in the quality score. The border penalty (-5 first tick, -10 second, etc.) becomes -1.5, -3.0, etc. These are small relative to `trackProgress * 0.35` (which can reach +0.35). The physics reward signal is diluted.

2. **Guide rewards only apply at Tier 0-1**: `guidanceAlpha > 0` check at line 262. At Tier 2+, no guide-following reward exists, so the car has no positive shaping signal beyond track progress and heading alignment.

3. **Wrong-direction penalty is flat (-5)**: No escalation. A car driving wrong for 50 ticks gets -5 per tick, which at 0.3 weight = -1.5 per tick. This may not be enough to overcome the forwardSpeed reward (up to +0.25) when the car is moving fast in the wrong direction.

4. **The quality score does not directly encode "improvement"**: `toDrivingQuality` produces an absolute quality value, not a delta. The adaptation engine uses `evaluateRacingTrendScore` which computes `scoreMean + scoreTrend * 0.5 + complexityBonus`. The trend term (`last - first`) captures improvement, but the mean term can dominate if the car consistently drives at a moderate quality level without improving.

5. **The improvement threshold (0.01) is very low**: A candidate mutation that produces even a tiny score improvement (0.01) gets committed. This means the network may commit mutations that don't meaningfully improve driving — they just barely beat the baseline by noise.

**Slice 04d fix plan:**
- Increase guide-following reward magnitude or its weight in `toDrivingQuality`
- Add escalating wrong-direction penalty (similar to border contact escalation)
- Make border penalty stronger (increase weight in `toDrivingQuality` or increase base penalty)
- Consider adding a "wrong direction duration" tracker similar to `consecutiveBorderContactTicks`
- Ensure guide rewards are available at more tiers or provide alternative positive shaping at Tier 2+

## Decision

### Extraction Plan

1. **Slice 04a (Foundation):** Add `NgeGrowStabilizeConfig`, `NgeGrowStabilizeState`, `NgeGrowStabilizeInput`, `NgeGrowStabilizeResult`, `NgeGrowStabilizePhase`, `NgeQualitySignal` types to `neat.nge-juvenile.types.ts`. Add default constants to `neat.nge-juvenile.constants.ts`. Break the barrel-import cycle by refactoring `neat.nge-lifecycle.ts` to import from specific source files.

2. **Slice 04b (Core Extraction):** Create `neat.nge-juvenile.grow-stabilize.ts` with `runNgeGrowStabilizeCycle` and all extracted functions. Add barrel re-export. Refactor `runtime.adaptation.ts` to call `runNgeGrowStabilizeCycle` instead of containing the inline logic. Remove `resolveAdaptiveHysteresis`, `isPlateauReached`, `applyWeightMutations`, and all moved constants from `runtime.adaptation.ts` (No Deferred Cleanup). Keep cadence, evidence window, hysteresis advancement, cooldowns, and domain-specific evaluation in the app layer.

3. **Slice 04c (All-Cars):** Verify the per-car loop covers all cars. The wiring is already correct — focus on ensuring no edge case drops cars and consider visualizing multiple cars' growth.

4. **Slice 04d (Driving Quality):** Strengthen reward shaping in `environment.step.service.ts` and adjust `toDrivingQuality` weights in `runtime.adaptation.ts` to give physics reward more influence.

5. **Slice 04e (Growth Speed):** Wire `maxStructuralEditsPerStep` into the grow-stabilize cycle so multiple structural edits apply per growth phase tick. Raise default from 1 to 5-10.

6. **Slice 04f (Test Fixes):** Fix `resolveTierPromotionFromLapCount` missing export from `browser-entry.ts`. Fix TS type mismatch in `nge-e2e-growth.test.ts`. Add Tier 2 remap test.

### Cycle-Break Strategy

**Preferred: dependency injection.** `runNgeGrowStabilizeCycle` accepts a `lifecycleRunner` callback. The app layer passes `runNgeLifecycle`. This eliminates the circular dependency without touching `neat.nge-lifecycle.ts` imports.

**Fallback: direct import refactor.** If dependency injection is too invasive, refactor `neat.nge-lifecycle.ts` to import from specific sub-module files instead of the barrel. Both approaches break the cycle; the injection approach is cleaner and more testable.

## Risks

1. **Behavioral regression risk**: The extraction must preserve exact runtime behavior. The 144 existing tests serve as the safety net. Any behavioral change (even improvement) in slice 04b is a defect — behavioral changes belong in slices 04c-04f.

2. **`resolveObservationVector` coupling**: `buildCandidateScoreWindow` calls `collectForwardPassOutputs` which calls `resolveObservationVector`. The latter is racing-specific (tiles `RacingQualitySignal` fields). The core module must accept an `observationMapper` callback to stay domain-agnostic, or the functions must be split so the generic parts (forward-pass, sample selection, variance) move to core and the racing-specific observation mapping stays in the app layer.

3. **`evaluateScore` coupling**: The grow-stabilize cycle calls the caller-supplied `evaluateScore` function. This is already dependency-injected, so the core can accept it as a parameter without change.

4. **`runNgeLifecycle` coupling**: The growth phase calls `runNgeLifecycle` (line 583). This is the most significant coupling — the core module would need to either import `runNgeLifecycle` (circular dependency risk) or accept it as a callback. The callback approach is recommended.

5. **`buildModuleMetricsSnapshot` / `buildGrowthBudget` / `buildPruneBudget` coupling**: These functions bridge app-layer limits and racing quality scores to NGE lifecycle inputs. They stay in the app layer and are called before invoking the core cycle.

6. **Test defect risk**: `resolveTierPromotionFromLapCount` is imported by `browser-entry.progression.test.ts` but is NOT exported from `browser-entry.ts`. This is a pre-existing defect. The function exists in the test expectations (returns `{nextTier, didAdvance, remainingLaps}`) but the implementation is missing. Slice 04f must add this export.