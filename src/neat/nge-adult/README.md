# neat/nge-adult

Rolling reward-delta evidence tracked while the adult phase checks for plateau behavior.

## neat/nge-adult/neat.nge-adult.types.ts

### AdultState

Persistent adult-phase state shelf reserved for plateau, cooling, and equilibrium passes.

### EquilibriumCandidate

Typed equilibrium snapshot emitted once one adult zone looks ready for compaction-first policy.

### GainStabilityRecord

Rolling gain evidence tracked while the adult phase checks for stable optimization pressure.

### PlateauRecord

Rolling reward-delta evidence tracked while the adult phase checks for plateau behavior.

## neat/nge-adult/neat.nge-adult.ts

### advanceAdultState

```ts
advanceAdultState(
  input: AdvanceAdultStateInput,
): AdvanceAdultStateResult
```

Advance one owner-local adult optimization cycle.

Parameters:
- `input` - Runtime inputs for the current adult cycle.

Returns: The composed adult transition for the current cycle.

### AdvanceAdultStateInput

Runtime inputs for one adult-phase orchestration step.

### AdvanceAdultStateResult

Top-level adult orchestration result for one evaluation cycle.

## neat/nge-adult/neat.nge-adult.errors.ts

Error raised when adult plateau evaluation cannot produce a valid local decision.

### NgeAdult_EquilibriumError

Error raised when adult equilibrium evaluation cannot produce a valid local decision.

### NgeAdult_PlateauError

Error raised when adult plateau evaluation cannot produce a valid local decision.

## neat/nge-adult/neat.nge-adult.cooling.ts

### AdultGrowthCoolingDecision

Budget split emitted by the adult cooling policy for one focus score.

### AdultPruneCandidate

Ranked adult prune candidate evaluated before compact is considered.

### AdultPruneCompactDecision

Local dominance decision produced by the adult prune and compact arbitration step.

### arbitratePruneCompactDominance

```ts
arbitratePruneCompactDominance(
  candidates: readonly AdultPruneCandidate[],
  compactEligible: boolean,
): AdultPruneCompactDecision
```

Arbitrate whether adult prune and compact dominate the cooled morph budget.

Parameters:
- `candidates` - Ranked prune candidates visible to the local adult module.
- `compactEligible` - Whether compact still has headroom in the current window.

Returns: A placeholder decision that is intentionally incorrect during the red phase.

### resolveGrowthCoolingDecision

```ts
resolveGrowthCoolingDecision(
  focusScore: number,
  growthCoolingFactor: number,
  focusFloor: number,
): AdultGrowthCoolingDecision
```

Resolve the current adult cooling decision for one normalized focus score.

Parameters:
- `focusScore` - Normalized adult focus score for the active module.
- `growthCoolingFactor` - Residual growth fraction preserved by adult cooling.
- `focusFloor` - Minimum focus score that keeps residual growth alive.

Returns: A placeholder decision that is intentionally incorrect during the red phase.

## neat/nge-adult/neat.nge-adult.plateau.ts

### advancePlateauRecord

```ts
advancePlateauRecord(
  plateauRecord: PlateauRecord,
  rewardDelta: number,
  marginalEpsilon: number,
): PlateauRecord
```

Advances one rolling plateau record with a new adult reward delta.

Parameters:
- `plateauRecord` - Existing reward-delta evidence for the active adult zone.
- `rewardDelta` - Latest normalized reward delta observed for the current evaluation window.
- `marginalEpsilon` - Smallest improvement that still counts as meaningful positive return.

Returns: Updated plateau state with the retained reward window and stagnation decision.

### computeMarginalReturn

```ts
computeMarginalReturn(
  rewardDelta: number,
  structuralEditCount: number,
): number
```

Resolves one marginal-return score for the current adult structural edit batch.

Parameters:
- `rewardDelta` - Normalized reward improvement produced by the current edit batch.
- `structuralEditCount` - Number of structural edits responsible for the observed reward delta.

Returns: Reward improvement normalized by the structural edit count.

## neat/nge-adult/neat.nge-adult.constants.ts

Rolling evaluation-window length used by the adult plateau detector.

### NGE_ADULT_DEFAULT_GAIN_STABILITY_TOLERANCE

Allowed gain deviation before one adult zone is considered unstable again.

### NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW

Rolling evaluation-window length used by the adult gain-stability detector.

### NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR

Residual growth-budget fraction preserved while the adult phase biases toward prune and compact.

### NGE_ADULT_DEFAULT_GROWTH_FOCUS_FLOOR

Minimum normalized focus score required before adult growth remains eligible at all.

### NGE_ADULT_DEFAULT_MARGINAL_EPSILON

Smallest meaningful reward improvement that still counts as positive adult return.

### NGE_ADULT_DEFAULT_PLATEAU_WINDOW

Rolling evaluation-window length used by the adult plateau detector.

## neat/nge-adult/neat.nge-adult.equilibrium.ts

### advanceGainStabilityRecord

```ts
advanceGainStabilityRecord(
  gainStabilityRecord: GainStabilityRecord,
  gainMeasurement: number,
  gainStabilityTolerance: number,
): GainStabilityRecord
```

Advances one rolling gain-stability record with a new adult neuromodulator gain sample.

Parameters:
- `gainStabilityRecord` - Existing gain evidence for the active adult zone.
- `gainMeasurement` - Latest gain measurement observed for the current evaluation window.
- `gainStabilityTolerance` - Allowed deviation around the rolling mean.

Returns: Updated gain-stability evidence for the active adult zone.

### detectEquilibriumCandidate

```ts
detectEquilibriumCandidate(
  zoneId: string,
  plateauRecord: PlateauRecord,
  gainStabilityRecord: GainStabilityRecord,
): EquilibriumCandidate | null
```

Detects whether one adult zone is ready to emit an equilibrium candidate.

Parameters:
- `zoneId` - Stable adult-zone identifier under evaluation.
- `plateauRecord` - Plateau evidence for the current adult zone.
- `gainStabilityRecord` - Gain-stability evidence for the current adult zone.

Returns: An equilibrium candidate when both adult guards hold; otherwise null.

## neat/nge-adult/neat.nge-adult.utils.ts

### createAdultState

```ts
createAdultState(
  zoneId: string,
): AdultState
```

Create the initial owner-local adult state for one zone.

Parameters:
- `zoneId` - Stable adult-zone identifier that anchors the seeded state.

Returns: A compile-only placeholder while the Step 06 red contract is active.
