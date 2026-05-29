# neat/nge-juvenile

Error raised when one juvenile morph delta would exceed a DNA-configured budget.

## neat/nge-juvenile/neat.nge-juvenile.types.ts

### NgeFocusScore

One module-level focus result containing both the raw and normalized score.

### NgeFocusVector

Normalized focus vector emitted for one juvenile evaluation window.
Contains per-module probability-like scores produced by the weighted focus formula.

### NgeGrowthBudget

DNA-configured structural caps and live counts for one locally growing module.

### NgeHysteresisState

JSON-safe hysteresis state tracked persistently across juvenile morphology evaluation windows.
Persists growth and prune streak counts plus cooldown counters between successive windows.

### NgeJuvenileFocusWeights

Focus-weight shelf consumed by the juvenile weighted focus formula for module scoring.
Each weight scales one normalized metric — utilization, reward, novelty, stability, or cost.

### NgeJuvenilePhaseConfig

Resolved juvenile-phase configuration for focus scoring and later morph guards.

### NgeModuleMetricsSnapshot

Cheap per-module metrics snapshot consumed by the juvenile focus scorer.

### NgeMorphDelta

Dry-run structural delta that later juvenile passes can validate or roll back.

### NgeProbeDecision

Pure scheduler decision emitted for one epoch's cadence check in the juvenile phase.
Signals whether the gate is open and which probe kind has been selected for execution.

### NgeProbeKind

Canonical probe kinds cycled by the juvenile perturbation scheduler across episodes.
Each kind suppresses, perturbs, or gates a different aspect of module behavior.

### NgeProbeLedgerEntry

Append-only probe ledger entry produced by one juvenile scheduled perturbation pass.
Records the probe kind, target module, epoch index, and signed reward delta for analysis.

### NgeProbeSchedulerConfig

Fully resolved configuration for the cadence-gated juvenile perturbation probe scheduler.
Controls probe-kind rotation, ledger size cap, and per-kind severity parameters.

### NgeProbeSchedulerState

JSON-safe probe scheduler state tracked persistently across juvenile evaluation windows.
Holds the last-fired epoch, probe-kind rotation index, and the append-only probe ledger.

### NgePruneBudget

DNA-configured structural floors and permanent prune exemptions for one juvenile module.
Guards the minimum edge and node counts that no morph action may reduce below.

### NgePruneCandidate

One scored prune candidate supplied by the caller for dry-run ranking.

## neat/nge-juvenile/neat.nge-juvenile.grow.ts

### advanceGrowthHysteresis

```ts
advanceGrowthHysteresis(
  hysteresis: NgeHysteresisState,
  isPositiveFocusWindow: boolean,
): NgeHysteresisState
```

Advance the growth-side hysteresis counters for one evaluation window and return fresh state.

Parameters:
- `hysteresis` - Previous hysteresis state.
- `isPositiveFocusWindow` - Whether the current window carried positive focus evidence.

Returns: A fresh hysteresis state with the growth streak and cooldown advanced.

### canGrowNow

```ts
canGrowNow(
  hysteresis: NgeHysteresisState,
  config: NgeJuvenilePhaseConfig,
): boolean
```

Check whether juvenile growth may commit in the current window.

Parameters:
- `hysteresis` - Current hysteresis state tracked across windows.
- `config` - Resolved juvenile-phase configuration.

Returns: `true` when the positive-focus streak is satisfied and cooldown is clear.

### commitGrowth

```ts
commitGrowth(
  hysteresis: NgeHysteresisState,
  morphKind: NgeGrowthMorphKind,
  config: NgeJuvenilePhaseConfig,
): NgeHysteresisState
```

Commit one growth-side hysteresis update after a validated morph is applied.

Parameters:
- `hysteresis` - Previous hysteresis state.
- `morphKind` - Concrete growth morph kind that committed.
- `config` - Resolved juvenile-phase configuration.

Returns: A fresh hysteresis state ready for the next cooldown window.

### planEdgeDensification

```ts
planEdgeDensification(
  moduleId: string,
  budget: NgeGrowthBudget,
  focusScore: NgeFocusScore,
): NgeMorphDelta
```

Plan one local edge-densification delta for a single module, validating the DNA edge budget.

Parameters:
- `moduleId` - Module receiving the planned densification.
- `budget` - DNA-configured growth caps and current live counts.
- `focusScore` - Focus score for the target module.

Returns: One dry-run edge densification delta.

### planGrowthMorphs

```ts
planGrowthMorphs(
  moduleId: string,
  focusScore: NgeFocusScore,
  metrics: NgeModuleMetricsSnapshot,
  budget: NgeGrowthBudget,
  config: NgeJuvenilePhaseConfig,
  hysteresis: NgeHysteresisState,
): NgeMorphDelta[]
```

Plan all eligible local growth deltas for one module in edge-first priority order.

Parameters:
- `moduleId` - Module receiving all planned local growth actions.
- `focusScore` - Focus score for the target module.
- `metrics` - Module metrics whose utilization and reward delta drive eligibility.
- `budget` - DNA-configured growth caps and current live counts.
- `config` - Resolved juvenile-phase configuration.
- `hysteresis` - Current growth-side hysteresis state.

Returns: Zero or more validated dry-run morph deltas in edge-first priority order.

### planNodeAddition

```ts
planNodeAddition(
  moduleId: string,
  budget: NgeGrowthBudget,
  rewardDelta: number,
): NgeMorphDelta
```

Plan one rare evidence-gated node-addition delta for a single module.

Parameters:
- `moduleId` - Module receiving the planned node addition.
- `budget` - DNA-configured growth caps and current live counts.
- `rewardDelta` - Measured reward delta acting as the positive-evidence signal.

Returns: One dry-run node-addition delta.

### planSlotExpansion

```ts
planSlotExpansion(
  moduleId: string,
  hitRate: number,
  budget: NgeGrowthBudget,
  focusScore: NgeFocusScore,
  config: NgeJuvenilePhaseConfig,
): NgeMorphDelta
```

Plan one local episodic-slot expansion delta for a single module.

Phase B uses `metrics.utilization` as the hit-rate proxy until a dedicated episodic
hit-rate field lands in a later step.

Parameters:
- `moduleId` - Module receiving the planned slot expansion.
- `hitRate` - Episodic hit-rate proxy for the target module.
- `budget` - DNA-configured growth caps and current live counts.
- `focusScore` - Focus score for the target module.
- `config` - Resolved juvenile-phase configuration.

Returns: One dry-run slot expansion delta.

### validateMorphDelta

```ts
validateMorphDelta(
  delta: NgeMorphDelta,
  budget: NgeGrowthBudget,
): void
```

Re-validate one dry-run morph delta against the current structural budget.

Parameters:
- `delta` - Planned morph delta to validate.
- `budget` - DNA-configured growth caps and current live counts.

## neat/nge-juvenile/neat.nge-juvenile.focus.ts

### computeFocusScores

```ts
computeFocusScores(
  snapshots: readonly NgeModuleMetricsSnapshot[],
  config: Partial<NgeJuvenilePhaseConfig>,
): NgeFocusVector
```

Compute the weighted juvenile focus vector for one evaluation window.

The score math is deterministic for a fixed snapshot and config. The `computedAt`
field is metadata only and must not participate in any deterministic fingerprint.

Parameters:
- `snapshots` - Module metrics observed in the active evaluation slice.
- `config` - Partial or fully resolved juvenile focus configuration.

Returns: A focus vector carrying raw and normalized module scores.

### resolveFocusConfig

```ts
resolveFocusConfig(
  partial: Partial<NgeJuvenilePhaseConfig>,
): NgeJuvenilePhaseConfig
```

Resolve a partial juvenile focus config against the seeded plan defaults.

Parameters:
- `partial` - Partial config whose omitted fields should resolve conservatively.

Returns: A fully resolved config packet ready for deterministic focus scoring.

## neat/nge-juvenile/neat.nge-juvenile.probe.ts

### advanceSchedulerState

```ts
advanceSchedulerState(
  state: NgeProbeSchedulerState,
  epochIndex: number,
  entry: NgeProbeLedgerEntry,
  config: NgeProbeSchedulerConfig,
): NgeProbeSchedulerState
```

Advance the scheduler state after one measured probe result is available.

Parameters:
- `state` - Previous scheduler state.
- `epochIndex` - Epoch that produced the new probe result.
- `entry` - Append-only ledger entry describing the probe outcome.
- `config` - Fully resolved scheduler config.

Returns: A fresh scheduler state with updated cadence metadata and ledger.

### appendProbeLedgerEntry

```ts
appendProbeLedgerEntry(
  ledger: NgeProbeLedgerEntry[],
  entry: NgeProbeLedgerEntry,
  maxEntries: number,
): NgeProbeLedgerEntry[]
```

Append one probe entry while preserving immutability and the bounded ledger cap.

Parameters:
- `ledger` - Existing append-only probe ledger.
- `entry` - New entry to append.
- `maxEntries` - Maximum number of entries preserved in the returned ledger.

Returns: A new bounded ledger with the newest entry preserved.

### buildProbeLedgerEntry

```ts
buildProbeLedgerEntry(
  kind: NgeProbeKind,
  targetModuleId: string,
  epochIndex: number,
  rewardBefore: number,
  rewardAfter: number,
): NgeProbeLedgerEntry
```

Build one append-only probe ledger entry from caller-measured before and after reward readings.

Parameters:
- `kind` - Probe kind applied to the target module.
- `targetModuleId` - Module whose local behavior was perturbed.
- `epochIndex` - Epoch that recorded the probe result.
- `rewardBefore` - Reward measured before the perturbation.
- `rewardAfter` - Reward measured after the perturbation.

Returns: One append-only probe ledger entry.

### computeProbeRewardDelta

```ts
computeProbeRewardDelta(
  ledger: NgeProbeLedgerEntry[],
  moduleId: string,
): number
```

Compute the mean signed probe delta for one module across the current ledger.

Parameters:
- `ledger` - Append-only probe ledger.
- `moduleId` - Module whose probe deltas should be averaged.

Returns: Mean signed reward delta, or `0` when no matching probes exist.

### decideProbe

```ts
decideProbe(
  epochIndex: number,
  state: NgeProbeSchedulerState,
  config: NgeProbeSchedulerConfig,
): NgeProbeDecision
```

Decide whether the current epoch may execute one expensive perturbation probe.

Parameters:
- `epochIndex` - Current training or evaluation epoch.
- `state` - Current scheduler state.
- `config` - Fully resolved scheduler config.

Returns: A pure decision packet describing cadence and the selected probe kind.

### defaultProbeSchedulerState

```ts
defaultProbeSchedulerState(): NgeProbeSchedulerState
```

Build the zeroed scheduler state used before any probe has fired.

Returns: A JSON-safe scheduler state packet for one episode.

### deserializeLedger

```ts
deserializeLedger(
  json: string,
): NgeProbeLedgerEntry[]
```

Deserialize one JSON-serialized probe ledger previously produced by `serializeLedger`.
Throws `NgeJuvenile_ProbeError` when the payload cannot be parsed or is not a JSON array.

Parameters:
- `json` - JSON string previously produced by `serializeLedger`.

Returns: Parsed probe ledger entries when the payload is a valid JSON array.

### resolveProbeSchedulerConfig

```ts
resolveProbeSchedulerConfig(
  partial: Partial<NgeProbeSchedulerConfig>,
): NgeProbeSchedulerConfig
```

Resolve a partial probe scheduler config against the seeded plan defaults.

Parameters:
- `partial` - Partial config whose omitted fields should resolve conservatively.

Returns: A fully resolved probe scheduler config packet.

### serializeLedger

```ts
serializeLedger(
  ledger: NgeProbeLedgerEntry[],
): string
```

Serialize the append-only probe ledger into a stable JSON string for checkpoint storage.

Parameters:
- `ledger` - Probe ledger to serialize.

Returns: JSON string containing the ledger entries in order.

## neat/nge-juvenile/neat.nge-juvenile.prune.ts

### advancePruneHysteresis

```ts
advancePruneHysteresis(
  hysteresis: NgeHysteresisState,
  isUnderuseWindow: boolean,
): NgeHysteresisState
```

Advance the prune-side hysteresis counters for one evaluation window and return fresh state.

Parameters:
- `hysteresis` - Previous hysteresis state.
- `isUnderuseWindow` - Whether the current window carried prune evidence.

Returns: A fresh hysteresis state with the prune streak and cooldown advanced.

### canPruneNow

```ts
canPruneNow(
  hysteresis: NgeHysteresisState,
  config: NgeJuvenilePhaseConfig,
): boolean
```

Check whether juvenile prune or compact actions may commit in the current window.

Parameters:
- `hysteresis` - Current prune-side hysteresis state tracked across windows.
- `config` - Resolved juvenile-phase configuration.

Returns: `true` when the underuse streak is satisfied and cooldown is clear.

### commitPrune

```ts
commitPrune(
  hysteresis: NgeHysteresisState,
  morphKind: NgePruneMorphKind,
  config: NgeJuvenilePhaseConfig,
): NgeHysteresisState
```

Commit one prune-side hysteresis update after a validated morph is applied.

Parameters:
- `hysteresis` - Previous hysteresis state.
- `morphKind` - Concrete prune or compact morph kind that committed.
- `config` - Resolved juvenile-phase configuration.

Returns: A fresh hysteresis state ready for the next cooldown window.

### planCompact

```ts
planCompact(
  moduleId: string,
  budget: NgePruneBudget,
): NgeMorphDelta
```

Plan one dry-run compact delta for a single module, verifying the node floor before returning.

Parameters:
- `moduleId` - Module receiving the planned compact action.
- `budget` - DNA floors and current structural counts for the module.

Returns: One dry-run compact delta.

### planEdgePrune

```ts
planEdgePrune(
  moduleId: string,
  candidate: NgePruneCandidate,
  budget: NgePruneBudget,
): NgeMorphDelta
```

Plan one dry-run edge-prune delta for a single module, respecting cost-exempt edges and DNA floor.

Parameters:
- `moduleId` - Module receiving the planned edge prune.
- `candidate` - Candidate edge selected for dry-run pruning.
- `budget` - DNA floors and prune exemptions for the module.

Returns: One dry-run edge-prune delta.

### planPruneMorphs

```ts
planPruneMorphs(
  moduleId: string,
  budget: NgePruneBudget,
  candidates: readonly NgePruneCandidate[],
  config: NgeJuvenilePhaseConfig,
  hysteresis: NgeHysteresisState,
): NgeMorphDelta[]
```

Plan all eligible dry-run prune deltas for one module in prune-before-compact order.

Parameters:
- `moduleId` - Module receiving all planned prune-side actions.
- `budget` - DNA floors, exemptions, and current structural counts.
- `candidates` - Caller-supplied edge candidates ranked locally within the module.
- `config` - Resolved juvenile-phase configuration.
- `hysteresis` - Current prune-side hysteresis state.

Returns: Zero or more validated dry-run prune deltas in priority order.

### selectPruneCandidate

```ts
selectPruneCandidate(
  candidates: readonly NgePruneCandidate[],
  budget: NgePruneBudget,
): NgePruneCandidate
```

Select the highest-priority non-exempt prune candidate for one module, sorted by wiring cost.

Parameters:
- `candidates` - Caller-supplied candidate edges for one prune pass.
- `budget` - DNA floors and permanent prune exemptions for the module.

Returns: The highest-priority non-exempt candidate, ordered by wiring cost then edge length.

### validatePruneDelta

```ts
validatePruneDelta(
  delta: NgeMorphDelta,
  budget: NgePruneBudget,
): void
```

Re-validate one dry-run prune delta against the current structural floors.

Parameters:
- `delta` - Planned morph delta to validate.
- `budget` - DNA floors and current structural counts for the module.

## neat/nge-juvenile/neat.nge-juvenile.errors.ts

### NgeJuvenile_BudgetError

Error raised when one juvenile morph delta would exceed a DNA-configured budget.

### NgeJuvenile_MorphError

Error raised when one dry-run juvenile morph validation fails locally.

### NgeJuvenile_ProbeError

Error raised when probe scheduling or ledger deserialization fails validation.

## neat/nge-juvenile/neat.nge-juvenile.constants.ts

### NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT

Minimum viable edge increment applied by one approved juvenile densification step.
Each committed grow pass adds at least this many edges to the target module.

### NGE_JUVENILE_DEFAULT_EPISODIC_HIT_RATE_THRESHOLD

Hit-rate floor required before episodic slot growth becomes eligible for commit.
Slot expansion is blocked when a module's episodic recall rate falls at or below this value.

### NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS

Seed focus weights from the plan's initial default-threshold section for the juvenile scorer.
Drives the weighted formula that ranks modules by utilization, reward, novelty, stability, and cost.

### NGE_JUVENILE_DEFAULT_GAIN_STABILITY_TOLERANCE

Allowed mean-gain deviation before one module is treated as unstable.

### NGE_JUVENILE_DEFAULT_GAIN_STABILITY_WINDOW

Rolling-window length used for neuromodulator gain stabilization checks in the juvenile phase.
The mean gain is averaged over this many windows before stability is evaluated.

### NGE_JUVENILE_DEFAULT_GATING_EDGE_LENGTH_THRESHOLD

Default Euclidean edge-length threshold targeted by juvenile gating probes.
Edges shorter than this value are de-prioritized when selecting gating perturbation targets.

### NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT

Consecutive evaluation windows required before a growth or prune action may commit.
Prevents premature structural changes caused by transient evaluation signal spikes.

### NGE_JUVENILE_DEFAULT_LESION_SEVERITY

Default lesion severity where `1.0` suppresses the full target edge set.

### NGE_JUVENILE_DEFAULT_MIN_EDGE_FLOOR

Safe absolute minimum edge floor when DNA supplies no tighter prune bound.

### NGE_JUVENILE_DEFAULT_NODE_REWARD_DELTA_FLOOR

Exclusive lower bound on reward delta required to evidence-gate a node addition.
Node growth is blocked when the module's focus reward delta is at or below this floor.

### NGE_JUVENILE_DEFAULT_NOISE_SIGMA

Default Gaussian standard deviation applied to module activations during noise probes.
Smaller values produce fine-grained perturbations; larger values create more disruptive noise.

### NGE_JUVENILE_DEFAULT_PROBE_CADENCE_EPOCHS

Probe scheduler cadence floor keeping expensive perturbations sparse across training epochs.
At least this many epochs must elapse between successive juvenile probe executions.

### NGE_JUVENILE_DEFAULT_PROBE_KINDS

Default deterministic probe-kind rotation applied by the juvenile perturbation scheduler.
Each probe epoch advances the rotation index to cycle through lesion, noise, and gating kinds.

### NGE_JUVENILE_DEFAULT_PROBE_MAX_LEDGER_ENTRIES

Maximum number of append-only probe ledger entries preserved per episode for analysis.
Older entries are evicted in FIFO order when the ledger reaches this cap.

### NGE_JUVENILE_DEFAULT_PRUNE_COST_PRESSURE_THRESHOLD

Cost-pressure fraction above which one window counts as prune evidence.

### NGE_JUVENILE_DEFAULT_RECURRENT_REFRESH_FLOOR

Hidden-state refresh floor below which recurrent state becomes prune evidence.

### NGE_JUVENILE_DEFAULT_SLOT_EXPANSION_COUNT

Minimum viable episodic slot increment applied by one expansion step.

## neat/nge-juvenile/neat.nge-juvenile.utils.ts

Normalize one numeric vector with min-max scaling.

Degenerate vectors with a single value or zero range return uniform weights so
downstream focus math never emits `NaN`.

### minMaxNormalize

```ts
minMaxNormalize(
  values: readonly number[],
): number[]
```

Normalize one numeric vector with min-max scaling.

Degenerate vectors with a single value or zero range return uniform weights so
downstream focus math never emits `NaN`.

Parameters:
- `values` - Raw numeric vector to normalize.

Returns: A normalized vector in the `[0, 1]` range or uniform degenerate weights.

### softmaxTopK

```ts
softmaxTopK(
  items: readonly T[],
  k: number,
): T[]
```

Return the top-k items after softmax normalization of the `score` field.

Parameters:
- `items` - Candidate items carrying one scalar score.
- `k` - Maximum number of items to return.

Returns: The highest-ranked `k` items with stable tie-breaking by input index.
