# neat/pruning/core

Minimal NEAT host contract required by pruning helpers.

The pruning boundary only needs schedule settings, population structure, and
two shared adaptive fields, so this contract stays intentionally narrow.

## neat/pruning/core/pruning.types.ts

### AdaptivePruningOptions

Adaptive pruning options extracted from the NEAT host.

### EvolutionPruningOptions

Evolution pruning options extracted from the NEAT host.

### NeatLikeForPruning

Minimal NEAT host contract required by pruning helpers.

The pruning boundary only needs schedule settings, population structure, and
two shared adaptive fields, so this contract stays intentionally narrow.

### PopulationMetrics

Summary of population metrics used by adaptive pruning.

## neat/pruning/core/pruning.core.ts

Pruning mechanics used by scheduled and adaptive NEAT pruning.

This chapter holds the policy resolution and metric math that sit underneath
the public pruning entrypoints.

### applyAdaptivePruneLevelToPopulation

```ts
applyAdaptivePruneLevelToPopulation(
  host: NeatLikeForPruning,
  pruneLevel: number,
): void
```

Apply the shared adaptive prune level to every compatible genome.

Parameters:
- `host` - - NEAT host exposing the population.
- `pruneLevel` - - Prune level to apply.

Returns: Nothing. Compatible genomes are pruned in place.

### applyPruningToPopulation

```ts
applyPruningToPopulation(
  host: NeatLikeForPruning,
  options: { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; },
  targetSparsity: number,
): void
```

Apply scheduled pruning to each genome in the population.

Parameters:
- `host` - - NEAT host exposing the population.
- `options` - - Active scheduled pruning options.
- `targetSparsity` - - Target sparsity to apply.

Returns: Nothing. Genomes are pruned in place when supported.

### computeMeanConnectionCount

```ts
computeMeanConnectionCount(
  host: NeatLikeForPruning,
): number
```

Compute the average connection count per genome.

Parameters:
- `host` - - NEAT host exposing the population.

Returns: Average number of connections per genome.

### computeMeanNodeCount

```ts
computeMeanNodeCount(
  host: NeatLikeForPruning,
): number
```

Compute the average node count per genome.

Parameters:
- `host` - - NEAT host exposing the population.

Returns: Average number of nodes per genome.

### computeNextAdaptivePruneLevel

```ts
computeNextAdaptivePruneLevel(
  options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; },
  currentPruneLevel: number,
  currentMetricValue: number,
  targetRemainingMetric: number,
): number
```

Compute the next adaptive prune level.

Parameters:
- `options` - - Adaptive pruning options.
- `currentPruneLevel` - - Current shared prune level.
- `currentMetricValue` - - Current observed metric value.
- `targetRemainingMetric` - - Target remaining metric value.

Returns: Updated prune level clamped into the valid sparsity range.

### computePopulationMetrics

```ts
computePopulationMetrics(
  host: NeatLikeForPruning,
): PopulationMetrics
```

Compute the population metrics used by adaptive pruning.

Parameters:
- `host` - - NEAT host exposing the population.

Returns: Summary of mean node and connection counts.

### computeRampFraction

```ts
computeRampFraction(
  host: NeatLikeForPruning,
  options: { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; },
): number
```

Compute the ramp completion fraction for scheduled pruning.

Parameters:
- `host` - - NEAT host exposing generation state.
- `options` - - Active scheduled pruning options.

Returns: Fraction in `[0, 1]` indicating ramp completion.

### computeTargetRemainingMetric

```ts
computeTargetRemainingMetric(
  options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; },
  adaptivePruneBaseline: number,
): number
```

Compute the target remaining metric implied by the desired sparsity.

Parameters:
- `options` - - Adaptive pruning options.
- `adaptivePruneBaseline` - - Baseline metric value.

Returns: Target remaining metric value.

### computeTargetSparsityNow

```ts
computeTargetSparsityNow(
  host: NeatLikeForPruning,
  options: { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; },
): number
```

Compute the target sparsity for the current generation.

Parameters:
- `host` - - NEAT host exposing generation state.
- `options` - - Active scheduled pruning options.

Returns: Target sparsity for the current generation.

### initializeAdaptivePruningState

```ts
initializeAdaptivePruningState(
  host: NeatLikeForPruning,
): void
```

Ensure the adaptive pruning state exists on the host.

Parameters:
- `host` - - NEAT host exposing adaptive pruning state.

Returns: Nothing. The shared prune level is initialized when missing.

### resolveActiveAdaptivePruningOptions

```ts
resolveActiveAdaptivePruningOptions(
  host: NeatLikeForPruning,
): { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; } | null
```

Resolve adaptive pruning options when enabled.

Parameters:
- `host` - - NEAT host exposing adaptive pruning options.

Returns: Adaptive pruning options when enabled, otherwise `null`.

### resolveActiveEvolutionPruningOptions

```ts
resolveActiveEvolutionPruningOptions(
  host: NeatLikeForPruning,
): { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; } | null
```

Resolve scheduled pruning options when they are active for the current generation.

Parameters:
- `host` - - NEAT host exposing generation and pruning options.

Returns: Evolution pruning options when active, otherwise `null`.

### resolveAdaptivePruneBaseline

```ts
resolveAdaptivePruneBaseline(
  host: NeatLikeForPruning,
  currentMetricValue: number,
): number
```

Resolve and persist the adaptive pruning baseline.

Parameters:
- `host` - - NEAT host exposing adaptive baseline state.
- `currentMetricValue` - - Currently observed metric value.

Returns: Baseline metric value used for adaptation.

### resolveObservedMetricValue

```ts
resolveObservedMetricValue(
  options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; },
  metrics: PopulationMetrics,
): number
```

Resolve the currently observed population metric for adaptive pruning.

Parameters:
- `options` - - Adaptive pruning options.
- `metrics` - - Population metric summary.

Returns: Current observed metric value used for adaptation.

### shouldAdjustAdaptivePruning

```ts
shouldAdjustAdaptivePruning(
  options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; },
  currentMetricValue: number,
  targetRemainingMetric: number,
  adaptivePruneBaseline: number,
): boolean
```

Decide whether adaptive pruning should adjust the prune level.

Parameters:
- `options` - - Adaptive pruning options.
- `currentMetricValue` - - Current observed metric value.
- `targetRemainingMetric` - - Target remaining metric value.
- `adaptivePruneBaseline` - - Baseline metric value.

Returns: `true` when the normalized drift exceeds the configured tolerance.
