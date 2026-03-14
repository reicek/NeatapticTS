# neat/adaptive/mutation

Mutation-side adaptive controllers.

This category covers how NEAT changes per-genome mutation pressure and how
operator success statistics are decayed so exploration stays responsive over
long training runs.

## neat/adaptive/mutation/adaptive.mutation.ts

### applyAdaptiveMutation

```ts
applyAdaptiveMutation(): void
```

Self-adaptive per-genome mutation tuning.

This function implements several strategies to adjust each genome's
internal mutation rate (`g._mutRate`) and optionally its mutation
amount (`g._mutAmount`) over time. Strategies include:
- `twoTier`: push top and bottom halves in opposite directions to
  create exploration/exploitation balance.
- `exploreLow`: preferentially increase mutation for lower-scoring
  genomes to promote exploration.
- `anneal`: gradually reduce mutation deltas over time.

The method reads `this.options.adaptiveMutation` for configuration
and mutates genomes in-place.

Example:

// configuration example:
// options.adaptiveMutation = { enabled: true, initialRate: 0.5, adaptEvery: 1, strategy: 'twoTier', minRate: 0.01, maxRate: 1 }
engine.applyAdaptiveMutation();

### applyMutationsToPopulation

```ts
applyMutationsToPopulation(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
  partitions: MutationPartitions,
  settings: MutationSettings,
  randomSource: () => number,
): MutationOutcome
```

Apply mutation updates to the population.

Parameters:
- `population` - - Full population to mutate.
- `partitions` - - Scored partitions.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.

Returns: Mutation outcome flags.

### applyOperatorDecay

```ts
applyOperatorDecay(
  stats: Map<string, { success: number; attempts: number; }>,
  entries: [string, { success: number; attempts: number; }][],
  decay: number,
): void
```

Apply exponential decay to each operator statistic entry.

Parameters:
- `stats` - - Operator statistics map.
- `entries` - - Operator stat entries to update.
- `decay` - - Decay factor.

### applyTwoTierFallback

```ts
applyTwoTierFallback(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
  settings: MutationSettings,
): void
```

Apply two-tier fallback balancing.

Parameters:
- `population` - - Population of genomes.
- `settings` - - Resolved settings.

### resolveMutationSettings

```ts
resolveMutationSettings(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; },
): MutationSettings
```

Resolve mutation settings derived from configuration and engine state.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Adaptive mutation configuration.

Returns: Resolved mutation settings.

### resolveOperatorDecay

```ts
resolveOperatorDecay(
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; alpha?: number | undefined; decay?: number | undefined; },
): number
```

Resolve the decay factor for operator statistics.

Parameters:
- `config` - - Operator adaptation configuration.

Returns: Decay factor for exponential smoothing.

### shouldAdaptThisGeneration

```ts
shouldAdaptThisGeneration(
  generation: number,
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; },
): boolean
```

Check whether mutation adaptation should run this generation.

Parameters:
- `generation` - - Current generation index.
- `config` - - Adaptive mutation configuration.

Returns: True if adaptation should run.

### shouldApplyTwoTierFallback

```ts
shouldApplyTwoTierFallback(
  strategy: string,
  outcome: MutationOutcome,
): boolean
```

Determine whether a two-tier fallback is needed.

Parameters:
- `strategy` - - Mutation strategy identifier.
- `outcome` - - Mutation outcome flags.

Returns: True if fallback should run.

## neat/adaptive/mutation/adaptive.mutation.utils.ts

### applyAnnealDelta

```ts
applyAnnealDelta(
  baseDelta: number,
  settings: MutationSettings,
): number
```

Apply annealing adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `settings` - - Resolved settings.

Returns: Adjusted delta.

### applyExploreLowDelta

```ts
applyExploreLowDelta(
  baseDelta: number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Apply explore-low adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyMutationAmount

```ts
applyMutationAmount(
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  settings: MutationSettings,
  randomSource: () => number,
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): void
```

Apply mutation-amount adjustments to a genome.

Parameters:
- `genome` - - Current genome.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

### applyMutationsToPopulation

```ts
applyMutationsToPopulation(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
  partitions: MutationPartitions,
  settings: MutationSettings,
  randomSource: () => number,
): MutationOutcome
```

Apply mutation updates to the population.

Parameters:
- `population` - - Full population to mutate.
- `partitions` - - Scored partitions.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.

Returns: Mutation outcome flags.

### applyTwoTierAmountDelta

```ts
applyTwoTierAmountDelta(
  baseDelta: number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Apply two-tier adjustments to amount delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierDelta

```ts
applyTwoTierDelta(
  baseDelta: number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Apply two-tier adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierFallback

```ts
applyTwoTierFallback(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
  settings: MutationSettings,
): void
```

Apply two-tier fallback balancing.

Parameters:
- `population` - - Population of genomes.
- `settings` - - Resolved settings.

### clampValue

```ts
clampValue(
  value: number,
  min: number,
  max: number,
): number
```

Clamp a value between min and max bounds.

Parameters:
- `value` - - Value to clamp.
- `min` - - Minimum bound.
- `max` - - Maximum bound.

Returns: Clamped value.

### collectScoredGenomes

```ts
collectScoredGenomes(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
): { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]
```

Collect genomes with numeric scores.

Parameters:
- `population` - - Population of genomes.

Returns: Scored genomes.

### createRandomDelta

```ts
createRandomDelta(
  sigmaBase: number,
  randomSource: () => number,
): number
```

Create a signed random delta scaled by sigma.

Parameters:
- `sigmaBase` - - Sigma scaling factor.
- `randomSource` - - Random number provider.

Returns: Signed delta.

### resolveAmountDelta

```ts
resolveAmountDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Resolve mutation-amount delta based on strategy.

Parameters:
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Signed mutation amount delta.

### resolveMutationSettings

```ts
resolveMutationSettings(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; },
): MutationSettings
```

Resolve mutation settings derived from configuration and engine state.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Adaptive mutation configuration.

Returns: Resolved mutation settings.

### resolveRandomSource

```ts
resolveRandomSource(
  engine: NeatLikeWithAdaptive,
): () => number
```

Resolve a random source that matches the legacy RNG usage.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Random number provider.

### resolveRateDelta

```ts
resolveRateDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Resolve mutation-rate delta based on strategy.

Parameters:
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Signed mutation rate delta.

### shouldAdaptThisGeneration

```ts
shouldAdaptThisGeneration(
  generation: number,
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; },
): boolean
```

Check whether mutation adaptation should run this generation.

Parameters:
- `generation` - - Current generation index.
- `config` - - Adaptive mutation configuration.

Returns: True if adaptation should run.

### shouldApplyTwoTierFallback

```ts
shouldApplyTwoTierFallback(
  strategy: string,
  outcome: MutationOutcome,
): boolean
```

Determine whether a two-tier fallback is needed.

Parameters:
- `strategy` - - Mutation strategy identifier.
- `outcome` - - Mutation outcome flags.

Returns: True if fallback should run.

### sortScoredGenomes

```ts
sortScoredGenomes(
  scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
): { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]
```

Sort scored genomes in ascending score order.

Parameters:
- `scoredGenomes` - - Scored genomes.

Returns: Sorted genomes.

### splitScoredGenomes

```ts
splitScoredGenomes(
  scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
): MutationPartitions
```

Split scored genomes into top and bottom halves.

Parameters:
- `scoredGenomes` - - Sorted scored genomes.

Returns: Partitions used by strategy rules.

## neat/adaptive/mutation/adaptive.operator.utils.ts

### applyOperatorDecay

```ts
applyOperatorDecay(
  stats: Map<string, { success: number; attempts: number; }>,
  entries: [string, { success: number; attempts: number; }][],
  decay: number,
): void
```

Apply exponential decay to each operator statistic entry.

Parameters:
- `stats` - - Operator statistics map.
- `entries` - - Operator stat entries to update.
- `decay` - - Decay factor.

### collectOperatorStatsEntries

```ts
collectOperatorStatsEntries(
  stats: Map<string, { success: number; attempts: number; }>,
): [string, { success: number; attempts: number; }][]
```

Collect operator statistic entries for processing.

Parameters:
- `stats` - - Operator statistics map.

Returns: Array of operator stat entries.

### decayOperatorStat

```ts
decayOperatorStat(
  operatorStat: { success: number; attempts: number; },
  decay: number,
): { success: number; attempts: number; }
```

Apply decay to a single operator statistic record.

Parameters:
- `operatorStat` - - Operator statistic record.
- `decay` - - Decay factor.

Returns: Decayed operator statistic record.

### resolveOperatorDecay

```ts
resolveOperatorDecay(
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; alpha?: number | undefined; decay?: number | undefined; },
): number
```

Resolve the decay factor for operator statistics.

Parameters:
- `config` - - Operator adaptation configuration.

Returns: Decay factor for exponential smoothing.
