# evaluation

Public evaluation contracts for scoring Flappy Bird policies.

This folder exists to answer a narrower question than the trainer does:
given one network, one or more deterministic seeds, and one scoring policy,
what evidence should evolution use when deciding whether that network is any
good?

The answer in this example is intentionally stricter than a toy demo. A
single rollout is useful for inspection, but shared-seed batches are the real
selection surface because they reduce luck and expose instability.

Read the exports in that order:

- `FlappyRolloutOptions` defines what a caller may ask evaluation to do.
- `FlappyEpisodeResult` captures what happened in one seeded episode.
- `FlappySeedBatchEvaluation` captures the trainer-facing evidence used to
  rank genomes more fairly.

## evaluation/evaluation.types.ts

### FlappyEpisodeResult

Summary metrics for a single Flappy episode rollout.

The result intentionally keeps both a single scalar `fitness` and the channel
breakdown that produced it, which makes reward debugging much easier.

### FlappyNetworkLike

Minimal network contract required by Flappy evaluation.

The evaluation layer depends only on activation plus an optional stable id
used for deterministic seed mixing.

### FlappyRolloutOptions

Runtime controls for one rollout evaluation.

This is the public control surface for evaluation callers. The rollout layer
later normalizes these options into execution-safe context values.

### FlappySeedBatchEvaluation

Aggregate statistics from evaluating one network across shared seeds.

These statistics are the trainer-facing view of evaluation quality: mean,
median, $p90$, stability, and average gameplay progress.

## evaluation/evaluation.constants.ts

Default difficulty scale for rollouts when caller does not provide one.

A value of `1` means full adaptive difficulty is enabled during evaluation.

### FLAPPY_EVALUATION_DEFAULT_DIFFICULTY_SCALE

Default difficulty scale for rollouts when caller does not provide one.

A value of `1` means full adaptive difficulty is enabled during evaluation.

### FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_CONSECUTIVE_FRAMES

Default consecutive unrecoverable frames required for early termination.

### FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_GRACE_FRAMES

Default grace period (frames) before early termination checks begin.

### FLAPPY_EVALUATION_DEFAULT_PIPE_PROGRESS_TARGET

Default pipe-progress target used when normalizing rollout fitness.

This target anchors the progress channel so normalization remains meaningful
even when individual episodes vary widely in difficulty and duration.

### FLAPPY_EVALUATION_DENSE_SHAPING_FRAMES_NORMALIZER

Dense shaping normalization factor per survived frame.

Calibrated to match the new centering-focused weight profile. The max
per-frame dense reward with perfect centering is approximately:
alignment (3.0) + clearance (2.5) + centering quality (3.0) + velocity
stability (1.5) = 10.0. Setting the normalizer to 10.0 keeps a perfectly
centered bird near a normalizedDenseShaping of 1.0.

### FLAPPY_EVALUATION_NORMALIZED_DENSE_WEIGHT

Dense-shaping channel weight in normalized fitness composition.

### FLAPPY_EVALUATION_NORMALIZED_PROGRESS_WEIGHT

Pipe-progress channel weight in normalized fitness composition.

### FLAPPY_EVALUATION_NORMALIZED_SURVIVAL_WEIGHT

Survival channel weight in normalized fitness composition.

### FLAPPY_EVALUATION_NORMALIZED_TERMINAL_WEIGHT

Terminal-shaping channel weight in normalized fitness composition.

### FLAPPY_EVALUATION_ROBUST_STDDEV_PENALTY

Robust fitness penalty multiplier applied to standard deviation.

A higher value penalizes instability more strongly when computing robust
fitness from a shared-seed batch.

### FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_A

Seed-mix first multiplicative avalanche constant.

### FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_B

Seed-mix second multiplicative avalanche constant.

### FLAPPY_EVALUATION_SEED_MIX_XOR_SALT

Seed-mix additive constant used to decorrelate nearby genome ids.

Together with the multiplicative constants below, this creates a small
avalanche-style mixing pipeline for deterministic seed derivation.

### FLAPPY_EVALUATION_UNRECOVERABLE_ABOVE_GAP_DELTA

Upper-gap delta threshold used by early termination heuristic.

### FLAPPY_EVALUATION_UNRECOVERABLE_BELOW_GAP_DELTA

Lower-gap delta threshold used by early termination heuristic.

### FLAPPY_EVALUATION_UNRECOVERABLE_CLEARANCE_THRESHOLD

Unrecoverable clearance threshold used by early termination heuristic.

### FLAPPY_EVALUATION_UNRECOVERABLE_FALLING_VELOCITY

Falling-speed threshold used by early termination heuristic.

### FLAPPY_EVALUATION_UNRECOVERABLE_RISING_VELOCITY

Rising-speed threshold used by early termination heuristic.

## evaluation/evaluation.fitness.utils.ts

### evaluateFlappyFitness

```ts
evaluateFlappyFitness(
  network: FlappyNetworkLike,
  rolloutOptions: FlappyRolloutOptions,
): number
```

Evaluate a network on a single deterministic Flappy Bird episode.

This is the simplest evaluation entrypoint: one policy, one rollout, one
scalar fitness.

Parameters:

- `network` - Genome/network to evaluate.
- `rolloutOptions` - Optional rollout controls.

Returns: Fitness score (higher is better).

### evaluateFlappyFitnessAcrossSeeds

```ts
evaluateFlappyFitnessAcrossSeeds(
  network: FlappyNetworkLike,
  sharedSeeds: readonly number[],
  rolloutOptions: FlappyRolloutOptions,
): FlappySeedBatchEvaluation
```

Evaluate a network on a shared batch of deterministic seeds.

Educational note:
Shared-seed evaluation reduces luck. Every genome in the same comparison set
sees the same rollout seeds, which makes the aggregate statistics much more
useful for selection than a single lucky episode.

Parameters:

- `network` - Genome/network to evaluate.
- `sharedSeeds` - Shared deterministic seeds used for all genomes.
- `rolloutOptions` - Optional rollout controls.

Returns: Robust aggregate metrics for selection/ranking.

Example:

```ts
const aggregate = evaluateFlappyFitnessAcrossSeeds(network, [11, 22, 33], {
  normalizeFitness: true,
});
```

### evaluateFlappyFitnessAcrossSeedsWithInferenceChannel

```ts
evaluateFlappyFitnessAcrossSeedsWithInferenceChannel(
  network: default,
  sharedSeeds: readonly number[],
  options: { rolloutOptions?: FlappyRolloutOptions | undefined; workerUrl: string; },
): Promise<FlappySeedBatchEvaluation>
```

Evaluate a network through one persistent inference channel across shared seeds.

The same worker-side predictor is reset between seeded episodes so the
browser worker can reuse warm transport state without leaking recurrent
memory across rollout boundaries.

Parameters:

- `network` - Network to evaluate through one persistent inference channel.
- `sharedSeeds` - Shared deterministic seeds used for all genomes.
- `options` - Rollout controls plus the browser worker bundle URL.

Returns: Robust aggregate metrics for selection/ranking.

### evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker

```ts
evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker(
  sharedInferenceWorker: SharedInferenceWorker,
  sharedSeeds: readonly number[],
  options: { networkId?: number | undefined; rolloutOptions?: FlappyRolloutOptions | undefined; },
): Promise<FlappySeedBatchEvaluation>
```

Evaluate one shared-memory predictor across a deterministic seed batch.

This helper keeps one `SharedInferenceWorker` warm across the whole seed set
so the caller can parallelize across genomes without paying one bootstrap
cost per seeded rollout.

Parameters:

- `sharedInferenceWorker` - Persistent shared-memory predictor for one genome.
- `sharedSeeds` - Shared deterministic seeds used for the evaluation batch.
- `options` - Optional rollout controls plus a stable network id for seed mixing.

Returns: Robust aggregate metrics for selection/ranking.

### evaluateFlappyFitnessWithInferenceChannel

```ts
evaluateFlappyFitnessWithInferenceChannel(
  network: default,
  options: { rolloutOptions?: FlappyRolloutOptions | undefined; workerUrl: string; },
): Promise<number>
```

Evaluate a network through one persistent inference channel on a single seeded episode.

This browser-worker-oriented helper reuses one worker-side predictor instead
of calling `network.activate(...)` directly on the hot rollout path.

Parameters:

- `network` - Network to evaluate through one persistent inference channel.
- `options` - Rollout controls plus the browser worker bundle URL.

Returns: Fitness score (higher is better).

### runClearedRolloutEpisode

```ts
runClearedRolloutEpisode(
  network: FlappyNetworkLike,
  rolloutOptions: FlappyRolloutOptions,
): FlappyEpisodeResult
```

Runs one rollout after resetting any carried recurrent network state.

Stateful builders such as NARX, GRU, and LSTM must start each deterministic
Flappy rollout from a clean memory slate. Feed-forward networks ignore the
optional `clear()` hook, but recurrent networks use it to avoid leaking state
across shared-seed evaluations.

Parameters:

- `network` - Network being evaluated.
- `rolloutOptions` - Rollout controls for this episode.

Returns: One deterministic episode result.

## evaluation/evaluation.seed.utils.ts

### mixGenomeEvaluationSeed

```ts
mixGenomeEvaluationSeed(
  genomeId: number,
): number
```

Mixes a genome identifier into a stable uint32 rollout seed.

This keeps evaluation deterministic per genome while still spreading nearby
genome ids across the RNG state space to reduce correlated rollouts.

If you want background reading, the Wikipedia article on "hash function"
gives a reasonable intuition for why a few avalanche-style mixing steps help
nearby ids map to less-correlated seed values.

Parameters:

- `genomeId` - Genome id from NEAT bookkeeping.

Returns: Mixed uint32 seed.

## evaluation/evaluation.rollout.service.ts

Public rollout compatibility facade.

Keeping this file at the evaluation layer preserves the established import
path while the actual rollout orchestration lives behind the dedicated
rollout-owned module boundary.

This is the public evaluation-layer shelf for callers that should not need to
know about the rollout subfolder layout.

Minimal usage sketch:

```ts
const result = rolloutEpisode(network, {
  seed: 123,
  normalizeFitness: true,
});
```

### rolloutEpisode

```ts
rolloutEpisode(
  network: FlappyNetworkLike,
  rolloutOptions: FlappyRolloutOptions,
): FlappyEpisodeResult
```

Roll out an episode and return details.

Parameters:

- `network` - Genome/network to evaluate.
- `rolloutOptions` - Optional rollout controls.

Returns: Episode result details.

Example:

```ts
const result = rolloutEpisode(network, {
  seed: 123,
  normalizeFitness: true,
  maxFrames: 2_000,
});

console.log(result.fitness, result.doneReason);
```

### rolloutEpisodeWithPredictor

```ts
rolloutEpisodeWithPredictor(
  options: { predict: (observationVector: number[]) => Promise<unknown>; rolloutOptions?: FlappyRolloutOptions | undefined; networkId?: number | undefined; },
): Promise<FlappyEpisodeResult>
```

Roll out an episode against one async predictor callback.

This browser-worker-oriented variant preserves the same seeded rollout and
shaping semantics as `rolloutEpisode(...)` while sourcing control decisions
from an async inference boundary such as `InferenceChannel.predict(...)`.

Parameters:

- `options` - Predictor callback plus optional rollout controls.

Returns: Episode result details.

## evaluation/evaluation.worker-pool.ts

### FlappyEvaluationWorkerPool

Bounded shared-worker pool for parallel Flappy evaluation across genomes.

The pool parallelizes the expensive cross-genome part of evaluation while
keeping each individual genome on one persistent predictor for its seed
batch. That preserves recurrent reset semantics and avoids reopening a worker
for every single seeded rollout.

#### dispose

```ts
dispose(): Promise<void>
```

Releases every active shared worker and clears cached population state.

Returns: Nothing.

#### evaluateGenomesAcrossSeeds

```ts
evaluateGenomesAcrossSeeds(
  genomes: readonly WorkerPoolGenome[],
  sharedSeeds: readonly number[],
  rolloutOptions: FlappyRolloutOptions,
): Promise<Map<WorkerPoolGenome, FlappySeedBatchEvaluation>>
```

Evaluate one genome subset across a shared deterministic seed batch.

Parameters:

- `genomes` - Ordered genome shelf to score.
- `sharedSeeds` - Shared deterministic seeds used for each genome.
- `rolloutOptions` - Rollout controls reused across the whole batch.

Returns: Aggregates keyed by genome in the caller's original order.

#### initialize

```ts
initialize(
  genomes: readonly WorkerPoolGenome[],
): Promise<void>
```

Prepares exported payloads and empty slot state for one population shelf.

Parameters:

- `genomes` - Population that may be evaluated during the next generation.

Returns: Nothing.

#### parallelWorkerPool

Exposes the shared generic scheduler for public ordered batch helpers.

#### resolveOrderedPayloads

```ts
resolveOrderedPayloads(
  genomes: readonly WorkerPoolGenome[],
): Promise<TransferableInferencePayload[]>
```

Resolves the ordered transferable payload shelf for one genome batch.

Parameters:

- `genomes` - Genome batch that may be evaluated next.

Returns: Ordered transferable payload shelf aligned to the input genomes.

### FlappyEvaluationWorkerPoolOptions

Optional delivery controls for the Flappy shared-worker evaluation pool.

Browser examples usually resolve the worker bundle URL relative to the
evolution worker location. Tests can omit this and inject mocks instead.
