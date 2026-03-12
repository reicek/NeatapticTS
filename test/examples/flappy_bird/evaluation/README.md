# evaluation

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

### evaluation.constants

Default difficulty scale for rollouts when caller does not provide one.

A value of `1` means full adaptive difficulty is enabled during evaluation.

### FLAPPY_EVALUATION_DEFAULT_DIFFICULTY_SCALE

### FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_CONSECUTIVE_FRAMES

### FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_GRACE_FRAMES

### FLAPPY_EVALUATION_DEFAULT_PIPE_PROGRESS_TARGET

### FLAPPY_EVALUATION_DENSE_SHAPING_FRAMES_NORMALIZER

### FLAPPY_EVALUATION_NORMALIZED_DENSE_WEIGHT

### FLAPPY_EVALUATION_NORMALIZED_PROGRESS_WEIGHT

### FLAPPY_EVALUATION_NORMALIZED_SURVIVAL_WEIGHT

### FLAPPY_EVALUATION_NORMALIZED_TERMINAL_WEIGHT

### FLAPPY_EVALUATION_ROBUST_STDDEV_PENALTY

### FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_A

### FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_B

### FLAPPY_EVALUATION_SEED_MIX_XOR_SALT

### FLAPPY_EVALUATION_UNRECOVERABLE_ABOVE_GAP_DELTA

### FLAPPY_EVALUATION_UNRECOVERABLE_BELOW_GAP_DELTA

### FLAPPY_EVALUATION_UNRECOVERABLE_CLEARANCE_THRESHOLD

### FLAPPY_EVALUATION_UNRECOVERABLE_FALLING_VELOCITY

### FLAPPY_EVALUATION_UNRECOVERABLE_RISING_VELOCITY

## evaluation/evaluation.rollout.service.ts

### evaluation.rollout.service

Public rollout compatibility facade.

Keeping this file at the evaluation layer preserves the established import
path while the actual rollout orchestration lives behind the dedicated
rollout-owned module boundary.

This is the public evaluation-layer shelf for callers that should not need to
know about the rollout subfolder layout.

### rolloutEpisode

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutOptions: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyEpisodeResult`

Roll out an episode and return details.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutOptions` - - Optional rollout controls.

Returns: Episode result details.

## evaluation/evaluation.seed.utils.ts

### mixGenomeEvaluationSeed

`(genomeId: number) => number`

Mixes a genome identifier into a stable uint32 rollout seed.

This keeps evaluation deterministic per genome while still spreading nearby
genome ids across the RNG state space to reduce correlated rollouts.

If you want background reading, the Wikipedia article on "hash function"
gives a reasonable intuition for why a few avalanche-style mixing steps help
nearby ids map to less-correlated seed values.

Parameters:
- `genomeId` - - Genome id from NEAT bookkeeping.

Returns: Mixed uint32 seed.

## evaluation/evaluation.fitness.utils.ts

### evaluateFlappyFitness

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutOptions: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => number`

Evaluate a network on a single deterministic Flappy Bird episode.

This is the simplest evaluation entrypoint: one policy, one rollout, one
scalar fitness.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutOptions` - - Optional rollout controls.

Returns: Fitness score (higher is better).

### evaluateFlappyFitnessAcrossSeeds

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, sharedSeeds: readonly number[], rolloutOptions: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation`

Evaluate a network on a shared batch of deterministic seeds.

Educational note:
Shared-seed evaluation reduces luck. Every genome in the same comparison set
sees the same rollout seeds, which makes the aggregate statistics much more
useful for selection than a single lucky episode.

Parameters:
- `network` - - Genome/network to evaluate.
- `sharedSeeds` - - Shared deterministic seeds used for all genomes.
- `rolloutOptions` - - Optional rollout controls.

Returns: Robust aggregate metrics for selection/ranking.
