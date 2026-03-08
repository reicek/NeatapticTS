# evaluation

## evaluation/evaluation.types.ts

### FlappyEpisodeResult

Summary metrics for a single Flappy episode rollout.

### FlappyNetworkLike

Minimal network contract required by Flappy evaluation.

### FlappyRolloutOptions

Runtime controls for one rollout evaluation.

### FlappySeedBatchEvaluation

Aggregate statistics from evaluating one network across shared seeds.

## evaluation/evaluation.constants.ts

### evaluation.constants

Default difficulty scale for rollouts when caller does not provide one.

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

### rolloutEpisode

`(network: import("C:/NeatapticTS/test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutOptions: import("C:/NeatapticTS/test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => import("C:/NeatapticTS/test/examples/flappy_bird/evaluation/evaluation.types").FlappyEpisodeResult`

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

Parameters:

- `genomeId` - - Genome id from NEAT bookkeeping.

Returns: Mixed uint32 seed.

## evaluation/evaluation.fitness.utils.ts

### evaluateFlappyFitness

`(network: import("C:/NeatapticTS/test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutOptions: import("C:/NeatapticTS/test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => number`

Evaluate a network on a single deterministic Flappy Bird episode.

Parameters:

- `network` - - Genome/network to evaluate.
- `rolloutOptions` - - Optional rollout controls.

Returns: Fitness score (higher is better).

### evaluateFlappyFitnessAcrossSeeds

`(network: import("C:/NeatapticTS/test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, sharedSeeds: readonly number[], rolloutOptions: import("C:/NeatapticTS/test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => import("C:/NeatapticTS/test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation`

Evaluate a network on a shared batch of deterministic seeds.

Parameters:

- `network` - - Genome/network to evaluate.
- `sharedSeeds` - - Shared deterministic seeds used for all genomes.
- `rolloutOptions` - - Optional rollout controls.

Returns: Robust aggregate metrics for selection/ranking.
