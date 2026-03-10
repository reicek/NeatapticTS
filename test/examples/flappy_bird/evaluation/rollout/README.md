# evaluation/rollout

## evaluation/rollout/evaluation.rollout.types.ts

### evaluation.rollout.types

Rollout-internal type contracts.

This file will host runtime-only rollout types that should not widen the
public evaluation-level API surface.

### DenseShapingRewardComponents

Per-frame dense shaping channels resolved from consecutive observations.

### RolloutEpisodeContext

Immutable rollout options normalized into execution-safe ranges.

### RolloutEpisodeRuntimeState

Mutable runtime state accumulated while one rollout episode executes.

### RolloutFitnessBreakdown

Fitness-channel breakdown used to compose the public episode result.

## evaluation/rollout/evaluation.rollout.service.ts

### evaluation.rollout.service

Rollout orchestration module.

This file will host the internal rollout orchestration entry while the
public evaluation-level service remains a stable compatibility facade.

### rolloutEpisode

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutOptions: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyEpisodeResult`

Roll out an episode and return details.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutOptions` - - Optional rollout controls.

Returns: Episode result details.

## evaluation/rollout/evaluation.rollout.services.ts

### evaluation.rollout.services

Rollout runtime services.

This file will host context resolution, runtime initialization, frame loop,
and early-termination behavior for rollout execution.

### applyRolloutEarlyTerminationIfNeeded

`(rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState, currentObservationFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => void`

Applies the optional early-termination heuristic for unrecoverable starts.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.
- `currentObservationFeatures` - - Post-step observation features.

Returns: Nothing.

### createRolloutEpisodeRuntimeState

`(rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext) => import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState`

Creates mutable runtime state for one rollout episode.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.

Returns: Mutable runtime state.

### finalizeRolloutEpisodeState

`(rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState) => void`

Finalizes episode state after the main rollout loop exits.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.

Returns: Nothing.

### resolveRolloutEpisodeContext

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutOptions: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext`

Resolves normalized rollout configuration from user options.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutOptions` - - Optional rollout controls.

Returns: Normalized rollout configuration.

### resolveRolloutFrameFlapDecision

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState) => boolean`

Resolves the flap decision for one control substep and commits memory state.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.

Returns: Whether the bird should flap.

### runRolloutEpisodeFrame

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState) => void`

Runs one rollout frame including control, shaping, and early termination.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.

Returns: Nothing.

### runRolloutEpisodeLoop

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState) => void`

Runs the main rollout loop until termination or frame-budget exhaustion.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.

Returns: Nothing.

## evaluation/rollout/evaluation.rollout.constants.ts

### evaluation.rollout.constants

Rollout-local constants.

This file will host rollout-only constants and sentinels that belong to the
rollout subsystem rather than the wider evaluation surface.

### FLAPPY_ROLLOUT_DEFAULT_GENOME_ID

### FLAPPY_ROLLOUT_DONE_REASON_COLLISION

### FLAPPY_ROLLOUT_DONE_REASON_TIMEOUT

### FLAPPY_ROLLOUT_MIN_EARLY_TERMINATION_CONSECUTIVE_FRAMES

### FLAPPY_ROLLOUT_MIN_EARLY_TERMINATION_GRACE_FRAMES

### FLAPPY_ROLLOUT_MIN_MAX_FRAMES

### FLAPPY_ROLLOUT_ZERO_FITNESS

## evaluation/rollout/evaluation.rollout.utils.ts

### evaluation.rollout.utils

Rollout shaping and result helpers.

This file will host rollout-local fitness composition, shaping utilities,
and terminal result assembly helpers.

### composeNormalizedFitness

`(framesValue: number, pipesPassedValue: number, denseShapingValue: number, terminalShapingValue: number, maxFramesValue: number, pipeProgressTarget: number | undefined) => number`

Normalize and cap fitness channels so no single reward term dominates.

Parameters:
- `framesValue` - - Frames survived for the episode.
- `pipesPassedValue` - - Pipes passed during the episode.
- `denseShapingValue` - - Accumulated dense shaping reward.
- `terminalShapingValue` - - Terminal shaping reward.
- `maxFramesValue` - - Frame budget used for the episode.
- `pipeProgressTarget` - - Optional target used to normalize pipe progress.

Returns: Normalized composite fitness.

### composeRolloutEpisodeResult

`(rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState) => import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyEpisodeResult`

Composes the final rollout result from the terminal game state.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.

Returns: Episode result details.

### computeDenseShapingReward

`(previousFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures, currentFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => number`

Computes dense reward shaping from consecutive observations.

Parameters:
- `previousFeatures` - - Observation before stepping the environment.
- `currentFeatures` - - Observation after stepping the environment.

Returns: Per-step shaped reward.

### computeTerminalShapingFitness

`(episodeState: import("test/examples/flappy_bird/environment/environment.types").FlappyGameState, difficultyScale: number) => number`

Adds small terminal bonuses from final progress/alignment signals.

Parameters:
- `episodeState` - - Final rollout state.
- `difficultyScale` - - Active rollout difficulty scale.

Returns: Terminal shaping reward.

### isBirdLikelyUnrecoverable

`(observationFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => boolean`

Detects trajectories that are usually irrecoverable in early warmup.

Parameters:
- `observationFeatures` - - Post-step observation features.

Returns: Whether the current trajectory appears unrecoverable.

### resolveDenseShapingRewardComponents

`(previousFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures, currentFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").DenseShapingRewardComponents`

Resolves every dense-shaping reward component from consecutive observations.

Parameters:
- `previousFeatures` - - Observation before stepping the environment.
- `currentFeatures` - - Observation after stepping the environment.

Returns: Dense-shaping reward components.

### resolveRolloutFitnessBreakdown

`(rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState, framesSurvived: number, pipesPassed: number) => import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutFitnessBreakdown`

Resolves the raw fitness channels from the final episode state.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.
- `framesSurvived` - - Final frame count.
- `pipesPassed` - - Final pipe-pass count.

Returns: Fitness-channel breakdown.

### resolveUnnormalizedRolloutFitness

`(rolloutFitnessBreakdown: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutFitnessBreakdown) => number`

Resolves raw fitness by summing every fitness channel.

Parameters:
- `rolloutFitnessBreakdown` - - Fitness-channel breakdown.

Returns: Raw unnormalized fitness.
