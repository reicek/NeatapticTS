# evaluation/rollout

## evaluation/rollout/evaluation.rollout.types.ts

### evaluation.rollout.types

Rollout-internal type contracts.

This file will host runtime-only rollout types that should not widen the
public evaluation-level API surface.

That separation keeps the public evaluation API compact even as rollout
internals become more detailed.

### DenseShapingRewardComponents

Per-frame dense shaping channels resolved from consecutive observations.

The shaping system rewards more than survival: it also tracks approach,
centering, clearance, and stable motion.

### RolloutEpisodeContext

Immutable rollout options normalized into execution-safe ranges.

Every field here is ready for direct use inside the episode loop.

### RolloutEpisodeRuntimeState

Mutable runtime state accumulated while one rollout episode executes.

This is the mutable side of the rollout: world state, RNG, temporal memory,
and the counters accumulated during execution.

### RolloutFitnessBreakdown

Fitness-channel breakdown used to compose the public episode result.

Named channels make reward design easier to audit than a single opaque number.

## evaluation/rollout/evaluation.rollout.service.ts

### evaluation.rollout.service

Rollout orchestration module.

This file will host the internal rollout orchestration entry while the
public evaluation-level service remains a stable compatibility facade.

Educational note:
A rollout is one deterministic episode for one policy under one seed. This
module keeps that lifecycle readable: normalize inputs, create runtime state,
simulate until termination, then fold the result into a public episode report.

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

The companion utils file owns reward shaping and result composition. This file
owns the mechanics of actually running the episode.

### applyRolloutEarlyTerminationIfNeeded

`(rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState, currentObservationFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => void`

Applies the optional early-termination heuristic for unrecoverable starts.

Educational note:
Early termination is an evaluation-speed heuristic, not a gameplay rule. It
exists to stop obviously doomed warmup trajectories from consuming excessive
rollout budget.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.
- `currentObservationFeatures` - - Post-step observation features.

Returns: Nothing.

### createRolloutEpisodeRuntimeState

`(rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext) => import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState`

Creates mutable runtime state for one rollout episode.

The runtime state carries the seeded RNG, the mutable environment, the
temporal observation memory, and the shaping counters accumulated during the
episode.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.

Returns: Mutable runtime state.

### finalizeRolloutEpisodeState

`(rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState) => void`

Finalizes episode state after the main rollout loop exits.

Timeouts are applied here instead of inside the loop body so natural episode
endings stay distinct from budget exhaustion.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.

Returns: Nothing.

### resolveRolloutEpisodeContext

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutOptions: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext`

Resolves normalized rollout configuration from user options.

This is the rollout safety boundary: caller-provided values are clamped into
deterministic, execution-safe ranges before the main loop touches them.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutOptions` - - Optional rollout controls.

Returns: Normalized rollout configuration.

### resolveRolloutFrameFlapDecision

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState) => boolean`

Resolves the flap decision for one control substep and commits memory state.

The temporal memory is updated immediately after the decision so subsequent
substeps can see short-term action history without needing recurrent state.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.

Returns: Whether the bird should flap.

### runRolloutEpisodeFrame

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState) => void`

Runs one rollout frame including control, shaping, and early termination.

Educational note:
Each frame follows a compact pipeline: observe, act, step the environment,
accumulate shaping reward, then optionally prune the trajectory.

Parameters:
- `network` - - Genome/network to evaluate.
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.

Returns: Nothing.

### runRolloutEpisodeLoop

`(network: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyNetworkLike, rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState) => void`

Runs the main rollout loop until termination or frame-budget exhaustion.

This is the episode heartbeat: keep stepping while the bird is alive and the
rollout still has budget left.

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

Naming these sentinels explicitly keeps rollout code easier to read than a
sea of raw `0`, `1`, and string literals.

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

Educational note:
The rollout subsystem separates simulation from scoring on purpose. The
services file determines what happened; this file determines how that episode
should be interpreted as fitness.

### composeNormalizedFitness

`(framesValue: number, pipesPassedValue: number, denseShapingValue: number, terminalShapingValue: number, maxFramesValue: number, pipeProgressTarget: number | undefined) => number`

Normalize and cap fitness channels so no single reward term dominates.

Educational note:
Channel normalization is a pragmatic way to keep the objective balanced across
episodes of different lengths and levels of progress.

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

This is the final fold step for rollout execution: internal counters and
shaping channels become the public `FlappyEpisodeResult` consumed by training
and reporting.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.

Returns: Episode result details.

### computeDenseShapingReward

`(previousFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures, currentFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => number`

Computes dense reward shaping from consecutive observations.

Dense shaping rewards incremental improvement throughout an episode instead of
paying out only at the end, which gives evolution a more informative signal.

Parameters:
- `previousFeatures` - - Observation before stepping the environment.
- `currentFeatures` - - Observation after stepping the environment.

Returns: Per-step shaped reward.

### computeTerminalShapingFitness

`(episodeState: import("test/examples/flappy_bird/environment/environment.types").FlappyGameState, difficultyScale: number) => number`

Adds small terminal bonuses from final progress/alignment signals.

Terminal bonuses refine the final ranking, but they are intentionally smaller
than the main survival and pipe-progress channels.

Parameters:
- `episodeState` - - Final rollout state.
- `difficultyScale` - - Active rollout difficulty scale.

Returns: Terminal shaping reward.

### isBirdLikelyUnrecoverable

`(observationFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => boolean`

Detects trajectories that are usually irrecoverable in early warmup.

The heuristic focuses on obvious early failures, where spending more rollout
budget is least informative.

Parameters:
- `observationFeatures` - - Post-step observation features.

Returns: Whether the current trajectory appears unrecoverable.

### resolveDenseShapingRewardComponents

`(previousFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures, currentFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").DenseShapingRewardComponents`

Resolves every dense-shaping reward component from consecutive observations.

If you want background reading, the Wikipedia article on "reward shaping" is
a good high-level companion concept for why these components exist.

Parameters:
- `previousFeatures` - - Observation before stepping the environment.
- `currentFeatures` - - Observation after stepping the environment.

Returns: Dense-shaping reward components.

### resolveRolloutFitnessBreakdown

`(rolloutEpisodeContext: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeContext, rolloutEpisodeRuntimeState: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutEpisodeRuntimeState, framesSurvived: number, pipesPassed: number) => import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutFitnessBreakdown`

Resolves the raw fitness channels from the final episode state.

Separating raw channels from final composition makes reward rebalancing much
easier to reason about.

Parameters:
- `rolloutEpisodeContext` - - Normalized rollout configuration.
- `rolloutEpisodeRuntimeState` - - Mutable runtime state.
- `framesSurvived` - - Final frame count.
- `pipesPassed` - - Final pipe-pass count.

Returns: Fitness-channel breakdown.

### resolveUnnormalizedRolloutFitness

`(rolloutFitnessBreakdown: import("test/examples/flappy_bird/evaluation/rollout/evaluation.rollout.types").RolloutFitnessBreakdown) => number`

Resolves raw fitness by summing every fitness channel.

This is the legacy unnormalized objective. The normalized path below caps
channels so no single term dominates the whole score.

Parameters:
- `rolloutFitnessBreakdown` - - Fitness-channel breakdown.

Returns: Raw unnormalized fitness.
