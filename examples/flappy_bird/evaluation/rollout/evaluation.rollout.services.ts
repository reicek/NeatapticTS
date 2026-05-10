/**
 * Rollout runtime services.
 *
 * This file owns the mechanics of running an episode once a caller has decided
 * to do a rollout: normalize the options, create the seeded runtime, loop over
 * frames, and stop early when continued simulation is no longer informative.
 *
 * The companion utils file owns reward shaping and result composition. This file
 * owns the episode heartbeat itself.
 *
 * Minimal usage sketch:
 * ```ts
 * const rolloutEpisodeContext = resolveRolloutEpisodeContext(network, {
 *   seed: 123,
 *   enableEarlyTermination: true,
 * });
 * const rolloutEpisodeRuntimeState = createRolloutEpisodeRuntimeState(
 *   rolloutEpisodeContext,
 * );
 * runRolloutEpisodeLoop(
 *   network,
 *   rolloutEpisodeContext,
 *   rolloutEpisodeRuntimeState,
 * );
 * finalizeRolloutEpisodeState(
 *   rolloutEpisodeContext,
 *   rolloutEpisodeRuntimeState,
 * );
 * ```
 */
import {
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
} from '../../constants/constants';
import {
  FLAPPY_EVALUATION_DEFAULT_DIFFICULTY_SCALE,
  FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
  FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_GRACE_FRAMES,
} from '../evaluation.constants';
import {
  createInitialFlappyState,
  getFlappyObservationFeatures,
  stepFlappyStateWithControlSubsteps,
  stepFlappyStateWithControlSubstepsAsync,
  type FlappyObservationFeatures,
} from '../../flappyEnvironment.ts';
import {
  clampValue,
  commitSharedObservationMemoryStep,
  createSharedObservationMemoryState,
  resolveFlapDecision,
  resolveTemporalObservationVector,
} from '../../flappy.simulation.shared.utils';
import { createXorshift32 } from '../../rng';
import { mixGenomeEvaluationSeed } from '../evaluation.seed.utils';
import {
  FLAPPY_ROLLOUT_DEFAULT_GENOME_ID,
  FLAPPY_ROLLOUT_DONE_REASON_COLLISION,
  FLAPPY_ROLLOUT_DONE_REASON_TIMEOUT,
  FLAPPY_ROLLOUT_MIN_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
  FLAPPY_ROLLOUT_MIN_EARLY_TERMINATION_GRACE_FRAMES,
  FLAPPY_ROLLOUT_MIN_MAX_FRAMES,
  FLAPPY_ROLLOUT_ZERO_FITNESS,
} from './evaluation.rollout.constants';
import type {
  RolloutEpisodeContext,
  RolloutEpisodeRuntimeState,
} from './evaluation.rollout.types';
import type {
  FlappyNetworkLike,
  FlappyRolloutOptions,
} from '../evaluation.types';
import {
  computeDenseShapingReward,
  isBirdLikelyUnrecoverable,
} from './evaluation.rollout.utils';

/**
 * Resolves normalized rollout configuration from user options.
 *
 * This is the rollout safety boundary: caller-provided values are clamped into
 * deterministic, execution-safe ranges before the main loop touches them.
 *
 * @param network - Genome/network to evaluate.
 * @param rolloutOptions - Optional rollout controls.
 * @returns Normalized rollout configuration.
 */
export function resolveRolloutEpisodeContext(
  network: Pick<FlappyNetworkLike, '_id'>,
  rolloutOptions: FlappyRolloutOptions,
): RolloutEpisodeContext {
  // Step 1: Resolve the deterministic evaluation seed from the explicit option or genome id.
  const genomeId = network._id ?? FLAPPY_ROLLOUT_DEFAULT_GENOME_ID;
  const seed = rolloutOptions.seed ?? mixGenomeEvaluationSeed(genomeId);

  // Step 2: Normalize the rollout scalar options into safe ranges.
  return {
    seed,
    difficultyScale: clampValue(
      rolloutOptions.difficultyScale ??
        FLAPPY_EVALUATION_DEFAULT_DIFFICULTY_SCALE,
      FLAPPY_ROLLOUT_ZERO_FITNESS,
      FLAPPY_ROLLOUT_MIN_MAX_FRAMES,
    ),
    maxFramesPerEpisode: Math.max(
      FLAPPY_ROLLOUT_MIN_MAX_FRAMES,
      Math.min(
        FLAPPY_MAX_FRAMES_PER_EPISODE,
        Math.trunc(rolloutOptions.maxFrames ?? FLAPPY_MAX_FRAMES_PER_EPISODE),
      ),
    ),
    earlyTerminationGraceFrames: Math.max(
      FLAPPY_ROLLOUT_MIN_EARLY_TERMINATION_GRACE_FRAMES,
      Math.trunc(
        rolloutOptions.earlyTerminationGraceFrames ??
          FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_GRACE_FRAMES,
      ),
    ),
    earlyTerminationConsecutiveFrames: Math.max(
      FLAPPY_ROLLOUT_MIN_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
      Math.trunc(
        rolloutOptions.earlyTerminationConsecutiveFrames ??
          FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
      ),
    ),
    enableEarlyTermination: rolloutOptions.enableEarlyTermination === true,
    normalizeFitness: rolloutOptions.normalizeFitness === true,
    pipeProgressTarget: rolloutOptions.pipeProgressTarget,
  };
}

/**
 * Creates mutable runtime state for one rollout episode.
 *
 * The runtime state carries the seeded RNG, the mutable environment, the
 * shared observation-memory compatibility state, and the shaping counters
 * accumulated during the episode.
 *
 * @param rolloutEpisodeContext - Normalized rollout configuration.
 * @returns Mutable runtime state.
 */
export function createRolloutEpisodeRuntimeState(
  rolloutEpisodeContext: RolloutEpisodeContext,
): RolloutEpisodeRuntimeState {
  // Step 1: Create the seeded RNG and initial game state.
  const rng = createXorshift32(rolloutEpisodeContext.seed);
  const state = createInitialFlappyState(rng);

  // Step 2: Create the shared compatibility memory state carried across runtimes.
  return {
    rng,
    state,
    observationMemoryState: createSharedObservationMemoryState(),
    denseShapingFitness: FLAPPY_ROLLOUT_ZERO_FITNESS,
    unrecoverableFrameCount: FLAPPY_ROLLOUT_ZERO_FITNESS,
  };
}

/**
 * Runs the main rollout loop until termination or frame-budget exhaustion.
 *
 * This is the episode heartbeat: keep stepping while the bird is alive and the
 * rollout still has budget left.
 *
 * @param network - Genome/network to evaluate.
 * @param rolloutEpisodeContext - Normalized rollout configuration.
 * @param rolloutEpisodeRuntimeState - Mutable runtime state.
 * @returns Nothing.
 */
export function runRolloutEpisodeLoop(
  network: FlappyNetworkLike,
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
): void {
  // Step 1: Continue stepping while the episode remains active and within the frame cap.
  while (
    !rolloutEpisodeRuntimeState.state.done &&
    rolloutEpisodeRuntimeState.state.frameIndex <
      rolloutEpisodeContext.maxFramesPerEpisode
  ) {
    runRolloutEpisodeFrame(
      network,
      rolloutEpisodeContext,
      rolloutEpisodeRuntimeState,
    );
  }
}

/**
 * Runs the main rollout loop against one async predictor callback.
 *
 * This keeps the rollout semantics aligned with the synchronous evaluation
 * surface while allowing the control decision itself to come from a persistent
 * worker-hosted predictor.
 *
 * @param predictOutputs - Async predictor callback for one observation vector.
 * @param rolloutEpisodeContext - Normalized rollout configuration.
 * @param rolloutEpisodeRuntimeState - Mutable runtime state.
 * @returns Nothing.
 */
export async function runRolloutEpisodeLoopWithPredictor(
  predictOutputs: (observationVector: number[]) => Promise<unknown>,
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
): Promise<void> {
  // Step 1: Continue stepping while the episode remains active and within the frame cap.
  while (
    !rolloutEpisodeRuntimeState.state.done &&
    rolloutEpisodeRuntimeState.state.frameIndex <
      rolloutEpisodeContext.maxFramesPerEpisode
  ) {
    await runRolloutEpisodeFrameWithPredictor(
      predictOutputs,
      rolloutEpisodeContext,
      rolloutEpisodeRuntimeState,
    );
  }
}

/**
 * Finalizes episode state after the main rollout loop exits.
 *
 * Timeouts are applied here instead of inside the loop body so natural episode
 * endings stay distinct from budget exhaustion.
 *
 * @param rolloutEpisodeContext - Normalized rollout configuration.
 * @param rolloutEpisodeRuntimeState - Mutable runtime state.
 * @returns Nothing.
 */
export function finalizeRolloutEpisodeState(
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
): void {
  // Step 1: Leave the state unchanged when the episode already finished naturally.
  if (
    rolloutEpisodeRuntimeState.state.done ||
    rolloutEpisodeRuntimeState.state.frameIndex <
      rolloutEpisodeContext.maxFramesPerEpisode
  ) {
    return;
  }

  // Step 2: Mark a clean timeout when the frame budget was exhausted.
  rolloutEpisodeRuntimeState.state.done = true;
  rolloutEpisodeRuntimeState.state.doneReason =
    FLAPPY_ROLLOUT_DONE_REASON_TIMEOUT;
}

/**
 * Runs one rollout frame including control, shaping, and early termination.
 *
 * Educational note:
 * Each frame follows a compact pipeline: observe, act, step the environment,
 * accumulate shaping reward, then optionally prune the trajectory.
 *
 * @param network - Genome/network to evaluate.
 * @param rolloutEpisodeContext - Normalized rollout configuration.
 * @param rolloutEpisodeRuntimeState - Mutable runtime state.
 * @returns Nothing.
 */
function runRolloutEpisodeFrame(
  network: FlappyNetworkLike,
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
): void {
  // Step 1: Capture the pre-step observation used by dense shaping.
  const previousObservationFeatures = getFlappyObservationFeatures(
    rolloutEpisodeRuntimeState.state,
    rolloutEpisodeContext.difficultyScale,
  );

  // Step 2: Advance the environment using network-driven flap control.
  stepFlappyStateWithControlSubsteps(
    rolloutEpisodeRuntimeState.state,
    rolloutEpisodeRuntimeState.rng,
    () =>
      resolveRolloutFrameFlapDecision(
        network,
        rolloutEpisodeContext,
        rolloutEpisodeRuntimeState,
      ),
    rolloutEpisodeContext.difficultyScale,
    FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  );

  // Step 3: Update dense shaping from the pre-step and post-step observations.
  const currentObservationFeatures = getFlappyObservationFeatures(
    rolloutEpisodeRuntimeState.state,
    rolloutEpisodeContext.difficultyScale,
  );
  rolloutEpisodeRuntimeState.denseShapingFitness += computeDenseShapingReward(
    previousObservationFeatures,
    currentObservationFeatures,
  );

  // Step 4: Apply the optional early-termination heuristic when enabled.
  applyRolloutEarlyTerminationIfNeeded(
    rolloutEpisodeContext,
    rolloutEpisodeRuntimeState,
    currentObservationFeatures,
  );
}

async function runRolloutEpisodeFrameWithPredictor(
  predictOutputs: (observationVector: number[]) => Promise<unknown>,
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
): Promise<void> {
  // Step 1: Capture the pre-step observation used by dense shaping.
  const previousObservationFeatures = getFlappyObservationFeatures(
    rolloutEpisodeRuntimeState.state,
    rolloutEpisodeContext.difficultyScale,
  );

  // Step 2: Advance the environment using predictor-driven flap control.
  await stepFlappyStateWithControlSubstepsAsync(
    rolloutEpisodeRuntimeState.state,
    rolloutEpisodeRuntimeState.rng,
    () =>
      resolveRolloutFrameFlapDecisionWithPredictor(
        predictOutputs,
        rolloutEpisodeContext,
        rolloutEpisodeRuntimeState,
      ),
    rolloutEpisodeContext.difficultyScale,
    FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  );

  // Step 3: Update dense shaping from the pre-step and post-step observations.
  const currentObservationFeatures = getFlappyObservationFeatures(
    rolloutEpisodeRuntimeState.state,
    rolloutEpisodeContext.difficultyScale,
  );
  rolloutEpisodeRuntimeState.denseShapingFitness += computeDenseShapingReward(
    previousObservationFeatures,
    currentObservationFeatures,
  );

  // Step 4: Apply the optional early-termination heuristic when enabled.
  applyRolloutEarlyTerminationIfNeeded(
    rolloutEpisodeContext,
    rolloutEpisodeRuntimeState,
    currentObservationFeatures,
  );
}

/**
 * Resolves the flap decision for one control substep and commits memory state.
 *
 * The shared memory surface is updated at the same post-decision boundary used
 * by browser playback and worker simulation. The current controller input still
 * reads only the current normalized frame, but keeping the bookkeeping point
 * stable avoids runtime drift if an opt-in external-history experiment returns.
 *
 * @param network - Genome/network to evaluate.
 * @param rolloutEpisodeContext - Normalized rollout configuration.
 * @param rolloutEpisodeRuntimeState - Mutable runtime state.
 * @returns Whether the bird should flap.
 */
function resolveRolloutFrameFlapDecision(
  network: FlappyNetworkLike,
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
): boolean {
  // Step 1: Resolve the current observation features from the mutable game state.
  const observationFeatures = getFlappyObservationFeatures(
    rolloutEpisodeRuntimeState.state,
    rolloutEpisodeContext.difficultyScale,
  );

  // Step 2: Build the current-frame controller input and query the network outputs.
  const observation = resolveTemporalObservationVector(
    observationFeatures,
    rolloutEpisodeRuntimeState.observationMemoryState,
  );
  const outputs = network.activate(observation);
  const shouldFlap = resolveFlapDecision(outputs);

  // Step 3: Commit the observation and decision to the shared compatibility state.
  commitSharedObservationMemoryStep(
    rolloutEpisodeRuntimeState.observationMemoryState,
    observationFeatures,
    shouldFlap,
  );
  return shouldFlap;
}

async function resolveRolloutFrameFlapDecisionWithPredictor(
  predictOutputs: (observationVector: number[]) => Promise<unknown>,
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
): Promise<boolean> {
  // Step 1: Resolve the current observation features from the mutable game state.
  const observationFeatures = getFlappyObservationFeatures(
    rolloutEpisodeRuntimeState.state,
    rolloutEpisodeContext.difficultyScale,
  );

  // Step 2: Build the current-frame controller input and query the async predictor.
  const observation = resolveTemporalObservationVector(
    observationFeatures,
    rolloutEpisodeRuntimeState.observationMemoryState,
  );
  const outputs = await predictOutputs(observation);
  const shouldFlap = resolveFlapDecision(outputs);

  // Step 3: Commit the observation and decision to the shared compatibility state.
  commitSharedObservationMemoryStep(
    rolloutEpisodeRuntimeState.observationMemoryState,
    observationFeatures,
    shouldFlap,
  );
  return shouldFlap;
}

/**
 * Applies the optional early-termination heuristic for unrecoverable starts.
 *
 * Educational note:
 * Early termination is an evaluation-speed heuristic, not a gameplay rule. It
 * exists to stop obviously doomed warmup trajectories from consuming excessive
 * rollout budget.
 *
 * @param rolloutEpisodeContext - Normalized rollout configuration.
 * @param rolloutEpisodeRuntimeState - Mutable runtime state.
 * @param currentObservationFeatures - Post-step observation features.
 * @returns Nothing.
 */
function applyRolloutEarlyTerminationIfNeeded(
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
  currentObservationFeatures: FlappyObservationFeatures,
): void {
  // Step 1: Exit immediately when early termination is disabled.
  if (!rolloutEpisodeContext.enableEarlyTermination) {
    return;
  }

  // Step 2: Track consecutive unrecoverable frames only during the warmup phase.
  const earlyTerminationEligible =
    rolloutEpisodeRuntimeState.state.pipesPassed ===
      FLAPPY_ROLLOUT_ZERO_FITNESS &&
    rolloutEpisodeRuntimeState.state.frameIndex >=
      rolloutEpisodeContext.earlyTerminationGraceFrames;
  const birdLikelyUnrecoverable = isBirdLikelyUnrecoverable(
    currentObservationFeatures,
  );
  rolloutEpisodeRuntimeState.unrecoverableFrameCount =
    earlyTerminationEligible && birdLikelyUnrecoverable
      ? rolloutEpisodeRuntimeState.unrecoverableFrameCount + 1
      : FLAPPY_ROLLOUT_ZERO_FITNESS;

  // Step 3: Stop the episode once the unrecoverable streak reaches the configured threshold.
  if (
    rolloutEpisodeRuntimeState.unrecoverableFrameCount <
    rolloutEpisodeContext.earlyTerminationConsecutiveFrames
  ) {
    return;
  }

  rolloutEpisodeRuntimeState.state.done = true;
  rolloutEpisodeRuntimeState.state.doneReason =
    FLAPPY_ROLLOUT_DONE_REASON_COLLISION;
}
