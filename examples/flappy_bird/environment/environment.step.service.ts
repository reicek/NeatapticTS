import {
  FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
  FLAPPY_GRAVITY_PX_PER_FRAME2,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_WIDTH_PX,
} from '../constants/constants';
import type { FlappyRng } from '../rng';
import {
  clampValue,
  resolveAdaptiveDifficultyProfile,
  resolveNextSpawnGapCenterY,
  resolveNextSpawnGapSize,
  resolveNextSpawnIntervalFrames,
} from '../flappy.simulation.shared.utils';
import {
  FLAPPY_ENVIRONMENT_DEFAULT_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE,
  FLAPPY_ENVIRONMENT_MAX_FRAMES_PER_EPISODE,
} from './environment.constants';
import { updateCollisionAndProgressState } from './environment.collision.utils';
import type {
  FlappyDifficultyScale,
  FlappyGameState,
} from './environment.types';

/**
 * Advance the simulation by one frame.
 *
 * This is the simplest stepping surface: one logical frame and one flap choice.
 * More advanced callers can use the control-substep variant below.
 *
 * @param state - Mutable state object to update in-place.
 * @param rng - Random source used to spawn pipes.
 * @param flap - If true, applies an upward velocity impulse.
 * @param difficultyScale - Curriculum difficulty scale in [0, 1].
 * @returns Nothing.
 */
export function stepFlappyState(
  state: FlappyGameState,
  rng: FlappyRng,
  flap: boolean,
  difficultyScale: FlappyDifficultyScale = FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE,
): void {
  stepFlappyStateWithControlSubsteps(
    state,
    rng,
    () => flap,
    difficultyScale,
    1,
  );
}

/**
 * Advance one logical frame using multiple control/physics substeps.
 *
 * This allows policies to react multiple times before `frameIndex` advances,
 * improving responsiveness in high-difficulty scenarios.
 *
 * Educational note:
 * Splitting a logical frame into smaller control steps is a simple numerical
 * stability trick. It reduces the chance that fast pipes or large velocity
 * updates make the environment feel artificially coarse.
 *
 * For background reading, the Wikipedia article on "numerical integration"
 * provides the general idea behind updating continuous motion in small steps.
 *
 * @param state - Mutable state object to update in-place.
 * @param rng - Random source used to spawn pipes.
 * @param shouldFlapForSubstep - Callback deciding flap action per substep.
 * @param difficultyScale - Curriculum difficulty scale in [0, 1].
 * @param controlSubstepsPerFrame - Number of substeps to run this frame.
 * @returns Nothing.
 */
export function stepFlappyStateWithControlSubsteps(
  state: FlappyGameState,
  rng: FlappyRng,
  shouldFlapForSubstep: () => boolean,
  difficultyScale: FlappyDifficultyScale = FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE,
  controlSubstepsPerFrame: number = FLAPPY_ENVIRONMENT_DEFAULT_CONTROL_SUBSTEPS_PER_FRAME,
): void {
  if (state.done) return;

  const difficultyProfile = resolveAdaptiveDifficultyProfile(
    state.pipesPassed,
    difficultyScale,
  );
  const substepCount = Math.max(1, Math.trunc(controlSubstepsPerFrame));
  const substepDelta = 1 / substepCount;

  // Step 1: Run high-frequency control/physics substeps.
  for (
    let controlSubstepIndex = 0;
    controlSubstepIndex < substepCount && !state.done;
    controlSubstepIndex++
  ) {
    const flap = shouldFlapForSubstep();
    if (flap) {
      state.bird.velocityYPxPerFrame = FLAPPY_FLAP_VELOCITY_PX_PER_FRAME;
    }

    state.bird.velocityYPxPerFrame = clampValue(
      state.bird.velocityYPxPerFrame +
        FLAPPY_GRAVITY_PX_PER_FRAME2 * substepDelta,
      -Infinity,
      FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
    );
    state.bird.yPx += state.bird.velocityYPxPerFrame * substepDelta;

    for (const pipe of state.pipes) {
      pipe.xPx -= difficultyProfile.pipeSpeedPxPerFrame * substepDelta;
    }
    state.pipes = state.pipes.filter(
      (pipe) => pipe.xPx + FLAPPY_PIPE_WIDTH_PX > 0,
    );

    state.framesUntilNextPipeSpawn -= substepDelta;
    if (state.framesUntilNextPipeSpawn <= 0) {
      const nextGapSizePx = resolveNextSpawnGapSize(
        state.lastSpawnedPipeGapPx,
        difficultyProfile,
        rng,
      );
      const nextSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
        state.lastSpawnedPipeSpawnIntervalFrames,
        difficultyProfile,
      );
      const nextGapCenterYPx = resolveNextSpawnGapCenterY(
        state.lastSpawnedPipeGapCenterYPx,
        rng,
        nextGapSizePx,
      );
      state.pipes.push({
        xPx: FLAPPY_WORLD_WIDTH_PX + FLAPPY_PIPE_WIDTH_PX,
        gapCenterYPx: nextGapCenterYPx,
        gapSizePx: nextGapSizePx,
        passed: false,
      });
      state.lastSpawnedPipeGapPx = nextGapSizePx;
      state.lastSpawnedPipeGapCenterYPx = nextGapCenterYPx;
      state.lastSpawnedPipeSpawnIntervalFrames = nextSpawnIntervalFrames;
      state.framesUntilNextPipeSpawn += nextSpawnIntervalFrames;
    }

    if (!state.done) {
      updateCollisionAndProgressState(state);
    }
  }

  // Step 2: Frame accounting and timeout.
  state.frameIndex++;
  if (
    !state.done &&
    state.frameIndex >= FLAPPY_ENVIRONMENT_MAX_FRAMES_PER_EPISODE
  ) {
    state.done = true;
    state.doneReason = 'timeout';
  }
}
