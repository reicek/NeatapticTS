/**
 * Rollout-local constants.
 *
 * This file will host rollout-only constants and sentinels that belong to the
 * rollout subsystem rather than the wider evaluation surface.
 */
import type { FlappyGameState } from '../../flappyEnvironment.ts';

/** Default genome id used when a network does not expose one. */
export const FLAPPY_ROLLOUT_DEFAULT_GENOME_ID = 0;

/** Minimum positive frame-like scalar used by rollout normalization. */
export const FLAPPY_ROLLOUT_MIN_MAX_FRAMES = 1;

/** Minimum grace period allowed before early termination can activate. */
export const FLAPPY_ROLLOUT_MIN_EARLY_TERMINATION_GRACE_FRAMES = 0;

/** Minimum unrecoverable-frame streak required for early termination. */
export const FLAPPY_ROLLOUT_MIN_EARLY_TERMINATION_CONSECUTIVE_FRAMES = 1;

/** Shared zero baseline used across rollout fitness and counters. */
export const FLAPPY_ROLLOUT_ZERO_FITNESS = 0;

/** Rollout done reason used by heuristic early termination. */
export const FLAPPY_ROLLOUT_DONE_REASON_COLLISION: FlappyGameState['doneReason'] =
  'collision';

/** Rollout done reason used when the episode exhausts its frame budget. */
export const FLAPPY_ROLLOUT_DONE_REASON_TIMEOUT: FlappyGameState['doneReason'] =
  'timeout';
