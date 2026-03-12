import {
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
} from '../constants/constants';

/**
 * Default curriculum difficulty scale used by environment stepping.
 *
 * A value of `1` means the environment uses the full adaptive difficulty ramp.
 */
export const FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE = 1;

/**
 * Default number of control/physics substeps executed per simulation frame.
 *
 * Reusing the shared control-substep count keeps the environment and browser
 * playback aligned on the same stepping granularity.
 */
export const FLAPPY_ENVIRONMENT_DEFAULT_CONTROL_SUBSTEPS_PER_FRAME =
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME;

/**
 * Maximum frame budget before the environment forces timeout termination.
 *
 * Timeouts stop extremely long survival loops from dominating evaluation cost.
 */
export const FLAPPY_ENVIRONMENT_MAX_FRAMES_PER_EPISODE =
  FLAPPY_MAX_FRAMES_PER_EPISODE;
