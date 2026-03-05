import {
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
} from '../constants/constants';

/** Default curriculum difficulty scale used by environment stepping. */
export const FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE = 1;

/** Default number of control/physics substeps executed per simulation frame. */
export const FLAPPY_ENVIRONMENT_DEFAULT_CONTROL_SUBSTEPS_PER_FRAME =
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME;

/** Maximum frame budget before the environment forces timeout termination. */
export const FLAPPY_ENVIRONMENT_MAX_FRAMES_PER_EPISODE =
  FLAPPY_MAX_FRAMES_PER_EPISODE;
