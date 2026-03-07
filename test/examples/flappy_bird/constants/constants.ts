/**
 * Legacy Flappy constants barrel.
 *
 * Constants are now organized into small, category-focused modules
 * (`constants.*.ts`). This file remains as a compatibility export surface so
 * existing imports continue to work while callers migrate gradually.
 */

export * from './constants.world';
export * from './constants.physics';
export * from './constants.pipes';
export * from './constants.network';
export * from './constants.fitness';
export * from './constants.difficulty';
export * from './constants.rendering';
export * from './constants.observation';

export {
  DEFAULT_CONTAINER_ID,
  FLAPPY_EMULATION_SPEED_MULTIPLIER,
  FLAPPY_HUD_UPDATE_INTERVAL_FRAMES,
  FLAPPY_BROWSER_POPULATION_SIZE,
  FLAPPY_BROWSER_ELITISM_COUNT,
  FLAPPY_DEFAULT_RNG_SEED,
  FLAPPY_FLAP_THRESHOLD,
  FLAPPY_HALF,
  FLAPPY_HUD_ZERO_TEXT,
  FLAPPY_HUD_ZERO_DECIMAL_TEXT,
  FLAPPY_HUD_OFF_TEXT,
  FLAPPY_HUD_INITIALIZING_TEXT,
  FLAPPY_STATUS_PLAYING_TEXT,
  FLAPPY_STATUS_EVOLVING_TEXT,
  FLAPPY_HUD_UPDATES_WINDOW_MS,
  FLAPPY_HUD_UPDATES_WINDOW_SECONDS,
  FLAPPY_MINOR_GC_WINDOW_MS,
} from './constants.runtime';
export * from './constants.frame';
export * from './constants.palette';
export * from './constants.layout';
export * from './constants.stats';
export * from './constants.network-view';
export * from './constants.birds';
export {
  FLAPPY_PIPE_OUTLINE_GLOW_ALPHA,
  FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX,
} from './constants.pipe-render';
export * from './constants.starfield';
