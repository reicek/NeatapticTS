/**
 * Legacy Flappy constants barrel.
 *
 * Constants are now organized into small, category-focused modules
 * (`constants.*.ts`). This file remains as a compatibility export surface so
 * existing imports continue to work while callers migrate gradually.
 *
 * Educational note:
 * Treat this file as the folder map, not the best place to memorize individual
 * values. The category modules below are where the real stories live: world
 * geometry, physics, pipes, observation, runtime defaults, rendering chrome,
 * network visualization, and browser HUD behavior.
 *
 * A useful way to read this folder is to ask one question first: "what kind of
 * knob am I trying to change?"
 *
 * - If the bird feels wrong, start with physics.
 * - If the course feels unfair or too easy, start with pipes and difficulty.
 * - If the policy sees the wrong world, start with observation.
 * - If the browser demo feels noisy or cramped, start with runtime, layout, and stats.
 * - If the network panel is hard to read, start with network-view and palette.
 *
 * Quick import example:
 * ```ts
 * import {
 *   FLAPPY_GRAVITY_PX_PER_FRAME2,
 *   FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
 * } from './constants/constants';
 * ```
 *
 * Constant-family map:
 * ```mermaid
 * flowchart TB
 *     Constants["constants.ts"] --> World["world\ncourse geometry"]
 *     Constants --> Physics["physics\ncontrol and motion"]
 *     Constants --> Pipes["pipes + difficulty\nspawn cadence and spacing"]
 *     Constants --> Observation["observation\nfeature scaling"]
 *     Constants --> Runtime["runtime + stats\nbrowser defaults and HUD"]
 *     Constants --> Rendering["frame + layout + birds + starfield\nvisual shell"]
 *     Constants --> NetworkView["network-view + palette\ninspection and legend"]
 * ```
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
  FLAPPY_STARTUP_PREVIEW_LEGEND_TEXT,
  FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS,
  FLAPPY_GENERATION_PREVIEW_HOLD_DURATION_MS,
  FLAPPY_STARTUP_PREVIEW_FRAME_DURATION_MS,
  FLAPPY_STARTUP_PREVIEW_LEGEND_FONT_WEIGHT,
  FLAPPY_STARTUP_PREVIEW_LEGEND_FONT_SIZE_RATIO,
  FLAPPY_STARTUP_PREVIEW_LEGEND_MIN_FONT_SIZE_PX,
  FLAPPY_STARTUP_PREVIEW_LEGEND_MAX_FONT_SIZE_PX,
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
  FLAPPY_PIPE_ENTRY_RIM_INSET_PX,
  FLAPPY_PIPE_OUTLINE_GLOW_ALPHA,
  FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX,
} from './constants.pipe-render';
export * from './constants.starfield';
