/**
 * Shared constants for the Neatenstein display worker and eval worker.
 *
 * Extracted from inline magic numbers scattered across `display.worker.ts`,
 * `eval.worker.ts`, and the `display.worker.*.utils.ts` files. This file is
 * the single source of truth for all worker-layer constants.
 *
 * Shared constants from `browser-entry/constants.ts` are re-exported here so
 * worker files can import everything from one module.
 *
 * @module
 */

import {
  WORKER_MSG_INIT,
  WORKER_MSG_RESIZE,
  WORKER_MSG_SIM_STATE,
  WORKER_MSG_INITIALIZED,
  WORKER_MSG_FRAME,
  EVAL_MSG_EVALUATE,
  EVAL_MSG_EVAL_COMPLETE,
  SNAPSHOT_KIND_MLP,
  SNAPSHOT_KIND_SWARM,
  TICK_INPUT_SOURCE_AUTO,
  TICK_INPUT_SOURCE_HUMAN,
  NEATENSTEIN_CANVAS_2D_CONTEXT,
  RENDER_TIER_WORKER,
  RENDER_TIER_CPU,
  RENDER_TIER_GPU,
} from '../constants';

// Re-export shared constants for worker-layer convenience.
export {
  WORKER_MSG_INIT,
  WORKER_MSG_RESIZE,
  WORKER_MSG_SIM_STATE,
  WORKER_MSG_INITIALIZED,
  WORKER_MSG_FRAME,
  EVAL_MSG_EVALUATE,
  EVAL_MSG_EVAL_COMPLETE,
  SNAPSHOT_KIND_MLP,
  SNAPSHOT_KIND_SWARM,
  TICK_INPUT_SOURCE_AUTO,
  TICK_INPUT_SOURCE_HUMAN,
  NEATENSTEIN_CANVAS_2D_CONTEXT,
  RENDER_TIER_WORKER,
  RENDER_TIER_CPU,
  RENDER_TIER_GPU,
};

/**
 * Maximum network node count for NEAT evolution.
 *
 * Duplicated in both `display.worker.ts` and `eval.worker.ts` before
 * extraction — this is now the single source.
 */
export const MAX_NODES = 64;

/**
 * Maximum network connection count for NEAT evolution.
 *
 * Duplicated in both `display.worker.ts` and `eval.worker.ts` before
 * extraction — this is now the single source.
 */
export const MAX_CONNECTIONS = 256;

/** Population size for the hoisted NEAT evaluation. */
export const NEAT_POPSIZE = 4;

/** Vertical feather height in pixels for fog-wall top/bottom gradient edges. */
export const FOG_FEATHER_PX = 6;

/** Alpha value for the ambient pulse glow shadow colour. */
export const PULSE_GLOW_ALPHA = 0.42;

/** Golden-angle rotation in degrees for distributing enemy team hues. */
export const GOLDEN_ANGLE_DEG = 137.508;

/** Full hue wheel range in degrees (used as the modulo base for HSL hue). */
export const COLOR_HSL_HUE = 360;

/** HSL saturation for enemy team colour resolution. */
export const COLOR_HSL_SAT = 0.65;

/** HSL lightness for enemy team colour resolution. */
export const COLOR_HSL_LIGHT = 0.5;

/** HSL hue segment size in degrees (each segment spans 60° of the hue wheel). */
export const COLOR_HSL_ALPHA = 60;

/** Fire cooldown in ticks for the fallback auto-mode AI. */
export const FALLBACK_FIRE_RANGE = 25;

/** Half-angle of the forward firing arc for the fallback auto-mode AI. */
export const FALLBACK_FIRE_ANGLE = Math.PI / 6;