/**
 * Flappy policy-network and temporal memory constants.
 *
 * This module defines observation window shape and initial architecture sizing
 * used when seeding agents for evolution.
 */

/** Number of core per-frame observation features retained for temporal stacking. */
export const FLAPPY_MEMORY_CORE_FEATURE_COUNT = 12;

/** Number of temporal frames included in the stacked observation window. */
export const FLAPPY_MEMORY_STACKED_FRAME_COUNT = 3;

/** Number of past actions retained for the action-memory channel. */
export const FLAPPY_MEMORY_ACTION_WINDOW_STEPS = 12;

/** Number of observation features fed into each Flappy policy network. */
export const FLAPPY_NETWORK_INPUT_SIZE =
  FLAPPY_MEMORY_CORE_FEATURE_COUNT * FLAPPY_MEMORY_STACKED_FRAME_COUNT + 2;

/** Number of output action scores emitted by each Flappy policy network. */
export const FLAPPY_NETWORK_OUTPUT_SIZE = 2;

/**
 * Hidden-layer sizes used to seed initial Flappy policy architectures.
 *
 * Using at least two hidden layers improves representational flexibility for
 * precise vertical control near narrow, fast-changing pipe targets.
 */
export const FLAPPY_NETWORK_HIDDEN_LAYER_SIZES = [10, 8, 6, 4] as const;
