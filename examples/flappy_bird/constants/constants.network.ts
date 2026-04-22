/**
 * Flappy policy-network and temporal memory constants.
 *
 * This module defines observation window shape and initial architecture sizing
 * used when seeding agents for evolution.
 */

/**
 * Number of compact current-frame observation features fed into Flappy policies.
 *
 * The controller keeps only bird-state and next-gap geometry on the public
 * input shelf. Higher-level control-pressure hints remain available as derived
 * features for shaping and heuristics, but they are no longer wired directly
 * into the network input.
 */
export const FLAPPY_MEMORY_CORE_FEATURE_COUNT = 6;

/**
 * Effective controller frame count kept in the external observation window.
 *
 * All Flappy architectures now consume only the current normalized frame so
 * recurrent profiles are not double-fed with both built-in state and a
 * hand-authored temporal stack.
 */
export const FLAPPY_MEMORY_STACKED_FRAME_COUNT = 1;

/**
 * Size of the legacy action-history buffer.
 *
 * This stays at zero so no controller receives hand-authored action memory on
 * top of its learned state.
 */
export const FLAPPY_MEMORY_ACTION_WINDOW_STEPS = 0;

/** Number of observation features fed into each Flappy policy network. */
export const FLAPPY_NETWORK_INPUT_SIZE = FLAPPY_MEMORY_CORE_FEATURE_COUNT;

/** Number of output action scores emitted by each Flappy policy network. */
export const FLAPPY_NETWORK_OUTPUT_SIZE = 2;

/**
 * Hidden-layer sizes used to seed initial Flappy policy architectures.
 *
 * Using at least two hidden layers improves representational flexibility for
 * precise vertical control near narrow, fast-changing pipe targets.
 */
export const FLAPPY_NETWORK_HIDDEN_LAYER_SIZES = [24, 12, 6] as const;
