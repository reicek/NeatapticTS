/**
 * Flappy policy-network and temporal memory constants.
 *
 * This module defines observation window shape and initial architecture sizing
 * used when seeding agents for evolution.
 */

/**
 * Number of compact current-frame observation features fed into Flappy policies.
 *
 * The nine channels are split into three semantic groups:
 *
 * **Bird state (2):** normalized height and vertical velocity — the controller's
 * body-state check before any pipe geometry matters.
 *
 * **Next gap (4):** distance to the pipe exit, signed vertical offset from the
 * gap center, normalized top and bottom boundaries of the safe corridor.
 *
 * **Look-ahead (3):** signed distance to the pipe *entrance* (goes negative
 * while the bird is traversing the pipe body, giving a clear in-pipe signal),
 * signed gap clearance (how centered the bird currently is inside the opening),
 * and signed vertical offset from the *second* upcoming gap center (gives
 * the network a reason to plan ahead instead of staying level).
 */
export const FLAPPY_MEMORY_CORE_FEATURE_COUNT = 9;

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
