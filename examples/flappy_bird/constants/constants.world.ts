/**
 * Flappy world and episode-shape constants.
 *
 * This module groups stable geometry and run-budget values that define the
 * simulation envelope. Keeping these together makes it easier to reason about
 * how large the world is, where the bird is anchored, and when an episode ends.
 */

/** Width of the simulated world (pixels). */
export const FLAPPY_WORLD_WIDTH_PX = 288;

/** Height of the simulated world (pixels). */
export const FLAPPY_WORLD_HEIGHT_PX = 512;

/**
 * Fixed horizontal position of the bird (pixels).
 *
 * A fixed x-anchor turns the task into primarily vertical control while pipes
 * move left, making policy behavior easier to visualize and debug.
 */
export const FLAPPY_BIRD_X_PX = 60;

/** Bird collision radius (pixels). */
export const FLAPPY_BIRD_RADIUS_PX = 4;

/** Bird collision height (diameter, pixels). */
export const FLAPPY_BIRD_HEIGHT_PX = FLAPPY_BIRD_RADIUS_PX * 2;

/**
 * Enables runtime telemetry counters used for profiling diagnostics.
 *
 * Keep disabled during normal demo runs to avoid instrumentation overhead and
 * to preserve a cleaner educational rendering path.
 */
export const FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION = false;

/**
 * Episode terminates after this many frames even if still alive.
 *
 * This prevents extremely long outlier episodes from dominating generation
 * runtime and keeps evolution throughput predictable.
 */
export const FLAPPY_MAX_FRAMES_PER_EPISODE = 5_000;
