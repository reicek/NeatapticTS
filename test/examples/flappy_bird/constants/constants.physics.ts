/**
 * Flappy physics and control-cadence constants.
 *
 * Values in this module shape how quickly the bird falls, how strongly a flap
 * responds, and how often a policy can react inside each visible frame.
 */

/**
 * Gravity acceleration applied each frame (pixels/frame²).
 *
 * Slightly increased so birds settle faster after each flap and can make
 * finer vertical corrections around narrow targets.
 */
export const FLAPPY_GRAVITY_PX_PER_FRAME2 = 0.22;

/**
 * Instantaneous upward velocity applied on flap (pixels/frame).
 *
 * Reduced so each flap produces a smaller hop, improving precision when
 * threading tighter gaps.
 */
export const FLAPPY_FLAP_VELOCITY_PX_PER_FRAME = -5;

/**
 * Maximum downward speed clamp (pixels/frame).
 *
 * This cap limits runaway fall acceleration and keeps trajectories learnable.
 */
export const FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME = 7;

/**
 * Number of policy decision substeps executed inside each logical frame.
 *
 * Values > 1 give agents finer temporal control in tight scenarios by allowing
 * multiple react-and-integrate passes before the frame counter advances.
 */
export const FLAPPY_CONTROL_SUBSTEPS_PER_FRAME = 5;

/**
 * Target control cadence used for endgame reachability calculations.
 *
 * A smaller value means the policy is expected to correct more frequently
 * (effectively "jumping more often"), which supports narrower endgame gaps.
 */
export const FLAPPY_TARGET_FLAP_INTERVAL_FRAMES = 2;
