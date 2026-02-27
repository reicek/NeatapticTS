/**
 * Flappy Bird demo constants.
 *
 * These values define the physics and geometry for a lightweight Flappy Bird–style
 * environment used to exercise NeatapticTS neuroevolution.
 */

/** Width of the simulated world (pixels). */
export const FLAPPY_WORLD_WIDTH_PX = 288;

/** Height of the simulated world (pixels). */
export const FLAPPY_WORLD_HEIGHT_PX = 512;

/** Fixed horizontal position of the bird (pixels). */
export const FLAPPY_BIRD_X_PX = 60;

/** Bird collision radius (pixels). */
export const FLAPPY_BIRD_RADIUS_PX = 4;

/**
 * Gravity acceleration applied each frame (pixels/frame²).
 *
 * Slightly increased so birds settle faster after each flap and can make
 * finer vertical corrections around narrow targets.
 */
export const FLAPPY_GRAVITY_PX_PER_FRAME2 = 0.56;

/**
 * Instantaneous upward velocity applied on flap (pixels/frame).
 *
 * Reduced so each flap produces a smaller hop, improving precision when
 * threading tighter gaps.
 */
export const FLAPPY_FLAP_VELOCITY_PX_PER_FRAME = -5.8;

/** Maximum downward speed clamp (pixels/frame). */
export const FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME = 8.8;

/** Pipe width (pixels). */
export const FLAPPY_PIPE_WIDTH_PX = 40;

/** Vertical opening size of each pipe gap (pixels). */
export const FLAPPY_PIPE_GAP_PX = 150;

/** Pipe horizontal speed (pixels/frame). */
export const FLAPPY_PIPE_SPEED_PX_PER_FRAME = 4;

/** Frames between spawning new pipes. */
export const FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES = 100;

/**
 * Number of policy decision substeps executed inside each logical frame.
 *
 * Values > 1 give agents finer temporal control in tight scenarios by allowing
 * multiple react-and-integrate passes before the frame counter advances.
 */
export const FLAPPY_CONTROL_SUBSTEPS_PER_FRAME = 6;

/** Bird collision height (diameter, pixels). */
export const FLAPPY_BIRD_HEIGHT_PX = FLAPPY_BIRD_RADIUS_PX * 2;

/**
 * Target control cadence used for endgame reachability calculations.
 *
 * A smaller value means the policy is expected to correct more frequently
 * (effectively "jumping more often"), which supports narrower endgame gaps.
 */
export const FLAPPY_TARGET_FLAP_INTERVAL_FRAMES = 2;

/** Small geometric buffer so "barely possible" remains physically solvable. */
export const FLAPPY_MIN_CLEARANCE_MARGIN_PX = 20;

/**
 * Hard floor on time between pipes at max speed so controllers can recover.
 *
 * This prevents endgame spacing from becoming too tight for realistic policy
 * reaction and vertical correction.
 */
export const FLAPPY_MIN_PIPE_RECOVERY_FRAMES = 50;

/**
 * Minimum edge-to-edge spacing needed to recover between consecutive pipes.
 *
 * Derived from bird size + expected control drop budget + a small margin.
 */
export const FLAPPY_MIN_EDGE_TO_EDGE_PIPE_SPACING_PX =
  FLAPPY_BIRD_HEIGHT_PX +
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME * FLAPPY_TARGET_FLAP_INTERVAL_FRAMES +
  FLAPPY_MIN_CLEARANCE_MARGIN_PX;

/** Minimum allowed gap center height (pixels). */
export const FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX = 90;

/** Maximum allowed gap center height (pixels). */
export const FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX =
  FLAPPY_WORLD_HEIGHT_PX - 90;

/** Episode terminates after this many frames even if still alive. */
export const FLAPPY_MAX_FRAMES_PER_EPISODE = 5_000;

/** Fitness bonus added per pipe successfully passed. */
export const FLAPPY_FITNESS_BONUS_PER_PIPE = 1_000;

/** Number of observation features fed into each Flappy policy network. */
export const FLAPPY_NETWORK_INPUT_SIZE = 12;

/** Number of output action scores emitted by each Flappy policy network. */
export const FLAPPY_NETWORK_OUTPUT_SIZE = 2;

/**
 * Hidden-layer sizes used to seed initial Flappy policy architectures.
 *
 * Using at least two hidden layers improves representational flexibility for
 * precise vertical control near narrow, fast-changing pipe targets.
 */
export const FLAPPY_NETWORK_HIDDEN_LAYER_SIZES = [6, 6] as const;

/**
 * Fraction of parent bird opacity used for drawing trails.
 *
 * For example, 0.5 means a bird at 20% opacity gets a 10% opacity trail.
 */
export const FLAPPY_TRAIL_OPACITY_FACTOR = 0.5;

/**
 * Fraction of raw frame-survival reward kept in total fitness.
 *
 * Lower values reduce the incentive to merely stay alive and increase pressure
 * to center on gaps and pass pipes cleanly.
 */
export const FLAPPY_FITNESS_SURVIVAL_WEIGHT = 0.65;

/** Per-frame reward weight for staying vertically aligned with the next gap. */
export const FLAPPY_FITNESS_ALIGNMENT_WEIGHT_PER_FRAME = 0.9;

/** Reward scale for reducing distance to the next pipe between consecutive frames. */
export const FLAPPY_FITNESS_APPROACH_PROGRESS_WEIGHT = 320;

/** Reward scale for reducing vertical error to the next gap center. */
export const FLAPPY_FITNESS_CENTERING_PROGRESS_WEIGHT = 320;

/** Per-frame reward weight for keeping the bird inside next-gap clearance. */
export const FLAPPY_FITNESS_CLEARANCE_WEIGHT_PER_FRAME = 0.35;

/** Per-frame reward weight for pre-aligning with the second upcoming gap. */
export const FLAPPY_FITNESS_SECOND_GAP_ALIGNMENT_WEIGHT_PER_FRAME = 0.2;

/** Per-frame reward weight for maintaining controllable vertical velocity. */
export const FLAPPY_FITNESS_STABLE_VELOCITY_WEIGHT_PER_FRAME = 0.2;

/** Terminal bonus based on final alignment with the next gap center. */
export const FLAPPY_FITNESS_TERMINAL_ALIGNMENT_BONUS_WEIGHT = 180;

/** Terminal bonus based on final progress toward the next pipe. */
export const FLAPPY_FITNESS_TERMINAL_PROGRESS_BONUS_WEIGHT = 80;

/** Minimum pipe gap used at peak adaptive difficulty. */
export const FLAPPY_PIPE_GAP_MIN_PX =
  FLAPPY_BIRD_HEIGHT_PX +
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME * FLAPPY_TARGET_FLAP_INTERVAL_FRAMES +
  FLAPPY_MIN_CLEARANCE_MARGIN_PX;

/** Initial spawn gap multiplier relative to the current hardest gap target. */
export const FLAPPY_PIPE_GAP_START_MULTIPLIER = 2.15;

/** Per-pipe gap shrink step toward the current hardest target gap (pixels). */
export const FLAPPY_PIPE_GAP_SHRINK_PER_PIPE_PX = 4;

/** Random jitter range applied to each spawned pipe gap (pixels). */
export const FLAPPY_PIPE_GAP_RANDOM_JITTER_PX = 6;

/**
 * Maximum allowed vertical jump between consecutive pipe gap centers (pixels).
 *
 * This reduces abrupt zig-zag transitions that are often unrecoverable once
 * spacing tightens at higher difficulty.
 */
export const FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX = 72;

/** Maximum pipe speed used at peak adaptive difficulty. */
export const FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME = 3;

/** Minimum spawn interval used at peak adaptive difficulty. */
export const FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES = Math.max(
  FLAPPY_MIN_PIPE_RECOVERY_FRAMES,
  Math.ceil(
    (FLAPPY_PIPE_WIDTH_PX + FLAPPY_MIN_EDGE_TO_EDGE_PIPE_SPACING_PX) /
      FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME,
  ),
);

/** Initial spawn-interval multiplier relative to the current hardest interval target. */
export const FLAPPY_PIPE_SPAWN_INTERVAL_START_MULTIPLIER = 2.35;

/** Per-pipe spawn-interval shrink step toward the current hardest interval target (frames). */
export const FLAPPY_PIPE_SPAWN_INTERVAL_SHRINK_PER_PIPE_FRAMES = 1;

/**
 * Pipe-pass count needed to reach maximum adaptive difficulty.
 *
 * After this point, spacing does not tighten further, so proficient agents can
 * sustain long runs without additional spacing compression.
 */
export const FLAPPY_DIFFICULTY_RAMP_PIPES = 25;
