/**
 * Curriculum soft-target and case-weight constants for the Neatenstein
 * warm-start module.
 *
 * Extracts the ~13 soft-target probability constants, the input jitter
 * amplitude, the combat-case weight, and the movement/stalled case indices so
 * that `enemy-warmstart.curriculum.utils.ts` contains only the curriculum
 * builder logic with no inline magic numbers.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Soft-target probabilities
// ---------------------------------------------------------------------------

/** Soft-target probability for the primary action. */
export const TARGET_HIGH = 0.8;

/** Soft-target probability for inactive actions. */
export const TARGET_LOW = 0.05;

/** Soft-target probability for fire during movement (encourages opportunistic firing). */
export const TARGET_FIRE_MOVE = 0.3;

/** Soft-target probability for fire during aggressive pursuit. */
export const TARGET_FIRE_PURSUE = 0.3;

/** Soft-target probability for fire during stalled/turning states. */
export const TARGET_FIRE_STALL = 0.3;

/** Soft-target for moderate movement (stalled/reorienting). */
export const TARGET_MOVE_STALLED = 0.45;

/** Soft-target for moderate turning (stalled/reorienting). */
export const TARGET_TURN_STALLED = 0.3;

/** Soft-target for strafe primary action. */
export const TARGET_STRAFE = 0.4;

/** Soft-target for moderate strafe context actions. */
export const TARGET_STRAFE_CONTEXT = 0.3;

/** Soft-target for turn primary action. */
export const TARGET_TURN = 0.7;

/** Soft-target for fire during turn (looking for player). */
export const TARGET_FIRE_TURN = 0.35;

/** Soft-target for very low movement (fire dominant). */
export const TARGET_VERY_LOW = 0.05;

// ---------------------------------------------------------------------------
// Jitter & weighting constants
// ---------------------------------------------------------------------------

/** Amplitude of deterministic input jitter applied to curriculum cases (±). */
export const CURRICULUM_JITTER_AMPLITUDE = 0.1;

/** Loss weight applied to combat cases (fire, strafe, turn, pursue). */
export const CURRICULUM_COMBAT_CASE_WEIGHT = 2.0;

/** Loss weight applied to movement and stalled cases. */
export const CURRICULUM_MOVEMENT_CASE_WEIGHT = 1.0;

/** Index of the first combat case in the curriculum (0-based). */
export const CURRICULUM_FIRST_COMBAT_CASE_INDEX = 17;

/** Deterministic seed string for the curriculum jitter PRNG. */
export const CURRICULUM_JITTER_SEED = 'neatenstein:curriculum:jitter:42';