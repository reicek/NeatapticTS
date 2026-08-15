/**
 * Curriculum executors for the Neatenstein warm-start module.
 *
 * Contains the soft-target constants, curriculum case builder, and per-case
 * loss weighting. The curriculum defines ~23 curated combat+maze training
 * cases with deterministic jitter.
 *
 * @module enemy-warmstart.curriculum.utils
 */

import seedrandom from 'seedrandom';

import {
  TARGET_HIGH,
  TARGET_LOW,
  TARGET_FIRE_MOVE,
  TARGET_FIRE_PURSUE,
  TARGET_FIRE_STALL,
  TARGET_MOVE_STALLED,
  TARGET_TURN_STALLED,
  TARGET_STRAFE,
  TARGET_STRAFE_CONTEXT,
  TARGET_TURN,
  TARGET_FIRE_TURN,
  TARGET_VERY_LOW,
  CURRICULUM_JITTER_AMPLITUDE,
  CURRICULUM_COMBAT_CASE_WEIGHT,
  CURRICULUM_MOVEMENT_CASE_WEIGHT,
  CURRICULUM_FIRST_COMBAT_CASE_INDEX,
  CURRICULUM_JITTER_SEED,
} from './curriculum.constants';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * One supervised training case used by the Neatenstein warm-start curriculum.
 */
export interface CurriculumCase {
  /** Six-dimensional vision input [compassScalar, openN, openE, openS, openW, progressDelta]. */
  input: number[];
  /** Four-dimensional soft action target [move, strafe, turn, fire]. */
  target: number[];
}

// ---------------------------------------------------------------------------
// Soft-target constants — re-exported from curriculum.constants.ts
// ---------------------------------------------------------------------------

export {
  TARGET_HIGH,
  TARGET_LOW,
  TARGET_FIRE_MOVE,
  TARGET_FIRE_PURSUE,
  TARGET_FIRE_STALL,
  TARGET_MOVE_STALLED,
  TARGET_TURN_STALLED,
  TARGET_STRAFE,
  TARGET_STRAFE_CONTEXT,
  TARGET_TURN,
  TARGET_FIRE_TURN,
  TARGET_VERY_LOW,
} from './curriculum.constants';

// ---------------------------------------------------------------------------
// Curriculum builder
// ---------------------------------------------------------------------------

/**
 * Build the Neatenstein warm-start curriculum.
 *
 * Generates ~23 base cases mapping combat+maze vision inputs to soft action
 * targets, with deterministic jitter (±0.1 on inputs) for robustness. The
 * structure mirrors the asciiMaze `buildLamarckianTrainingSet` pattern:
 *
 * - Single-path corridors (4 cases, one per cardinal direction)
 * - Strong-progress corridors (2 cases)
 * - Two-way junctions with directional bias (8 cases)
 * - Regressing/stalled movement (3 cases)
 * - Combat scenarios: fire (2), strafe (2), turn (1), pursue (1)
 *
 * Each case maps a 6-dimensional input `[compassScalar, openN, openE, openS,
 * openW, progressDelta]` to a 4-dimensional soft target
 * `[move, strafe, turn, fire]` with values in [0, 1].
 *
 * Combat cases use input signatures that are clearly distinguishable from
 * movement cases (e.g., all directions open + very high progress for fire,
 * two opposite directions open for strafe, all closed for turn) so the
 * network can learn non-linear decision boundaries in ≤60 iterations.
 *
 * @returns Array of curriculum cases. The output is deterministic — the same
 *   array is produced on every call.
 *
 * @example
 * ```ts
 * const curriculum = buildNeatensteinCurriculum();
 * console.log(curriculum.length); // ~23
 * console.log(curriculum[0].input.length); // 6
 * console.log(curriculum[0].target.length); // 4
 * ```
 */
export function buildNeatensteinCurriculum(): CurriculumCase[] {
  const cases: CurriculumCase[] = [];

  const pushCase = (input: number[], target: number[]): void => {
    cases.push({ input, target });
  };

  // === Movement cases (17 cases) ===

  // Single open path — one direction open, moderate progress (4 cases)
  pushCase(
    [0, 1, 0, 0, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.25, 0, 1, 0, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.5, 0, 0, 1, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.75, 0, 0, 0, 1, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );

  // Strong progress — closing on target (2 cases)
  pushCase(
    [0, 1, 0, 0, 0, 0.9],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_PURSUE],
  );
  pushCase(
    [0.25, 0, 1, 0, 0, 0.9],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_PURSUE],
  );

  // Two-way junctions — compass guides primary direction (8 cases)
  pushCase(
    [0, 1, 0.6, 0, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0, 1, 0, 0.6, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.25, 0.6, 1, 0, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.25, 0, 1, 0.6, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.5, 0, 0.6, 1, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.5, 0, 0, 1, 0.6, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.75, 0, 0, 0.6, 1, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.75, 0.6, 0, 0, 1, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );

  // Regressing/stalled — low progress, need to reorient (3 cases)
  pushCase(
    [0, 1, 0.3, 0, 0, 0.1],
    [TARGET_MOVE_STALLED, TARGET_LOW, TARGET_TURN_STALLED, TARGET_FIRE_STALL],
  );
  pushCase(
    [0.25, 0.5, 1, 0.4, 0, 0.1],
    [TARGET_MOVE_STALLED, TARGET_LOW, TARGET_TURN_STALLED, TARGET_FIRE_STALL],
  );
  pushCase(
    [0.5, 0, 0.3, 1, 0.2, 0.1],
    [TARGET_MOVE_STALLED, TARGET_LOW, TARGET_TURN_STALLED, TARGET_FIRE_STALL],
  );

  // === Combat cases (6 cases) ===

  // Player very close, all directions open — fire (2 cases)
  // Input signature: all-open + very high progress is unique to fire cases.
  pushCase(
    [0, 1, 1, 1, 1, 0.95],
    [TARGET_VERY_LOW, TARGET_LOW, TARGET_LOW, TARGET_HIGH],
  );
  pushCase(
    [0.25, 1, 1, 1, 1, 0.95],
    [TARGET_VERY_LOW, TARGET_LOW, TARGET_LOW, TARGET_HIGH],
  );

  // Player flanking — two opposite directions open, strafe (2 cases)
  // Input signature: opposite-pair-open is unique to strafe cases.
  pushCase(
    [0.5, 1, 0, 1, 0, 0.6],
    [
      TARGET_STRAFE_CONTEXT,
      TARGET_STRAFE,
      TARGET_STRAFE_CONTEXT,
      TARGET_FIRE_MOVE,
    ],
  );
  pushCase(
    [0.75, 0, 1, 0, 1, 0.6],
    [
      TARGET_STRAFE_CONTEXT,
      TARGET_STRAFE,
      TARGET_STRAFE_CONTEXT,
      TARGET_FIRE_MOVE,
    ],
  );

  // Stalled, all closed — turn to find player (1 case)
  // Input signature: all-closed + very low progress is unique to turn case.
  pushCase(
    [0.5, 0, 0, 0, 0, 0.05],
    [TARGET_VERY_LOW, TARGET_LOW, TARGET_TURN, TARGET_FIRE_TURN],
  );

  // Player retreating — moderate progress, pursue (1 case)
  pushCase(
    [0, 1, 0, 0, 0, 0.3],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );

  // === Deterministic jitter (±0.1 on inputs) ===
  const rng = seedrandom(CURRICULUM_JITTER_SEED);
  const jitterAmp = CURRICULUM_JITTER_AMPLITUDE;
  for (const c of cases) {
    for (let i = 0; i < c.input.length; i++) {
      c.input[i] = Math.max(
        0,
        Math.min(1, c.input[i] + (rng() * 2 - 1) * jitterAmp),
      );
    }
  }

  return cases;
}

/**
 * Per-case loss weighting for the Neatenstein curriculum.
 *
 * Combat cases (fire, strafe, turn, pursue) are a minority of the curriculum
 * but represent critical behavioural modes. Without weighting, the majority
 * movement cases dominate the gradient and the network fails to learn combat
 * behaviours. This function returns a weight array giving combat cases 2×
 * influence so the network can learn both movement and combat within ≤60
 * iterations.
 *
 * @param numCases - Number of curriculum cases (must match the curriculum).
 * @returns Array of per-case weights (1.0 for movement, 2.0 for combat).
 */
export function getCurriculumCaseWeights(numCases: number): number[] {
  // The curriculum layout is: 14 movement (0-13), 3 stalled (14-16),
  // and combat cases (17+). Combat cases represent distinct behavioural
  // modes that are minority cases; weighting them 2× ensures the network
  // learns both movement and combat within ≤60 iterations.
  const weights = new Array<number>(numCases);
  for (let i = 0; i < numCases; i++) {
    weights[i] = i >= CURRICULUM_FIRST_COMBAT_CASE_INDEX ? CURRICULUM_COMBAT_CASE_WEIGHT : CURRICULUM_MOVEMENT_CASE_WEIGHT;
  }
  return weights;
}
