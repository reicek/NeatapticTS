/**
 * NeatRacing fitness shaping (slice `P2-S3-E`, TP-08, AC-029 support).
 *
 * THE LAW (spec section 9.1, E8): the growth-engine commit rule
 * (`bestVariantScore > baselineScore + threshold`) consumes ONE
 * HIGHER-IS-BETTER fitness number, while the recycled team-fitness resolver
 * (`resolveTeamFitness`) is LOWEST-POSITION-WINS. The recycled position
 * ordering therefore enters this module INVERTED — `maxCars − position` — so
 * finishing P1 scores the highest of the field. The shaped value is a PURE
 * function of three probe terms:
 *
 * ```text
 * fitness = W_POS · (maxCars − position) − W_TIME · time + W_LP · lapProgress01
 * ```
 *
 * Monotonic directions (the pinned red-test invariants, each term held with
 * the others fixed):
 *
 * - POSITION: fitness strictly INCREASES as the position number DECREASES —
 *   each inverted position step pays exactly `W_POS` (P1 > P2 > ... > Pn).
 * - TIME: fitness strictly INCREASES as the elapsed race time DECREASES —
 *   each elapsed second pays exactly `W_TIME` (finishing faster is better).
 * - LAP PROGRESS: fitness strictly INCREASES with `lapProgress01` — each
 *   unit of progress pays exactly `W_LP` (more laps driven is better).
 *
 * THE FORMULA SHAPE IS CONTRACT; THE WEIGHTS ARE TP: the sum-of-terms shape
 * above (including the inverted position term) is the deliverable contract,
 * while `NEATRACING_FITNESS_WEIGHT_POSITION`, `..._TIME` and
 * `..._LAP_PROGRESS` are TP-08 tuning points — arbitrary starting values,
 * retunable in Phase 2 as one-constant changes (spec risk R-04). A retune
 * MUST keep every weight strictly positive or the monotonic directions
 * above (and therefore the higher-is-better contract) break.
 *
 * THE ACCUMULATION ORDER IS PINNED: the terms are summed left-to-right
 * (position -> time -> progress) so the same runtime keeps byte-identical
 * scores for identical inputs (same-runtime Level-2 ordered-deterministic;
 * never a cross-environment bit-exact claim — floating-point associativity
 * is not guaranteed across runtimes).
 *
 * THE STRICT-REJECTION GUARDS keep the bounded-output contract: a
 * non-finite or out-of-range input throws a `RangeError`
 * (`neatRacing fitness shaping failed` voice) instead of silently shaping a
 * NaN into the commit rule, where a NaN score would disable the
 * `bestVariantScore > baselineScore + threshold` comparison. `lapProgress01`
 * callers are expected to be already-normalized by the observation
 * assembler's `normalizeNeatRacingProbability` clamp path; this module
 * surfaces out-of-band data loudly rather than re-clamping it.
 *
 * THE CONSUMER: the eval worker's variant evaluation reports the
 * deterministic `score` per variant slot (AC-029 support) — this module is
 * the pure library-side shaping surface behind that protocol field; it owns
 * no RNG, no wall clock, no worker globals.
 *
 * @module
 */

/* ------------------------------------------------------------------------- *
 * Section 9.1 — the TP-08 fitness shaping weight constants
 * ------------------------------------------------------------------------- */

/**
 * Weight of the INVERTED position term, paid per position step
 * (`maxCars − position`): P1 scores `maxCars − 1` steps above the last-placed
 * car, so the P1-vs-last gap scales with grid size (the lvl-3 five-car grid
 * spans 4 steps).
 *
 * TP-08 tuning point — arbitrary starting value, NOT contract; retunable in
 * Phase 2. Must stay strictly positive (spec E8 monotonicity).
 */
export const NEATRACING_FITNESS_WEIGHT_POSITION = 10;

/**
 * Weight of the elapsed-race-time penalty, paid per elapsed second: less
 * race time is strictly higher fitness.
 *
 * TP-08 tuning point — arbitrary starting value chosen so a full lvl-3 grid
 * position step (10 points) outweighs several minutes of race time; retunable
 * in Phase 2. Must stay strictly positive (spec E8 monotonicity).
 */
export const NEATRACING_FITNESS_WEIGHT_TIME = 0.01;

/**
 * Weight of the lap-progress term, paid per unit of `lapProgress01` (the
 * observation assembler's clamped 0..1 progress): a full lap of progress
 * scores exactly this weight.
 *
 * TP-08 tuning point — arbitrary starting value chosen so one full lap of
 * progress ties a 100-second reduction in race time; retunable in Phase 2.
 * Must stay strictly positive (spec E8 monotonicity).
 */
export const NEATRACING_FITNESS_WEIGHT_LAP_PROGRESS = 1;

/* ------------------------------------------------------------------------- *
 * Section 9.1 — the shaping function
 * ------------------------------------------------------------------------- */

/**
 * Computes the shaped HIGHER-IS-BETTER fitness of one car probe.
 *
 * The spec's formula shape is `fitness(position, time, lapProgress)`; this
 * module names the terms for the lane's data shapes (`position` is the
 * 1-based finishing position, `timeSeconds` is elapsed race time in seconds
 * and `lapProgress01` is the assembler's clamped 0..1 lap progress) and
 * takes `maxCars` explicitly because the recycled position ordering is only
 * invertible relative to the grid size it ranks against.
 *
 * @param position - The car's 1-based finishing position in `[1, maxCars]`
 *   (P1 = 1). A non-integer or out-of-grid value is rejected.
 * @param timeSeconds - The elapsed race time in seconds (`>= 0`). Less time
 *   is strictly higher fitness.
 * @param lapProgress01 - The clamped lap progress in `[0, 1]`. More progress
 *   is strictly higher fitness.
 * @param maxCars - The grid size the position ranks against — a positive
 *   integer `>= 1` (the lvl-3 E2 layout carries 5 cars: 1 main agent plus
 *   4 rivals).
 * @returns The shaped fitness. Finite for every accepted input; strictly
 *   monotonic in each term per the module-header directions.
 * @throws {RangeError} With the `neatRacing fitness shaping failed` voice
 *   when any input is non-finite, `position` is not an integer within
 *   `[1, maxCars]`, `timeSeconds` is negative, `lapProgress01` is outside
 *   `[0, 1]` or `maxCars` is not a positive integer.
 *
 * @example
 * ```ts
 * // lvl-3 grid (5 cars): P1 after 41.5 s with 76% lap progress
 * const fitness = computeNeatRacingShapedFitness(1, 41.5, 0.76, 5);
 * // 10 * (5 - 1) - 0.01 * 41.5 + 1 * 0.76 === 40.345
 * ```
 */
export function computeNeatRacingShapedFitness(
  position: number,
  timeSeconds: number,
  lapProgress01: number,
  maxCars: number,
): number {
  // Step 1: strict-rejection guards — a bad input is a loud RangeError, never
  // a NaN score silently poisoning the commit-rule comparison.
  assertNeatRacingShapedFitnessInputs(
    position,
    timeSeconds,
    lapProgress01,
    maxCars,
  );

  // Step 2: shape the three contract terms (the accumulation order is pinned:
  // position term, then the time penalty, then the progress bonus).
  const positionTerm =
    NEATRACING_FITNESS_WEIGHT_POSITION * (maxCars - position);
  const timeTerm = NEATRACING_FITNESS_WEIGHT_TIME * timeSeconds;
  const lapProgressTerm =
    NEATRACING_FITNESS_WEIGHT_LAP_PROGRESS * lapProgress01;

  return positionTerm - timeTerm + lapProgressTerm;
}

/* ------------------------------------------------------------------------- *
 * Section 9.1 — private guards (below the fold)
 * ------------------------------------------------------------------------- */

/**
 * The shared error voice of the fitness-shaping guard stack, matching the
 * lane's `neatRacing <surface> failed` RangeError convention.
 */
const NEATRACING_FITNESS_SHAPING_ERROR_PREFIX =
  'neatRacing fitness shaping failed';

/**
 * Validates the whole shaped-fitness input bundle before any term is shaped.
 *
 * `maxCars` is guarded first because it bounds the position range the
 * position guard checks against.
 *
 * @param position - The 1-based finishing position.
 * @param timeSeconds - The elapsed race time in seconds.
 * @param lapProgress01 - The clamped lap progress.
 * @param maxCars - The grid size the position ranks against.
 * @throws {RangeError} When any input violates its term contract.
 */
function assertNeatRacingShapedFitnessInputs(
  position: number,
  timeSeconds: number,
  lapProgress01: number,
  maxCars: number,
): void {
  assertFitnessMaxCars(maxCars);
  assertFitnessPosition(position, maxCars);
  assertFitnessTimeSeconds(timeSeconds);
  assertFitnessLapProgress01(lapProgress01);
}

/**
 * Guards one shaped-fitness term value against non-finite data (NaN and the
 * infinities have no committable score meaning here).
 *
 * @param value - The term value to guard.
 * @param label - The term's name (for the static message surface).
 * @throws {RangeError} When the value is not a finite number.
 */
function assertFitnessFiniteNumber(value: number, label: string): void {
  if (!Number.isFinite(value)) {
    throw new RangeError(
      `${NEATRACING_FITNESS_SHAPING_ERROR_PREFIX}: ${label} must be a finite number`,
    );
  }
}

/**
 * Guards the grid size the position ranks against (a rank is only invertible
 * relative to a well-formed grid).
 *
 * @param maxCars - The grid size to guard.
 * @throws {RangeError} When the grid size is not a positive integer.
 */
function assertFitnessMaxCars(maxCars: number): void {
  assertFitnessFiniteNumber(maxCars, 'maxCars');
  if (!Number.isInteger(maxCars) || maxCars < 1) {
    throw new RangeError(
      `${NEATRACING_FITNESS_SHAPING_ERROR_PREFIX}: maxCars must be a positive (>= 1) integer`,
    );
  }
}

/**
 * Guards the 1-based finishing position against the well-formed grid.
 *
 * @param position - The position to guard.
 * @param maxCars - The grid size the position ranks against.
 * @throws {RangeError} When the position is not an integer within
 *   `[1, maxCars]`.
 */
function assertFitnessPosition(position: number, maxCars: number): void {
  assertFitnessFiniteNumber(position, 'position');
  if (!Number.isInteger(position) || position < 1 || position > maxCars) {
    throw new RangeError(
      `${NEATRACING_FITNESS_SHAPING_ERROR_PREFIX}: position must be an integer within [1, maxCars]`,
    );
  }
}

/**
 * Guards the elapsed race time (a race probe can never record negative time;
 * the time term is a monotonic penalty, so `0` is the best time).
 *
 * @param timeSeconds - The elapsed race time to guard.
 * @throws {RangeError} When the time is non-finite or negative.
 */
function assertFitnessTimeSeconds(timeSeconds: number): void {
  assertFitnessFiniteNumber(timeSeconds, 'timeSeconds');
  if (timeSeconds < 0) {
    throw new RangeError(
      `${NEATRACING_FITNESS_SHAPING_ERROR_PREFIX}: timeSeconds must not be negative`,
    );
  }
}

/**
 * Guards the clamped lap progress (the observation assembler's
 * `normalizeNeatRacingProbability` path already clamps to `[0, 1]`; an
 * out-of-band value here is bad upstream data to surface, not noise to absorb).
 *
 * @param lapProgress01 - The lap progress to guard.
 * @throws {RangeError} When the progress is non-finite or outside `[0, 1]`.
 */
function assertFitnessLapProgress01(lapProgress01: number): void {
  assertFitnessFiniteNumber(lapProgress01, 'lapProgress01');
  if (lapProgress01 < 0 || lapProgress01 > 1) {
    throw new RangeError(
      `${NEATRACING_FITNESS_SHAPING_ERROR_PREFIX}: lapProgress01 must be within [0, 1]`,
    );
  }
}
