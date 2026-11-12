/**
 * Green tests for the NeatRacing fitness shaping (`fitness.shaping.utils.ts`
 * — Step 04, slice P2-S3-E, TDD lane green-only).
 *
 * Contracts pinned here (spec section 9.1, E8 / TP-08, AC-029 support):
 * - HIGHER-IS-BETTER: the shaped fitness feeds the growth-engine commit rule
 *   (`bestVariantScore > baselineScore + threshold`), so a better-finishing
 *   variant must score strictly above its baseline (AC-029 support).
 * - THE INVERTED POSITION TERM: the recycled team-fitness resolver is
 *   lowest-position-wins, so the position term enters as `maxCars − position`
 *   — P1 scores highest, each better position step pays exactly the TP-08
 *   position weight, and the sweep over the REAL lvl-3 team layout
 *   (`NEATRACING_TEAM_LAYOUTS[3]`, 1 main agent + 4 rivals) is strictly
 *   monotonic.
 * - STRICT MONOTONICITY IN EACH TERM (the E8 red-test invariants): position,
 *   time and lap progress each move the fitness strictly in their pinned
 *   direction with the other two terms held fixed.
 * - DETERMINISM: the same inputs shape the same fitness (a pure function —
 *   same-runtime Level-2 ordered-deterministic; no RNG, no wall clock).
 * - BOUNDED OUTPUTS: every accepted (finite, in-range) input yields a finite
 *   score, and the all-floor corner pins exactly 0.
 * - THE FORMULA SHAPE IS CONTRACT; THE WEIGHTS ARE TP: the three named TP-08
 *   weights drive the term math (the delta tests READ the constants, so a
 *   Phase-2 retune stays green as long as the shape holds) and every weight
 *   must stay strictly positive or the monotonic directions break.
 * - THE STRICT-REJECTION GUARDS: non-finite or out-of-range terms throw a
 *   `RangeError` with the `neatRacing fitness shaping failed` voice naming
 *   the offending term — a NaN score must never silently poison the commit
 *   rule.
 *
 * @module
 */

import { NEATRACING_TEAM_LAYOUTS } from './observation.assembler';
import {
  computeNeatRacingShapedFitness,
  NEATRACING_FITNESS_WEIGHT_LAP_PROGRESS,
  NEATRACING_FITNESS_WEIGHT_POSITION,
  NEATRACING_FITNESS_WEIGHT_TIME,
} from './fitness.shaping.utils';

/** The real lvl-3 E2 team layout grid size (1 main agent + 4 rivals). */
const MAX_CARS = NEATRACING_TEAM_LAYOUTS[3].length;

/** Probe race time shared by the fixed-term sweeps. */
const PROBE_TIME_SECONDS = 41.5;

/** Probe lap progress shared by the fixed-term sweeps. */
const PROBE_LAP_PROGRESS_01 = 0.76;

/** Ascending race-time sweep (in seconds) for the time-term monotonicity. */
const TIME_SWEEP_SECONDS = [0, 20, 40, 60, 80] as const;

/** Ascending lap-progress sweep for the progress-term monotonicity. */
const PROGRESS_SWEEP_01 = [0, 0.25, 0.5, 0.75, 1] as const;

/** Valid input corners exercised for the bounded-output contract. */
const SHAPED_FITNESS_CORNERS = [
  [1, 0, 1, MAX_CARS],
  [MAX_CARS, 0, 0, MAX_CARS],
  [1, 120, 0.5, MAX_CARS],
  [MAX_CARS, 600, 0, MAX_CARS],
] as const;

describe('TP-08 fitness shaping weights (named constants)', () => {
  it('exposes all three named weights strictly positive (the monotonicity precondition)', () => {
    expect(NEATRACING_FITNESS_WEIGHT_POSITION).toBeGreaterThan(0);
    expect(NEATRACING_FITNESS_WEIGHT_TIME).toBeGreaterThan(0);
    expect(NEATRACING_FITNESS_WEIGHT_LAP_PROGRESS).toBeGreaterThan(0);
  });
});

describe('higher-is-better monotonicity (E8 red-test invariants)', () => {
  it('strictly decreases the fitness across the real lvl-3 position field', () => {
    const fitnessByAscendingPosition = Array.from(
      { length: MAX_CARS },
      (_, gridIndex) =>
        computeNeatRacingShapedFitness(
          gridIndex + 1,
          PROBE_TIME_SECONDS,
          PROBE_LAP_PROGRESS_01,
          MAX_CARS,
        ),
    );
    expect(isStrictlyDecreasing(fitnessByAscendingPosition)).toBe(true);
  });

  it('strictly decreases the fitness as the elapsed race time grows', () => {
    const fitnessByAscendingTime = TIME_SWEEP_SECONDS.map((timeSeconds) =>
      computeNeatRacingShapedFitness(
        1,
        timeSeconds,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      ),
    );
    expect(isStrictlyDecreasing(fitnessByAscendingTime)).toBe(true);
  });

  it('strictly increases the fitness as the lap progress grows', () => {
    const fitnessByAscendingProgress = PROGRESS_SWEEP_01.map((lapProgress01) =>
      computeNeatRacingShapedFitness(
        1,
        PROBE_TIME_SECONDS,
        lapProgress01,
        MAX_CARS,
      ),
    );
    expect(isStrictlyIncreasing(fitnessByAscendingProgress)).toBe(true);
  });
});

describe('term math (the formula shape is contract)', () => {
  it('pays exactly the TP-08 position weight per better position step', () => {
    const oneBetterPositionDelta =
      computeNeatRacingShapedFitness(
        1,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      ) -
      computeNeatRacingShapedFitness(
        2,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      );
    expect(oneBetterPositionDelta).toBeCloseTo(
      NEATRACING_FITNESS_WEIGHT_POSITION,
      12,
    );
  });

  it('pays exactly the TP-08 time weight per elapsed second', () => {
    const oneMoreSecondDelta =
      computeNeatRacingShapedFitness(
        1,
        PROBE_TIME_SECONDS + 1,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      ) -
      computeNeatRacingShapedFitness(
        1,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      );
    expect(oneMoreSecondDelta).toBeCloseTo(-NEATRACING_FITNESS_WEIGHT_TIME, 12);
  });

  it('pays exactly the TP-08 lap-progress weight per unit of progress', () => {
    const halfProgressDelta =
      computeNeatRacingShapedFitness(1, PROBE_TIME_SECONDS, 0.5, MAX_CARS) -
      computeNeatRacingShapedFitness(1, PROBE_TIME_SECONDS, 0, MAX_CARS);
    expect(halfProgressDelta).toBeCloseTo(
      0.5 * NEATRACING_FITNESS_WEIGHT_LAP_PROGRESS,
      12,
    );
  });

  it('pins the all-floor corner (last place, zero time, zero progress) to exactly zero', () => {
    expect(computeNeatRacingShapedFitness(MAX_CARS, 0, 0, MAX_CARS)).toBe(0);
  });
});

describe('determinism (E8 red-test invariant)', () => {
  it('shapes bit-identical fitness for repeated identical inputs', () => {
    const firstCall = computeNeatRacingShapedFitness(3, 57.25, 0.34, MAX_CARS);
    const secondCall = computeNeatRacingShapedFitness(3, 57.25, 0.34, MAX_CARS);
    expect(secondCall).toBe(firstCall);
  });
});

describe('bounded outputs (finite for every accepted input)', () => {
  it('shapes a finite score at every valid input corner', () => {
    expect(
      SHAPED_FITNESS_CORNERS.every(
        ([position, timeSeconds, lapProgress01, maxCars]) =>
          Number.isFinite(
            computeNeatRacingShapedFitness(
              position,
              timeSeconds,
              lapProgress01,
              maxCars,
            ),
          ),
      ),
    ).toBe(true);
  });
});

describe('strict-rejection guards (the commit rule never sees a NaN score)', () => {
  it('rejects a non-finite position', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        Number.NaN,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      ),
    ).toThrow(RangeError);
  });

  it('rejects a position below the grid', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        0,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      ),
    ).toThrow(RangeError);
  });

  it('rejects a position above the grid', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        MAX_CARS + 1,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      ),
    ).toThrow(RangeError);
  });

  it('rejects a fractional position', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        2.5,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      ),
    ).toThrow(RangeError);
  });

  it('rejects a negative race time', () => {
    expect(() =>
      computeNeatRacingShapedFitness(1, -0.5, PROBE_LAP_PROGRESS_01, MAX_CARS),
    ).toThrow(RangeError);
  });

  it('rejects a non-finite race time', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        1,
        Number.POSITIVE_INFINITY,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      ),
    ).toThrow(RangeError);
  });

  it('rejects a lap progress below zero', () => {
    expect(() =>
      computeNeatRacingShapedFitness(1, PROBE_TIME_SECONDS, -0.01, MAX_CARS),
    ).toThrow(RangeError);
  });

  it('rejects a lap progress above one', () => {
    expect(() =>
      computeNeatRacingShapedFitness(1, PROBE_TIME_SECONDS, 1.01, MAX_CARS),
    ).toThrow(RangeError);
  });

  it('rejects a non-finite lap progress', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        1,
        PROBE_TIME_SECONDS,
        Number.NaN,
        MAX_CARS,
      ),
    ).toThrow(RangeError);
  });

  it('rejects a grid size below one', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        1,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        0,
      ),
    ).toThrow(RangeError);
  });

  it('rejects a fractional grid size', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        1,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        2.5,
      ),
    ).toThrow(RangeError);
  });

  it('rejects a non-finite grid size', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        1,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        Number.POSITIVE_INFINITY,
      ),
    ).toThrow(RangeError);
  });

  it('speaks the shared `neatRacing fitness shaping failed` RangeError voice naming the offending term', () => {
    expect(() =>
      computeNeatRacingShapedFitness(
        Number.NaN,
        PROBE_TIME_SECONDS,
        PROBE_LAP_PROGRESS_01,
        MAX_CARS,
      ),
    ).toThrow(
      /^neatRacing fitness shaping failed: position must be a finite number$/,
    );
  });
});

describe('commit-rule support (AC-029)', () => {
  it('scores a better-finishing variant strictly above its baseline under the commit rule', () => {
    const baselineScore = computeNeatRacingShapedFitness(
      3,
      PROBE_TIME_SECONDS,
      PROBE_LAP_PROGRESS_01,
      MAX_CARS,
    );
    const variantScore = computeNeatRacingShapedFitness(
      1,
      PROBE_TIME_SECONDS,
      PROBE_LAP_PROGRESS_01,
      MAX_CARS,
    );
    expect(variantScore).toBeGreaterThan(baselineScore);
  });
});

/* ------------------------------------------------------------------------- *
 * Test helpers (below the fold)
 * ------------------------------------------------------------------------- */

/**
 * Reports whether every value is strictly greater than its predecessor — the
 * strict-increase form of the E8 per-term monotonicity invariant.
 *
 * @param values - The fitness values along one fixed-other-terms sweep.
 * @returns True when the sweep is strictly increasing.
 */
function isStrictlyIncreasing(values: readonly number[]): boolean {
  return values.every(
    (value, index) => index === 0 || value > values[index - 1]!,
  );
}

/**
 * Reports whether every value is strictly smaller than its predecessor — the
 * strict-decrease form used by the inverted position and time sweeps.
 *
 * @param values - The fitness values along one fixed-other-terms sweep.
 * @returns True when the sweep is strictly decreasing.
 */
function isStrictlyDecreasing(values: readonly number[]): boolean {
  return values.every(
    (value, index) => index === 0 || value < values[index - 1]!,
  );
}
