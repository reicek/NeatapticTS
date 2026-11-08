/**
 * Pins the slice P2-S4-A contracts of the NeatRacing AI-mode goal state
 * machine (AC-033).
 *
 * The suite is PURE and node-lane: no jsdom, no mocks, no timers — the
 * goal machine is a deterministic realm, so every case is a plain
 * function call over hand-built fixtures.
 *
 * The pinned laws, restated per describe:
 * - the lap floor is a shared PRECONDITION, not a fourth condition
 *   (OQ-002): the machine reads the floor through the shared predicate's
 *   eligibility gate;
 * - COUNT-THEN-EVALUATE: the crossing that completes a lap counts BEFORE
 *   the predicate reads the counter, so the floor flips exactly at lap
 *   20;
 * - THE LAP INCREMENT IS THE HOST'S COMPLETED-CIRCUIT FACT: the banked
 *   increment is `lapsSinceLastCrossing` floored and clamped at 0 (never
 *   a bare +1), so a forward crossing that completed no circuit banks no
 *   lap and reports itself as `crossingWithoutLap`;
 * - NON-FINITE MEASUREMENTS ARE NORMALIZED AT THE MACHINE BOUNDARY: the
 *   machine normalizes every crossing measurement to finite-or-NaN
 *   before the shared predicate reads it, so a non-finite measurement
 *   can never promote;
 * - ONE EVALUATION PATH: only forward crossings count a lap and only
 *   forward crossings are evaluated;
 * - LEVEL 3 IS UNBOUNDED (LD-10): at the cap the machine still counts
 *   laps and still evaluates, but it never promotes and has no terminal
 *   state;
 * - PROMOTION IS A ROUND BOUNDARY (E6/D4): a promoted crossing re-bases
 *   the session onto the next rung with `sessionLaps` 0, so the lap
 *   floor re-applies per round.
 *
 * @module
 */

import type { NeatRacingObservationLevel } from './observation.assembler';
import { resolveOscillationPromotionNodeThreshold } from './oscillation-penalty.utils';
import {
  NEATRACING_AI_MIN_SESSION_LAPS,
  NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
  type NeatRacingDrivabilityMeasurement,
} from './promotion.rules';
import {
  createNeatRacingAiModeGoalSession,
  describeNeatRacingAiModeGoalPhase,
  recordNeatRacingAiModeGoalCrossing,
  NEATRACING_AI_MODE_START_LEVEL,
  type NeatRacingAiModeGoalCrossing,
  type NeatRacingAiModeGoalCrossingWithoutLapOutcome,
  type NeatRacingAiModeGoalLapCountedOutcome,
  type NeatRacingAiModeGoalPromotedOutcome,
  type NeatRacingAiModeGoalSession,
} from './ai-mode.goals';

/* ------------------------------------------------------------------------- *
 * The ladder rungs (the shared 1/2/3 observation levels).
 * ------------------------------------------------------------------------- */

const LEVEL_FIRST = 1;
const LEVEL_SECOND = 2;
const LEVEL_FINAL = 3;

/* ------------------------------------------------------------------------- *
 * The promotion thresholds — read from the single source of truth.
 * ------------------------------------------------------------------------- */

const THRESHOLD_LVL1 = resolveOscillationPromotionNodeThreshold(LEVEL_FIRST);
const ABOVE_LVL1 = THRESHOLD_LVL1 + 1;
const THRESHOLD_LVL2 = resolveOscillationPromotionNodeThreshold(LEVEL_SECOND);
const ABOVE_LVL2 = THRESHOLD_LVL2 + 1;

/* ------------------------------------------------------------------------- *
 * The OQ-002 lap floor.
 * ------------------------------------------------------------------------- */

const BELOW_FLOOR_LAPS = NEATRACING_AI_MIN_SESSION_LAPS - 1;
const AT_FLOOR_LAPS = NEATRACING_AI_MIN_SESSION_LAPS;

/**
 * A drivability measurement that passes the shared gate: 25 >= 0.75 * 30,
 * 0.2 <= 0.35 and 0.05 <= 0.1.
 */
const PASSING_DRIVABILITY: NeatRacingDrivabilityMeasurement = {
  meanForwardSpeed: 25,
  scriptedBaselineLapPace: 30,
  oscillationMetric: 0.2,
  offTrackFraction: 0.05,
};

/** The goal outcomes that carry a shared-predicate evaluation. */
type EvaluatedGoalOutcome =
  | NeatRacingAiModeGoalCrossingWithoutLapOutcome
  | NeatRacingAiModeGoalLapCountedOutcome
  | NeatRacingAiModeGoalPromotedOutcome;

/* ------------------------------------------------------------------------- *
 * The lap floor is the shared precondition (OQ-002).
 * ------------------------------------------------------------------------- */

describe('the lap floor is the shared precondition (OQ-002)', () => {
  it('boots the AI session on the first ladder rung', () => {
    expect(NEATRACING_AI_MODE_START_LEVEL).toBe(LEVEL_FIRST);
  });

  it('starts a fresh session below the floor with the full lap budget', () => {
    expect(
      describeNeatRacingAiModeGoalPhase(
        createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      ),
    ).toEqual({ kind: 'lapFloor', remainingLaps: NEATRACING_AI_MIN_SESSION_LAPS });
  });

  it('counts forward crossings below the floor but stays ineligible', () => {
    const outcomes = recordForwardCrossings(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      BELOW_FLOOR_LAPS,
    );

    expect(lastOutcome(outcomes).evaluation.eligible).toBe(false);
  });

  it('flips eligibility on the crossing that completes the floor lap', () => {
    const outcomes = recordForwardCrossings(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      AT_FLOOR_LAPS,
      buildCrossing({ nodeCount: THRESHOLD_LVL1 }),
    );

    expect(lastOutcome(outcomes).evaluation.eligible).toBe(true);
  });
});

/* ------------------------------------------------------------------------- *
 * Growth-goal continuation (promotion-threshold-driven).
 * ------------------------------------------------------------------------- */

describe('growth-goal continuation (promotion-threshold-driven)', () => {
  it('promotes on the first floor-meeting crossing with all three conditions met', () => {
    const outcomes = recordForwardCrossings(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      AT_FLOOR_LAPS,
    );

    expect(lastOutcome(outcomes).kind).toBe('promoted');
  });

  it('re-bases the session onto the fresh round after a promotion (E6)', () => {
    const outcomes = recordForwardCrossings(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      AT_FLOOR_LAPS,
    );

    expect(lastOutcome(outcomes).session).toEqual({
      level: LEVEL_SECOND,
      sessionLaps: 0,
    });
  });

  it('never promotes on an at-threshold node count (the threshold is strict)', () => {
    const outcomes = recordForwardCrossings(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      AT_FLOOR_LAPS,
      buildCrossing({ nodeCount: THRESHOLD_LVL1 }),
    );

    expect(lastOutcome(outcomes).evaluation.promoted).toBe(false);
  });

  it('keeps counting laps on the same rung when the evaluation fails', () => {
    const outcomes = recordForwardCrossings(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      AT_FLOOR_LAPS + 1,
      buildCrossing({ nodeCount: THRESHOLD_LVL1 }),
    );

    expect(lastOutcome(outcomes).session.sessionLaps).toBe(AT_FLOOR_LAPS + 1);
  });

  it('re-applies the lap floor on the fresh round after a promotion', () => {
    const outcomes = recordForwardCrossings(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      AT_FLOOR_LAPS,
    );

    expect(
      describeNeatRacingAiModeGoalPhase(lastOutcome(outcomes).session).kind,
    ).toBe('lapFloor');
  });

  it('names the next rung while the session chases its growth goal', () => {
    expect(describeNeatRacingAiModeGoalPhase(buildFloorReachedSession())).toEqual({
      kind: 'chasingGoal',
      promotionTargetLevel: LEVEL_SECOND,
    });
  });

  it('never promotes the level-2 round on an at-threshold node count', () => {
    const promotedToLevelTwo = recordForwardCrossings(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      AT_FLOOR_LAPS,
      buildCrossing({ nodeCount: ABOVE_LVL1 }),
    );
    const levelTwoRound = recordForwardCrossings(
      lastOutcome(promotedToLevelTwo).session,
      AT_FLOOR_LAPS,
      buildCrossing({ nodeCount: THRESHOLD_LVL2 }),
    );

    expect(lastOutcome(levelTwoRound).evaluation.promoted).toBe(false);
  });

  it('banks every completed circuit the host reports', () => {
    // The node count sits exactly on the threshold (the threshold is
    // strict), so the crossing banks its laps WITHOUT promoting: the
    // assertion reads the floor counter, not the round boundary.
    const outcome = recordNeatRacingAiModeGoalCrossing(
      buildFloorReachedSession(),
      buildCrossing({ lapsSinceLastCrossing: 2, nodeCount: THRESHOLD_LVL1 }),
    );

    expect(outcome.session.sessionLaps).toBe(AT_FLOOR_LAPS + 2);
  });

  it('banks no lap on a forward crossing that completed no circuit', () => {
    const outcome = recordNeatRacingAiModeGoalCrossing(
      buildFloorReachedSession(),
      buildCrossing({ lapsSinceLastCrossing: 0 }),
    );

    expect(outcome.kind).toBe('crossingWithoutLap');
  });

  it('keeps the lap counter untouched on a zero-circuit crossing', () => {
    const outcome = recordNeatRacingAiModeGoalCrossing(
      buildFloorReachedSession(),
      buildCrossing({ lapsSinceLastCrossing: 0 }),
    );

    expect(outcome.session.sessionLaps).toBe(AT_FLOOR_LAPS);
  });

  it('fails the circuit condition honestly on a zero-circuit crossing', () => {
    const outcome = recordNeatRacingAiModeGoalCrossing(
      buildFloorReachedSession(),
      buildCrossing({ lapsSinceLastCrossing: 0 }),
    );
    if (outcome.kind !== 'crossingWithoutLap') {
      throw new Error('a zero-circuit crossing must never bank a lap');
    }

    expect(outcome.evaluation.failedConditions).toContain('crossing');
  });

  it('can never promote on a non-finite measurement', () => {
    // `+Infinity` is ABOVE the node-count threshold, so the raw
    // measurement would have passed the shared comparison.
    const outcome = recordNeatRacingAiModeGoalCrossing(
      buildFloorReachedSession(),
      buildCrossing({
        lapsSinceLastCrossing: NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
        nodeCount: Number.POSITIVE_INFINITY,
      }),
    );
    if (outcome.kind === 'ignored') {
      throw new Error('a forward crossing must never be ignored');
    }

    expect(outcome.evaluation.promoted).toBe(false);
  });

  it('banks no lap on a non-finite circuit fact', () => {
    const outcome = recordNeatRacingAiModeGoalCrossing(
      buildFloorReachedSession(),
      buildCrossing({ lapsSinceLastCrossing: Number.NaN }),
    );

    expect(outcome.kind).toBe('crossingWithoutLap');
  });
});

/* ------------------------------------------------------------------------- *
 * Level 3 — the unbounded cap (until the user stops).
 * ------------------------------------------------------------------------- */

describe('level 3 — the unbounded cap (until the user stops)', () => {
  it('reaches the cap through chained ladder promotions', () => {
    expect(buildLevelThreeSession().level).toBe(LEVEL_FINAL);
  });

  it('reports the unbounded phase on the capped rung above the floor', () => {
    const outcomes = recordForwardCrossings(
      buildLevelThreeSession(),
      AT_FLOOR_LAPS,
      buildCrossing({ nodeCount: ABOVE_LVL2 }),
    );

    expect(describeNeatRacingAiModeGoalPhase(lastOutcome(outcomes).session)).toEqual({
      kind: 'unboundedCap',
    });
  });

  it('still evaluates at the cap and still never promotes', () => {
    const outcomes = recordForwardCrossings(
      buildLevelThreeSession(),
      AT_FLOOR_LAPS,
      buildCrossing({ nodeCount: ABOVE_LVL2 }),
    );

    expect(lastOutcome(outcomes).evaluation.ladderCapped).toBe(true);
  });

  it('stays on the top rung while the cap holds', () => {
    const outcomes = recordForwardCrossings(
      buildLevelThreeSession(),
      AT_FLOOR_LAPS,
      buildCrossing({ nodeCount: ABOVE_LVL2 }),
    );

    expect(lastOutcome(outcomes).session.level).toBe(LEVEL_FINAL);
  });

  it('re-applies the lap floor on the fresh capped round', () => {
    expect(describeNeatRacingAiModeGoalPhase(buildLevelThreeSession()).kind).toBe(
      'lapFloor',
    );
  });
});

/* ------------------------------------------------------------------------- *
 * Reverse crossings are ignored (ONE EVALUATION PATH).
 * ------------------------------------------------------------------------- */

describe('reverse crossings are ignored', () => {
  it('keeps the round untouched and never evaluates a reverse crossing', () => {
    const session = createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL);
    const outcome = recordNeatRacingAiModeGoalCrossing(
      session,
      buildCrossing({ forwardCrossing: false }),
    );

    expect(outcome).toEqual({ kind: 'ignored', session });
  });
});

/* ------------------------------------------------------------------------- *
 * Validation.
 * ------------------------------------------------------------------------- */

describe('validation', () => {
  it('rejects an out-of-band boot level', () => {
    // The cast is domain-safe on purpose: 7 is not a ladder rung.
    expect(() =>
      createNeatRacingAiModeGoalSession(7 as NeatRacingObservationLevel),
    ).toThrow(RangeError);
  });

  it('rejects a corrupted session record', () => {
    // The cast is domain-safe on purpose: a negative lap count cannot be
    // produced by the machine itself.
    expect(() =>
      recordNeatRacingAiModeGoalCrossing(
        { level: LEVEL_SECOND, sessionLaps: -1 } as NeatRacingAiModeGoalSession,
        buildCrossing(),
      ),
    ).toThrow(RangeError);
  });

  it('rejects a session record without a level', () => {
    // The cast is domain-safe on purpose: a record without a level
    // cannot be produced by the machine itself.
    expect(() =>
      recordNeatRacingAiModeGoalCrossing(
        {} as NeatRacingAiModeGoalSession,
        buildCrossing(),
      ),
    ).toThrow(RangeError);
  });

  it('rejects a session record without a lap counter', () => {
    // The cast is domain-safe on purpose: a record without a lap
    // counter cannot be produced by the machine itself.
    expect(() =>
      recordNeatRacingAiModeGoalCrossing(
        { level: LEVEL_FIRST } as Partial<NeatRacingAiModeGoalSession> as NeatRacingAiModeGoalSession,
        buildCrossing(),
      ),
    ).toThrow(RangeError);
  });
});

/* ------------------------------------------------------------------------- *
 * The fixture builders.
 * ------------------------------------------------------------------------- */

/**
 * Builds a forward crossing fixture over the passing level-1 base.
 */
function buildCrossing(
  overrides: Partial<NeatRacingAiModeGoalCrossing> = {},
): NeatRacingAiModeGoalCrossing {
  return {
    forwardCrossing: true,
    lapsSinceLastCrossing: NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
    nodeCount: ABOVE_LVL1,
    drivability: PASSING_DRIVABILITY,
    ...overrides,
  };
}

/**
 * Records a chain of forward crossings and returns every outcome.
 */
function recordForwardCrossings(
  session: NeatRacingAiModeGoalSession,
  count: number,
  crossing: NeatRacingAiModeGoalCrossing = buildCrossing(),
): EvaluatedGoalOutcome[] {
  const outcomes: EvaluatedGoalOutcome[] = [];
  let current: NeatRacingAiModeGoalSession = session;
  for (let index = 0; index < count; index += 1) {
    const outcome = recordNeatRacingAiModeGoalCrossing(current, crossing);
    if (outcome.kind === 'ignored') {
      throw new Error('a forward crossing must never be ignored');
    }
    outcomes.push(outcome);
    current = outcome.session;
  }
  return outcomes;
}

/**
 * Returns the last outcome of a crossing chain.
 */
function lastOutcome(outcomes: EvaluatedGoalOutcome[]): EvaluatedGoalOutcome {
  const outcome = outcomes.at(-1);
  if (outcome === undefined) {
    throw new Error('the crossing chain must not be empty');
  }
  return outcome;
}

/**
 * Builds a session that met the lap floor WITHOUT promoting: the node
 * count sits exactly on the threshold, so condition (b) never holds.
 */
function buildFloorReachedSession(): NeatRacingAiModeGoalSession {
  const crossing = buildCrossing({ nodeCount: THRESHOLD_LVL1 });
  let session: NeatRacingAiModeGoalSession = createNeatRacingAiModeGoalSession(
    NEATRACING_AI_MODE_START_LEVEL,
  );
  for (let index = 0; index < AT_FLOOR_LAPS; index += 1) {
    const outcome = recordNeatRacingAiModeGoalCrossing(session, crossing);
    if (outcome.kind === 'ignored') {
      throw new Error('a forward crossing must never be ignored');
    }
    session = outcome.session;
    if (describeNeatRacingAiModeGoalPhase(session).kind !== 'lapFloor') {
      return session;
    }
  }
  throw new Error('the lap floor was never reached');
}

/**
 * Chains two promotable rounds: level 1 -> level 2 -> level 3.
 */
function buildLevelThreeSession(): NeatRacingAiModeGoalSession {
  const firstRound = recordForwardCrossings(
    createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
    AT_FLOOR_LAPS,
    buildCrossing({ nodeCount: ABOVE_LVL1 }),
  );
  const promotedToLevelTwo = lastOutcome(firstRound);
  if (promotedToLevelTwo.kind !== 'promoted') {
    throw new Error('the first round must promote to level 2');
  }
  const secondRound = recordForwardCrossings(
    promotedToLevelTwo.session,
    AT_FLOOR_LAPS,
    buildCrossing({ nodeCount: ABOVE_LVL2 }),
  );
  const promotedToLevelThree = lastOutcome(secondRound);
  if (promotedToLevelThree.kind !== 'promoted') {
    throw new Error('the second round must promote to level 3');
  }
  return promotedToLevelThree.session;
}
