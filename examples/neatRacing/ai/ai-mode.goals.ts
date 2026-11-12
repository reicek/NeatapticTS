/**
 * NeatRacing AI-mode goal state machine (slice P2-S4-A, AC-033).
 *
 * THE LAW (spec section 9.4, LD-10 / LD-12): AI mode runs a MINIMUM of
 * `NEATRACING_AI_MIN_SESSION_LAPS` laps and then CONTINUES until the
 * current level's growth goal is met — that is the three-part promotion
 * predicate of `promotion.rules.ts` (the promotion thresholds) firing at a
 * forward crossing. Level 3 continues until the user stops the session.
 * Switching mode in EITHER direction restarts the race: the selector's
 * notify contract (`../host/mode-selector.ts`) IS the restart hook, and a
 * fresh AI goal session boots at the first rung of the ladder.
 *
 * (1) THE LAP FLOOR IS A PRECONDITION, NOT A FOURTH CONDITION (OQ-002):
 *     floor membership is read through the SHARED predicate's eligibility
 *     gate — this module NEVER re-declares the floor constant, the
 *     threshold table, or the conditions: `promotion.rules.ts` owns them
 *     and this machine only asks it.
 *
 * (2) COUNT-THEN-EVALUATE: the crossing that completes a lap increments
 *     `sessionLaps` BEFORE the predicate reads it, so the floor flips
 *     EXACTLY at the crossing that completes lap 20 (the promotion.rules
 *     GREEN fixture pins the promotion AT `sessionLaps` 20). The
 *     increment is the HOST's completed-circuit fact — the same
 *     `lapsSinceLastCrossing` field the promotion predicate's condition
 *     (a) reads — floored and clamped at 0 and NEVER a bare +1, so the
 *     OQ-002 floor counts COMPLETED laps.
 *
 * (3) ONE EVALUATION PATH: every FORWARD finish-line crossing routes
 *     through `evaluateNeatRacingPromotion`. Non-forward crossings never
 *     count a lap and never evaluate — the predicate is evaluated at
 *     FORWARD crossings only.
 *
 * (4) LEVEL 3 IS UNBOUNDED (LD-10): at the cap, crossings still count
 *     laps and the predicate verdict still runs (it reports `ladderCapped`
 *     and never promotes); the machine has NO terminal state — stopping is
 *     a HOST action (the user stops the session or switches mode).
 *
 * (5) PROMOTION IS A ROUND BOUNDARY (E6 / D4): a promoted crossing
 *     re-bases the goal session to the next rung with `sessionLaps` 0, so
 *     the lap floor RE-APPLIES per round. The D4 carrier state (seed
 *     network, innovation counter, adapter reset) belongs to the
 *     `promotion.rules.ts` round-tuple primitives consumed by the host
 *     wiring (P2-S4-C), NOT to this bookkeeping machine.
 *
 * (6) NON-FINITE MEASUREMENTS ARE NORMALIZED AT THE MACHINE BOUNDARY:
 *     `+Infinity` would PASS the shared predicate's comparisons (a
 *     `+Infinity` node count is `> threshold`, a `+Infinity` circuit
 *     count is `>= required`, a `+Infinity` lap pace is `> 0`), so the
 *     machine normalizes every measurement to finite-or-NaN FIRST: a
 *     non-finite measurement can never promote, while every forward
 *     crossing still routes through the ONE shared predicate.
 *
 * (7) THE REALM: pure, headless and deterministic — no RNG, no clock
 *     reads, no DOM, no worker globals. The machine never mutates its
 *     inputs; the counted and promoted outcomes carry a FRESH session
 *     record.
 *
 * @module
 */

import type { NeatRacingObservationLevel } from './observation.assembler';
import {
  evaluateNeatRacingPromotion,
  NEATRACING_AI_MIN_SESSION_LAPS,
  NEATRACING_PROMOTION_MAX_LEVEL,
  type NeatRacingDrivabilityMeasurement,
  type NeatRacingPromotionEvaluation,
} from './promotion.rules';

/**
 * The ladder rung a fresh AI-mode goal session boots at (AC-033).
 *
 * WHY: the AI lane's goal ladder starts at the FIRST rung — the same
 * numeric rung as the human lane's `NEATRACING_HUMAN_START_LEVEL` in
 * `../physics/race-rules.ts`, but deliberately NOT imported: the AI
 * lane's constants live here and the human lane's never leave
 * `race-rules.ts`.
 *
 * @example
 * ```ts
 * const session = createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL);
 * // { level: 1, sessionLaps: 0 }
 * ```
 */
export const NEATRACING_AI_MODE_START_LEVEL: NeatRacingObservationLevel = 1;

/**
 * The AI-mode goal session record — the whole state the machine owns.
 *
 * WHY: the session is the ONLY thing the goal state machine carries. It
 * is a plain, immutable snapshot: every transition produces a fresh
 * record instead of mutating the previous one, so a host can keep the old
 * record for a replay or an audit trail.
 *
 * @example
 * ```ts
 * const session = createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL);
 * const phase = describeNeatRacingAiModeGoalPhase(session);
 * // { kind: 'lapFloor', remainingLaps: 20 }
 * ```
 */
export interface NeatRacingAiModeGoalSession {
  /** Ladder rung the session evolves on (the shared 1/2/3 observation level). */
  readonly level: NeatRacingObservationLevel;
  /** Forward finish-line crossings completed in the CURRENT round (the OQ-002 floor counter). */
  readonly sessionLaps: number;
}

/**
 * The phase the goal session is currently in — what the session still owes
 * the user.
 *
 * WHY: the phase is the human- and HUD-readable summary of the session.
 * It never re-derives the promotion conditions: it only reports whether
 * the shared floor has been met and whether a rung above still exists.
 *
 * The three kinds:
 * - `lapFloor` — the session still owes the minimum lap budget
 *   (`remainingLaps > 0`); no promotion is even reachable yet.
 * - `chasingGoal` — the floor is met and a promotion rung exists: the
 *   session continues until the three-part predicate holds at a forward
 *   crossing.
 * - `unboundedCap` — the ladder top: the session keeps evolving until the
 *   user stops it or switches mode (no promotion target exists).
 *
 * @example
 * ```ts
 * describeNeatRacingAiModeGoalPhase({ level: 2, sessionLaps: 20 });
 * // { kind: 'chasingGoal', promotionTargetLevel: 3 }
 * ```
 */
export type NeatRacingAiModeGoalPhase =
  | { readonly kind: 'lapFloor'; readonly remainingLaps: number }
  | {
      readonly kind: 'chasingGoal';
      readonly promotionTargetLevel: NeatRacingObservationLevel;
    }
  | { readonly kind: 'unboundedCap' };

/**
 * The host-owned facts about one finish-line crossing.
 *
 * WHY: the goal machine does not detect crossings — the host's line
 * detector does. The machine only consumes the verdict plus the two
 * measurements the shared predicate needs, so the same lap counter and
 * the same drivability sample feed the goal decision and the promotion
 * decision.
 *
 * @example
 * ```ts
 * const outcome = recordNeatRacingAiModeGoalCrossing(session, {
 *   forwardCrossing: true,
 *   lapsSinceLastCrossing: 1,
 *   nodeCount: 301,
 *   drivability: { meanForwardSpeed: 25, scriptedBaselineLapPace: 30, oscillationMetric: 0.2, offTrackFraction: 0.05 },
 * });
 * ```
 */
export interface NeatRacingAiModeGoalCrossing {
  /** The detector's forward-crossing verdict at this tick. */
  readonly forwardCrossing: boolean;
  /** Full laps completed since the car's last finish-line crossing (host-owned lap counter). */
  readonly lapsSinceLastCrossing: number;
  /** The crossing network's live node count at the crossing tick. */
  readonly nodeCount: number;
  /** The crossing lap's drivability measurement (feeds the shared gate). */
  readonly drivability: NeatRacingDrivabilityMeasurement;
}

/**
 * The result of feeding one crossing to the goal machine.
 *
 * WHY: the outcome is a discriminated union so a host can branch on the
 * KIND without inspecting the evaluation: `ignored` (nothing happened),
 * `crossingWithoutLap` (a forward crossing that completed no circuit —
 * the round is untouched), `lapCounted` (the lap was banked, the round
 * continues) and `promoted` (the round ended, the next rung begins).
 *
 * @example
 * ```ts
 * if (outcome.kind === 'promoted') {
 *   session = outcome.session; // re-based on the next rung
 * }
 * ```
 */
export type NeatRacingAiModeGoalCrossingOutcome =
  | NeatRacingAiModeGoalIgnoredOutcome
  | NeatRacingAiModeGoalCrossingWithoutLapOutcome
  | NeatRacingAiModeGoalLapCountedOutcome
  | NeatRacingAiModeGoalPromotedOutcome;

/**
 * The outcome of a crossing that the machine ignores.
 *
 * WHY: a non-forward crossing is not a lap and not a promotion trigger,
 * so the round is returned untouched and NO evaluation was run (the
 * shared predicate is a FORWARD-crossing evaluator).
 *
 * @example
 * ```ts
 * recordNeatRacingAiModeGoalCrossing(session, { ...crossing, forwardCrossing: false });
 * // { kind: 'ignored', session }
 * ```
 */
export interface NeatRacingAiModeGoalIgnoredOutcome {
  readonly kind: 'ignored';
  readonly session: NeatRacingAiModeGoalSession;
}

/**
 * The outcome of a FORWARD crossing that completed NO full circuit.
 *
 * WHY: the host reported `lapsSinceLastCrossing` below one lap (or a
 * non-finite value), so the round is untouched — NO lap is banked — but
 * the shared predicate still ran (ONE evaluation path): condition (a)
 * reports its failure honestly, and such a crossing can never promote
 * without a completed circuit.
 *
 * @example
 * ```ts
 * recordNeatRacingAiModeGoalCrossing(session, { ...crossing, lapsSinceLastCrossing: 0 });
 * // { kind: 'crossingWithoutLap', session }
 * ```
 */
export interface NeatRacingAiModeGoalCrossingWithoutLapOutcome {
  readonly kind: 'crossingWithoutLap';
  readonly session: NeatRacingAiModeGoalSession;
  readonly evaluation: NeatRacingPromotionEvaluation;
}

/**
 * The outcome of a forward crossing that banked a lap without promoting.
 *
 * WHY: the lap is counted and the shared predicate still ran (and its
 * verdict is reported), but the round continues on the SAME rung — the
 * evaluation explains WHY the growth goal is still unmet.
 *
 * @example
 * ```ts
 * outcome.evaluation.eligible; // true — the lap floor is met
 * outcome.evaluation.promoted; // false — the threshold was not crossed
 * ```
 */
export interface NeatRacingAiModeGoalLapCountedOutcome {
  readonly kind: 'lapCounted';
  readonly session: NeatRacingAiModeGoalSession;
  readonly evaluation: NeatRacingPromotionEvaluation;
}

/**
 * The outcome of a forward crossing that promoted the session.
 *
 * WHY: a promotion ends the round: the session is re-based on the next
 * rung with `sessionLaps` 0 (the lap floor re-applies per round) and the
 * evaluation carries the promotion verdict the host needs to swap the
 * round tuple.
 *
 * @example
 * ```ts
 * outcome.session; // { level: 2, sessionLaps: 0 }
 * ```
 */
export interface NeatRacingAiModeGoalPromotedOutcome {
  readonly kind: 'promoted';
  readonly session: NeatRacingAiModeGoalSession;
  readonly evaluation: NeatRacingPromotionEvaluation;
}

/**
 * Creates a fresh AI-mode goal session on a ladder rung.
 *
 * WHY: a fresh AI goal session boots at the first rung with an empty lap
 * counter — the lap floor is owed in full. The boot level is validated
 * here so that out-of-band rungs fail loudly at the boundary instead of
 * corrupting a round later on.
 *
 * @param level - The ladder rung the session starts on.
 * @returns A fresh session record with `sessionLaps` 0.
 * @throws {RangeError} When the level is not an integer between 1 and
 *   `NEATRACING_PROMOTION_MAX_LEVEL` (out-of-band levels are rejected at
 *   runtime even though the TypeScript type narrows them, so JavaScript
 *   callers fail loudly).
 *
 * @example
 * ```ts
 * const session = createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL);
 * // { level: 1, sessionLaps: 0 }
 * ```
 */
export function createNeatRacingAiModeGoalSession(
  level: NeatRacingObservationLevel,
): NeatRacingAiModeGoalSession {
  assertAiModeGoalLevel(level);
  return { level, sessionLaps: 0 };
}

/**
 * Describes the phase the goal session is currently in.
 *
 * WHY: the phase answers "what does this session still owe?" — it reads
 * the SHARED floor constant (`NEATRACING_AI_MIN_SESSION_LAPS`) and the
 * SHARED cap (`NEATRACING_PROMOTION_MAX_LEVEL`) instead of re-declaring
 * them, so the machine and the predicate can never disagree about the
 * floor.
 *
 * @param session - The session record to describe.
 * @returns The current phase: `lapFloor` while the minimum lap budget is
 *   still owed, `chasingGoal` once the floor is met and a rung above
 *   exists, `unboundedCap` at the ladder top.
 * @throws {RangeError} When the session record is not a well-formed
 *   session (see {@link createNeatRacingAiModeGoalSession} for the same
 *   guard on the boot level).
 *
 * @example
 * ```ts
 * describeNeatRacingAiModeGoalPhase({ level: 1, sessionLaps: 0 });
 * // { kind: 'lapFloor', remainingLaps: 20 }
 * describeNeatRacingAiModeGoalPhase({ level: 1, sessionLaps: 20 });
 * // { kind: 'chasingGoal', promotionTargetLevel: 2 }
 * describeNeatRacingAiModeGoalPhase({ level: 3, sessionLaps: 20 });
 * // { kind: 'unboundedCap' }
 * ```
 */
export function describeNeatRacingAiModeGoalPhase(
  session: NeatRacingAiModeGoalSession,
): NeatRacingAiModeGoalPhase {
  assertAiModeGoalSession(session);
  if (session.sessionLaps < NEATRACING_AI_MIN_SESSION_LAPS) {
    return {
      kind: 'lapFloor',
      remainingLaps: NEATRACING_AI_MIN_SESSION_LAPS - session.sessionLaps,
    };
  }
  if (session.level < NEATRACING_PROMOTION_MAX_LEVEL) {
    // The branch proves the rung is 1 or 2, so the next rung is 2 or 3.
    const promotionTargetLevel: NeatRacingObservationLevel =
      session.level === 1 ? 2 : 3;
    return { kind: 'chasingGoal', promotionTargetLevel };
  }
  return { kind: 'unboundedCap' };
}

/**
 * Feeds one finish-line crossing to the goal state machine.
 *
 * WHY: this is the single decision point of the AI-mode goal. A
 * non-forward crossing is ignored outright (no lap, no evaluation — the
 * shared predicate is a FORWARD-crossing evaluator). A forward crossing
 * COUNTS the host's completed-circuit fact FIRST and only then asks the
 * shared predicate, so the eligibility floor flips exactly at the
 * crossing that completes lap 20.
 *
 * COUNT-THEN-EVALUATE: the banked increment is the HOST's
 * completed-circuit fact — `lapsSinceLastCrossing` floored and clamped
 * at 0 — never a bare +1. It is the SAME fact the shared predicate's
 * condition (a) consumes, so the machine's floor counter and the
 * predicate can never disagree: a forward crossing that completed no
 * circuit banks NO lap (the `crossingWithoutLap` outcome) and the
 * predicate reports condition (a) as failed.
 *
 * The session record is NEVER mutated: the `ignored` and
 * `crossingWithoutLap` outcomes return the INPUT session reference (the
 * round is untouched), the counted outcome spreads the input record into
 * a fresh one, and the promoted outcome re-bases onto the next rung with
 * `sessionLaps` 0 (the lap floor re-applies per round). The D4 carrier
 * state (seed network, innovation counter, adapter reset) is NOT part of
 * this bookkeeping — the host consumes the `promotion.rules.ts` round-tuple
 * primitives for it (P2-S4-C).
 *
 * The crossing measurements are NORMALIZED before the shared predicate
 * reads them: the machine runs every measurement through
 * `finiteOrNaN`, so an Infinity-laced measurement can never pass a
 * comparison it would otherwise pass (`+Infinity > threshold`,
 * `+Infinity >= 1` and `+Infinity >= required` were all true inside the
 * shared predicate). The normalization re-declares no threshold and
 * keeps the single evaluation path.
 *
 * @param session - The session record to advance.
 * @param crossing - The host-owned crossing facts.
 * @returns The outcome: `ignored` for a non-forward crossing,
 *   `crossingWithoutLap` for a forward crossing that completed no
 *   circuit, `lapCounted` when the round continues, `promoted` when the
 *   round ended and the next rung began.
 * @throws {RangeError} When the session record is not a well-formed
 *   session.
 *
 * @example
 * ```ts
 * const outcome = recordNeatRacingAiModeGoalCrossing(session, crossing);
 * if (outcome.kind === 'promoted') {
 *   session = outcome.session; // re-based on the next rung
 * }
 * ```
 */
export function recordNeatRacingAiModeGoalCrossing(
  session: NeatRacingAiModeGoalSession,
  crossing: NeatRacingAiModeGoalCrossing,
): NeatRacingAiModeGoalCrossingOutcome {
  assertAiModeGoalSession(session);
  if (!crossing.forwardCrossing) {
    return { kind: 'ignored', session };
  }
  // COUNT-THEN-EVALUATE: the banked increment is the HOST's
  // completed-circuit fact, never a bare +1: a forward crossing that
  // completed no circuit banks no lap. No floor/threshold/condition is
  // re-declared here.
  const normalizedLapsSinceLastCrossing: number = finiteOrNaN(
    crossing.lapsSinceLastCrossing,
  );
  const completedLaps: number = Number.isFinite(normalizedLapsSinceLastCrossing)
    ? Math.max(0, Math.floor(normalizedLapsSinceLastCrossing))
    : 0;
  const sessionLaps: number = session.sessionLaps + completedLaps;
  const evaluation: NeatRacingPromotionEvaluation = evaluateNeatRacingPromotion(
    {
      level: session.level,
      sessionLaps,
      nodeCount: finiteOrNaN(crossing.nodeCount),
      forwardCrossing: true,
      lapsSinceLastCrossing: normalizedLapsSinceLastCrossing,
      drivability: normalizeNeatRacingAiModeGoalDrivability(
        crossing.drivability,
      ),
    },
  );
  if (evaluation.promoted) {
    return {
      kind: 'promoted',
      session: { level: resolveLadderRung(evaluation.toLevel), sessionLaps: 0 },
      evaluation,
    };
  }
  // The zero-circuit path can never promote (condition (a) fails when the
  // normalized circuit fact is below one lap or NaN), so the promoted
  // branch above is unreachable there — it stays first for clarity.
  if (completedLaps === 0) {
    return { kind: 'crossingWithoutLap', session, evaluation };
  }
  return {
    kind: 'lapCounted',
    session: { ...session, sessionLaps },
    evaluation,
  };
}

/**
 * Normalizes a host measurement onto the shared predicate's own rejection
 * path.
 *
 * WHY: every check a NaN touches inside `promotion.rules.ts` (`>`,
 * `>=`, `<=`) evaluates false, so a normalized measurement can never
 * pass its check. Only NaN, -Infinity and +Infinity are normalized —
 * every finite value passes through verbatim, so the machine re-declares
 * no threshold and loses no precision.
 *
 * @param value - The host measurement to normalize.
 * @returns The measurement itself when it is finite, NaN otherwise.
 */
function finiteOrNaN(value: number): number {
  return Number.isFinite(value) ? value : Number.NaN;
}

/**
 * Normalizes a drivability measurement onto the shared predicate's own
 * rejection path.
 *
 * WHY: the shared gate compares each of the four fields (`>`, `>=`,
 * `<=`), and a NaN field always fails its own comparison. Normalizing
 * the whole measurement at the machine boundary leaves the shared gate's
 * arithmetic untouched while making a non-finite sample unable to
 * promote.
 *
 * @param measurement - The host's drivability measurement.
 * @returns The measurement with every field normalized to finite-or-NaN.
 */
function normalizeNeatRacingAiModeGoalDrivability(
  measurement: NeatRacingDrivabilityMeasurement,
): NeatRacingDrivabilityMeasurement {
  return {
    meanForwardSpeed: finiteOrNaN(measurement.meanForwardSpeed),
    scriptedBaselineLapPace: finiteOrNaN(measurement.scriptedBaselineLapPace),
    oscillationMetric: finiteOrNaN(measurement.oscillationMetric),
    offTrackFraction: finiteOrNaN(measurement.offTrackFraction),
  };
}

/**
 * Narrows a level number onto the ladder's own rungs (1, 2 or 3).
 *
 * WHY: the shared promotion evaluation reports its levels as plain
 * numbers, while the session record carries the ladder's rung type. This
 * narrowing keeps the module cast-free while still refusing an
 * out-of-band rung — a corrupted evaluation can never re-base a round
 * onto an unknown rung.
 *
 * @param level - The level number to narrow.
 * @returns The ladder rung the level names.
 * @throws {RangeError} When the level is outside the ladder.
 */
function resolveLadderRung(level: number): NeatRacingObservationLevel {
  if (level === 1) {
    return 1;
  }
  if (level === 2) {
    return 2;
  }
  if (level === 3) {
    return 3;
  }
  throw new RangeError(
    `neatracing ai-mode goals failed: the ladder rung must be an integer between 1 and ${NEATRACING_PROMOTION_MAX_LEVEL}`,
  );
}

/**
 * Guards a session record handed in from the outside.
 *
 * WHY: the goal machine is a pure realm, but a host can still hand it a
 * corrupted record. Rejecting a malformed session at the boundary keeps
 * the state machine total: every outcome it produces is derived from a
 * well-formed session.
 *
 * @param session - The candidate session record.
 * @throws {RangeError} When the record is null, not an object, an array,
 *   carries an out-of-band rung, or a negative/non-finite lap count.
 */
function assertAiModeGoalSession(
  session: unknown,
): asserts session is NeatRacingAiModeGoalSession {
  if (
    typeof session !== 'object' ||
    session === null ||
    Array.isArray(session)
  ) {
    throw new RangeError(
      'neatracing ai-mode goals failed: the session record must be an object',
    );
  }
  if (!('level' in session)) {
    throw new RangeError(
      'neatracing ai-mode goals failed: the session record must carry a level',
    );
  }
  assertAiModeGoalLevel(session.level);
  if (!('sessionLaps' in session)) {
    throw new RangeError(
      'neatracing ai-mode goals failed: the session record must carry a lap count',
    );
  }
  assertAiModeGoalLapCount(session.sessionLaps);
}

/**
 * Guards the ladder rung of a session record.
 *
 * WHY: the rung must be a real ladder rung. The TypeScript type already
 * narrows it, but a JavaScript caller can pass anything, so the runtime
 * check keeps out-of-band rungs loud.
 *
 * @param level - The candidate rung.
 * @throws {RangeError} When the rung is not an integer between 1 and
 *   `NEATRACING_PROMOTION_MAX_LEVEL`.
 */
function assertAiModeGoalLevel(
  level: unknown,
): asserts level is NeatRacingObservationLevel {
  if (
    typeof level !== 'number' ||
    !Number.isInteger(level) ||
    level < 1 ||
    level > NEATRACING_PROMOTION_MAX_LEVEL
  ) {
    throw new RangeError(
      `neatracing ai-mode goals failed: the ladder rung must be an integer between 1 and ${NEATRACING_PROMOTION_MAX_LEVEL}`,
    );
  }
}

/**
 * Guards the lap counter of a session record.
 *
 * WHY: the counter is the OQ-002 floor counter — a negative or fractional
 * value would silently misreport the remaining lap budget.
 *
 * @param sessionLaps - The candidate lap count.
 * @throws {RangeError} When the lap count is not a finite, non-negative
 *   integer.
 */
function assertAiModeGoalLapCount(
  sessionLaps: unknown,
): asserts sessionLaps is number {
  if (
    typeof sessionLaps !== 'number' ||
    !Number.isInteger(sessionLaps) ||
    sessionLaps < 0
  ) {
    throw new RangeError(
      'neatracing ai-mode goals failed: the session lap count must be a finite, non-negative integer',
    );
  }
}
