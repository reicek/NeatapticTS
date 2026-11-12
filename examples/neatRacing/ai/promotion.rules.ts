/**
 * NeatRacing AI promotion rules (slice `P2-S3-C`, AC-031).
 *
 * THE LAW (spec section 9.4, LD-10 / LD-30, E1 / E6): promotion is NOT
 * gameable — a big node count alone (or one lucky crossing) never promotes.
 * The promotion rule is a THREE-PART PURE PREDICATE; ALL THREE conditions
 * must hold AT THE CROSSING TICK:
 *
 * - (a) FORWARD FINISH-LINE CROSSING AFTER ONE COMPLETE CIRCUIT — the car
 *   crosses the start/finish line in the forward direction having completed at
 *   least one full lap since its last crossing (no back-and-forth over the
 *   line, no crossing on the first pass);
 * - (b) NODE COUNT ABOVE THE LEVEL'S THRESHOLD AT THE CROSSING TICK —
 *   `nodeCount > threshold(currentLevel)` (TP-07: 300 for the lvl 1 -> 2
 *   band, 600 for the lvl 2 -> 3 band);
 * - (c) DRIVABILITY GATE OVER THE CROSSING LAP — measured over the lap that
 *   JUST completed: the mean forward speed against the PINNED scripted-baseline
 *   lap pace, the TP-09 oscillation metric and the off-track fraction.
 *
 * THE LAP FLOOR IS A PRECONDITION, NOT A FOURTH CONDITION (OQ-002): the
 * predicate is evaluated at forward finish-line crossings ONLY once
 * `sessionLaps >= NEATRACING_AI_MIN_SESSION_LAPS = 20`. The predicate itself
 * stays three-part; the floor only decides WHETHER the predicate runs at all.
 *
 * THE NODE-THRESHOLD BAND IS LEVEL-RELATIVE AND CLIMBS ONE RUNG: the
 * threshold is the CURRENT level's promotion threshold — the very threshold
 * the oscillation law engages at (`resolveOscillationPromotionNodeThreshold`,
 * owned by `oscillation-penalty.utils.ts`) — so a promotion re-bases the floor
 * (300 -> 600) exactly as the oscillation law expects. A session therefore
 * climbs the ladder one rung per promotion (lvl 1 -> lvl 2 -> lvl 3) and the
 * ladder CAPS at level 3 (`NEATRACING_PROMOTION_MAX_LEVEL`), where no further
 * rung exists. The thresholds are NOT re-declared here (no dual path): this
 * module IMPORTS them.
 *
 * THE INTENTIONAL BOUNDARY ASYMMETRY (spec section 9.5 — do NOT "fix"): the
 * oscillation law engages at `nodeCount >= threshold` while promotion fires
 * at `nodeCount > threshold`. The STRICT `>` is required here so the penalty
 * is already ramping at the promotion instant (no step change in the score).
 *
 * THE PINNED BASELINE (N3): the drivability gate's speed reference is the
 * per-course profile-augmented scripted controller (AC-028's throttle profile
 * table) run on the SAME track/seed in a SCRIPTED-INPUT PROBE EPISODE — no
 * human input, no evolving agents. The probe's mean lap speed is a pure
 * deterministic function of the track tuple, so the gate is reproducible.
 *
 * THE ROUND BOUNDARY (E6, D4): a promotion is a ROUND BOUNDARY, never a
 * mid-session event. The crossing network becomes the SEED NETWORK of the
 * next round; the next round starts from a FRESH reproducibility tuple (new
 * `agentCount`, new `opponentSnapshot` matching the new level's car count);
 * the adaptation-engine state RESETS; and the innovation counter CARRIES with
 * the seed network. NO MID-SESSION CAR MATERIALIZATION: the cars of the new
 * level exist from the FIRST tick of the new round.
 *
 * THE TWO EXPLICIT D4 FIELDS: the fresh round tuple carries `seedNetwork`
 * (the crossing network's `CarGenome.serialize()` payload as a TYPED ARRAY
 * plus its NGE envelope — NOT `toJSON`, because typed-array serialization
 * avoids key-order sensitivity) and `carriedInnovationCounter` (the counter
 * value carried from the prior round). With those two named fields the round
 * N+1 session is REPLAYABLE FROM ITS OWN TUPLE ALONE.
 *
 * THE REALM: pure, headless and deterministic — no RNG, no wall-clock reads,
 * no DOM, no worker globals. The probe episode is the ONE place that drives
 * the real physics, and it drives it deterministically (a fixed spawn seed, a
 * scripted input, no random source), so the same track tuple always yields
 * the same baseline pace.
 *
 * @module
 */

import { NGE_DNA_SCHEMA_VERSION } from '../../../src/neat/nge-dna/neat.nge-dna.constants';
import type { NgeDnaCanonicalEnvelope } from '../../../src/neat/nge-dna/neat.nge-dna.types';
import Network from '../../../src/architecture/network';
import type {
  SplineSample,
  TrackSpec,
} from '../../racing_curriculum/track/track.generator.types';
import {
  createInitialPhysicsCarState,
  stepPhysics,
  type PhysicsCarState,
  type PhysicsCarSpawnPose,
} from '../physics/physics.step.service';
import {
  deriveNeatRacingCarNetworkChildSeed,
  serializeNeatRacingReproducibilityTuple,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';
import {
  computeNeatRacingScriptedControl,
  createNeatRacingScriptedControllerState,
} from './scripted.controller';
import {
  createOscillationPredicateState,
  resolveOscillationPromotionNodeThreshold,
  type OscillationPredicateState,
} from './oscillation-penalty.utils';
import {
  NEATRACING_TEAM_LAYOUTS,
  resolveNeatRacingObservationChannelCount,
  type NeatRacingObservationLevel,
} from './observation.assembler';
import { NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT } from './nge.controller';

/* ------------------------------------------------------------------------- *
 * Section 9.4 — the promotion law's named constants
 * ------------------------------------------------------------------------- */

/**
 * OQ-002 session-lap floor of the AI lane: the promotion predicate is
 * evaluated at forward finish-line crossings ONLY once the session has
 * completed this many laps. It is a PRECONDITION of evaluation, never a
 * fourth condition of the three-part predicate.
 */
export const NEATRACING_AI_MIN_SESSION_LAPS = 20;

/**
 * The highest rung of the AI level ladder (lvl 1 -> 2 -> 3). The ladder is
 * CAPPED here: at level 3 the session keeps evolving until the user stops it
 * or switches mode, so no promotion target exists beyond this level.
 */
export const NEATRACING_PROMOTION_MAX_LEVEL = 3;

/**
 * TP-17 drivability gate — the fraction of the scripted-baseline lap pace the
 * crossing lap's mean forward speed must reach. Arbitrary starting tuning
 * point (like TP-07), tuned in Phase 2.
 */
export const NEATRACING_DRIVABILITY_SPEED_FRACTION = 0.75;

/**
 * TP-17 drivability gate — the highest oscillation metric the crossing lap
 * may show. The metric keeps the TP-09 shape (the deadband
 * `RACING_OSCILLATION_MIN_MEAN_STEERING = 0.15` is preserved by the metric
 * detector). The bound sits BELOW the oscillation growth-commit gate
 * (`RACING_OSCILLATION_COMMIT_THRESHOLD = 0.5`), so a drivable car never
 * trips the commit boost. Arbitrary starting tuning point (like TP-07), tuned
 * in Phase 2.
 */
export const NEATRACING_DRIVABILITY_OSCILLATION_MAX = 0.35;

/**
 * TP-17 drivability gate — the highest fraction of the crossing lap the car
 * may spend off track. Arbitrary starting tuning point (like TP-07), tuned in
 * Phase 2.
 */
export const NEATRACING_DRIVABILITY_OFF_TRACK_MAX_FRACTION = 0.1;

/**
 * Condition (a)'s minimum circuit count: the car must have completed at least
 * this many full laps since its last finish-line crossing, so a back-and-forth
 * over the line and a crossing on the first pass never promote.
 */
export const NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS = 1;

/**
 * The probe episode's default tick budget (at the 60 Hz physics step: 200 s
 * of driving — generous enough for several laps of a long course).
 */
export const NEATRACING_PROBE_DEFAULT_MAX_TICKS = 12_000;

/**
 * The smallest number of spline samples a course must own for the probe
 * episode to attempt a lap (a shorter course has no meaningful circuit).
 */
export const NEATRACING_PROBE_MIN_SAMPLE_COUNT = 8;

/**
 * The sample-index tolerance of the probe's finish-line wrap detection: the
 * wrap counts when the car has reached the LAST samples of the course and
 * returns to the FIRST ones (the tolerance absorbs the sample-density
 * jitter of the closest-sample lookup).
 */
export const NEATRACING_PROBE_LAP_WRAP_TOLERANCE = 2;

/**
 * The probe episode's fixed spawn seed (the deterministic lane offset of the
 * spawned car; the probe never reads a random source).
 */
export const NEATRACING_PROBE_SPAWN_SEED = 1337;

/* ------------------------------------------------------------------------- *
 * Section 9.4 — the promotion predicate's public types
 * ------------------------------------------------------------------------- */

/**
 * The names of the predicate's three conditions (in evaluation order).
 */
export type NeatRacingPromotionConditionName =
  'crossing' | 'nodeCount' | 'drivability';

/**
 * The names of the drivability gate's three checks.
 */
export type NeatRacingDrivabilityCheckName =
  'speed' | 'oscillation' | 'offTrack';

/**
 * The drivability measurement of ONE completed lap (the crossing lap).
 */
export interface NeatRacingDrivabilityMeasurement {
  /** The mean forward speed over the crossing lap, in m/s. */
  readonly meanForwardSpeed: number;
  /**
   * The PINNED scripted-baseline lap pace for the same track/seed, in m/s
   * (the probe episode's mean lap speed).
   */
  readonly scriptedBaselineLapPace: number;
  /** The TP-09 oscillation metric of the crossing lap, in [0, 1]. */
  readonly oscillationMetric: number;
  /** The fraction of the crossing lap spent off track, in [0, 1]. */
  readonly offTrackFraction: number;
}

/**
 * The drivability gate's verdict over the crossing lap.
 */
export interface NeatRacingDrivabilityVerdict {
  /** True when ALL THREE drivability checks hold. */
  readonly passed: boolean;
  /** The required mean forward speed (the fraction of the baseline pace). */
  readonly requiredForwardSpeed: number;
  /** The speed check's verdict. */
  readonly speedOk: boolean;
  /** The oscillation check's verdict. */
  readonly oscillationOk: boolean;
  /** The off-track check's verdict. */
  readonly offTrackOk: boolean;
  /** The checks that did not hold (empty when the gate passed). */
  readonly failedChecks: readonly NeatRacingDrivabilityCheckName[];
}

/**
 * The predicate's per-condition verdicts at the crossing tick.
 */
export interface NeatRacingPromotionConditions {
  /** Condition (a): the forward finish-line crossing after one circuit. */
  readonly crossing: boolean;
  /** Condition (b): the node count above the level's threshold. */
  readonly nodeCount: boolean;
  /** Condition (c): the drivability gate over the crossing lap. */
  readonly drivability: boolean;
}

/**
 * The crossing tick's evaluation input.
 */
export interface NeatRacingPromotionEvaluationInput {
  /** The session level the evaluation runs at. */
  readonly level: number;
  /** The laps the session has completed so far (the OQ-002 floor). */
  readonly sessionLaps: number;
  /** The crossing network's node count at the crossing tick. */
  readonly nodeCount: number;
  /** True when the car crossed the finish line in the FORWARD direction. */
  readonly forwardCrossing: boolean;
  /** The full laps completed since the last finish-line crossing. */
  readonly lapsSinceLastCrossing: number;
  /** The crossing lap's drivability measurement. */
  readonly drivability: NeatRacingDrivabilityMeasurement;
}

/**
 * The predicate's verdict at the crossing tick.
 */
export interface NeatRacingPromotionEvaluation {
  /** The OQ-002 lap-floor precondition's verdict (the evaluation gate). */
  readonly eligible: boolean;
  /** True when the three conditions hold AND a further rung exists. */
  readonly promoted: boolean;
  /** The session level before the evaluation. */
  readonly fromLevel: number;
  /** The session level after the evaluation (unchanged when not promoted). */
  readonly toLevel: number;
  /** The rung the node-count band selects (the ladder cap applies). */
  readonly targetLevel: number;
  /** True when the ladder already sits on its highest rung. */
  readonly ladderCapped: boolean;
  /** The three conditions' per-condition verdicts. */
  readonly conditions: NeatRacingPromotionConditions;
  /** The conditions that did not hold (empty when all three held). */
  readonly failedConditions: readonly NeatRacingPromotionConditionName[];
}

/* ------------------------------------------------------------------------- *
 * Section 9.4 / E6 / D4 — the round boundary's public types
 * ------------------------------------------------------------------------- */

/**
 * The SEED NETWORK payload (D4): the crossing network's serialized weights as
 * a TYPED ARRAY plus the NGE envelope that travels with it.
 */
export interface NeatRacingSeedNetworkPayload {
  /** The crossing network's `CarGenome.serialize()` payload (Float32Array). */
  readonly weights: Float32Array;
  /** The NGE envelope that accompanies the serialized weights. */
  readonly envelope: NgeDnaCanonicalEnvelope;
}

/**
 * The fresh round's opponent snapshot: the new level and the car count the
 * new level materializes (the E6 "new `opponentSnapshot` matching the new
 * level's car count" requirement).
 */
export interface NeatRacingPromotionRoundSnapshot {
  /** The level the fresh round starts at. */
  readonly level: number;
  /** The number of cars materialized from the fresh round's first tick. */
  readonly carCount: number;
}

/**
 * The fresh round's reproducibility tuple: the AC-026 tuple PLUS the two
 * explicit D4 fields (`seedNetwork` and `carriedInnovationCounter`), so the
 * round N+1 session is replayable from its own tuple alone.
 */
export interface NeatRacingPromotionRoundTuple extends NeatRacingReproducibilityTuple {
  /** The fresh round's opponent snapshot (the new level's car count). */
  readonly opponentSnapshot: NeatRacingPromotionRoundSnapshot;
  /** The fresh round's seed network (D4). */
  readonly seedNetwork: NeatRacingSeedNetworkPayload;
  /**
   * The innovation counter carried from the prior round (D4). A top-level
   * identity field of the replay anchor: it must be a finite non-negative
   * integer (a non-finite / non-integer / negative counter is rejected with
   * a RangeError instead of silently serializing to `null`).
   */
  readonly carriedInnovationCounter: number;
}

/**
 * The state a fresh round starts from (E6): the adaptation-engine state is
 * RESET, the session counters restart at zero, and the innovation counter
 * CARRIES with the seed network.
 */
export interface NeatRacingRoundBoundaryState {
  /** The level the fresh round runs at. */
  readonly level: number;
  /** The laps completed in the fresh round (zero at the boundary). */
  readonly sessionLaps: number;
  /** The laps since the last finish-line crossing (zero at the boundary). */
  readonly lapsSinceLastCrossing: number;
  /** The carried innovation counter (NOT reset at the boundary). */
  readonly innovationCounter: number;
  /** The oscillation predicate's reset hysteresis state. */
  readonly oscillationState: OscillationPredicateState;
  /** The cars materialized from the fresh round's FIRST tick. */
  readonly materializedCarCount: number;
}

/**
 * The fresh round tuple's build input.
 */
export interface NeatRacingNextRoundTupleOptions {
  /** The prior round's reproducibility tuple (the fresh tuple's base). */
  readonly base: NeatRacingReproducibilityTuple;
  /** The level the fresh round starts at. */
  readonly targetLevel: number;
  /** The crossing network's seed-network payload (D4). */
  readonly seedNetwork: NeatRacingSeedNetworkPayload;
  /**
   * The innovation counter carried from the prior round (D4). Must be a
   * finite non-negative integer (rejected with a RangeError otherwise).
   */
  readonly carriedInnovationCounter: number;
}

/* ------------------------------------------------------------------------- *
 * Section 9.4 — the pinned scripted-baseline probe episode's public types
 * ------------------------------------------------------------------------- */

/**
 * The scripted-baseline probe episode's input.
 */
export interface NeatRacingScriptedBaselineProbeOptions {
  /** The course the probe drives (the same track/seed as the session). */
  readonly trackSpec: TrackSpec;
  /**
   * The probe's tick budget; defaults to
   * {@link NEATRACING_PROBE_DEFAULT_MAX_TICKS}.
   */
  readonly maxTicks?: number;
}

/**
 * The scripted-baseline probe episode's result.
 */
export interface NeatRacingScriptedBaselineProbeResult {
  /** The mean forward speed over the completed lap, in m/s (zero without a lap). */
  readonly meanLapSpeed: number;
  /** The ticks the completed lap took (zero without a lap). */
  readonly lapTicks: number;
  /** True when the probe completed one full circuit. */
  readonly completedLap: boolean;
  /** The ticks the probe consumed (the whole budget without a lap). */
  readonly consumedTicks: number;
}

/* ------------------------------------------------------------------------- *
 * Section 9.4 — the three-part promotion predicate
 * ------------------------------------------------------------------------- */

/**
 * Resolves the OQ-002 session-lap floor's eligibility at a crossing tick.
 *
 * The floor is a PRECONDITION of the evaluation, never a fourth condition of
 * the predicate: below the floor the predicate is not evaluated at all.
 *
 * @param sessionLaps - The laps the session has completed so far.
 * @returns True when the predicate may be evaluated at this crossing.
 *
 * @example
 * ```ts
 * isNeatRacingPromotionEvaluationEligible(20); // true
 * isNeatRacingPromotionEvaluationEligible(19); // false
 * ```
 */
export function isNeatRacingPromotionEvaluationEligible(
  sessionLaps: number,
): boolean {
  return sessionLaps >= NEATRACING_AI_MIN_SESSION_LAPS;
}

/**
 * Resolves the rung the node-count band selects for a session level.
 *
 * The band is LEVEL-RELATIVE: the threshold is the current level's promotion
 * threshold (`resolveOscillationPromotionNodeThreshold`), so a promotion
 * climbs exactly ONE rung and re-bases the floor (300 -> 600) exactly as the
 * oscillation law expects. The comparison is STRICT (`nodeCount > threshold`).
 * The ladder CAPS at {@link NEATRACING_PROMOTION_MAX_LEVEL}.
 *
 * @param level - The session level.
 * @param nodeCount - The crossing network's node count at the crossing tick.
 * @returns The level the session would climb to (the current level when the
 *   band is not crossed or the ladder is capped).
 *
 * @example
 * ```ts
 * resolveNeatRacingPromotionTargetLevel(1, 301); // 2
 * resolveNeatRacingPromotionTargetLevel(1, 300); // 1 (strict '>')
 * resolveNeatRacingPromotionTargetLevel(3, 900); // 3 (the ladder is capped)
 * ```
 */
export function resolveNeatRacingPromotionTargetLevel(
  level: number,
  nodeCount: number,
): number {
  const currentLevel = clampPromotionLevel(level);
  if (currentLevel >= NEATRACING_PROMOTION_MAX_LEVEL) {
    return currentLevel;
  }
  const threshold = resolveOscillationPromotionNodeThreshold(currentLevel);
  return nodeCount > threshold ? currentLevel + 1 : currentLevel;
}

/**
 * Evaluates the drivability gate (condition (c)) over the crossing lap.
 *
 * The gate holds when ALL THREE checks hold: the mean forward speed reaches
 * the named fraction of the PINNED scripted-baseline lap pace, the TP-09
 * oscillation metric stays at or below its bound, and the off-track fraction
 * stays at or below its bound. A MISSING baseline (a non-positive pinned
 * pace) can never satisfy the speed check — an unpinned baseline must not
 * promote.
 *
 * @param measurement - The crossing lap's drivability measurement.
 * @returns The gate's verdict.
 *
 * @example
 * ```ts
 * evaluateNeatRacingDrivabilityGate({
 *   meanForwardSpeed: 30,
 *   scriptedBaselineLapPace: 40,
 *   oscillationMetric: 0.1,
 *   offTrackFraction: 0,
 * }).passed; // true (30 >= 0.75 * 40)
 * ```
 */
export function evaluateNeatRacingDrivabilityGate(
  measurement: NeatRacingDrivabilityMeasurement,
): NeatRacingDrivabilityVerdict {
  const requiredForwardSpeed =
    measurement.scriptedBaselineLapPace * NEATRACING_DRIVABILITY_SPEED_FRACTION;
  const speedOk =
    measurement.scriptedBaselineLapPace > 0 &&
    measurement.meanForwardSpeed >= requiredForwardSpeed;
  const oscillationOk =
    measurement.oscillationMetric <= NEATRACING_DRIVABILITY_OSCILLATION_MAX;
  const offTrackOk =
    measurement.offTrackFraction <=
    NEATRACING_DRIVABILITY_OFF_TRACK_MAX_FRACTION;
  const failedChecks: NeatRacingDrivabilityCheckName[] = [];
  if (!speedOk) {
    failedChecks.push('speed');
  }
  if (!oscillationOk) {
    failedChecks.push('oscillation');
  }
  if (!offTrackOk) {
    failedChecks.push('offTrack');
  }
  return {
    passed: speedOk && oscillationOk && offTrackOk,
    requiredForwardSpeed,
    speedOk,
    oscillationOk,
    offTrackOk,
    failedChecks,
  };
}

/**
 * Evaluates the THREE-PART promotion predicate at a crossing tick.
 *
 * ALL THREE conditions must hold at the crossing tick: the forward
 * finish-line crossing after one complete circuit (a), the node count above
 * the level's threshold (b), and the drivability gate over the crossing lap
 * (c). The OQ-002 lap floor is a PRECONDITION of the evaluation, not a
 * fourth condition: below the floor the predicate never promotes.
 *
 * @param input - The crossing tick's evaluation input.
 * @returns The predicate's verdict, carrying the per-condition verdicts and
 *   the level the session ends up at.
 *
 * @example
 * ```ts
 * const evaluation = evaluateNeatRacingPromotion({
 *   level: 1,
 *   sessionLaps: 20,
 *   nodeCount: 320,
 *   forwardCrossing: true,
 *   lapsSinceLastCrossing: 1,
 *   drivability: {
 *     meanForwardSpeed: 30,
 *     scriptedBaselineLapPace: 40,
 *     oscillationMetric: 0.1,
 *     offTrackFraction: 0,
 *   },
 * });
 * evaluation.promoted; // true
 * evaluation.toLevel; // 2
 * ```
 */
export function evaluateNeatRacingPromotion(
  input: NeatRacingPromotionEvaluationInput,
): NeatRacingPromotionEvaluation {
  const level = clampPromotionLevel(input.level);
  const conditions = resolvePromotionConditions(input, level);
  const targetLevel = resolveNeatRacingPromotionTargetLevel(
    level,
    input.nodeCount,
  );
  const ladderCapped = level >= NEATRACING_PROMOTION_MAX_LEVEL;
  const eligible = isNeatRacingPromotionEvaluationEligible(input.sessionLaps);
  const promoted =
    eligible &&
    conditions.crossing &&
    conditions.nodeCount &&
    conditions.drivability &&
    !ladderCapped;
  return {
    eligible,
    promoted,
    fromLevel: level,
    toLevel: promoted ? targetLevel : level,
    targetLevel,
    ladderCapped,
    conditions,
    failedConditions: collectFailedConditions(conditions),
  };
}

/* ------------------------------------------------------------------------- *
 * Section 9.4 — the pinned scripted-baseline probe episode (N3)
 * ------------------------------------------------------------------------- */

/**
 * Runs the PINNED scripted-baseline probe episode for one course.
 *
 * The probe drives the per-course profile-augmented scripted controller
 * (AC-028's throttle profile table) on the SAME track/seed with a scripted
 * input only — no human input and no evolving agents — and measures the mean
 * forward speed of the lap it completes. The result is a pure deterministic
 * function of the track tuple, so the drivability gate's speed reference is
 * reproducible.
 *
 * The lap is detected on the course's own sample order: the car has reached
 * the LAST spline samples and returns to the FIRST ones (the start/finish
 * wrap). A course that is too short to own a circuit, or a probe that never
 * completes a lap, reports `completedLap: false` and a zero pace — an
 * unpinned baseline can never promote.
 *
 * @param options - The probe's course and tick budget.
 * @returns The probe's result.
 *
 * @example
 * ```ts
 * const probe = runNeatRacingScriptedBaselineProbeEpisode({ trackSpec });
 * probe.completedLap; // true on a course with a real circuit
 * ```
 */
export function runNeatRacingScriptedBaselineProbeEpisode(
  options: NeatRacingScriptedBaselineProbeOptions,
): NeatRacingScriptedBaselineProbeResult {
  const samples = options.trackSpec.splineSamples;
  const maxTicks = options.maxTicks ?? NEATRACING_PROBE_DEFAULT_MAX_TICKS;
  if (samples.length < NEATRACING_PROBE_MIN_SAMPLE_COUNT) {
    return {
      meanLapSpeed: 0,
      lapTicks: 0,
      completedLap: false,
      consumedTicks: 0,
    };
  }

  const controllerState = createNeatRacingScriptedControllerState();
  let car = createInitialPhysicsCarState(
    NEATRACING_PROBE_SPAWN_SEED,
    resolveProbeSpawnPose(samples),
  );
  let furthestSampleIndex = 0;
  let forwardSpeedSum = 0;

  for (let tick = 0; tick < maxTicks; tick += 1) {
    const control = computeNeatRacingScriptedControl(
      car,
      options.trackSpec,
      controllerState,
    );
    car = stepPhysics(car, control);
    forwardSpeedSum += car.speed;
    const sampleIndex = findClosestSplineSampleIndex(car, samples);
    if (
      isFinishLineCrossing(furthestSampleIndex, sampleIndex, samples.length)
    ) {
      const lapTicks = tick + 1;
      return {
        meanLapSpeed: forwardSpeedSum / lapTicks,
        lapTicks,
        completedLap: true,
        consumedTicks: lapTicks,
      };
    }
    furthestSampleIndex = Math.max(furthestSampleIndex, sampleIndex);
  }

  return {
    meanLapSpeed: 0,
    lapTicks: 0,
    completedLap: false,
    consumedTicks: maxTicks,
  };
}

/* ------------------------------------------------------------------------- *
 * Section 9.4 / E6 / D4 — the round boundary
 * ------------------------------------------------------------------------- */

/**
 * Resolves the number of cars a level materializes (the E2 team layouts are
 * the single source of truth: level 1 owns one car, level 2 two, level 3
 * five).
 *
 * @param level - The session level.
 * @returns The car's count materialized from the round's first tick.
 *
 * @example
 * ```ts
 * resolveNeatRacingLevelCarCount(1); // 1
 * resolveNeatRacingLevelCarCount(3); // 5
 * ```
 */
export function resolveNeatRacingLevelCarCount(level: number): number {
  return NEATRACING_TEAM_LAYOUTS[clampPromotionLevel(level)].length;
}

/**
 * Builds the seed-network payload (D4) of the fresh round.
 *
 * The weights are copied out of the caller's array, so the payload owns its
 * typed array and a later mutation of the source can never change the round
 * tuple.
 *
 * @param weights - The crossing network's `CarGenome.serialize()` payload.
 * @param envelope - The NGE envelope that travels with the weights.
 * @returns The seed-network payload.
 * @throws {RangeError} When the weights array is empty (a seed network
 *   without weights can never seed a round).
 * @throws {RangeError} When any weight is not a finite number (a network
 *   with a NaN or Infinity weight cannot seed a round either — the upstream
 *   `CarGenome.serialize()` does not validate, so the round boundary is the
 *   only gate over a corrupted genome).
 *
 * @example
 * ```ts
 * const payload = createNeatRacingSeedNetworkPayload(
 *   new Float32Array([0.5, -0.25]),
 *   envelope,
 * );
 * ```
 */
export function createNeatRacingSeedNetworkPayload(
  weights: Float32Array,
  envelope: NgeDnaCanonicalEnvelope,
): NeatRacingSeedNetworkPayload {
  if (weights.length === 0) {
    throw new RangeError(
      'neatRacing promotion rules failed: the seed network payload needs at least one weight',
    );
  }
  for (const weight of weights) {
    if (!Number.isFinite(weight)) {
      throw new RangeError(
        'neatRacing promotion rules failed: the seed network payload needs finite weights',
      );
    }
  }
  return { weights: weights.slice(), envelope };
}

/**
 * Builds the FRESH round tuple of the next round (E6, D4).
 *
 * The fresh tuple carries the prior round's seed, track and pack-schema
 * identity, a NEW `agentCount` and a NEW `opponentSnapshot` matching the new
 * level's car count, plus the two explicit D4 fields: the `seedNetwork` (the
 * crossing network's typed-array payload and its NGE envelope) and the
 * `carriedInnovationCounter`.
 *
 * @param options - The fresh tuple's build input.
 * @returns The fresh round tuple.
 * @throws {RangeError} When `carriedInnovationCounter` is not a finite
 *   non-negative integer (a top-level identity field of the replay anchor —
 *   never silently serialized to `null`).
 *
 * @example
 * ```ts
 * const nextRound = buildNeatRacingNextRoundTuple({
 *   base,
 *   targetLevel: 2,
 *   seedNetwork,
 *   carriedInnovationCounter: 42,
 * });
 * nextRound.agentCount; // 2 (the level-2 car count)
 * ```
 */
export function buildNeatRacingNextRoundTuple(
  options: NeatRacingNextRoundTupleOptions,
): NeatRacingPromotionRoundTuple {
  assertCarriedInnovationCounter(options.carriedInnovationCounter);
  const level = clampPromotionLevel(options.targetLevel);
  const carCount = resolveNeatRacingLevelCarCount(level);
  return {
    seed: options.base.seed,
    agentCount: carCount,
    packSchemaVersion: options.base.packSchemaVersion,
    opponentSnapshot: { level, carCount },
    trackId: options.base.trackId,
    featureFlags: options.base.featureFlags,
    seedNetwork: options.seedNetwork,
    carriedInnovationCounter: options.carriedInnovationCounter,
  };
}

/**
 * The frozen canonical NGE DNA envelope the promotion boundary ships with
 * the seed network (D4).
 *
 * Educational note — why this is pinned BY VALUE here: the round boundary
 * carries the crossing genome under the lane's CANONICAL `A.1.0` envelope, a
 * frozen substrate/policy document that never mutates between rounds. It is
 * deliberately a module-private constant rather than an import from the NGE
 * DNA runtime: the browser bundle must not pull the DNA engine's
 * `node:crypto` dependency (`neat.nge-dna.utils.ts` is its only consumer), so
 * the canonical document is reproduced here while the crypto-bearing runtime
 * stays out of the bundle. The schema version still comes from the repo's own
 * crypto-free constants module, so there is ONE source of truth for it.
 *
 * Educational note — why the fingerprint is the RUNTIME's own digest: the
 * envelope's `fingerprint` field is the SHA-256 hex digest of the canonical
 * envelope content (the repo's `computeFingerprint` over the key-sorted
 * canonical JSON), so the value pinned below is the digest the NGE runtime
 * itself produces for exactly THIS content
 * (`new NGE_DNA().toCanonical().fingerprint`). A label-shaped placeholder
 * would misreport the genome identity, so the real digest is pinned instead.
 */
const NEATRACING_NGE_PROMOTION_ENVELOPE: NgeDnaCanonicalEnvelope =
  Object.freeze({
    schemaVersion: NGE_DNA_SCHEMA_VERSION,
    compatibilityVersion: '1.0.0',
    encodingMode: 'lossless',
    fingerprint:
      'b3a7c3cd77b9cee5c440924602a3a04e3cae98bfd39f82bd02bdb6f3bffb4328',
    substrate: {
      budgetOverride: {
        maxEdges: 1024,
        maxNodes: 512,
      },
      dimensions: 3,
      normalization: 'unit-cube',
      zonePartition: {
        x: { count: 3 },
        y: { count: 3 },
        z: { count: 3 },
      },
    },
    reproductionPolicy: {
      assignedRegionStrategy: 'roundRobin',
      mode: 'sexual',
      modeIsEvolvable: false,
      parthenogenesisMutationRate: 0.01,
      polyandricDroneContributionFraction: 0.1,
      polyandricDroneCount: 2,
      queenBias: 1,
      seedPolicy: {
        siblingsDifferBySeed: true,
        twinsAllowed: false,
      },
    },
    rulePasses: [],
    cppnPrograms: [],
    moduleArchetypes: [],
  } satisfies NgeDnaCanonicalEnvelope);

/**
 * The agent-car index whose network seeds the promotion boundary's
 * `seedNetwork` payload (D4).
 *
 * Educational note — why index 1: car 0 of every round is the PLAYER's car
 * (it never carries a network), so the promoted round's FIRST adaptation car
 * is index 1 at every rung. The shipped genome therefore derives from the
 * SAME child-seed derivation the live session will rebuild that car with
 * (`deriveNeatRacingCarNetworkChildSeed` over the promoted tuple's seed), so
 * the boundary payload and the session-built network share one deterministic
 * source.
 */
const NEATRACING_PROMOTED_ROUND_SEED_AGENT_CAR_INDEX = 1;

/**
 * Serializes one network's connection weights into a fresh typed array.
 *
 * Educational note — why a manual positional walk: the D4 seed genome ships
 * as a `Float32Array` weight vector plus the canonical envelope (NOT the
 * `toJSON`-style compact serialization), so the boundary copies each
 * connection weight positionally. The fresh array detaches the payload from
 * the live network — later evolution of the source must never mutate the
 * shipped genome.
 *
 * @param network - The network whose connection weights serialize.
 * @returns The fresh weight vector (one entry per connection).
 */
function serializeNetworkWeightsToFloat32Array(network: Network): Float32Array {
  const weights = new Float32Array(network.connections.length);
  for (let index = 0; index < network.connections.length; index += 1) {
    weights[index] = network.connections[index].weight;
  }
  return weights;
}

/** The promoted round-tuple builder's input (E6/D4). */
export interface NeatRacingPromotedRoundTupleOptions {
  /** The just-completed round's reproducibility tuple (the fresh base). */
  readonly base: NeatRacingReproducibilityTuple;
  /** The rung the promotion lands on (clamped onto the ladder). */
  readonly targetLevel: number;
  /** The innovation counter carried from the prior round. */
  readonly carriedInnovationCounter: number;
}

/**
 * Builds the PROMOTED round's fresh tuple from the boundary's own evidence
 * (E6, D4, AC-031).
 *
 * Educational note — what makes this the promotion-flavored variant: unlike
 * `buildNeatRacingNextRoundTuple` (which consumes an already-built seed
 * payload), this builder DERIVES the seed network from the promoted round's
 * own fingerprint — the first agent car's network is constructed from the
 * promoted tuple's seed and the promoted LEVEL's observation width (the same
 * width the live session rebuilds that car with), then its weights ship with
 * the frozen canonical NGE envelope. The innovation counter stays an INPUT
 * (the realm never reads a global): the host reads the live counter at the
 * crossing and hands it in, so the boundary records exactly the value the
 * prior round's evolution reached.
 *
 * @param options - The promoted tuple's build input.
 * @returns The promoted round's fresh tuple (seed network and carried
 *   innovation counter included).
 * @throws {RangeError} When the derived network yields no finite weights
 *   (the seed payload gate) or the carried counter is malformed (delegated
 *   to `buildNeatRacingNextRoundTuple`).
 *
 * @example
 * ```ts
 * const promoted = buildNeatRacingPromotedRoundTuple({
 *   base,
 *   targetLevel: 2,
 *   carriedInnovationCounter: 42,
 * });
 * promoted.seedNetwork.weights; // Float32Array (the car-1 genome)
 * promoted.carriedInnovationCounter; // 42
 * ```
 */
export function buildNeatRacingPromotedRoundTuple(
  options: NeatRacingPromotedRoundTupleOptions,
): NeatRacingPromotionRoundTuple {
  const level = clampPromotionLevel(options.targetLevel);
  const seedNetwork = new Network(
    resolveNeatRacingObservationChannelCount(level),
    NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
    {
      seed: deriveNeatRacingCarNetworkChildSeed(
        options.base.seed,
        NEATRACING_PROMOTED_ROUND_SEED_AGENT_CAR_INDEX,
      ),
    },
  );
  return buildNeatRacingNextRoundTuple({
    base: options.base,
    targetLevel: level,
    seedNetwork: createNeatRacingSeedNetworkPayload(
      serializeNetworkWeightsToFloat32Array(seedNetwork),
      NEATRACING_NGE_PROMOTION_ENVELOPE,
    ),
    carriedInnovationCounter: options.carriedInnovationCounter,
  });
}

/**
 * Creates the state a fresh round starts from (E6).
 *
 * The adaptation-engine state RESETS (the oscillation predicate's hysteresis
 * state returns to its zeroed state), the session counters restart at zero
 * and the innovation counter CARRIES with the seed network. The fresh round's
 * cars are materialized FROM ITS FIRST TICK (no mid-session materialization).
 *
 * @param targetLevel - The level the fresh round runs at.
 * @param carriedInnovationCounter - The innovation counter carried from the
 *   prior round.
 * @returns The fresh round's boundary state.
 * @throws {RangeError} When `carriedInnovationCounter` is not a finite
 *   non-negative integer (`restoreNeatRacingRoundFromTuple` inherits this
 *   rejection because it routes through this function).
 *
 * @example
 * ```ts
 * const state = createNeatRacingRoundBoundaryState(2, 42);
 * state.sessionLaps; // 0 (reset)
 * state.innovationCounter; // 42 (carried)
 * ```
 */
export function createNeatRacingRoundBoundaryState(
  targetLevel: number,
  carriedInnovationCounter: number,
): NeatRacingRoundBoundaryState {
  assertCarriedInnovationCounter(carriedInnovationCounter);
  const level = clampPromotionLevel(targetLevel);
  return {
    level,
    sessionLaps: 0,
    lapsSinceLastCrossing: 0,
    innovationCounter: carriedInnovationCounter,
    oscillationState: createOscillationPredicateState(),
    materializedCarCount: resolveNeatRacingLevelCarCount(level),
  };
}

/**
 * Restores the fresh round's boundary state from the round's OWN tuple.
 *
 * This is the D4 replay path: the tuple alone (its seed, its level snapshot
 * and its carried innovation counter) rebuilds the round's boundary state, so
 * round N+1 never depends on state that only the prior round held.
 *
 * @param tuple - The fresh round tuple.
 * @returns The fresh round's boundary state.
 * @throws {RangeError} When the tuple carries no promotion opponent snapshot
 *   or when the snapshot's car count disagrees with the level's own layout.
 *
 * @example
 * ```ts
 * const restored = restoreNeatRacingRoundFromTuple(nextRound);
 * restored.level; // the fresh round's level
 * ```
 */
export function restoreNeatRacingRoundFromTuple(
  tuple: NeatRacingPromotionRoundTuple,
): NeatRacingRoundBoundaryState {
  const snapshot = tuple.opponentSnapshot;
  if (!isPromotionRoundSnapshot(snapshot)) {
    throw new RangeError(
      'neatRacing promotion rules failed: the round tuple carries no promotion opponent snapshot',
    );
  }
  const state = createNeatRacingRoundBoundaryState(
    snapshot.level,
    tuple.carriedInnovationCounter,
  );
  if (state.materializedCarCount !== snapshot.carCount) {
    throw new RangeError(
      'neatRacing promotion rules failed: the round tuple snapshot disagrees with the level layout',
    );
  }
  return state;
}

/**
 * Serializes the fresh round tuple into its canonical string shape.
 *
 * The AC-026 tuple's own canonical serial is REUSED for the base fields (one
 * serialization path, never a second one) and the two D4 fields are appended:
 * the `seedNetwork` (the typed-array weights encoded as a byte-exact hex
 * string plus the key-sorted NGE envelope — a typed array is NEVER `toJSON`ed,
 * because JSON key order would make the serial key-order sensitive) and the
 * `carriedInnovationCounter`.
 *
 * @param tuple - The fresh round tuple.
 * @returns The canonical serial (the round's replay identity anchor).
 * @throws {RangeError} When the tuple violates the AC-026 serialization
 *   contract, or when `carriedInnovationCounter` is not a finite non-negative
 *   integer (the counter is a top-level identity field of the anchor — the
 *   serial throws instead of silently carrying a `null` in its place).
 *
 * @example
 * ```ts
 * serializeNeatRacingPromotionRoundTuple(nextRound);
 * // '{"seed":1,...,"seedNetwork":{"weights":"0000803f",...},"carriedInnovationCounter":42}'
 * ```
 */
export function serializeNeatRacingPromotionRoundTuple(
  tuple: NeatRacingPromotionRoundTuple,
): string {
  assertCarriedInnovationCounter(tuple.carriedInnovationCounter);
  const base = JSON.parse(
    serializeNeatRacingReproducibilityTuple(tuple),
  ) as Record<string, unknown>;
  return JSON.stringify({
    ...base,
    seedNetwork: {
      weights: encodeFloat32ArrayAsHex(tuple.seedNetwork.weights),
      envelope: canonicalizeEnvelopeJsonValue(tuple.seedNetwork.envelope),
    },
    carriedInnovationCounter: tuple.carriedInnovationCounter,
  });
}

/* ------------------------------------------------------------------------- *
 * Section 9.4 — private helpers (below the fold)
 * ------------------------------------------------------------------------- */

/**
 * Validates the carried innovation counter (D4) of a fresh round.
 *
 * The counter is a top-level numeric identity field of the round tuple's
 * replay anchor, exactly parallel to the base tuple's `seed` (which the AC-026
 * serializer validates as finite): a non-finite, non-integer or negative
 * counter is rejected at the round boundary instead of being silently copied
 * into a `JSON.stringify` that would collapse it to `null` — leaving the
 * canonical serial and the restored round state disagreeing over the very
 * counter the D4 field exists to carry.
 *
 * @param carriedInnovationCounter - The counter to validate.
 * @throws {RangeError} When the counter is not a finite non-negative integer.
 */
function assertCarriedInnovationCounter(
  carriedInnovationCounter: number,
): void {
  if (
    !Number.isFinite(carriedInnovationCounter) ||
    !Number.isInteger(carriedInnovationCounter) ||
    carriedInnovationCounter < 0
  ) {
    throw new RangeError(
      'neatRacing promotion rules failed: the carried innovation counter must be a finite non-negative integer',
    );
  }
}

/**
 * Resolves the predicate's three per-condition verdicts at the crossing tick.
 *
 * @param input - The crossing tick's evaluation input.
 * @param level - The clamped session level.
 * @returns The three conditions' verdicts.
 */
function resolvePromotionConditions(
  input: NeatRacingPromotionEvaluationInput,
  level: number,
): NeatRacingPromotionConditions {
  const threshold = resolveOscillationPromotionNodeThreshold(level);
  return {
    crossing:
      input.forwardCrossing &&
      input.lapsSinceLastCrossing >= NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
    nodeCount: input.nodeCount > threshold,
    drivability: evaluateNeatRacingDrivabilityGate(input.drivability).passed,
  };
}

/**
 * Collects the conditions that did not hold at the crossing tick.
 *
 * @param conditions - The three conditions' verdicts.
 * @returns The names of the conditions that did not hold.
 */
function collectFailedConditions(
  conditions: NeatRacingPromotionConditions,
): NeatRacingPromotionConditionName[] {
  const failed: NeatRacingPromotionConditionName[] = [];
  if (!conditions.crossing) {
    failed.push('crossing');
  }
  if (!conditions.nodeCount) {
    failed.push('nodeCount');
  }
  if (!conditions.drivability) {
    failed.push('drivability');
  }
  return failed;
}

/**
 * Clamps a session level onto the ladder's own rungs (1, 2 or 3).
 *
 * @param level - The requested level.
 * @returns The clamped level.
 */
function clampPromotionLevel(level: number): NeatRacingObservationLevel {
  const truncated = Number.isFinite(level) ? Math.trunc(level) : 1;
  if (truncated <= 1) {
    return 1;
  }
  if (truncated >= NEATRACING_PROMOTION_MAX_LEVEL) {
    return NEATRACING_PROMOTION_MAX_LEVEL;
  }
  return 2;
}

/**
 * Resolves the probe's spawn pose: the first spline sample's position, facing
 * the second sample (the course's own forward direction).
 *
 * @param samples - The course's spline samples.
 * @returns The spawn pose.
 */
function resolveProbeSpawnPose(
  samples: readonly SplineSample[],
): PhysicsCarSpawnPose {
  const first = samples[0];
  const second = samples[1] ?? first;
  return {
    x: first.x,
    y: first.y,
    heading: Math.atan2(second.y - first.y, second.x - first.x),
  };
}

/**
 * Finds the spline sample the car is closest to (the course's own sample
 * order is the lap's progress measure).
 *
 * @param car - The car's current state.
 * @param samples - The course's spline samples.
 * @returns The index of the closest sample.
 */
function findClosestSplineSampleIndex(
  car: PhysicsCarState,
  samples: readonly SplineSample[],
): number {
  let closestIndex = 0;
  let closestDistanceSquared = Number.POSITIVE_INFINITY;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index];
    const deltaX = sample.x - car.x;
    const deltaY = sample.y - car.y;
    const distanceSquared = deltaX * deltaX + deltaY * deltaY;
    if (distanceSquared < closestDistanceSquared) {
      closestDistanceSquared = distanceSquared;
      closestIndex = index;
    }
  }
  return closestIndex;
}

/**
 * Resolves whether the car wrapped the start/finish line on the course's own
 * sample order (reached the last samples and returned to the first ones).
 *
 * @param furthestSampleIndex - The furthest sample index reached so far.
 * @param currentSampleIndex - The sample index the car is closest to now.
 * @param sampleCount - The course's sample count.
 * @returns True when the car wrapped the finish line.
 */
function isFinishLineCrossing(
  furthestSampleIndex: number,
  currentSampleIndex: number,
  sampleCount: number,
): boolean {
  const reachedLastSamples =
    furthestSampleIndex >=
    sampleCount - 1 - NEATRACING_PROBE_LAP_WRAP_TOLERANCE;
  const returnedToFirstSamples =
    currentSampleIndex <= NEATRACING_PROBE_LAP_WRAP_TOLERANCE;
  return reachedLastSamples && returnedToFirstSamples;
}

/**
 * Guards the fresh round tuple's opponent snapshot's shape.
 *
 * @param value - The tuple's opponent snapshot.
 * @returns True when the snapshot is a promotion round snapshot.
 */
function isPromotionRoundSnapshot(
  value: unknown,
): value is NeatRacingPromotionRoundSnapshot {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const candidate = value as { level?: unknown; carCount?: unknown };
  return (
    typeof candidate.level === 'number' &&
    typeof candidate.carCount === 'number'
  );
}

/**
 * Encodes a typed array as a byte-exact hex string (the typed-array
 * serialization D4 requires — never `toJSON`, whose key order would make the
 * serial key-order sensitive).
 *
 * @param weights - The typed array to encode.
 * @returns The hex string of the array's raw bytes.
 */
function encodeFloat32ArrayAsHex(weights: Float32Array): string {
  const bytes = new Uint8Array(
    weights.buffer as ArrayBuffer,
    weights.byteOffset,
    weights.byteLength,
  );
  let hex = '';
  for (const byte of bytes) {
    hex += byte.toString(16).padStart(2, '0');
  }
  return hex;
}

/**
 * Canonicalizes the NGE envelope into a key-sorted JSON value.
 *
 * The base tuple's canonicalizer is private to its own module, so this module
 * owns the envelope's canonical form (a NEW payload, never a second path over
 * the AC-026 tuple). Keys are sorted, so the serial can never depend on the
 * envelope's insertion order. The STRICTNESS CONTRACT is the base's own (the
 * one canonical serial may not carry two regimes):
 *
 * - Plain objects only (prototype `Object.prototype` or `null`): a Date, Map,
 *   Set or class instance has no canonical JSON form and is rejected instead
 *   of silently collapsing to `{}` — the base's `isPlainJsonObject` guard is
 *   mirrored one-for-one.
 * - Non-finite numbers are rejected (`numbers must be finite` — the base's
 *   own message) instead of being coerced to `null`, which would make the
 *   serial non-injective (NaN, null and an absent key could no longer be
 *   told apart by the replay anchor).
 * - Undefined-valued object keys DROP (JSON semantics — the base's
 *   `canonicalizeReproducibilityObject` drops them the same way) instead of
 *   materializing a `null` the base region would never have carried.
 *
 * @param value - The envelope (or any of its nested values).
 * @returns The canonical JSON value.
 * @throws {RangeError} When the envelope carries a value that has no
 *   canonical JSON form: a non-plain object (Date/Map/Set/class instance), a
 *   typed array outside the `seedNetwork.weights` field, a non-finite number
 *   (NaN/Infinity), or any other non-JSON material. Undefined-valued object
 *   keys drop rather than serialize.
 */
function canonicalizeEnvelopeJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((entry) => canonicalizeEnvelopeJsonValue(entry));
  }
  if (typeof value === 'object' && value !== null) {
    if (ArrayBuffer.isView(value)) {
      throw new RangeError(
        'neatRacing promotion tuple serialization failed: typed arrays belong in the seedNetwork.weights field',
      );
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new RangeError(
        'neatRacing promotion tuple serialization failed: the envelope carries a non-plain object (no canonical JSON form)',
      );
    }
    const canonical: Record<string, unknown> = {};
    for (const key of Object.keys(value).toSorted()) {
      const entry = (value as Record<string, unknown>)[key];
      if (entry === undefined) {
        continue; // JSON semantics: undefined-valued keys drop (base parity).
      }
      canonical[key] = canonicalizeEnvelopeJsonValue(entry);
    }
    return canonical;
  }
  if (
    value === null ||
    typeof value === 'boolean' ||
    typeof value === 'string'
  ) {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new RangeError(
        'neatRacing promotion tuple serialization failed: numbers must be finite',
      );
    }
    return value;
  }
  throw new RangeError(
    'neatRacing promotion tuple serialization failed: the envelope carries a value with no canonical JSON form',
  );
}
