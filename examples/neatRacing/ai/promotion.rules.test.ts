/**
 * Green tests for the NeatRacing promotion rules (`promotion.rules.ts` —
 * Step 03, slice P2-S3-C, TDD lane green-only).
 *
 * Contracts pinned here (AC-031):
 * - ONE test per condition (a/b/c) plus ONE all-conditions test: the
 *   promotion rule is a THREE-PART pure predicate (spec section 9.4, E1) and
 *   ALL THREE conditions must hold at the crossing tick — a big node count
 *   alone (or one lucky crossing) never promotes.
 * - The OQ-002 session-lap floor (`NEATRACING_AI_MIN_SESSION_LAPS = 20`) is
 *   a PRECONDITION of evaluation, NOT a fourth condition: below the floor
 *   the evaluation is ineligible while the three conditions stay verdicts of
 *   their own — the floor is never listed as a failed CONDITION.
 * - Condition (b) is LEVEL-RELATIVE and STRICT: the threshold is the current
 *   level's promotion threshold (TP-07 via the canonical
 *   `resolveOscillationPromotionNodeThreshold` table — no dual path), the
 *   comparison is `nodeCount > threshold`, exactly-at-threshold never
 *   promotes, and the ladder CAPS at `NEATRACING_PROMOTION_MAX_LEVEL`.
 * - The INTENTIONAL boundary asymmetry with the oscillation law (spec
 *   section 9.5 — do NOT "fix"): the law ENGAGES at `nodeCount >= threshold`
 *   while promotion fires at `nodeCount > threshold`, so the penalty is
 *   already ramping at the promotion instant.
 * - THE DRIVABILITY GATE (TP-17): the mean forward speed must reach the
 *   named fraction of the PINNED scripted-baseline lap pace, the oscillation
 *   metric (deadband `RACING_OSCILLATION_MIN_MEAN_STEERING = 0.15` kept in
 *   the detectors) must stay at or below its bound, and the off-track
 *   fraction must stay at or below its bound.
 * - THE PINNED BASELINE (N3): the scripted-baseline probe episode drives the
 *   per-course profile-augmented scripted controller on the SAME track/seed
 *   with a scripted input only, deterministic in the track tuple; a probe
 *   that never completes a lap pins a zero pace — an unpinned baseline can
 *   never promote.
 * - THE ROUND BOUNDARY (E6, D4): the fresh round tuple re-sizes
 *   `agentCount` to the target level's E2 team layout, swaps the
 *   `opponentSnapshot` to the fresh round's level/car count, carries the
 *   AC-026 identity fields verbatim, and carries the two EXPLICIT D4 named
 *   fields — `seedNetwork` (typed-array weights encoded as byte-exact hex
 *   plus the key-sorted NGE envelope, NEVER `toJSON`) and
 *   `carriedInnovationCounter` — so round N+1 is REPLAYABLE FROM ITS OWN
 *   TUPLE ALONE; the adaptation-engine state RESETS and the session-state
 *   counters restart at zero with NO mid-session car materialization.
 *
 * @module
 */

import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type { TrackSpec } from '../../racing_curriculum/track/track.generator.types';
import type { NgeDnaCanonicalEnvelope } from '../../../src/neat/nge-dna/neat.nge-dna.types';
import {
  serializeNeatRacingReproducibilityTuple,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';
import { NEATRACING_TEAM_LAYOUTS } from './observation.assembler';
import {
  createOscillationPredicateState,
  evaluateOscillationEngagement,
  NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
  NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3,
} from './oscillation-penalty.utils';
import {
  NEATRACING_AI_MIN_SESSION_LAPS,
  NEATRACING_DRIVABILITY_OFF_TRACK_MAX_FRACTION,
  NEATRACING_DRIVABILITY_OSCILLATION_MAX,
  NEATRACING_DRIVABILITY_SPEED_FRACTION,
  NEATRACING_PROMOTION_MAX_LEVEL,
  NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
  NEATRACING_PROBE_DEFAULT_MAX_TICKS,
  buildNeatRacingNextRoundTuple,
  createNeatRacingRoundBoundaryState,
  createNeatRacingSeedNetworkPayload,
  evaluateNeatRacingDrivabilityGate,
  evaluateNeatRacingPromotion,
  isNeatRacingPromotionEvaluationEligible,
  resolveNeatRacingLevelCarCount,
  resolveNeatRacingPromotionTargetLevel,
  restoreNeatRacingRoundFromTuple,
  runNeatRacingScriptedBaselineProbeEpisode,
  serializeNeatRacingPromotionRoundTuple,
  type NeatRacingPromotionEvaluationInput,
  type NeatRacingPromotionRoundTuple,
  type NeatRacingRoundBoundaryState,
} from './promotion.rules';

// ── Fixtures ─────────────────────────────────────────────────────────────────

/** The first level of the NeatRacing level ladder (TP-12). */
const LEVEL_FIRST = 1;

/** The second level of the level ladder (TP-12). */
const LEVEL_SECOND = 2;

/** The final level of the level ladder (TP-12 cap). */
const LEVEL_FINAL = 3;

/** The OQ-002 floor lap the precondition allows evaluation from. */
const AT_FLOOR_LAPS = NEATRACING_AI_MIN_SESSION_LAPS;

/** One lap below the OQ-002 floor. */
const BELOW_FLOOR_LAPS = NEATRACING_AI_MIN_SESSION_LAPS - 1;

/** The pinned scripted-baseline lap pace fixture (m/s). */
const BASELINE_LAP_PACE = 40;

/** The mean forward speed the speed fraction requires at the fixture pace. */
const REQUIRED_FORWARD_SPEED =
  BASELINE_LAP_PACE * NEATRACING_DRIVABILITY_SPEED_FRACTION;

/** A crossing-lap speed ABOVE the speed fraction (32 >= 0.75 * 40 = 30). */
const SPEED_ABOVE_FRACTION = REQUIRED_FORWARD_SPEED + 2;

/** A crossing-lap speed strictly BELOW the speed fraction. */
const SPEED_BELOW_FRACTION = REQUIRED_FORWARD_SPEED - 1;

/** A low oscillation metric (deadband-filtered gentle driving). */
const OSCILLATION_MILD = 0.1;

/** An oscillation metric above the drivability bound. */
const OSCILLATION_ABOVE_BOUND = NEATRACING_DRIVABILITY_OSCILLATION_MAX + 0.01;

/** A low off-track fraction. */
const OFF_TRACK_LOW = 0.05;

/** An off-track fraction above the drivability bound. */
const OFF_TRACK_ABOVE_BOUND = NEATRACING_DRIVABILITY_OFF_TRACK_MAX_FRACTION + 0.01;

/** The lvl 1 -> 2 band's TP-07 node threshold (300). */
const AT_LVL1_THRESHOLD = NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2;

/** One node above the lvl 1 -> 2 threshold. */
const ABOVE_LVL1_THRESHOLD = NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2 + 1;

/** The lvl 2 -> 3 band's TP-07 node threshold (600). */
const AT_LVL2_THRESHOLD = NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3;

/** One node above the lvl 2 -> 3 threshold. */
const ABOVE_LVL2_THRESHOLD = NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3 + 1;

/** The probe's tiny tick budget that cannot complete any lap. */
const TINY_PROBE_TICK_BUDGET = 16;

/** The seed-network weight fixture (three weights is a valid genome shape). */
const SEED_WEIGHTS = new Float32Array([0.5, -0.25, 0.75]);

/** The base-2-fixture byte-exact hex encoding of a Float32Array([1]). */
const FLOAT32_ONE_HEX = '0000803f';

/** The AC-026 packed-frame schema version the fixtures replay. */
const FIXTURE_PACK_SCHEMA_VERSION = 'neatracing-packed-v1';

/** The carried innovation counter fixture value (D4). */
const CARRIED_INNOVATION_COUNTER = 42;

/**
 * Builds a minimal canonical NGE DNA envelope fixture (the repo's
 * `neat.nge-juvenile.dna.test.ts` shape), so the round-boundary tests carry
 * a JSON-safe, key-sortable envelope without constructing full DNA.
 *
 * @returns The canonical envelope fixture.
 */
function buildCanonicalEnvelopeFixture(): NgeDnaCanonicalEnvelope {
  return {
    schemaVersion:
      'v1' as unknown as import('../../../src/neat/nge-dna/neat.nge-dna.types').NgeSchemaVersion,
    compatibilityVersion: '1.0.0',
    encodingMode: 'lossless',
    fingerprint: 'promotion-fixture-fingerprint',
    substrate: {
      dimensions: 3 as const,
      normalization: 'unit-cube' as const,
      zonePartition: {
        x: { count: 2 },
        y: { count: 2 },
        z: { count: 2 },
      },
    },
    reproductionPolicy: {
      mode: 'parthenogenesis' as const,
      parthenogenesisMutationRate: 0,
      polyandricDroneCount: 0,
      polyandricDroneContributionFraction: 0,
      queenBias: 1,
      assignedRegionStrategy: 'roundRobin' as const,
      modeIsEvolvable: false,
      seedPolicy: {
        siblingsDifferBySeed: false,
        twinsAllowed: false,
      },
    },
    rulePasses: [],
    cppnPrograms: [],
    moduleArchetypes: [],
  } as NgeDnaCanonicalEnvelope;
}

/** The shared canonical NGE envelope fixture. */
const ENVELOPE_FIXTURE = buildCanonicalEnvelopeFixture();

/**
 * Builds the prior round's reproducibility tuple (the fresh tuple's base).
 *
 * @returns The base tuple fixture.
 */
function buildBaseTupleFixture(): NeatRacingReproducibilityTuple {
  return {
    seed: 4242,
    agentCount: NEATRACING_TEAM_LAYOUTS[LEVEL_FIRST].length,
    packSchemaVersion: FIXTURE_PACK_SCHEMA_VERSION,
    opponentSnapshot: { level: LEVEL_FIRST, carCount: 1 },
    trackId: 'neatracing-fixtures',
    featureFlags: { replayD1: true },
  };
}

/**
 * Builds one clean drivability measurement for the crossing lap.
 *
 * @param overrides - The fields to override.
 * @returns The drivability measurement.
 */
function buildDrivabilityFixture(overrides?: {
  readonly meanForwardSpeed?: number;
  readonly scriptedBaselineLapPace?: number;
  readonly oscillationMetric?: number;
  readonly offTrackFraction?: number;
}): NeatRacingPromotionEvaluationInput['drivability'] {
  return {
    meanForwardSpeed: SPEED_ABOVE_FRACTION,
    scriptedBaselineLapPace: BASELINE_LAP_PACE,
    oscillationMetric: OSCILLATION_MILD,
    offTrackFraction: OFF_TRACK_LOW,
    ...overrides,
  };
}

/**
 * Builds one fully-promotable crossing-tick evaluation input.
 *
 * @param overrides - The fields to override.
 * @returns The evaluation input.
 */
function buildEvaluationFixture(overrides?: {
  readonly level?: number;
  readonly sessionLaps?: number;
  readonly nodeCount?: number;
  readonly forwardCrossing?: boolean;
  readonly lapsSinceLastCrossing?: number;
  readonly drivability?: NeatRacingPromotionEvaluationInput['drivability'];
}): NeatRacingPromotionEvaluationInput {
  return {
    level: LEVEL_FIRST,
    sessionLaps: AT_FLOOR_LAPS,
    nodeCount: ABOVE_LVL1_THRESHOLD,
    forwardCrossing: true,
    lapsSinceLastCrossing: NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
    drivability: buildDrivabilityFixture(),
    ...overrides,
  };
}

/**
 * Builds the fresh round tuple from the fixture state at a target level.
 *
 * @param targetLevel - The level the fresh round starts at.
 * @returns The fresh round tuple.
 */
function buildNextRoundTupleFixture(targetLevel: number) {
  return buildNeatRacingNextRoundTuple({
    base: buildBaseTupleFixture(),
    targetLevel,
    seedNetwork: createNeatRacingSeedNetworkPayload(
      SEED_WEIGHTS,
      ENVELOPE_FIXTURE,
    ),
    carriedInnovationCounter: CARRIED_INNOVATION_COUNTER,
  });
}

/** The real generated course the probe episode drives (deterministic). */
const REAL_TRACK_SPEC: TrackSpec = generateTrack({
  seed: 4242,
  layoutVersion: 1,
  sizeBucket: 'medium',
});

// ── The OQ-002 session-lap floor precondition ────────────────────────────────

describe('the OQ-002 session-lap floor precondition', () => {
  it('allows evaluation from the floor lap on and rejects the lap below it', () => {
    expect(isNeatRacingPromotionEvaluationEligible(AT_FLOOR_LAPS)).toBe(true);
    expect(isNeatRacingPromotionEvaluationEligible(BELOW_FLOOR_LAPS)).toBe(false);
  });

  it('never promotes below the floor even with a fully-promotable tick', () => {
    const evaluation = evaluateNeatRacingPromotion(
      buildEvaluationFixture({ sessionLaps: BELOW_FLOOR_LAPS }),
    );
    expect(evaluation.promoted).toBe(false);
  });

  it('keeps the floor OUT of the three-part predicate (not a fourth condition)', () => {
    const evaluation = evaluateNeatRacingPromotion(
      buildEvaluationFixture({ sessionLaps: BELOW_FLOOR_LAPS }),
    );
    expect(evaluation.eligible).toBe(false);
    expect(evaluation.failedConditions).not.toContain('crossing');
    expect(evaluation.conditions.crossing).toBe(true);
  });
});

// ── Condition (a) — the forward finish-line crossing ─────────────────────────

describe('condition (a) — the forward finish-line crossing', () => {
  it('rejects a back-and-forth over the line (reverse crossing)', () => {
    const evaluation = evaluateNeatRacingPromotion(
      buildEvaluationFixture({ forwardCrossing: false }),
    );
    expect(evaluation.conditions.crossing).toBe(false);
    expect(evaluation.failedConditions).toContain('crossing');
    expect(evaluation.promoted).toBe(false);
  });

  it('rejects a crossing on the first pass (no full circuit yet)', () => {
    const evaluation = evaluateNeatRacingPromotion(
      buildEvaluationFixture({ lapsSinceLastCrossing: 0 }),
    );
    expect(evaluation.conditions.crossing).toBe(false);
    expect(evaluation.failedConditions).toContain('crossing');
    expect(evaluation.promoted).toBe(false);
  });

  it('accepts one forward crossing after one complete circuit', () => {
    const evaluation = evaluateNeatRacingPromotion(
      buildEvaluationFixture({
        lapsSinceLastCrossing: NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
      }),
    );
    expect(evaluation.conditions.crossing).toBe(true);
    expect(evaluation.failedConditions).not.toContain('crossing');
  });
});

// ── Condition (b) — the STRICT level-relative node threshold ─────────────────

describe('condition (b) — the STRICT level-relative node threshold', () => {
  it('rejects exactly-at-threshold node counts (strict >)', () => {
    const evaluation = evaluateNeatRacingPromotion(
      buildEvaluationFixture({ nodeCount: AT_LVL1_THRESHOLD }),
    );
    expect(evaluation.conditions.nodeCount).toBe(false);
    expect(evaluation.failedConditions).toContain('nodeCount');
  });

  it('accepts one node above the threshold', () => {
    const evaluation = evaluateNeatRacingPromotion(
      buildEvaluationFixture({ nodeCount: ABOVE_LVL1_THRESHOLD }),
    );
    expect(evaluation.conditions.nodeCount).toBe(true);
    expect(evaluation.failedConditions).not.toContain('nodeCount');
  });

  it('stays asymmetric with the oscillation law at the threshold (do NOT fix)', () => {
    const engagement = evaluateOscillationEngagement(
      {
        nodeCount: AT_LVL1_THRESHOLD,
        oscillationMetric: 1,
        level: LEVEL_FIRST,
      },
      createOscillationPredicateState(),
    );
    const evaluation = evaluateNeatRacingPromotion(
      buildEvaluationFixture({ nodeCount: AT_LVL1_THRESHOLD }),
    );
    expect(engagement.engaged).toBe(true);
    expect(evaluation.conditions.nodeCount).toBe(false);
  });

  it('re-bases the strict threshold at lvl 2 (600)', () => {
    const atThreshold = evaluateNeatRacingPromotion(
      buildEvaluationFixture({ level: LEVEL_SECOND, nodeCount: AT_LVL2_THRESHOLD }),
    );
    const above = evaluateNeatRacingPromotion(
      buildEvaluationFixture({
        level: LEVEL_SECOND,
        nodeCount: ABOVE_LVL2_THRESHOLD,
      }),
    );
    expect(atThreshold.conditions.nodeCount).toBe(false);
    expect(above.conditions.nodeCount).toBe(true);
  });

  it('climbs exactly one rung and caps at the ladder end', () => {
    expect(resolveNeatRacingPromotionTargetLevel(LEVEL_FIRST, ABOVE_LVL1_THRESHOLD)).toBe(LEVEL_SECOND);
    expect(
      resolveNeatRacingPromotionTargetLevel(LEVEL_SECOND, ABOVE_LVL2_THRESHOLD),
    ).toBe(LEVEL_FINAL);
    expect(
      resolveNeatRacingPromotionTargetLevel(LEVEL_FINAL, 900),
    ).toBe(NEATRACING_PROMOTION_MAX_LEVEL);
  });
});

// ── Condition (c) — the drivability gate ─────────────────────────────────────

describe('condition (c) — the drivability gate', () => {
  it('rejects a mean forward speed below the pinned speed fraction', () => {
    const verdict = evaluateNeatRacingDrivabilityGate(
      buildDrivabilityFixture({ meanForwardSpeed: SPEED_BELOW_FRACTION }),
    );
    expect(verdict.speedOk).toBe(false);
    expect(verdict.failedChecks).toContain('speed');
    expect(verdict.passed).toBe(false);
  });

  it('accepts a speed exactly at the fraction boundary', () => {
    const verdict = evaluateNeatRacingDrivabilityGate(
      buildDrivabilityFixture({ meanForwardSpeed: REQUIRED_FORWARD_SPEED }),
    );
    expect(verdict.speedOk).toBe(true);
    expect(verdict.requiredForwardSpeed).toBe(REQUIRED_FORWARD_SPEED);
  });

  it('rejects an oscillation metric above the TP-17 bound', () => {
    const verdict = evaluateNeatRacingDrivabilityGate(
      buildDrivabilityFixture({ oscillationMetric: OSCILLATION_ABOVE_BOUND }),
    );
    expect(verdict.oscillationOk).toBe(false);
    expect(verdict.failedChecks).toContain('oscillation');
  });

  it('rejects an off-track fraction above the TP-17 bound', () => {
    const verdict = evaluateNeatRacingDrivabilityGate(
      buildDrivabilityFixture({ offTrackFraction: OFF_TRACK_ABOVE_BOUND }),
    );
    expect(verdict.offTrackOk).toBe(false);
    expect(verdict.failedChecks).toContain('offTrack');
  });

  it('never passes the speed check without a positive pinned baseline', () => {
    const verdict = evaluateNeatRacingDrivabilityGate(
      buildDrivabilityFixture({ scriptedBaselineLapPace: 0 }),
    );
    expect(verdict.speedOk).toBe(false);
    expect(verdict.passed).toBe(false);
  });
});

// ── All conditions together — the three-part predicate ───────────────────────

describe('all conditions together — the three-part predicate promotes', () => {
  it('promotes the lvl-1 crossing network to lvl 2 when a, b and c all hold', () => {
    const evaluation = evaluateNeatRacingPromotion(buildEvaluationFixture());
    expect(evaluation.promoted).toBe(true);
    expect(evaluation.toLevel).toBe(LEVEL_SECOND);
    expect(evaluation.failedConditions).toEqual([]);
  });

  it('keeps promoting through the ladder when the lvl-2 conditions hold', () => {
    const evaluation = evaluateNeatRacingPromotion(
      buildEvaluationFixture({
        level: LEVEL_SECOND,
        nodeCount: ABOVE_LVL2_THRESHOLD,
        drivability: buildDrivabilityFixture(),
      }),
    );
    expect(evaluation.promoted).toBe(true);
    expect(evaluation.toLevel).toBe(LEVEL_FINAL);
  });
});

// ── The pinned scripted-baseline probe episode (N3) ──────────────────────────

describe('the pinned scripted-baseline probe episode (N3)', () => {
  it('is deterministic in the track tuple (same input, same result)', () => {
    const first = runNeatRacingScriptedBaselineProbeEpisode({
      trackSpec: REAL_TRACK_SPEC,
    });
    const second = runNeatRacingScriptedBaselineProbeEpisode({
      trackSpec: REAL_TRACK_SPEC,
    });
    expect(second).toEqual(first);
  });

  it('completes exactly one circuit on a real course and pins a positive pace', () => {
    const probe = runNeatRacingScriptedBaselineProbeEpisode({
      trackSpec: REAL_TRACK_SPEC,
    });
    expect(probe.completedLap).toBe(true);
    expect(probe.lapTicks).toBeGreaterThan(0);
    expect(probe.meanLapSpeed).toBeGreaterThan(0);
  });

  it('reports the consumed ticks as the lap ticks when a lap completes', () => {
    const probe = runNeatRacingScriptedBaselineProbeEpisode({
      trackSpec: REAL_TRACK_SPEC,
    });
    expect(probe.consumedTicks).toBe(probe.lapTicks);
  });

  it('reports an unpinned (zero) pace when the budget runs out without a lap', () => {
    const probe = runNeatRacingScriptedBaselineProbeEpisode({
      trackSpec: REAL_TRACK_SPEC,
      maxTicks: TINY_PROBE_TICK_BUDGET,
    });
    expect(probe.completedLap).toBe(false);
    expect(probe.meanLapSpeed).toBe(0);
    expect(probe.consumedTicks).toBe(TINY_PROBE_TICK_BUDGET);
  });

  it('refuses to attempt a lap on a course too short to own a circuit', () => {
    const probe = runNeatRacingScriptedBaselineProbeEpisode({
      trackSpec: { ...REAL_TRACK_SPEC, splineSamples: REAL_TRACK_SPEC.splineSamples.slice(0, 2) },
    });
    expect(probe.completedLap).toBe(false);
    expect(probe.meanLapSpeed).toBe(0);
    expect(probe.consumedTicks).toBe(0);
  });

  it('matches the default tick budget constant', () => {
    expect(NEATRACING_PROBE_DEFAULT_MAX_TICKS).toBe(12_000);
  });
});

// ── The round boundary (E6/D4) ───────────────────────────────────────────────

describe('the round boundary (E6/D4) — fresh tuple, reset, carried counter', () => {
  it("re-sizes the fresh tuple's agentCount to the target level's layout", () => {
    expect(
      buildNeatRacingNextRoundTuple({
        base: buildBaseTupleFixture(),
        targetLevel: LEVEL_SECOND,
        seedNetwork: createNeatRacingSeedNetworkPayload(
          SEED_WEIGHTS,
          ENVELOPE_FIXTURE,
        ),
        carriedInnovationCounter: CARRIED_INNOVATION_COUNTER,
      }).agentCount,
    ).toBe(NEATRACING_TEAM_LAYOUTS[LEVEL_SECOND].length);
    expect(resolveNeatRacingLevelCarCount(LEVEL_FINAL)).toBe(5);
  });

  it('swaps the opponent snapshot to the fresh round level and car count', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    expect(nextRound.opponentSnapshot.level).toBe(LEVEL_SECOND);
    expect(nextRound.opponentSnapshot.carCount).toBe(
      NEATRACING_TEAM_LAYOUTS[LEVEL_SECOND].length,
    );
  });

  it('carries the AC-026 identity fields verbatim', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    expect(nextRound).toMatchObject({
      seed: 4242,
      packSchemaVersion: FIXTURE_PACK_SCHEMA_VERSION,
      trackId: 'neatracing-fixtures',
      featureFlags: { replayD1: true },
    });
  });

  it('carries the D4 seed network with a COPY of the typed-array weights', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    expect(nextRound.seedNetwork.weights).not.toBe(SEED_WEIGHTS);
    expect(nextRound.seedNetwork.weights).toEqual(SEED_WEIGHTS);
    expect(nextRound.seedNetwork.envelope).toBe(ENVELOPE_FIXTURE);
  });

  it('carries the D4 innovation counter from the prior round', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    expect(nextRound.carriedInnovationCounter).toBe(CARRIED_INNOVATION_COUNTER);
  });

  it('rejects a seed-network payload without weights', () => {
    expect(() =>
      createNeatRacingSeedNetworkPayload(new Float32Array([]), ENVELOPE_FIXTURE),
    ).toThrow(RangeError);
  });
});

describe('the fresh round boundary state (E6 — reset and carry)', () => {
  it('resets the session-state counters and materializes the fleet', () => {
    const state = createNeatRacingRoundBoundaryState(
      LEVEL_SECOND,
      CARRIED_INNOVATION_COUNTER,
    );
    expect(state.sessionLaps).toBe(0);
    expect(state.lapsSinceLastCrossing).toBe(0);
    expect(state.materializedCarCount).toBe(
      NEATRACING_TEAM_LAYOUTS[LEVEL_SECOND].length,
    );
  });

  it('resets the oscillation hysteresis and CARRIES the innovation counter', () => {
    const state = createNeatRacingRoundBoundaryState(
      LEVEL_SECOND,
      CARRIED_INNOVATION_COUNTER,
    );
    expect(state.oscillationState.ramp).toBe(0);
    expect(state.oscillationState.sustainedTicks).toBe(0);
    expect(state.innovationCounter).toBe(CARRIED_INNOVATION_COUNTER);
  });
});

describe('the D4 replay path — round N+1 from its own tuple alone', () => {
  it('rebuilds the boundary state from the round tuple with NO prior state', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    const restored = restoreNeatRacingRoundFromTuple(nextRound);
    const expected: NeatRacingRoundBoundaryState =
      createNeatRacingRoundBoundaryState(
        LEVEL_SECOND,
        CARRIED_INNOVATION_COUNTER,
      );
    expect(restored).toEqual(expected);
  });

  it('rejects a round tuple whose snapshot disagrees with the level layout', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    const tampered = {
      ...nextRound,
      opponentSnapshot: { level: LEVEL_SECOND, carCount: 5 },
    };
    expect(() => restoreNeatRacingRoundFromTuple(tampered)).toThrow(RangeError);
  });

  it('rejects a round tuple with no promotion opponent snapshot', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    const stripped = {
      ...nextRound,
      opponentSnapshot: { carriedRivals: [] },
    } as unknown as NeatRacingPromotionRoundTuple;
    expect(() => restoreNeatRacingRoundFromTuple(stripped)).toThrow(RangeError);
  });
});

describe('the canonical round-tuple serial (D4 — typed arrays, never toJSON)', () => {
  it('serializes the base fields through the ONE AC-026 serial path', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    const baseSerial = JSON.parse(
      serializeNeatRacingReproducibilityTuple(nextRound),
    ) as Record<string, unknown>;
    const fullSerial = JSON.parse(
      serializeNeatRacingPromotionRoundTuple(nextRound),
    ) as Record<string, unknown>;
    expect(String(fullSerial.seed)).toBe(String(baseSerial.seed));
    expect(String(fullSerial.packSchemaVersion)).toBe(
      String(baseSerial.packSchemaVersion),
    );
  });

  it('encodes the seed-network weights as byte-exact hex (not a toJSON call)', () => {
    const nextRound = buildNeatRacingNextRoundTuple({
      base: buildBaseTupleFixture(),
      targetLevel: LEVEL_SECOND,
      seedNetwork: createNeatRacingSeedNetworkPayload(
        new Float32Array([1]),
        ENVELOPE_FIXTURE,
      ),
      carriedInnovationCounter: CARRIED_INNOVATION_COUNTER,
    });
    const parsed = JSON.parse(
      serializeNeatRacingPromotionRoundTuple(nextRound),
    ) as { seedNetwork: { weights: string; envelope: unknown } };
    expect(parsed.seedNetwork.weights).toBe(FLOAT32_ONE_HEX);
    expect(parsed.seedNetwork.envelope).not.toBeUndefined();
  });

  it('appends the carried innovation counter to the serial', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    const serial = serializeNeatRacingPromotionRoundTuple(nextRound);
    const parsed = JSON.parse(serial) as Record<string, unknown>;
    expect(parsed.carriedInnovationCounter).toBe(CARRIED_INNOVATION_COUNTER);
    expect(typeof serial).toBe('string');
  });

  it('is byte-stable across repeated serializations of the same tuple', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    const first = serializeNeatRacingPromotionRoundTuple(nextRound);
    const second = serializeNeatRacingPromotionRoundTuple(nextRound);
    expect(second).toBe(first);
  });
});

// ── The replay guard helpers ─────────────────────────────────────────────────

describe('the replay guard helpers', () => {
  it('keeps the level ladder at 1..3 in the car-count resolver', () => {
    expect(resolveNeatRacingLevelCarCount(LEVEL_FIRST)).toBe(
      NEATRACING_TEAM_LAYOUTS[LEVEL_FIRST].length,
    );
    expect(resolveNeatRacingLevelCarCount(99)).toBe(
      NEATRACING_TEAM_LAYOUTS[LEVEL_FINAL].length,
    );
  });
});

// ── The carried-innovation-counter contract (fix packet F1) ──────────────────

describe('the carried-innovation-counter contract (D4 identity field)', () => {
  it('rejects a non-finite counter at the round boundary state', () => {
    expect(() =>
      createNeatRacingRoundBoundaryState(LEVEL_SECOND, Number.NaN),
    ).toThrow(RangeError);
    expect(() =>
      createNeatRacingRoundBoundaryState(LEVEL_SECOND, Number.POSITIVE_INFINITY),
    ).toThrow(RangeError);
  });

  it('rejects a negative or fractional counter at the round boundary state', () => {
    expect(() => createNeatRacingRoundBoundaryState(LEVEL_SECOND, -1)).toThrow(
      RangeError,
    );
    expect(() => createNeatRacingRoundBoundaryState(LEVEL_SECOND, 1.5)).toThrow(
      RangeError,
    );
  });

  it('rejects a non-finite counter when building the fresh round tuple', () => {
    expect(() =>
      buildNeatRacingNextRoundTuple({
        base: buildBaseTupleFixture(),
        targetLevel: LEVEL_SECOND,
        seedNetwork: createNeatRacingSeedNetworkPayload(
          SEED_WEIGHTS,
          ENVELOPE_FIXTURE,
        ),
        carriedInnovationCounter: Number.NaN,
      }),
    ).toThrow(RangeError);
  });

  it('serializes a valid counter in place and throws on a null one', () => {
    const nextRound = buildNextRoundTupleFixture(LEVEL_SECOND);
    expect(serializeNeatRacingPromotionRoundTuple(nextRound)).toContain(
      '"carriedInnovationCounter":42',
    );
    expect(() =>
      serializeNeatRacingPromotionRoundTuple({
        ...nextRound,
        carriedInnovationCounter: Number.NaN,
      }),
    ).toThrow(RangeError);
  });
});

// ── The strict envelope canonicalizer (fix packet F2) ────────────────────────

/** A class-instance fixture for the canonicalizer's plain-object guard. */
class FixtureClassInstance {}

/**
 * Builds a fresh round tuple carrying an envelope corrupted at one field.
 *
 * @param corruptFieldValue - The runtime value that must not serialize.
 * @returns The round tuple with the corrupted envelope.
 */
function buildTupleWithCorruptEnvelope(
  corruptFieldValue: unknown,
) {
  return buildNeatRacingNextRoundTuple({
    base: buildBaseTupleFixture(),
    targetLevel: LEVEL_SECOND,
    seedNetwork: createNeatRacingSeedNetworkPayload(
      SEED_WEIGHTS,
      {
        ...ENVELOPE_FIXTURE,
        fingerprint: corruptFieldValue,
      } as unknown as NgeDnaCanonicalEnvelope,
    ),
    carriedInnovationCounter: CARRIED_INNOVATION_COUNTER,
  });
}

describe('the strict envelope canonicalizer (never a silent null)', () => {
  it('throws when the envelope carries a NaN number', () => {
    expect(() =>
      serializeNeatRacingPromotionRoundTuple(
        buildTupleWithCorruptEnvelope(Number.NaN),
      ),
    ).toThrow(RangeError);
  });

  it('throws when the envelope carries Infinity', () => {
    expect(() =>
      serializeNeatRacingPromotionRoundTuple(
        buildTupleWithCorruptEnvelope(Number.POSITIVE_INFINITY),
      ),
    ).toThrow(RangeError);
  });

  it('throws for a Date, a Map and a class instance (one case each)', () => {
    expect(() =>
      serializeNeatRacingPromotionRoundTuple(
        buildTupleWithCorruptEnvelope(new Date('2026-11-07T00:00:00Z')),
      ),
    ).toThrow(RangeError);
    expect(() =>
      serializeNeatRacingPromotionRoundTuple(
        buildTupleWithCorruptEnvelope(new Map([['fingerprint', 'x']])),
      ),
    ).toThrow(RangeError);
    expect(() =>
      serializeNeatRacingPromotionRoundTuple(
        buildTupleWithCorruptEnvelope(new FixtureClassInstance()),
      ),
    ).toThrow(RangeError);
  });

  it('keeps the serial injective over the envelope region', () => {
    const nullEnvelopeTuple = buildTupleWithCorruptEnvelope(null);
    const nullSerial = serializeNeatRacingPromotionRoundTuple(nullEnvelopeTuple);
    expect(typeof nullSerial).toBe('string');
    expect(() =>
      serializeNeatRacingPromotionRoundTuple(
        buildTupleWithCorruptEnvelope(Number.NaN),
      ),
    ).toThrow(RangeError);
  });
});

// ── The finite-weights guard (fix packet F3) ─────────────────────────────────

describe('the finite-weights guard on the seed-network payload', () => {
  it('rejects a NaN weight', () => {
    expect(() =>
      createNeatRacingSeedNetworkPayload(
        new Float32Array([Number.NaN]),
        ENVELOPE_FIXTURE,
      ),
    ).toThrow(RangeError);
  });

  it('rejects an Infinity weight', () => {
    expect(() =>
      createNeatRacingSeedNetworkPayload(
        new Float32Array([Number.POSITIVE_INFINITY]),
        ENVELOPE_FIXTURE,
      ),
    ).toThrow(RangeError);
  });
});
