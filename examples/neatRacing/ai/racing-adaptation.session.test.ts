/**
 * Owner-local test surface for the neatRacing HOST ADAPTATION SESSION
 * (Phase 4, Step 23, slice `P4-S1-eval-wiring`).
 *
 * Educational note — what this file owns:
 * The RED home (`workers/simulation-worker/simulation-worker.control-seam.test.ts`)
 * pins the OBSERVABLE contract of the host boot: a PRODUCTION module OUTSIDE
 * the eval-worker subtree boots the eval-worker session. It deliberately does
 * NOT pin HOW the session is built, so the production wiring stays free.
 *
 * This file owns the session's OWN laws:
 * - the eval-worker BOOT (ONE session per AI-lane lifecycle, the BOUNDED
 *   per-car variant window and the section 9.6 shaped fitness scored through
 *   it),
 * - the LANE ISOLATION of the human sample (car 0 is SCORED but NEVER
 *   adapted),
 * - the AC-030 oscillation predicate (evaluated EXACTLY ONCE per adaptation
 *   tick, with its hysteresis: the hard node-count gate, the one-quarter
 *   grace ramp, the SYMMETRIC penalty on the baseline AND the variant score
 *   and the additive commit boost),
 * - the INJECTED fleet-control seam (the session publishes the LIVE adopted
 *   per-car network, never a cached snapshot),
 * - the D1 determinism law (the same tuple replays the same decisions and
 *   `Math.random` is never called),
 * - the ROUND-BOUNDARY law (a reset boots a FRESH eval-worker session,
 *   because the protocol's `stop` request is TERMINAL).
 *
 * THE IMPORT HAZARD: `../workers/eval-worker/eval-worker` binds
 * `self.onmessage` at MODULE SCOPE and creates a module-scope session at
 * import time, so under Jest (`testEnvironment: 'node'`) the import THROWS
 * when `self` is undefined. The house pattern (copied from the RED home)
 * installs a fake `globalThis.self` at MODULE scope and loads the session
 * module through a DYNAMIC import inside `beforeAll`.
 *
 * Every fixture is DETERMINISTIC: fixed seeds, the SHIPPED autopilot fixture
 * track, NO RNG, NO timers, NO wall-clock reads.
 *
 * @module
 */

import { afterAll, beforeAll, describe, expect, it, jest } from '@jest/globals';

import Network from '../../../src/architecture/network';
import type { TrackSpec } from '../../racing_curriculum/track/track.generator.types';
import { createTrackSpecForAutopilotFixture } from '../physics/ai.autopilot.fixtures';
import { createInitialPhysicsCarState } from '../physics/physics.step.service';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from '../physics/physics.step.service';
import { NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT } from '../workers/eval-worker/eval.protocol';
import { createNeatRacingFleetControl } from '../workers/simulation-worker/nge.fleet.control';
import {
  createNeatRacingXorShift32Stream,
  deriveNeatRacingAdaptationChildSeed,
  deriveNeatRacingCarNetworkChildSeed,
  NEATRACING_REPLAY_ADAPTATION_TICKS,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';
import { computeNeatRacingShapedFitness } from './fitness.shaping.utils';
import { NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT } from './nge.controller';
import {
  NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT,
} from './observation.assembler';
import {
  createOscillationPredicateState,
  evaluateOscillationEngagement,
  RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST,
  RACING_OSCILLATION_PENALTY_WEIGHT,
  resolveOscillationPromotionNodeThreshold,
} from './oscillation-penalty.utils';
import { resolveNeatRacingLevelCarCount } from './promotion.rules';
import {
  createRuntimeAdaptationEngine,
  type NeatRacingAdaptationCadenceOptions,
  type NeatRacingAdaptationTelemetry,
  type NeatRacingQualitySignal,
} from './runtime.adaptation';
import type {
  NeatRacingAdaptationCarInput,
  NeatRacingAdaptationSession,
  NeatRacingAdaptationSessionTickResult,
} from './racing-adaptation.session';

/* ------------------------------------------------------------------------- *
 * The fake worker scope (the house pattern of the RED home)
 * ------------------------------------------------------------------------- */

/**
 * Fake DedicatedWorkerGlobalScope the eval-worker module binds its message
 * adapter to at IMPORT time.
 */
const evalWorkerScope: {
  onmessage: unknown;
  postMessage: jest.Mock;
  addEventListener: jest.Mock;
} = {
  onmessage: undefined,
  postMessage: jest.fn(),
  addEventListener: jest.fn(),
};

/** Original `self` property descriptor captured before the test fakes it. */
const originalSelfDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'self',
);

Object.defineProperty(globalThis, 'self', {
  configurable: true,
  value: evalWorkerScope,
});

afterAll(() => {
  if (originalSelfDescriptor === undefined) {
    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: undefined,
    });
  } else {
    Object.defineProperty(globalThis, 'self', originalSelfDescriptor);
  }
});

/** The session module loaded in `beforeAll` once the fake scope has bound. */
type AdaptationSessionModule = typeof import('./racing-adaptation.session');

/** The session module, loaded through the dynamic import of `beforeAll`. */
let adaptationSessionModule: AdaptationSessionModule;

beforeAll(async () => {
  adaptationSessionModule = await import('./racing-adaptation.session');
});

/* ------------------------------------------------------------------------- *
 * The deterministic fixtures
 * ------------------------------------------------------------------------- */

/** The observation ladder level the fixture session observes at (solo rung). */
const SESSION_TEST_LEVEL = 1;

/** The fleet size of the fixture session (car 0 is the human sample). */
const SESSION_TEST_CAR_COUNT = 3;

/** The grid size the section 9.6 shaped fitness ranks the position against. */
const SESSION_TEST_MAX_CARS = 3;

/** The AC-026 tuple seed every derived stream of the fixture session hangs off. */
const SESSION_TEST_TUPLE_SEED = 42_424_242;

/**
 * The live node count of one agent car: the observation width of the level
 * plus the controller's output channels (never a magic number).
 */
const AGENT_CAR_NODE_COUNT =
  NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT +
  NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT;

/** The AC-026 reproducibility tuple of the fixture session. */
const SESSION_TEST_TUPLE: NeatRacingReproducibilityTuple = {
  seed: SESSION_TEST_TUPLE_SEED,
  agentCount: SESSION_TEST_CAR_COUNT,
  packSchemaVersion: 'adaptation-session-test-pack',
  opponentSnapshot: null,
  trackId: 'adaptation-session-test-track',
  featureFlags: undefined,
};

/**
 * Builds the fixture track: the SHIPPED autopilot fixture line (one straight
 * closed loop of 40 lane-center samples along +x at y = 0) plus the segment
 * the observation assembler and the fleet-control seam read.
 *
 * @returns The frozen fixture track geometry.
 */
function createSessionTestTrackSpec(): TrackSpec {
  return {
    seed: SESSION_TEST_TUPLE_SEED,
    layoutVersion: 1,
    sizeBucket: 'adaptation-session-fixture',
    segments: [{ startX: 0, startY: 0, endX: 39, endY: 0, width: 12 }],
    splineSamples: createTrackSpecForAutopilotFixture(),
    laneCount: 2,
  };
}

/** The frozen fixture track geometry. */
const SESSION_TEST_TRACK_SPEC = createSessionTestTrackSpec();

/**
 * Builds the fixture fleet: car 0 is the HUMAN car (no adopted network),
 * cars 1 and 2 are the AGENT cars the session adapts.
 *
 * @returns The three deterministic fixture car states.
 */
function createSessionTestFleet(): PhysicsCarState[] {
  return [
    createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 }),
    createInitialPhysicsCarState(43, { x: 14, y: 0, heading: 0 }),
    createInitialPhysicsCarState(44, { x: 26, y: 0, heading: 0 }),
  ];
}

/** The frozen fixture fleet. */
const SESSION_TEST_FLEET = createSessionTestFleet();

/**
 * The per-car inputs of one fixture tick: car 0 is the human sample, cars 1
 * and 2 are the agent cars.
 *
 * Educational note: the rolling score history is deliberately SHORT (two
 * entries) so the fixture tick stops at the engine's evidence gate and never
 * enters the growth path — the growth path consumes the ambient
 * `Math.random` (the D1 law keeps it out of the deterministic surface).
 */
const SESSION_TEST_CARS: readonly NeatRacingAdaptationCarInput[] = [
  {
    carIndex: 0,
    scoreHistory: [1, 0.5],
    position: 2,
    timeSeconds: 30,
    lapProgress01: 0.5,
  },
  {
    carIndex: 1,
    scoreHistory: [1, 0.5],
    position: 1,
    timeSeconds: 28,
    lapProgress01: 0.6,
  },
  {
    carIndex: 2,
    scoreHistory: [1, 0.5],
    position: 3,
    timeSeconds: 35,
    lapProgress01: 0.4,
  },
];

/**
 * The section 9.6 shaped fitness every fixture car must be scored with.
 *
 * @returns The expected per-car shaped fitness (indexed by car index).
 */
function computeSessionTestShapedFitness(): readonly number[] {
  return SESSION_TEST_CARS.map((car) =>
    computeNeatRacingShapedFitness(
      car.position,
      car.timeSeconds,
      car.lapProgress01,
      SESSION_TEST_MAX_CARS,
    ),
  );
}

/**
 * Creates the fixture adaptation session.
 *
 * @returns A fresh session bound to the fixture tuple and track.
 */
function createSessionTestSession(): NeatRacingAdaptationSession {
  return adaptationSessionModule.createNeatRacingAdaptationSession({
    tuple: SESSION_TEST_TUPLE,
    trackSpec: SESSION_TEST_TRACK_SPEC,
    level: SESSION_TEST_LEVEL,
    carCount: SESSION_TEST_CAR_COUNT,
    maxCars: SESSION_TEST_MAX_CARS,
  });
}

/**
 * Projects a published control into a comparable plain-data shape.
 *
 * Educational note: the fleet-control seam publishes a REUSED per-car buffer
 * (the zero-allocation law), so a consumer must copy the values BEFORE the
 * next tick overwrites them — the projection does exactly that.
 *
 * @param control - The published control (null when the car has none).
 * @returns The projected control (null when the car has none).
 */
function projectSessionTestControl(
  control: PhysicsStepInput | null,
): { brake: number; steer: number; throttle: number } | null {
  if (control === null) {
    return null;
  }
  return {
    brake: control.brake,
    steer: Number(control.steer.toFixed(6)),
    throttle: Number(control.throttle.toFixed(6)),
  };
}

/**
 * Builds the reference network of one fleet car.
 *
 * Educational note: the reference network is built exactly like the session
 * builds its own per-car brains — the shipped `Network` bootstrap over the
 * level-shaped observation width, CONSTRUCTED from the car's `car-<i>` child
 * seed (AC-026).
 *
 * @param networkSeed - The seed the reference network is CONSTRUCTED with.
 * @returns The reference network.
 */
function createSessionTestReferenceNetwork(networkSeed: number): Network {
  return new Network(
    NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
    NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
    { seed: networkSeed },
  );
}

/**
 * Builds a REFERENCE seam that publishes one reference network to a car.
 *
 * Educational note: the reference seam is the CONTROL of the session's own
 * seam — same tuple seed, same track geometry, same observation level, same
 * per-car child seed — so the two seams MUST publish the same control. A
 * session that cached a stale snapshot (or seeded the car from another
 * stream) would publish a different control.
 *
 * @param networkSeed - The seed the reference network is CONSTRUCTED with.
 * @returns The reference seam.
 */
function createSessionTestReferenceSeam(
  networkSeed: number,
): ReturnType<typeof createNeatRacingFleetControl> {
  const referenceNetwork = createSessionTestReferenceNetwork(networkSeed);
  return createNeatRacingFleetControl({
    tupleSeed: SESSION_TEST_TUPLE_SEED,
    trackSpec: SESSION_TEST_TRACK_SPEC,
    relay: { readCarController: () => referenceNetwork },
    level: SESSION_TEST_LEVEL,
  });
}

/* ------------------------------------------------------------------------- *
 * The AC-030 engine probe
 * ------------------------------------------------------------------------- */

/** The number of predicate evaluations the AC-030 probe runs. */
const OSCILLATION_PROBE_TICK_COUNT = 4;

/**
 * The oscillating score window of the AC-030 probe: three peaks and two
 * troughs, so the score-window detector reports the maximum metric of 1.
 */
const OSCILLATION_PROBE_SCORE_HISTORY: readonly number[] = [1, 0, 1, 0, 1];

/**
 * The CONSTANT score the symmetric-penalty probe drives the engine with: a
 * constant baseline exposes the engagement's penalty fraction directly.
 */
const OSCILLATION_PROBE_CONSTANT_SCORE = 10;

/** The fleet index the AC-030 probe derives its seeded stream from. */
const OSCILLATION_PROBE_CAR_INDEX = 1;

/**
 * The expected hysteresis ramp of the AC-030 probe: ONE grace step per
 * predicate evaluation, reaching the full engagement after exactly the
 * four-tick grace window.
 */
const OSCILLATION_PROBE_EXPECTED_RAMPS: readonly number[] = [
  0.25, 0.5, 0.75, 1,
];

/**
 * The expected penalty fractions of the AC-030 probe, computed with the
 * SAME expression the shared predicate uses (so the comparison is exact and
 * not a rounded approximation).
 */
const OSCILLATION_PROBE_EXPECTED_PENALTIES: readonly number[] =
  OSCILLATION_PROBE_EXPECTED_RAMPS.map(
    (ramp) => ramp * 1 * RACING_OSCILLATION_PENALTY_WEIGHT,
  );

/**
 * The expected commit boosts of the AC-030 probe, computed with the SAME
 * expression the shared predicate uses.
 */
const OSCILLATION_PROBE_EXPECTED_BOOSTS: readonly number[] =
  OSCILLATION_PROBE_EXPECTED_RAMPS.map(
    (ramp) => ramp * RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST,
  );

/**
 * Runs the AC-030 probe: ONE network ABOVE the level's promotion node
 * threshold, ONE oscillating score window and a cadence that never fires, so
 * every tick evaluates the shared predicate and returns without entering the
 * growth path.
 *
 * @param cadence - The cadence policy of the probe engine.
 * @param evaluateScore - The score hook of the probe engine.
 * @returns The telemetry of every probe tick.
 */
async function runSessionTestOscillationProbe(
  cadence: NeatRacingAdaptationCadenceOptions,
  evaluateScore: (
    network: Network,
    scoreHistory: readonly (number | NeatRacingQualitySignal)[],
  ) => number,
): Promise<readonly NeatRacingAdaptationTelemetry[]> {
  const engine = createRuntimeAdaptationEngine({
    random: createNeatRacingXorShift32Stream(
      deriveNeatRacingAdaptationChildSeed(
        SESSION_TEST_TUPLE_SEED,
        OSCILLATION_PROBE_CAR_INDEX,
      ),
    ),
    cadence,
    level: SESSION_TEST_LEVEL,
    evaluateScore,
    limits: { mutationCooldownTicks: 0, rollbackCooldownTicks: 0 },
  });

  const network = new Network(
    resolveOscillationPromotionNodeThreshold(SESSION_TEST_LEVEL),
    NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
    { seed: SESSION_TEST_TUPLE_SEED },
  );

  const telemetry: NeatRacingAdaptationTelemetry[] = [];
  for (let tick = 0; tick < OSCILLATION_PROBE_TICK_COUNT; tick += 1) {
    telemetry.push(
      await engine.adaptOnTick({
        tick,
        network,
        scoreHistory: OSCILLATION_PROBE_SCORE_HISTORY,
        completedLaps: 0,
      }),
    );
  }
  return telemetry;
}

/* ------------------------------------------------------------------------- *
 * The session's own laws
 * ------------------------------------------------------------------------- */

describe('neatRacing host adaptation session (P4-S1-eval-wiring)', () => {
  describe('the eval-worker boot and the shaped-fitness window', () => {
    it('boots ONE eval-worker session and scores the bounded per-car window through it', async () => {
      // Arrange: the fixture session (its factory boots the eval worker).
      const session = createSessionTestSession();

      // Act: run one adaptation tick.
      const results = await session.adaptOnTick({
        tick: 0,
        cars: SESSION_TEST_CARS,
      });

      // Assert: the eval worker is READY (the boot happened, and the round
      // was never cancelled).
      expect(session.snapshot().evalPhase).toBe('ready');

      // Assert: every car of the bounded window was scored with the
      // section 9.6 shaped fitness.
      expect(results.map((result) => result.shapedFitness)).toEqual(
        computeSessionTestShapedFitness(),
      );

      // Assert: the window carried ONE slot per car.
      expect(results).toHaveLength(SESSION_TEST_CAR_COUNT);
    });

    it('publishes the per-car shaped fitness and the live node counts in the snapshot', async () => {
      // Arrange: the fixture session.
      const session = createSessionTestSession();

      // Act: run one adaptation tick and read the plain-data snapshot.
      await session.adaptOnTick({ tick: 0, cars: SESSION_TEST_CARS });
      const snapshot = session.snapshot();

      // Assert: the snapshot publishes the tick and the per-car shaped
      // fitness of the window.
      expect(snapshot.tick).toBe(0);
      expect(snapshot.shapedFitness).toEqual(computeSessionTestShapedFitness());

      // Assert: the snapshot publishes the LIVE per-car node counts (the
      // human sample owns no network).
      expect(snapshot.nodeCounts).toEqual([
        0,
        AGENT_CAR_NODE_COUNT,
        AGENT_CAR_NODE_COUNT,
      ]);
    });

    it('scores the human sample but NEVER adapts it (lane isolation)', async () => {
      // Arrange: the fixture session.
      const session = createSessionTestSession();

      // Act: run one adaptation tick.
      const results = await session.adaptOnTick({
        tick: 0,
        cars: SESSION_TEST_CARS,
      });

      // Assert: the human sample is SCORED in the shaped-fitness window.
      expect(results[0].shapedFitness).toBe(
        computeSessionTestShapedFitness()[0],
      );

      // Assert: the human sample is NEVER adapted (no commit, no engine
      // reason — the lane-isolation reason).
      expect(results[0].reason).toBe('human_car_not_adapted');
      expect(results[0].committed).toBe(false);
    });
  });

  describe('the AC-030 oscillation law', () => {
    it('keeps a young network fully disengaged (the hard node-count gate)', async () => {
      // Arrange: the fixture session.
      const session = createSessionTestSession();

      // Act: run one adaptation tick and read the per-car engagements.
      await session.adaptOnTick({ tick: 0, cars: SESSION_TEST_CARS });
      const engagements = session.snapshot().oscillationEngagements;

      // Assert: the agent car's engagement is the shared predicate's verdict
      // for a network BELOW the level's promotion threshold (the hard gate).
      expect(engagements[1]).toEqual(
        evaluateOscillationEngagement(
          {
            nodeCount: AGENT_CAR_NODE_COUNT,
            oscillationMetric: 0,
            level: SESSION_TEST_LEVEL,
          },
          createOscillationPredicateState(),
        ),
      );

      // Assert: the law is fully disengaged (no penalty, no commit boost).
      expect(engagements[1]?.engaged).toBe(false);

      // Assert: the human sample owns no engagement (it is never adapted).
      expect(engagements[0]).toBeNull();
    });

    it('evaluates the shared predicate EXACTLY ONCE per adaptation tick (one grace step per tick)', async () => {
      // Act: run the probe (the cadence never fires, so ONLY the predicate
      // evaluation moves the law).
      const telemetry = await runSessionTestOscillationProbe(
        { mode: 'lap_boundary' },
        (network, scoreHistory) => scoreHistory.length,
      );

      // Assert: the hysteresis ramp advanced by EXACTLY ONE grace step per
      // tick (a double evaluation would ramp twice as fast).
      expect(
        telemetry.map((entry) => entry.oscillationEngagement.ramp),
      ).toEqual(OSCILLATION_PROBE_EXPECTED_RAMPS);

      // Assert: the penalty fraction followed the ramp (multiplicative law).
      expect(
        telemetry.map((entry) => entry.oscillationEngagement.penaltyFraction),
      ).toEqual(OSCILLATION_PROBE_EXPECTED_PENALTIES);

      // Assert: the additive commit boost followed the ramp as well.
      expect(
        telemetry.map((entry) => entry.oscillationEngagement.commitBoost),
      ).toEqual(OSCILLATION_PROBE_EXPECTED_BOOSTS);
    });

    it('counts the TP-09 grace window in predicate EVALUATIONS (never in raw simulation ticks)', async () => {
      // Act: run the probe.
      const telemetry = await runSessionTestOscillationProbe(
        { mode: 'lap_boundary' },
        (network, scoreHistory) => scoreHistory.length,
      );

      // Assert: the sustained-evaluation counter advanced once per
      // evaluation and CAPPED at the grace window.
      expect(
        telemetry.map(
          (entry) => entry.oscillationEngagement.state.sustainedTicks,
        ),
      ).toEqual([1, 2, 3, 4]);

      // Assert: the probe's cadence never fired, so the ramp advanced
      // PURELY through the predicate evaluations.
      expect(telemetry.map((entry) => entry.reason)).toEqual([
        'cadence_not_reached',
        'cadence_not_reached',
        'cadence_not_reached',
        'cadence_not_reached',
      ]);
    });

    it('applies the tick engagement SYMMETRICALLY to the baseline and the variant score', async () => {
      // Act: run the probe with a CONSTANT score, so both sides of the
      // comparison expose the engagement's penalty fraction.
      const telemetry = await runSessionTestOscillationProbe(
        { mode: 'every_tick' },
        () => OSCILLATION_PROBE_CONSTANT_SCORE,
      );
      const last = telemetry[telemetry.length - 1];
      const penaltyFraction = last.oscillationEngagement.penaltyFraction;

      // Assert: the law really was engaged on the probe's last tick.
      expect(penaltyFraction).toBeGreaterThan(0);

      // Assert: the BASELINE score of the tick carried the penalty.
      expect(last.scoreBefore).toBeCloseTo(
        OSCILLATION_PROBE_CONSTANT_SCORE * (1 - penaltyFraction),
      );

      // Assert: the VARIANT score of the SAME tick carried the SAME
      // penalty (the symmetric application law).
      expect(last.scoreAfter).toBeCloseTo(
        OSCILLATION_PROBE_CONSTANT_SCORE * (1 - penaltyFraction),
      );
    });
  });

  describe('the injected fleet-control seam', () => {
    it('publishes a control for every adopted agent car and none for the human sample', () => {
      // Arrange: the fixture session (the production boot injects the seam).
      const session = createSessionTestSession();

      // Act: ask the seam for every car's control.
      const humanControl = session.controller.computeControl(
        SESSION_TEST_FLEET,
        0,
      );
      const firstAgentControl = session.controller.computeControl(
        SESSION_TEST_FLEET,
        1,
      );
      const secondAgentControl = session.controller.computeControl(
        SESSION_TEST_FLEET,
        2,
      );

      // Assert: the human sample owns no adopted network, so the seam
      // publishes NO control for it (lane isolation).
      expect(humanControl).toBeNull();

      // Assert: both agent cars own an adopted network, so the seam
      // publishes a control for each.
      expect(firstAgentControl).not.toBeNull();
      expect(secondAgentControl).not.toBeNull();
    });

    it('drives the session LIVE per-car network through the seam (the car-<i> child seed)', () => {
      // Arrange: the fixture session and a REFERENCE seam over an
      // independently built network carrying car 1's child seed.
      const session = createSessionTestSession();
      const sameSeedSeam = createSessionTestReferenceSeam(
        deriveNeatRacingCarNetworkChildSeed(SESSION_TEST_TUPLE_SEED, 1),
      );

      // Act: publish car 1's control from the session seam and from the
      // reference seam.
      const sessionControl = projectSessionTestControl(
        session.controller.computeControl(SESSION_TEST_FLEET, 1),
      );
      const sameSeedControl = projectSessionTestControl(
        sameSeedSeam.computeControl(SESSION_TEST_FLEET, 1),
      );

      // Assert: the session's seam drives the SAME live per-car network the
      // reference seam drives (a stale or differently-seeded snapshot would
      // publish a different control).
      expect(sessionControl).toEqual(sameSeedControl);

      // Assert: the published control honours the mapping law (the brake is
      // fixed at zero and the throttle axis stays inside its physics domain).
      expect(sessionControl?.brake).toBe(0);
      expect(sessionControl?.throttle).toBeGreaterThanOrEqual(0);
      expect(sessionControl?.throttle).toBeLessThanOrEqual(1);
    });

    it('republishes the seam control on every tick (the relay is read, never cached)', () => {
      // Arrange: the fixture session and the first published control.
      const session = createSessionTestSession();
      const firstControl = projectSessionTestControl(
        session.controller.computeControl(SESSION_TEST_FLEET, 1),
      );

      // Act: publish the SAME car's control again (the seam asks the relay
      // for the currently adopted network on every tick).
      const secondControl = projectSessionTestControl(
        session.controller.computeControl(SESSION_TEST_FLEET, 1),
      );

      // Assert: the seam keeps publishing the live brain's control.
      expect(secondControl).toEqual(firstControl);

      // Assert: a car OUTSIDE the fleet never publishes a control.
      expect(
        session.controller.computeControl(SESSION_TEST_FLEET, 7),
      ).toBeNull();
    });

    it('gives every car its OWN brain (the per-car child-seed table)', () => {
      // Arrange: the two per-car reference networks — car 1's child seed and
      // car 2's child seed (the §9.7.2 `car-<i>` table).
      const firstCarNetwork = createSessionTestReferenceNetwork(
        deriveNeatRacingCarNetworkChildSeed(SESSION_TEST_TUPLE_SEED, 1),
      );
      const secondCarNetwork = createSessionTestReferenceNetwork(
        deriveNeatRacingCarNetworkChildSeed(SESSION_TEST_TUPLE_SEED, 2),
      );

      // Act: read each brain's bias-only forward pass (the ZERO observation
      // isolates the per-car seed from the observation law).
      const firstCarOutputs = firstCarNetwork.activate(
        new Array<number>(NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT).fill(0),
      );
      const secondCarOutputs = secondCarNetwork.activate(
        new Array<number>(NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT).fill(0),
      );

      // Assert: the per-car child seeds select DIFFERENT brains — the fleet
      // never shares one stream.
      //
      // Educational note: the published CONTROL channel alone cannot carry
      // this distinction. The shipped bootstrap draws its starter IO weights
      // at the `rand * input * sqrt(2 / input)` scale, so at the level-1
      // width (82 channels) the logistic activation SATURATES on any real
      // observation and every seeded brain publishes the same rail control.
      // The per-car seeding law is therefore proven on the brain itself.
      expect(firstCarOutputs).not.toEqual(secondCarOutputs);
    });
  });

  describe('the D1 determinism law', () => {
    it('replays the same decisions from the same tuple and never calls Math.random', async () => {
      // Arrange: the random spy (installed BEFORE the sessions are built,
      // so a construction-time draw would be caught) and TWO sessions built
      // from the SAME tuple.
      const randomSpy = jest.spyOn(Math, 'random');
      const firstSession = createSessionTestSession();
      const secondSession = createSessionTestSession();

      // Act: run the SAME tick on both sessions.
      const firstResults: readonly NeatRacingAdaptationSessionTickResult[] =
        await firstSession.adaptOnTick({ tick: 0, cars: SESSION_TEST_CARS });
      const secondResults: readonly NeatRacingAdaptationSessionTickResult[] =
        await secondSession.adaptOnTick({ tick: 0, cars: SESSION_TEST_CARS });

      // Assert: both sessions replayed the SAME per-car decisions.
      expect(firstResults).toEqual(secondResults);

      // Assert: both sessions published the SAME plain-data snapshot.
      expect(firstSession.snapshot()).toEqual(secondSession.snapshot());

      // Assert: the ambient RNG was NEVER consulted (D1).
      expect(randomSpy).not.toHaveBeenCalled();
      randomSpy.mockRestore();
    });
  });

  describe('the round boundary', () => {
    it('resets the per-car bookkeeping for the next round', async () => {
      // Arrange: a session that already ran a tick.
      const session = createSessionTestSession();
      await session.adaptOnTick({ tick: 7, cars: SESSION_TEST_CARS });

      // Act: reset the session.
      session.reset();
      const snapshot = session.snapshot();

      // Assert: the tick index is back at the round's origin.
      expect(snapshot.tick).toBe(0);

      // Assert: the per-car node counts and the shaped fitness are cleared.
      expect(snapshot.nodeCounts).toEqual([0, 0, 0]);
      expect(snapshot.shapedFitness).toEqual([null, null, null]);
    });

    it('boots a FRESH eval-worker session on reset (the stop request is terminal)', async () => {
      // Arrange: a session that already ran a tick.
      const session = createSessionTestSession();
      await session.adaptOnTick({ tick: 0, cars: SESSION_TEST_CARS });

      // Act: reset the session and read its eval-worker phase.
      session.reset();

      // Assert: the outgoing session was stopped and a FRESH one took over
      // (a terminal session would answer the next window with an error).
      expect(session.snapshot().evalPhase).toBe('ready');

      // Act: run a tick on the fresh session.
      const results = await session.adaptOnTick({
        tick: 0,
        cars: SESSION_TEST_CARS,
      });

      // Assert: the fresh session scores the window normally.
      expect(results.map((result) => result.shapedFitness)).toEqual(
        computeSessionTestShapedFitness(),
      );
    });
  });
});

/* ------------------------------------------------------------------------- *
 * The five-car fleet under the bounded variant window (slice
 * `P4-S2-concurrent-play`).
 *
 * Educational note — WHAT THIS BLOCK OWNS: the three-car fixture above pins
 * the session's OWN laws at the SOLO rung. This block pins the SAME laws at
 * the FIVE-CAR rung (the level-3 team's own fleet size) — the fleet the AC-026
 * preservation demands under CONCURRENT play:
 * - the bounded per-car variant window stays INSIDE the eval worker's dispatch
 *   bound (ONE slot per car, never more than
 *   `NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT`),
 * - the five-car fleet publishes FIVE live node counts (the human sample owns
 *   none) and the human lane stays isolated,
 * - the D1 determinism law still holds at the five-car rung (the same tuple
 *   replays the same decisions and `Math.random` is never called).
 *
 * The fixtures are DETERMINISTIC: fixed seeds, the SHIPPED autopilot fixture
 * track, a deliberately SHORT score history (the engine stops at its evidence
 * gate and never enters the growth path), NO RNG, NO timers, NO wall-clock
 * reads.
 * ------------------------------------------------------------------------- */

/** The observation ladder level of the five-car fixture (the team rung). */
const SESSION_TEST_FIVE_CAR_LEVEL = 3;

/** The fleet size of the five-car fixture (car 0 is the human sample). */
const SESSION_TEST_FIVE_CAR_COUNT = 5;

/** The grid size the shaped fitness of the five-car fixture ranks against. */
const SESSION_TEST_FIVE_CAR_MAX_CARS = 5;

/**
 * The AC-026 tuple seed the five-car fixture hangs off (its OWN stream, so
 * the five-car fixture never shares a stream with the solo fixture).
 */
const SESSION_TEST_FIVE_CAR_TUPLE_SEED = 42_424_243;

/**
 * The live node count of one five-car AGENT car: the level-3 observation
 * width plus the controller's output channels (never a magic number).
 */
const FIVE_CAR_AGENT_NODE_COUNT =
  NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT +
  NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT;

/** The AC-026 reproducibility tuple of the five-car fixture session. */
const SESSION_TEST_FIVE_CAR_TUPLE: NeatRacingReproducibilityTuple = {
  seed: SESSION_TEST_FIVE_CAR_TUPLE_SEED,
  agentCount: SESSION_TEST_FIVE_CAR_COUNT,
  packSchemaVersion: 'adaptation-session-five-car-test-pack',
  opponentSnapshot: null,
  trackId: 'adaptation-session-five-car-test-track',
  featureFlags: undefined,
};

/**
 * The per-car inputs of one five-car tick: car 0 is the human sample, cars
 * 1..4 are the agent cars.
 *
 * Educational note: the rolling score history is deliberately SHORT (two
 * entries) for the same reason as the solo fixture — the tick stops at the
 * engine's evidence gate (four entries) and never enters the growth path, so
 * the ambient RNG stays untouched (the D1 law).
 */
const SESSION_TEST_FIVE_CAR_CARS: readonly NeatRacingAdaptationCarInput[] = [
  {
    carIndex: 0,
    scoreHistory: [1, 0.5],
    position: 2,
    timeSeconds: 30,
    lapProgress01: 0.5,
  },
  {
    carIndex: 1,
    scoreHistory: [1, 0.5],
    position: 1,
    timeSeconds: 28,
    lapProgress01: 0.6,
  },
  {
    carIndex: 2,
    scoreHistory: [1, 0.5],
    position: 3,
    timeSeconds: 35,
    lapProgress01: 0.4,
  },
  {
    carIndex: 3,
    scoreHistory: [1, 0.5],
    position: 4,
    timeSeconds: 32,
    lapProgress01: 0.45,
  },
  {
    carIndex: 4,
    scoreHistory: [1, 0.5],
    position: 5,
    timeSeconds: 31,
    lapProgress01: 0.55,
  },
];

/**
 * The section 9.6 shaped fitness every five-car fixture car must be scored
 * with.
 *
 * @returns The expected per-car shaped fitness (indexed by car index).
 */
function computeSessionTestFiveCarShapedFitness(): readonly number[] {
  return SESSION_TEST_FIVE_CAR_CARS.map((car) =>
    computeNeatRacingShapedFitness(
      car.position,
      car.timeSeconds,
      car.lapProgress01,
      SESSION_TEST_FIVE_CAR_MAX_CARS,
    ),
  );
}

/**
 * Creates the five-car fixture adaptation session.
 *
 * @returns A fresh five-car session bound to the five-car tuple and track.
 */
function createSessionTestFiveCarSession(): NeatRacingAdaptationSession {
  return adaptationSessionModule.createNeatRacingAdaptationSession({
    tuple: SESSION_TEST_FIVE_CAR_TUPLE,
    trackSpec: SESSION_TEST_TRACK_SPEC,
    level: SESSION_TEST_FIVE_CAR_LEVEL,
    carCount: SESSION_TEST_FIVE_CAR_COUNT,
    maxCars: SESSION_TEST_FIVE_CAR_MAX_CARS,
  });
}

describe('neatRacing host adaptation session - the five-car fleet (P4-S2-concurrent-play)', () => {
  it('bounds the five-car variant window to one slot per car inside the eval-worker dispatch bound', async () => {
    // Arrange: the five-car fixture session (its factory boots the eval
    // worker).
    const session = createSessionTestFiveCarSession();

    // Act: run the WHOLE AC-026 adaptation window (the >= 32-tick depth) and
    // track the LARGEST per-tick window the fleet produced.
    const firstTickResults: readonly NeatRacingAdaptationSessionTickResult[] =
      await session.adaptOnTick({ tick: 0, cars: SESSION_TEST_FIVE_CAR_CARS });
    let maxObservedWindowSlots = firstTickResults.length;
    for (let tick = 1; tick < NEATRACING_REPLAY_ADAPTATION_TICKS; tick += 1) {
      const results: readonly NeatRacingAdaptationSessionTickResult[] =
        await session.adaptOnTick({
          tick,
          cars: SESSION_TEST_FIVE_CAR_CARS,
        });
      maxObservedWindowSlots = Math.max(maxObservedWindowSlots, results.length);
    }

    // Assert: the window carried ONE slot per car, the window stayed INSIDE
    // the eval worker's dispatch bound on EVERY tick of the AC-026 window,
    // the five-car fleet IS the level-3 rung's own fleet, and every car was
    // scored with the section 9.6 shaped fitness.
    expect({
      windowSlots: firstTickResults.length,
      maxObservedWindowSlots,
      insideDispatchBound:
        maxObservedWindowSlots <= NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT,
      levelRungCarCount: resolveNeatRacingLevelCarCount(
        SESSION_TEST_FIVE_CAR_LEVEL,
      ),
      shapedFitness: firstTickResults.map((result) => result.shapedFitness),
    }).toEqual({
      windowSlots: SESSION_TEST_FIVE_CAR_COUNT,
      maxObservedWindowSlots: SESSION_TEST_FIVE_CAR_COUNT,
      insideDispatchBound: true,
      levelRungCarCount: SESSION_TEST_FIVE_CAR_COUNT,
      shapedFitness: computeSessionTestFiveCarShapedFitness(),
    });
  });

  it('publishes the five live node counts and keeps the human lane isolated', async () => {
    // Arrange: the five-car fixture session.
    const session = createSessionTestFiveCarSession();

    // Act: run one adaptation tick and read the plain-data snapshot.
    const results = await session.adaptOnTick({
      tick: 0,
      cars: SESSION_TEST_FIVE_CAR_CARS,
    });
    const snapshot = session.snapshot();

    // Assert: the human sample owns NO node count, every agent car owns the
    // level-3 live node count, and the human sample is still SCORED but
    // NEVER adapted (lane isolation).
    expect({
      nodeCounts: snapshot.nodeCounts,
      humanReason: results[0]?.reason,
      humanCommitted: results[0]?.committed,
    }).toEqual({
      nodeCounts: [
        0,
        FIVE_CAR_AGENT_NODE_COUNT,
        FIVE_CAR_AGENT_NODE_COUNT,
        FIVE_CAR_AGENT_NODE_COUNT,
        FIVE_CAR_AGENT_NODE_COUNT,
      ],
      humanReason: 'human_car_not_adapted',
      humanCommitted: false,
    });
  });

  it('replays the five-car decisions from the same tuple and never calls Math.random', async () => {
    // Arrange: the random spy (installed BEFORE the sessions are built, so
    // a construction-time draw would be caught) and TWO five-car sessions
    // built from the SAME tuple.
    const randomSpy = jest.spyOn(Math, 'random');
    try {
      const firstSession = createSessionTestFiveCarSession();
      const secondSession = createSessionTestFiveCarSession();

      // Act: run the SAME AC-026 adaptation window (the >= 32-tick depth) on
      // both sessions.
      const firstWindow: Array<
        readonly NeatRacingAdaptationSessionTickResult[]
      > = [];
      const secondWindow: Array<
        readonly NeatRacingAdaptationSessionTickResult[]
      > = [];
      for (let tick = 0; tick < NEATRACING_REPLAY_ADAPTATION_TICKS; tick += 1) {
        firstWindow.push(
          await firstSession.adaptOnTick({
            tick,
            cars: SESSION_TEST_FIVE_CAR_CARS,
          }),
        );
        secondWindow.push(
          await secondSession.adaptOnTick({
            tick,
            cars: SESSION_TEST_FIVE_CAR_CARS,
          }),
        );
      }

      // Assert: both sessions replayed the SAME per-car decisions across the
      // WHOLE window.
      expect(firstWindow).toEqual(secondWindow);

      // Assert: both sessions published the SAME plain-data snapshot.
      expect(firstSession.snapshot()).toEqual(secondSession.snapshot());

      // Assert: the ambient RNG was NEVER consulted (D1).
      expect(randomSpy).not.toHaveBeenCalled();
    } finally {
      randomSpy.mockRestore();
    }
  });
});
