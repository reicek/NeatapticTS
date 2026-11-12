import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from '../../physics/physics.step.service';
import type { NeatRacingFleetControlSeam } from './nge.fleet.control';
import type { RaceRulesState, RaceStatus } from '../../physics/race-rules';
import {
  NEATRACING_PACK_SCHEMA_VERSION,
  createNeatRacingFramePackPool,
} from './frame.pack.service';
import type { NeatRacingPackedFrame } from './frame.pack.service';
import {
  NEATRACING_ADAPTATION_SURFACE_TAG_PREFIX,
  NEATRACING_CAR_NETWORK_SURFACE_TAG_PREFIX,
  NEATRACING_REPLAY_ADAPTATION_TICKS,
  NEATRACING_RIVAL_SWAP_SEED_STRIDE,
  NEATRACING_RIVAL_SWAP_SURFACE_TAG,
  NEATRACING_TRACK_SURFACE_TAG,
  NEATRACING_VARIANT_PATCH_SEED_OFFSET,
  NEATRACING_VARIANT_PATCH_SEED_STRIDE_FACTOR,
  NEATRACING_VARIANT_STREAM_SURFACE_TAG,
  NEATRACING_VISUAL_SEED,
  NEATRACING_XORSHIFT32_ZERO_SEED_FALLBACK,
  combineNeatRacingChildSeed,
  computeNeatRacingPackedFrameByteDigest,
  computeNeatRacingReproducibilityTupleFingerprint,
  createNeatRacingXorShift32Stream,
  deriveNeatRacingAdaptationChildSeed,
  deriveNeatRacingCarNetworkChildSeed,
  deriveNeatRacingRivalSwapChildSeed,
  deriveNeatRacingTrackChildSeed,
  deriveNeatRacingVariantStreamChildSeed,
  serializeNeatRacingReproducibilityTuple,
} from './reproducibility.utils';
import type {
  NeatRacingReproducibilityTuple,
  NeatRacingXorShift32Stream,
} from './reproducibility.utils';
import type {
  NeatRacingSimWorkerFramePayload,
  NeatRacingSimWorkerFramesPayload,
  NeatRacingSimWorkerInitPayload,
  NeatRacingSimWorkerRequest,
  NeatRacingSimWorkerResponse,
} from './simulation-worker';

/**
 * Green tests for the neatRacing reproducibility tuple utilities
 * (Phase 2, Step 01, slice P2-S1-C — tdd_sequence `green-only`).
 *
 * The red-test note for this slice is superseded by the green-only
 * sequence: there is no RED phase, and this suite asserts the slice's
 * acceptance surface — the §9.7.2 child-seed derivation table (five named
 * surfaces plus the visual-seed exemption), the seeded xorshift32 stream
 * (D1: no `Math.random` sim-side), the canonical AC-026 tuple
 * serialization shape, the packed-frame byte digest, and the AC-026
 * byte-parity window itself: two sessions booted from one fixed tuple
 * produce byte-identical packed frames across
 * `NEATRACING_REPLAY_ADAPTATION_TICKS` adaptation ticks.
 *
 * Testing pattern: worker test precedents — pure-utility describes test
 * the exported derivation/serialization functions headless, and the
 * AC-026 parity describe follows the flappy-worker precedent
 * (`globalThis.self` is replaced with a fake worker scope BEFORE the
 * worker module is dynamically imported; the fake is restored in
 * `afterAll`). Every fixture is owner-local: helpers are never imported
 * from other test files, and the byte digest is computed through the
 * P2-S1-B packer pool so this suite is the packer's first real consumer.
 */

/** Lazily-loaded worker module type (import in `beforeAll` after the stub). */
type SimWorkerModule = typeof import('./simulation-worker');

/** Session handle type alias for helper signatures. */
type SimWorkerSession = ReturnType<
  SimWorkerModule['NEATRACING_SIMULATION_WORKER_INTERNALS']['createSimWorkerSession']
>;

/** Packer-pool handle type alias for helper signatures. */
type NeatRacingTestFramePool = ReturnType<typeof createNeatRacingFramePackPool>;

/** Fake DedicatedWorkerGlobalScope the module's onmessage adapter binds to. */
const simWorkerScope: { onmessage: unknown; postMessage: jest.Mock } = {
  onmessage: undefined,
  postMessage: jest.fn(),
};

/** Original `self` property descriptor captured before the test fakes it. */
const originalSelfDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'self',
);

Object.defineProperty(globalThis, 'self', {
  configurable: true,
  value: simWorkerScope,
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

/** Worker module loaded in `beforeAll` once the fake scope has bound. */
let simWorkerModule: SimWorkerModule;

beforeAll(async () => {
  simWorkerModule = await import('./simulation-worker');
});

/** Fixture tuple seed (matches the sim-worker test fixture). */
const REPRO_TEST_SEED = 42;

/** Fixture fleet size: the AC-033 demo fleet (car 0 plus two agents). */
const REPRO_TEST_AGENT_COUNT = 3;

/** Fixture track identifier (matches the sim-worker test fixture). */
const REPRO_TEST_TRACK_ID = 'fixture-square-loop';

/** Fixture track width in meters (matches the sim-worker test fixture). */
const REPRO_TEST_TRACK_WIDTH_M = 60;

/** Fixture frame step in ms: one frame releases exactly one sim tick. */
const REPRO_TEST_FRAME_STEP_MS = 16.667;

/** Short xorshift32 draw window for sequence replay assertions. */
const REPRO_TEST_DRAW_COUNT = 8;

/** Long xorshift32 draw window for the `[0, 1)` range assertion. */
const REPRO_TEST_STRESS_DRAW_COUNT = 256;

/**
 * Builds the closed-loop square fixture track (the sim-worker test
 * pattern): four axis-aligned segments whose endpoints close exactly
 * ((0,0) → (L,0) → (L,L) → (0,0)), with one spline sample per segment
 * midpoint so the autopilot has a frozen racing line to target.
 *
 * @param segmentLengthM - Side length in meters (larger geometry moves
 *   the spawn line and the splines, changing every later frame).
 * @returns A minimal deterministic track spec fixture.
 */
function createReproTestFixtureTrackSpec(segmentLengthM = 800) {
  const sideM = segmentLengthM;
  const halfLengthM = segmentLengthM / 2;
  const segments = [
    { startX: 0, startY: 0, endX: sideM, endY: 0 },
    { startX: sideM, startY: 0, endX: sideM, endY: sideM },
    { startX: sideM, startY: sideM, endX: 0, endY: sideM },
    { startX: 0, startY: sideM, endX: 0, endY: 0 },
  ].map((segment) => ({
    ...segment,
    width: REPRO_TEST_TRACK_WIDTH_M,
  }));
  const splineSamples = [
    { x: halfLengthM, y: 0 },
    { x: segmentLengthM, y: halfLengthM },
    { x: halfLengthM, y: segmentLengthM },
    { x: 0, y: halfLengthM },
  ].map((sample, globalIndex) => ({
    ...sample,
    width: REPRO_TEST_TRACK_WIDTH_M,
    segmentIndex: globalIndex,
    sampleIndexWithinSegment: 0,
    globalIndex,
  }));
  return {
    seed: REPRO_TEST_SEED,
    layoutVersion: 1,
    sizeBucket: 'fixture',
    segments,
    splineSamples,
    laneCount: 2,
  };
}

/**
 * Builds the fixture AC-026 reproducibility tuple (worker-tuple fields
 * only, no owned geometry).
 *
 * @returns A complete deterministic tuple fixture.
 */
function createReproTestFixtureTuple(): NeatRacingReproducibilityTuple {
  return {
    seed: REPRO_TEST_SEED,
    agentCount: REPRO_TEST_AGENT_COUNT,
    packSchemaVersion: NEATRACING_PACK_SCHEMA_VERSION,
    opponentSnapshot: null,
    trackId: REPRO_TEST_TRACK_ID,
    featureFlags: { aiAutopilot: true },
  };
}

/**
 * Builds the fixture init payload (the AC-026 tuple + owned geometry).
 *
 * Educational note — the injected controller: the legacy scripted lane is
 * retired (AC-029), so an agent car with NO injected controller has no
 * driver. The fixture therefore boots WITH the deterministic controller
 * double, and the fleet size stays 3 (the multi-car coverage survives).
 *
 * @param overrides - Partial fields to override on the fixture payload.
 * @returns A complete init payload for one deterministic fixture boot.
 */
function createReproTestFixtureInitPayload(
  overrides: Partial<NeatRacingSimWorkerInitPayload> = {},
): NeatRacingSimWorkerInitPayload {
  return {
    ...createReproTestFixtureTuple(),
    trackSpec: createReproTestFixtureTrackSpec(),
    raceMode: 'ai',
    controller: createReproTestControllerDouble(),
    ...overrides,
  };
}

/**
 * The deterministic control answer the agent cars publish in this suite.
 *
 * Educational note — why a FIXED answer: the retired scripted lane is gone
 * (AC-029), so the AGENT cars take the CONTROLLER's answer. A frozen answer
 * keeps the AC-026 byte-parity contract deterministic — no RNG, no network,
 * no clock.
 */
const REPRO_TEST_AGENT_CONTROL: Readonly<PhysicsStepInput> = Object.freeze({
  throttle: 1,
  brake: 0,
  steer: 0.25,
});

/**
 * Creates the deterministic controller double the agent cars run.
 *
 * Educational note — the seam contract: `computeControl(fleet, carIndex)`
 * publishes the control input of ONE fleet car. The double is STATELESS and
 * PURE (frozen constants plus the addressed car's own heading, no mutable
 * state, no call counter) because the AC-026 contract compares TWO
 * independently booted sessions element-for-element — a stateful double
 * would make their digest streams diverge.
 *
 * @returns A deterministic fleet-control seam double.
 */
function createReproTestControllerDouble(): NeatRacingFleetControlSeam {
  return {
    computeControl(
      fleet: readonly PhysicsCarState[],
      carIndex: number,
    ): PhysicsStepInput {
      const addressedCar = fleet[carIndex];
      return {
        throttle: REPRO_TEST_AGENT_CONTROL.throttle,
        brake: REPRO_TEST_AGENT_CONTROL.brake,
        steer: REPRO_TEST_AGENT_CONTROL.steer + addressedCar.heading,
      };
    },
  };
}

/**
 * Quantizes a fixture value to Float32 (the packed channel storage type).
 *
 * @param value - Fixture float value.
 * @returns The Float32-exact quantized value.
 */
function reproTestFloat32(value: number): number {
  return Math.fround(value);
}

/**
 * Builds one deterministic fixture car whose floats are Float32-exact
 * (the P2-S1-B packer-test pattern, owner-local copy).
 *
 * @param carIndex - Fleet index of the fixture car.
 * @returns A Float32-exact deterministic car state.
 */
function createReproTestCar(carIndex: number): PhysicsCarState {
  return {
    // Physics pose (Float32-exact base values; index offsets stay exact).
    x: reproTestFloat32(1.5 + carIndex * 0.25),
    y: reproTestFloat32(-0.25 + carIndex * 0.5),
    heading: reproTestFloat32(0.75 + carIndex * 0.125),
    speed: reproTestFloat32(12.5 + carIndex * 2),
    yawRate: reproTestFloat32(0.02 * (carIndex + 1)),
    slipAngle: reproTestFloat32(0.125 + carIndex * 0.03125),
    travelHeadingRad: reproTestFloat32(0.875 + carIndex * 0.25),
    tractionSeverity: reproTestFloat32(0.25 * carIndex),
    lateralVelocity: reproTestFloat32(0.375 + carIndex * 0.125),
    // Drift channels (committed + visible pose, float32-exact values).
    drift: {
      committed: {
        slip: reproTestFloat32(-0.125 - carIndex * 0.0625),
        turnSign: carIndex % 2 === 0 ? 1 : -1,
        reArmed: carIndex % 2 === 1,
      },
      pose: {
        pose: reproTestFloat32(0.0625 * (carIndex + 1)),
        poseRate: reproTestFloat32(0.5 - carIndex * 0.0625),
        articulation: reproTestFloat32(-0.25 + carIndex * 0.125),
      },
    },
    // Discrete per-car tick counter (exact small int in float32).
    tick: reproTestFloat32(7 + carIndex * 3),
  };
}

/**
 * Builds a deterministic fixture fleet of `size` cars.
 *
 * @param size - Fleet size (car 0 plus `size - 1` agents).
 * @returns The fixture fleet array.
 */
function createReproTestFleet(size: number): PhysicsCarState[] {
  const cars: PhysicsCarState[] = [];
  for (let carIndex = 0; carIndex < size; carIndex++) {
    cars.push(createReproTestCar(carIndex));
  }
  return cars;
}

/**
 * Builds deterministic fixture rules for a packed frame.
 *
 * @param raceStatus - Fixture lifecycle status (default running).
 * @param nextRaceLevel - Fixture carried level (default null mid-race).
 * @returns A deterministic rules fixture.
 */
function createReproTestRules(
  raceStatus: RaceStatus = 'running',
  nextRaceLevel: number | null = null,
): RaceRulesState {
  return {
    level: 2,
    lapsCompleted: 1,
    checkpointBudgetS: 6.25,
    elapsedS: 4.125,
    raceStatus,
    nextRaceLevel,
  };
}

/**
 * Creates a fixture packer pool sized for a fixture fleet.
 *
 * @param fleetSize - Fleet size the pool must pack (one channel per car).
 * @returns A fresh `neatracing-packed-v1` packer pool.
 */
function createReproTestFramePool(
  fleetSize: number = REPRO_TEST_AGENT_COUNT,
): NeatRacingTestFramePool {
  return createNeatRacingFramePackPool({ carCapacity: fleetSize });
}

/**
 * Packs one deterministic fixture frame through the fixture pool.
 *
 * @param pool - The fixture packer pool.
 * @param overrides - Optional header/fleet overrides for a variant frame.
 * @returns The packed frame (a digest-able `neatracing-packed-v1` frame).
 */
function packReproTestFrame(
  pool: NeatRacingTestFramePool,
  overrides: {
    cars?: PhysicsCarState[];
    fromTick?: number;
    toTick?: number;
  } = {},
): NeatRacingPackedFrame {
  return pool.packFrame({
    fromTick: overrides.fromTick ?? 0,
    toTick: overrides.toTick ?? 1,
    cars: overrides.cars ?? createReproTestFleet(REPRO_TEST_AGENT_COUNT),
    rules: createReproTestRules('running', null),
  });
}

/**
 * Collects `drawCount` consecutive draws from one xorshift32 stream.
 *
 * @param stream - The seeded draw function.
 * @param drawCount - Number of draws to collect.
 * @returns The collected draw window.
 */
function collectXorShiftDraws(
  stream: NeatRacingXorShift32Stream,
  drawCount: number,
): number[] {
  const draws: number[] = [];
  for (let drawIndex = 0; drawIndex < drawCount; drawIndex++) {
    draws.push(stream());
  }
  return draws;
}

describe('combineNeatRacingChildSeed + §9.7.2 derivation table', () => {
  it('derives track, car-network, and adaptation child seeds from the named tags', () => {
    // Arrange: the fixture tuple seed and the expected manual mixes.
    const tupleSeed = REPRO_TEST_SEED;

    // Act: derive each named surface through the §9.7.2 table helpers.

    // Assert: each table row pins its `surfaceTag` vocabulary exactly.
    expect(deriveNeatRacingCarNetworkChildSeed(tupleSeed, 0)).toBe(
      combineNeatRacingChildSeed(
        tupleSeed,
        `${NEATRACING_CAR_NETWORK_SURFACE_TAG_PREFIX}0`,
      ),
    );
    expect(deriveNeatRacingTrackChildSeed(tupleSeed)).toBe(
      combineNeatRacingChildSeed(tupleSeed, NEATRACING_TRACK_SURFACE_TAG),
    );
    expect(deriveNeatRacingAdaptationChildSeed(tupleSeed, 2)).toBe(
      combineNeatRacingChildSeed(
        tupleSeed,
        `${NEATRACING_ADAPTATION_SURFACE_TAG_PREFIX}2`,
      ),
    );
  });

  it('derives variant and rival-swap stream bases and pins the visual exemption', () => {
    // Arrange: the fixture tuple seed.
    const tupleSeed = REPRO_TEST_SEED;

    // Act: derive the two stream-surface bases at index 0.

    // Assert: each row 4/5 base is the raw tag mix; TP-20 visual streams
    // are exempt from the table and use the named seed verbatim.
    expect(deriveNeatRacingVariantStreamChildSeed(tupleSeed, 0)).toBe(
      combineNeatRacingChildSeed(
        tupleSeed,
        NEATRACING_VARIANT_STREAM_SURFACE_TAG,
      ),
    );
    expect(deriveNeatRacingRivalSwapChildSeed(tupleSeed, 0)).toBe(
      combineNeatRacingChildSeed(tupleSeed, NEATRACING_RIVAL_SWAP_SURFACE_TAG),
    );
    expect(NEATRACING_VISUAL_SEED).toBe(1337);
  });

  it('keeps the mix deterministic and sensitive to tuple seed and tag', () => {
    // Arrange: the fixture tuple seed plus distinct perturbations.

    // Act: combine the same inputs twice, then perturbed inputs.

    // Assert: pure-function determinism, and both inputs matter.
    expect(
      combineNeatRacingChildSeed(REPRO_TEST_SEED, NEATRACING_TRACK_SURFACE_TAG),
    ).toBe(
      combineNeatRacingChildSeed(REPRO_TEST_SEED, NEATRACING_TRACK_SURFACE_TAG),
    );
    expect(
      combineNeatRacingChildSeed(
        REPRO_TEST_SEED + 1,
        NEATRACING_TRACK_SURFACE_TAG,
      ),
    ).not.toBe(
      combineNeatRacingChildSeed(REPRO_TEST_SEED, NEATRACING_TRACK_SURFACE_TAG),
    );
    expect(
      combineNeatRacingChildSeed(
        REPRO_TEST_SEED,
        NEATRACING_VARIANT_STREAM_SURFACE_TAG,
      ),
    ).not.toBe(
      combineNeatRacingChildSeed(REPRO_TEST_SEED, NEATRACING_TRACK_SURFACE_TAG),
    );
  });

  it('mirrors the recycled per-variant stride composition', () => {
    // Arrange: the variant stream base and the recycled offset/factor
    // constants (`patchSize` 0 falls back to the offset base).
    const variantBase = combineNeatRacingChildSeed(
      REPRO_TEST_SEED,
      NEATRACING_VARIANT_STREAM_SURFACE_TAG,
    );

    // Act: derive slots for the offset-dominated and factor-dominated
    // stride branches (patchSize 300 → stride 1200 > offset 1000).

    // Assert: index 0 ignores the stride; the named stride formula holds
    // on both branches.
    expect(
      deriveNeatRacingVariantStreamChildSeed(REPRO_TEST_SEED, 0, 300),
    ).toBe(variantBase);
    expect(deriveNeatRacingVariantStreamChildSeed(REPRO_TEST_SEED, 2)).toBe(
      (variantBase + 2 * NEATRACING_VARIANT_PATCH_SEED_OFFSET) >>> 0,
    );
    expect(
      deriveNeatRacingVariantStreamChildSeed(REPRO_TEST_SEED, 2, 300),
    ).toBe(
      (variantBase +
        2 * (300 * NEATRACING_VARIANT_PATCH_SEED_STRIDE_FACTOR)) >>>
        0,
    );
  });

  it('spaces rival-swap event seeds by the named stride', () => {
    // Arrange: the rival-swap stream base.
    const rivalBase = combineNeatRacingChildSeed(
      REPRO_TEST_SEED,
      NEATRACING_RIVAL_SWAP_SURFACE_TAG,
    );

    // Act: derive disjoint swap-event slots.

    // Assert: each event advances exactly one named stride.
    expect(deriveNeatRacingRivalSwapChildSeed(REPRO_TEST_SEED, 2)).toBe(
      (rivalBase + 2 * NEATRACING_RIVAL_SWAP_SEED_STRIDE) >>> 0,
    );
    expect(deriveNeatRacingRivalSwapChildSeed(REPRO_TEST_SEED, 1)).toBe(
      (rivalBase + NEATRACING_RIVAL_SWAP_SEED_STRIDE) >>> 0,
    );
  });

  it('rejects empty tags and invalid derivation indices', () => {
    // Arrange: the fixture tuple seed.

    // Act: mix an empty tag and derive with negative / fractional indices.

    // Assert: loud static RangeErrors at the derivation boundary.
    expect(() => combineNeatRacingChildSeed(REPRO_TEST_SEED, '')).toThrow(
      RangeError,
    );
    expect(() =>
      deriveNeatRacingCarNetworkChildSeed(REPRO_TEST_SEED, -1),
    ).toThrow(RangeError);
    expect(() =>
      deriveNeatRacingAdaptationChildSeed(REPRO_TEST_SEED, 0.5),
    ).toThrow(RangeError);
  });

  it('rejects non-finite tuple seeds', () => {
    // Arrange: a non-finite seed.

    // Act: mix it into any surface.

    // Assert: a loud static RangeError (no silent NaN propagation).
    expect(() =>
      combineNeatRacingChildSeed(Number.NaN, NEATRACING_TRACK_SURFACE_TAG),
    ).toThrow(RangeError);
  });
});

describe('createNeatRacingXorShift32Stream', () => {
  it('replays identical draw sequences from identical child seeds', () => {
    // Arrange: the track child seed and two fresh streams from it.
    const childSeed = deriveNeatRacingTrackChildSeed(REPRO_TEST_SEED);

    // Act: draw the short window from each stream.
    const drawsA = collectXorShiftDraws(
      createNeatRacingXorShift32Stream(childSeed),
      REPRO_TEST_DRAW_COUNT,
    );
    const drawsB = collectXorShiftDraws(
      createNeatRacingXorShift32Stream(childSeed),
      REPRO_TEST_DRAW_COUNT,
    );

    // Assert: identical seeds replay identical sequences (D1).
    expect(drawsB).toEqual(drawsA);
  });

  it('diverges across different child seeds', () => {
    // Arrange: two distinct unsigned child seeds.

    // Act: draw the short window from each stream.

    // Assert: different seeds seed different streams.
    expect(
      collectXorShiftDraws(
        createNeatRacingXorShift32Stream(1),
        REPRO_TEST_DRAW_COUNT,
      ),
    ).not.toEqual(
      collectXorShiftDraws(
        createNeatRacingXorShift32Stream(2),
        REPRO_TEST_DRAW_COUNT,
      ),
    );
  });

  it('stays within [0, 1) across a long draw window', () => {
    // Arrange: an adaptation child seed and its stream.
    const childSeed = deriveNeatRacingAdaptationChildSeed(REPRO_TEST_SEED, 1);

    // Act: draw the long stress window.
    const draws = collectXorShiftDraws(
      createNeatRacingXorShift32Stream(childSeed),
      REPRO_TEST_STRESS_DRAW_COUNT,
    );

    // Assert: every draw is a unit float in [0, 1).
    expect(Math.min(...draws)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...draws)).toBeLessThan(1);
  });

  it('guards a zero child seed onto the named fallback stream', () => {
    // Arrange: a zero seed and the named zero-state fallback seed.

    // Act: draw the short window from both streams.

    // Assert: the zero seed streams exactly like the fallback (xorshift32
    // would otherwise be stuck at zero forever).
    expect(
      collectXorShiftDraws(
        createNeatRacingXorShift32Stream(0),
        REPRO_TEST_DRAW_COUNT,
      ),
    ).toEqual(
      collectXorShiftDraws(
        createNeatRacingXorShift32Stream(
          NEATRACING_XORSHIFT32_ZERO_SEED_FALLBACK,
        ),
        REPRO_TEST_DRAW_COUNT,
      ),
    );
  });

  it('rejects a non-finite child seed', () => {
    // Arrange: a non-finite child seed.

    // Act: build a stream from it.

    // Assert: a loud static RangeError.
    expect(() => createNeatRacingXorShift32Stream(Number.NaN)).toThrow(
      RangeError,
    );
  });
});

describe('serializeNeatRacingReproducibilityTuple', () => {
  it('pins the AC-026 field order in the canonical serialization', () => {
    // Arrange: the fixture tuple.

    // Act: serialize and re-parse the canonical JSON text.

    // Assert: the serialized key order follows the pinned AC-026 order.
    const parsed = JSON.parse(
      serializeNeatRacingReproducibilityTuple(createReproTestFixtureTuple()),
    ) as Record<string, unknown>;
    expect(Object.keys(parsed)).toEqual([
      'seed',
      'agentCount',
      'packSchemaVersion',
      'opponentSnapshot',
      'trackId',
      'featureFlags',
    ]);
  });

  it('canonicalizes nested object key order without touching array order', () => {
    // Arrange: two tuples differing only in nested key INSERTION order.
    const tupleA = createReproTestFixtureTuple();
    tupleA.opponentSnapshot = {
      zeta: 1,
      alphabet: [1, 2, 3],
      nested: { beta: 2, alpha: 1 },
    };
    const tupleB = createReproTestFixtureTuple();
    tupleB.opponentSnapshot = {
      nested: { alpha: 1, beta: 2 },
      alphabet: [1, 2, 3],
      zeta: 1,
    };

    // Act: serialize both.
    const serialA = serializeNeatRacingReproducibilityTuple(tupleA);
    const serialB = serializeNeatRacingReproducibilityTuple(tupleB);

    // Assert: identical serials (key order canonicalized), and the array
    // order survives canonicalization untouched.
    expect(serialB).toBe(serialA);
    expect(
      (JSON.parse(serialA) as { opponentSnapshot: unknown }).opponentSnapshot,
    ).toEqual({ zeta: 1, alphabet: [1, 2, 3], nested: { beta: 2, alpha: 1 } });
  });

  it('carries absent featureFlags as an explicit null slot and present flags verbatim', () => {
    // Arrange: the fixture tuple, once without and once with feature flags.
    const tupleWithoutFlags: NeatRacingReproducibilityTuple = {
      ...createReproTestFixtureTuple(),
      featureFlags: undefined,
    };

    // Act: serialize both and re-parse.

    // Assert: absent flags collapse to an explicit null slot in the same
    // field position; present flags are carried verbatim.
    expect(
      (
        JSON.parse(
          serializeNeatRacingReproducibilityTuple(tupleWithoutFlags),
        ) as {
          featureFlags: unknown;
        }
      ).featureFlags,
    ).toBe(null);
    expect(
      (
        JSON.parse(
          serializeNeatRacingReproducibilityTuple(
            createReproTestFixtureTuple(),
          ),
        ) as { featureFlags: unknown }
      ).featureFlags,
    ).toEqual({ aiAutopilot: true });
  });

  it('fingerprints deterministically and differentiates across tuple seeds', () => {
    // Arrange: the fixture tuple plus a one-seed-perturbed tuple.
    const tuple = createReproTestFixtureTuple();
    const tupleSeedShifted = { ...tuple, seed: tuple.seed + 1 };

    // Act: fingerprint the deterministic tuple twice and the shifted tuple.

    // Assert: deterministic repeats, seed-sensitive serials, and
    // seed-sensitive fingerprints (both hashes stay aligned).
    expect(computeNeatRacingReproducibilityTupleFingerprint({ ...tuple })).toBe(
      computeNeatRacingReproducibilityTupleFingerprint({ ...tuple }),
    );
    expect(serializeNeatRacingReproducibilityTuple(tupleSeedShifted)).not.toBe(
      serializeNeatRacingReproducibilityTuple(tuple),
    );
    expect(
      computeNeatRacingReproducibilityTupleFingerprint(tupleSeedShifted),
    ).not.toBe(computeNeatRacingReproducibilityTupleFingerprint(tuple));
  });

  it('rejects non-JSON-safe tuple values', () => {
    // Arrange: fixture tuples carrying non-JSON-safe or non-finite fields.

    // Act: serialize each rejected tuple.

    // Assert: loud static RangeErrors (never silent JSON coercion).
    expect(() =>
      serializeNeatRacingReproducibilityTuple({
        ...createReproTestFixtureTuple(),
        opponentSnapshot: () => null,
      }),
    ).toThrow(RangeError);
    expect(() =>
      serializeNeatRacingReproducibilityTuple({
        ...createReproTestFixtureTuple(),
        opponentSnapshot: Symbol('opponent'),
      }),
    ).toThrow(RangeError);
    expect(() =>
      serializeNeatRacingReproducibilityTuple({
        ...createReproTestFixtureTuple(),
        seed: Number.NaN,
      }),
    ).toThrow(RangeError);
  });
});

describe('computeNeatRacingPackedFrameByteDigest', () => {
  it('digests one fixed packed frame deterministically', () => {
    // Arrange: the fixture pool and one packed fixture frame.

    // Act: digest the same packed frame twice.

    // Assert: the digest witness is deterministic.
    const pool = createReproTestFramePool();
    const packedFrame = packReproTestFrame(pool);
    expect(computeNeatRacingPackedFrameByteDigest(packedFrame)).toBe(
      computeNeatRacingPackedFrameByteDigest(packedFrame),
    );
  });

  it('flips the digest on any channel-byte change', () => {
    // Arrange: two fleets differing in one float channel (car 0 speed).
    const pool = createReproTestFramePool();
    const fleetA = createReproTestFleet(REPRO_TEST_AGENT_COUNT);
    const fleetB = createReproTestFleet(REPRO_TEST_AGENT_COUNT);
    fleetB[0] = { ...fleetB[0], speed: reproTestFloat32(12.75) };

    // Act: pack and digest both frames (channel change from 12.5 → 12.75).

    // Assert: any channel-byte change flips the packed-frame digest — the
    // AC-026 byte-parity witness has no silent channel tolerance.
    expect(
      computeNeatRacingPackedFrameByteDigest(
        packReproTestFrame(pool, { cars: fleetB }),
      ),
    ).not.toBe(
      computeNeatRacingPackedFrameByteDigest(
        packReproTestFrame(pool, { cars: fleetA }),
      ),
    );
  });

  it('rejects a frame with a foreign schema sentinel', () => {
    // Arrange: a fake packed frame stamped with a foreign version string.
    const foreignFrame = {
      schemaVersion: 'neatracing-packed-vX',
    } as unknown as NeatRacingPackedFrame;

    // Act: digest it.

    // Assert: a loud RangeError from the schema sentinel check.
    expect(() => computeNeatRacingPackedFrameByteDigest(foreignFrame)).toThrow(
      RangeError,
    );
  });

  it('rejects a frame with missing header scalars', () => {
    // Arrange: a fake packed frame stamped with the correct schema
    // sentinel but carrying no header scalars or channel views.
    const hollowFrame = {
      schemaVersion: NEATRACING_PACK_SCHEMA_VERSION,
    } as unknown as NeatRacingPackedFrame;

    // Act: digest it.

    // Assert: a loud RangeError from the header-fold validation.
    expect(() => computeNeatRacingPackedFrameByteDigest(hollowFrame)).toThrow(
      RangeError,
    );
  });
});

describe('AC-026 byte parity over the adaptation-tick window', () => {
  /**
   * Boots one fixture session from a fixture init payload.
   *
   * @param payload - The fixture init payload (the AC-026 tuple + track).
   * @returns The booted session (init acknowledged via a quiet guard).
   */
  function bootReproTestSession(
    payload: NeatRacingSimWorkerInitPayload,
  ): SimWorkerSession {
    if (!simWorkerModule) {
      throw new Error('simulation-worker module not loaded in beforeAll');
    }
    const session =
      simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS.createSimWorkerSession();
    const responses = session.handleRequest({ type: 'init', payload });
    if (responses[0]?.type !== 'ready') {
      throw new Error('sim worker boot did not acknowledge init with ready');
    }
    return session;
  }

  /**
   * Narrowly-typed accessor for a `frames` reply's payload.
   *
   * @param responses - The single-response list from `handleRequest`.
   * @returns The `frames` payload of the sole response.
   */
  function framePayloadOf(
    responses: NeatRacingSimWorkerResponse[],
  ): NeatRacingSimWorkerFramesPayload {
    expect(responses).toHaveLength(1);
    expect(responses[0].type).toBe('frames');
    return (responses[0] as { payload: NeatRacingSimWorkerFramesPayload })
      .payload;
  }

  /**
   * Drives one session through the zero-tick anchor plus
   * `NEATRACING_REPLAY_ADAPTATION_TICKS` one-tick frames, packing each
   * structured reply and digesting it immediately.
   *
   * @param session - The booted sim-worker session to pace.
   * @param pool - The shared fixture packer pool (digests run before any
   *   subsequent packFrame call, so slot rotation cannot alias bytes).
   * @returns The digest stream, one entry per windowed adaptation tick.
   */
  function digestAdaptationTickWindow(
    session: SimWorkerSession,
    pool: NeatRacingTestFramePool,
  ): string[] {
    // Arrange: warm the accumulator with the zero-tick anchor frame.
    session.handleRequest({
      type: 'frame',
      payload: { timestampMs: 0 } satisfies NeatRacingSimWorkerFramePayload,
    });

    // Act: pace exactly one adaptation-tick window of once-per-tick frames.
    const digests: string[] = [];
    for (
      let frameIndex = 0;
      frameIndex < NEATRACING_REPLAY_ADAPTATION_TICKS;
      frameIndex++
    ) {
      const request: NeatRacingSimWorkerRequest = {
        type: 'frame',
        payload: {
          timestampMs:
            REPRO_TEST_FRAME_STEP_MS + frameIndex * REPRO_TEST_FRAME_STEP_MS,
        } satisfies NeatRacingSimWorkerFramePayload,
      };
      const framesPayload = framePayloadOf(session.handleRequest(request));
      digests.push(
        computeNeatRacingPackedFrameByteDigest(pool.packFrame(framesPayload)),
      );
    }
    return digests;
  }

  it('produces identical packed-frame digest streams across NEATRACING_REPLAY_ADAPTATION_TICKS ticks', () => {
    // Arrange: two independent sessions booted from the SAME tuple, and
    // one shared packer pool.
    const sessionA = bootReproTestSession(createReproTestFixtureInitPayload());
    const sessionB = bootReproTestSession(createReproTestFixtureInitPayload());
    const pool = createReproTestFramePool();

    // Act: pace both sessions through identical anchor + window pacing.
    const digestsA = digestAdaptationTickWindow(sessionA, pool);
    const digestsB = digestAdaptationTickWindow(sessionB, pool);

    // Assert: the window covers exactly the named tick count, every
    // digest is 8-char lowercase hex (the 32-bit FNV-1a witness shape),
    // and the two digest streams are element-for-element identical — the
    // AC-026 same-runtime byte-parity claim at this boundary.
    expect(digestsA).toHaveLength(NEATRACING_REPLAY_ADAPTATION_TICKS);
    expect(digestsA.every((digest) => /^[0-9a-f]{8}$/.test(digest))).toBe(true);
    expect(digestsB).toEqual(digestsA);
  });

  it('diverges when the tuple fleet size differs (negative differentiator)', () => {
    // Arrange: two sessions whose tuples differ in agentCount (fleet size).
    // The tuple seed itself does NOT move sim bytes at this boundary — the
    // worker fixes each spawn pose explicitly (so the physics seed is
    // ignored) and the derived child seeds feed the sim only from Step 03
    // onward, while a larger fleet adds a channel block to every frame.
    const sessionA = bootReproTestSession(createReproTestFixtureInitPayload());
    const sessionB = bootReproTestSession(
      createReproTestFixtureInitPayload({ agentCount: 4 }),
    );
    const poolA = createReproTestFramePool();
    const poolB = createReproTestFramePool(4);

    // Act: pace both sessions through the identical window.
    const digestsA = digestAdaptationTickWindow(sessionA, poolA);
    const digestsB = digestAdaptationTickWindow(sessionB, poolB);

    // Assert: a tuple fleet-size change must change at least one packed
    // frame's bytes inside the window (the parity witness is not
    // channel-blind end-to-end through real session traffic).
    expect(digestsB).not.toEqual(digestsA);
  });
});
