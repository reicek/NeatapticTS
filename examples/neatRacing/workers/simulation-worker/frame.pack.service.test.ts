/**
 * Frame pack service acceptance suite (Phase 2, Step 01, slice P2-S1-B).
 *
 * Educational note — what this suite pins:
 * Every acceptance criterion of the packed-frames wire contract in one
 * place: the `'neatracing-packed-v1'` sentinel stamping and guard, the
 * recycled D8b feature-flag bit table (NeatRacing pins tires = 0,
 * radio = 1, pits = 0), the preallocated single-buffer slot layout with
 * computed byte offsets, deterministic Float32 channel quantization, the
 * zero-allocation pool reuse contract, the exactly-one transfer-buffer
 * resolution, and the static validation errors.
 *
 * GREEN-ONLY note: the red-test note for this slice is superseded by the
 * green-only sequence — tests here are the slice's own acceptance suite
 * and assert pack/unpack round-trip and zero-alloc transfer behavior
 * directly (per the step packet's green-only note).
 *
 * Determinism discipline (D8a): fixtures quantize every float field with
 * `Math.fround` so round-trip equality is exact; the lossiness of Float32
 * is asserted explicitly exactly once (non-representable value test).
 */

import { describe, expect, it } from '@jest/globals';
import type { PhysicsCarState } from '../../physics/physics.step.service';
import type { RaceRulesState, RaceStatus } from '../../physics/race-rules';
import {
  NEATRACING_DEFAULT_FEATURE_FLAG_BITFIELD,
  NEATRACING_FEATURE_FLAG_BITS,
  NEATRACING_FRAME_PACK_MAX_POOL_SLOTS,
  NEATRACING_FRAME_PACK_POOL_SLOTS,
  NEATRACING_PACKED_CAR_CHANNELS,
  NEATRACING_PACKED_CAR_CHANNEL_COUNT,
  NEATRACING_PACKED_MAX_CAR_CAPACITY,
  NEATRACING_PACKED_RULES_FLOAT_COUNT,
  NEATRACING_PACKED_TRANSFER_BUFFER_COUNT,
  NEATRACING_PACKED_WORD_BYTES,
  NEATRACING_PACK_NULL_LEVEL_SENTINEL,
  NEATRACING_PACK_SCHEMA_VERSION,
  assertNeatRacingPackedFrameSchemaVersion,
  createNeatRacingFramePackPool,
  decodeNeatRacingFeatureFlags,
  encodeNeatRacingFeatureFlags,
  resolveNeatRacingPackedChannelFirstFloatWord,
  resolveNeatRacingPackedFrameFloatWordCount,
  resolveNeatRacingPackedFrameTransferList,
  resolveNeatRacingPackedSlotByteLength,
} from './frame.pack.service';
import type { NeatRacingFeatureFlagState } from './frame.pack.service';

/** Fixture fleet size for the multi-car round-trip. */
const DEMO_FLEET_SIZE = 3;
/** Fixture pool capacity used by the layout tests. */
const LAYOUT_TEST_CAPACITY = 3;

/** Quantizes a fixture value to Float32 (the packed channel storage type). */
function expectedFloat32(value: number): number {
  return Math.fround(value);
}

/** Builds one deterministic fixture car whose floats are Float32-exact. */
function createFixtureCarState(carIndex: number): PhysicsCarState {
  return {
    // Physics pose (Float32-exact base values; index offsets stay exact).
    x: expectedFloat32(1.5 + carIndex * 0.25),
    y: expectedFloat32(-0.25 + carIndex * 0.5),
    heading: expectedFloat32(0.75 + carIndex * 0.125),
    speed: expectedFloat32(12.5 + carIndex * 2),
    yawRate: expectedFloat32(0.02 * (carIndex + 1)),
    slipAngle: expectedFloat32(0.125 + carIndex * 0.03125),
    travelHeadingRad: expectedFloat32(0.875 + carIndex * 0.25),
    tractionSeverity: expectedFloat32(0.25 * carIndex),
    lateralVelocity: expectedFloat32(0.375 + carIndex * 0.125),
    // Drift channels (committed + visible pose, float32-exact values).
    drift: {
      committed: {
        slip: expectedFloat32(-0.125 - carIndex * 0.0625),
        turnSign: carIndex % 2 === 0 ? 1 : -1,
        reArmed: carIndex % 2 === 1,
      },
      pose: {
        pose: expectedFloat32(0.0625 * (carIndex + 1)),
        poseRate: expectedFloat32(0.5 - carIndex * 0.0625),
        articulation: expectedFloat32(-0.25 + carIndex * 0.125),
      },
    },
    // Discrete per-car tick counter (exact small int in float32).
    tick: expectedFloat32(7 + carIndex * 3),
  };
}

/** Builds a deterministic demo fleet of `size` cars. */
function createDemoFleet(size: number): PhysicsCarState[] {
  const cars: PhysicsCarState[] = [];
  for (let carIndex = 0; carIndex < size; carIndex++) {
    cars.push(createFixtureCarState(carIndex));
  }
  return cars;
}

/** Builds deterministic rules for a packed round-trip. */
function createFixtureRules(
  raceStatus: RaceStatus,
  nextRaceLevel: number | null,
  overrides?: Partial<Pick<RaceRulesState, 'elapsedS' | 'checkpointBudgetS'>>,
): RaceRulesState {
  return {
    level: 2,
    lapsCompleted: 1,
    checkpointBudgetS: overrides?.checkpointBudgetS ?? 6.25,
    elapsedS: overrides?.elapsedS ?? 4.125,
    raceStatus,
    nextRaceLevel,
  };
}

/** Packs one fixture frame with default NeatRacing flags. */
function packFixtureFrame(
  carCapacity: number,
  cars: readonly PhysicsCarState[],
  rules: RaceRulesState,
) {
  const pool = createNeatRacingFramePackPool({ carCapacity });
  return pool.packFrame({ fromTick: 10, toTick: 13, cars, rules });
}

describe('neatracing-packed-v1 frame pack service', () => {
  describe('schema sentinel', () => {
    it('stamps the neatracing-packed-v1 sentinel on packed frames', () => {
      const frame = packFixtureFrame(
        1,
        createDemoFleet(1),
        createFixtureRules('running', null),
      );
      expect(frame.schemaVersion).toBe(NEATRACING_PACK_SCHEMA_VERSION);
      expect(() =>
        assertNeatRacingPackedFrameSchemaVersion({
          schemaVersion: 'racing-packed-v1',
        }),
      ).toThrow(RangeError);
    });

    it('rejects a missing sentinel on received frames', () => {
      expect(() =>
        assertNeatRacingPackedFrameSchemaVersion({ schemaVersion: undefined }),
      ).toThrow(/schemaVersion/);
    });
  });

  describe('feature flag bit table (D8b)', () => {
    it('encodes single flags on the recycled bit assignments', () => {
      const tiresOnly = encodeNeatRacingFeatureFlags({
        tiresEnabled: true,
        radioEnabled: false,
        pitsEnabled: false,
      });
      const radioOnly = encodeNeatRacingFeatureFlags({
        tiresEnabled: false,
        radioEnabled: true,
        pitsEnabled: false,
      });
      const pitsOnly = encodeNeatRacingFeatureFlags({
        tiresEnabled: false,
        radioEnabled: false,
        pitsEnabled: true,
      });
      expect(tiresOnly).toBe(NEATRACING_FEATURE_FLAG_BITS.tiresEnabled);
      expect(radioOnly).toBe(NEATRACING_FEATURE_FLAG_BITS.radioEnabled);
      expect(pitsOnly).toBe(NEATRACING_FEATURE_FLAG_BITS.pitsEnabled);
    });

    it('packs the NeatRacing pinned defaults as bitfield 2', () => {
      const frame = packFixtureFrame(
        1,
        createDemoFleet(1),
        createFixtureRules('running', null),
      );
      expect(frame.featureFlags).toBe(NEATRACING_DEFAULT_FEATURE_FLAG_BITFIELD);
    });

    it('decodes all eight masked bitfields exactly', () => {
      const decodes: Record<number, NeatRacingFeatureFlagState> = {};
      for (let bitfield = 0; bitfield < 8; bitfield++) {
        decodes[bitfield] = decodeNeatRacingFeatureFlags(bitfield);
      }
      expect(decodes).toEqual({
        0: { tiresEnabled: false, radioEnabled: false, pitsEnabled: false },
        1: { tiresEnabled: true, radioEnabled: false, pitsEnabled: false },
        2: { tiresEnabled: false, radioEnabled: true, pitsEnabled: false },
        3: { tiresEnabled: true, radioEnabled: true, pitsEnabled: false },
        4: { tiresEnabled: false, radioEnabled: false, pitsEnabled: true },
        5: { tiresEnabled: true, radioEnabled: false, pitsEnabled: true },
        6: { tiresEnabled: false, radioEnabled: true, pitsEnabled: true },
        7: { tiresEnabled: true, radioEnabled: true, pitsEnabled: true },
      });
    });

    it('round-trips custom flags through encode/decode', () => {
      const decoded = decodeNeatRacingFeatureFlags(
        encodeNeatRacingFeatureFlags({
          tiresEnabled: true,
          radioEnabled: true,
          pitsEnabled: true,
        }),
      );
      expect(decoded).toEqual({
        tiresEnabled: true,
        radioEnabled: true,
        pitsEnabled: true,
      });
    });
  });

  describe('preallocated slot layout (AC-042)', () => {
    it('places the rules span at word 0 and named channels at table offsets', () => {
      const frame = packFixtureFrame(
        LAYOUT_TEST_CAPACITY,
        createDemoFleet(LAYOUT_TEST_CAPACITY),
        createFixtureRules('running', null),
      );
      expect([
        frame.rulesChannel.byteOffset,
        frame.rulesChannel.length,
      ]).toEqual([0, NEATRACING_PACKED_RULES_FLOAT_COUNT]);

      const viewLayout: Record<string, number[]> = {};
      for (
        let channelIndex = 0;
        channelIndex < NEATRACING_PACKED_CAR_CHANNEL_COUNT;
        channelIndex++
      ) {
        const channel = NEATRACING_PACKED_CAR_CHANNELS[channelIndex];
        const view = (frame as unknown as Record<string, Float32Array>)[
          channel.name
        ];
        viewLayout[channel.name] = [view.byteOffset, view.length];
      }
      const expectedLayout: Record<string, number[]> = {};
      for (
        let channelIndex = 0;
        channelIndex < NEATRACING_PACKED_CAR_CHANNEL_COUNT;
        channelIndex++
      ) {
        const channel = NEATRACING_PACKED_CAR_CHANNELS[channelIndex];
        expectedLayout[channel.name] = [
          resolveNeatRacingPackedChannelFirstFloatWord(
            channelIndex,
            LAYOUT_TEST_CAPACITY,
          ) * NEATRACING_PACKED_WORD_BYTES,
          LAYOUT_TEST_CAPACITY,
        ];
      }
      expect(viewLayout).toEqual(expectedLayout);
    });

    it('sizes slot buffers from the resolver contract', () => {
      const pool = createNeatRacingFramePackPool({
        carCapacity: LAYOUT_TEST_CAPACITY,
      });
      expect(pool.slotByteLength).toBe(
        resolveNeatRacingPackedSlotByteLength(LAYOUT_TEST_CAPACITY),
      );
      expect(
        resolveNeatRacingPackedFrameFloatWordCount(LAYOUT_TEST_CAPACITY),
      ).toBe(
        NEATRACING_PACKED_RULES_FLOAT_COUNT +
          NEATRACING_PACKED_CAR_CHANNEL_COUNT * LAYOUT_TEST_CAPACITY,
      );
    });
  });

  describe('packing validation', () => {
    it('guards the car count bounds with static errors', () => {
      const pool = createNeatRacingFramePackPool({ carCapacity: 2 });
      expect(() =>
        pool.packFrame({
          fromTick: 0,
          toTick: 1,
          cars: [],
          rules: createFixtureRules('running', null),
        }),
      ).toThrow('frame pack failed: at least one car is required');
      expect(() =>
        pool.packFrame({
          fromTick: 0,
          toTick: 1,
          cars: createDemoFleet(3),
          rules: createFixtureRules('running', null),
        }),
      ).toThrow('frame pack failed: car count exceeds the packer carCapacity');
    });

    it('guards non-finite channel values with named static errors', () => {
      const car = createFixtureCarState(0);
      const invalidCar: PhysicsCarState = { ...car, speed: Number.NaN };
      expect(() =>
        packFixtureFrame(1, [invalidCar], createFixtureRules('running', null)),
      ).toThrow(
        'frame pack failed: car channel carSpeed must be a finite number',
      );
    });

    it('guards rules fields with static errors', () => {
      const cars = createDemoFleet(1);
      const badStatus = createFixtureRules('running', null);
      expect(() =>
        packFixtureFrame(1, cars, {
          ...badStatus,
          raceStatus: 'unknown' as unknown as RaceStatus,
        }),
      ).toThrow('frame pack failed: raceStatus is not a packed code');
      const badLevel = createFixtureRules('running', null);
      expect(() =>
        packFixtureFrame(1, cars, {
          ...badLevel,
          level: Number.NaN,
        }),
      ).toThrow('frame pack failed: rules level must be a finite number');
      const badNextLevel = createFixtureRules('running', null);
      expect(() =>
        packFixtureFrame(1, cars, {
          ...badNextLevel,
          nextRaceLevel: Number.NaN,
        }),
      ).toThrow(
        'frame pack failed: rules nextRaceLevel must be null or finite',
      );
    });

    it('guards the released-tick envelope ordering', () => {
      const pool = createNeatRacingFramePackPool({ carCapacity: 1 });
      expect(() =>
        pool.packFrame({
          fromTick: 5,
          toTick: 4,
          cars: createDemoFleet(1),
          rules: createFixtureRules('running', null),
        }),
      ).toThrow('frame pack failed: toTick must not precede fromTick');
    });

    it('guards pool options with static errors', () => {
      expect(() =>
        createNeatRacingFramePackPool({
          carCapacity: NEATRACING_PACKED_MAX_CAR_CAPACITY + 1,
        }),
      ).toThrow(
        'frame pack pool failed: carCapacity must be an integer in [1, 32]',
      );
      expect(() =>
        createNeatRacingFramePackPool({
          carCapacity: 1,
          poolSlots: NEATRACING_FRAME_PACK_MAX_POOL_SLOTS + 1,
        }),
      ).toThrow(
        'frame pack pool failed: poolSlots must be an integer in [1, 4]',
      );
    });
  });

  describe('pack/unpack round-trip', () => {
    it('round-trips a single car exactly on Float32-exact fixtures', () => {
      const snapshot = createNeatRacingFramePackPool({
        carCapacity: 1,
      }).unpackFrame(
        packFixtureFrame(
          1,
          createDemoFleet(1),
          createFixtureRules('running', null),
        ),
      );
      expect(snapshot.cars).toEqual([createFixtureCarState(0)]);
    });

    it('round-trips a three-car fleet in channel order', () => {
      const snapshot = createNeatRacingFramePackPool({
        carCapacity: DEMO_FLEET_SIZE,
      }).unpackFrame(
        packFixtureFrame(
          DEMO_FLEET_SIZE,
          createDemoFleet(DEMO_FLEET_SIZE),
          createFixtureRules('running', null),
        ),
      );
      expect(snapshot.cars).toEqual(createDemoFleet(DEMO_FLEET_SIZE));
      expect(snapshot.agentCount).toBe(DEMO_FLEET_SIZE);
    });

    it('round-trips the exact integer header fields', () => {
      const frame = packFixtureFrame(
        1,
        createDemoFleet(1),
        createFixtureRules('raceComplete', 2),
      );
      const snapshot = createNeatRacingFramePackPool({
        carCapacity: 1,
      }).unpackFrame(frame);
      expect([
        snapshot.fromTick,
        snapshot.toTick,
        snapshot.agentCount,
        snapshot.featureFlags,
        snapshot.rules.level,
        snapshot.rules.lapsCompleted,
        snapshot.rules.raceStatus,
        snapshot.rules.nextRaceLevel,
      ]).toEqual([
        10,
        13,
        1,
        NEATRACING_DEFAULT_FEATURE_FLAG_BITFIELD,
        2,
        1,
        'raceComplete',
        2,
      ]);
    });

    it('round-trips race status codes and the null-level sentinel', () => {
      const pool = createNeatRacingFramePackPool({ carCapacity: 1 });
      const cars = createDemoFleet(1);
      const running = pool.unpackFrame(
        pool.packFrame({
          fromTick: 0,
          toTick: 1,
          cars,
          rules: createFixtureRules('running', null),
        }),
      );
      expect(running.rules.raceStatus).toBe('running');
      const raceComplete = pool.unpackFrame(
        pool.packFrame({
          fromTick: 0,
          toTick: 1,
          cars,
          rules: createFixtureRules('raceComplete', 2),
        }),
      );
      expect(raceComplete.rules.raceStatus).toBe('raceComplete');
      const outOfTime = pool.unpackFrame(
        pool.packFrame({
          fromTick: 0,
          toTick: 1,
          cars,
          rules: createFixtureRules('outOfTime', null),
        }),
      );
      expect(outOfTime.rules.nextRaceLevel).toBeNull();
    });

    it('writes the null-level sentinel into the packed header', () => {
      const frame = packFixtureFrame(
        1,
        createDemoFleet(1),
        createFixtureRules('running', null),
      );
      expect(frame.nextRaceLevel).toBe(NEATRACING_PACK_NULL_LEVEL_SENTINEL);
    });

    it('round-trips the continuous rules channel with Float32 quantization', () => {
      const pool = createNeatRacingFramePackPool({ carCapacity: 1 });
      const snapshot = pool.unpackFrame(
        pool.packFrame({
          fromTick: 0,
          toTick: 1,
          cars: createDemoFleet(1),
          rules: createFixtureRules('running', null, {
            // 5.1 is not Float32-representable: the lossy contract shows.
            checkpointBudgetS: 5.1,
            elapsedS: 0.1,
          }),
        }),
      );
      expect(snapshot.rules.checkpointBudgetS).toBe(expectedFloat32(5.1));
      expect(snapshot.rules.elapsedS).toBe(expectedFloat32(0.1));
      expect(snapshot.rules.checkpointBudgetS).not.toBe(5.1);
    });
  });

  describe('determinism and zero-alloc pool (AC-042)', () => {
    it('packs identical inputs to bit-identical channel values', () => {
      const pool = createNeatRacingFramePackPool({
        carCapacity: DEMO_FLEET_SIZE,
      });
      const input = {
        fromTick: 10,
        toTick: 13,
        cars: createDemoFleet(DEMO_FLEET_SIZE),
        rules: createFixtureRules('running', null),
      };
      const first = pool.packFrame(input);
      const second = pool.packFrame(input);
      const channels = (frame: typeof first): number[] =>
        NEATRACING_PACKED_CAR_CHANNELS.map(
          (channel) =>
            (frame as unknown as Record<string, Float32Array>)[channel.name][0],
        );
      expect(channels(second)).toEqual(channels(first));
    });

    it('reuses exactly poolSlots distinct frame objects across packs', () => {
      const pool = createNeatRacingFramePackPool({
        carCapacity: 1,
      });
      const input = {
        fromTick: 0,
        toTick: 1,
        cars: createDemoFleet(1),
        rules: createFixtureRules('running', null),
      };
      const first = pool.packFrame(input);
      const second = pool.packFrame(input);
      const third = pool.packFrame(input);
      const fourth = pool.packFrame(input);
      const uniqueFrames = new Set([first, second, third, fourth]);
      expect(uniqueFrames.size).toBe(NEATRACING_FRAME_PACK_POOL_SLOTS);
      expect(first).toBe(third);
      expect(second).toBe(fourth);
    });

    it('keeps buffer identity stable across in-place repacks', () => {
      const pool = createNeatRacingFramePackPool({ carCapacity: 1 });
      const input = {
        fromTick: 0,
        toTick: 1,
        cars: createDemoFleet(1),
        rules: createFixtureRules('running', null),
      };
      // Pack three frames: slots round-robin, so the third pack lands on the
      // same slot as the first and must reuse its views and buffer in place.
      const first = pool.packFrame(input);
      pool.packFrame(input);
      const third = pool.packFrame(input);
      expect(third.rulesChannel).toBe(first.rulesChannel);
      expect(third.carX.buffer).toBe(first.carX.buffer);
    });
  });

  describe('transfer list resolution (AC-042)', () => {
    it('exposes exactly one preallocated transfer buffer', () => {
      const frame = packFixtureFrame(
        1,
        createDemoFleet(1),
        createFixtureRules('running', null),
      );
      const transferList = resolveNeatRacingPackedFrameTransferList(frame);
      expect(transferList.length).toBe(NEATRACING_PACKED_TRANSFER_BUFFER_COUNT);
      expect(transferList[0]).toBe(frame.carX.buffer);
    });
  });
});
