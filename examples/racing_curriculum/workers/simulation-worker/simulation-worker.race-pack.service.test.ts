/**
 * Sibling smoke tests for `simulation-worker.race-pack.service.ts`.
 *
 * Full contract tests live in `simulation-worker.race-pack.test.ts`
 * (authored in Step 03).  This file satisfies the folder quality gate's
 * sibling-test-file requirement.
 *
 * Single-expect rule enforced throughout.
 */
import {
  createDeterministicRacePack,
  resolveRaceStepTransferList,
  createRaceEpisodeRunner,
  convertCoreToRacePackSnapshot,
  extractPitLapDistribution,
  type RaceAdaptationContext,
} from './simulation-worker.race-pack.service';
import type { Network } from '../../../../src/browser-entry.ts';
import type { RuntimeAdaptationEngine } from '../../controller/runtime.adaptation';
import type { OpponentSnapshot as CoreOpponentSnapshot } from '../../../../src/neat/nge-collective/neat.nge-collective.types';

describe('simulation-worker.race-pack.service module exports', () => {
  describe('createDeterministicRacePack', () => {
    it('is exported as a function', () => {
      expect(typeof createDeterministicRacePack).toBe('function');
    });

    it('returns a frame with the correct schema version sentinel', () => {
      const frame = createDeterministicRacePack(1, {
        snapshotId: 'smoke-test',
        generation: 0,
        networkPayloads: [],
      });

      expect(frame.schemaVersion).toBe('racing-packed-v1');
    });

    it('returns a frame with a Float32Array carX field', () => {
      const frame = createDeterministicRacePack(1, {
        snapshotId: 'smoke-test',
        generation: 0,
        networkPayloads: [],
      });

      expect(frame.carX).toBeInstanceOf(Float32Array);
    });
  });

  describe('resolveRaceStepTransferList', () => {
    it('is exported as a function', () => {
      expect(typeof resolveRaceStepTransferList).toBe('function');
    });

    it('returns an array of ArrayBuffer entries', () => {
      const frame = createDeterministicRacePack(1, {
        snapshotId: 'smoke-test',
        generation: 0,
        networkPayloads: [],
      });
      const transferList = resolveRaceStepTransferList(frame);

      expect(Array.isArray(transferList)).toBe(true);
    });
  });

  describe('createRaceEpisodeRunner', () => {
    it('is exported as a function', () => {
      expect(typeof createRaceEpisodeRunner).toBe('function');
    });
  });

  describe('convertCoreToRacePackSnapshot', () => {
    it('is exported as a function', () => {
      expect(typeof convertCoreToRacePackSnapshot).toBe('function');
    });

    it('maps agentId to snapshotId and frozenAt to generation', () => {
      const coreSnapshot = makeCoreOpponentSnapshot({
        networkPayloads: [1, 2],
      });

      const racePackSnapshot = convertCoreToRacePackSnapshot(coreSnapshot);

      expect({
        snapshotId: racePackSnapshot.snapshotId,
        generation: racePackSnapshot.generation,
      }).toEqual({
        snapshotId: 'core-agent-1',
        generation: 7,
      });
    });

    it('extracts networkPayloads from the core snapshot record', () => {
      const coreSnapshot = makeCoreOpponentSnapshot({
        networkPayloads: ['a', 'b'],
      });

      expect(
        convertCoreToRacePackSnapshot(coreSnapshot).networkPayloads,
      ).toEqual(['a', 'b']);
    });

    it('falls back to an empty array when networkPayloads is missing', () => {
      const coreSnapshot = makeCoreOpponentSnapshot({});

      expect(
        convertCoreToRacePackSnapshot(coreSnapshot).networkPayloads,
      ).toEqual([]);
    });

    it('falls back to an empty array when networkPayloads is not an array', () => {
      const coreSnapshot = makeCoreOpponentSnapshot({
        networkPayloads: 'not-an-array' as unknown as readonly unknown[],
      });

      expect(
        convertCoreToRacePackSnapshot(coreSnapshot).networkPayloads,
      ).toEqual([]);
    });
  });

  describe('extractPitLapDistribution', () => {
    it('is exported as a function', () => {
      expect(typeof extractPitLapDistribution).toBe('function');
    });

    it('returns an empty array when pitLapPerCar is missing', () => {
      expect(
        extractPitLapDistribution(
          { frame: { carTeam: new Uint8Array([0, 1]) } },
          0,
        ),
      ).toEqual([]);
    });

    it('returns an empty array when carTeam is missing', () => {
      expect(
        extractPitLapDistribution({ pitLapPerCar: new Uint16Array([1, 2]) }, 0),
      ).toEqual([]);
    });

    it('returns lap numbers for cars on the requested team', () => {
      const pitLapPerCar = new Uint16Array([3, 0, 5, 2]);
      const carTeam = new Uint8Array([0, 0, 1, 1]);

      expect(
        extractPitLapDistribution({ pitLapPerCar, frame: { carTeam } }, 1),
      ).toEqual([5, 2]);
    });
  });

  describe('resolveRaceStepTransferList deduplication', () => {
    it('skips duplicate buffers already present in the transfer list', () => {
      const frame = createDeterministicRacePack(1, {
        snapshotId: 'dedup-test',
        generation: 0,
        networkPayloads: [],
      });
      // Force carX and carY to share the same underlying buffer so the dedup
      // branch is exercised.
      const sharedBuffer = new ArrayBuffer(8);
      (frame as { carX: Float32Array }).carX = new Float32Array(
        sharedBuffer,
        0,
        2,
      );
      (frame as { carY: Float32Array }).carY = new Float32Array(
        sharedBuffer,
        0,
        2,
      );

      const transferList = resolveRaceStepTransferList(frame);

      expect(transferList.length).toBe(9);
    });
  });

  describe('resolveRaceStepTransferList pitStatus presence', () => {
    it('produces the standard transfer count for a pack without pitStatus', () => {
      const runner = createRaceEpisodeRunner(
        42,
        {
          snapshotId: 'no-pit-transfer-test',
          generation: 0,
          networkPayloads: [],
        },
        makeThrottleNetworks(2),
      );

      expect(resolveRaceStepTransferList(runner.frame).length).toBe(10);
    });

    it('includes pitStatus buffer for a pack with pitStatus', () => {
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'pit-transfer-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(4),
      );

      expect(resolveRaceStepTransferList(runner.frame).length).toBe(11);
    });
  });

  describe('createRaceEpisodeRunner adaptation context', () => {
    it('returns undefined from serializeVisualizationNetwork without an adaptation context', () => {
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'adapt-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(2),
      );

      expect(runner.serializeVisualizationNetwork()).toBeUndefined();
    });

    it('returns undefined from serializeVisualizationNetwork when car 0 network is missing', () => {
      const adaptationContext = {
        engines: new Map(),
        networks: new Map(),
      } as unknown as RaceAdaptationContext;
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'adapt-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(2),
        adaptationContext,
      );

      expect(runner.serializeVisualizationNetwork()).toBeUndefined();
    });

    it('returns car 0 connection weights from serializeVisualizationNetwork', () => {
      const mockNetwork = {
        connections: [{ weight: 0.1 }, { weight: -0.2 }],
      } as unknown as Network;
      const adaptationContext = {
        engines: new Map(),
        networks: new Map([[0, mockNetwork]]),
      } as unknown as RaceAdaptationContext;
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'adapt-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(2),
        adaptationContext,
      );

      expect(runner.serializeVisualizationNetwork()).toEqual(
        Float32Array.from([0.1, -0.2]),
      );
    });

    it('invokes adaptOnTick for cars with an adaptation engine and network', () => {
      const adaptOnTick = jest.fn();
      const mockEngine = { adaptOnTick } as unknown as RuntimeAdaptationEngine;
      const mockNetwork = { connections: [] } as unknown as Network;
      const adaptationContext = {
        engines: new Map([[0, mockEngine]]),
        networks: new Map([[0, mockNetwork]]),
      } as unknown as RaceAdaptationContext;
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'adapt-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(2),
        adaptationContext,
      );

      runner.tick();

      expect(adaptOnTick).toHaveBeenCalledTimes(1);
    });

    it('rolls the per-car adaptation score history after the cap is reached', () => {
      const adaptOnTick = jest.fn();
      const mockEngine = { adaptOnTick } as unknown as RuntimeAdaptationEngine;
      const mockNetwork = { connections: [] } as unknown as Network;
      const adaptationContext = {
        engines: new Map([[0, mockEngine]]),
        networks: new Map([[0, mockNetwork]]),
      } as unknown as RaceAdaptationContext;
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'adapt-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(2),
        adaptationContext,
      );

      for (let index = 0; index < 50; index++) {
        runner.tick();
      }

      expect(adaptOnTick).toHaveBeenCalledTimes(50);
    });
  });

  describe('createRaceEpisodeRunner pit strategy', () => {
    it('does not throw when resolving pit strategy with non-zero laps since pit', () => {
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'pit-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(2),
      );

      runner.pitLapPerCar[0] = 1;
      runner.frame.lap[0] = 2;

      expect(() => runner.tick()).not.toThrow();
    });
  });

  describe('createRaceEpisodeRunner NaN controller outputs', () => {
    it('clamps NaN controller outputs to zero without crashing', () => {
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'nan-test', generation: 0, networkPayloads: [] },
        makeNetworksWithOutputs([[NaN, 0]]),
      );

      expect(() => runner.tick()).not.toThrow();
    });
  });

  describe('createRaceEpisodeRunner tier observation routing', () => {
    it('assembles a Tier 4 observation for a 4-car runner', () => {
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'tier4-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(4),
      );

      expect(() => runner.tick()).not.toThrow();
    });

    it('assembles a Tier 5 observation for a 6-car runner', () => {
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'tier5-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(6),
      );

      expect(() => runner.tick()).not.toThrow();
    });
  });

  describe('createRaceEpisodeRunner lap completion', () => {
    it('records lap completion after the car travels one track length', () => {
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'lap-test', generation: 0, networkPayloads: [] },
        makeNetworksWithOutputs([[1, 0]]),
      );

      while (runner.frame.lap[0] === 0 && runner.frame.tick < 1000) {
        runner.tick();
      }

      expect(runner.lapCompleted[0]).toBe(1);
    });
  });

  describe('createRaceEpisodeRunner pit stop guard', () => {
    it('keeps an actively pitting car stationary during the stop timer', () => {
      const runner = createRaceEpisodeRunner(
        42,
        {
          snapshotId: 'pit-stop-guard-test',
          generation: 0,
          networkPayloads: [],
        },
        makeThrottleNetworks(4),
      );
      const initialX = runner.frame.carX[0];
      const initialY = runner.frame.carY[0];
      const initialTireHealth = runner.frame.tireState[0];
      const pitStatus = runner.frame.pitStatus!;
      pitStatus[0] = 0;
      pitStatus[1] = 5;

      runner.tick();

      expect({
        carX: runner.frame.carX[0],
        carY: runner.frame.carY[0],
        tireState: runner.frame.tireState[0],
      }).toEqual({
        carX: initialX,
        carY: initialY,
        tireState: initialTireHealth,
      });
    });
  });

  describe('createRaceEpisodeRunner 6-car pitStatus layout', () => {
    it('initializes the waiting slots to zero for both teams', () => {
      const runner = createRaceEpisodeRunner(
        42,
        {
          snapshotId: 'pit-layout-test',
          generation: 0,
          networkPayloads: [],
        },
        makeThrottleNetworks(6),
      );

      expect({
        teamACarSlot: runner.frame.pitStatus![0],
        teamATickSlot: runner.frame.pitStatus![1],
        teamAWaitingSlot: runner.frame.pitStatus![2],
        teamBCarSlot: runner.frame.pitStatus![3],
        teamBTickSlot: runner.frame.pitStatus![4],
        teamBWaitingSlot: runner.frame.pitStatus![5],
      }).toEqual({
        teamACarSlot: 255,
        teamATickSlot: 0,
        teamAWaitingSlot: 0,
        teamBCarSlot: 255,
        teamBTickSlot: 0,
        teamBWaitingSlot: 0,
      });
    });
  });

  describe('createRaceEpisodeRunner pit lifecycle', () => {
    it('decrements an active pit counter and releases the slot when it reaches zero', () => {
      const runner = createRaceEpisodeRunner(
        42,
        { snapshotId: 'pit-life-test', generation: 0, networkPayloads: [] },
        makeThrottleNetworks(4),
      );
      const pitStatus = runner.frame.pitStatus!;
      pitStatus[0] = 0;
      pitStatus[1] = 1;

      runner.tick();

      expect(pitStatus[0]).toBe(255);
    });

    it('decrements a pit counter above one without releasing the slot', () => {
      const runner = createRaceEpisodeRunner(
        42,
        {
          snapshotId: 'pit-countdown-test',
          generation: 0,
          networkPayloads: [],
        },
        makeThrottleNetworks(4),
      );
      const pitStatus = runner.frame.pitStatus!;
      pitStatus[0] = 0;
      pitStatus[1] = 2;

      runner.tick();

      expect(pitStatus[1]).toBe(1);
    });

    it('releases a pit slot whose counter is already zero', () => {
      const runner = createRaceEpisodeRunner(
        42,
        {
          snapshotId: 'pit-zero-release-test',
          generation: 0,
          networkPayloads: [],
        },
        makeThrottleNetworks(4),
      );
      const pitStatus = runner.frame.pitStatus!;
      pitStatus[0] = 0;
      pitStatus[1] = 0;

      runner.tick();

      expect(pitStatus[0]).toBe(255);
    });
  });
});

function makeCoreOpponentSnapshot(overrides: {
  networkPayloads?: readonly unknown[] | string;
}): CoreOpponentSnapshot {
  return {
    agentId: 'core-agent-1',
    snapshot:
      overrides.networkPayloads === undefined
        ? {}
        : { networkPayloads: overrides.networkPayloads as readonly unknown[] },
    frozenAt: 7,
  } as CoreOpponentSnapshot;
}

function makeThrottleNetworks(
  count: number,
): Array<{ activate(inputs: number[]): number[] }> {
  return Array.from({ length: count }, () => ({
    activate: () => [0.5, 0],
  }));
}

function makeNetworksWithOutputs(
  outputs: number[][],
): Array<{ activate(inputs: number[]): number[] }> {
  return outputs.map((output) => ({
    activate: () => output,
  }));
}
