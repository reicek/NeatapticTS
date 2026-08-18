import { beforeEach, describe, expect, it, jest } from '@jest/globals';

import type { Network } from 'neataptic';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/worker/eval.worker.ts.
 *
 * Covers AC-SLICE-C-002: the dedicated evaluation worker threads the P5S1 soft
 * fire gate through its fitness episode, matching the display worker.
 */

import { workerSelf } from './display.worker.test-helpers';

describe('Neatenstein eval worker', () => {
  beforeEach(() => {
    jest.resetModules();
    jest.restoreAllMocks();
  });

  describe('P5S1: fire gate wired through evaluation episodes', () => {
    it('suppresses fire when no enemy is visible', async () => {
      const enemyNavModule =
        (await import('../shared/enemy-navigation')) as unknown as {
          extractSensors: jest.Mock;
        };
      jest
        .spyOn(enemyNavModule, 'extractSensors')
        .mockReturnValue(new Array(15).fill(0));

      const evalModule = (await import('./eval.worker.ts')) as {
        runFitnessEpisode: (
          network: Network,
          episodeSeed: number,
        ) => { shotsFired?: number };
      };

      const network = {
        activate: jest.fn().mockReturnValue([0, 0, 0, 1, 0]),
      } as unknown as Network;

      const signal = evalModule.runFitnessEpisode(network, 99);
      expect(signal.shotsFired).toBe(0);
    });

    it('allows fire when an enemy is visible', async () => {
      const enemyNavModule =
        (await import('../shared/enemy-navigation')) as unknown as {
          extractSensors: jest.Mock;
        };
      const sensors = new Array(15).fill(0);
      sensors[12] = 1;
      jest.spyOn(enemyNavModule, 'extractSensors').mockReturnValue(sensors);

      const evalModule = (await import('./eval.worker.ts')) as {
        runFitnessEpisode: (
          network: Network,
          episodeSeed: number,
        ) => { shotsFired?: number };
      };

      const network = {
        activate: jest.fn().mockReturnValue([0, 0, 0, 1, 0]),
      } as unknown as Network;

      const signal = evalModule.runFitnessEpisode(network, 99);
      expect(signal.shotsFired ?? 0).toBeGreaterThan(0);
    });

    it('falls back to zero outputs when activate returns a non-array', async () => {
      jest.doMock('../host/game/tick', () => ({
        gameTick: jest.fn().mockImplementation((state) => state),
      }));

      const evalModule = (await import('./eval.worker.ts')) as {
        runFitnessEpisode: (
          network: Network,
          episodeSeed: number,
        ) => { shotsFired?: number };
      };

      const network = {
        activate: jest.fn().mockReturnValue(null),
      } as unknown as Network;

      expect(() => evalModule.runFitnessEpisode(network, 99)).not.toThrow();
      jest.unmock('../host/game/tick');
    });

    it('sanitises non-finite values in network output', async () => {
      jest.doMock('../host/game/tick', () => ({
        gameTick: jest.fn().mockImplementation((state) => state),
      }));

      const evalModule = (await import('./eval.worker.ts')) as {
        runFitnessEpisode: (
          network: Network,
          episodeSeed: number,
        ) => { shotsFired?: number };
      };

      const network = {
        activate: jest.fn().mockReturnValue([NaN, Infinity, -Infinity, 1, 0]),
      } as unknown as Network;

      expect(() => evalModule.runFitnessEpisode(network, 99)).not.toThrow();
      jest.unmock('../host/game/tick');
    });

    it('sanitises non-numeric values in network output', async () => {
      jest.doMock('../host/game/tick', () => ({
        gameTick: jest.fn().mockImplementation((state) => state),
      }));

      const evalModule = (await import('./eval.worker.ts')) as {
        runFitnessEpisode: (
          network: Network,
          episodeSeed: number,
        ) => { shotsFired?: number };
      };

      const network = {
        activate: jest.fn().mockReturnValue(['not-a-number', 1, 0, 0, 0]),
      } as unknown as Network;

      expect(() => evalModule.runFitnessEpisode(network, 99)).not.toThrow();
      jest.unmock('../host/game/tick');
    });

    it('falls back to zero when the enemy-visible sensor is undefined', async () => {
      jest.doMock('../host/game/tick', () => ({
        gameTick: jest.fn().mockImplementation((state) => state),
      }));

      const enemyNavModule =
        (await import('../shared/enemy-navigation')) as unknown as {
          extractSensors: jest.Mock;
        };
      const sensors = new Array(15).fill(0);
      sensors[12] = undefined;
      jest.spyOn(enemyNavModule, 'extractSensors').mockReturnValue(sensors);

      const evalModule = (await import('./eval.worker.ts')) as {
        runFitnessEpisode: (
          network: Network,
          episodeSeed: number,
        ) => { shotsFired?: number };
      };

      const network = {
        activate: jest.fn().mockReturnValue([0, 0, 0, 0, 0]),
      } as unknown as Network;

      expect(() => evalModule.runFitnessEpisode(network, 99)).not.toThrow();
      jest.unmock('../host/game/tick');
    });

    it('posts an evalComplete message for a valid evaluate request', async () => {
      jest.doMock('../host/game/tick', () => ({
        gameTick: jest.fn().mockImplementation((state) => state),
      }));

      const enemyNavModule =
        (await import('../shared/enemy-navigation')) as unknown as {
          extractSensors: jest.Mock;
        };
      const sensors = new Array(15).fill(0);
      sensors[12] = 1;
      jest.spyOn(enemyNavModule, 'extractSensors').mockReturnValue(sensors);

      const armsRaceModule =
        (await import('../harness/arms-race')) as unknown as {
          runArmsRaceGeneration: jest.Mock;
        };
      jest
        .spyOn(armsRaceModule, 'runArmsRaceGeneration')
        .mockReturnValue({ generation: 7 });

      const evalModule = (await import('./eval.worker.ts')) as {
        runFitnessEpisode: (
          network: Network,
          episodeSeed: number,
        ) => { shotsFired?: number };
        __testOnlySetNeatConstructor: (getter: () => Promise<unknown>) => void;
      };

      const mockNetwork = {
        activate: jest.fn().mockReturnValue([0, 0, 0, 1, 0]),
        toJSON: jest.fn().mockReturnValue({ nodes: [], connections: [] }),
      };
      const fakeNeat = jest.fn().mockImplementation((...args: unknown[]) => {
        const fitnessFn = args[2] as (n: Network) => number;
        return {
          evaluate: jest.fn().mockImplementation(() => {
            fitnessFn(mockNetwork as unknown as Network);
            return Promise.resolve();
          }),
          evolve: jest.fn().mockImplementation(() => Promise.resolve()),
          getFittest: jest.fn().mockReturnValue(mockNetwork),
        };
      });

      evalModule.__testOnlySetNeatConstructor(async () => ({
        Neat: fakeNeat as unknown as typeof import('neataptic').Neat,
      }));

      if (typeof workerSelf.onmessage === 'function') {
        workerSelf.onmessage({
          data: {
            type: 'evaluate',
            seed: 1,
            generation: 2,
            enemySnapshot: { kind: 'mlp', weights: new Float32Array(8) },
            humanMode: false,
          },
        } as unknown as MessageEvent);
      }

      // Wait for the async onmessage handler.
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(workerSelf.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'evalComplete',
          generation: 7,
          championNetworkJSON: { nodes: [], connections: [] },
        }),
      );

      jest.unmock('../host/game/tick');
    });

    it('ignores non-evaluate messages', async () => {
      const evalModule = (await import('./eval.worker.ts')) as {
        __testOnlySetNeatConstructor: (getter: () => Promise<unknown>) => void;
      };

      evalModule.__testOnlySetNeatConstructor(async () => ({
        Neat: jest.fn() as unknown as typeof import('neataptic').Neat,
      }));

      workerSelf.postMessage.mockClear();

      if (typeof workerSelf.onmessage === 'function') {
        workerSelf.onmessage({
          data: { type: 'unknown' },
        } as unknown as MessageEvent);
      }

      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(workerSelf.postMessage).not.toHaveBeenCalled();
    });

    it('lazy-loads the default Neat constructor', async () => {
      const evalModule = (await import('./eval.worker.ts')) as unknown as {
        __testOnlyGetNeatConstructor: () => () => Promise<
          Pick<typeof import('neataptic'), 'Neat'>
        >;
      };

      const loader = evalModule.__testOnlyGetNeatConstructor();
      const mod = await loader();

      expect(mod.Neat).toBeDefined();
    });
  });
});
