import { Network } from '../../../neataptic';
import type { EvolutionSettings, NeatRuntime } from '../network.types';
import { runEvolutionLoop } from './network.evolve.loop.utils';
import { DISABLED_TARGET_ERROR } from './network.evolve.utils.types';

jest.setTimeout(10000);

const baseSettings: EvolutionSettings = {
  growth: 0,
  cost: { name: 'MSE' },
  amount: 1,
  log: 0,
  schedule: undefined,
  clear: false,
  threads: 1,
  targetError: DISABLED_TARGET_ERROR,
};

function buildRuntime(scorePerCall: number): {
  runtime: NeatRuntime;
  callCount: () => number;
} {
  const state = { generation: 0, calls: 0 };
  const mockGenome = new Network(2, 1);
  mockGenome.score = scorePerCall;

  const runtime: NeatRuntime = {
    get generation() {
      return state.generation;
    },
    options: {},
    evolve: async () => {
      state.generation++;
      state.calls++;
      return mockGenome;
    },
  };

  return { runtime, callCount: () => state.calls };
}

describe('runEvolutionLoop()', () => {
  describe('given the evolved genome always produces an invalid (NaN) score', () => {
    describe('when the consecutive-invalid-error threshold is reached', () => {
      it('breaks the loop early', async () => {
        // Arrange – undefined score → NaN fitness → Infinity error → increments invalid counter
        const state = { generation: 0 };
        const mockGenome = new Network(2, 1);
        mockGenome.score = NaN;

        const runtime: NeatRuntime = {
          get generation() {
            return state.generation;
          },
          options: {},
          evolve: async () => {
            state.generation++;
            return mockGenome;
          },
        };

        // Act – iterations=100 but loop aborts after 5 consecutive invalid errors
        const result = await runEvolutionLoop(
          runtime,
          { ...baseSettings },
          DISABLED_TARGET_ERROR,
          100,
        );

        // Assert – loop returned early; generation < 100
        expect(state.generation).toBeLessThan(100);
        expect(result.error).toBe(Infinity);
      });
    });
  });
  describe('given the evolved genome has an undefined score', () => {
    describe('when the loop derives fitness from the genome', () => {
      it('treats undefined score as -Infinity fitness and runs the invalid-error path', async () => {
        // Arrange – no score set → score = undefined → ?? -Infinity fires (covers line 113)
        const state = { generation: 0 };
        const mockGenome = new Network(2, 1);
        // score is intentionally not set (undefined)

        const runtime: NeatRuntime = {
          get generation() {
            return state.generation;
          },
          options: {},
          evolve: async () => {
            state.generation++;
            return mockGenome;
          },
        };

        // Act – runs until invalid-error threshold (5 consecutive)
        const result = await runEvolutionLoop(
          runtime,
          { ...baseSettings },
          DISABLED_TARGET_ERROR,
          100,
        );

        // Assert
        expect(result.error).toBe(Infinity);
      });
    });
  });

  describe('given no iterations limit is specified', () => {
    describe('when the target error is met on the first evolve pass', () => {
      it('returns after the first pass without iterating further', async () => {
        // Arrange – fitness -0.1 → error 0.1; targetError 0.5
        const { runtime, callCount } = buildRuntime(-0.1);

        // Act – undefined iterations → iterationsSpecified = false (covers line 82 TRUE arm)
        await runEvolutionLoop(runtime, { ...baseSettings }, 0.5, undefined);

        // Assert
        expect(callCount()).toBe(1);
      });
    });
  });

  describe('given a schedule with iterations=2 and the loop runs for 3 generations', () => {
    describe('when reading which generations fired the schedule callback', () => {
      it('fires the callback only on even-numbered generations', async () => {
        // Arrange
        const scheduleFiredAtGenerations: number[] = [];
        let mockGeneration = 0;
        const mockGenome = new Network(2, 1);
        mockGenome.score = -0.1;

        const runtime: NeatRuntime = {
          get generation() {
            return mockGeneration;
          },
          options: {},
          evolve: async () => {
            mockGeneration++;
            return mockGenome;
          },
        };

        const scheduleConfig = {
          iterations: 2,
          function: ({
            iteration,
          }: {
            fitness: number;
            error: number;
            iteration: number;
          }) => {
            scheduleFiredAtGenerations.push(iteration);
          },
        };

        // Act – 3 iterations: gens 1,2,3. gen%2!=0 at gen 1 and 3 → covers line 219 TRUE arm
        await runEvolutionLoop(
          runtime,
          { ...baseSettings, schedule: scheduleConfig },
          DISABLED_TARGET_ERROR,
          3,
        );

        // Assert – schedule fired at generation 2 but not at 1 or 3
        expect(scheduleFiredAtGenerations).toEqual([2]);
      });
    });
  });
});
