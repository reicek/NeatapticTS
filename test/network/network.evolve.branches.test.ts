/**
 * Branch-focused tests for evolveNetwork covering:
 *  - iterations=0 early termination + _warnIfNoBestGenome path
 *  - infinite error break via repeated Infinity errors (mocked Neat)
 *  - adoption else branch when no best genome captured
 */
import Network from '../../src/architecture/network';
import { evolveNetwork } from '../../src/architecture/network/network.evolve';

type TrainingSet = Parameters<typeof evolveNetwork>[0];

// Mock the dynamically imported Neat module used inside evolveNetwork.
jest.mock('../../src/neat', () => {
  type FitnessFunction = (network: Network) => number;
  type OptionsRecord = Record<string, unknown>;
  return {
    __esModule: true,
    default: class MockNeat {
      input: number;
      output: number;
      fitnessFn: FitnessFunction;
      options: OptionsRecord;
      generation = 0;
      constructor(
        input: number,
        output: number,
        fitnessFn: FitnessFunction,
        options: OptionsRecord
      ) {
        this.input = input;
        this.output = output;
        this.fitnessFn = fitnessFn;
        this.options = options;
      }
      async evolve(): Promise<Network> {
        // Step 1: Increment generation counter to emulate progress.
        this.generation += 1;
        // Step 2: Return a network whose score remains NaN to trigger infiniteErrorCount path.
        const genome = new Network(this.input, this.output);
        genome.score = NaN;
        return genome;
      }
      _warnIfNoBestGenome() {
        console.warn(
          'Evolution completed without finding a valid best genome (mock)'
        );
      }
    },
  };
});

describe('Network.evolveNetwork branch coverage', () => {
  describe('Scenario: iterations=0 triggers warning path', () => {
    it('emits warning via _warnIfNoBestGenome when zero iterations specified', async () => {
      // Arrange
      const spy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      const net = new Network(1, 1, { seed: 60 });
      const trainingSet: TrainingSet = [{ input: [0.1], output: [0.2] }];
      // Act
      await evolveNetwork.call(net, trainingSet, { iterations: 0 });
      const warned = spy.mock.calls.some(([message]) =>
        /valid best genome/.test(String(message))
      );
      spy.mockRestore();
      // Assert
      expect(warned).toBe(true);
    });
  });

  describe('Scenario: infinite error break after repeated Infinity errors', () => {
    it('terminates early (iterations less than configured upper bound)', async () => {
      // Arrange
      const net = new Network(1, 1, { seed: 61 });
      const trainingSet: TrainingSet = [{ input: [0.9], output: [0.1] }];
      // Act
      const result = await evolveNetwork.call(net, trainingSet, {
        iterations: 50,
      });
      // Assert (loop should break after reaching infinite error threshold << 50)
      expect(result.iterations < 50).toBe(true);
    });
  });
});
