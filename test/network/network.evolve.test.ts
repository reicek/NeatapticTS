import Network from '../../src/architecture/network';
import Multi from '../../src/multithreading/multi';
import { Workers } from '../../src/multithreading/workers/workers';
import { evolveNetwork } from '../../src/architecture/network/network.utils';

type TrainingSet = Parameters<typeof evolveNetwork>[0];

/**
 * Evolution helper tests target uncovered control-flow in evolveNetwork:
 *  - dataset validation errors
 *  - missing stopping conditions error
 *  - error-only path (iterations auto-set to 0 => zero loop iterations)
 *  - iterations-only path (targetError=-1 sentinel)
 *  - multi-thread fallback (threads>1 but likely no workers available)
 *  - schedule + log callback execution
 */
describe('Network.evolveNetwork', () => {
  describe('Scenario: dataset dimensionality invalid', () => {
    it('throws descriptive size mismatch error', async () => {
      // Arrange
      const net = new Network(2, 1, { seed: 1 });
      const invalidTrainingSet: TrainingSet = [
        {
          input: [1],
          output: [0],
        },
      ]; // wrong input length
      // Act / Assert
      await expect(
        evolveNetwork.call(net, invalidTrainingSet, { iterations: 1 }),
      ).rejects.toThrow(/Dataset is invalid/);
    });
  });

  describe('Scenario: missing stopping conditions', () => {
    it('throws when neither iterations nor error specified', async () => {
      // Arrange
      const net = new Network(1, 1, { seed: 2 });
      const trainingSet: TrainingSet = [{ input: [0.1], output: [0.2] }];
      // Act / Assert
      await expect(evolveNetwork.call(net, trainingSet, {})).rejects.toThrow(
        /At least one stopping condition/,
      );
    });
  });

  describe('Scenario: error-only stopping (zero-iteration path)', () => {
    it('returns iterations = 0 when only error target supplied', async () => {
      // Arrange
      const net = new Network(1, 1, { seed: 3 });
      const trainingSet: TrainingSet = [{ input: [0.1], output: [0.2] }];
      // Act
      const result = await evolveNetwork.call(net, trainingSet, {
        error: 0.5,
      });
      // Assert
      expect(result.iterations).toBe(0);
    });
  });

  describe('Scenario: iterations-only stopping (targetError sentinel -1)', () => {
    it('performs exactly one generation when iterations=1', async () => {
      // Arrange
      const net = new Network(1, 1, { seed: 4 });
      const trainingSet: TrainingSet = [{ input: [0.1], output: [0.2] }];
      // Act
      const result = await evolveNetwork.call(net, trainingSet, {
        iterations: 1,
        popsize: 6,
      });
      // Assert
      expect(result.iterations).toBe(1);
    });
  });

  describe('Scenario: multi-thread fallback (no workers available)', () => {
    it('completes evolution with threads>1 by falling back to single thread', async () => {
      // Arrange
      const net = new Network(1, 1, { seed: 5 });
      const trainingSet: TrainingSet = [{ input: [0.3], output: [0.7] }];
      // Step 1: Override worker discovery hooks with a subclass that rejects to force fallback.
      const originalWorkers = Multi.workers;
      class DisabledWorkers extends Workers {
        static getNodeTestWorker(): Promise<never> {
          return Promise.reject(new Error('worker disabled for test'));
        }

        static getBrowserTestWorker(): Promise<never> {
          return Promise.reject(new Error('worker disabled for test'));
        }
      }
      Multi.workers = DisabledWorkers;
      try {
        const result = await evolveNetwork.call(net, trainingSet, {
          iterations: 1,
          threads: 4,
        });
        // Assert
        expect(result.iterations).toBe(1);
      } finally {
        Multi.workers = originalWorkers;
      }
    });
  });

  describe('Scenario: schedule + log callbacks inside loop', () => {
    it('invokes schedule function at specified interval', async () => {
      // Arrange
      const net = new Network(1, 1, { seed: 6 });
      const trainingSet: TrainingSet = [{ input: [0.9], output: [0.1] }];
      const scheduleFn = jest.fn();
      // Act
      await evolveNetwork.call(net, trainingSet, {
        iterations: 1,
        schedule: { iterations: 1, function: scheduleFn },
        log: 1,
      });
      // Assert
      expect(scheduleFn).toHaveBeenCalledTimes(1);
    });
  });
});
