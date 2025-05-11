jest.setTimeout(10000);

import { Architect } from '../../src/neataptic';

describe('Async Evolution', () => {
  describe('Scenario: valid dataset and reachable error', () => {
    let result: any;
    beforeAll(async () => {
      // Arrange
      const dataset = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const net = Architect.perceptron(2, 4, 1);
      // Act
      result = await net.evolve(dataset, { iterations: 2, error: 0.5, amount: 1, threads: 1, popsize: 2 });
    });
    test('returns a result with error property', () => {
      // Assert
      expect(typeof result.error).toBe('number');
    });
    test('returns a result with iterations property', () => {
      // Assert
      expect(typeof result.iterations).toBe('number');
    });
    test('returns a result with time property', () => {
      // Assert
      expect(typeof result.time).toBe('number');
    });
    test('iterations is less than or equal to max', () => {
      // Assert
      expect(result.iterations).toBeLessThanOrEqual(2);
    });
  });

  describe('Scenario: empty dataset', () => {
    test('throws or rejects', async () => {
      // Arrange
      const net = Architect.perceptron(2, 4, 1);
      // Act & Assert
      await expect(net.evolve([], { iterations: 2, error: 0.5, amount: 1, threads: 1, popsize: 2 })).rejects.toThrow();
    });
  });

  describe('Scenario: input/output size mismatch', () => {
    describe('input size mismatch', () => {
      test('throws or rejects', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        const dataset = [
          { input: [0], output: [0] },
          { input: [1], output: [1] }
        ];
        // Act & Assert
        await expect(net.evolve(dataset, { iterations: 2, error: 0.5, amount: 1, threads: 1, popsize: 2 })).rejects.toThrow();
      });
    });
    describe('output size mismatch', () => {
      test('throws or rejects', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        const dataset = [
          { input: [0, 1], output: [0, 1] },
          { input: [1, 0], output: [1, 0] }
        ];
        // Act & Assert
        await expect(net.evolve(dataset, { iterations: 2, error: 0.5, amount: 1, threads: 1, popsize: 2 })).rejects.toThrow();
      });
    });
  });

  describe('Scenario: missing stopping conditions', () => {
    test('throws or rejects if neither iterations nor error is specified', async () => {
      // Arrange
      const net = Architect.perceptron(2, 4, 1);
      const dataset = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] }
      ];
      // Act & Assert
      await expect(net.evolve(dataset, {})).rejects.toThrow();
    });
  });

  describe('Scenario: only error stopping condition', () => {
    test('resolves with result when only error is specified', async () => {
      // Arrange
      const dataset = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const net = Architect.perceptron(2, 4, 1);
      
      // Act
      const result = await net.evolve(dataset, { error: 0.5, amount: 1, threads: 1, popsize: 2 });
      
      // Assert
      expect(typeof result.error).toBe('number');
      expect(typeof result.time).toBe('number');
      expect(typeof result.iterations).toBe('number');
    });
  });

  describe('Scenario: evolution completes with no valid best genome', () => {
    test('warns if no valid best genome is found', async () => {
      // Arrange
      const net = Architect.perceptron(2, 4, 1);
      // Spy
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      // Act
      await net.evolve([{ input: [0, 0], output: [0] }], { iterations: 0, error: 0.00000001, amount: 1, threads: 1, popsize: 2 });
      // Assert
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Evolution completed without finding a valid best genome'));
      warnSpy.mockRestore();
    });
  });

  describe('Advanced Evolution Scenarios', () => {
    test('should evolve with POWER selection', async () => {
      // Arrange
      const dataset = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const net = Architect.perceptron(2, 4, 1);
      // Act
      const result = await net.evolve(dataset, { iterations: 2, error: 0.5, amount: 1, threads: 1, popsize: 2, selection: 'POWER' });
      // Assert
      expect(typeof result.error).toBe('number');
    });

    test('should evolve with FITNESS_PROPORTIONATE selection', async () => {
      // Arrange
      const dataset = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const net = Architect.perceptron(2, 4, 1);
      // Act
      const result = await net.evolve(dataset, { iterations: 2, error: 0.5, amount: 1, threads: 1, popsize: 2, selection: 'FITNESS_PROPORTIONATE' });
      // Assert
      expect(typeof result.error).toBe('number');
    });

    test('should evolve with TOURNAMENT selection', async () => {
      // Arrange
      const dataset = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const net = Architect.perceptron(2, 4, 1);
      // Act
      const result = await net.evolve(dataset, { iterations: 2, error: 0.5, amount: 1, threads: 1, popsize: 2, selection: { name: 'TOURNAMENT', size: 2, probability: 1 } });
      // Assert
      expect(typeof result.error).toBe('number');
    });

    test('should evolve with maxNodes constraint', async () => {
      // Arrange
      const dataset = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const net = Architect.perceptron(2, 4, 1);
      // Act
      const result = await net.evolve(dataset, { iterations: 2, error: 0.5, amount: 1, threads: 1, popsize: 2, maxNodes: 3 });
      // Assert
      expect(typeof result.error).toBe('number');
    });

    test('should evolve with empty population', async () => {
      // Arrange
      const dataset = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const net = Architect.perceptron(2, 4, 1);
      // Force empty population
      (net as any).population = [];
      // Act
      const result = await net.evolve(dataset, { iterations: 2, error: 0.5, amount: 1, threads: 1, popsize: 0 });
      // Assert
      expect(typeof result.error).toBe('number');
    });
  });
});
