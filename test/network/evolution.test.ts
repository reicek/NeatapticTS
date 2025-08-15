jest.setTimeout(10000);

import { Architect } from '../../src/neataptic';

describe('Async Evolution', () => {
  describe('Scenario: valid dataset and reachable error', () => {
    const dataset = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ];
    let net: any;
    beforeEach(() => {
      // Arrange
      net = Architect.perceptron(2, 4, 1);
    });
    describe('when evolving', () => {
      let result: any;
      beforeEach(async () => {
        // Act
        result = await net.evolve(dataset, {
          iterations: 2,
          error: 0.5,
          amount: 1,
          threads: 1,
          popsize: 2,
        });
      });
      it('returns a result with error property', () => {
        // Assert
        expect(typeof result.error).toBe('number');
      });
      it('returns a result with iterations property', () => {
        // Assert
        expect(typeof result.iterations).toBe('number');
      });
      it('returns a result with time property', () => {
        // Assert
        expect(typeof result.time).toBe('number');
      });
      it('iterations is less than or equal to max', () => {
        // Assert
        expect(result.iterations).toBeLessThanOrEqual(2);
      });
    });
  });

  describe('Scenario: empty dataset', () => {
    it('throws or rejects when dataset is empty', async () => {
      // Arrange
      const net = Architect.perceptron(2, 4, 1);
      // Act & Assert
      await expect(
        net.evolve([], {
          iterations: 2,
          error: 0.5,
          amount: 1,
          threads: 1,
          popsize: 2,
        })
      ).rejects.toThrow();
    });
  });

  describe('Scenario: input/output size mismatch', () => {
    describe('input size mismatch', () => {
      it('throws or rejects when input size does not match', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        const dataset = [
          { input: [0], output: [0] },
          { input: [1], output: [1] },
        ];
        // Act & Assert
        await expect(
          net.evolve(dataset, {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
          })
        ).rejects.toThrow();
      });
    });
    describe('output size mismatch', () => {
      it('throws or rejects when output size does not match', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        const dataset = [
          { input: [0, 1], output: [0, 1] },
          { input: [1, 0], output: [1, 0] },
        ];
        // Act & Assert
        await expect(
          net.evolve(dataset, {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
          })
        ).rejects.toThrow();
      });
    });
  });

  describe('Scenario: missing stopping conditions', () => {
    it('throws or rejects if neither iterations nor error is specified', async () => {
      // Arrange
      const net = Architect.perceptron(2, 4, 1);
      const dataset = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
      ];
      // Act & Assert
      await expect(net.evolve(dataset, {})).rejects.toThrow();
    });
  });

  describe('Scenario: only error stopping condition', () => {
    const dataset = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ];
    let net: any;
    beforeEach(() => {
      // Arrange
      net = Architect.perceptron(2, 4, 1);
    });
    describe('when evolving with only error specified', () => {
      let result: any;
      beforeEach(async () => {
        // Act
        result = await net.evolve(dataset, {
          error: 0.5,
          amount: 1,
          threads: 1,
          popsize: 2,
        });
      });
      it('returns a result with error property', () => {
        // Assert
        expect(typeof result.error).toBe('number');
      });
      it('returns a result with time property', () => {
        // Assert
        expect(typeof result.time).toBe('number');
      });
      it('returns a result with iterations property', () => {
        // Assert
        expect(typeof result.iterations).toBe('number');
      });
    });
  });

  describe('Scenario: evolution completes with no valid best genome', () => {
    describe('when no valid best genome is found', () => {
      let warnSpy: jest.SpyInstance;
      beforeEach(() => {
        // Spy
        warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      });
      afterEach(() => {
        warnSpy.mockRestore();
      });
      it('warns if no valid best genome is found', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        // Act
        await net.evolve([{ input: [0, 0], output: [0] }], {
          iterations: 0,
          error: 0.00000001,
          amount: 1,
          threads: 1,
          popsize: 2,
        });
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(
          expect.stringContaining(
            'Evolution completed without finding a valid best genome'
          )
        );
      });
    });
  });

  describe('Advanced Evolution Scenarios', () => {
    const dataset = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ];
    describe('with POWER selection', () => {
      it('returns a result with error property', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        // Act
        const result = await net.evolve(dataset, {
          iterations: 2,
          error: 0.5,
          amount: 1,
          threads: 1,
          popsize: 2,
          selection: 'POWER',
        });
        // Assert
        expect(typeof result.error).toBe('number');
      });
    });
    describe('with FITNESS_PROPORTIONATE selection', () => {
      it('returns a result with error property', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        // Act
        const result = await net.evolve(dataset, {
          iterations: 2,
          error: 0.5,
          amount: 1,
          threads: 1,
          popsize: 2,
          selection: 'FITNESS_PROPORTIONATE',
        });
        // Assert
        expect(typeof result.error).toBe('number');
      });
    });
    describe('with TOURNAMENT selection', () => {
      it('returns a result with error property', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        // Act
        const result = await net.evolve(dataset, {
          iterations: 2,
          error: 0.5,
          amount: 1,
          threads: 1,
          popsize: 2,
          selection: { name: 'TOURNAMENT', size: 2, probability: 1 },
        });
        // Assert
        expect(typeof result.error).toBe('number');
      });
    });
    describe('with maxNodes constraint', () => {
      it('returns a result with error property', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        // Act
        const result = await net.evolve(dataset, {
          iterations: 2,
          error: 0.5,
          amount: 1,
          threads: 1,
          popsize: 2,
          maxNodes: 3,
        });
        // Assert
        expect(typeof result.error).toBe('number');
      });
    });
    describe('with empty population', () => {
      it('returns a result with error property', async () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        // Force empty population
        (net as any).population = [];
        // Act
        const result = await net.evolve(dataset, {
          iterations: 2,
          error: 0.5,
          amount: 1,
          threads: 1,
          popsize: 0,
        });
        // Assert
        expect(typeof result.error).toBe('number');
      });
    });
  });
});
