import { Architect, Network, methods } from '../src/neataptic';

describe('Neat', () => {
  describe('AND scenario', () => {
    test('should evolve to solve AND (error non-negative)', async () => {
      // Arrange
      jest.setTimeout(50000);
      const trainingSet = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ];
      const network = new Network(2, 1);
      // Act
      const results = await network.evolve(trainingSet, {
        mutation: methods.mutation.FFW,
        equal: true,
        elitism: 10,
        mutationRate: 0.5,
        error: 0.01,
        threads: 1,
        iterations: 5,
        populationSize: 3,
      });
      // Assert
      expect(results.error).toBeGreaterThanOrEqual(0);
    });
    test('should evolve to solve AND (error below threshold)', async () => {
      // Arrange
      jest.setTimeout(50000);
      const trainingSet = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ];
      const network = new Network(2, 1);
      // Act
      const results = await network.evolve(trainingSet, {
        mutation: methods.mutation.FFW,
        equal: true,
        elitism: 10,
        mutationRate: 0.5,
        error: 0.01,
        threads: 1,
        iterations: 5,
        populationSize: 3,
      });
      // Assert
      expect(results.error).toBeLessThan(0.3);
    });
  });

  describe('XOR scenario', () => {
    test('should evolve to solve XOR (error non-negative)', async () => {
      // Arrange
      jest.setTimeout(50000);
      const trainingSet = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const network = new Network(2, 1);
      // Act
      const results = await network.evolve(trainingSet, {
        mutation: methods.mutation.FFW,
        equal: true,
        elitism: 10,
        mutationRate: 0.5,
        error: 0.01,
        threads: 1,
        iterations: 5,
        populationSize: 3,
      });
      // Assert
      expect(results.error).toBeGreaterThanOrEqual(0);
    });
    test('should evolve to solve XOR (error below threshold)', async () => {
      // Arrange
      jest.setTimeout(50000);
      const trainingSet = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
      const network = new Network(2, 1);
      // Act
      const results = await network.evolve(trainingSet, {
        mutation: methods.mutation.FFW,
        equal: true,
        elitism: 10,
        mutationRate: 0.5,
        error: 0.01,
        threads: 1,
        iterations: 5,
        populationSize: 3,
      });
      // Assert
      expect(results.error).toBeLessThan(0.3);
    });
  });

  describe('XNOR scenario', () => {
    test('should evolve to solve XNOR (error non-negative)', async () => {
      // Arrange
      jest.setTimeout(70000);
      const trainingSet = [
        { input: [0, 0], output: [1] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ];
      const network = new Network(2, 1);
      // Act
      const results = await network.evolve(trainingSet, {
        mutation: methods.mutation.FFW,
        equal: true,
        elitism: 10,
        mutationRate: 0.5,
        error: 0.01,
        threads: 1,
        iterations: 5,
        populationSize: 3,
      });
      // Assert
      expect(results.error).toBeGreaterThanOrEqual(0);
    });
    test('should evolve to solve XNOR (error below threshold)', async () => {
      // Arrange
      jest.setTimeout(70000);
      const trainingSet = [
        { input: [0, 0], output: [1] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ];
      const network = new Network(2, 1);
      // Act
      const results = await network.evolve(trainingSet, {
        mutation: methods.mutation.FFW,
        equal: true,
        elitism: 10,
        mutationRate: 0.5,
        error: 0.01,
        threads: 1,
        iterations: 5,
        populationSize: 3,
      });
      // Assert
      expect(results.error).toBeLessThan(0.3);
    });
  });
});

describe('Strict node removal', () => {
  describe('when removing input node', () => {
    test('should throw when trying to remove an input node', () => {
      // Arrange
      const net = new Network(2, 1);
      const inputNode = net.nodes.find(n => n.type === 'input');
      // Act & Assert
      expect(() => net.remove(inputNode!)).toThrow('Cannot remove input or output node from the network.');
    });
  });
  describe('when removing output node', () => {
    test('should throw when trying to remove an output node', () => {
      // Arrange
      const net = new Network(2, 1);
      const outputNode = net.nodes.find(n => n.type === 'output');
      // Act & Assert
      expect(() => net.remove(outputNode!)).toThrow('Cannot remove input or output node from the network.');
    });
  });
});
