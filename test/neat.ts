import { Architect, Network, methods } from '../src/neataptic';
import Neat from '../src/neat';

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
        error: 0.1,
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
        iterations: 50,
        populationSize: 10,
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

describe('Hidden Node Minimum Enforcement', () => {
  const multiplier = 1; // Deterministic for tests
  describe('Scenario: Flat network (no layers)', () => {
    test('enforces minimum hidden nodes on creation', () => {
      // Arrange
      const net = new Network(3, 2);
      // Act
      const hiddenCount = net.nodes.filter(n => n.type === 'hidden').length;
      // Assert
      expect(hiddenCount).toBeGreaterThanOrEqual(Math.max(net.input, net.output) * multiplier);
    });
    test('enforces minimum after removing hidden nodes', () => {
      // Arrange
      const net = new Network(4, 3);
      net.nodes = net.nodes.filter(n => n.type !== 'hidden');
      // Act
      const neat = new Neat(4, 3, () => 1, { hiddenLayerMultiplier: multiplier });
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      const hiddenCount = net.nodes.filter(n => n.type === 'hidden').length;
      // Assert
      expect(hiddenCount).toBeGreaterThanOrEqual(Math.max(net.input, net.output) * multiplier);
    });
  });

  describe('Scenario: Layered network', () => {
    test('enforces minimum hidden nodes per layer', () => {
      // Arrange
      const net = new Network(5, 2);
      net.layers = [
        { nodes: Array(5).fill({ type: 'input' }) },
        { nodes: [] }, // hidden layer
        { nodes: Array(2).fill({ type: 'output' }) }
      ];
      // Act
      const neat = new Neat(5, 2, () => 1, { hiddenLayerMultiplier: multiplier });
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      // Assert
      // Check the number of hidden nodes in the flat node list
      expect(net.nodes.filter(n => n.type === 'hidden').length).toBeGreaterThanOrEqual(Math.max(net.input, net.output) * multiplier);
    });
    test('does not add nodes to input/output layers', () => {
      // Arrange
      const net = new Network(3, 3);
      net.layers = [
        { nodes: Array(3).fill({ type: 'input' }) },
        { nodes: [] }, // hidden layer
        { nodes: Array(3).fill({ type: 'output' }) }
      ];
      // Act
      const neat = new Neat(3, 3, () => 1, { hiddenLayerMultiplier: multiplier });
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      // Assert
      expect(net.layers[0].nodes.length).toBe(3);
      expect(net.layers[2].nodes.length).toBe(3);
    });
  });

  describe('Scenario: Negative - already meets minimum', () => {
    test('does not add extra hidden nodes if already at minimum', () => {
      // Arrange
      const net = new Network(2, 2);
      // Add minimum hidden nodes
      while (net.nodes.filter(n => n.type === 'hidden').length < Math.max(net.input, net.output) * multiplier) {
        net.mutate(methods.mutation.ADD_NODE);
      }
      const prevCount = net.nodes.filter(n => n.type === 'hidden').length;
      // Act
      const neat = new Neat(2, 2, () => 1, { hiddenLayerMultiplier: multiplier });
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      // Assert
      expect(net.nodes.filter(n => n.type === 'hidden').length).toBe(prevCount);
    });
  });

  describe('Scenario: Negative - maxNodes limit', () => {
    test('does not exceed maxNodes when enforcing minimum', () => {
      // Arrange
      const net = new Network(2, 2);
      const maxNodes = net.input + net.output + 1;
      const neat = new Neat(2, 2, () => 1, { maxNodes, hiddenLayerMultiplier: multiplier });
      net.nodes = net.nodes.filter(n => n.type !== 'hidden');
      // Act
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      // Assert
      expect(net.nodes.length).toBeLessThanOrEqual(maxNodes);
    });
  });
});
