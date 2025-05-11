import { Architect, Network, methods } from '../../src/neataptic';

describe('Genetic Operations', () => {
  describe('Crossover', () => {
    describe('Scenario: equal fitness, different hidden node counts', () => {
      let net1: Network;
      let net2: Network;
      let offspring: Network;
      let minNodes: number;
      let maxNodes: number;
      beforeAll(() => {
        // Arrange
        net1 = new Network(2, 1);
        net2 = new Network(2, 1);
        net1.mutate(methods.mutation.ADD_NODE);
        net2.mutate(methods.mutation.ADD_NODE);
        net2.mutate(methods.mutation.ADD_NODE);
        net1.score = 1;
        net2.score = 1;
        // Act
        offspring = Network.crossOver(net1, net2, true);
        minNodes = Math.min(net1.nodes.length, net2.nodes.length);
        maxNodes = Math.max(net1.nodes.length, net2.nodes.length);
      });
      test('offspring node count is at least min of parents', () => {
        // Assert
        expect(offspring.nodes.length).toBeGreaterThanOrEqual(minNodes);
      });
      test('offspring node count is at most max of parents', () => {
        // Assert
        expect(offspring.nodes.length).toBeLessThanOrEqual(maxNodes);
      });
    });
    describe('Scenario: fitter parent, different hidden node counts', () => {
      let net1: Network;
      let net2: Network;
      let offspring: Network;
      beforeAll(() => {
        // Arrange
        net1 = new Network(2, 1);
        net2 = new Network(2, 1);
        net1.mutate(methods.mutation.ADD_NODE);
        net2.mutate(methods.mutation.ADD_NODE);
        net2.mutate(methods.mutation.ADD_NODE);
        net1.score = 1;
        net2.score = 2;
        // Act
        offspring = Network.crossOver(net1, net2, false);
      });
      test('offspring node count matches fitter parent', () => {
        // Assert
        expect(offspring.nodes.length).toBe(net2.nodes.length);
      });
    });
    describe('Scenario: input/output size mismatch', () => {
      test('throws', () => {
        // Arrange
        const net1 = new Network(2, 1);
        const net2 = new Network(3, 1);
        // Act
        const act = () => Network.crossOver(net1, net2);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: output size mismatch', () => {
      test('throws', () => {
        // Arrange
        const net1 = new Network(2, 1);
        const net2 = new Network(2, 2);
        // Act
        const act = () => Network.crossOver(net1, net2);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: input size mismatch', () => {
      test('throws', () => {
        // Arrange
        const net1 = new Network(2, 1);
        const net2 = new Network(3, 1);
        // Act
        const act = () => Network.crossOver(net1, net2);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: both parents are undefined', () => {
      test('throws', () => {
        // Act
        const act = () => Network.crossOver(undefined as any, undefined as any);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: both parents are null', () => {
      test('throws', () => {
        // Act
        const act = () => Network.crossOver(null as any, null as any);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: both parents have no connections', () => {
      test('offspring node count matches parents', () => {
        // Arrange
        const net1 = new Network(2, 1);
        net1.connections = [];
        const net2 = new Network(2, 1);
        net2.connections = [];
        // Act
        const offspring = Network.crossOver(net1, net2, true);
        // Assert
        expect(offspring.nodes.length).toBe(net1.nodes.length);
      });
    });
    describe('Scenario: identical parents', () => {
      test('offspring node count matches parents', () => {
        // Arrange
        const net1 = new Network(2, 1);
        const net2 = new Network(2, 1);
        // Act
        const offspring = Network.crossOver(net1, net2, true);
        // Assert
        expect(offspring.nodes.length).toBe(net1.nodes.length);
      });
    });
    describe('Scenario: both parents have no hidden nodes', () => {
      test('offspring node count matches parents', () => {
        // Arrange
        const net1 = new Network(2, 1);
        const net2 = new Network(2, 1);
        // Act
        const offspring = Network.crossOver(net1, net2, true);
        // Assert
        expect(offspring.nodes.length).toBe(net1.nodes.length);
      });
    });
    describe('Scenario: same score and same node count', () => {
      test('offspring node count matches parents', () => {
        // Arrange
        const net1 = new Network(2, 1);
        const net2 = new Network(2, 1);
        net1.score = 1;
        net2.score = 1;
        // Act
        const offspring = Network.crossOver(net1, net2, true);
        // Assert
        expect(offspring.nodes.length).toBe(net1.nodes.length);
      });
    });
  });

  describe('Advanced Crossover Scenarios', () => {
    test('should throw if one parent is missing', () => {
      // Act & Assert
      expect(() => Network.crossOver(undefined as any, new Network(2, 1))).toThrow();
      expect(() => Network.crossOver(new Network(2, 1), undefined as any)).toThrow();
    });
  });
});
