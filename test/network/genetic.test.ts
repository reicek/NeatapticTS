import { Architect, Network, methods } from '../../src/neataptic';

describe('Genetic Operations', () => {
  describe('Crossover', () => {
    describe('Scenario: equal fitness, different hidden node counts', () => {
      let net1: Network;
      let net2: Network;
      let offspring: Network;
      beforeEach(() => {
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
      });
      describe('offspring node count', () => {
        it('is at least min of parents', () => {
          // Assert
          const minNodes = Math.min(net1.nodes.length, net2.nodes.length);
          expect(offspring.nodes.length).toBeGreaterThanOrEqual(minNodes);
        });
        it('is at most max of parents', () => {
          // Assert
          const maxNodes = Math.max(net1.nodes.length, net2.nodes.length);
          expect(offspring.nodes.length).toBeLessThanOrEqual(maxNodes);
        });
      });
    });
    describe('Scenario: fitter parent, different hidden node counts', () => {
      let net1: Network;
      let net2: Network;
      let offspring: Network;
      beforeEach(() => {
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
      it('offspring node count matches fitter parent', () => {
        // Assert
        expect(offspring.nodes.length).toBe(net2.nodes.length);
      });
    });
    describe('Scenario: input/output size mismatch', () => {
      it('throws', () => {
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
      it('throws', () => {
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
      it('throws', () => {
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
      it('throws', () => {
        // Arrange
        const originalWarn = console.warn;
        console.warn = jest.fn(); // Suppress warning
        // Act
        const act = () => Network.crossOver(undefined as any, undefined as any);
        // Assert
        expect(act).toThrow();
        console.warn = originalWarn;
      });
    });
    describe('Scenario: both parents are null', () => {
      it('throws', () => {
        // Arrange
        const originalWarn = console.warn;
        console.warn = jest.fn(); // Suppress warning
        // Act
        const act = () => Network.crossOver(null as any, null as any);
        // Assert
        expect(act).toThrow();
        console.warn = originalWarn;
      });
    });
    describe('Scenario: both parents have no connections', () => {
      let net1: Network;
      let net2: Network;
      let offspring: Network;
      beforeEach(() => {
        // Arrange
        net1 = new Network(2, 1);
        net1.connections = [];
        net2 = new Network(2, 1);
        net2.connections = [];
        // Act
        offspring = Network.crossOver(net1, net2, true);
      });
      it('offspring node count matches parents', () => {
        // Assert
        expect(offspring.nodes.length).toBe(net1.nodes.length);
      });
    });
    describe('Scenario: identical parents', () => {
      let net1: Network;
      let net2: Network;
      let offspring: Network;
      beforeEach(() => {
        // Arrange
        net1 = new Network(2, 1);
        net2 = new Network(2, 1);
        // Act
        offspring = Network.crossOver(net1, net2, true);
      });
      it('offspring node count matches parents', () => {
        // Assert
        expect(offspring.nodes.length).toBe(net1.nodes.length);
      });
    });
    describe('Scenario: both parents have no hidden nodes', () => {
      let net1: Network;
      let net2: Network;
      let offspring: Network;
      beforeEach(() => {
        // Arrange
        net1 = new Network(2, 1);
        net2 = new Network(2, 1);
        // Act
        offspring = Network.crossOver(net1, net2, true);
      });
      it('offspring node count matches parents', () => {
        // Assert
        expect(offspring.nodes.length).toBe(net1.nodes.length);
      });
    });
    describe('Scenario: same score and same node count', () => {
      let net1: Network;
      let net2: Network;
      let offspring: Network;
      beforeEach(() => {
        // Arrange
        net1 = new Network(2, 1);
        net2 = new Network(2, 1);
        net1.score = 1;
        net2.score = 1;
        // Act
        offspring = Network.crossOver(net1, net2, true);
      });
      it('offspring node count matches parents', () => {
        // Assert
        expect(offspring.nodes.length).toBe(net1.nodes.length);
      });
    });
  });

  describe('Advanced Crossover Scenarios', () => {
    describe('Scenario: first parent is missing', () => {
      it('throws', () => {
        // Arrange
        const originalWarn = console.warn;
        console.warn = jest.fn(); // Suppress warning
        // Act & Assert
        expect(() => Network.crossOver(undefined as any, new Network(2, 1))).toThrow();
        console.warn = originalWarn;
      });
    });
    describe('Scenario: second parent is missing', () => {
      it('throws', () => {
        // Arrange
        const originalWarn = console.warn;
        console.warn = jest.fn(); // Suppress warning
        // Act & Assert
        expect(() => Network.crossOver(new Network(2, 1), undefined as any)).toThrow();
        console.warn = originalWarn;
      });
    });
  });
});
