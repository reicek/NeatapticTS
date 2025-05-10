import { Architect, Network, methods } from '../../src/neataptic';
import Node from '../../src/architecture/node';

// Helper function to verify that a given mutation method alters the network's output.
function checkMutation(method: any): void {
  const network = Architect.perceptron(2, 4, 4, 4, 2);
  network.mutate(methods.mutation.ADD_GATE);
  network.mutate(methods.mutation.ADD_BACK_CONN);
  network.mutate(methods.mutation.ADD_SELF_CONN);
  const originalOutput: number[][] = [];
  let i: number, j: number;
  for (i = 0; i <= 10; i++) {
    for (j = 0; j <= 10; j++) {
      originalOutput.push(network.activate([i / 10, j / 10]));
    }
  }
  network.mutate(method);
  const mutatedOutput: number[][] = [];
  for (i = 0; i <= 10; i++) {
    for (j = 0; j <= 10; j++) {
      mutatedOutput.push(network.activate([i / 10, j / 10]));
    }
  }
  expect(originalOutput).not.toEqual(mutatedOutput);
}

// Helper for deep approximate equality with max delta logging
function arraysClose(a: any, b: any, epsilon = 1e-5, logDelta = false): boolean {
  let maxDelta = 0;
  function compare(x: any, y: any): boolean {
    if (Array.isArray(x) && Array.isArray(y) && x.length === y.length) {
      return x.every((v, i) => compare(v, y[i]));
    }
    if (typeof x === 'number' && typeof y === 'number') {
      const delta = Math.abs(x - y);
      if (delta > maxDelta) maxDelta = delta;
      return delta < epsilon;
    }
    return x === y;
  }
  const result = compare(a, b);
  if (!result && logDelta) {
    // eslint-disable-next-line no-console
    console.log('Max delta:', maxDelta);
  }
  return result;
}

// Deterministic random seed helper
function setSeed(seed: number) {
  let s = seed;
  Math.random = function () {
    // Simple LCG for deterministic randomness
    s = Math.imul(48271, s) | 0 % 2147483647;
    return ((s & 2147483647) / 2147483647);
  };
}

describe('Mutation Effects', () => {
  let originalWarn: any;
  let originalLog: any;

  beforeEach(() => {
    setSeed(42); // Arrange: deterministic randomness
    // Arrange: Patch console.warn and console.log to suppress expected output
    originalWarn = console.warn;
    originalLog = console.log;
    console.warn = jest.fn(); // Spy
    console.log = jest.fn(); // Spy
  });

  afterEach(() => {
    // Restore console methods
    console.warn = originalWarn;
    console.log = originalLog;
  });

  describe('Network output after mutation', () => {
    Object.values(methods.mutation).forEach((method) => {
      if ((typeof method === 'function' && method.name) || (typeof method === 'object' && method !== null && 'name' in method)) {
        describe(`Scenario: mutation ${method.name}`, () => {
          test('alters network output', () => {
            // Arrange
            setSeed(42);
            const network = Architect.perceptron(2, 4, 4, 4, 2);
            network.mutate(methods.mutation.ADD_GATE);
            network.mutate(methods.mutation.ADD_BACK_CONN);
            network.mutate(methods.mutation.ADD_SELF_CONN);
            const originalOutput: number[][] = [];
            for (let i = 0; i <= 10; i++) {
              for (let j = 0; j <= 10; j++) {
                originalOutput.push(network.activate([i / 10, j / 10]));
              }
            }
            // Act
            setSeed(43);
            network.mutate(method);
            const mutatedOutput: number[][] = [];
            for (let i = 0; i <= 10; i++) {
              for (let j = 0; j <= 10; j++) {
                mutatedOutput.push(network.activate([i / 10, j / 10]));
              }
            }
            // Assert
            expect(arraysClose(originalOutput, mutatedOutput, 1e-5, true)).toBe(false);
          });

          test('does not alter output if mutation is not applied', () => {
            // Arrange
            setSeed(42);
            const network = Architect.perceptron(2, 4, 4, 4, 2);
            setSeed(42);
            network.mutate(methods.mutation.ADD_GATE);
            network.mutate(methods.mutation.ADD_BACK_CONN);
            network.mutate(methods.mutation.ADD_SELF_CONN);
            setSeed(42);
            const originalOutput: number[][] = [];
            for (let i = 0; i <= 10; i++) {
              for (let j = 0; j <= 10; j++) {
                originalOutput.push(network.activate([i / 10, j / 10]));
              }
            }
            // Act
            setSeed(42);
            const outputAfterNoMutation: number[][] = [];
            for (let i = 0; i <= 10; i++) {
              for (let j = 0; j <= 10; j++) {
                outputAfterNoMutation.push(network.activate([i / 10, j / 10]));
              }
            }
            // Assert
            expect(arraysClose(originalOutput, outputAfterNoMutation, 1e-5, true)).toBe(true);
          });
        });
      }
    });
  });

  // Mutation Edge Cases
  describe('Mutation Edge Cases', () => {
    describe('Scenario: no connections', () => {
      test('ADD_NODE does not throw and does not add a node', () => {
        // Arrange
        const net = new Network(2, 1);
        net.connections = [];
        const originalNodeCount = net.nodes.length;
        // Act
        net.mutate(methods.mutation.ADD_NODE);
        // Assert
        expect(net.nodes.length).toBe(originalNodeCount);
      });
      test('SUB_CONN does not throw and does not remove nodes', () => {
        // Arrange
        const net = new Network(2, 1);
        net.connections = [];
        const originalNodeCount = net.nodes.length;
        // Act
        net.mutate(methods.mutation.SUB_CONN);
        // Assert
        expect(net.nodes.length).toBe(originalNodeCount);
      });
    });
    describe('Scenario: all nodes already have self-connections', () => {
      test('ADD_SELF_CONN does not throw and does not add more self-conns', () => {
        // Arrange
        const net = new Network(2, 1);
        net.nodes.forEach(node => {
          if (node.type !== 'input') net.connect(node, node);
        });
        const originalSelfConns = net.selfconns.length;
        // Act
        net.mutate(methods.mutation.ADD_SELF_CONN);
        // Assert
        expect(net.selfconns.length).toBe(originalSelfConns);
      });
    });
    describe('Scenario: no self-connections', () => {
      test('SUB_SELF_CONN does not throw and does not remove nodes', () => {
        // Arrange
        const net = new Network(2, 1);
        net.selfconns = [];
        const originalNodeCount = net.nodes.length;
        // Act
        net.mutate(methods.mutation.SUB_SELF_CONN);
        // Assert
        expect(net.nodes.length).toBe(originalNodeCount);
      });
    });
    describe('Scenario: all connections are already gated', () => {
      test('ADD_GATE does not throw and does not add more gates', () => {
        // Arrange
        const net = new Network(2, 1);
        net.connections.forEach(conn => { conn.gater = net.nodes[net.nodes.length - 1]; });
        const originalGates = net.gates.length;
        // Act
        net.mutate(methods.mutation.ADD_GATE);
        // Assert
        expect(net.gates.length).toBe(originalGates);
      });
    });
    describe('Scenario: no gated connections', () => {
      test('SUB_GATE does not throw and does not remove gates', () => {
        // Arrange
        const net = new Network(2, 1);
        net.gates = [];
        const originalGates = net.gates.length;
        // Act
        net.mutate(methods.mutation.SUB_GATE);
        // Assert
        expect(net.gates.length).toBe(originalGates);
      });
    });
    describe('Scenario: no possible new connections', () => {
      test('ADD_CONN does not throw and does not add connections', () => {
        // Arrange
        const net = new Network(2, 1);
        for (let i = 0; i < net.nodes.length - net.output; i++) {
          for (let j = Math.max(i + 1, net.input); j < net.nodes.length; j++) {
            if (!net.nodes[i].isProjectingTo(net.nodes[j])) {
              net.connect(net.nodes[i], net.nodes[j]);
            }
          }
        }
        const originalConnCount = net.connections.length;
        // Act
        net.mutate(methods.mutation.ADD_CONN);
        // Assert
        expect(net.connections.length).toBe(originalConnCount);
      });
    });
    describe('Scenario: no possible new back-connections', () => {
      test('ADD_BACK_CONN does not throw and does not add connections', () => {
        // Arrange
        const net = new Network(2, 1);
        const originalConnCount = net.connections.length;
        // Act
        net.mutate(methods.mutation.ADD_BACK_CONN);
        // Assert
        expect(net.connections.length).toBe(originalConnCount);
      });
    });
    describe('Scenario: not enough nodes to swap', () => {
      test('SWAP_NODES does not throw and does not swap', () => {
        // Arrange
        const net = new Network(2, 1);
        const originalBiases = net.nodes.map(n => n.bias);
        // Act
        net.mutate(methods.mutation.SWAP_NODES);
        // Assert
        expect(net.nodes.map(n => n.bias)).toEqual(originalBiases);
      });
    });
    describe('Scenario: no hidden nodes to remove', () => {
      test('SUB_NODE does not throw and does not remove nodes', () => {
        // Arrange
        const net = new Network(2, 1);
        const originalNodeCount = net.nodes.length;
        // Act
        net.mutate(methods.mutation.SUB_NODE);
        // Assert
        expect(net.nodes.length).toBe(originalNodeCount);
      });
    });
    describe('Scenario: no connections to remove', () => {
      test('SUB_CONN does not throw and does not remove connections', () => {
        // Arrange
        const net = new Network(2, 1);
        net.connections = [];
        const originalConnCount = net.connections.length;
        // Act
        net.mutate(methods.mutation.SUB_CONN);
        // Assert
        expect(net.connections.length).toBe(originalConnCount);
      });
    });
    describe('Scenario: no removable back-connections', () => {
      test('SUB_BACK_CONN does not throw and does not remove connections', () => {
        // Arrange
        const net = new Network(2, 1);
        const originalConnCount = net.connections.length;
        // Act
        net.mutate(methods.mutation.SUB_BACK_CONN);
        // Assert
        expect(net.connections.length).toBe(originalConnCount);
      });
    });
  });

  // Remove Node scenarios
  describe('Remove Node', () => {
    describe('Scenario: removing a node not in the network', () => {
      test('should throw an error', () => {
        // Arrange
        const net = new Network(2, 1);
        const orphan = new Node('hidden');
        // Act & Assert
        expect(() => net.remove(orphan)).toThrow('Node not found in the network for removal.');
      });
    });
    describe('Scenario: removing a hidden node', () => {
      test('should decrease node count by 1', () => {
        // Arrange
        const net = new Network(2, 1);
        net.mutate(methods.mutation.ADD_NODE);
        const hidden = net.nodes.find(n => n.type === 'hidden');
        const originalNodeCount = net.nodes.length;
        // Act
        net.remove(hidden!);
        // Assert
        expect(net.nodes.length).toBe(originalNodeCount - 1);
      });
    });
    describe('Scenario: removing an input node', () => {
      test('should throw or not remove input node', () => {
        // Arrange
        const net = new Network(2, 1);
        const inputNode = net.nodes.find(n => n.type === 'input');
        // Act & Assert
        expect(() => net.remove(inputNode!)).toThrow();
      });
    });
    describe('Scenario: removing an output node', () => {
      test('should throw or not remove output node', () => {
        // Arrange
        const net = new Network(2, 1);
        const outputNode = net.nodes.find(n => n.type === 'output');
        // Act & Assert
        expect(() => net.remove(outputNode!)).toThrow();
      });
    });
    describe('Scenario: keep_gates is true', () => {
      test('should not leave any gated connections after gater is removed', () => {
        // Arrange
        const net = new Network(2, 1);
        net.mutate(methods.mutation.ADD_NODE);
        const hidden = net.nodes.find(n => n.type === 'hidden');
        const conn = net.connections[0];
        net.gate(hidden!, conn);
        methods.mutation.SUB_NODE.keep_gates = true;
        // Act
        net.remove(hidden!);
        // Assert
        const stillGated = net.connections.some(c => c.gater !== null);
        expect(stillGated).toBe(false);
        methods.mutation.SUB_NODE.keep_gates = false;
      });
    });
  });
});
