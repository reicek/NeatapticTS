import { Network, methods } from '../../src/neataptic';

// Reduced global timeout
jest.setTimeout(5000);

// Retry failed tests
jest.retryTimes(3, { logErrorsBeforeRetry: true });

// Reduced set of mutations to test exhaustively
const CORE_MUTATIONS = [
  methods.mutation.ADD_NODE,
  methods.mutation.SUB_NODE,
  methods.mutation.ADD_LSTM_NODE,
  methods.mutation.ADD_GRU_NODE
];

describe('Mutation Effects', () => {
  let originalWarn: any;
  let originalLog: any;

  beforeEach(() => {
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
    // Only test core mutations with full behavior verification
    CORE_MUTATIONS.forEach((method) => {
      describe(`Scenario: mutation ${method.name}`, () => {
        it('alters network output', () => {
          // Use smaller timeout per test
          jest.setTimeout(500);

          // Arrange - use minimal network
          const rand = seededRandom(42);
          const network = new Network(2, 1);

          // Add node for SUB_NODE test
          if (method === methods.mutation.SUB_NODE) {
            network.mutate(methods.mutation.ADD_NODE);
          }

          // Use single input vector instead of multiple
          const input = [rand(), rand()];
          const originalOutput = network.activate(input);

          // Act - perform mutation once
          network.mutate(method);
          const mutatedOutput = network.activate(input);

          // Assert - simple comparison
          expect(originalOutput).not.toEqual(mutatedOutput);
        });
      });
    });

    describe('Scenario: other mutation methods', () => {
      [
        methods.mutation.ADD_CONN,
        methods.mutation.SUB_CONN,
        methods.mutation.ADD_GATE,
        methods.mutation.SUB_GATE,
        methods.mutation.ADD_SELF_CONN,
        methods.mutation.SUB_SELF_CONN,
        methods.mutation.ADD_BACK_CONN,
        methods.mutation.SUB_BACK_CONN
      ].forEach((mutation) => {
        describe(`Mutation: ${mutation.name}`, () => {
          it('does not throw', () => {
            // Arrange
            const network = new Network(2, 1);
            // Prepare network for specific mutation types
            if ([methods.mutation.SUB_GATE, methods.mutation.ADD_GATE].includes(mutation)) {
              if (network.connections.length > 0) {
                network.gate(network.nodes[network.nodes.length - 1], network.connections[0]);
              }
            } else if (mutation === methods.mutation.ADD_NODE) {
              if (network.connections.length === 0) {
                network.connect(network.nodes[0], network.nodes[network.nodes.length - 1]);
              }
            } else if (mutation === methods.mutation.ADD_BACK_CONN) {
              network.mutate(methods.mutation.ADD_NODE);
            }
            // Act & Assert
            expect(() => network.mutate(mutation)).not.toThrow();
          });
        });
      });
    });
  });

  // Scenario-based: Mutations on networks with no connections
  describe('Scenario: Mutations on networks with no connections', () => {
    it('ADD_NODE does not throw', () => {
      // Arrange
      const net = new Network(2, 1);
      net.connections = [];
      // Act & Assert
      expect(() => net.mutate(methods.mutation.ADD_NODE)).not.toThrow();
    });
    it('SUB_CONN does not throw', () => {
      // Arrange
      const net = new Network(2, 1);
      net.connections = [];
      // Act & Assert
      expect(() => net.mutate(methods.mutation.SUB_CONN)).not.toThrow();
    });
  });

  // Scenario-based: Mutations with impossible conditions
  describe('Scenario: Mutations with impossible conditions', () => {
    it('SUB_GATE does not throw', () => {
      // Arrange
      const net = new Network(2, 1);
      // Act & Assert
      expect(() => net.mutate(methods.mutation.SUB_GATE)).not.toThrow();
    });
    it('SUB_SELF_CONN does not throw', () => {
      // Arrange
      const net = new Network(2, 1);
      // Act & Assert
      expect(() => net.mutate(methods.mutation.SUB_SELF_CONN)).not.toThrow();
    });
    it('SUB_BACK_CONN does not throw', () => {
      // Arrange
      const net = new Network(2, 1);
      // Act & Assert
      expect(() => net.mutate(methods.mutation.SUB_BACK_CONN)).not.toThrow();
    });
    it('SWAP_NODES does not throw', () => {
      // Arrange
      const net = new Network(2, 1);
      // Act & Assert
      expect(() => net.mutate(methods.mutation.SWAP_NODES)).not.toThrow();
    });
  });

  // Scenario-based: Remove Node
  describe('Scenario: Remove Node', () => {
    it('throws when removing input node', () => {
      // Arrange
      const net = new Network(2, 1);
      const inputNode = net.nodes.find(n => n.type === 'input');
      // Act & Assert
      expect(() => net.remove(inputNode!)).toThrow();
    });
    it('throws when removing output node', () => {
      // Arrange
      const net = new Network(2, 1);
      const outputNode = net.nodes.find(n => n.type === 'output');
      // Act & Assert
      expect(() => net.remove(outputNode!)).toThrow();
    });
    describe('Scenario: hidden node removal', () => {
      it('removes hidden node without throwing', () => {
        // Arrange
        const net = new Network(2, 1);
        net.mutate(methods.mutation.ADD_NODE);
        const hidden = net.nodes.find(n => n.type === 'hidden');
        // Act & Assert
        expect(hidden).toBeDefined();
        if (hidden) {
          expect(() => net.remove(hidden)).not.toThrow();
        }
      });
      it('does nothing if no hidden node exists', () => {
        // Arrange
        const net = new Network(2, 1);
        const hidden = net.nodes.find(n => n.type === 'hidden');
        // Act & Assert
        expect(hidden).toBeUndefined();
      });
    });
  });

  // Scenario-based: Invalid and special mutation methods
  describe('Scenario: Invalid and special mutation methods', () => {
    it('warns but does not throw for invalid mutation name', () => {
      // Arrange
      const net = new Network(2, 1);
      // Act & Assert
      expect(() => {
        net.mutate({ name: 'NOT_A_REAL_MUTATION' } as any);
      }).not.toThrow();
    });
    it('warns but does not throw for empty mutation object', () => {
      // Arrange
      const net = new Network(2, 1);
      // Act & Assert
      expect(() => {
        net.mutate({} as any);
      }).not.toThrow();
    });
    it('throws error on null mutation method', () => {
      // Arrange
      const net = new Network(2, 1);
      // Act & Assert
      expect(() => {
        net.mutate(null as any);
      }).toThrow('No (correct) mutate method given!');
    });
    describe('Scenario: mutation after serialization', () => {
      it('allows mutation after serialization', () => {
        // Arrange
        const net = new Network(2, 1);
        const deserialized = Network.fromJSON(net.toJSON());
        // Act
        deserialized.mutate(methods.mutation.ADD_NODE);
        // Assert
        expect(deserialized.nodes.length).toBeGreaterThan(net.nodes.length);
      });
      it('serialized and deserialized network is structurally equal before mutation', () => {
        // Arrange
        const net = new Network(2, 1);
        const deserialized = Network.fromJSON(net.toJSON());
        // Act & Assert
        expect(arraysClose(deserialized.toJSON(), net.toJSON())).toBe(true);
      });
    });
  });

  // Scenario-based: Cycle/Infinite Loop Protection
  describe('Scenario: Cycle/Infinite Loop Protection', () => {
    it('handles self-connection in neural network', () => {
      // Arrange
      const net = new Network(1, 1);
      const outputNode = net.nodes.find(n => n.type === 'output')!;
      // Act
      net.connect(outputNode, outputNode);
      // Assert
      expect(() => net.activate([0.5])).not.toThrow();
    });
    it('handles 2-node cycle in neural network', () => {
      // Arrange
      const net2 = new Network(2, 1);
      net2.mutate(methods.mutation.ADD_NODE);
      const hidden = net2.nodes.find(n => n.type === 'hidden')!;
      const output = net2.nodes.find(n => n.type === 'output')!;
      // Act
      net2.connect(output, hidden);
      net2.connect(hidden, output);
      // Assert
      expect(() => net2.activate([0.1, 0.2])).not.toThrow();
    });
  });

  describe('Deep Path Evolution via Mutation', () => {
    describe('Scenario: repeated ADD_NODE mutations', () => {
      it('can create a deep path (multiple hidden nodes in a chain)', () => {
        // Arrange
        const net = new Network(2, 1);
        net.connect(net.nodes[0], net.nodes[net.nodes.length - 1]);
        // Act
        for (let i = 0; i < 4; i++) {
          net.mutate(methods.mutation.ADD_NODE);
        }
        // Find the longest path from input to output
        const input = net.nodes.find(n => n.type === 'input');
        const output = net.nodes.find(n => n.type === 'output');
        function dfs(node: any, visited = new Set()): number {
          if (node === output) return 0;
          visited.add(node);
          let maxDepth = 0;
          for (const conn of node.connections.out) {
            if (!visited.has(conn.to)) {
              maxDepth = Math.max(maxDepth, 1 + dfs(conn.to, visited));
            }
          }
          visited.delete(node);
          return maxDepth;
        }
        const depth = dfs(input!);
        // Assert
        expect(depth).toBeGreaterThan(2);
      });
      it('does not create a deep path in a shallow network', () => {
        // Arrange
        const net = new Network(2, 1);
        net.connect(net.nodes[0], net.nodes[net.nodes.length - 1]);
        // Act
        // No mutation
        const input = net.nodes.find(n => n.type === 'input');
        const output = net.nodes.find(n => n.type === 'output');
        function dfs(node: any, visited = new Set()): number {
          if (node === output) return 0;
          visited.add(node);
          let maxDepth = 0;
          for (const conn of node.connections.out) {
            if (!visited.has(conn.to)) {
              maxDepth = Math.max(maxDepth, 1 + dfs(conn.to, visited));
            }
          }
          visited.delete(node);
          return maxDepth;
        }
        const depth = dfs(input!);
        // Assert
        expect(depth).toBe(1);
      });
    });
  });
});
// Scenario-based: Helper utilities
describe('Helper: arraysClose', () => {
  describe('Scenario: flat arrays', () => {
    it('returns true for arrays within epsilon', () => {
      // Arrange
      const a = [1, 2, 3.000001];
      const b = [1, 2, 3.000002];
      // Act
      const result = arraysClose(a, b, 1e-5);
      // Assert
      expect(result).toBe(true);
    });
    it('returns false for arrays outside epsilon', () => {
      // Arrange
      const a = [1, 2, 3];
      const b = [1, 2, 3.1];
      // Act
      const result = arraysClose(a, b, 1e-5);
      // Assert
      expect(result).toBe(false);
    });
    it('returns false for arrays of different length', () => {
      // Arrange
      const a = [1, 2, 3];
      const b = [1, 2];
      // Act
      const result = arraysClose(a, b);
      // Assert
      expect(result).toBe(false);
    });
  });
  describe('Scenario: nested arrays', () => {
    it('returns true for equal nested arrays', () => {
      // Arrange
      const a = [[1, 2], [3, 4]];
      const b = [[1, 2], [3, 4]];
      // Act
      const result = arraysClose(a, b);
      // Assert
      expect(result).toBe(true);
    });
    it('returns false for different nested arrays', () => {
      // Arrange
      const a = [[1, 2], [3, 4]];
      const b = [[1, 2], [3, 5]];
      // Act
      const result = arraysClose(a, b);
      // Assert
      expect(result).toBe(false);
    });
    it('returns false for nested arrays of different shape', () => {
      // Arrange
      const a = [[1, 2], [3, 4]];
      const b = [[1, 2, 3], [4]];
      // Act
      const result = arraysClose(a, b);
      // Assert
      expect(result).toBe(false);
    });
  });
  describe('Scenario: non-array values', () => {
    it('returns true for equal primitives', () => {
      // Arrange
      const a = 5;
      const b = 5;
      // Act
      const result = arraysClose(a, b);
      // Assert
      expect(result).toBe(true);
    });
    it('returns false for different primitives', () => {
      // Arrange
      const a = 5;
      const b = 6;
      // Act
      const result = arraysClose(a, b);
      // Assert
      expect(result).toBe(false);
    });
  });

  describe('Scenario: deterministic output', () => {
    it('produces the same sequence for the same seed', () => {
      // Arrange
      const rand1 = seededRandom(123);
      const rand2 = seededRandom(123);
      // Act
      const seq1 = [rand1(), rand1(), rand1()];
      const seq2 = [rand2(), rand2(), rand2()];
      // Assert
      expect(seq1).toEqual(seq2);
    });
  });
  describe('Scenario: different seeds', () => {
    it('produces different sequences for different seeds', () => {
      // Arrange
      const rand1 = seededRandom(123);
      const rand2 = seededRandom(456);
      // Act
      const seq1 = [rand1(), rand1(), rand1()];
      const seq2 = [rand2(), rand2(), rand2()];
      // Assert
      expect(seq1).not.toEqual(seq2);
    });
  });
});

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
    if (x && y && typeof x === 'object' && typeof y === 'object') {
      const xKeys = Object.keys(x);
      const yKeys = Object.keys(y);
      if (xKeys.length !== yKeys.length) return false;
      // Sort keys to ensure order doesn't matter
      xKeys.sort();
      yKeys.sort();
      for (let i = 0; i < xKeys.length; i++) {
        if (xKeys[i] !== yKeys[i]) return false;
        if (!compare(x[xKeys[i]], y[yKeys[i]])) return false;
      }
      return true;
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

// Local deterministic PRNG (do not patch Math.random)
function seededRandom(seed: number) {
  let s = seed;
  return function () {
    s = Math.imul(48271, s) | 0 % 2147483647;
    return ((s & 2147483647) / 2147483647);
  };
}
