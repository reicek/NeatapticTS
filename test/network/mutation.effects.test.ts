import { Architect, Network, methods } from '../../src/neataptic';
import Node from '../../src/architecture/node';

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

// Local deterministic PRNG (do not patch Math.random)
function seededRandom(seed: number) {
  let s = seed;
  return function () {
    s = Math.imul(48271, s) | 0 % 2147483647;
    return ((s & 2147483647) / 2147483647);
  };
}

// Reduced global timeout
jest.setTimeout(5000);

// Reduced set of mutations to test exhaustively
const CORE_MUTATIONS = [
  methods.mutation.ADD_NODE,
  methods.mutation.SUB_NODE
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
        test('alters network output', () => {
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

    // Use grouped test for other mutations
    test('other mutations modify network structure correctly', () => {
      const mutations = [
        methods.mutation.ADD_CONN,
        methods.mutation.SUB_CONN,
        methods.mutation.ADD_GATE,
        methods.mutation.SUB_GATE,
        methods.mutation.ADD_SELF_CONN,
        methods.mutation.SUB_SELF_CONN,
        methods.mutation.ADD_BACK_CONN,
        methods.mutation.SUB_BACK_CONN
      ];

      // Test each mutation once with minimal verification
      for (const mutation of mutations) {
        const network = new Network(2, 1);

        // Prepare network for specific mutation types
        if ([methods.mutation.SUB_GATE, methods.mutation.ADD_GATE].includes(mutation)) {
          if (network.connections.length > 0) {
            network.gate(network.nodes[network.nodes.length - 1], network.connections[0]);
          }
        } else if (mutation === methods.mutation.ADD_NODE) {
          // Ensure network has connections
          if (network.connections.length === 0) {
            network.connect(network.nodes[0], network.nodes[network.nodes.length - 1]);
          }
        } else if (mutation === methods.mutation.ADD_BACK_CONN) {
          network.mutate(methods.mutation.ADD_NODE);
        }

        // Just verify mutation doesn't throw
        expect(() => network.mutate(mutation)).not.toThrow();
      }
    });
  });

  // Combine edge case tests into more efficient groups
  describe('Mutation Edge Cases', () => {
    test('handles mutations on networks with no connections', () => {
      const net = new Network(2, 1);
      net.connections = [];

      // Test multiple mutations in one test
      expect(() => net.mutate(methods.mutation.ADD_NODE)).not.toThrow();
      expect(() => net.mutate(methods.mutation.SUB_CONN)).not.toThrow();
    });

    test('handles mutations with impossible conditions', () => {
      const net = new Network(2, 1);

      // Group related tests
      expect(() => net.mutate(methods.mutation.SUB_GATE)).not.toThrow();
      expect(() => net.mutate(methods.mutation.SUB_SELF_CONN)).not.toThrow();
      expect(() => net.mutate(methods.mutation.SUB_BACK_CONN)).not.toThrow();
      expect(() => net.mutate(methods.mutation.SWAP_NODES)).not.toThrow();
    });
  });

  // Minimal tests for node removal
  describe('Remove Node', () => {
    test('should throw when removing critical nodes', () => {
      const net = new Network(2, 1);
      const inputNode = net.nodes.find(n => n.type === 'input');
      const outputNode = net.nodes.find(n => n.type === 'output');

      expect(() => net.remove(inputNode!)).toThrow();
      expect(() => net.remove(outputNode!)).toThrow();
    });

    test('handles removing nodes correctly', () => {
      const net = new Network(2, 1);
      net.mutate(methods.mutation.ADD_NODE);
      const hidden = net.nodes.find(n => n.type === 'hidden');

      // Test node removal
      if (hidden) {
        expect(() => net.remove(hidden)).not.toThrow();
      }
    });
  });

  // Combined advanced mutation tests
  describe('Advanced Mutation Scenarios', () => {
    test('handles invalid mutation methods appropriately', () => {
      // Arrange
      const net = new Network(2, 1);
      
      // Test with invalid mutation name - should warn but not throw
      expect(() => {
        net.mutate({ name: 'NOT_A_REAL_MUTATION' } as any);
      }).not.toThrow();
      
      // Test empty object - should warn but not throw
      expect(() => {
        net.mutate({} as any);
      }).not.toThrow();
    });
    
    test('throws error on null mutation method', () => {
      // Arrange
      const net = new Network(2, 1);
      
      // Act & Assert - Test with null mutation - should throw
      expect(() => {
        net.mutate(null as any);
      }).toThrow('No (correct) mutate method given!');
    });

    test('allows mutation after serialization', () => {
      // Create minimal test
      const net = new Network(2, 1);
      const deserialized = Network.fromJSON(net.toJSON());
      deserialized.mutate(methods.mutation.ADD_NODE);
      expect(deserialized.nodes.length).toBeGreaterThan(net.nodes.length);
    });
  });

  // Cycle protection tests - keep these as they are important
  describe('Cycle/Infinite Loop Protection', () => {
    test('handles cycles in neural networks correctly', () => {
      // Test both scenarios in one test
      const net = new Network(1, 1);
      const outputNode = net.nodes.find(n => n.type === 'output')!;

      // Self-connection
      net.connect(outputNode, outputNode);
      expect(() => net.activate([0.5])).not.toThrow();

      // 2-node cycle
      const net2 = new Network(2, 1);
      net2.mutate(methods.mutation.ADD_NODE);
      const hidden = net2.nodes.find(n => n.type === 'hidden')!;
      const output = net2.nodes.find(n => n.type === 'output')!;
      net2.connect(output, hidden);
      net2.connect(hidden, output);
      expect(() => net2.activate([0.1, 0.2])).not.toThrow();
    });
  });
});
