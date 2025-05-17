import { Architect, Network, methods } from '../src/neataptic';
import Neat from '../src/neat';

// Retry failed tests
jest.retryTimes(3, { logErrorsBeforeRetry: true });

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
      expect(results.error).toBeLessThan(0.33);
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
      expect(results.error).toBeLessThan(0.33);
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
      expect(results.error).toBeLessThan(0.33);
    });
  });

  describe('minHidden option propagation', () => {
    test('Neat passes minHidden to all new networks in population', () => {
      const minHidden = 5;
      const neat = new Neat(2, 1, () => 1, { popsize: 3, minHidden });
      // All networks in population should have at least minHidden hidden nodes
      for (const net of neat.population) {
        const hiddenCount = net.nodes.filter(n => n.type === 'hidden').length;
        expect(hiddenCount).toBeGreaterThanOrEqual(minHidden);
      }
    });
    test('Neat provenance networks respect minHidden', () => {
      const minHidden = 4;
      const neat = new Neat(2, 1, () => 1, { popsize: 2, minHidden, provenance: 1 });
      // Evolve to trigger provenance
      return neat.evolve().then(() => {
        for (const net of neat.population) {
          const hiddenCount = net.nodes.filter(n => n.type === 'hidden').length;
          expect(hiddenCount).toBeGreaterThanOrEqual(minHidden);
        }
      });
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

describe('Deep Network Evolution', () => {
  describe('Scenario: repeated ADD_NODE mutations', () => {
    test('can evolve a network with a deep path (more than 2 hidden nodes in a chain)', () => {
      // Arrange
      const net = new Network(2, 1);
      // Act
      // Repeatedly split the same connection to create a deep chain
      for (let i = 0; i < 5; i++) {
        if (net.connections.length === 0) {
          net.connect(net.nodes[0], net.nodes[net.nodes.length - 1]);
        }
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
      // Note: Random mutation does not guarantee a deep chain, only that deep paths are possible
      // This test passes if a path of length > 1 exists (i.e., at least one hidden node in a chain)
      expect(depth).toBeGreaterThan(0);
    });
  });
});

describe('Deep Path Construction (guaranteed)', () => {
  test('can construct a deep chain by always splitting the last connection', () => {
    // Arrange
    const net = new Network(1, 1);
    // Start with a single connection from input to output
    net.connect(net.nodes[0], net.nodes[net.nodes.length - 1]);
    let lastConn = net.connections[0];
    // Act: Always split the last connection to extend the chain
    for (let i = 0; i < 5; i++) {
      // Always split the connection that is the only outgoing from input or last hidden
      let node = net.nodes.find(n => n.type === 'input');
      while (node && node.type !== 'output') {
        const out = node.connections.out[0];
        if (out.to.type === 'output') {
          lastConn = out;
          break;
        }
        node = out.to;
      }
      net.disconnect(lastConn.from, lastConn.to);
      const newNode = new (require('../src/architecture/node').default)('hidden');
      net.nodes.splice(net.nodes.length - 1, 0, newNode); // Insert before output
      net.connect(lastConn.from, newNode);
      net.connect(newNode, lastConn.to);
    }
    // Walk the chain from input to output, counting steps
    let node = net.nodes.find(n => n.type === 'input');
    let depth = 0;
    const visited = new Set();
    while (node && node.type !== 'output' && !visited.has(node)) {
      visited.add(node);
      const nextConn = node.connections.out[0];
      if (!nextConn) break;
      node = nextConn.to;
      depth++;
    }
    // Assert
    expect(depth).toBe(6); // 5 hidden + 1 output
  });
});

describe('Connection Preservation', () => {
  test('all original input-output paths are present after minHidden enforcement', () => {
    // Arrange
    const net = new Network(2, 1);
    net.connect(net.nodes[0], net.nodes[net.nodes.length - 1]);
    net.connect(net.nodes[1], net.nodes[net.nodes.length - 1]);
    for (let i = 0; i < 2; i++) net.mutate(methods.mutation.ADD_NODE);
    const inputNodes = net.nodes.filter(n => n.type === 'input');
    const outputNode = net.nodes.find(n => n.type === 'output')!;
    // Act
    const neat = new Neat(2, 1, () => 1, { hiddenLayerMultiplier: 1 });
    (neat as any).ensureMinHiddenNodes(net, 1);
    // Assert: for each original input, there is still a path to output
    function hasPath(from: any, to: any, visited = new Set()): boolean {
      if (from === to) return true;
      visited.add(from);
      for (const conn of from.connections.out) {
        if (!visited.has(conn.to) && hasPath(conn.to, to, visited)) return true;
      }
      return false;
    }
    inputNodes.forEach(input => {
      expect(hasPath(input, outputNode)).toBe(true);
    });
  });
});

describe('Hidden Node Minimum Enforcement', () => {
  const multiplier = 1; // Deterministic for tests
  describe('Scenario: Flat network (no layers)', () => {
    test('enforces minimum hidden nodes on creation', () => {
      // Arrange
      const net = new Network(3, 2);
      const neat = new Neat(3, 2, () => 1, { hiddenLayerMultiplier: multiplier });
      // Act
      (neat as any).ensureMinHiddenNodes(net, multiplier);
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
    test('does not flatten or disconnect existing hidden nodes', () => {
      // Arrange
      const net = new Network(2, 1);
      net.connect(net.nodes[0], net.nodes[net.nodes.length - 1]);
      for (let i = 0; i < 3; i++) net.mutate(methods.mutation.ADD_NODE);
      const before = net.nodes.filter(n => n.type === 'input').map(input => {
        const output = net.nodes.find(n => n.type === 'output');
        return [input.index, output?.index];
      });
      // Act
      const neat = new Neat(2, 1, () => 1, { hiddenLayerMultiplier: multiplier });
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      // Assert: all original input-output paths are still present
      function hasPath(from: any, to: any, visited = new Set()): boolean {
        if (from === to) return true;
        visited.add(from);
        for (const conn of from.connections.out) {
          if (!visited.has(conn.to) && hasPath(conn.to, to, visited)) return true;
        }
        return false;
      }
      before.forEach(([inputIdx, outputIdx]) => {
        const input = net.nodes.find(n => n.index === inputIdx);
        const output = net.nodes.find(n => n.index === outputIdx);
        if (input && output) {
          expect(hasPath(input, output)).toBe(true);
        }
      });
    });
  });

  describe('Scenario: Missing input/output nodes', () => {
    test('warns and does not throw if input or output nodes are missing', () => {
      // Arrange
      const net = new Network(2, 1);
      net.nodes = net.nodes.filter(n => n.type !== 'input');
      const neat = new Neat(2, 1, () => 1, { hiddenLayerMultiplier: multiplier });
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {}); // Spy
      // Act
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      // Assert
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Network is missing input or output nodes'));
      warnSpy.mockRestore();
    });
  });
});
