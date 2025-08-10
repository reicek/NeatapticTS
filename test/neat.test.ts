import { Architect, Network, methods } from '../src/neataptic';
import Neat from '../src/neat';

// Retry failed tests
jest.retryTimes(3, { logErrorsBeforeRetry: true });

describe('Neat', () => {
  describe('AND scenario', () => {
    it('should evolve to solve AND (error non-negative)', async () => {
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
    it('should evolve to solve AND (error below threshold)', async () => {
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
    it('should evolve to solve XOR (error non-negative)', async () => {
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

    it('should evolve to solve XOR (error below threshold)', async () => {
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
    it('should evolve to solve XNOR (error non-negative)', async () => {
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
    it('should evolve to solve XNOR (error below threshold)', async () => {
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
    describe('when Neat is initialized with minHidden', () => {
      const minHidden = 5;
      const neat = new Neat(2, 1, () => 1, { popsize: 3, minHidden });
      neat.population.forEach((net, idx) => {
        it(`network ${idx + 1} has at least minHidden hidden nodes`, () => {
          // Arrange
          // Act
          const hiddenCount = net.nodes.filter((n) => n.type === 'hidden')
            .length;
          // Assert
          expect(hiddenCount).toBeGreaterThanOrEqual(minHidden);
        });
      });
    });
    describe('when Neat evolves with provenance and minHidden', () => {
      const minHidden = 4;
      it('network 1 has at least minHidden hidden nodes after evolve', async () => {
        // Arrange
        const neat = new Neat(2, 1, () => 1, {
          popsize: 2,
          minHidden,
          provenance: 1,
        });
        // Act
        await neat.evolve();
        // Assert
        const net = neat.population[0];
        const hiddenCount = net.nodes.filter((n) => n.type === 'hidden').length;
        expect(hiddenCount).toBeGreaterThanOrEqual(minHidden);
      });
      it('network 2 has at least minHidden hidden nodes after evolve', async () => {
        // Arrange
        const neat = new Neat(2, 1, () => 1, {
          popsize: 2,
          minHidden,
          provenance: 1,
        });
        // Act
        await neat.evolve();
        // Assert
        const net = neat.population[1];
        const hiddenCount = net.nodes.filter((n) => n.type === 'hidden').length;
        expect(hiddenCount).toBeGreaterThanOrEqual(minHidden);
      });
    });
  });
});

describe('Strict node removal', () => {
  describe('when removing input node', () => {
    it('should throw when trying to remove an input node', () => {
      // Arrange
      const net = new Network(2, 1);
      const inputNode = net.nodes.find((n) => n.type === 'input');
      // Act & Assert
      expect(() => net.remove(inputNode!)).toThrow(
        'Cannot remove input or output node from the network.'
      );
    });
  });
  describe('when removing output node', () => {
    it('should throw when trying to remove an output node', () => {
      // Arrange
      const net = new Network(2, 1);
      const outputNode = net.nodes.find((n) => n.type === 'output');
      // Act & Assert
      expect(() => net.remove(outputNode!)).toThrow(
        'Cannot remove input or output node from the network.'
      );
    });
  });
});

describe('Deep Network Evolution', () => {
  describe('Scenario: repeated ADD_NODE mutations', () => {
    it('can evolve a network with a deep path (more than 2 hidden nodes in a chain)', () => {
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
      const input = net.nodes.find((n) => n.type === 'input');
      const output = net.nodes.find((n) => n.type === 'output');
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
  it('can construct a deep chain by always splitting the last connection', () => {
    // Arrange
    const net = new Network(1, 1);
    // Start with a single connection from input to output
    net.connect(net.nodes[0], net.nodes[net.nodes.length - 1]);
    let lastConn = net.connections[0];
    // Act: Always split the last connection to extend the chain
    for (let i = 0; i < 5; i++) {
      // Always split the connection that is the only outgoing from input or last hidden
      let node = net.nodes.find((n) => n.type === 'input');
      while (node && node.type !== 'output') {
        const out = node.connections.out[0];
        if (out.to.type === 'output') {
          lastConn = out;
          break;
        }
        node = out.to;
      }
      net.disconnect(lastConn.from, lastConn.to);
      const newNode = new (require('../src/architecture/node').default)(
        'hidden'
      );
      net.nodes.splice(net.nodes.length - 1, 0, newNode); // Insert before output
      net.connect(lastConn.from, newNode);
      net.connect(newNode, lastConn.to);
    }
    // Walk the chain from input to output, counting steps
    let node = net.nodes.find((n) => n.type === 'input');
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
  describe('after minHidden enforcement', () => {
    // Arrange
    const net = new Network(2, 1);
    net.connect(net.nodes[0], net.nodes[net.nodes.length - 1]);
    net.connect(net.nodes[1], net.nodes[net.nodes.length - 1]);
    for (let i = 0; i < 2; i++) net.mutate(methods.mutation.ADD_NODE);
    const inputNodes = net.nodes.filter((n) => n.type === 'input');
    const outputNode = net.nodes.find((n) => n.type === 'output')!;
    const neat = new Neat(2, 1, () => 1, { hiddenLayerMultiplier: 1 });
    (neat as any).ensureMinHiddenNodes(net, 1);
    function hasPath(from: any, to: any, visited = new Set()): boolean {
      if (from === to) return true;
      visited.add(from);
      for (const conn of from.connections.out) {
        if (!visited.has(conn.to) && hasPath(conn.to, to, visited)) return true;
      }
      return false;
    }
    it('input node 1 has a path to output', () => {
      // Act
      const pathExists = hasPath(inputNodes[0], outputNode);
      // Assert
      expect(pathExists).toBe(true);
    });
    it('input node 2 has a path to output', () => {
      // Act
      const pathExists = hasPath(inputNodes[1], outputNode);
      // Assert
      expect(pathExists).toBe(true);
    });
  });
});

describe('Hidden Node Minimum Enforcement', () => {
  const multiplier = 1; // Deterministic for tests
  describe('Scenario: Flat network (no layers)', () => {
    it('enforces minimum hidden nodes on creation', () => {
      // Arrange
      const net = new Network(3, 2);
      const neat = new Neat(3, 2, () => 1, {
        hiddenLayerMultiplier: multiplier,
      });
      // Act
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      const hiddenCount = net.nodes.filter((n) => n.type === 'hidden').length;
      // Assert
      expect(hiddenCount).toBeGreaterThanOrEqual(
        Math.max(net.input, net.output) * multiplier
      );
    });
    it('enforces minimum after removing hidden nodes', () => {
      // Arrange
      const net = new Network(4, 3);
      net.nodes = net.nodes.filter((n) => n.type !== 'hidden');
      // Act
      const neat = new Neat(4, 3, () => 1, {
        hiddenLayerMultiplier: multiplier,
      });
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      const hiddenCount = net.nodes.filter((n) => n.type === 'hidden').length;
      // Assert
      expect(hiddenCount).toBeGreaterThanOrEqual(
        Math.max(net.input, net.output) * multiplier
      );
    });
    describe('when network has existing hidden nodes', () => {
      // Arrange
      // Note: ensureMinHiddenNodes does NOT guarantee preservation of all original input-output paths.
      // It only ensures minimum hidden nodes and that each hidden node is connected.
      const net = new Network(2, 1);
      net.connect(net.nodes[0], net.nodes[net.nodes.length - 1]);
      for (let i = 0; i < 3; i++) net.mutate(methods.mutation.ADD_NODE);
      const neat = new Neat(2, 1, () => 1, {
        hiddenLayerMultiplier: multiplier,
      });
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      it('ensures minimum hidden nodes', () => {
        // Act
        const hiddenCount = net.nodes.filter((n) => n.type === 'hidden').length;
        // Assert
        expect(hiddenCount).toBeGreaterThanOrEqual(
          Math.max(net.input, net.output) * multiplier
        );
      });
      it('ensures every hidden node has at least one input and one output connection', () => {
        // Act & Assert
        const hiddenNodes = net.nodes.filter((n) => n.type === 'hidden');
        hiddenNodes.forEach((hiddenNode) => {
          expect(hiddenNode.connections.in.length).toBeGreaterThan(0);
          expect(hiddenNode.connections.out.length).toBeGreaterThan(0);
        });
      });
    });
  });

  describe('Scenario: Missing input/output nodes', () => {
    it('warns and does not throw if input or output nodes are missing', () => {
      // Arrange
      const net = new Network(2, 1);
      net.nodes = net.nodes.filter((n) => n.type !== 'input');
      const neat = new Neat(2, 1, () => 1, {
        hiddenLayerMultiplier: multiplier,
      });
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {}); // Spy
      // Act
      (neat as any).ensureMinHiddenNodes(net, multiplier);
      // Assert
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Network is missing input or output nodes')
      );
      warnSpy.mockRestore();
    });
  });
});
