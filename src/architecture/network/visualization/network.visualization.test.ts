import Network from '../network';
import { exportVisualizationGraph, toDot } from './network.visualization';
import {
  buildNodePositionMap,
  collectSortedConnections,
  connectionToVisualizationDescriptor,
  nodeToVisualizationDescriptor,
  resolveActivationName,
} from './network.visualization.utils';
import type { VisualizationGraphV1 } from './network.visualization.types';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Creates a deterministic 2-hidden-2-out MLP (seed not supported by createMLP — use constructor-seeded Network). */
function createSeededMLP(): Network {
  // createMLP does not accept a seed option; use a fixed shape export surface.
  return Network.createMLP(2, [3], 2);
}

/** Exports the default (all-options-on) graph from the seeded MLP. */
function exportDefault(): VisualizationGraphV1 {
  return exportVisualizationGraph(createSeededMLP());
}

// ---------------------------------------------------------------------------
// exportVisualizationGraph — schema shape
// ---------------------------------------------------------------------------

describe('exportVisualizationGraph()', () => {
  describe('given a seeded 2-3-2 MLP network', () => {
    describe('when exported with default options', () => {
      it('returns version 1', () => {
        // Arrange / Act
        const graph = exportDefault();
        // Assert
        expect(graph.version).toBe(1);
      });

      it('includes the correct total node count', () => {
        // Arrange
        const network = createSeededMLP();
        // Act
        const graph = exportVisualizationGraph(network);
        // Assert
        expect(graph.nodes).toHaveLength(network.nodes.length);
      });

      it('includes at least one edge', () => {
        // Arrange / Act
        const graph = exportDefault();
        // Assert
        expect(graph.edges.length).toBeGreaterThan(0);
      });

      it('exposes io.inputNodeIds with length matching network.input', () => {
        // Arrange
        const network = createSeededMLP();
        // Act
        const graph = exportVisualizationGraph(network);
        // Assert
        expect(graph.io.inputNodeIds).toHaveLength(network.input);
      });

      it('exposes io.outputNodeIds with length matching network.output', () => {
        // Arrange
        const network = createSeededMLP();
        // Act
        const graph = exportVisualizationGraph(network);
        // Assert
        expect(graph.io.outputNodeIds).toHaveLength(network.output);
      });

      it('assigns role "input" to nodes in io.inputNodeIds', () => {
        // Arrange
        const graph = exportDefault();
        const inputIdSet = new Set(graph.io.inputNodeIds);
        // Act
        const inputRoleNodes = graph.nodes.filter((node) =>
          inputIdSet.has(node.id),
        );
        // Assert
        expect(inputRoleNodes.every((node) => node.role === 'input')).toBe(
          true,
        );
      });

      it('assigns role "output" to nodes in io.outputNodeIds', () => {
        // Arrange
        const graph = exportDefault();
        const outputIdSet = new Set(graph.io.outputNodeIds);
        // Act
        const outputRoleNodes = graph.nodes.filter((node) =>
          outputIdSet.has(node.id),
        );
        // Assert
        expect(outputRoleNodes.every((node) => node.role === 'output')).toBe(
          true,
        );
      });

      it('assigns role "hidden" to non-IO nodes', () => {
        // Arrange
        const graph = exportDefault();
        const ioIdSet = new Set([
          ...graph.io.inputNodeIds,
          ...graph.io.outputNodeIds,
        ]);
        // Act
        const hiddenNodes = graph.nodes.filter((node) => !ioIdSet.has(node.id));
        // Assert
        expect(hiddenNodes.every((node) => node.role === 'hidden')).toBe(true);
      });

      it('includes bias on every node', () => {
        // Arrange / Act
        const graph = exportDefault();
        // Assert
        expect(graph.nodes.every((node) => node.bias !== undefined)).toBe(true);
      });

      it('includes weight on every edge', () => {
        // Arrange / Act
        const graph = exportDefault();
        // Assert
        expect(graph.edges.every((edge) => edge.weight !== undefined)).toBe(
          true,
        );
      });

      it('includes an activation name on every node', () => {
        // Arrange / Act
        const graph = exportDefault();
        // Assert
        expect(
          graph.nodes.every(
            (node) =>
              typeof node.activation === 'string' && node.activation.length > 0,
          ),
        ).toBe(true);
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Determinism
  // ---------------------------------------------------------------------------

  describe('given the same seeded network exported twice', () => {
    describe('when comparing node arrays', () => {
      it('produces identical node ids in the same order', () => {
        // Arrange
        const network = createSeededMLP();
        // Act
        const firstGraph = exportVisualizationGraph(network);
        const secondGraph = exportVisualizationGraph(network);
        // Assert
        expect(firstGraph.nodes.map((n) => n.id)).toEqual(
          secondGraph.nodes.map((n) => n.id),
        );
      });
    });

    describe('when comparing edge arrays', () => {
      it('produces identical (from, to) pairs in the same order', () => {
        // Arrange
        const network = createSeededMLP();
        // Act
        const firstGraph = exportVisualizationGraph(network);
        const secondGraph = exportVisualizationGraph(network);
        // Assert
        expect(firstGraph.edges.map((edge) => [edge.from, edge.to])).toEqual(
          secondGraph.edges.map((edge) => [edge.from, edge.to]),
        );
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Ordering
  // ---------------------------------------------------------------------------

  describe('given a network with multiple nodes', () => {
    describe('when checking node ordering', () => {
      it('nodes are sorted by gene id ascending', () => {
        // Arrange / Act
        const graph = exportDefault();
        const nodeIds = graph.nodes.map((n) => n.id);
        // Assert
        expect(nodeIds).toEqual([...nodeIds].toSorted((a, b) => a - b));
      });
    });

    describe('when checking edge ordering', () => {
      it('edges are sorted by from-gene-id then to-gene-id ascending', () => {
        // Arrange / Act
        const graph = exportDefault();
        const edgePairs = graph.edges.map((edge) => [edge.from, edge.to]);
        const sortedPairs = [...edgePairs].toSorted(
          ([firstFrom, firstTo], [secondFrom, secondTo]) => {
            const fromDelta = firstFrom - secondFrom;
            return fromDelta !== 0 ? fromDelta : firstTo - secondTo;
          },
        );
        // Assert
        expect(edgePairs).toEqual(sortedPairs);
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Option flags
  // ---------------------------------------------------------------------------

  describe('given includeBiases: false', () => {
    describe('when exported', () => {
      it('omits bias from all nodes', () => {
        // Arrange
        const network = createSeededMLP();
        // Act
        const graph = exportVisualizationGraph(network, {
          includeBiases: false,
        });
        // Assert
        expect(graph.nodes.every((node) => node.bias === undefined)).toBe(true);
      });
    });
  });

  describe('given includeWeights: false', () => {
    describe('when exported', () => {
      it('sets weight to 0 on all edges', () => {
        // Arrange
        const network = createSeededMLP();
        // Act
        const graph = exportVisualizationGraph(network, {
          includeWeights: false,
        });
        // Assert
        expect(graph.edges.every((edge) => edge.weight === 0)).toBe(true);
      });
    });
  });

  describe('given disabled edges are explicitly requested', () => {
    describe('when exported', () => {
      it('retains disabled edges in the schema', () => {
        const network = new Network(2, 1);
        const disabledConnection = network.connections[0];
        disabledConnection.enabled = false;

        const graph = exportVisualizationGraph(network, {
          includeDisabledEdges: true,
        });

        expect(graph.edges.some((edge) => edge.enabled === false)).toBe(true);
      });
    });
  });

  describe('given a network with a self-connection', () => {
    describe('when exported', () => {
      it('labels the self-connection edge kind as "self"', () => {
        // Arrange — unconstrained network so self-connections are not blocked.
        const network = new Network(2, 1);
        // Manually add a hidden node by connecting nodes so there is a non-IO node.
        const anyNode = network.nodes[0];
        network.connect(anyNode, anyNode);
        // Act
        const graph = exportVisualizationGraph(network);
        // Assert
        expect(graph.edges.some((edge) => edge.kind === 'self')).toBe(true);
      });
    });
  });

  describe('given a feed-forward network', () => {
    describe('when exported', () => {
      it('sets metadata.mode to "acyclic"', () => {
        // Arrange
        const network = Network.createMLP(2, [3], 1);
        // Act
        const graph = exportVisualizationGraph(network);
        // Assert
        expect(graph.metadata?.mode).toBe('acyclic');
      });
    });
  });
});

// ---------------------------------------------------------------------------
// toDot()
// ---------------------------------------------------------------------------

describe('toDot()', () => {
  describe('given a valid VisualizationGraphV1', () => {
    describe('when converted to DOT', () => {
      it('returns a string containing "digraph"', () => {
        // Arrange
        const graph = exportDefault();
        // Act
        const dot = toDot(graph);
        // Assert
        expect(dot).toContain('digraph');
      });

      it('contains a -> for each edge', () => {
        // Arrange
        const graph = exportDefault();
        // Act
        const dot = toDot(graph);
        const arrowCount = (dot.match(/->/g) ?? []).length;
        // Assert
        expect(arrowCount).toBe(graph.edges.length);
      });

      it('contains a line for each node', () => {
        // Arrange
        const graph = exportDefault();
        // Act
        const dot = toDot(graph);
        // Assert — each node id should appear at least once in the DOT string
        expect(graph.nodes.every((node) => dot.includes(String(node.id)))).toBe(
          true,
        );
      });

      it('includes "invtriangle" for input nodes', () => {
        // Arrange
        const graph = exportDefault();
        // Act
        const dot = toDot(graph);
        // Assert
        expect(dot).toContain('invtriangle');
      });

      it('includes "doublecircle" for output nodes', () => {
        // Arrange
        const graph = exportDefault();
        // Act
        const dot = toDot(graph);
        // Assert
        expect(dot).toContain('doublecircle');
      });

      it('falls back to hidden-node circles and omits edge labels when a node role or edge weight is missing', () => {
        const dot = toDot({
          version: 1,
          nodes: [{ id: 1, role: 'input' }, { id: 2 }],
          edges: [{ from: 1, to: 2 } as never],
          io: {
            inputNodeIds: [1],
            outputNodeIds: [],
          },
        });

        expect(dot).toContain('2 [label="2", shape=circle];\n  1 -> 2;');
      });

      it('renders disabled weighted edges as dashed DOT lines', () => {
        const dot = toDot({
          version: 1,
          nodes: [
            { id: 1, role: 'input' },
            { id: 2, role: 'output' },
          ],
          edges: [
            { enabled: false, from: 1, to: 2, weight: 0.5, kind: 'forward' },
          ],
          io: {
            inputNodeIds: [1],
            outputNodeIds: [2],
          },
        });

        expect(dot).toContain('1 -> 2 [label="0.500", style=dashed];');
      });
    });
  });
});

describe('network visualization helper utilities', () => {
  it('falls back to an unknown activation label when the squash function is unnamed', () => {
    expect(resolveActivationName({ name: undefined } as never)).toBe('unknown');
  });

  it('omits optional node label and bias fields when the source node has no label and biases are disabled', () => {
    const descriptor = nodeToVisualizationDescriptor(
      {
        bias: 0.75,
        geneId: 42,
        label: null,
        squash: { name: 'IDENTITY' },
      } as never,
      'hidden',
      false,
    );

    expect(descriptor).toEqual({
      activation: 'IDENTITY',
      id: 42,
      role: 'hidden',
    });
  });

  it('preserves node labels and bias values when the caller requests biases', () => {
    expect(
      nodeToVisualizationDescriptor(
        {
          bias: -0.5,
          geneId: 7,
          label: 'sensor-7',
          squash: { name: 'TANH' },
        } as never,
        'input',
        true,
      ),
    ).toEqual({
      activation: 'TANH',
      bias: -0.5,
      id: 7,
      label: 'sensor-7',
      role: 'input',
    });
  });

  it('builds recurrent edge descriptors from backward node positions', () => {
    const positionMap = buildNodePositionMap([
      { geneId: 1 } as never,
      { geneId: 2 } as never,
    ]);

    expect(
      connectionToVisualizationDescriptor(
        {
          enabled: true,
          from: { geneId: 2 },
          to: { geneId: 1 },
          weight: 0.25,
        } as never,
        positionMap,
        true,
      ),
    ).toEqual({
      enabled: true,
      from: 2,
      kind: 'recurrent',
      to: 1,
      weight: 0.25,
    });
  });

  it('falls back to forward edge inference when a node position is missing from the lookup map', () => {
    expect(
      connectionToVisualizationDescriptor(
        {
          enabled: true,
          from: { geneId: 9 },
          to: { geneId: 10 },
          weight: 0.4,
        } as never,
        new Map<number, number>(),
        true,
      ),
    ).toEqual({
      enabled: true,
      from: 9,
      kind: 'forward',
      to: 10,
      weight: 0.4,
    });
  });

  it('filters disabled connections when disabled-edge export is off', () => {
    const sortedConnections = collectSortedConnections(
      [
        {
          enabled: false,
          from: { geneId: 2 },
          to: { geneId: 3 },
        },
        {
          enabled: true,
          from: { geneId: 3 },
          to: { geneId: 4 },
        },
      ] as never,
      false,
    );

    expect(
      sortedConnections.map((connection) => [
        connection.from.geneId,
        connection.to.geneId,
      ]),
    ).toEqual([[3, 4]]);
  });
});
