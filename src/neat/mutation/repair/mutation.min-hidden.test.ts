import { createInnovationTracker } from '../../innovation-tracker/innovation-tracker';
import * as mutationAddConn from '../add-conn/mutation.add-conn';
import {
  chooseRandomNodeForMinHidden,
  computeMinimumHiddenSize,
  ensureHiddenNodeCountForMinHidden,
  ensureIncomingConnectionForMinHidden,
  ensureOutgoingConnectionForMinHidden,
  resolveMaxNodesForMinHidden,
  resolveMinHiddenForMinHidden,
} from './mutation.min-hidden';
import type {
  ConnectionWithMetadata,
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

describe('neat mutation min-hidden chapter', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('resolveMaxNodesForMinHidden', () => {
    describe('given the controller does not cap node growth', () => {
      it('falls back to Infinity when maxNodes is zero', () => {
        // Arrange
        const mutationController = createMutationController({
          maxNodes: 0,
        });

        // Act
        const resolvedMaxNodes = resolveMaxNodesForMinHidden(
          mutationController,
        );

        // Assert
        expect(resolvedMaxNodes).toBe(Infinity);
      });
    });
  });

  describe('resolveMinHiddenForMinHidden', () => {
    describe('given the controller does not expose a hidden-floor callback', () => {
      it('returns zero when no minimum-hidden policy is available', () => {
        // Arrange
        const genome = createGenome([
          createNode('input', 1),
          createNode('output', 2),
        ]);
        const mutationController = createMutationController({});

        // Act
        const resolvedMinimumHidden = resolveMinHiddenForMinHidden(
          genome,
          10,
          2,
          mutationController,
        );

        // Assert
        expect(resolvedMinimumHidden).toBe(0);
      });
    });
  });

  describe('computeMinimumHiddenSize', () => {
    describe('given a finite multiplier and no explicit minimum', () => {
      it('rounds the weighted endpoint total and uses that as the floor', () => {
        // Arrange
        const inputCount = 2;
        const outputCount = 1;
        const hiddenMultiplier = 1.5;

        // Act
        const computedMinimumHidden = computeMinimumHiddenSize(
          inputCount,
          outputCount,
          undefined,
          hiddenMultiplier,
        );

        // Assert
        expect(computedMinimumHidden).toBe(5);
      });
    });
  });

  describe('ensureHiddenNodeCountForMinHidden', () => {
    describe('given the add-node reuse path cannot add a new hidden node', () => {
      it('stops after the first non-progressing attempt', async () => {
        // Arrange
        const genome = createGenome([
          createNode('input', 1),
          createNode('output', 2),
        ]);
        const hiddenNodes = genome.nodes.filter((node) => node.type === 'hidden');
        const mutateAddNodeReuse = jest.fn(async () => undefined);
        const mutationController = createMutationController({
          mutateAddNodeReuse,
        });

        // Act
        await ensureHiddenNodeCountForMinHidden(
          genome,
          { hiddenNodes },
          1,
          5,
          mutationController,
        );

        // Assert
        expect({ hiddenCount: hiddenNodes.length, attempts: mutateAddNodeReuse.mock.calls.length }).toEqual({
          hiddenCount: 0,
          attempts: 1,
        });
      });
    });
  });

  describe('chooseRandomNodeForMinHidden', () => {
    describe('given no candidates are available', () => {
      it('returns null without sampling the controller RNG', () => {
        // Arrange
        const mutationController = createMutationController({});

        // Act
        const chosenNode = chooseRandomNodeForMinHidden([], mutationController);

        // Assert
        expect(chosenNode).toBeNull();
      });
    });
  });

  describe('ensureIncomingConnectionForMinHidden', () => {
    describe('given no source candidates are available', () => {
      it('leaves the hidden node disconnected', () => {
        // Arrange
        const hiddenNode = createNode('hidden', 1);
        const genome = createGenome([hiddenNode]);
        const mutationController = createMutationController({});

        // Act
        ensureIncomingConnectionForMinHidden(
          genome,
          {
            inputNodes: [],
            hiddenNodes: [hiddenNode],
          },
          hiddenNode,
          mutationController,
        );

        // Assert
        expect(genome.connections).toHaveLength(0);
      });
    });

    describe('given every candidate is rejected by the add-connection guard', () => {
      it('skips the inbound repair entirely', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNode = createNode('hidden', 2);
        const genome = createGenome([inputNode, hiddenNode]);
        const mutationController = createMutationController({});
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(false);

        // Act
        ensureIncomingConnectionForMinHidden(
          genome,
          {
            inputNodes: [inputNode],
            hiddenNodes: [hiddenNode],
          },
          hiddenNode,
          mutationController,
        );

        // Assert
        expect(genome.connections).toHaveLength(0);
      });
    });

    describe('given the random picker lands outside the legal candidate range', () => {
      it('returns without creating an inbound repair connection', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNode = createNode('hidden', 2);
        const genome = createGenome([inputNode, hiddenNode]);
        const mutationController = createMutationController({
          randomValue: 1,
        });
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(true);

        // Act
        ensureIncomingConnectionForMinHidden(
          genome,
          {
            inputNodes: [inputNode],
            hiddenNodes: [hiddenNode],
          },
          hiddenNode,
          mutationController,
        );

        // Assert
        expect(genome.connections).toHaveLength(0);
      });
    });
  });

  describe('ensureOutgoingConnectionForMinHidden', () => {
    describe('given no target candidates are available', () => {
      it('leaves the hidden node without an outbound repair path', () => {
        // Arrange
        const hiddenNode = createNode('hidden', 1);
        const genome = createGenome([hiddenNode]);
        const mutationController = createMutationController({});

        // Act
        ensureOutgoingConnectionForMinHidden(
          genome,
          {
            outputNodes: [],
            hiddenNodes: [hiddenNode],
          },
          hiddenNode,
          mutationController,
        );

        // Assert
        expect(genome.connections).toHaveLength(0);
      });
    });

    describe('given every candidate is rejected by the add-connection guard', () => {
      it('skips the outbound repair entirely', () => {
        // Arrange
        const hiddenNode = createNode('hidden', 1);
        const outputNode = createNode('output', 2);
        const genome = createGenome([hiddenNode, outputNode]);
        const mutationController = createMutationController({});
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(false);

        // Act
        ensureOutgoingConnectionForMinHidden(
          genome,
          {
            outputNodes: [outputNode],
            hiddenNodes: [hiddenNode],
          },
          hiddenNode,
          mutationController,
        );

        // Assert
        expect(genome.connections).toHaveLength(0);
      });
    });

    describe('given the random picker lands outside the legal candidate range', () => {
      it('returns without creating an outbound repair connection', () => {
        // Arrange
        const hiddenNode = createNode('hidden', 1);
        const outputNode = createNode('output', 2);
        const genome = createGenome([hiddenNode, outputNode]);
        const mutationController = createMutationController({
          randomValue: 1,
        });
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(true);

        // Act
        ensureOutgoingConnectionForMinHidden(
          genome,
          {
            outputNodes: [outputNode],
            hiddenNodes: [hiddenNode],
          },
          hiddenNode,
          mutationController,
        );

        // Assert
        expect(genome.connections).toHaveLength(0);
      });
    });
  });
});

function createMutationController(input: {
  allowRecurrent?: boolean;
  maxNodes?: number;
  minimumHiddenSize?: (multiplierOverride?: number) => number;
  mutateAddNodeReuse?: NeatControllerForMutation['_mutateAddNodeReuse'];
  randomValue?: number;
}): NeatControllerForMutation {
  return {
    population: [],
    options: {
      allowRecurrent: input.allowRecurrent,
      maxNodes: input.maxNodes,
    },
    _getRNG: () => () => input.randomValue ?? 0,
    _mutateAddNodeReuse:
      input.mutateAddNodeReuse ?? (async () => undefined),
    _mutateAddConnReuse: () => undefined,
    _invalidateGenomeCaches: () => undefined,
    _operatorStats: new Map(),
    _innovationTracker: createInnovationTracker(),
    getMinimumHiddenSize: input.minimumHiddenSize,
    selectMutationMethod: async () => null,
  };
}

function createGenome(nodes: NodeWithMetadata[]): GenomeWithMetadata {
  const genome: GenomeWithMetadata = {
    nodes,
    connections: [],
    gates: [],
    input: 1,
    output: 0,
    connect: (from, to, weight = 1) => {
      const connection: ConnectionWithMetadata = {
        from,
        to,
        weight,
      };
      from.connections.out.push(connection);
      to.connections.in.push(connection);
      genome.connections.push(connection);
      return [connection];
    },
    disconnect: (from, to) => {
      const matchingConnections = genome.connections.filter(
        (connection) => connection.from === from && connection.to === to,
      );

      genome.connections = genome.connections.filter(
        (connection) => !(connection.from === from && connection.to === to),
      );
      from.connections.out = from.connections.out.filter(
        (connection) => !matchingConnections.includes(connection),
      );
      to.connections.in = to.connections.in.filter(
        (connection) => !matchingConnections.includes(connection),
      );
    },
  };

  return genome;
}

function createNode(
  type: NodeWithMetadata['type'],
  geneId: number,
): NodeWithMetadata {
  const connections = {
    in: [],
    out: [],
    self: [],
  } as NodeWithMetadata['connections'] & {
    self: ConnectionWithMetadata[];
  };

  return {
    type,
    geneId,
    connections,
    isProjectingTo: (targetNode) =>
      connections.out.some((connection) => connection.to === targetNode),
  };
}