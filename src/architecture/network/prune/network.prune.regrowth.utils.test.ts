import Node from '../../node/node';
import { maybeRunRegrowth } from './network.prune.regrowth.utils';
import type Network from '../../network/network';
import type { RegrowthPlanContext } from '../network.types';

type RegrowthTestNetwork = {
  _enforceAcyclic?: boolean;
  _rand: () => number;
  connect: (sourceNode: Node, targetNode: Node) => void;
  connections: Array<{ from: Node; to: Node }>;
  nodes: Node[];
};

describe('network regrowth utility chapter', () => {
  describe('maybeRunRegrowth', () => {
    describe('given regrowth is disabled', () => {
      it('leaves the connection set unchanged', () => {
        // Arrange
        const network = createRegrowthNetwork({ randomSequence: [0.1, 0.8] });

        // Act
        maybeRunRegrowth(
          network as unknown as Network,
          createRegrowthContext({
            regrowFraction: 0,
            desiredRemainingConnections: 1,
          }),
        );

        // Assert
        expect(network.connections).toHaveLength(0);
      });
    });

    describe('given the requested regrowth count rounds down to zero', () => {
      it('does not schedule any new edges', () => {
        // Arrange
        const network = createRegrowthNetwork({ randomSequence: [0.1, 0.8] });

        // Act
        maybeRunRegrowth(
          network as unknown as Network,
          createRegrowthContext({
            prunedConnectionCount: 1,
            regrowFraction: 0.2,
            desiredRemainingConnections: 1,
          }),
        );

        // Assert
        expect(network.connections).toHaveLength(0);
      });
    });

    describe('given the random pair is valid', () => {
      it('adds one new connection and stops at the desired count', () => {
        // Arrange
        const network = createRegrowthNetwork({ randomSequence: [0.1, 0.8] });

        // Act
        maybeRunRegrowth(
          network as unknown as Network,
          createRegrowthContext(),
        );

        // Assert
        expect({
          connectionCount: network.connections.length,
          edge: [
            network.nodes.indexOf(network.connections[0].from),
            network.nodes.indexOf(network.connections[0].to),
          ],
        }).toEqual({ connectionCount: 1, edge: [0, 2] });
      });
    });

    describe('given the network has no nodes', () => {
      it('exhausts the attempt budget without adding edges', () => {
        // Arrange – empty nodes → pickRandomNode returns undefined → line 124 TRUE arm
        const network = createRegrowthNetwork({
          nodeCount: 0,
          randomSequence: [0.5],
        });

        // Act
        maybeRunRegrowth(
          network as unknown as Network,
          createRegrowthContext({ desiredRemainingConnections: 1 }),
        );

        // Assert
        expect(network.connections).toHaveLength(0);
      });
    });

    describe('given the network has no _rand property', () => {
      it('uses Math.random as the fallback RNG without throwing', () => {
        // Arrange – no _rand → ?? Math.random fallback (line 142)
        const connections: Array<{ from: Node; to: Node }> = [];
        const nodeA = new Node('hidden');
        const nodeB = new Node('hidden');
        const networkWithoutRand = {
          connect: (from: Node, to: Node) => {
            connections.push({ from, to });
          },
          connections,
          nodes: [nodeA, nodeB],
          // _rand intentionally absent
        };

        // Act – spy Math.random so we can verify the fallback was used
        const originalRandom = Math.random;
        let mathRandomCalled = false;
        Math.random = () => {
          mathRandomCalled = true;
          return originalRandom();
        };
        try {
          maybeRunRegrowth(
            networkWithoutRand as unknown as Network,
            createRegrowthContext({ desiredRemainingConnections: 1 }),
          );
        } finally {
          Math.random = originalRandom;
        }

        // Assert
        expect(mathRandomCalled).toBe(true);
      });
    });

    describe('given every sampled candidate is invalid', () => {
      it('stops after exhausting the attempt budget without adding edges', () => {
        // Arrange
        const network = createRegrowthNetwork({
          existingEdges: [[0, 2]],
          enforceAcyclic: true,
          randomSequence: [0.1, 0.1, 0.1, 0.8, 0.8, 0.1],
        });

        // Act
        maybeRunRegrowth(
          network as unknown as Network,
          createRegrowthContext({ desiredRemainingConnections: 2 }),
        );

        // Assert
        expect(network.connections).toHaveLength(1);
      });
    });
  });
});

function createRegrowthContext(
  overrides?: Partial<RegrowthPlanContext>,
): RegrowthPlanContext {
  return {
    desiredRemainingConnections: 1,
    prunedConnectionCount: 1,
    regrowFraction: 1,
    ...overrides,
  };
}

function createRegrowthNetwork(input: {
  existingEdges?: Array<[number, number]>;
  enforceAcyclic?: boolean;
  nodeCount?: number;
  randomSequence: number[];
}): RegrowthTestNetwork {
  const nodes = Array.from(
    { length: input.nodeCount ?? 3 },
    () => new Node('hidden'),
  );
  const connections = (input.existingEdges ?? []).map(
    ([fromIndex, toIndex]) =>
      ({ from: nodes[fromIndex], to: nodes[toIndex] }) as never,
  );
  let randomIndex = 0;

  return {
    _enforceAcyclic: input.enforceAcyclic,
    _rand: () => {
      const randomValue =
        input.randomSequence[randomIndex % input.randomSequence.length];
      randomIndex += 1;
      return randomValue;
    },
    connect: (sourceNode: Node, targetNode: Node) => {
      connections.push({ from: sourceNode, to: targetNode } as never);
    },
    connections,
    nodes,
  };
}
