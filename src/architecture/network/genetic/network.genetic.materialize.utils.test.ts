import Network from '../network';
import Node from '../../node';
import { materializeOffspringConnections } from './network.genetic.materialize.utils';
import type { ConnectionGene, GeneticNetwork } from '../network.types';

function createNodeWithGeneId(nodeType: Node['type'], geneId: number): Node {
  const node = new Node(nodeType);
  node.geneId = geneId;
  return node;
}

function createGeneticNetwork(input: {
  assignIndexes?: boolean;
  nodes: Node[];
  topologyIntent?: 'feed-forward' | 'unconstrained';
}): GeneticNetwork {
  const network = new Network(1, 1, {
    topologyIntent: input.topologyIntent,
  }) as unknown as GeneticNetwork;

  network.nodes = input.nodes;
  network.connections = [];
  network.selfconns = [];
  network.gates = [];

  if (input.assignIndexes !== false) {
    network.nodes.forEach((node, nodeIndex) => {
      node.index = nodeIndex;
    });
  }

  return network;
}

function createConnectionGene(input: {
  fromGeneId: number;
  gaterGeneId?: number | null;
  innovation: number;
  toGeneId: number;
  weight?: number;
}): ConnectionGene {
  return {
    enabled: true,
    fromGeneId: input.fromGeneId,
    gaterGeneId: input.gaterGeneId ?? null,
    innovation: input.innovation,
    toGeneId: input.toGeneId,
    weight: input.weight ?? 0.5,
  };
}

describe('network genetic materialize utility chapter', () => {
  describe('materializeOffspringConnections', () => {
    describe('given provisional offspring nodes include one hidden placeholder without a gene id', () => {
      describe('when inherited connections are materialized', () => {
        it('drops the placeholder and keeps the inherited interface projection', () => {
          // Arrange
          const inputNode = createNodeWithGeneId('input', 11);
          const placeholderHiddenNode = new Node('hidden');
          Reflect.set(placeholderHiddenNode, 'geneId', undefined);
          const outputNode = createNodeWithGeneId('output', 22);
          const offspring = createGeneticNetwork({
            nodes: [inputNode, placeholderHiddenNode, outputNode],
          });
          const chosenGene = createConnectionGene({
            fromGeneId: 11,
            innovation: 501,
            toGeneId: 22,
          });

          // Act
          materializeOffspringConnections(offspring, [chosenGene]);

          // Assert
          expect({
            connectionCount: offspring.connections.length,
            nodeGeneIds: offspring.nodes.map((node) => node.geneId ?? null),
          }).toEqual({
            connectionCount: 1,
            nodeGeneIds: [11, 22],
          });
        });
      });
    });

    describe('given one chosen gene references a missing endpoint gene id', () => {
      describe('when inherited connections are materialized', () => {
        it('skips the unresolved projection', () => {
          // Arrange
          const inputNode = createNodeWithGeneId('input', 31);
          const outputNode = createNodeWithGeneId('output', 32);
          const offspring = createGeneticNetwork({
            nodes: [inputNode, outputNode],
          });
          const chosenGene = createConnectionGene({
            fromGeneId: 999,
            innovation: 601,
            toGeneId: 32,
          });

          // Act
          materializeOffspringConnections(offspring, [chosenGene]);

          // Assert
          expect(offspring.connections.length).toBe(0);
        });
      });
    });

    describe('given one chosen gene references a missing gater gene id', () => {
      describe('when inherited connections are materialized', () => {
        it('keeps the connection ungated', () => {
          // Arrange
          const inputNode = createNodeWithGeneId('input', 51);
          const outputNode = createNodeWithGeneId('output', 52);
          const offspring = createGeneticNetwork({
            nodes: [inputNode, outputNode],
          });
          const chosenGene = createConnectionGene({
            fromGeneId: 51,
            gaterGeneId: 999,
            innovation: 801,
            toGeneId: 52,
          });

          // Act
          materializeOffspringConnections(offspring, [chosenGene]);

          // Assert
          expect(offspring.connections[0].gater ?? null).toBe(null);
        });
      });
    });

    describe('given the offspring connect hook declines to create a runtime edge', () => {
      describe('when inherited connections are materialized', () => {
        it('leaves the offspring connection list unchanged', () => {
          // Arrange
          const inputNode = createNodeWithGeneId('input', 61);
          const outputNode = createNodeWithGeneId('output', 62);
          const offspring = createGeneticNetwork({
            nodes: [inputNode, outputNode],
          });
          offspring.connect = jest.fn(() => []) as typeof offspring.connect;
          const chosenGene = createConnectionGene({
            fromGeneId: 61,
            innovation: 901,
            toGeneId: 62,
          });

          // Act
          materializeOffspringConnections(offspring, [chosenGene]);

          // Assert
          expect(offspring.connections.length).toBe(0);
        });
      });
    });

    describe('given two inherited hidden genes share the same preferred order across different source priorities', () => {
      describe('when inherited connections are materialized', () => {
        it('orders the rebuilt hidden nodes by source priority', () => {
          // Arrange
          const offspringInputNode = createNodeWithGeneId('input', 71);
          const offspringOutputNode = createNodeWithGeneId('output', 72);
          const offspring = createGeneticNetwork({
            nodes: [offspringInputNode, offspringOutputNode],
          });

          const firstSourceNetwork = createGeneticNetwork({
            nodes: [
              createNodeWithGeneId('input', 71),
              createNodeWithGeneId('hidden', 81),
              createNodeWithGeneId('output', 72),
            ],
          });
          const secondSourceNetwork = createGeneticNetwork({
            nodes: [
              createNodeWithGeneId('input', 71),
              createNodeWithGeneId('hidden', 82),
              createNodeWithGeneId('output', 72),
            ],
          });
          const chosenGenes = [
            createConnectionGene({
              fromGeneId: 71,
              innovation: 1001,
              toGeneId: 81,
            }),
            createConnectionGene({
              fromGeneId: 71,
              innovation: 1002,
              toGeneId: 82,
            }),
          ];

          // Act
          materializeOffspringConnections(offspring, chosenGenes, [
            firstSourceNetwork,
            secondSourceNetwork,
          ]);

          // Assert
          expect(
            offspring.nodes
              .filter((node) => node.type === 'hidden')
              .map((node) => node.geneId),
          ).toEqual([81, 82]);
        });
      });
    });
  });
});
