import Node from '../../node';
import Network from '../network';
import type { NetworkJSON } from '../network.types';
import {
  NetworkRemoveNodeNotFoundError,
  NetworkRemoveStructuralAnchorError,
} from './network.remove.errors';

function createRemoveNetwork(seed: number): Network {
  return new Network(2, 1, { seed });
}

function getNetworkRandomGenerator(network: Network): () => number {
  return Reflect.get(network, '_rand') as () => number;
}

function summarizeHydratedTemporalExtensionBag(network: Network): {
  recurrentModuleCount: number;
  gatedBlockCount: number;
  recurrentKinds: string[];
} {
  const hydratedExtensions = Reflect.get(network, '_serializedExtensions') as
    | NetworkJSON['extensions']
    | undefined;
  const extensionValues = hydratedExtensions?.values as
    | {
        recurrentModules?: Array<{ kind?: string }>;
        gatedBlocks?: Array<unknown>;
      }
    | undefined;
  const recurrentModules = Array.isArray(extensionValues?.recurrentModules)
    ? extensionValues.recurrentModules
    : [];
  const gatedBlocks = Array.isArray(extensionValues?.gatedBlocks)
    ? extensionValues.gatedBlocks
    : [];

  return {
    recurrentModuleCount: recurrentModules.length,
    gatedBlockCount: gatedBlocks.length,
    recurrentKinds: recurrentModules
      .map((recurrentModule) => recurrentModule.kind)
      .filter((kind): kind is string => typeof kind === 'string')
      .toSorted(),
  };
}

describe('network remove chapter', () => {
  describe('Network.remove()', () => {
    describe('given the removal target is an input node', () => {
      describe('when remove() is called', () => {
        it('throws the structural-anchor error', () => {
          // Arrange
          const network = createRemoveNetwork(390);
          const inputNode = network.nodes[0];

          // Act
          const removeInputNode = () => network.remove(inputNode);

          // Assert
          expect(removeInputNode).toThrow(NetworkRemoveStructuralAnchorError);
        });
      });
    });

    describe('given the removal target is an output node', () => {
      describe('when remove() is called', () => {
        it('throws the structural-anchor error', () => {
          // Arrange
          const network = createRemoveNetwork(391);
          const outputNode = network.nodes.at(-1);
          if (!outputNode) {
            throw new Error('Output node should exist');
          }

          // Act
          const removeOutputNode = () => network.remove(outputNode);

          // Assert
          expect(removeOutputNode).toThrow(NetworkRemoveStructuralAnchorError);
        });
      });
    });

    describe('given the removal target does not belong to the network', () => {
      describe('when remove() is called', () => {
        it('throws the node-not-found error', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 392 });
          const foreignNetwork = new Network(1, 1, { seed: 393 });
          const foreignNode = foreignNetwork.nodes[0];

          // Act
          const removeForeignNode = () => network.remove(foreignNode);

          // Assert
          expect(removeForeignNode).toThrow(NetworkRemoveNodeNotFoundError);
        });
      });
    });

    describe('given the removal target is a hidden bridge node', () => {
      describe('when remove() is called', () => {
        it('restores a direct predecessor-to-successor projection', () => {
          // Arrange
          const network = createRemoveNetwork(394);
          const inputNode = network.nodes[0];
          const outputNode = network.nodes.at(-1);
          if (!outputNode) {
            throw new Error('Output node should exist');
          }

          network.disconnect(inputNode, outputNode);
          const hiddenNode = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );
          network.nodes.splice(1, 0, hiddenNode);
          network.connect(inputNode, hiddenNode);
          network.connect(hiddenNode, outputNode);

          // Act
          network.remove(hiddenNode);
          const restoredProjectionExists = outputNode.connections.in.some(
            (candidateConnection) => candidateConnection.from === inputNode,
          );

          // Assert
          expect(restoredProjectionExists).toBe(true);
        });
      });
    });

    describe('given the removal target only owns gating on an external connection', () => {
      describe('when remove() is called', () => {
        it('retires the hydrated temporal descriptor bag immediately', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 395 });
          const gaterNode = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );
          network.nodes.splice(1, 0, gaterNode);

          const gatedConnection = network.connections[0];
          if (!gatedConnection) {
            throw new Error('Expected a bootstrap connection to gate.');
          }

          network.gate(gaterNode, gatedConnection);
          Reflect.set(network, '_serializedExtensions', {
            version: 1,
            values: {
              gatedBlocks: [
                {
                  blockId: 'gated:block:manual-fixture',
                  gaterGeneIds: [gaterNode.geneId ?? 0],
                  connectionInnovations: [gatedConnection.innovation],
                },
              ],
            },
          });
          const summaryBeforeRemove =
            summarizeHydratedTemporalExtensionBag(network);

          // Act
          network.remove(gaterNode);
          const summaryAfterRemove =
            summarizeHydratedTemporalExtensionBag(network);

          // Assert
          expect({
            summaryBeforeRemove,
            summaryAfterRemove,
          }).toEqual({
            summaryBeforeRemove: {
              recurrentModuleCount: 0,
              gatedBlockCount: 1,
              recurrentKinds: [],
            },
            summaryAfterRemove: {
              recurrentModuleCount: 0,
              gatedBlockCount: 0,
              recurrentKinds: [],
            },
          });
        });
      });
    });
  });
});
