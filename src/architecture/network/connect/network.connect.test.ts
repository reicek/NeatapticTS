import { Architect } from '../../../neataptic';
import Network from '../network';
import type { NetworkJSON } from '../network.types';

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

describe('network connect chapter', () => {
  describe('connect()', () => {
    describe('given a duplicate self-connection attempt on the same node', () => {
      it('returns an empty array when the self-connection already exists', () => {
        // Arrange: network with a hidden node; create first self-connection
        const network = new Network(1, 1);
        const hiddenNode = network.nodes[0];
        network.connect(hiddenNode, hiddenNode);

        // Act: second self-connection attempt → Node.connect returns []
        const result = network.connect(hiddenNode, hiddenNode);

        // Assert: no connection created → empty array
        expect(result).toEqual([]);
      });
    });
  });

  describe('disconnect()', () => {
    describe('given a temporal descriptor references the removed structural edge', () => {
      it('retires the hydrated temporal descriptor bag immediately', () => {
        // Arrange
        const network = Architect.lstm(1, 1, 1);
        const connectionToDisconnect =
          network.gates[0] ?? network.connections[0];

        if (!connectionToDisconnect) {
          throw new Error('Expected an LSTM fixture connection to disconnect.');
        }

        const summaryBeforeDisconnect =
          summarizeHydratedTemporalExtensionBag(network);

        // Act
        network.disconnect(
          connectionToDisconnect.from,
          connectionToDisconnect.to,
        );
        const summaryAfterDisconnect =
          summarizeHydratedTemporalExtensionBag(network);

        // Assert
        expect({
          summaryBeforeDisconnect,
          summaryAfterDisconnect,
        }).toEqual({
          summaryBeforeDisconnect: {
            recurrentModuleCount: 1,
            gatedBlockCount: 1,
            recurrentKinds: ['lstm'],
          },
          summaryAfterDisconnect: {
            recurrentModuleCount: 0,
            gatedBlockCount: 0,
            recurrentKinds: [],
          },
        });
      });
    });
  });
});
