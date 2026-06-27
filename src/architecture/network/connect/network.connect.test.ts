import { Architect } from '../../../neataptic';
import Network from '../network';
import type { NetworkJSON } from '../network.types';

type BatchConnectionRequest = {
  from: Network['nodes'][number];
  to: Network['nodes'][number];
  weight?: number;
};

function resolveConnectBatch(
  network: Network,
): (
  requests: readonly BatchConnectionRequest[],
) => ReturnType<Network['connect']> {
  const connectBatch = Reflect.get(network, 'connectBatch');

  if (typeof connectBatch !== 'function') {
    throw new Error('Expected Network.connectBatch() to exist.');
  }

  return connectBatch.bind(network) as (
    requests: readonly BatchConnectionRequest[],
  ) => ReturnType<Network['connect']>;
}

function disconnectAllConnections(network: Network): void {
  for (const connection of [...network.connections]) {
    network.disconnect(connection.from, connection.to);
  }

  for (const selfConnection of [...network.selfconns]) {
    network.disconnect(selfConnection.from, selfConnection.to);
  }
}

function createStarterBatchRequests(
  network: Network,
): BatchConnectionRequest[] {
  return [
    {
      from: network.nodes[0],
      to: network.nodes[2],
    },
    {
      from: network.nodes[1],
      to: network.nodes[2],
    },
    {
      from: network.nodes[0],
      to: network.nodes[3],
    },
    {
      from: network.nodes[1],
      to: network.nodes[3],
    },
  ];
}

function summarizeHydratedTemporalExtensionBag(network: Network): {
  recurrentModuleCount: number;
  gatedBlockCount: number;
  recurrentKinds: string[];
} {
  const hydratedExtensions = Reflect.get(network, '_serializedExtensions') as
    NetworkJSON['extensions'] | undefined;
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

  describe('connectBatch()', () => {
    describe('given the request shelf is empty', () => {
      it('returns an empty array without mutating connection storage', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 5_700 });
        const connectBatch = resolveConnectBatch(network);

        disconnectAllConnections(network);

        // Act
        const createdConnections = connectBatch([]);

        // Assert
        expect({
          createdConnectionCount: createdConnections.length,
          registeredConnectionCount: network.connections.length,
          registeredSelfConnectionCount: network.selfconns.length,
        }).toEqual({
          createdConnectionCount: 0,
          registeredConnectionCount: 0,
          registeredSelfConnectionCount: 0,
        });
      });
    });

    describe('given one deterministic starter wiring request shelf', () => {
      it('matches repeated connect activation output for the same request order', () => {
        // Arrange
        const repeatedNetwork = new Network(2, 2, { seed: 5_701 });
        const batchedNetwork = new Network(2, 2, { seed: 5_701 });
        const repeatedRequests = createStarterBatchRequests(repeatedNetwork);
        const batchedRequests = createStarterBatchRequests(batchedNetwork);
        const connectBatch = resolveConnectBatch(batchedNetwork);

        disconnectAllConnections(repeatedNetwork);
        disconnectAllConnections(batchedNetwork);

        // Act
        repeatedRequests.forEach((request) => {
          repeatedNetwork.connect(request.from, request.to, request.weight);
        });
        connectBatch(batchedRequests);

        const repeatedOutput = repeatedNetwork.activate([0.25, -0.5]);
        const batchedOutput = batchedNetwork.activate([0.25, -0.5]);

        // Assert
        expect(batchedOutput).toStrictEqual(repeatedOutput);
      });
    });

    describe('given one acyclic request shelf mixes forward and backward edges', () => {
      it('returns only the created forward edges in request order', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: true,
          seed: 5_702,
        });
        const connectBatch = resolveConnectBatch(network);

        disconnectAllConnections(network);

        // Act
        const createdConnections = connectBatch([
          {
            from: network.nodes[0],
            to: network.nodes[1],
            weight: 0.25,
          },
          {
            from: network.nodes[1],
            to: network.nodes[0],
            weight: 0.5,
          },
        ]);

        // Assert
        expect({
          createdConnectionCount: createdConnections.length,
          registeredPairs: network.connections.map((connection) => {
            return `${String(network.nodes.indexOf(connection.from))}->${String(network.nodes.indexOf(connection.to))}`;
          }),
        }).toEqual({
          createdConnectionCount: 1,
          registeredPairs: ['0->1'],
        });
      });
    });

    describe('given one duplicate self-connection request resolves to no created edges', () => {
      it('skips the no-op request without growing self-connection storage', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: false,
          seed: 5_703,
        });
        const connectBatch = resolveConnectBatch(network);

        network.connect(network.nodes[0], network.nodes[0], 0.25);

        const baselineSelfConnectionCount = network.selfconns.length;

        // Act
        const createdConnections = connectBatch([
          {
            from: network.nodes[0],
            to: network.nodes[0],
            weight: 0.25,
          },
        ]);

        // Assert
        expect({
          createdConnectionCount: createdConnections.length,
          registeredSelfConnectionCount: network.selfconns.length,
        }).toEqual({
          createdConnectionCount: 0,
          registeredSelfConnectionCount: baselineSelfConnectionCount,
        });
      });
    });

    describe('given one unconstrained self-connection request is legal', () => {
      it('registers the created edge in self-connection storage', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: false,
          seed: 5_704,
        });
        const connectBatch = resolveConnectBatch(network);

        disconnectAllConnections(network);

        // Act
        const createdConnections = connectBatch([
          {
            from: network.nodes[0],
            to: network.nodes[0],
            weight: 0.25,
          },
        ]);

        // Assert
        expect({
          createdConnectionCount: createdConnections.length,
          registeredSelfConnectionCount: network.selfconns.length,
        }).toEqual({
          createdConnectionCount: 1,
          registeredSelfConnectionCount: 1,
        });
      });
    });

    describe('given one acyclic self-connection request preserves legacy skip semantics', () => {
      it('returns the created edge without registering it in self-connection storage', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: true,
          seed: 5_705,
        });
        const connectBatch = resolveConnectBatch(network);

        disconnectAllConnections(network);

        // Act
        const createdConnections = connectBatch([
          {
            from: network.nodes[0],
            to: network.nodes[0],
            weight: 0.25,
          },
        ]);

        // Assert
        expect({
          createdConnectionCount: createdConnections.length,
          registeredSelfConnectionCount: network.selfconns.length,
        }).toEqual({
          createdConnectionCount: 1,
          registeredSelfConnectionCount: 0,
        });
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
