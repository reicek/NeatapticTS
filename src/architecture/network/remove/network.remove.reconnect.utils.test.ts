import type Connection from '../../connection';
import Node from '../../node';
import Network from '../network';
import type {
  NetworkRemoveProps,
  NodeConnectionSnapshotContext,
  NodeRemovalContext,
} from './network.remove.utils.types';
import { reconnectBridgedPaths } from './network.remove.reconnect.utils';

function createRemovalContext(): NodeRemovalContext {
  const network = new Network(1, 1, { seed: 9901 });
  const hiddenNode = new Node(
    'hidden',
    undefined,
    Reflect.get(network, '_rand') as () => number,
  );

  network.nodes.splice(1, 0, hiddenNode);

  return {
    internalNetwork: {} as NetworkRemoveProps,
    network,
    targetNode: hiddenNode,
    targetNodeIndex: 1,
  };
}

describe('network remove reconnect utility chapter', () => {
  describe('reconnectBridgedPaths', () => {
    describe('given a reconnect candidate is missing its source endpoint', () => {
      it('skips the invalid pair without reconnecting anything', () => {
        // Arrange
        const removalContext = createRemovalContext();
        const connectSpy = jest
          .spyOn(removalContext.network, 'connect')
          .mockImplementation(() => []);
        const invalidConnection = removalContext.network.connections[0];
        if (!invalidConnection) {
          throw new Error('Expected a bootstrap connection to exist');
        }

        Reflect.set(invalidConnection, 'from', undefined);
        const snapshotContext: NodeConnectionSnapshotContext = {
          inboundConnections: [invalidConnection],
          outboundConnections: [invalidConnection],
          selfConnectionCount: 0,
        };

        // Act
        reconnectBridgedPaths(removalContext, snapshotContext);

        // Assert
        expect(connectSpy.mock.calls.length).toBe(0);
      });
    });
  });
});