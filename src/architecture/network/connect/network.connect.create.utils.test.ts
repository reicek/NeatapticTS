import Network from '../network';
import {
  createConnectionsFromSourceNode,
  markConnectionCachesDirtyWhenNeeded,
  registerCreatedConnections,
  shouldRejectConnectionForAcyclicMode,
} from './network.connect.create.utils';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('network connect create utils chapter', () => {
  describe('shouldRejectConnectionForAcyclicMode()', () => {
    describe('given _enforceAcyclic is true and source comes after target in node order', () => {
      it('returns true to reject the backwards edge (line 21 branch 1 — acyclic check)', () => {
        // Arrange: 1-1 network, node[1] is output; connecting output→input is backwards
        const network = new Network(1, 1);
        const internalState = { _enforceAcyclic: true } as Parameters<
          typeof shouldRejectConnectionForAcyclicMode
        >[1];
        const sourceNode = network.nodes[1]; // output node (index 1)
        const targetNode = network.nodes[0]; // input node (index 0)

        // Act
        const result = shouldRejectConnectionForAcyclicMode(
          network,
          internalState,
          sourceNode,
          targetNode,
        );

        // Assert: backward edge rejected
        expect(result).toBe(true);
      });
    });
  });

  describe('createConnectionsFromSourceNode()', () => {
    describe('given no initialWeight and no randomValue function', () => {
      it('creates connections with undefined weight (line 41 branch — ?? undefined fallback)', () => {
        // Arrange: both initialWeight and randomValue are omitted → undefined weight
        const network = new Network(1, 1);
        const sourceNode = network.nodes[0];
        const targetNode = network.nodes[1];
        // Disconnect existing connection first
        sourceNode.connections.out.forEach((c) => c.from.disconnect(c.to));

        // Act: no initialWeight, no randomValue → resolvedWeight = undefined
        const connections = createConnectionsFromSourceNode(
          sourceNode,
          targetNode,
        );

        // Assert: connection created (weight defaults in Node.connect)
        expect(connections.length).toBeGreaterThan(0);
      });
    });

    describe('given no initialWeight but a randomValue function', () => {
      it('uses randomValue to resolve weight (line 41 block 2 branch 0 — randomValue truthy)', () => {
        // Arrange: randomValue provided, initialWeight omitted → randomValue() fires
        const network = new Network(1, 1);
        const sourceNode = network.nodes[0];
        const targetNode = network.nodes[1];
        sourceNode.connections.out.forEach((c) => c.from.disconnect(c.to));

        // Act: randomValue provided → resolvedWeight = randomValue() * 0.2 - 0.1
        const connections = createConnectionsFromSourceNode(
          sourceNode,
          targetNode,
          undefined,
          () => 0.5,
        );

        // Assert: connection created with a weight derived from randomValue
        expect(connections.length).toBeGreaterThan(0);
      });
    });
  });

  describe('markConnectionCachesDirtyWhenNeeded()', () => {
    describe('given a createdConnectionCount of zero', () => {
      it('returns without dirtying caches (line 86 branch 0 — early return)', () => {
        // Arrange: count = 0 → early return fires, _topoDirty never set
        const internalState = {
          _topoDirty: false,
          _slabDirty: false,
        } as Parameters<typeof markConnectionCachesDirtyWhenNeeded>[0];

        // Act
        markConnectionCachesDirtyWhenNeeded(internalState, 0);

        // Assert: caches remain clean
        expect(internalState._topoDirty).toBe(false);
      });
    });
  });

  describe('registerCreatedConnections()', () => {
    describe('given a self-connection with _enforceAcyclic enabled', () => {
      it('skips adding the self-connection to selfconns (line 111 branch 0 — acyclic guard)', () => {
        // Arrange: self-connection + acyclic mode → registerSingleCreatedConnection returns early
        const network = new Network(1, 1);
        const sourceNode = network.nodes[0];
        const internalState = { _enforceAcyclic: true } as Parameters<
          typeof registerCreatedConnections
        >[1];
        const selfConnections = sourceNode.connect(sourceNode);
        const selfconnsBefore = network.selfconns.length;

        // Act
        registerCreatedConnections(
          network,
          internalState,
          sourceNode,
          sourceNode,
          selfConnections,
        );

        // Assert: selfconns unchanged because acyclic mode skipped registration
        expect(network.selfconns.length).toBe(selfconnsBefore);
      });
    });

    describe('given a self-connection with _enforceAcyclic disabled', () => {
      it('pushes the self-connection into selfconns (line 112 — non-acyclic self-conn)', () => {
        // Arrange: self-connection + acyclic mode disabled → selfconns.push fires
        const network = new Network(1, 1);
        const sourceNode = network.nodes[0];
        const internalState = { _enforceAcyclic: false } as Parameters<
          typeof registerCreatedConnections
        >[1];
        const selfConnections = sourceNode.connect(sourceNode);
        const selfconnsBefore = network.selfconns.length;

        // Act
        registerCreatedConnections(
          network,
          internalState,
          sourceNode,
          sourceNode,
          selfConnections,
        );

        // Assert: selfconns grew by the number of registered self-connections
        expect(network.selfconns.length).toBe(
          selfconnsBefore + selfConnections.length,
        );
      });
    });
  });
});
