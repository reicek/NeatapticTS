import Network from '../network';
import Node from '../../node';

function markTopologyDirty(network: Network, isDirty: boolean): void {
  Reflect.set(network, '_topoDirty', isDirty);
}

describe('network runtime scheduling diagnostics', () => {
  describe('given an acyclic network compiled its activation schedule', () => {
    describe('when the diagnostics snapshot is read', () => {
      it('reports the compiled acyclic scheduling path', () => {
        // Arrange
        const network = new Network(2, 1, {
          seed: 401,
          enforceAcyclic: true,
        });
        network.activate([0.25, 0.75]);

        // Act
        const diagnostics = network.getActivationSchedulingDiagnostics();

        // Assert
        expect(diagnostics).toEqual({
          topologyIntent: 'feed-forward',
          requestedMode: 'acyclic',
          topologyDirty: false,
          executionPath: 'compiled-schedule',
          issue: null,
          message:
            'Activation is using the compiled acyclic schedule with stable wave ordering.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 2,
          recurrentComponentCount: 0,
          stateSemantics: null,
          cycleNodeIds: [],
          suggestions: [],
        });
      });
    });
  });

  describe('given an acyclic network contains a cycle', () => {
    describe('when the diagnostics snapshot is read after activation', () => {
      it('reports the cycle fallback path and the implicated node ids', () => {
        // Arrange
        const network = new Network(1, 1, {
          seed: 402,
          enforceAcyclic: true,
        });
        const inputNode = network.nodes[0];
        const outputNode = network.nodes[1];
        const hiddenNode = new Node('hidden');
        network.nodes.push(hiddenNode);

        inputNode.connect(hiddenNode);
        hiddenNode.connect(outputNode);
        outputNode.connect(hiddenNode);
        Network.rebuildConnections(network);
        network.activate([0.5]);

        // Act
        const diagnostics = network.getActivationSchedulingDiagnostics();

        // Assert
        expect(diagnostics).toEqual({
          topologyIntent: 'feed-forward',
          requestedMode: 'acyclic',
          topologyDirty: false,
          executionPath: 'cycle-fallback-order',
          issue: 'cycle-detected',
          message:
            'Acyclic scheduling detected a cycle, so activation falls back to raw node order until the cycle is removed or recurrent mode is enabled.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 0,
          recurrentComponentCount: 0,
          stateSemantics: null,
          cycleNodeIds: [outputNode.geneId, hiddenNode.geneId],
          suggestions: [
            'Remove the reported cycle or back-connection if this network should stay feed-forward.',
            'If recurrent behavior is intentional, switch the topology intent to unconstrained or disable acyclic enforcement.',
          ],
        });
      });
    });
  });

  describe('given a recurrent network compiled its activation schedule', () => {
    describe('when the diagnostics snapshot is read', () => {
      it('reports the compiled recurrent scheduling path and carried state semantics', () => {
        // Arrange
        const network = new Network(1, 1, {
          seed: 403,
          enforceAcyclic: false,
        });
        const inputNode = network.nodes[0];
        const outputNode = network.nodes[1];
        const hiddenNode = new Node('hidden');
        network.nodes = [inputNode, hiddenNode, outputNode];

        network.connections.slice().forEach((connection) => {
          network.disconnect(connection.from, connection.to);
        });

        network.connect(inputNode, hiddenNode);
        network.connect(hiddenNode, hiddenNode);
        network.connect(hiddenNode, outputNode);
        network.activate([1]);

        // Act
        const diagnostics = network.getActivationSchedulingDiagnostics();

        // Assert
        expect(diagnostics).toEqual({
          topologyIntent: 'unconstrained',
          requestedMode: 'recurrent',
          topologyDirty: false,
          executionPath: 'compiled-schedule',
          issue: null,
          message:
            'Activation is using the compiled recurrent schedule with explicit recurrent-component steps and carried recurrent state.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 3,
          recurrentComponentCount: 1,
          stateSemantics: 'carry',
          cycleNodeIds: [],
          suggestions: [
            'Call clear() before a new independent sequence when carried recurrent state should reset.',
          ],
        });
      });
    });
  });

  describe('given a compiled schedule became stale after a structural edit', () => {
    describe('when the diagnostics snapshot is read before the next activation', () => {
      it('reports that topology is dirty and suggests rebuilding the scheduling cache', () => {
        // Arrange
        const network = new Network(2, 1, {
          seed: 404,
          enforceAcyclic: true,
        });
        network.activate([0.1, 0.2]);
        markTopologyDirty(network, true);

        // Act
        const diagnostics = network.getActivationSchedulingDiagnostics();

        // Assert
        expect(diagnostics).toEqual({
          topologyIntent: 'feed-forward',
          requestedMode: 'acyclic',
          topologyDirty: true,
          executionPath: 'compiled-schedule',
          issue: null,
          message:
            'Activation is using the compiled acyclic schedule with stable wave ordering. Topology is currently dirty, so the next activation will rebuild scheduling state before it runs.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 2,
          recurrentComponentCount: 0,
          stateSemantics: null,
          cycleNodeIds: [],
          suggestions: [
            'Run activate() or noTraceActivate() after structural edits to refresh the compiled scheduling cache.',
          ],
        });
      });
    });
  });
});
