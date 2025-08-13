import Network from '../../src/architecture/network';
import mutation from '../../src/methods/mutation';
import { removeNode as gatingRemoveNode } from '../../src/architecture/network/network.gating';

/**
 * Tests for gating & node removal utilities (`network.gating.ts`).
 *
 * Covered behaviors:
 *  - gate: error when gater not in network.
 *  - gate: warning & no duplicate when gating already gated connection.
 *  - ungate: warning path when ungating connection not in global gates list.
 *  - removeNode: removes hidden node and bridges predecessors to successors.
 *  - removeNode: preserves & reassigns gating nodes when keep_gates enabled.
 */
describe('Network.gating & removal', () => {
  describe('Scenario: invalid gating node', () => {
    it('throws when gating with node from another network', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 7, enforceAcyclic: true });
      const foreign = new Network(1, 1, { seed: 8, enforceAcyclic: true });
      const conn = net.connections[0];
      // Act / Assert
      expect(() => net.gate(foreign.nodes[1], conn)).toThrow();
    });
  });

  describe('Scenario: double gating same connection', () => {
    it('does not duplicate connection in gates list after second gating attempt', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 11, enforceAcyclic: true });
      const conn = net.connections[0];
      net.gate(net.nodes[0], conn);
      // Act
      net.gate(net.nodes[0], conn);
      // Assert
      expect(net.gates.length).toBe(1);
    });
  });

  describe('Scenario: ungate non-gated connection', () => {
    it('keeps gates list length unchanged when ungating non-gated connection', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 12, enforceAcyclic: true });
      const conn = net.connections[0];
      const before = net.gates.length;
      // Act
      net.ungate(conn);
      // Assert
      expect(net.gates.length).toBe(before);
    });
  });

  describe('Scenario: remove hidden node (bridging) using public remove()', () => {
    it('removes the hidden node from network.nodes', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 21, enforceAcyclic: true });
      net.disconnect(net.nodes[0], net.nodes[1]);
      const NodeCtor = require('../../src/architecture/node').default as any;
      const hidden = new NodeCtor('hidden', undefined, (net as any)._rand);
      (net as any).nodes.splice(1, 0, hidden);
      net.connect(net.nodes[0], hidden);
      net.connect(hidden, net.nodes[net.nodes.length - 1]);
      // Act
      net.remove(hidden);
      const stillHasHidden = (net as any).nodes.includes(hidden);
      // Assert
      expect(stillHasHidden).toBe(false);
    });
  });

  describe('Scenario: gating removeNode preserves gating when keep_gates true', () => {
    it('retains at least one gated connection after removal (reassigned)', () => {
      // Arrange
      const originalKeep = mutation.SUB_NODE.keep_gates;
      mutation.SUB_NODE.keep_gates = true;
      const net = new Network(1, 1, { seed: 31, enforceAcyclic: true });
      net.disconnect(net.nodes[0], net.nodes[1]);
      const NodeCtor = require('../../src/architecture/node').default as any;
      const hidden = new NodeCtor('hidden', undefined, (net as any)._rand);
      (net as any).nodes.splice(1, 0, hidden);
      const cIn = net.connect(net.nodes[0], hidden)[0];
      const cOut = net.connect(hidden, net.nodes[net.nodes.length - 1])[0];
      net.gate(net.nodes[0], cIn);
      net.gate(net.nodes[net.nodes.length - 1], cOut);
      // Act
      gatingRemoveNode.call(net, hidden);
      const postRemovalGates = (net as any).gates.length;
      // Assert
      expect(postRemovalGates > 0).toBe(true);
      mutation.SUB_NODE.keep_gates = originalKeep;
    });
  });
});
