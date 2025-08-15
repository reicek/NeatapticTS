import Network from '../../src/architecture/network';
import { mutateImpl } from '../../src/architecture/network/network.mutate';
import mutation from '../../src/methods/mutation';

/** Edge case mutation tests targeting early-return / warning branches. */

describe('Network.mutateImpl edge cases', () => {
  describe('Scenario: SUB_NODE with no hidden nodes', () => {
    it('leaves hidden count unchanged', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 40 });
      const before = net.nodes.filter((n) => n.type === 'hidden').length;
      // Act
      mutateImpl.call(net, mutation.SUB_NODE);
      const after = net.nodes.filter((n) => n.type === 'hidden').length;
      // Assert
      expect(after).toBe(before);
    });
  });

  describe('Scenario: ADD_CONN with no available new forward pairs', () => {
    it('does not change connection count', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 41 }); // already fully connected
      const before = net.connections.length;
      // Act
      mutateImpl.call(net, mutation.ADD_CONN);
      const after = net.connections.length;
      // Assert
      expect(after).toBe(before);
    });
  });

  describe('Scenario: SUB_SELF_CONN with no self connections', () => {
    it('keeps self connection count at zero', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 42 });
      const before = net.selfconns.length;
      // Act
      mutateImpl.call(net, mutation.SUB_SELF_CONN);
      const after = net.selfconns.length;
      // Assert
      expect(after).toBe(before);
    });
  });

  describe('Scenario: ADD_SELF_CONN early return (all nodes already have self loops)', () => {
    it('does not increase self connection count', () => {
      // Arrange
      const net = new Network(1, 2, { seed: 43, enforceAcyclic: false });
      // Add self loops for every non-input node first
      net.nodes.forEach((n, idx) => {
        if (idx >= net.input) net.connect(n, n);
      });
      const before = net.selfconns.length;
      // Act
      mutateImpl.call(net, mutation.ADD_SELF_CONN);
      const after = net.selfconns.length;
      // Assert
      expect(after).toBe(before);
    });
  });

  describe('Scenario: ADD_LSTM_NODE blocked by acyclicity', () => {
    it('leaves node count unchanged', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 44, enforceAcyclic: true });
      const before = net.nodes.length;
      // Act
      mutateImpl.call(net, mutation.ADD_LSTM_NODE);
      const after = net.nodes.length;
      // Assert
      expect(after).toBe(before);
    });
  });

  describe('Scenario: ADD_GRU_NODE blocked by acyclicity', () => {
    it('leaves node count unchanged', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 45, enforceAcyclic: true });
      const before = net.nodes.length;
      // Act
      mutateImpl.call(net, mutation.ADD_GRU_NODE);
      const after = net.nodes.length;
      // Assert
      expect(after).toBe(before);
    });
  });

  describe('Scenario: BATCH_NORM with no hidden nodes', () => {
    it('does not tag any node', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 46 });
      // Act
      mutateImpl.call(net, mutation.BATCH_NORM);
      const tagged = net.nodes.some((n: any) => n._batchNorm);
      // Assert
      expect(tagged).toBe(false);
    });
  });
});
