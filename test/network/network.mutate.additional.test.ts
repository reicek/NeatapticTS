import Network from '../../src/architecture/network';
import { mutateImpl } from '../../src/architecture/network/network.mutate';
import mutation from '../../src/methods/mutation';

/**
 * Additional mutation operator coverage for success paths and complex heuristics.
 */

describe('Network.mutateImpl additional operators', () => {
  describe('Scenario: ADD_SELF_CONN success', () => {
    it('adds a self connection when acyclicity disabled', () => {
      // Arrange
      const net = new Network(2, 2, { seed: 20, enforceAcyclic: false });
      const before = net.selfconns.length;
      // Act
      mutateImpl.call(net, mutation.ADD_SELF_CONN);
      // Assert
      expect(net.selfconns.length > before).toBe(true);
    });
  });

  describe('Scenario: SUB_SELF_CONN removes existing self loop', () => {
    it('decreases self connection count by one', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 21, enforceAcyclic: false });
      mutateImpl.call(net, mutation.ADD_SELF_CONN);
      const before = net.selfconns.length;
      // Act
      mutateImpl.call(net, mutation.SUB_SELF_CONN);
      const after = net.selfconns.length;
      // Assert
      expect(before - after).toBe(1);
    });
  });

  describe('Scenario: ADD_BACK_CONN success', () => {
    it('adds at least one backward connection', () => {
      // Arrange
      const net = new Network(2, 2, { seed: 22, enforceAcyclic: false });
      const before = net.connections.filter(
        (c) => net.nodes.indexOf(c.from) > net.nodes.indexOf(c.to)
      ).length;
      // Act
      mutateImpl.call(net, mutation.ADD_BACK_CONN);
      const after = net.connections.filter(
        (c) => net.nodes.indexOf(c.from) > net.nodes.indexOf(c.to)
      ).length;
      // Assert
      expect(after > before).toBe(true);
    });
  });

  describe('Scenario: SUB_BACK_CONN removes one backward connection', () => {
    it('reduces backward connection count', () => {
      // Arrange
      const net = new Network(1, 2, { seed: 23, enforceAcyclic: false });
      // Create two hidden nodes to satisfy redundancy heuristics.
      mutateImpl.call(net, mutation.ADD_NODE);
      mutateImpl.call(net, mutation.ADD_NODE);
      // Manually add extra forward/outgoing edges from first hidden to outputs to inflate out-degree.
      const hiddenNodes = net.nodes.filter((n) => n.type === 'hidden');
      if (hiddenNodes.length) {
        const h = hiddenNodes[0];
        const outputs = net.nodes.filter((n) => n.type === 'output');
        outputs.forEach((o) => {
          if (!h.isProjectingTo(o)) net.connect(h, o);
        });
        // Add two backward connections to the single input (increase its in-degree)
        outputs.slice(0, 2).forEach((o) => {
          if (!o.isProjectingTo(net.nodes[0])) net.connect(o, net.nodes[0]);
        });
        if (!h.isProjectingTo(net.nodes[0])) net.connect(h, net.nodes[0]);
      }
      const before = net.connections.filter(
        (c) => net.nodes.indexOf(c.from) > net.nodes.indexOf(c.to)
      ).length;
      // Act
      mutateImpl.call(net, mutation.SUB_BACK_CONN);
      const after = net.connections.filter(
        (c) => net.nodes.indexOf(c.from) > net.nodes.indexOf(c.to)
      ).length;
      // Assert
      expect(after < before).toBe(true);
    });
  });

  describe('Scenario: REINIT_WEIGHT rewrites weights of a node', () => {
    it('changes at least one connection weight', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 24 });
      mutateImpl.call(net, mutation.ADD_NODE); // ensure hidden node available
      const beforeWeights = net.connections
        .map((c) => c.weight.toFixed(6))
        .join(',');
      // Act
      mutateImpl.call(net, mutation.REINIT_WEIGHT);
      const afterWeights = net.connections
        .map((c) => c.weight.toFixed(6))
        .join(',');
      // Assert
      expect(afterWeights === beforeWeights).toBe(false);
    });
  });

  describe('Scenario: BATCH_NORM tags a hidden node', () => {
    it('sets _batchNorm flag on some hidden node', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 25 });
      mutateImpl.call(net, mutation.ADD_NODE);
      // Act
      mutateImpl.call(net, mutation.BATCH_NORM);
      const tagged = net.nodes.some(
        (n: any) => n.type === 'hidden' && n._batchNorm === true
      );
      // Assert
      expect(tagged).toBe(true);
    });
  });

  describe('Scenario: ADD_LSTM_NODE expands connection into block', () => {
    it('increases node count when acyclicity disabled', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 26, enforceAcyclic: false });
      const before = net.nodes.length;
      // Act
      mutateImpl.call(net, mutation.ADD_LSTM_NODE);
      const after = net.nodes.length;
      // Assert
      expect(after > before).toBe(true);
    });
  });

  describe('Scenario: ADD_GRU_NODE expands connection into block', () => {
    it('increases node count when acyclicity disabled', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 27, enforceAcyclic: false });
      const before = net.nodes.length;
      // Act
      mutateImpl.call(net, mutation.ADD_GRU_NODE);
      const after = net.nodes.length;
      // Assert
      expect(after > before).toBe(true);
    });
  });
});
