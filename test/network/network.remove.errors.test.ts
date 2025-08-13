import Network from '../../src/architecture/network';
import { removeNode } from '../../src/architecture/network/network.remove';

/**
 * Standalone removeNode error path coverage.
 * Focuses on structural guards & not-in-network validation.
 */
describe('Network.removeNode (standalone) error guards', () => {
  describe('Scenario: attempt removal of input node', () => {
    it('throws structural anchor error', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 31 });
      const inputNode = net.nodes[0];
      // Act / Assert
      expect(() => removeNode.call(net, inputNode as any)).toThrow(
        /Cannot remove input or output node/
      );
    });
  });

  describe('Scenario: attempt removal of output node', () => {
    it('throws structural anchor error', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 32 });
      const outputNode = net.nodes[net.nodes.length - 1];
      // Act / Assert
      expect(() => removeNode.call(net, outputNode as any)).toThrow(
        /Cannot remove input or output node/
      );
    });
  });

  describe('Scenario: attempt removal of node not in network', () => {
    it('throws not in network error', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 33 });
      const otherNet = new Network(1, 1, { seed: 34 });
      const foreignNode = otherNet.nodes[0];
      // Act / Assert
      expect(() => removeNode.call(net, foreignNode as any)).toThrow(
        /Node not in network/
      );
    });
  });
});
