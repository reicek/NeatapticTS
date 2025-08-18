import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import * as methods from '../../src/methods/methods';
import { importFromONNX } from '../../src/architecture/onnx';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('ONNX Import', () => {
  describe('1-1 input-output network', () => {
    let net: Network; // source network
    let onnx: any; // exported ONNX representation
    let imported: Network; // imported network
    beforeEach(() => {
      // Arrange
      net = new Network(1, 1);
      net.nodes[1].squash = methods.Activation.tanh;
      onnx = net.toONNX();
      // Act
      imported = importFromONNX(onnx);
    });
    it('imports with 2 nodes', () => {
      // Assert
      const nodeCount = imported.nodes.length;
      expect(nodeCount).toBe(2);
    });
    it('preserves output activation', () => {
      const preserved = imported.nodes[1].squash === methods.Activation.tanh;
      expect(preserved).toBe(true);
    });
    it('preserves output node bias', () => {
      const sameBias =
        Math.abs(imported.nodes[1].bias - net.nodes[1].bias) < 1e-9;
      expect(sameBias).toBe(true);
    });
  });

  describe('2-2-1 MLP', () => {
    let net: Network; // original net
    let imported: Network; // imported net
    beforeEach(() => {
      // Arrange
      net = Network.createMLP(2, [2], 1);
      net.nodes[2].bias = 0.5;
      net.nodes[3].bias = -0.5;
      net.nodes[4].bias = 1.0;
      net.nodes[2].connections.in[0].weight = 0.1;
      net.nodes[2].connections.in[1].weight = 0.2;
      net.nodes[3].connections.in[0].weight = 0.3;
      net.nodes[3].connections.in[1].weight = 0.4;
      net.nodes[4].connections.in[0].weight = 0.5;
      net.nodes[4].connections.in[1].weight = 0.6;
      net.nodes[2].squash = methods.Activation.relu;
      net.nodes[3].squash = methods.Activation.relu;
      net.nodes[4].squash = methods.Activation.sigmoid;
      const onnx = net.toONNX();
      // Act
      imported = importFromONNX(onnx);
    });
    it('imports with 5 nodes', () => {
      const nodeCount = imported.nodes.length;
      expect(nodeCount).toBe(5);
    });
    it('preserves bias of hidden node 1', () => {
      const same = Math.abs(imported.nodes[2].bias - 0.5) < 1e-9;
      expect(same).toBe(true);
    });
    it('preserves bias of hidden node 2', () => {
      const same = Math.abs(imported.nodes[3].bias + 0.5) < 1e-9;
      expect(same).toBe(true);
    });
    it('preserves bias of output node', () => {
      const same = Math.abs(imported.nodes[4].bias - 1.0) < 1e-9;
      expect(same).toBe(true);
    });
    it('preserves weight hidden1<-input0', () => {
      const w = imported.nodes[2].connections.in[0].weight;
      expect(Math.abs(w - 0.1) < 1e-9).toBe(true);
    });
    it('preserves weight hidden1<-input1', () => {
      const w = imported.nodes[2].connections.in[1].weight;
      expect(Math.abs(w - 0.2) < 1e-9).toBe(true);
    });
    it('preserves weight hidden2<-input0', () => {
      const w = imported.nodes[3].connections.in[0].weight;
      expect(Math.abs(w - 0.3) < 1e-9).toBe(true);
    });
    it('preserves weight hidden2<-input1', () => {
      const w = imported.nodes[3].connections.in[1].weight;
      expect(Math.abs(w - 0.4) < 1e-9).toBe(true);
    });
    it('preserves weight output<-hidden1', () => {
      const w = imported.nodes[4].connections.in[0].weight;
      expect(Math.abs(w - 0.5) < 1e-9).toBe(true);
    });
    it('preserves weight output<-hidden2', () => {
      const w = imported.nodes[4].connections.in[1].weight;
      expect(Math.abs(w - 0.6) < 1e-9).toBe(true);
    });
    it('preserves activation of hidden node 1', () => {
      const same = imported.nodes[2].squash === methods.Activation.relu;
      expect(same).toBe(true);
    });
    it('preserves activation of hidden node 2', () => {
      const same = imported.nodes[3].squash === methods.Activation.relu;
      expect(same).toBe(true);
    });
    it('preserves activation of output node', () => {
      const same = imported.nodes[4].squash === methods.Activation.sigmoid;
      expect(same).toBe(true);
    });
  });
  // Negative/error scenarios
  describe('Error and edge scenarios', () => {
    it('throws if ONNX input is null', () => {
      const throws = () => importFromONNX(null as any);
      expect(throws).toThrow();
    });
    it('throws if ONNX input is undefined', () => {
      const throws = () => importFromONNX(undefined as any);
      expect(throws).toThrow();
    });
  });
});
