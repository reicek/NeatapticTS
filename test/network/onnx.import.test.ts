import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import * as methods from '../../src/methods/methods';
import { importFromONNX } from '../../src/architecture/onnx';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('ONNX Import', () => {
  describe('1-1 input-output network', () => {
    let net: Network;
    let onnx: any;
    let imported: Network;

    beforeEach(() => {
      // Arrange
      net = new Network(1, 1);
      net.nodes[1].squash = methods.Activation.tanh;
      onnx = net.toONNX();
      // Act
      imported = importFromONNX(onnx);
    });

    it('should have 2 nodes', () => {
      // Assert
      expect(imported.nodes.length).toBe(2);
    });

    it('should preserve the activation function', () => {
      // Assert
      expect(imported.nodes[1].squash).toBe(methods.Activation.tanh);
    });

    it('should preserve the output node bias', () => {
      // Assert
      expect(imported.nodes[1].bias).toBeCloseTo(net.nodes[1].bias);
    });
  });

  describe('2-2-1 MLP', () => {
    let net: Network;
    let onnx: any;
    let imported: Network;

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
      onnx = net.toONNX();
      // Act
      imported = importFromONNX(onnx);
    });

    it('should have 5 nodes', () => {
      // Assert
      expect(imported.nodes.length).toBe(5);
    });

    it('should preserve hidden node 1 bias', () => {
      // Assert
      expect(imported.nodes[2].bias).toBeCloseTo(0.5);
    });

    it('should preserve hidden node 2 bias', () => {
      // Assert
      expect(imported.nodes[3].bias).toBeCloseTo(-0.5);
    });

    it('should preserve output node bias', () => {
      // Assert
      expect(imported.nodes[4].bias).toBeCloseTo(1.0);
    });

    it('should preserve hidden node 1 input weights', () => {
      // Assert
      expect(imported.nodes[2].connections.in[0].weight).toBeCloseTo(0.1);
      expect(imported.nodes[2].connections.in[1].weight).toBeCloseTo(0.2);
    });

    it('should preserve hidden node 2 input weights', () => {
      // Assert
      expect(imported.nodes[3].connections.in[0].weight).toBeCloseTo(0.3);
      expect(imported.nodes[3].connections.in[1].weight).toBeCloseTo(0.4);
    });

    it('should preserve output node input weights', () => {
      // Assert
      expect(imported.nodes[4].connections.in[0].weight).toBeCloseTo(0.5);
      expect(imported.nodes[4].connections.in[1].weight).toBeCloseTo(0.6);
    });

    it('should preserve hidden node 1 activation', () => {
      // Assert
      expect(imported.nodes[2].squash).toBe(methods.Activation.relu);
    });

    it('should preserve hidden node 2 activation', () => {
      // Assert
      expect(imported.nodes[3].squash).toBe(methods.Activation.relu);
    });

    it('should preserve output node activation', () => {
      // Assert
      expect(imported.nodes[4].squash).toBe(methods.Activation.sigmoid);
    });
  });
  // Negative/error scenarios
  describe('Error and edge scenarios', () => {
    it('should throw if ONNX input is null', () => {
      // Arrange/Act/Assert
      expect(() => importFromONNX(null as any)).toThrow();
    });

    it('should throw if ONNX input is undefined', () => {
      // Arrange/Act/Assert
      expect(() => importFromONNX(undefined as any)).toThrow();
    });
  });
});
