import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import * as methods from '../../src/methods/methods';
import { exportToONNX } from '../../src/architecture/onnx';

// Helper to suppress console.warn for tests that expect warnings
const suppressConsoleWarn = (fn: () => void) => {
  const originalWarn = console.warn;
  console.warn = jest.fn();
  try { fn(); } finally { console.warn = originalWarn; }
};

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('ONNX Export', () => {
  describe('Minimal valid MLP', () => {
    it('exports a 1-1 input-output network', () => {
      // Arrange
      const net = new Network(1, 1, {});
      // Act
      const onnx = exportToONNX(net);
      // Assert
      expect(onnx.graph.inputs[0].type.tensor_type.shape.dim[0].dim_value).toBe(1);
    });
  });

  describe('Network with hidden layer', () => {
    it('exports a 2-2-1 network with correct layer structure', () => {
      // Arrange
      const net = Network.createMLP(2, [2], 1);
      // Act
      const onnx = exportToONNX(net);
      // Assert
      expect(onnx.graph.initializer.length).toBeGreaterThan(0);
    });
  });

  describe('Activation function mapping', () => {
    it('maps Tanh activation to ONNX Tanh', () => {
      // Arrange
      const net = new Network(1, 1, {});
      net.nodes[1].squash = methods.Activation.tanh;
      // Act
      const onnx = exportToONNX(net);
      // Assert
      expect(onnx.graph.node[0].op_type).toBe('Tanh');
    });
    it('maps Sigmoid activation to ONNX Sigmoid', () => {
      // Arrange
      const net = new Network(1, 1, {});
      net.nodes[1].squash = methods.Activation.logistic;
      // Act
      const onnx = exportToONNX(net);
      // Assert
      expect(onnx.graph.node[0].op_type).toBe('Sigmoid');
    });
    it('maps Relu activation to ONNX Relu', () => {
      // Arrange
      const net = new Network(1, 1, {});
      net.nodes[1].squash = methods.Activation.relu;
      // Act
      const onnx = exportToONNX(net);
      // Assert
      expect(onnx.graph.node[0].op_type).toBe('Relu');
    });
    it('maps unknown activation to ONNX Identity and warns', () => {
      // Arrange
      const net = new Network(1, 1, {});
      net.nodes[1].squash = function customSquash(x: number) { return x; };
      // Act & Assert
      suppressConsoleWarn(() => {
        const onnx = exportToONNX(net);
        expect(onnx.graph.node[0].op_type).toBe('Identity');
      });
    });
  });

  describe('Error scenarios', () => {
    it('throws if network has no connections', () => {
      // Arrange
      const net = new Network(1, 1, {});
      net.connections = [];
      // Also clear per-node connection lists to ensure the network is truly connectionless
      net.nodes.forEach(n => {
        n.connections.out = [];
        n.connections.in = [];
      });
      Network.rebuildConnections(net);
      // Act & Assert
      expect(() => exportToONNX(net)).toThrow('ONNX export currently only supports simple MLPs');
    });
    it('throws if network is not fully connected (invalid MLP)', () => {
      // Arrange
      // Build a valid 2-2-1 MLP first
      const net = new Network(2, 1, {});
      net.connections = [];
      const hidden1 = new Node('hidden');
      const hidden2 = new Node('hidden');
      net.nodes.splice(2, 0, hidden1, hidden2);
      const input0 = net.nodes[0];
      const input1 = net.nodes[1];
      input0.connect(hidden1, 0.5);
      input0.connect(hidden2, 0.5);
      input1.connect(hidden1, 0.5);
      input1.connect(hidden2, 0.5);
      hidden1.connect(net.nodes[4], 1.0);
      hidden2.connect(net.nodes[4], 1.0);
      net.connections = [
        ...input0.connections.out,
        ...input1.connections.out,
        ...hidden1.connections.out,
        ...hidden2.connections.out
      ];
      // Remove one input->hidden connection to break full connectivity
      // (e.g., remove input0->hidden1)
      net.connections = net.connections.filter(
        c => !(c.from === input0 && c.to === hidden1)
      );
      // Also update the per-node connection lists
      input0.connections.out = input0.connections.out.filter(c => c.to !== hidden1);
      hidden1.connections.in = hidden1.connections.in.filter(c => c.from !== input0);
      // Ensure network.connections is consistent with per-node connections
      Network.rebuildConnections(net);
      // Act & Assert
      expect(() => exportToONNX(net)).toThrow();
    });
  });

  describe('Helper/utility coverage', () => {
    it('assigns node indices before export', () => {
      // Arrange
      const net = new Network(1, 1, {});
      net.nodes.forEach(n => { n.index = undefined; });
      // Act
      exportToONNX(net);
      // Assert
      expect(net.nodes.every(n => typeof n.index === 'number')).toBe(true);
    });
  });
});
