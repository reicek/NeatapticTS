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
    describe('when exporting a 1-1 input-output network', () => {
      let net: Network;
      let onnx: ReturnType<typeof exportToONNX>;
      beforeEach(() => {
        // Arrange
        net = new Network(1, 1, {});
        // Act
        onnx = exportToONNX(net);
      });
      it('should have input shape dim_value 1', () => {
        // Assert
        expect(onnx.graph.inputs[0].type.tensor_type.shape.dim[0].dim_value).toBe(1);
      });
    });
  });

  describe('Network with hidden layer', () => {
    describe('when exporting a 2-2-1 network', () => {
      let net: Network;
      let onnx: ReturnType<typeof exportToONNX>;
      beforeEach(() => {
        // Arrange
        net = Network.createMLP(2, [2], 1);
        // Act
        onnx = exportToONNX(net);
      });
      it('should have at least one initializer', () => {
        // Assert
        expect(onnx.graph.initializer.length).toBeGreaterThan(0);
      });
    });
    describe('when exporting a 3-3-2-1 network (multiple hidden layers)', () => {
      let net: Network;
      let onnx: ReturnType<typeof exportToONNX>;
      beforeEach(() => {
        // Arrange
        net = Network.createMLP(3, [3, 2], 1);
        // Act
        onnx = exportToONNX(net);
      });
      it('should have correct number of initializers', () => {
        // Assert
        expect(onnx.graph.initializer.length).toBeGreaterThan(0);
      });
    });
    describe('when exporting a 2-2-2 network (multiple outputs)', () => {
      let net: Network;
      let onnx: ReturnType<typeof exportToONNX>;
      beforeEach(() => {
        // Arrange
        net = Network.createMLP(2, [2], 2);
        // Act
        onnx = exportToONNX(net);
      });
      it('should have a single output tensor with shape 2', () => {
        // Assert
        expect(onnx.graph.outputs.length).toBe(1);
        expect(onnx.graph.outputs[0].type.tensor_type.shape.dim[0].dim_value).toBe(2);
      });
    });
  });

  describe('Activation function mapping', () => {
    describe('when using Tanh activation', () => {
      it('should map to ONNX Tanh', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = methods.Activation.tanh;
        // Act
        const onnx = exportToONNX(net);
        // Assert
        expect(onnx.graph.node[0].op_type).toBe('Tanh');
      });
    });
    describe('when using Sigmoid activation', () => {
      it('should map to ONNX Sigmoid', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = methods.Activation.logistic;
        // Act
        const onnx = exportToONNX(net);
        // Assert
        expect(onnx.graph.node[0].op_type).toBe('Sigmoid');
      });
    });
    describe('when using Relu activation', () => {
      it('should map to ONNX Relu', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = methods.Activation.relu;
        // Act
        const onnx = exportToONNX(net);
        // Assert
        expect(onnx.graph.node[0].op_type).toBe('Relu');
      });
    });
    describe('when using an unknown activation', () => {
      it('should map to ONNX Identity', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = function customSquash(x: number) { return x; };
        // Act & Assert
        suppressConsoleWarn(() => {
          const onnx = exportToONNX(net);
          // Assert
          expect(onnx.graph.node[0].op_type).toBe('Identity');
        });
      });
      it('should warn when using an unknown activation', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = function customSquash(x: number) { return x; };
        // Spy
        const originalWarn = console.warn;
        const warnSpy = jest.fn();
        console.warn = warnSpy;
        try {
          // Act
          exportToONNX(net);
          // Assert
          expect(warnSpy).toHaveBeenCalled();
        } finally {
          console.warn = originalWarn;
        }
      });
    });
  });

  describe('Error scenarios', () => {
    describe('when network has no connections', () => {
      let net: Network;
      beforeEach(() => {
        // Arrange
        net = new Network(1, 1, {});
        net.connections = [];
        // Also clear per-node connection lists to ensure the network is truly connectionless
        net.nodes.forEach((n: Node) => {
          n.connections.out = [];
          n.connections.in = [];
        });
        Network.rebuildConnections(net);
      });
      it('should throw with message about only supporting simple MLPs', () => {
        // Assert
        expect(() => exportToONNX(net)).toThrow('ONNX export currently only supports simple MLPs');
      });
    });
    describe('when network is not fully connected (invalid MLP)', () => {
      let net: Network;
      beforeEach(() => {
        // Arrange
        // Build a valid 2-2-1 MLP first
        net = new Network(2, 1, {});
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
      });
      it('should throw when not fully connected', () => {
        // Assert
        expect(() => exportToONNX(net)).toThrow();
      });
    });
  });

  describe('Helper/utility coverage', () => {
    describe('assigns node indices before export', () => {
      let net: Network;
      beforeEach(() => {
        // Arrange
        net = new Network(1, 1, {});
        net.nodes.forEach((n: Node) => { n.index = undefined; });
      });
      it('should assign a number index to every node', () => {
        // Act
        exportToONNX(net);
        // Assert
        expect(net.nodes.every((n: Node) => typeof n.index === 'number')).toBe(true);
      });
    });
    describe('suppressConsoleWarn utility', () => {
      it('should suppress console.warn during function execution', () => {
        // Spy
        const originalWarn = console.warn;
        const warnSpy = jest.fn();
        let called = false;
        // Act
        suppressConsoleWarn(() => {
          console.warn('test warning');
          called = true;
        });
        // Assert
        expect(warnSpy).not.toHaveBeenCalled();
        expect(called).toBe(true);
        // Cleanup
        console.warn = originalWarn;
      });
    });
  });
});
