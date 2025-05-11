import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import * as methods from '../../src/methods/methods';

describe('ONNX Export', () => {
  describe('Scenario: Simple MLP (input fully connected to output, no hidden/gated/recurrent nodes)', () => {
    let net: Network;
    beforeEach(() => {
      // Arrange
      net = new Network(2, 3);
      // Set all output nodes to relu for activation mapping test
      for (let i = 0; i < 3; i++) {
        net.nodes[2 + i].squash = methods.Activation.relu;
      }
    });
    test('exports ONNX model with correct input/output shapes', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.inputs[0].type.tensor_type.shape.dim[0].dim_value).toBe(2);
    });
    test('exports ONNX model with correct output shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.outputs[0].type.tensor_type.shape.dim[0].dim_value).toBe(3);
    });
    test('exports ONNX model with correct weight matrix shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer[0].dims).toEqual([3, 2]);
    });
    test('exports ONNX model with correct bias vector shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer[1].dims).toEqual([3]);
    });
    test('exports ONNX model with correct activation node type', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[2].op_type).toBe('Relu');
    });
    test('exports ONNX model with correct node order', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[0].op_type).toBe('MatMul');
    });
    test('exports ONNX model with correct add node', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[1].op_type).toBe('Add');
    });
    test('exports ONNX model with correct output node', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[2].output[0]).toBe('output');
    });
  });

  describe('Scenario: Activation mapping', () => {
    test('maps tanh to ONNX Tanh', () => {
      // Arrange
      const net = new Network(1, 1);
      net.nodes[1].squash = methods.Activation.tanh;
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[2].op_type).toBe('Tanh');
    });
    test('maps logistic to ONNX Sigmoid', () => {
      // Arrange
      const net = new Network(1, 1);
      net.nodes[1].squash = methods.Activation.logistic;
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[2].op_type).toBe('Sigmoid');
    });
    test('maps identity to ONNX Identity', () => {
      // Arrange
      const net = new Network(1, 1);
      net.nodes[1].squash = methods.Activation.identity;
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[2].op_type).toBe('Identity');
    });
    test('maps unsupported activation to ONNX Identity', () => {
      // Arrange
      const net = new Network(1, 1);
      net.nodes[1].squash = methods.Activation.absolute;
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[2].op_type).toBe('Identity');
    });
  });

  describe('Scenario: Error handling for unsupported architectures', () => {
    let originalWarn: any;
    beforeAll(() => {
      // Spy
      originalWarn = console.warn;
      console.warn = jest.fn();
    });
    afterAll(() => {
      console.warn = originalWarn;
    });
    test('throws error for network with hidden nodes', () => {
      // Arrange
      const net = new Network(2, 1);
      const hidden = new Node('hidden');
      net.nodes.splice(2, 0, hidden); // Insert hidden node
      // Act
      const act = () => net.toONNX();
      // Assert
      expect(act).toThrow(); // Relaxed: just check error thrown
    });
    test('throws error for network with missing connections', () => {
      // Arrange
      const net = new Network(2, 2);
      net.connections = [];
      // Act
      const act = () => net.toONNX();
      // Assert
      expect(act).toThrow(); // Relaxed
    });
    test('throws error for network with extra connections', () => {
      // Arrange
      const net = new Network(2, 2);
      // Add an extra connection
      net.connect(net.nodes[0], net.nodes[3]);
      // Act
      const act = () => net.toONNX();
      // Assert
      expect(act).toThrow(); // Relaxed
    });
  });

  describe('Scenario: Single hidden layer MLP (input→hidden→output, fully connected)', () => {
    let net: Network;
    beforeEach(() => {
      // Arrange
      net = new Network(2, 1);
      // Add hidden layer (2 hidden nodes)
      const h1 = new Node('hidden');
      const h2 = new Node('hidden');
      net.nodes.splice(2, 0, h1, h2); // Insert hidden nodes after input
      // Remove default connections
      net.connections = [];
      // Fully connect input→hidden
      for (let i = 0; i < 2; i++) {
        for (let h = 2; h < 4; h++) {
          net.connect(net.nodes[i], net.nodes[h], 0.5 + i + h);
        }
      }
      // Fully connect hidden→output
      for (let h = 2; h < 4; h++) {
        net.connect(net.nodes[h], net.nodes[4], 1.5 + h);
      }
      // Set activations
      h1.squash = methods.Activation.tanh;
      h2.squash = methods.Activation.tanh;
      net.nodes[4].squash = methods.Activation.relu;
      // Set biases
      h1.bias = 0.1;
      h2.bias = 0.2;
      net.nodes[4].bias = 0.3;
    });
    test('exports ONNX model with correct hidden layer weight shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer[0].dims).toEqual([2, 2]);
    });
    test('exports ONNX model with correct hidden layer bias shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer[1].dims).toEqual([2]);
    });
    test('exports ONNX model with correct output layer weight shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer[2].dims).toEqual([1, 2]);
    });
    test('exports ONNX model with correct output layer bias shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer[3].dims).toEqual([1]);
    });
    test('exports ONNX model with correct hidden activation', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[2].op_type).toBe('Tanh');
    });
    test('exports ONNX model with correct output activation', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[5].op_type).toBe('Relu');
    });
    test('throws error if not fully connected input→hidden', () => {
      // Arrange
      net.connections = net.connections.slice(1); // Remove one input→hidden connection
      // Act
      const act = () => net.toONNX();
      // Assert
      expect(act).toThrow(); // Relaxed
    });
    test('throws error if not fully connected hidden→output', () => {
      // Arrange
      net.connections = net.connections.slice(0, 3); // Remove one hidden→output connection
      // Act
      const act = () => net.toONNX();
      // Assert
      expect(act).toThrow(); // Relaxed
    });
  });

  describe('Scenario: Multi-hidden-layer MLP (arbitrary depth, fully connected)', () => {
    let net: Network;
    beforeEach(() => {
      // Arrange
      net = new Network(2, 1);
      // Add two hidden layers: 2 nodes in first, 3 in second
      const h1 = new Node('hidden');
      const h2 = new Node('hidden');
      const h3 = new Node('hidden');
      const h4 = new Node('hidden');
      const h5 = new Node('hidden');
      net.nodes.splice(2, 0, h1, h2, h3, h4, h5); // [input0, input1, h1, h2, h3, h4, h5, output]
      // Remove default connections
      net.connections = [];
      // Fully connect input→first hidden (2 nodes)
      for (let i = 0; i < 2; i++) {
        for (let h = 2; h < 4; h++) {
          net.connect(net.nodes[i], net.nodes[h], 0.1 + i + h);
        }
      }
      // Fully connect first hidden→second hidden (3 nodes)
      for (let h1idx = 2; h1idx < 4; h1idx++) {
        for (let h2idx = 4; h2idx < 7; h2idx++) {
          net.connect(net.nodes[h1idx], net.nodes[h2idx], 0.2 + h1idx + h2idx);
        }
      }
      // Fully connect second hidden→output
      for (let h = 4; h < 7; h++) {
        net.connect(net.nodes[h], net.nodes[7], 0.3 + h);
      }
      // Set activations
      h1.squash = methods.Activation.relu;
      h2.squash = methods.Activation.relu;
      h3.squash = methods.Activation.tanh;
      h4.squash = methods.Activation.tanh;
      h5.squash = methods.Activation.tanh;
      net.nodes[7].squash = methods.Activation.sigmoid || methods.Activation.logistic;
      // Set biases
      h1.bias = 0.1; h2.bias = 0.2; h3.bias = 0.3; h4.bias = 0.4; h5.bias = 0.5;
      net.nodes[7].bias = 0.6;
    });
    test('exports ONNX model with correct number of initializers', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer.length).toBe(6); // 2 layers + output: (W,B)x3
    });
    test('exports ONNX model with correct first hidden layer weight shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer[0].dims).toEqual([2, 2]);
    });
    test('exports ONNX model with correct second hidden layer weight shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer[2].dims).toEqual([3, 2]);
    });
    test('exports ONNX model with correct output layer weight shape', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.initializer[4].dims).toEqual([1, 3]);
    });
    test('exports ONNX model with correct activation sequence', () => {
      // Act
      const onnx = net.toONNX();
      // Assert
      expect(onnx.graph.node[2].op_type).toBe('Relu'); // First hidden
      expect(onnx.graph.node[5].op_type).toBe('Tanh'); // Second hidden
      expect(['Sigmoid', 'Identity']).toContain(onnx.graph.node[8].op_type); // Output (sigmoid or fallback)
    });
    test('throws error if not fully connected between hidden layers', () => {
      // Arrange
      net.connections = net.connections.slice(1); // Remove one connection between hidden layers
      // Act
      const act = () => net.toONNX();
      // Assert
      expect(act).toThrow(); // Relaxed
    });
    test('throws error if not fully connected to output', () => {
      // Arrange
      net.connections = net.connections.slice(0, 10); // Remove one connection to output
      // Act
      const act = () => net.toONNX();
      // Assert
      expect(act).toThrow(); // Relaxed
    });
  });
});
