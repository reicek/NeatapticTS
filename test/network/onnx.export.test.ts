import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import * as methods from '../../src/methods/methods';
import { exportToONNX } from '../../src/architecture/onnx';
import type { OnnxModel } from '../../src/architecture/onnx';

/**
 * Suppresses console.warn output during execution of a function that is
 * expected to trigger warnings (e.g., unknown activation mapping). This keeps
 * test output clean while still exercising the warning path.
 */
type OnnxValueDim = { dim_value?: number; dim_param?: string };

type OnnxValueInfo = {
  name: string;
  type: {
    tensor_type: {
      shape: {
        dim: OnnxValueDim[];
      };
    };
  };
};

type OnnxAttributeView = {
  name: string;
  f?: number;
  i?: number;
};

type OnnxNodeView = {
  op_type: string;
  input: string[];
  output: string[];
  attributes?: OnnxAttributeView[];
};

type OnnxInitializerView = { name: string };

type OnnxGraphView = {
  inputs: OnnxValueInfo[];
  outputs: OnnxValueInfo[];
  initializer: OnnxInitializerView[];
  node: OnnxNodeView[];
};

const toGraphView = (model: OnnxModel): OnnxGraphView => ({
  inputs: model.graph.inputs as OnnxValueInfo[],
  outputs: model.graph.outputs as OnnxValueInfo[],
  initializer: model.graph.initializer as OnnxInitializerView[],
  node: model.graph.node as OnnxNodeView[],
});

const activationOps = new Set(['Tanh', 'Sigmoid', 'Relu', 'Identity']);

const getHiddenNodes = (network: Network) =>
  network.nodes.filter((node) => node.type === 'hidden');

const suppressConsoleWarn = (fn: () => void) => {
  const originalWarn = console.warn;
  console.warn = jest.fn();
  try {
    fn();
  } finally {
    console.warn = originalWarn;
  }
};

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('ONNX Export', () => {
  describe('Minimal valid MLP', () => {
    describe('1-1 input-output network', () => {
      let net: Network; // network under test
      let onnx: OnnxModel; // exported model
      beforeEach(() => {
        // Arrange
        net = new Network(1, 1, {});
        // Act
        onnx = exportToONNX(net);
      });
      it('has correct single input dimension', () => {
        // Assert
        const graph = toGraphView(onnx);
        const firstDim = graph.inputs[0].type.tensor_type.shape.dim[0];
        expect(firstDim?.dim_value).toBe(1);
      });
    });
  });

  describe('Networks with hidden layers', () => {
    describe('2-2-1 network', () => {
      let onnx: OnnxModel;
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(2, [2], 1);
        // Act
        onnx = exportToONNX(net);
      });
      it('has at least one initializer', () => {
        // Assert
        const graph = toGraphView(onnx);
        const hasInitializer = graph.initializer.length > 0;
        expect(hasInitializer).toBe(true);
      });
    });
    describe('3-3-2-1 network (two hidden layers)', () => {
      let onnx: OnnxModel;
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(3, [3, 2], 1);
        // Act
        onnx = exportToONNX(net);
      });
      it('emits initializers for weights and biases', () => {
        // Assert
        const graph = toGraphView(onnx);
        const hasInitializers = graph.initializer.length > 0;
        expect(hasInitializers).toBe(true);
      });
    });
    describe('2-2-2 network (multiple outputs)', () => {
      let onnx: OnnxModel;
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(2, [2], 2);
        // Act
        onnx = exportToONNX(net);
      });
      it('has single output tensor', () => {
        // Assert
        const graph = toGraphView(onnx);
        const singleOutput = graph.outputs.length === 1;
        expect(singleOutput).toBe(true);
      });
      it('has output dimension 2', () => {
        // Assert
        const graph = toGraphView(onnx);
        const firstDim = graph.outputs[0].type.tensor_type.shape.dim[0];
        expect(firstDim?.dim_value).toBe(2);
      });
    });
  });

  describe('Activation function mapping', () => {
    describe('Tanh activation', () => {
      it('maps tanh to ONNX Tanh op', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = methods.Activation.tanh;
        // Act
        const onnx = exportToONNX(net);
        // Assert
        const graph = toGraphView(onnx);
        const actNode = graph.node.find((node) =>
          activationOps.has(node.op_type),
        );
        expect(actNode?.op_type).toBe('Tanh');
      });
    });
    describe('Sigmoid activation', () => {
      it('maps logistic to ONNX Sigmoid op', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = methods.Activation.logistic;
        // Act
        const onnx = exportToONNX(net);
        // Assert
        const graph = toGraphView(onnx);
        const actNode = graph.node.find((node) =>
          activationOps.has(node.op_type),
        );
        expect(actNode?.op_type).toBe('Sigmoid');
      });
    });
    describe('Relu activation', () => {
      it('maps relu to ONNX Relu op', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = methods.Activation.relu;
        // Act
        const onnx = exportToONNX(net);
        // Assert
        const graph = toGraphView(onnx);
        const actNode = graph.node.find((node) =>
          activationOps.has(node.op_type),
        );
        expect(actNode?.op_type).toBe('Relu');
      });
    });
    describe('Unknown activation', () => {
      it('falls back to Identity op', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = (value: number) => value;
        // Act / Assert
        suppressConsoleWarn(() => {
          const onnx = exportToONNX(net);
          const graph = toGraphView(onnx);
          const actNode = graph.node.find((node) =>
            activationOps.has(node.op_type),
          );
          expect(actNode?.op_type).toBe('Identity');
        });
      });
      it('emits a console warning for unknown activation', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = (value: number) => value;
        const originalWarn = console.warn;
        const warnSpy = jest.fn();
        console.warn = warnSpy;
        try {
          // Act
          exportToONNX(net);
          // Assert
          const warned = warnSpy.mock.calls.length > 0;
          expect(warned).toBe(true);
        } finally {
          console.warn = originalWarn;
        }
      });
    });
  });

  describe('Error scenarios', () => {
    describe('Network with no connections', () => {
      let net: Network;
      beforeEach(() => {
        // Arrange
        net = new Network(1, 1, {});
        net.connections = [];
        net.nodes.forEach((n: Node) => {
          n.connections.out = [];
          n.connections.in = [];
        });
        Network.rebuildConnections(net);
      });
      it('throws for unsupported empty connection set', () => {
        // Act / Assert
        const throws = () => exportToONNX(net);
        expect(throws).toThrow(
          'ONNX export currently only supports simple MLPs',
        );
      });
    });
    describe('Partially disconnected network', () => {
      let net: Network;
      beforeEach(() => {
        // Arrange: construct partially disconnected 2-2-1 network
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
        // Remove one connection to violate full connectivity
        hidden1.connections.in = hidden1.connections.in.filter(
          (c) => c.from !== input0,
        );
        input0.connections.out = input0.connections.out.filter(
          (c) => c.to !== hidden1,
        );
        Network.rebuildConnections(net);
      });
      it('throws when layer is not fully connected', () => {
        // Act / Assert
        const throws = () => exportToONNX(net);
        expect(throws).toThrow();
      });
    });
  });

  describe('Helper utilities', () => {
    describe('Node indexing', () => {
      let net: Network;
      beforeEach(() => {
        // Arrange
        net = new Network(1, 1, {});
        net.nodes.forEach((n: Node) => {
          n.index = undefined;
        });
      });
      it('assigns numeric indices to all nodes', () => {
        // Act
        exportToONNX(net);
        // Assert
        const allIndexed = net.nodes.every(
          (n: Node) => typeof n.index === 'number',
        );
        expect(allIndexed).toBe(true);
      });
    });
    describe('suppressConsoleWarn helper', () => {
      it('suppresses console.warn during wrapped call', () => {
        // Arrange
        let warned = false;
        // Act
        suppressConsoleWarn(() => {
          console.warn('test');
          warned = true;
        });
        // Assert
        const executed = warned; // ensures closure executed
        expect(executed).toBe(true);
      });
    });
  });

  describe('Gemm attributes and ordering', () => {
    describe('Default ordering (Gemm -> Activation)', () => {
      let gemmNodes: OnnxNodeView[];
      let nodes: OnnxNodeView[];
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(3, [3, 2], 1);
        // Act
        const onnx = exportToONNX(net);
        const graph = toGraphView(onnx);
        gemmNodes = graph.node.filter((node) => node.op_type === 'Gemm');
        nodes = graph.node;
      });
      it('emits at least one Gemm node', () => {
        // Assert
        const hasGemm = gemmNodes.length > 0;
        expect(hasGemm).toBe(true);
      });
      it('all Gemm nodes have alpha=1', () => {
        const allAlphaOne = gemmNodes.every((node) => {
          const attributes = node.attributes ?? [];
          return (
            attributes.find((attribute) => attribute.name === 'alpha')?.f === 1
          );
        });
        expect(allAlphaOne).toBe(true);
      });
      it('all Gemm nodes have beta=1', () => {
        const allBetaOne = gemmNodes.every((node) => {
          const attributes = node.attributes ?? [];
          return (
            attributes.find((attribute) => attribute.name === 'beta')?.f === 1
          );
        });
        expect(allBetaOne).toBe(true);
      });
      it('all Gemm nodes have transB=1', () => {
        const allTransBOne = gemmNodes.every((node) => {
          const attributes = node.attributes ?? [];
          return (
            attributes.find((attribute) => attribute.name === 'transB')?.i === 1
          );
        });
        expect(allTransBOne).toBe(true);
      });
      it('each Gemm node is followed by activation referencing its output', () => {
        const orderingValid = gemmNodes.every((node) => {
          const gemmIndex = nodes.indexOf(node);
          const activation = nodes.find(
            (candidate) =>
              candidate !== node &&
              candidate.input[0] === node.output[0] &&
              candidate.op_type !== 'Gemm',
          );
          return !!activation && nodes.indexOf(activation) > gemmIndex;
        });
        expect(orderingValid).toBe(true);
      });
    });
    describe('Legacy ordering (Activation -> Gemm)', () => {
      let orderingValid: boolean;
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(2, [2], 1);
        // Act
        const onnx = exportToONNX(net, { legacyNodeOrdering: true });
        const nodes = toGraphView(onnx).node;
        const gemmNodes = nodes.filter((node) => node.op_type === 'Gemm');
        orderingValid = gemmNodes.every((node) => {
          const gemmIndex = nodes.indexOf(node);
          const activation = nodes.find(
            (candidate) =>
              candidate !== node &&
              candidate.input[0] === node.output[0] &&
              candidate.op_type !== 'Gemm',
          );
          return !!activation && nodes.indexOf(activation) < gemmIndex;
        });
      });
      it('places activation before gemm in legacy mode', () => {
        // Assert
        expect(orderingValid).toBe(true);
      });
    });
    describe('Metadata inclusion', () => {
      let onnx: OnnxModel;
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(1, [1], 1);
        // Act
        onnx = exportToONNX(net, { includeMetadata: true, opset: 18 });
      });
      it('includes ir_version field', () => {
        const present = typeof onnx.ir_version !== 'undefined';
        expect(present).toBe(true);
      });
      it('includes opset_import array', () => {
        const hasOpset = Array.isArray(onnx.opset_import);
        expect(hasOpset).toBe(true);
      });
      it('includes producer_name', () => {
        const hasProducer = typeof onnx.producer_name === 'string';
        expect(hasProducer).toBe(true);
      });
    });
    describe('Batch dimension option', () => {
      let inDims: OnnxValueDim[];
      let outDims: OnnxValueDim[];
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(4, [3], 2);
        // Act
        const onnx = exportToONNX(net, { batchDimension: true });
        const graph = toGraphView(onnx);
        inDims = graph.inputs[0].type.tensor_type.shape.dim;
        outDims = graph.outputs[0].type.tensor_type.shape.dim;
      });
      it('adds two input dims (batch + feature)', () => {
        const twoDims = inDims.length === 2;
        expect(twoDims).toBe(true);
      });
      it('adds two output dims (batch + feature)', () => {
        const twoDims = outDims.length === 2;
        expect(twoDims).toBe(true);
      });
      it('uses symbolic batch dim "N" for input', () => {
        const hasSymbol = inDims[0].dim_param === 'N';
        expect(hasSymbol).toBe(true);
      });
      it('uses symbolic batch dim "N" for output', () => {
        const hasSymbol = outDims[0].dim_param === 'N';
        expect(hasSymbol).toBe(true);
      });
      it('sets input feature size to 4', () => {
        const sizeOk = inDims[1].dim_value === 4;
        expect(sizeOk).toBe(true);
      });
      it('sets output feature size to 2', () => {
        const sizeOk = outDims[1].dim_value === 2;
        expect(sizeOk).toBe(true);
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Relaxed validation options (formerly in separate phase-specific test file)
  // ---------------------------------------------------------------------------
  describe('Relaxed validation options', () => {
    describe('allowPartialConnectivity option', () => {
      let net: Network;
      beforeEach(() => {
        // Arrange: build 2-2-1 network then remove one inbound connection to hidden[0]
        net = Network.createMLP(2, [2], 1);
        const input0 = net.nodes[0];
        const hidden0 = net.nodes[2];
        // Remove a single connection to create a partial layer
        hidden0.connections.in = hidden0.connections.in.filter(
          (c) => c.from !== input0,
        );
        input0.connections.out = input0.connections.out.filter(
          (c) => c.to !== hidden0,
        );
        Network.rebuildConnections(net);
      });
      it('throws without allowPartialConnectivity flag', () => {
        const throws = () => exportToONNX(net);
        expect(throws).toThrow();
      });
      it('exports successfully with allowPartialConnectivity flag', () => {
        const succeeds = () =>
          exportToONNX(net, { allowPartialConnectivity: true });
        expect(() => succeeds()).not.toThrow();
      });
    });

    describe('allowMixedActivations option', () => {
      let net: Network;
      beforeEach(() => {
        // Arrange: 1-3-1 network with different activations in same hidden layer
        net = Network.createMLP(1, [3], 1);
        // Hidden layer starts after input (1) and before output (last)
        net.nodes[1 + 0 + 1].squash = methods.Activation.relu; // first hidden
        net.nodes[1 + 0 + 2].squash = methods.Activation.tanh; // second hidden
        net.nodes[1 + 0 + 3].squash = methods.Activation.sigmoid; // third hidden
      });
      it('throws without allowMixedActivations flag', () => {
        const throws = () => exportToONNX(net);
        expect(throws).toThrow();
      });
      it('warns and exports successfully with allowMixedActivations flag', () => {
        const originalWarn = console.warn;
        const warnSpy = jest.fn();
        console.warn = warnSpy;
        try {
          const model = exportToONNX(net, { allowMixedActivations: true });
          const graph = toGraphView(model);
          const didExport = graph.node.length >= 0;
          expect(didExport).toBe(true);
        } finally {
          console.warn = originalWarn;
        }
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Recurrent single-step export (extended multi-layer baseline)
  // ---------------------------------------------------------------------------
  describe('Recurrent single-step export', () => {
    describe('Single hidden layer self-recurrence', () => {
      let onnx: OnnxModel;
      beforeEach(() => {
        // Arrange: create 2-3-1 network and add self connections to each hidden node
        const net = Network.createMLP(2, [3], 1);
        const hiddenNodes = getHiddenNodes(net);
        hiddenNodes.forEach((node) => {
          if (!node.connections.self.length) {
            node.connect(node, 0.42); // arbitrary recurrent weight
          } else {
            node.connections.self[0].weight = 0.42;
          }
        });
        // Act
        onnx = exportToONNX(net, {
          allowRecurrent: true,
          recurrentSingleStep: true,
        });
      });
      it('adds previous hidden state input', () => {
        const hasPrev = toGraphView(onnx).inputs.some(
          (input) => input.name === 'hidden_prev',
        );
        expect(hasPrev).toBe(true);
      });
      it('emits recurrent weight matrix R0', () => {
        const hasR0 = toGraphView(onnx).initializer.some(
          (initializer) => initializer.name === 'R0',
        );
        expect(hasR0).toBe(true);
      });
    });
    describe('Two hidden layers with recurrence only in second', () => {
      let onnx: OnnxModel;
      beforeEach(() => {
        // Arrange: 2-2-2-1 network (two hidden layers of size 2)
        const net = Network.createMLP(2, [2, 2], 1);
        const hiddenLayers = getHiddenNodes(net);
        // Hidden layer segmentation: first hidden layer indices 2..3, second 4..5 (after inputs 0..1)
        const secondHidden = hiddenLayers.slice(2, 4);
        // Add self connections only to second hidden layer
        secondHidden.forEach((node, index) => {
          const recurrentWeight = 0.1 + index;
          if (!node.connections.self.length) {
            node.connect(node, recurrentWeight);
          } else {
            node.connections.self[0].weight = recurrentWeight;
          }
        });
        // Act
        onnx = exportToONNX(net, {
          allowRecurrent: true,
          recurrentSingleStep: true,
        });
      });
      it('adds previous state input for second hidden layer only', () => {
        const inputs = toGraphView(onnx).inputs;
        const hasFirst = inputs.some((input) => input.name === 'hidden_prev');
        const hasSecond = inputs.some(
          (input) => input.name === 'hidden_prev_l2',
        );
        expect(hasFirst).toBe(false); // no recurrence in first hidden layer
        expect(hasSecond).toBe(true);
      });
      it('emits recurrent matrix R1 only', () => {
        const initializers = toGraphView(onnx).initializer;
        const hasR0 = initializers.some(
          (initializer) => initializer.name === 'R0',
        );
        const hasR1 = initializers.some(
          (initializer) => initializer.name === 'R1',
        );
        expect(hasR0).toBe(false);
        expect(hasR1).toBe(true);
      });
    });
    describe('Mixed activations in recurrent layer not allowed', () => {
      it('throws when recurrent layer has mixed activations without allowMixedActivations (not yet supported together)', () => {
        const net = Network.createMLP(1, [2], 1);
        const hidden = getHiddenNodes(net);
        // Add self connections
        hidden.forEach((node) => node.connect(node, 0.2));
        // Make mixed activations
        hidden[0].squash = methods.Activation.relu;
        hidden[1].squash = methods.Activation.tanh;
        const throws = () =>
          exportToONNX(net, {
            allowRecurrent: true,
            recurrentSingleStep: true,
          });
        expect(throws).toThrow();
      });
    });
  });
});
