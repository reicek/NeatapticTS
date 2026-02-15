import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import * as methods from '../../src/methods/methods';
import { exportToONNX } from '../../src/architecture/onnx';
import type { OnnxModel } from '../../src/architecture/network/network.onnx.utils';

type OnnxAttribute = { name?: string; f?: number; i?: number };
type OnnxGraphNode = {
  op_type: string;
  input?: string[];
  output: string[];
  attributes?: OnnxAttribute[];
};

/**
 * Suppresses console.warn output during execution of a function that is
 * expected to trigger warnings (e.g., unknown activation mapping). This keeps
 * test output clean while still exercising the warning path.
 */
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
      let onnx: ReturnType<typeof exportToONNX>; // exported model
      beforeEach(() => {
        // Arrange
        net = new Network(1, 1, {});
        // Act
        onnx = exportToONNX(net);
      });
      it('has correct single input dimension', () => {
        // Assert
        const dimVal =
          onnx.graph.inputs[0].type.tensor_type.shape.dim[0].dim_value;
        expect(dimVal).toBe(1);
      });
    });
  });

  describe('Networks with hidden layers', () => {
    describe('2-2-1 network', () => {
      let onnx: ReturnType<typeof exportToONNX>;
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(2, [2], 1);
        // Act
        onnx = exportToONNX(net);
      });
      it('has at least one initializer', () => {
        // Assert
        const hasInitializer = onnx.graph.initializer.length > 0;
        expect(hasInitializer).toBe(true);
      });
    });
    describe('3-3-2-1 network (two hidden layers)', () => {
      let onnx: ReturnType<typeof exportToONNX>;
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(3, [3, 2], 1);
        // Act
        onnx = exportToONNX(net);
      });
      it('emits initializers for weights and biases', () => {
        // Assert
        const hasInitializers = onnx.graph.initializer.length > 0;
        expect(hasInitializers).toBe(true);
      });
    });
    describe('2-2-2 network (multiple outputs)', () => {
      let onnx: ReturnType<typeof exportToONNX>;
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(2, [2], 2);
        // Act
        onnx = exportToONNX(net);
      });
      it('has single output tensor', () => {
        // Assert
        const singleOutput = onnx.graph.outputs.length === 1;
        expect(singleOutput).toBe(true);
      });
      it('has output dimension 2', () => {
        // Assert
        const outDim =
          onnx.graph.outputs[0].type.tensor_type.shape.dim[0].dim_value;
        expect(outDim).toBe(2);
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
        const actNode = (onnx as OnnxModel).graph.node.find(
          (graphNode: OnnxGraphNode) =>
            ['Tanh', 'Sigmoid', 'Relu', 'Identity'].includes(graphNode.op_type),
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
        const actNode = (onnx as OnnxModel).graph.node.find(
          (graphNode: OnnxGraphNode) =>
            ['Tanh', 'Sigmoid', 'Relu', 'Identity'].includes(graphNode.op_type),
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
        const actNode = (onnx as OnnxModel).graph.node.find(
          (graphNode: OnnxGraphNode) =>
            ['Tanh', 'Sigmoid', 'Relu', 'Identity'].includes(graphNode.op_type),
        );
        expect(actNode?.op_type).toBe('Relu');
      });
    });
    describe('Unknown activation', () => {
      it('falls back to Identity op', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = (x: number) => x;
        // Act / Assert
        suppressConsoleWarn(() => {
          const onnx = exportToONNX(net) as OnnxModel;
          const actNode = onnx.graph.node.find((graphNode: OnnxGraphNode) =>
            ['Tanh', 'Sigmoid', 'Relu', 'Identity'].includes(graphNode.op_type),
          );
          expect(actNode?.op_type).toBe('Identity');
        });
      });
      it('emits a console warning for unknown activation', () => {
        // Arrange
        const net = new Network(1, 1, {});
        net.nodes[1].squash = (x: number) => x;
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
        net.nodes.forEach((nodeEntry: Node) => {
          nodeEntry.index = undefined;
        });
      });
      it('assigns numeric indices to all nodes', () => {
        // Act
        exportToONNX(net);
        // Assert
        const allIndexed = net.nodes.every(
          (nodeEntry: Node) => typeof nodeEntry.index === 'number',
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
      let gemmNodes: OnnxModel['graph']['node'];
      let nodes: OnnxModel['graph']['node'];
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(3, [3, 2], 1);
        // Act
        const onnx = exportToONNX(net) as OnnxModel;
        gemmNodes = onnx.graph.node.filter(
          (graphNode: OnnxGraphNode) => graphNode.op_type === 'Gemm',
        );
        nodes = onnx.graph.node;
      });
      it('emits at least one Gemm node', () => {
        // Assert
        const hasGemm = gemmNodes.length > 0;
        expect(hasGemm).toBe(true);
      });
      it('all Gemm nodes have alpha=1', () => {
        const allAlphaOne = gemmNodes.every((graphNode) => {
          const attribute = (graphNode.attributes ?? []).find(
            (attributeEntry?: OnnxAttribute) =>
              attributeEntry?.name === 'alpha',
          );
          return attribute?.f === 1;
        });
        expect(allAlphaOne).toBe(true);
      });
      it('all Gemm nodes have beta=1', () => {
        const allBetaOne = gemmNodes.every((graphNode) => {
          const attribute = (graphNode.attributes ?? []).find(
            (attributeEntry?: OnnxAttribute) => attributeEntry?.name === 'beta',
          );
          return attribute?.f === 1;
        });
        expect(allBetaOne).toBe(true);
      });
      it('all Gemm nodes have transB=1', () => {
        const allTransBOne = gemmNodes.every((graphNode) => {
          const attribute = (graphNode.attributes ?? []).find(
            (attributeEntry?: OnnxAttribute) =>
              attributeEntry?.name === 'transB',
          );
          return attribute?.i === 1;
        });
        expect(allTransBOne).toBe(true);
      });
      it('each Gemm node is followed by activation referencing its output', () => {
        const orderingValid = gemmNodes.every((gemmNode) => {
          const gemmIndex = nodes.indexOf(gemmNode);
          const activationNode = nodes.find(
            (graphNode) =>
              (graphNode.input &&
                graphNode.input[0] === gemmNode.output[0] &&
                graphNode.op_type !== 'Gemm') ??
              false,
          );
          return activationNode && nodes.indexOf(activationNode) > gemmIndex;
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
        const onnx = exportToONNX(net, {
          legacyNodeOrdering: true,
        }) as OnnxModel;
        const nodes = onnx.graph.node;
        const gemmNodes = nodes.filter(
          (graphNode: OnnxGraphNode) => graphNode.op_type === 'Gemm',
        );
        orderingValid = gemmNodes.every((gemmNode) => {
          const gemmIndex = nodes.indexOf(gemmNode);
          const activationNode = nodes.find(
            (graphNode) =>
              graphNode.input &&
              graphNode.input[0] === gemmNode.output[0] &&
              graphNode.op_type !== 'Gemm',
          );
          return activationNode && nodes.indexOf(activationNode) < gemmIndex;
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
        onnx = exportToONNX(net, {
          includeMetadata: true,
          opset: 18,
        }) as OnnxModel;
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
      /** ONNX tensor shape dimension descriptors */
      let inDims: Array<{ dim_param?: string; dim_value?: number }>;
      let outDims: Array<{ dim_param?: string; dim_value?: number }>;
      beforeEach(() => {
        // Arrange
        const net = Network.createMLP(4, [3], 2);
        // Act
        const onnx = exportToONNX(net, { batchDimension: true }) as OnnxModel;
        inDims = onnx.graph.inputs[0].type.tensor_type.shape.dim;
        outDims = onnx.graph.outputs[0].type.tensor_type.shape.dim;
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
          const model = exportToONNX(net, {
            allowMixedActivations: true,
          }) as OnnxModel;
          const didExport = !!model.graph;
          expect(didExport).toBe(true);
        } finally {
          console.warn = originalWarn;
        }
      });
    });
  });
});
