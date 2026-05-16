import * as methods from '../../../../methods/methods';
import Network from '../../network';
import type { OnnxModel } from '../schema/network.onnx.schema.types';
import { assignActivationFunctions } from './network.onnx.import-activations.utils';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function buildOnnx(graphNodes: OnnxModel['graph']['node'] = []): OnnxModel {
  return {
    graph: { inputs: [], outputs: [], initializer: [], node: graphNodes },
  };
}

describe('network onnx import-activations utils chapter', () => {
  describe('assignActivationFunctions()', () => {
    describe('given a graph node with an unsupported op_type like Gemm', () => {
      it('skips the node without throwing (line 192 — return null for unsupported op)', () => {
        // Arrange: op_type 'Gemm' is not in SUPPORTED_ACTIVATION_OPERATIONS → null
        const network = new Network(1, 1);
        const onnx = buildOnnx([
          { op_type: 'Gemm', name: 'act_l1', input: [], output: [] },
        ]);

        // Act + Assert
        expect(() =>
          assignActivationFunctions(network, onnx, []),
        ).not.toThrow();
      });
    });

    describe('given a graph node with a null name', () => {
      it('skips the node without throwing (line 143 branch 1 — null ?? EMPTY_NODE_NAME)', () => {
        // Arrange: null name hits the ?? fallback → empty string → no regex match → node skipped
        const network = new Network(1, 1);
        const onnx = buildOnnx([
          {
            op_type: 'Tanh',
            name: null as unknown as string,
            input: [],
            output: [],
          },
        ]);

        // Act + Assert: must not throw even though node name is null
        expect(() =>
          assignActivationFunctions(network, onnx, []),
        ).not.toThrow();
      });
    });

    describe('given a per-neuron activation node name containing a neuron suffix', () => {
      it('assigns the named activation to the matching hidden node', () => {
        // Arrange: act_l1_n0 has match[2]='0' (line 174 branch 0 — TRUE arm)
        const network = Network.createMLP(1, [1], 1);
        const hiddenNode = network.nodes.find((n) => n.type === 'hidden');
        const onnx = buildOnnx([
          { op_type: 'Tanh', name: 'act_l1_n0', input: [], output: [] },
        ]);

        // Act
        assignActivationFunctions(network, onnx, [1]);

        // Assert
        expect(hiddenNode?.squash).toBe(methods.Activation.tanh);
      });
    });

    describe('given a two-hidden-layer network', () => {
      it('uses the cumulative offset for the second hidden layer (line 247 branch 1)', () => {
        // Arrange: second traversal context has previousContext !== undefined
        const network = Network.createMLP(1, [1, 1], 1);
        const hiddenNodes = network.nodes.filter((n) => n.type === 'hidden');
        const onnx = buildOnnx([
          { op_type: 'Relu', name: 'act_l1', input: [], output: [] },
          { op_type: 'Tanh', name: 'act_l2', input: [], output: [] },
        ]);

        // Act
        assignActivationFunctions(network, onnx, [1, 1]);

        // Assert: second hidden node gets tanh from layer 2
        expect(hiddenNodes[1].squash).toBe(methods.Activation.tanh);
      });
    });

    describe('given hiddenLayerSizes that exceed the actual hidden node count', () => {
      it('skips assignment for out-of-range indices without throwing (line 318 branch 0)', () => {
        // Arrange: network has no hidden nodes; hiddenLayerSizes claims one exists
        const network = new Network(1, 1);
        const onnx = buildOnnx([
          { op_type: 'Relu', name: 'act_l1', input: [], output: [] },
        ]);

        // Act + Assert: must not throw
        expect(() =>
          assignActivationFunctions(network, onnx, [1]),
        ).not.toThrow();
      });
    });

    describe('given a hidden layer with no activation nodes present in ONNX', () => {
      it('assigns the identity activation to the hidden node via default fallback (lines 351+422)', () => {
        // Arrange: empty ONNX → operationsByLayer = {} → [] → DEFAULT_ACTIVATION_OPERATION
        const network = Network.createMLP(1, [1], 1);
        const hiddenNode = network.nodes.find((n) => n.type === 'hidden');
        const onnx = buildOnnx([]);

        // Act
        assignActivationFunctions(network, onnx, [1]);

        // Assert
        expect(hiddenNode?.squash).toBe(methods.Activation.identity);
      });
    });

    describe('given an output-only network with no activation nodes in ONNX', () => {
      it('assigns the identity activation to the output node via default fallback (line 392)', () => {
        // Arrange: empty ONNX → output layer operations = [] → DEFAULT_ACTIVATION_OPERATION
        const network = new Network(1, 1);
        const onnx = buildOnnx([]);

        // Act
        assignActivationFunctions(network, onnx, []);

        // Assert
        expect(network.nodes[1].squash).toBe(methods.Activation.identity);
      });
    });
    describe('given a Softplus activation node', () => {
      it('assigns the softplus activation to the matching hidden node', () => {
        // Arrange
        const network = Network.createMLP(1, [1], 1);
        const hiddenNode = network.nodes.find((nodeEntry) => nodeEntry.type === 'hidden');
        const onnx = buildOnnx([
          { op_type: 'Softplus', name: 'act_l1', input: [], output: [] },
        ]);

        // Act
        assignActivationFunctions(network, onnx, [1]);

        // Assert
        expect(hiddenNode?.squash).toBe(methods.Activation.softplus);
      });
    });

    describe('given a Mish activation node', () => {
      it('assigns the mish activation to the matching hidden node', () => {
        // Arrange
        const network = Network.createMLP(1, [1], 1);
        const hiddenNode = network.nodes.find((nodeEntry) => nodeEntry.type === 'hidden');
        const onnx = buildOnnx([
          { op_type: 'Mish', name: 'act_l1', input: [], output: [] },
        ]);

        // Act
        assignActivationFunctions(network, onnx, [1]);

        // Assert
        expect(hiddenNode?.squash).toBe(methods.Activation.mish);
      });
    });

    describe('given a Gelu activation node', () => {
      it('assigns the gelu activation to the matching hidden node', () => {
        // Arrange
        const network = Network.createMLP(1, [1], 1);
        const hiddenNode = network.nodes.find((nodeEntry) => nodeEntry.type === 'hidden');
        const onnx = buildOnnx([
          { op_type: 'Gelu', name: 'act_l1', input: [], output: [] },
        ]);

        // Act
        assignActivationFunctions(network, onnx, [1]);

        // Assert
        expect(hiddenNode?.squash).toBe(methods.Activation.gelu);
      });
    });
  });
});
