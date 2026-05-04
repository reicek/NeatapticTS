import Node from '../../../../node';
import { emitPerNeuronLayer } from './network.onnx.export-dense.utils';
import type { OnnxModel } from '../../schema/network.onnx.schema.types';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function createOnnxModel(): OnnxModel {
  return {
    graph: {
      inputs: [],
      outputs: [],
      initializer: [],
      node: [],
    },
  };
}

describe('network onnx export dense utility chapter', () => {
  describe('emitPerNeuronLayer()', () => {
    describe('given batchDimension is true', () => {
      it('emits concat node with axis 1', () => {
        // Arrange
        const prevNode = new Node('input');
        const targetNode = new Node('hidden');
        prevNode.connect(targetNode, 0.3);
        const model = createOnnxModel();

        // Act
        emitPerNeuronLayer({
          model,
          layerIndex: 1,
          previousOutputName: 'Input_0',
          previousLayerNodes: [prevNode],
          currentLayerNodes: [targetNode],
          options: {},
          batchDimension: true,
        });

        const concatNode = model.graph.node.find(
          (node) => node.op_type === 'Concat',
        );

        // Assert
        expect(concatNode?.attributes?.[0]?.i).toBe(1);
      });
    });

    describe('given a previous layer node has no connection to the target neuron', () => {
      it('fills the missing inbound weight with zero', () => {
        // Arrange
        const prevNode1 = new Node('input');
        const prevNode2 = new Node('input');
        const targetNode = new Node('hidden');
        prevNode1.connect(targetNode, 0.5);
        // prevNode2 intentionally left unconnected

        const model = createOnnxModel();

        // Act
        emitPerNeuronLayer({
          model,
          layerIndex: 1,
          previousOutputName: 'Input_0',
          previousLayerNodes: [prevNode1, prevNode2],
          currentLayerNodes: [targetNode],
          options: {},
          batchDimension: false,
        });

        const weightInitializer = model.graph.initializer.find(
          (init) => init.name === 'W0_n0',
        );

        // Assert: second weight entry is 0 because prevNode2 has no connection to targetNode
        expect(weightInitializer?.float_data[1]).toBe(0);
      });
    });
  });
});
