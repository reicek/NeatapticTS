import Node from '../../../node';
import { buildOnnxModel } from './network.onnx.export-build.utils';

function createLayer(nodeType: 'input' | 'hidden' | 'output', nodeCount: number): Node[] {
  return Array.from({ length: nodeCount }, () => new Node(nodeType));
}

describe('network onnx export build utils', () => {
  describe('buildOnnxModel', () => {
    it('uses default options when the options argument is omitted', () => {
      // Arrange
      const layers = [createLayer('input', 2), createLayer('output', 1)];

      // Act
      const onnxModel = buildOnnxModel({} as never, layers);

      // Assert
      expect(onnxModel.graph.outputs[0]?.name).toBe('output');
    });

    it('collects hidden layer metadata when a hidden layer exists', () => {
      // Arrange
      const layers = [
        createLayer('input', 2),
        createLayer('hidden', 3),
        createLayer('output', 1),
      ];

      // Act
      const onnxModel = buildOnnxModel({} as never, layers, {
        includeMetadata: true,
      });

      // Assert
      expect(onnxModel.graph.outputs[0]?.name).toBe('output');
    });
  });
});
