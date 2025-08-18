import Network from '../../src/architecture/network';
import { exportToONNX, importFromONNX } from '../../src/architecture/onnx';

describe('ONNX Flatten after Pooling (Phase 4 extension)', () => {
  describe('when flattenAfterPooling enabled', () => {
    it('records flatten_layers metadata', () => {
      // Arrange
      const net = Network.createMLP(9, [4], 2);
      const onnx = exportToONNX(net, {
        includeMetadata: true,
        pool2dMappings: [
          {
            afterLayerIndex: 1,
            type: 'MaxPool',
            kernelHeight: 2,
            kernelWidth: 2,
            strideHeight: 1,
            strideWidth: 1,
          },
        ],
        flattenAfterPooling: true,
      });
      // Act
      const hasFlatten = (onnx.metadata_props || []).some(
        (m) => m.key === 'flatten_layers'
      );
      // Assert
      expect(hasFlatten).toBe(true);
    });
  });
});
