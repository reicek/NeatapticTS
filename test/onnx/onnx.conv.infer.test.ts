import Network from '../../src/architecture/network';
import { exportToONNX, importFromONNX } from '../../src/architecture/onnx';

/**
 * Tests heuristic conv inference metadata + pooling metadata attachment on import.
 */

describe('ONNX Conv Inference & Pool Import Attachment', () => {
  describe('Heuristic conv inference (single-channel square, 3x3->2x2 spatial)', () => {
    it('adds conv2d_inferred_layers metadata', () => {
      // Arrange: prev=5x5 (25) -> kernel 3 -> out 3x3 (9)
      const net = Network.createMLP(25, [9], 2);
      const onnx = exportToONNX(net, { includeMetadata: true });
      const inferred = (onnx.metadata_props || []).find(
        (m: any) => m.key === 'conv2d_inferred_layers'
      );
      expect(!!inferred).toBe(true);
    });
  });
  describe('Pooling import attachment metadata', () => {
    it('attaches _onnxPooling object to network', () => {
      const net = Network.createMLP(6, [4], 1);
      const onnx = exportToONNX(net, {
        includeMetadata: true,
        pool2dMappings: [
          {
            afterLayerIndex: 1,
            type: 'MaxPool',
            kernelHeight: 2,
            kernelWidth: 2,
            strideHeight: 2,
            strideWidth: 2,
          },
        ],
      });
      const imported = importFromONNX(onnx) as any;
      expect(!!imported._onnxPooling).toBe(true);
    });
  });
});
