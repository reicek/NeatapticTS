import Network from '../../src/architecture/network';
import { exportToONNX, importFromONNX } from '../../src/architecture/onnx';

/**
 * Phase 4 groundwork tests: explicit Conv2D mapping export scaffolding.
 * We verify structural emission (Conv node + weights/bias) and metadata presence.
 * Round-trip currently expands Conv back into dense form (weights applied) so forward pass should still be deterministic.
 */

describe('ONNX Conv2D Groundwork Export', () => {
  describe('Single hidden conv-mapped layer (manual mapping)', () => {
    // Common Arrange data for nested scenarios
    const inputChannels = 1;
    const inH = 3;
    const inW = 3;
    const kernelH = 2;
    const kernelW = 2;
    const strideH = 1;
    const strideW = 1;
    const pad = 0;
    const outChannels = 2;
    const outH = inH - kernelH + 1; // 2
    const outW = inW - kernelW + 1; // 2
    const inputSize = inputChannels * inH * inW; // 9
    const hiddenSize = outChannels * outH * outW; // 8
    const outputSize = 3;

    it('emits Conv node and metadata', () => {
      // Arrange
      const net = Network.createMLP(inputSize, [hiddenSize], outputSize);
      // Deterministic weight fill: weight = (idx * 0.01)
      net.connections.forEach((c: any, idx: number) => {
        c.weight = idx * 0.01;
      });
      net.nodes.forEach((n: any, idx: number) => {
        if (n.type !== 'input') n.bias = idx * 0.001;
      });
      const mapping = [
        {
          layerIndex: 1,
          inHeight: inH,
          inWidth: inW,
          inChannels: inputChannels,
          kernelHeight: kernelH,
          kernelWidth: kernelW,
          strideHeight: strideH,
          strideWidth: strideW,
          padTop: pad,
          padBottom: pad,
          padLeft: pad,
          padRight: pad,
          outHeight: outH,
          outWidth: outW,
          outChannels: outChannels,
        },
      ];
      // Act
      const onnx = exportToONNX(net, {
        includeMetadata: true,
        conv2dMappings: mapping,
      });
      // Assert (single expectation): ensure Conv node present & metadata recorded
      const hasConv = onnx.graph.node.some((n: any) => n.op_type === 'Conv');
      const metaConvLayers = (onnx.metadata_props || []).find(
        (m: any) => m.key === 'conv2d_layers'
      );
      expect(hasConv && !!metaConvLayers).toBe(true);
    });
  });

  describe('Conv mapping dimension mismatch fallback', () => {
    it('skips Conv emission when dimensions invalid', () => {
      // Arrange
      const net = Network.createMLP(10, [6], 2); // Incompatible sizes for declared mapping
      const badMapping = [
        {
          layerIndex: 1,
          inHeight: 2,
          inWidth: 2,
          inChannels: 2, // would imply prev width 8, but actual previous width 10
          kernelHeight: 2,
          kernelWidth: 2,
          strideHeight: 1,
          strideWidth: 1,
          outHeight: 1,
          outWidth: 1,
          outChannels: 3, // implies layer width 3, actual hidden width 6
        },
      ];
      // Act
      const onnx = exportToONNX(net, { conv2dMappings: badMapping });
      // Assert
      const hasConv = onnx.graph.node.some((n: any) => n.op_type === 'Conv');
      expect(hasConv).toBe(false);
    });
  });
});
