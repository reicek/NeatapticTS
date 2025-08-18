import Network from '../../src/architecture/network';
import { exportToONNX } from '../../src/architecture/onnx';

/**
 * Phase 4 tests: pooling emission + conv weight sharing validation metadata.
 * Each test keeps a single expectation as per project testing guidelines.
 */

describe('ONNX Conv2D + Pooling Validation (Phase 4)', () => {
  describe('Conv sharing validation success', () => {
    // Arrange shared deterministic conv-like dense layer: we manually enforce weight sharing by copying weights.
    const inC = 1;
    const inH = 3;
    const inW = 3; // 3x3x1 input
    const kH = 2;
    const kW = 2;
    const stride = 1;
    const outC = 1;
    const outH = inH - kH + 1;
    const outW = inW - kW + 1; // 2x2 spatial
    const inputSize = inC * inH * inW; // 9
    const hiddenSize = outC * outH * outW; // 4
    const net = Network.createMLP(inputSize, [hiddenSize], 2);
    // Define a canonical kernel pattern (relative weights) size kH*kW.
    const kernelPattern = [0.11, -0.07, 0.05, 0.02]; // order: (0,0),(0,1),(1,0),(1,1)
    // Helper to set weights of a neuron corresponding to spatial position (oh,ow).
    const hiddenNeurons = net.nodes.filter((n: any) => n.type === 'hidden');
    hiddenNeurons.forEach((neuron: any, idx: number) => {
      const oh = Math.floor(idx / outW); // since outC=1
      const ow = idx % outW;
      const ihBase = oh * stride;
      const iwBase = ow * stride; // no padding
      // Build map from input node to desired kernel weight by relative coordinate
      neuron.connections.in.forEach((conn: any) => {
        conn.weight = 0;
      });
      for (let kh = 0; kh < kH; kh++) {
        for (let kw = 0; kw < kW; kw++) {
          const ih = ihBase + kh;
          const iw = iwBase + kw;
          const inputIndex = ih * inW + iw; // single channel
          const relativeIdx = kh * kW + kw;
          const sourceNode = net.nodes.filter((n: any) => n.type === 'input')[
            inputIndex
          ];
          const c = neuron.connections.in.find(
            (cc: any) => cc.from === sourceNode
          );
          if (c) c.weight = kernelPattern[relativeIdx];
        }
      }
      neuron.bias = 0.123; // constant bias across spatial positions
    });
    const mapping = [
      {
        layerIndex: 1,
        inHeight: inH,
        inWidth: inW,
        inChannels: inC,
        kernelHeight: kH,
        kernelWidth: kW,
        strideHeight: stride,
        strideWidth: stride,
        outHeight: outH,
        outWidth: outW,
        outChannels: outC,
      },
    ];
    it('records conv2d_sharing_verified metadata', () => {
      // Act
      const onnx = exportToONNX(net, {
        includeMetadata: true,
        conv2dMappings: mapping,
        validateConvSharing: true,
      });
      const verified = (onnx.metadata_props || []).find(
        (m: any) => m.key === 'conv2d_sharing_verified'
      );
      // Assert
      expect(!!verified).toBe(true);
    });
  });

  describe('Conv sharing validation mismatch', () => {
    const inC = 1;
    const inH = 3;
    const inW = 3;
    const kH = 2;
    const kW = 2;
    const stride = 1;
    const outC = 1;
    const outH = inH - kH + 1;
    const outW = inW - kW + 1; // 2x2
    const inputSize = inC * inH * inW;
    const hiddenSize = outC * outH * outW;
    const net = Network.createMLP(inputSize, [hiddenSize], 1);
    // Deliberately vary one spatial neuron's weight to break sharing
    const hidden = net.nodes.filter((n: any) => n.type === 'hidden');
    hidden.forEach((n: any, neuronIdx: number) => {
      n.connections.in.forEach((c: any, idx: number) => {
        c.weight = (idx + 1) * 0.02 + neuronIdx * 0.001;
      });
      n.bias = neuronIdx * 0.01;
    });
    const mapping = [
      {
        layerIndex: 1,
        inHeight: inH,
        inWidth: inW,
        inChannels: inC,
        kernelHeight: kH,
        kernelWidth: kW,
        strideHeight: stride,
        strideWidth: stride,
        outHeight: outH,
        outWidth: outW,
        outChannels: outC,
      },
    ];
    it('records conv2d_sharing_mismatch metadata', () => {
      const onnx = exportToONNX(net, {
        includeMetadata: true,
        conv2dMappings: mapping,
        validateConvSharing: true,
      });
      const mismatch = (onnx.metadata_props || []).find(
        (m: any) => m.key === 'conv2d_sharing_mismatch'
      );
      expect(!!mismatch).toBe(true);
    });
  });

  describe('Pooling emission metadata', () => {
    const net = Network.createMLP(6, [4], 2); // simple layer
    // Uniform weights
    net.connections.forEach((c: any, idx: number) => {
      c.weight = (idx + 1) * 0.01;
    });
    net.nodes.forEach((n: any, idx: number) => {
      if (n.type !== 'input') n.bias = idx * 0.001;
    });
    it('emits pool2d_layers metadata when pool mapping provided', () => {
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
      const meta = (onnx.metadata_props || []).find(
        (m: any) => m.key === 'pool2d_layers'
      );
      expect(!!meta).toBe(true);
    });
  });
});
