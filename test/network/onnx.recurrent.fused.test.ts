import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import { exportToONNX, importFromONNX } from '../../src/architecture/onnx';
import type { OnnxModel } from '../../src/architecture/onnx';

const getHiddenNodes = (network: Network) =>
  network.nodes.filter((node) => node.type === 'hidden');

const hasMetadataKey = (model: OnnxModel, key: string) =>
  (model.metadata_props ?? []).some((metadata) => metadata.key === key);

/** Utility: create synthetic layer partition for LSTM heuristic */
const buildPartitionedLSTM = (
  input: number,
  unit: number,
  output: number,
): Network => {
  // Total hidden = 5 * unit (input, forget, cell, output gate, output block)
  const hiddenSize = unit * 5;
  const net = Network.createMLP(input, [hiddenSize], output);
  const hiddenNodes = getHiddenNodes(net);
  // Assign simple biases & ensure self connections only for cell group
  const cellStart = unit * 2;
  hiddenNodes.forEach((node: Node, index: number) => {
    node.bias = index * 0.01;
    if (index >= cellStart && index < cellStart + unit) {
      const recurrentWeight = 0.5 + (index - cellStart) * 0.01;
      if (!node.connections.self.length) {
        node.connect(node, recurrentWeight);
      } else {
        node.connections.self[0].weight = recurrentWeight;
      }
    }
  });
  return net;
};

/** Utility: create synthetic layer partition for GRU heuristic */
const buildPartitionedGRU = (
  input: number,
  unit: number,
  output: number,
): Network => {
  // Total hidden = 4 * unit (update, reset, candidate, output block)
  const hiddenSize = unit * 4;
  const net = Network.createMLP(input, [hiddenSize], output);
  const hiddenNodes = getHiddenNodes(net);
  // Self connections only for candidate group (third group)
  const candidateStart = unit * 2;
  hiddenNodes.forEach((node: Node, index: number) => {
    node.bias = index * 0.02;
    if (index >= candidateStart && index < candidateStart + unit) {
      const recurrentWeight = 0.7 + (index - candidateStart) * 0.02;
      if (!node.connections.self.length) {
        node.connect(node, recurrentWeight);
      } else {
        node.connections.self[0].weight = recurrentWeight;
      }
    }
  });
  return net;
};

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('ONNX Export (Experimental Fused Recurrent)', () => {
  describe('LSTM heuristic emission', () => {
    let onnx: OnnxModel;
    const unit = 2;
    beforeEach(() => {
      const net = buildPartitionedLSTM(3, unit, 1);
      onnx = exportToONNX(net, { allowRecurrent: true });
    });
    it('emits LSTM initializers', () => {
      const hasW = onnx.graph.initializer.some(
        (tensor) => tensor.name === 'LSTM_W0',
      );
      expect(hasW).toBe(true);
    });
    it('emits LSTM node', () => {
      const hasNode = onnx.graph.node.some((node) => node.op_type === 'LSTM');
      expect(hasNode).toBe(true);
    });
    it('records metadata for LSTM emission', () => {
      const meta = hasMetadataKey(onnx, 'lstm_emitted_layers');
      expect(meta).toBe(true);
    });
  });

  describe('GRU heuristic emission', () => {
    let onnx: OnnxModel;
    const unit = 3;
    beforeEach(() => {
      const net = buildPartitionedGRU(2, unit, 1);
      onnx = exportToONNX(net, { allowRecurrent: true });
    });
    it('emits GRU initializers', () => {
      const hasW = onnx.graph.initializer.some(
        (tensor) => tensor.name === 'GRU_W0',
      );
      expect(hasW).toBe(true);
    });
    it('emits GRU node', () => {
      const hasNode = onnx.graph.node.some((node) => node.op_type === 'GRU');
      expect(hasNode).toBe(true);
    });
    it('records metadata for GRU emission', () => {
      const meta = hasMetadataKey(onnx, 'gru_emitted_layers');
      expect(meta).toBe(true);
    });
  });

  describe('Fallback pattern metadata (near-miss)', () => {
    let onnx: OnnxModel;
    beforeEach(() => {
      // Construct hidden size triggering fallback (size 9 between GRU(8) and LSTM(10) thresholds)
      const net = Network.createMLP(2, [9], 1);
      // Add self connections to half just to simulate recurrence but not matching exact partition
      const hidden = getHiddenNodes(net);
      hidden.forEach((node: Node, index: number) => {
        if (index % 2 === 0) node.connect(node, 0.3);
      });
      onnx = exportToONNX(net, { allowRecurrent: true });
    });
    it('records rnn_pattern_fallback metadata', () => {
      const hasFallback = hasMetadataKey(onnx, 'rnn_pattern_fallback');
      expect(hasFallback).toBe(true);
    });
  });

  describe('Import reconstruction (LSTM)', () => {
    let roundTrip: Network;
    const unit = 2;
    beforeEach(() => {
      const net = buildPartitionedLSTM(2, unit, 1);
      const exported = exportToONNX(net, { allowRecurrent: true });
      roundTrip = importFromONNX(exported);
    });
    it('rebuilds a network with same input/output sizes', () => {
      const inputs = roundTrip.nodes.filter(
        (node) => node.type === 'input',
      ).length;
      expect(inputs).toBe(2);
    });
  });

  describe('Import reconstruction (GRU)', () => {
    let roundTrip: Network;
    const unit = 2;
    beforeEach(() => {
      const net = buildPartitionedGRU(2, unit, 1);
      const exported = exportToONNX(net, { allowRecurrent: true });
      roundTrip = importFromONNX(exported);
    });
    it('rebuilds a network with same output size', () => {
      const outputs = roundTrip.nodes.filter(
        (node) => node.type === 'output',
      ).length;
      expect(outputs).toBe(1);
    });
  });

  describe('Missing recurrent initializer safety', () => {
    let onnx: OnnxModel;
    beforeEach(() => {
      const net = buildPartitionedLSTM(2, 2, 1);
      onnx = exportToONNX(net, { allowRecurrent: true });
      // Remove one initializer to simulate corruption
      onnx.graph.initializer = onnx.graph.initializer.filter(
        (tensor) => !tensor.name.startsWith('LSTM_R'),
      );
    });
    it('imports without throwing when recurrent tensors missing', () => {
      const importModel = () => importFromONNX(onnx);
      expect(() => importModel()).not.toThrow();
    });
  });
});
