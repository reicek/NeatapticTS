import Network from '../../src/architecture/network';
import * as methods from '../../src/methods/methods';
import { exportToONNX, importFromONNX } from '../../src/architecture/onnx';

/** Utility: create synthetic layer partition for LSTM heuristic */
function buildPartitionedLSTM(input: number, unit: number, output: number) {
  // Total hidden = 5 * unit (input, forget, cell, output gate, output block)
  const hiddenSize = unit * 5;
  const net = Network.createMLP(input, [hiddenSize], output);
  const hidden = net.nodes.filter((n: any) => n.type === 'hidden');
  // Assign simple biases & ensure self connections only for cell group
  const cellStart = unit * 2;
  for (let i = 0; i < hidden.length; i++) {
    hidden[i].bias = i * 0.01;
    if (i >= cellStart && i < cellStart + unit) {
      // ensure self connection
      const h = hidden[i];
      if (!h.connections.self.length)
        h.connect(h, 0.5 + (i - cellStart) * 0.01);
      else h.connections.self[0].weight = 0.5 + (i - cellStart) * 0.01;
    }
  }
  return net;
}

/** Utility: create synthetic layer partition for GRU heuristic */
function buildPartitionedGRU(input: number, unit: number, output: number) {
  // Total hidden = 4 * unit (update, reset, candidate, output block)
  const hiddenSize = unit * 4;
  const net = Network.createMLP(input, [hiddenSize], output);
  const hidden = net.nodes.filter((n: any) => n.type === 'hidden');
  // Self connections only for candidate group (third group)
  const candStart = unit * 2;
  for (let i = 0; i < hidden.length; i++) {
    hidden[i].bias = i * 0.02;
    if (i >= candStart && i < candStart + unit) {
      const h = hidden[i];
      if (!h.connections.self.length)
        h.connect(h, 0.7 + (i - candStart) * 0.02);
      else h.connections.self[0].weight = 0.7 + (i - candStart) * 0.02;
    }
  }
  return net;
}

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('ONNX Export (Experimental Fused Recurrent)', () => {
  describe('LSTM heuristic emission', () => {
    let onnx: any;
    const unit = 2;
    beforeEach(() => {
      const net = buildPartitionedLSTM(3, unit, 1);
      onnx = exportToONNX(net, { allowRecurrent: true });
    });
    it('emits LSTM initializers', () => {
      const hasW = onnx.graph.initializer.some(
        (t: any) => t.name === 'LSTM_W0'
      );
      expect(hasW).toBe(true);
    });
    it('emits LSTM node', () => {
      const hasNode = onnx.graph.node.some((n: any) => n.op_type === 'LSTM');
      expect(hasNode).toBe(true);
    });
    it('records metadata for LSTM emission', () => {
      const meta = (onnx.metadata_props || []).some(
        (m: any) => m.key === 'lstm_emitted_layers'
      );
      expect(meta).toBe(true);
    });
  });

  describe('GRU heuristic emission', () => {
    let onnx: any;
    const unit = 3;
    beforeEach(() => {
      const net = buildPartitionedGRU(2, unit, 1);
      onnx = exportToONNX(net, { allowRecurrent: true });
    });
    it('emits GRU initializers', () => {
      const hasW = onnx.graph.initializer.some((t: any) => t.name === 'GRU_W0');
      expect(hasW).toBe(true);
    });
    it('emits GRU node', () => {
      const hasNode = onnx.graph.node.some((n: any) => n.op_type === 'GRU');
      expect(hasNode).toBe(true);
    });
    it('records metadata for GRU emission', () => {
      const meta = (onnx.metadata_props || []).some(
        (m: any) => m.key === 'gru_emitted_layers'
      );
      expect(meta).toBe(true);
    });
  });

  describe('Fallback pattern metadata (near-miss)', () => {
    let onnx: any;
    beforeEach(() => {
      // Construct hidden size triggering fallback (size 9 between GRU(8) and LSTM(10) thresholds)
      const net = Network.createMLP(2, [9], 1);
      // Add self connections to half just to simulate recurrence but not matching exact partition
      const hidden = net.nodes.filter((n: any) => n.type === 'hidden');
      hidden.forEach((h: any, i: number) => {
        if (i % 2 === 0) h.connect(h, 0.3);
      });
      onnx = exportToONNX(net, { allowRecurrent: true });
    });
    it('records rnn_pattern_fallback metadata', () => {
      const hasFallback = (onnx.metadata_props || []).some(
        (m: any) => m.key === 'rnn_pattern_fallback'
      );
      expect(hasFallback).toBe(true);
    });
  });

  describe('Import reconstruction (LSTM)', () => {
    let roundTrip: any;
    const unit = 2;
    beforeEach(() => {
      const net = buildPartitionedLSTM(2, unit, 1);
      const exported = exportToONNX(net, { allowRecurrent: true });
      roundTrip = importFromONNX(exported);
    });
    it('rebuilds a network with same input/output sizes', () => {
      const inputs = roundTrip.nodes.filter((n: any) => n.type === 'input')
        .length;
      expect(inputs).toBe(2);
    });
  });

  describe('Import reconstruction (GRU)', () => {
    let roundTrip: any;
    const unit = 2;
    beforeEach(() => {
      const net = buildPartitionedGRU(2, unit, 1);
      const exported = exportToONNX(net, { allowRecurrent: true });
      roundTrip = importFromONNX(exported);
    });
    it('rebuilds a network with same output size', () => {
      const outputs = roundTrip.nodes.filter((n: any) => n.type === 'output')
        .length;
      expect(outputs).toBe(1);
    });
  });

  describe('Missing recurrent initializer safety', () => {
    let onnx: any;
    beforeEach(() => {
      const net = buildPartitionedLSTM(2, 2, 1);
      onnx = exportToONNX(net, { allowRecurrent: true });
      // Remove one initializer to simulate corruption
      onnx.graph.initializer = onnx.graph.initializer.filter(
        (t: any) => !t.name.startsWith('LSTM_R')
      );
    });
    it('imports without throwing when recurrent tensors missing', () => {
      const imp = () => importFromONNX(onnx);
      expect(() => imp()).not.toThrow();
    });
  });
});
