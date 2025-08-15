import Network from '../../src/architecture/network';
import * as methods from '../../src/methods/methods';

// Helper dataset
const xor = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

describe('Network structural pruning & sparsification', () => {
  describe('magnitude pruning schedule reduces connections toward target sparsity', () => {
    const net = Network.createMLP(2, [4, 4], 1);
    const initialConns = net.connections.length;
    net.configurePruning({
      start: 1,
      end: 5,
      targetSparsity: 0.5,
      regrowFraction: 0,
      frequency: 1,
    });
    for (let i = 0; i < 5; i++) {
      net.train(xor, { iterations: 1, rate: 0.1, error: 0.00001, log: 0 });
    }
    test('sparsity progressed above 0', () => {
      expect(net.getCurrentSparsity()).toBeGreaterThan(0);
    });
    test('connections not below half minus tolerance', () => {
      expect(net.connections.length).toBeGreaterThanOrEqual(
        Math.floor(initialConns * 0.4)
      );
    });
  });

  describe('regrow fraction adds some new connections after pruning', () => {
    const net = Network.createMLP(2, [3], 1);
    const initial = net.connections.length;
    net.configurePruning({
      start: 1,
      end: 2,
      targetSparsity: 0.4,
      regrowFraction: 0.5,
    });
    net.train(xor, { iterations: 1, rate: 0.1, error: 0.00001, log: 0 });
    net.train(xor, { iterations: 1, rate: 0.1, error: 0.00001, log: 0 });
    const after = net.connections.length;
    test('after pruning+regrow connections not extremely low', () =>
      expect(after).toBeGreaterThan(initial * 0.3));
  });
});

describe('Acyclic enforcement', () => {
  describe('disallows back connections and self connections when enabled', () => {
    const net = new Network(2, 1);
    net.setEnforceAcyclic(true);
    const before = net.connections.length;
    // Try to add back connection by forcing mutation ADD_BACK_CONN repeatedly
    for (let i = 0; i < 10; i++) net.mutate(methods.mutation.ADD_BACK_CONN);
    for (let i = 0; i < 10; i++) net.mutate(methods.mutation.ADD_SELF_CONN);
    const after = net.connections.length;
    test('no new recurrent/self connections added', () =>
      expect(after).toBe(before));
  });
});
