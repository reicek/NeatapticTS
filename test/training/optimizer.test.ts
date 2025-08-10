import Network from '../../src/architecture/network';

// Simple dataset: y = 0 for x = 1
const set = [{ input: [1], output: [0] }];

const optimizers: Array<any> = [
  'sgd',
  'rmsprop',
  'adagrad',
  'adam',
  'adamw',
];

describe('Training with optional optimizers', () => {
  it('updates weights for each optimizer and tracks moments where applicable', () => {
    for (const opt of optimizers) {
      const net = new Network(1, 1);
      const conn = net.connections[0];
      const w0 = conn.weight;

      net.train(set, { iterations: 1, rate: 0.2, batchSize: 1, optimizer: opt });

      expect(conn.weight).not.toBe(w0);
      if (opt === 'adam' || opt === 'adamw') {
        expect(typeof conn.opt_m !== 'undefined').toBe(true);
        expect(typeof conn.opt_v !== 'undefined').toBe(true);
      }
      if (opt === 'rmsprop' || opt === 'adagrad') {
        expect(typeof conn.opt_cache !== 'undefined').toBe(true);
      }
    }
  });
});
