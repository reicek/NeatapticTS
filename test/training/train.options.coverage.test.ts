import { Architect, methods } from '../../src/neataptic';
import Network from '../../src/architecture/network';

describe('Network.train options coverage', () => {
  it('covers crossValidate, shuffle, dropout, schedule, log, clear', () => {
    const net = new Network(2, 1);
    const set = [
      { input: [0, 0], output: [0] },
      { input: [1, 1], output: [1] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
    ];

    const scheduleFn = jest.fn();
    const result = net.train(set, {
      rate: 0.2,
      iterations: 2,
      shuffle: true,
      dropout: 0.5,
      batchSize: 2,
      cost: (t: number[], o: number[]) => 0, // force zero error to exercise early stop branch
      crossValidate: { testSize: 0.5, testError: 0 },
      log: 1,
      schedule: { iterations: 1, function: scheduleFn },
      clear: true,
      optimizer: 'sgd',
    });

    expect(typeof result.error).toBe('number');
    expect(result.iterations).toBeGreaterThanOrEqual(1);
    expect(scheduleFn).toHaveBeenCalled();

    // ensure dropout masks reset
    net.nodes.forEach(n => {
      if (n.type === 'hidden') expect(n.mask).toBe(1);
    });
  });
});
