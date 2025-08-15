import Network from '../../src/architecture/network';

// This suite exercises a broad set of training options in a single run while
// keeping one expectation per "it" as required.
describe('Network.train with { crossValidate, shuffle, dropout, schedule, log, clear } options', () => {
  let net: Network;
  let result: any;
  let scheduleFn: jest.Mock;
  let hiddenMasksReset: boolean;

  beforeAll(() => {
    net = new Network(2, 1);
    const set = [
      { input: [0, 0], output: [0] },
      { input: [1, 1], output: [1] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
    ];

    scheduleFn = jest.fn();
    result = net.train(set, {
      rate: 0.2,
      iterations: 2,
      shuffle: true,
      dropout: 0.5,
      batchSize: 2,
      cost: () => 0, // force zero error to exercise early stop & log branches
      crossValidate: { testSize: 0.5, testError: 0 },
      log: 1,
      schedule: { iterations: 1, function: scheduleFn },
      clear: true,
      optimizer: 'sgd',
    });

    hiddenMasksReset = net.nodes
      .filter((n) => n.type === 'hidden')
      .every((n) => n.mask === 1);
  });

  it('reports numeric error', () => {
    expect(typeof result.error).toBe('number');
  });
  it('performs at least one iteration', () => {
    expect(result.iterations).toBeGreaterThanOrEqual(1);
  });
  it('invokes schedule function', () => {
    expect(scheduleFn).toHaveBeenCalled();
  });
  it('resets hidden node dropout masks to 1', () => {
    expect(hiddenMasksReset).toBe(true);
  });
});
