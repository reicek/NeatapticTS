import Network from '../../src/architecture/network';

// Helper to build tiny network (1 input -> 1 output) for deterministic gradient on identity activation
const buildNet = (): Network => {
  const net = new Network(1, 1);
  // Force deterministic weights/bias
  net.connections.forEach((connection) => {
    connection.weight = 0.5;
  });
  net.nodes
    .filter((node) => node.type !== 'input')
    .forEach((node) => {
      node.bias = 0;
    });
  return net;
};

// Simple dataset: y = 2x so gradient direction is clear
const data = Array.from({ length: 5 }, (_, i) => ({
  input: [i],
  output: [2 * i],
}));

type OptimizerConfig = { type: string } & Record<string, unknown>;

// Run few iterations and collect first weight change sign/magnitude
const trainWith = (net: Network, optimizerConfig: OptimizerConfig) => {
  const before = net.connections[0].weight;
  net.train(data, {
    iterations: 3,
    rate: 0.01,
    error: 0,
    cost: {
      fn: (t: number[], o: number[]) => (o[0] - t[0]) ** 2,
      calculate: (t: number[], o: number[]) => (o[0] - t[0]) ** 2,
    },
    optimizer: optimizerConfig,
    batchSize: 1,
  });
  const after = net.connections[0].weight;
  return { before, after, delta: after - before };
};

describe('Advanced optimizers', () => {
  describe('adamax', () => {
    const { delta } = trainWith(buildNet(), { type: 'adamax' });
    it('updates weight (non-zero delta)', () => {
      expect(delta).not.toBe(0);
    });
  });
  describe('nadam', () => {
    const { delta } = trainWith(buildNet(), { type: 'nadam' });
    it('applies nesterov style update (delta non-zero)', () => {
      expect(delta).not.toBe(0);
    });
  });
  describe('radam', () => {
    const { delta } = trainWith(buildNet(), { type: 'radam' });
    it('performs rectified adaptive update (delta non-zero)', () => {
      expect(delta).not.toBe(0);
    });
  });
  describe('lion', () => {
    const { delta } = trainWith(buildNet(), { type: 'lion' });
    it('uses sign-based update (delta non-zero)', () => {
      expect(delta).not.toBe(0);
    });
  });
  describe('adabelief', () => {
    const { delta } = trainWith(buildNet(), { type: 'adabelief' });
    it('adapts with belief variance (delta non-zero)', () => {
      expect(delta).not.toBe(0);
    });
  });
});

describe('Lookahead wrapper', () => {
  const net = buildNet();
  const result = trainWith(net, {
    type: 'lookahead',
    baseType: 'adam',
    la_k: 2,
    la_alpha: 0.5,
  });
  it('applies lookahead blended update (delta non-zero)', () => {
    expect(result.delta).not.toBe(0);
  });
});
