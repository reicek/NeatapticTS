import Network from '../../src/architecture/network';

// Deterministic small dataset y = 2x
const data = Array.from({ length: 4 }, (_, index) => ({
  input: [index + 1],
  output: [2 * (index + 1)],
}));

const buildNet = (): Network => {
  const net = new Network(1, 1);
  // set fixed weights
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

describe('Optimizer specific behaviors', () => {
  describe('adamw', () => {
    const net = buildNet();
    const before = net.connections[0].weight;
    net.train(data, {
      iterations: 5,
      rate: 0.01,
      optimizer: { type: 'adamw', weightDecay: 0.1 },
    });
    const after = net.connections[0].weight;
    it('applies decoupled weight decay (weight decreased)', () => {
      expect(after).toBeLessThan(before);
    });
  });

  describe('lion', () => {
    const net = buildNet();
    const before = net.connections[0].weight;
    net.train(data, {
      iterations: 5,
      rate: 0.01,
      optimizer: { type: 'lion', beta1: 0.9, beta2: 0.99 },
    });
    const after = net.connections[0].weight;
    it('updates weight (changed from before)', () => {
      expect(after).not.toBe(before);
    });
    // Bound magnitude in separate single-expectation test
    it('produces bounded cumulative magnitude (|delta| < 0.25)', () => {
      expect(Math.abs(after - before)).toBeLessThan(0.25);
    });
  });

  describe('lookahead sync', () => {
    const net = buildNet();
    net.train(data, {
      iterations: 6,
      rate: 0.01,
      optimizer: {
        type: 'lookahead',
        baseType: 'adam',
        la_k: 3,
        la_alpha: 0.5,
      },
    });
    const connection = net.connections[0];
    it('creates shadow weight', () => {
      expect(connection.lookaheadShadowWeight).toBeDefined();
    });
    it('synchronizes weight to shadow on k-multiple', () => {
      const latestConnection = net.connections[0];
      expect(latestConnection.weight).toBeCloseTo(
        latestConnection.lookaheadShadowWeight!,
        10
      );
    });
  });

  describe('string optimizer normalization', () => {
    const net = buildNet();
    net.train(data, { iterations: 1, rate: 0.01, optimizer: 'adam' });
    const connection = net.connections[0];
    it('creates first moment', () => {
      expect(connection.firstMoment).toBeDefined();
    });
    it('creates second moment', () => {
      expect(connection.secondMoment).toBeDefined();
    });
  });

  describe('invalid optimizer name', () => {
    it('throws error', () => {
      const net = buildNet();
      expect(() =>
        net.train(data, { iterations: 1, rate: 0.01, optimizer: 'nope' })
      ).toThrow('Unknown optimizer type');
    });
  });

  describe('nested lookahead rejection', () => {
    it('throws error', () => {
      const net = buildNet();
      expect(() =>
        net.train(data, {
          iterations: 1,
          rate: 0.01,
          optimizer: { type: 'lookahead', baseType: 'lookahead' },
        })
      ).toThrow('Nested lookahead');
    });
  });
});
