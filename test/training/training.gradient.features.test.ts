import Network from '../../src/architecture/network';

// Helper dataset linear y=2x
const data = Array.from({ length: 8 }, (_, i) => ({
  input: [i],
  output: [2 * i],
}));
function buildNet() {
  return new Network(1, 1);
}

describe('training.gradient.features', () => {
  describe('accumulationSteps approximate equivalence', () => {
    let matchingDirection = false;
    let similarMagnitude = false;
    beforeAll(() => {
      const netA = buildNet();
      const netB = buildNet();
      for (let i = 0; i < netA.connections.length; i++)
        netB.connections[i].weight = netA.connections[i].weight;
      for (let n = 0; n < netA.nodes.length; n++)
        (netB.nodes[n] as any).bias = (netA.nodes[n] as any).bias;
      const origWeight = netA.connections[0].weight;
      const subset = data.slice(0, 4);
      netA.train(subset, {
        iterations: 1,
        rate: 0.05,
        batchSize: 1,
        accumulationSteps: 4,
        optimizer: 'adam',
      });
      netB.train(subset, {
        iterations: 1,
        rate: 0.05,
        batchSize: 4,
        accumulationSteps: 1,
        optimizer: 'adam',
      });
      const deltaA = netA.connections[0].weight - origWeight;
      const deltaB = netB.connections[0].weight - origWeight;
      matchingDirection = Math.sign(deltaA) === Math.sign(deltaB);
      const ratio = Math.abs(deltaA) / Math.max(1e-9, Math.abs(deltaB));
      similarMagnitude = ratio < 2;
    });
    it('update direction matches', () => {
      expect(matchingDirection).toBe(true);
    });
    it('magnitude within 2x', () => {
      expect(similarMagnitude).toBe(true);
    });
  });

  describe('global norm clipping limits update magnitude', () => {
    let limited = false;
    beforeAll(() => {
      const net = buildNet();
      net.train(data, {
        iterations: 1,
        rate: 5,
        batchSize: 4,
        accumulationSteps: 1,
        optimizer: 'adam',
        gradientClip: { mode: 'norm', maxNorm: 0.01 },
      });
      const w = net.connections[0].weight;
      limited = Math.abs(w) < 20;
    });
    it('limits excessive growth (|w| < 20)', () => {
      expect(limited).toBe(true);
    });
  });

  describe('percentile clipping caps outliers', () => {
    let finite = false;
    beforeAll(() => {
      const net = buildNet();
      net.train(data, {
        iterations: 1,
        rate: 2,
        batchSize: 4,
        optimizer: 'adam',
        gradientClip: { mode: 'percentile', percentile: 90 },
      });
      const w = net.connections[0].weight;
      finite = Number.isFinite(w);
    });
    it('weight finite after percentile clipping', () => {
      expect(finite).toBe(true);
    });
  });

  describe('mixed precision applies loss scaling', () => {
    let hasMasterCopy = false;
    beforeAll(() => {
      const net = buildNet();
      net.train(data, {
        iterations: 1,
        rate: 0.01,
        batchSize: 4,
        optimizer: 'adam',
        mixedPrecision: { lossScale: 512 },
      });
      const conn: any = net.connections[0];
      hasMasterCopy = typeof conn._fp32Weight !== 'undefined';
    });
    it('stores master fp32 weight copy', () => {
      expect(hasMasterCopy).toBe(true);
    });
  });
});
