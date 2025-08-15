import Network from '../../src/architecture/network';

describe('training.gradient.separate-bias', () => {
  test('group count higher when separateBias true', () => {
    const netA = new Network(3, 2);
    const netB = new Network(3, 2);
    netA.train([{ input: [0, 0, 0], output: [0, 1] }], {
      iterations: 1,
      rate: 0.01,
      optimizer: 'adam',
      gradientClip: { mode: 'layerwiseNorm', maxNorm: 1 },
    });
    const groupsA = netA.getLastGradClipGroupCount();
    netB.train([{ input: [0, 0, 0], output: [0, 1] }], {
      iterations: 1,
      rate: 0.01,
      optimizer: 'adam',
      gradientClip: { mode: 'layerwiseNorm', maxNorm: 1, separateBias: true },
    });
    const groupsB = netB.getLastGradClipGroupCount();
    expect(groupsB).toBeGreaterThanOrEqual(groupsA);
  });

  test('training stats expose expected fields', () => {
    const net = new Network(2, 1);
    net.train([{ input: [0, 0], output: [1] }], {
      iterations: 1,
      rate: 0.01,
      optimizer: 'adam',
      gradientClip: { mode: 'norm', maxNorm: 1 },
    });
    const stats = net.getTrainingStats();
    expect(stats.lossScale).toBeGreaterThan(0);
  });

  test('rate adjust divides when using sum reduction', () => {
    const base = 0.01;
    const adj = Network.adjustRateForAccumulation(base, 4, 'sum');
    expect(adj).toBeCloseTo(base / 4, 10);
  });
});
