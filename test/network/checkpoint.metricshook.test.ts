import Network from '../../src/architecture/network';

/**
 * Tests checkpoint and metrics hook basics
 */
describe('Training Extensions', () => {
  describe('Checkpoint + Metrics Hook', () => {
    const dataset = [
      { input: [0, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ];
    const saves: any[] = [];
    const net = new Network(2, 1);
    it('invokes metricsHook with gradNorm', () => {
      let called = false;
      net.train(dataset, {
        iterations: 1,
        rate: 0.1,
        history: false,
        metricsHook: ({ gradNorm }: { gradNorm: number }) => {
          if (typeof gradNorm === 'number') called = true;
        },
      });
      expect(called).toBe(true);
    });
    it('saves best checkpoint snapshot', () => {
      net.train(dataset, {
        iterations: 1,
        rate: 0.1,
        checkpoint: { best: true, save: (d: any) => saves.push(d) },
      });
      const hasBest = saves.some((s) => s && s.type === 'best');
      expect(hasBest).toBe(true);
    });
  });
});
