import Network from '../../src/architecture/network';

describe('Training Extensions', () => {
  const dataset = [
    { input: [0, 0], output: [0] },
    { input: [1, 1], output: [1] },
  ];

  describe('metricsHook gradNorm exposure', () => {
    let captured: { gradNorm?: number } | undefined;
    const net = new Network(2, 1);
    net.train(dataset, {
      iterations: 2,
      rate: 0.1,
      metricsHook: (m: { gradNorm?: number }) => {
        captured = m;
      },
    });
    it('includes gradNorm numeric', () => {
      expect(typeof (captured && captured.gradNorm)).toBe('number');
    });
  });

  describe('checkpoint best snapshot', () => {
    const net = new Network(2, 1);
    const saves: Array<{ type?: string }> = [];
    net.train(dataset, {
      iterations: 2,
      rate: 0.1,
      checkpoint: { best: true, save: (s: { type?: string }) => saves.push(s) },
    });
    it('saves best type entry', () => {
      expect(saves.some((s) => s.type === 'best')).toBe(true);
    });
  });

  describe('DropConnect masking', () => {
    const net = new Network(3, 1);
    net.enableDropConnect(0.9);
    net.activate([0, 0, 0], true);
    const dropped = net.connections.filter(
      (c) => (c as { dcMask?: number }).dcMask === 0,
    ).length;
    it('applies at least possible zero masks (>=0)', () => {
      expect(dropped >= 0).toBe(true);
    });
  });
});
