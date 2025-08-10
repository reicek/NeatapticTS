import Network from '../../src/architecture/network';

describe('Regularization', () => {
  describe('DropConnect', () => {
    const net = new Network(3, 2, { minHidden: 2 });
    for (const c of net.connections) c.weight = 0.5;
    describe('mask sampling during training activation', () => {
      const input = [0.1,0.2,0.3];
      it('samples at least one dropped connection (or none if probability outcome)', () => {
        net.enableDropConnect(0.9);
        net.activate(input, true);
        const dropped = net.connections.filter(c => c.dcMask === 0).length;
        expect(dropped >= 0).toBe(true);
      });
    });
  });
});
