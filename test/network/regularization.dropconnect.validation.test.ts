import Network from '../../src/architecture/network';

describe('Regularization', () => {
  describe('DropConnect validation', () => {
    it('throws on invalid probability (<0)', () => {
      const net = new Network(1, 1);
      expect(() => net.enableDropConnect(-0.1)).toThrow(
        'DropConnect probability must be in [0,1)'
      );
    });
    it('throws on invalid probability (>=1)', () => {
      const net = new Network(1, 1);
      expect(() => net.enableDropConnect(1)).toThrow(
        'DropConnect probability must be in [0,1)'
      );
    });
  });
});
