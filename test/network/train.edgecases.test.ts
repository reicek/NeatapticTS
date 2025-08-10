import Network from '../../src/architecture/network';
import { config } from '../../src/config';

describe('Network.train edge cases', () => {
  const set = [ { input:[0], output:[0] } ];

  describe('invalid dataset dimensions', () => {
    it('throws dimension mismatch error', () => {
      const net = new Network(2,1);
      expect(()=> net.train(set as any, { iterations:1, rate:0.1 })).toThrow('Dataset is invalid');
    });
  });

  describe('missing stopping condition', () => {
    it('throws when neither iterations nor error provided', () => {
      const net = new Network(1,1);
      const goodSet = [ { input:[0], output:[0] } ];
      expect(()=> net.train(goodSet, { rate:0.1 })).toThrow();
    });
  });

  describe('invalid cost object', () => {
    it('throws invalid cost function error', () => {
      const net = new Network(1,1);
      const goodSet = [ { input:[0], output:[0] } ];
      expect(()=> net.train(goodSet, { iterations:1, rate:0.1, cost: { nope:true } as any })).toThrow('Invalid cost function');
    });
  });

  describe('batchSize larger than dataset', () => {
    it('throws batch size error', () => {
      const net = new Network(1,1);
      const goodSet = [ { input:[0], output:[0] } ];
      expect(()=> net.train(goodSet, { iterations:1, rate:0.1, batchSize: 5 })).toThrow('larger than the dataset length');
    });
  });

  describe('warnings for missing rate (enabled warnings)', () => {
    it('emits console warnings once', () => {
      const net = new Network(1,1);
      const goodSet = [ { input:[0], output:[0] } ];
      const original = config.warnings; config.warnings = true;
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation(()=>{});
      net.train(goodSet, { iterations:1 });
      const count = warnSpy.mock.calls.length;
      warnSpy.mockRestore(); config.warnings = original;
      expect(count).toBeGreaterThan(0);
    });
  });
});
