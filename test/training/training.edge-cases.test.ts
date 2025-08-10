import Network from '../../src/architecture/network';
import { config } from '../../src/config';

describe('training.edge-cases', () => {
  const set = [{ input: [0], output: [0] }];

  describe('invalid dataset dimensions', () => {
    let threw = false;
    beforeAll(() => {
      const net = new Network(2, 1);
      try {
        net.train(set as any, { iterations: 1, rate: 0.1 });
      } catch (e: any) {
        threw = e.message.includes('Dataset is invalid');
      }
    });
    it('throws dimension mismatch error', () => {
      expect(threw).toBe(true);
    });
  });

  describe('missing stopping condition', () => {
    let threw = false;
    beforeAll(() => {
      const net = new Network(1, 1);
      const goodSet = [{ input: [0], output: [0] }];
      try {
        net.train(goodSet, { rate: 0.1 });
      } catch {
        threw = true;
      }
    });
    it('throws when neither iterations nor error provided', () => {
      expect(threw).toBe(true);
    });
  });

  describe('invalid cost object', () => {
    let threw = false;
    beforeAll(() => {
      const net = new Network(1, 1);
      const goodSet = [{ input: [0], output: [0] }];
      try {
        net.train(goodSet, {
          iterations: 1,
          rate: 0.1,
          cost: { nope: true } as any,
        });
      } catch (e: any) {
        threw = e.message.includes('Invalid cost function');
      }
    });
    it('throws invalid cost function error', () => {
      expect(threw).toBe(true);
    });
  });

  describe('batchSize larger than dataset', () => {
    let threw = false;
    beforeAll(() => {
      const net = new Network(1, 1);
      const goodSet = [{ input: [0], output: [0] }];
      try {
        net.train(goodSet, { iterations: 1, rate: 0.1, batchSize: 5 });
      } catch (e: any) {
        threw = e.message.includes('larger than the dataset length');
      }
    });
    it('throws batch size error', () => {
      expect(threw).toBe(true);
    });
  });

  describe('warnings for missing rate (enabled warnings)', () => {
    let warned = false;
    beforeAll(() => {
      const net = new Network(1, 1);
      const goodSet = [{ input: [0], output: [0] }];
      const original = config.warnings;
      config.warnings = true;
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      net.train(goodSet, { iterations: 1 });
      warned = warnSpy.mock.calls.length > 0;
      warnSpy.mockRestore();
      config.warnings = original;
    });
    it('emits console warnings when rate missing', () => {
      expect(warned).toBe(true);
    });
  });
});
