import { activationArrayPool } from '../../src/architecture/activationArrayPool';
import { config } from '../../src/config';

describe('ActivationArrayPool', () => {
  describe('default mode', () => {
    beforeEach(() => {
      config.float32Mode = false;
      activationArrayPool.clear();
    });
    const size = 5;
    const arr = () => activationArrayPool.acquire(size);
    it('acquires array of requested length', () => {
      expect(arr().length).toBe(size);
    });
    describe('reuse', () => {
      const acquired = activationArrayPool.acquire(size);
      acquired[0] = 123 as any;
      activationArrayPool.release(acquired);
      const again = activationArrayPool.acquire(size);
      it('returns the same array reference after release', () => {
        expect(again).toBe(acquired);
      });
      it('zero-fills on reuse', () => {
        expect(again[0]).toBe(0 as any);
      });
    });
    it('clear empties all buckets', () => {
      const a = activationArrayPool.acquire(size);
      activationArrayPool.release(a);
      activationArrayPool.clear();
      const b = activationArrayPool.acquire(size);
      expect(b === a).toBe(false);
    });
  });

  describe('float32 mode', () => {
    beforeEach(() => {
      config.float32Mode = true;
      activationArrayPool.clear();
    });
    it('acquires Float32Array when enabled', () => {
      const arr = activationArrayPool.acquire(3);
      expect(arr instanceof Float32Array).toBe(true);
    });
  });
});
