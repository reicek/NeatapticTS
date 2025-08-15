import Network from '../../src/architecture/network';

/**
 * Activation path tests covering fast slab usage, fallback behavior, batch utilities,
 * raw / no-trace wrappers and validation error branches. Each test follows AAA and
 * uses a single expectation to keep intent focused and educative.
 */
describe('Network.activation helpers', () => {
  describe('Scenario: noTraceActivate fast slab path', () => {
    it('returns value produced by fast slab stub when fast path succeeds', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 11, enforceAcyclic: true });
      // Monkey patch internal fast path eligibility + implementation to force that branch deterministically.
      (net as any)._canUseFastSlab = () => true;
      (net as any)._fastSlabActivate = (input: number[]) => [
        input[0] + input[1],
      ]; // simple deterministic stub
      // Act
      const out = net.noTraceActivate([0.25, 0.75]);
      // Assert
      expect(out[0]).toBe(1.0);
    });
  });

  describe('Scenario: noTraceActivate fallback path after fast slab throws', () => {
    it('produces output vector of correct length via standard loop when fast slab fails', () => {
      // Arrange
      const net = new Network(3, 2, { seed: 12, enforceAcyclic: true });
      (net as any)._canUseFastSlab = () => true;
      (net as any)._fastSlabActivate = () => {
        throw new Error('forced');
      };
      const input = [0.1, 0.2, 0.3];
      // Act
      const out = net.noTraceActivate(input);
      // Assert
      expect(out.length).toBe(2);
    });
  });

  describe('Scenario: noTraceActivate input validation', () => {
    it('throws on size mismatch', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 13 });
      // Act / Assert
      expect(() => net.noTraceActivate([1, 2, 3])).toThrow(
        /Input size mismatch/
      );
    });
  });

  describe('Scenario: activateRaw delegation (reuseActivationArrays=false)', () => {
    it('returns output vector of expected length', () => {
      // Arrange
      const net = new Network(2, 2, { seed: 14 });
      // Act
      const out = net.activateRaw([0.4, 0.6]);
      // Assert
      expect(out.length).toBe(2);
    });
  });

  describe('Scenario: activateRaw delegation (reuseActivationArrays=true)', () => {
    it('still returns output vector of expected length when reuse enabled', () => {
      // Arrange
      const net = new Network(2, 3, {
        seed: 15,
        reuseActivationArrays: true,
        returnTypedActivations: true,
      });
      // Act
      const out = net.activateRaw([0.5, 0.5]);
      // Assert
      expect(out.length).toBe(3);
    });
  });

  describe('Scenario: activateBatch happy path', () => {
    it('returns one output row per input row', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 16 });
      const batch = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
      ];
      // Act
      const out = net.activateBatch(batch);
      // Assert
      expect(out.length).toBe(3);
    });
  });

  describe('Scenario: activateBatch row size mismatch', () => {
    it('throws descriptive error mentioning index', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 17 });
      // Act / Assert
      expect(() => net.activateBatch([[1, 2], [3]])).toThrow(
        /Input\[1] size mismatch/
      );
    });
  });

  describe('Scenario: activateBatch non-array input', () => {
    it('throws when inputs argument is not an array', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 18 });
      // Act / Assert
      expect(() => (net as any).activateBatch('nope')).toThrow(
        /inputs must be an array/
      );
    });
  });
});
