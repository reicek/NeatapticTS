import { gaussianRand } from './network.activate.core.utils';

describe('network activate core utilities sibling seam', () => {
  describe('gaussianRand', () => {
    it('uses the Box-Muller transform on the supplied random pair', () => {
      const randomValues = [0.25, 0.5];
      let randomValueIndex = 0;
      const gaussianSample = gaussianRand(
        () => randomValues[randomValueIndex++] ?? 0.5,
      );

      expect(gaussianSample).toBeCloseTo(-Math.sqrt(-2 * Math.log(0.25)));
    });
  });
});
