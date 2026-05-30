import Network from '../network';
import { activateBatch } from './network.activate.utils';

describe('network activate utilities sibling seam', () => {
  describe('activateBatch', () => {
    it('returns one output row per batch input row', () => {
      const network = new Network(1, 1, { seed: 7 });
      const batchOutput = activateBatch.call(
        network,
        [[0], [1]],
        false,
      );

      expect(batchOutput).toHaveLength(2);
    });
  });
});