import Network from '../network';
import { createBatchActivationContext } from './network.activate.helpers.utils';

describe('network activate helper re-export chapter', () => {
  describe('createBatchActivationContext', () => {
    it('re-exports the batch context creator for helper callers', () => {
      const network = new Network(1, 1, { seed: 4 });
      const activationContext = createBatchActivationContext(
        network,
        [[1]],
        false,
      );

      expect(activationContext.expectedInputSize).toBe(1);
    });
  });
});
