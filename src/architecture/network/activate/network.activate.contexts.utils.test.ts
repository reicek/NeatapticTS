import Network from '../network';
import {
  createBatchActivationContext,
  createNoTraceActivationContext,
  createRawActivationContext,
} from './network.activate.contexts.utils';

describe('network activate contexts utility chapter', () => {
  describe('createNoTraceActivationContext', () => {
    it('captures the bound network input width', () => {
      const network = new Network(2, 1, { seed: 1 });
      const activationContext = createNoTraceActivationContext(
        network,
        [0.2, 0.8],
      );

      expect(activationContext.expectedInputSize).toBe(2);
    });
  });

  describe('createRawActivationContext', () => {
    it('preserves the explicit activation-depth guard', () => {
      const network = new Network(1, 1, { seed: 2 });
      const activationContext = createRawActivationContext(
        network,
        [0.5],
        false,
        64,
      );

      expect(activationContext.maximumActivationDepth).toBe(64);
    });
  });

  describe('createBatchActivationContext', () => {
    it('keeps the caller batch row count available to downstream helpers', () => {
      const network = new Network(1, 1, { seed: 3 });
      const activationContext = createBatchActivationContext(
        network,
        [[0], [1]],
        false,
      );

      expect(activationContext.batchInputs.length).toBe(2);
    });
  });
});
