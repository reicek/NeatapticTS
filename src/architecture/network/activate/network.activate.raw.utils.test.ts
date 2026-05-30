import Network from '../network';
import { createRawActivationContext } from './network.activate.contexts.utils';
import { executeRawActivation } from './network.activate.raw.utils';

describe('network activate raw utility chapter', () => {
  describe('executeRawActivation', () => {
    it('returns the reusable typed output buffer when typed activations are enabled', () => {
      const network = new Network(1, 1, {
        activationPrecision: 'f32',
        reuseActivationArrays: true,
        returnTypedActivations: true,
        seed: 6,
      });

      Reflect.set(network, '_canUseFastSlab', () => false);

      const activationOutput = executeRawActivation(
        createRawActivationContext(network, [0.5], false, 32),
      );

      expect(activationOutput).toBeInstanceOf(Float32Array);
    });
  });
});