import { Network } from '../../../neataptic';
import { NetworkEvolveStoppingConditionRequiredError } from './network.evolve.errors';
import { evolveNetwork } from './network.evolve.utils';

describe('network evolve utils', () => {
  describe('evolveNetwork()', () => {
    describe('given stopping conditions are omitted', () => {
      describe('when the helper is called directly', () => {
        it('rejects with the stopping-condition-required error', async () => {
          // Arrange
          const network = new Network(2, 1);
          const trainingSet = [{ input: [0, 1], output: [1] }];

          // Act
          const evolvePromise = evolveNetwork.call(network, trainingSet);

          // Assert
          await expect(evolvePromise).rejects.toThrow(
            NetworkEvolveStoppingConditionRequiredError,
          );
        });
      });
    });
  });
});
