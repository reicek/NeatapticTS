import type { BatchActivationContext } from './network.activate.utils.types';
import { executeBatchActivation } from './network.activate.batch.utils';

function createBatchActivationContext(
  batchInputs: number[][],
): BatchActivationContext {
  return {
    batchInputs,
    expectedInputSize: 2,
    isTraining: false,
    networkInternal: {
      activate: jest.fn(() => [1]),
    } as unknown as BatchActivationContext['networkInternal'],
  };
}

describe('network activate batch utility chapter', () => {
  describe('executeBatchActivation', () => {
    describe('given one batch row is undefined instead of an input array', () => {
      it('reports the received row width as undefined', () => {
        // Arrange
        const activationContext = createBatchActivationContext([
          undefined,
        ] as unknown as number[][]);
        const activateInvalidBatch = () =>
          executeBatchActivation(activationContext);

        // Assert
        expect(activateInvalidBatch).toThrow(
          'Input[0] size mismatch: expected 2, got undefined',
        );
      });
    });
  });
});
