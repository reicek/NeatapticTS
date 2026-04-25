import type Network from '../../network/network';
import { executeNoTraceActivation } from './network.activate.notrace.utils';
import type { NoTraceActivationContext } from './network.activate.utils.types';

describe('network activate no-trace helper chapter', () => {
  describe('executeNoTraceActivation', () => {
    describe('given the input vector is undefined at runtime', () => {
      it('throws an input-size mismatch that reports an undefined length', () => {
        // Arrange
        const activationContext = {
          expectedInputSize: 2,
          inputVector: undefined as unknown as number[],
          network: {
            output: 1,
          } as unknown as Network,
          networkInternal: {
            _canUseFastSlab: jest.fn(() => false),
            _computeTopoOrder: jest.fn(),
            _fastSlabActivate: jest.fn(() => []),
            _topoDirty: false,
          },
        } as unknown as NoTraceActivationContext;

        // Act
        const executeWithUndefinedInput = () =>
          executeNoTraceActivation(activationContext);

        // Assert
        expect(executeWithUndefinedInput).toThrow(
          'Input size mismatch: expected 2, got undefined',
        );
      });
    });
  });
});
