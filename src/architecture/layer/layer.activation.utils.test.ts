import {
  assertActivationInputSize,
  fillActivationOutput,
  resolveLayerMask,
} from './layer.activation.utils';
import { LayerSizeMismatchError } from './layer.errors';

describe('layer activation utility chapter', () => {
  describe('assertActivationInputSize', () => {
    describe('given explicit activation values do not match the layer node count', () => {
      describe('when the size guard runs', () => {
        it('throws the layer size mismatch error', () => {
          // Arrange
          const assertMismatchedInputSize = () =>
            assertActivationInputSize(3, [0.1, 0.2]);

          // Assert
          expect(assertMismatchedInputSize).toThrow(LayerSizeMismatchError);
        });
      });
    });
  });

  describe('resolveLayerMask', () => {
    describe('given training mode is enabled and the random draw falls below the dropout rate', () => {
      afterEach(() => {
        jest.restoreAllMocks();
      });

      describe('when the layer mask is resolved', () => {
        it('returns the disabled mask value', () => {
          // Arrange
          jest.spyOn(Math, 'random').mockReturnValue(0.2);

          // Act
          const mask = resolveLayerMask(0.5, true);

          // Assert
          expect(mask).toBe(0);
        });
      });
    });

    describe('given training mode is enabled and the random draw clears the dropout rate', () => {
      afterEach(() => {
        jest.restoreAllMocks();
      });

      describe('when the layer mask is resolved', () => {
        it('returns the enabled mask value', () => {
          // Arrange
          jest.spyOn(Math, 'random').mockReturnValue(0.8);

          // Act
          const mask = resolveLayerMask(0.5, true);

          // Assert
          expect(mask).toBe(1);
        });
      });
    });
  });

  describe('fillActivationOutput', () => {
    describe('given no explicit input values are provided', () => {
      describe('when the output buffer is filled', () => {
        it('activates each node through the implicit activation path', () => {
          // Arrange
          const output = [0, 0];
          const nodeList = [
            { activate: jest.fn(() => 0.25) },
            { activate: jest.fn(() => 0.75) },
          ] as unknown as Parameters<typeof fillActivationOutput>[0];

          // Act
          fillActivationOutput(nodeList, undefined, output);

          // Assert
          expect(output).toEqual([0.25, 0.75]);
        });
      });
    });
  });
});