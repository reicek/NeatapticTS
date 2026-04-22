import type Node from '../node';
import {
  assertTargetInputSize,
  propagateNodesInReverse,
} from './layer.propagation.utils';

describe('layer propagation utilities chapter', () => {
  describe('assertTargetInputSize()', () => {
    describe('given no target array', () => {
      describe('when validating the input size', () => {
        it('accepts the missing target payload', () => {
          // Arrange
          const runValidation = () => assertTargetInputSize(2);

          // Act
          const thrownMessage = captureErrorMessage(runValidation);

          // Assert
          expect(thrownMessage).toBeUndefined();
        });
      });
    });

    describe('given a target array with the expected size', () => {
      describe('when validating the input size', () => {
        it('accepts the aligned target payload', () => {
          // Arrange
          const runValidation = () => assertTargetInputSize(2, [1, 0]);

          // Act
          const thrownMessage = captureErrorMessage(runValidation);

          // Assert
          expect(thrownMessage).toBeUndefined();
        });
      });
    });

    describe('given a target array with the wrong size', () => {
      describe('when validating the input size', () => {
        it('throws the shared layer size mismatch error message', () => {
          // Arrange
          const runValidation = () => assertTargetInputSize(2, [1]);

          // Act
          const thrownMessage = captureErrorMessage(runValidation);

          // Assert
          expect(thrownMessage).toBe(
            'Array with values should be same as the amount of nodes!',
          );
        });
      });
    });
  });

  describe('propagateNodesInReverse()', () => {
    describe('given a propagation context with three nodes and no explicit targets', () => {
      describe('when propagating in reverse', () => {
        it('calls each node in reverse order with the hidden-layer signature', () => {
          // Arrange
          const callLog: Array<{
            args: Array<boolean | number>;
            label: string;
          }> = [];
          const context = {
            nodes: [
              createPropagationNode('first', callLog),
              createPropagationNode('second', callLog),
              createPropagationNode('third', callLog),
            ],
          };

          // Act
          propagateNodesInReverse(context, 0.3, 0.1);

          // Assert
          expect(callLog).toStrictEqual([
            { args: [0.3, 0.1, true, 0], label: 'third' },
            { args: [0.3, 0.1, true, 0], label: 'second' },
            { args: [0.3, 0.1, true, 0], label: 'first' },
          ]);
        });
      });
    });

    describe('given a propagation context with three nodes and explicit targets', () => {
      describe('when propagating in reverse', () => {
        it('calls each node in reverse order with its aligned target value', () => {
          // Arrange
          const callLog: Array<{
            args: Array<boolean | number>;
            label: string;
          }> = [];
          const context = {
            nodes: [
              createPropagationNode('first', callLog),
              createPropagationNode('second', callLog),
              createPropagationNode('third', callLog),
            ],
          };

          // Act
          propagateNodesInReverse(context, 0.3, 0.1, [0.2, 0.4, 0.6]);

          // Assert
          expect(callLog).toStrictEqual([
            { args: [0.3, 0.1, true, 0, 0.6], label: 'third' },
            { args: [0.3, 0.1, true, 0, 0.4], label: 'second' },
            { args: [0.3, 0.1, true, 0, 0.2], label: 'first' },
          ]);
        });
      });
    });
  });
});

function createPropagationNode(
  label: string,
  callLog: Array<{ args: Array<boolean | number>; label: string }>,
): Node {
  return {
    propagate: (...args: Array<boolean | number>) => {
      callLog.push({ args, label });
    },
  } as unknown as Node;
}

function captureErrorMessage(runAction: () => unknown): string | undefined {
  try {
    runAction();
  } catch (error: unknown) {
    return error instanceof Error ? error.message : String(error);
  }

  return undefined;
}