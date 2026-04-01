import Cost from './cost';
import { LENGTH_MISMATCH_MESSAGE } from './cost.utils';

describe('Cost', () => {
  describe('crossEntropy()', () => {
    describe('given binary targets and probabilities', () => {
      describe('when the predictions are evaluated', () => {
        it('returns the expected mean cross-entropy', () => {
          // Arrange
          const targets = [0, 1, 0, 1];
          const outputs = [0.1, 0.9, 0.2, 0.8];
          const expectedLoss =
            -(Math.log(0.9) + Math.log(0.9) + Math.log(0.8) + Math.log(0.8)) /
            4;

          // Act
          const actualLoss = Cost.crossEntropy(targets, outputs);

          // Assert
          expect(actualLoss).toBeCloseTo(expectedLoss, 12);
        });
      });
    });

    describe('given mismatched target and output lengths', () => {
      describe('when the loss is evaluated', () => {
        it('throws the canonical length-mismatch error', () => {
          // Arrange
          const runLoss = () => Cost.crossEntropy([0, 1], [0.5]);

          // Act
          const thrownMessage = captureErrorMessage(runLoss);

          // Assert
          expect(thrownMessage).toBe(LENGTH_MISMATCH_MESSAGE);
        });
      });
    });
  });

  describe('softmaxCrossEntropy()', () => {
    describe('given raw class scores', () => {
      describe('when the loss is evaluated', () => {
        it('applies a stable softmax before computing cross-entropy', () => {
          // Arrange
          const targets = [1, 0, 0];
          const outputs = [2, 1, 0];
          const exponentials = outputs.map((outputValue) =>
            Math.exp(outputValue - 2),
          );
          const probabilitySum = exponentials.reduce(
            (runningSum, probabilityValue) => runningSum + probabilityValue,
            0,
          );
          const expectedLoss = -Math.log(
            exponentials[0] / Math.max(probabilitySum, 1),
          );

          // Act
          const actualLoss = Cost.softmaxCrossEntropy(targets, outputs);

          // Assert
          expect(actualLoss).toBeCloseTo(expectedLoss, 12);
        });
      });
    });
  });

  describe('mse()', () => {
    describe('given regression targets and predictions', () => {
      describe('when the loss is evaluated', () => {
        it('returns the expected mean squared error', () => {
          // Arrange
          const targets = [1, 2, 3];
          const outputs = [1.5, 2.5, 2.5];
          const expectedLoss = ((-0.5) ** 2 + (-0.5) ** 2 + 0.5 ** 2) / 3;

          // Act
          const actualLoss = Cost.mse(targets, outputs);

          // Assert
          expect(actualLoss).toBeCloseTo(expectedLoss, 12);
        });
      });
    });
  });

  describe('binary()', () => {
    describe('given binary targets and thresholded probabilities', () => {
      describe('when the error rate is evaluated', () => {
        it('returns the expected misclassification fraction', () => {
          // Arrange
          const targets = [0, 1, 0, 1];
          const outputs = [0.2, 0.7, 0.6, 0.3];
          const expectedErrorRate = 0.5;

          // Act
          const actualErrorRate = Cost.binary(targets, outputs);

          // Assert
          expect(actualErrorRate).toBe(expectedErrorRate);
        });
      });
    });
  });

  describe('mae()', () => {
    describe('given regression targets and predictions', () => {
      describe('when the loss is evaluated', () => {
        it('returns the expected mean absolute error', () => {
          // Arrange
          const targets = [1, 2, 3];
          const outputs = [1.5, 2.5, 2.5];
          const expectedLoss = (0.5 + 0.5 + 0.5) / 3;

          // Act
          const actualLoss = Cost.mae(targets, outputs);

          // Assert
          expect(actualLoss).toBeCloseTo(expectedLoss, 12);
        });
      });
    });
  });

  describe('mape()', () => {
    describe('given regression targets and predictions', () => {
      describe('when the loss is evaluated', () => {
        it('returns the expected mean absolute percentage error', () => {
          // Arrange
          const targets = [2, 4];
          const outputs = [1, 5];
          const expectedLoss = (0.5 + 0.25) / 2;

          // Act
          const actualLoss = Cost.mape(targets, outputs);

          // Assert
          expect(actualLoss).toBeCloseTo(expectedLoss, 12);
        });
      });
    });
  });

  describe('msle()', () => {
    describe('given non-negative regression targets and predictions', () => {
      describe('when the loss is evaluated', () => {
        it('returns the expected mean squared logarithmic error', () => {
          // Arrange
          const targets = [0, 3];
          const outputs = [1, 1];
          const expectedLoss =
            (Math.log1p(0) - Math.log1p(1)) ** 2 / 2 +
            (Math.log1p(3) - Math.log1p(1)) ** 2 / 2;

          // Act
          const actualLoss = Cost.msle(targets, outputs);

          // Assert
          expect(actualLoss).toBeCloseTo(expectedLoss, 12);
        });
      });
    });
  });

  describe('hinge()', () => {
    describe('given signed targets and raw scores', () => {
      describe('when the loss is evaluated', () => {
        it('returns the expected mean hinge loss', () => {
          // Arrange
          const targets = [1, -1, 1];
          const outputs = [0.8, 0.2, -0.5];
          const expectedLoss = (0.2 + 1.2 + 1.5) / 3;

          // Act
          const actualLoss = Cost.hinge(targets, outputs);

          // Assert
          expect(actualLoss).toBeCloseTo(expectedLoss, 12);
        });
      });
    });
  });

  describe('focalLoss()', () => {
    describe('given the default focal parameters', () => {
      describe('when the loss is evaluated', () => {
        it('applies the default alpha and gamma weights', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [0.9, 0.2];
          const expectedLoss =
            (-0.25 * (1 - 0.9) ** 2 * Math.log(0.9) -
              0.75 * (1 - 0.8) ** 2 * Math.log(0.8)) /
            2;

          // Act
          const actualLoss = Cost.focalLoss(targets, outputs);

          // Assert
          expect(actualLoss).toBeCloseTo(expectedLoss, 12);
        });
      });
    });
  });

  describe('labelSmoothing()', () => {
    describe('given the default smoothing factor', () => {
      describe('when the loss is evaluated', () => {
        it('uses the smoothed targets against the bounded probabilities', () => {
          // Arrange
          const targets = [1, 0];
          const outputs = [0.9, 0.2];
          const positiveTarget = 1 * 0.9 + 0.5 * 0.1;
          const negativeTarget = 0 * 0.9 + 0.5 * 0.1;
          const expectedLoss =
            (-(
              positiveTarget * Math.log(0.9) +
              (1 - positiveTarget) * Math.log(0.1)
            ) -
              (negativeTarget * Math.log(0.2) +
                (1 - negativeTarget) * Math.log(0.8))) /
            2;

          // Act
          const actualLoss = Cost.labelSmoothing(targets, outputs);

          // Assert
          expect(actualLoss).toBeCloseTo(expectedLoss, 12);
        });
      });
    });
  });
});

function captureErrorMessage(runLoss: () => number): string | undefined {
  try {
    runLoss();
  } catch (error: unknown) {
    return error instanceof Error ? error.message : String(error);
  }

  return undefined;
}
