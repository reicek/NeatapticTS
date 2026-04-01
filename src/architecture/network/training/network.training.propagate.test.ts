import Network from '../network';
import { NetworkTrainingOutputTargetLengthError } from './network.training.errors';

function createPropagationNetwork(seed: number): Network {
  return new Network(2, 1, { seed });
}

describe('network training chapter', () => {
  describe('propagation target validation', () => {
    describe('given the target vector has too many values', () => {
      describe('when propagate() is called', () => {
        it('throws the output-target-length error', () => {
          // Arrange
          const network = createPropagationNetwork(230);
          network.activate([0, 1]);

          // Act
          const propagateWithTooManyTargets = () => {
            network.propagate(0.1, 0, true, [1, 2]);
          };

          // Assert
          expect(propagateWithTooManyTargets).toThrow(
            NetworkTrainingOutputTargetLengthError,
          );
        });
      });
    });

    describe('given the target vector has too few values', () => {
      describe('when propagate() is called', () => {
        it('throws the output-target-length error', () => {
          // Arrange
          const network = createPropagationNetwork(231);
          network.activate([0, 1]);

          // Act
          const propagateWithTooFewTargets = () => {
            network.propagate(0.1, 0, true, []);
          };

          // Assert
          expect(propagateWithTooFewTargets).toThrow(
            NetworkTrainingOutputTargetLengthError,
          );
        });
      });
    });

    describe('given the target vector matches the output width', () => {
      describe('when propagate() is called', () => {
        it('does not throw', () => {
          // Arrange
          const network = createPropagationNetwork(232);
          network.activate([0, 1]);

          // Act
          const propagateWithValidTarget = () => {
            network.propagate(0.1, 0, true, [1]);
          };

          // Assert
          expect(propagateWithValidTarget).not.toThrow();
        });
      });
    });
  });
});
