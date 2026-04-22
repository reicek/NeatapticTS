import {
  NetworkTrainingInvalidCostFunctionError,
  NetworkTrainingInvalidOptimizerOptionError,
  NetworkTrainingUnknownLookaheadBaseTypeError,
} from './network.training.errors';

describe('network training errors chapter', () => {
  describe('NetworkTrainingInvalidCostFunctionError', () => {
    describe('given a training run receives an unsupported cost function descriptor', () => {
      it('captures the message, name, and forwarded cause', () => {
        // Arrange
        const cause = new Error('invalid-cost-function');

        // Act
        const error = new NetworkTrainingInvalidCostFunctionError(
          'Invalid cost function',
          { cause },
        );

        // Assert
        expect({
          cause: error.cause,
          message: error.message,
          name: error.name,
        }).toEqual({
          cause,
          message: 'Invalid cost function',
          name: 'NetworkTrainingInvalidCostFunctionError',
        });
      });
    });
  });

  describe('NetworkTrainingInvalidOptimizerOptionError', () => {
    describe('given a non-object optimizer configuration shape was provided', () => {
      it('captures the message, name, and forwarded cause', () => {
        // Arrange
        const cause = new Error('optimizer-shape');

        // Act
        const error = new NetworkTrainingInvalidOptimizerOptionError(
          'Invalid optimizer configuration',
          { cause },
        );

        // Assert
        expect({
          cause: error.cause,
          message: error.message,
          name: error.name,
        }).toEqual({
          cause,
          message: 'Invalid optimizer configuration',
          name: 'NetworkTrainingInvalidOptimizerOptionError',
        });
      });
    });
  });

  describe('NetworkTrainingUnknownLookaheadBaseTypeError', () => {
    describe('given a lookahead wrapper references an unsupported base optimizer', () => {
      it('captures the message, name, and forwarded cause', () => {
        // Arrange
        const cause = new Error('lookahead-base-type');

        // Act
        const error = new NetworkTrainingUnknownLookaheadBaseTypeError(
          'Unknown lookahead base optimizer',
          { cause },
        );

        // Assert
        expect({
          cause: error.cause,
          message: error.message,
          name: error.name,
        }).toEqual({
          cause,
          message: 'Unknown lookahead base optimizer',
          name: 'NetworkTrainingUnknownLookaheadBaseTypeError',
        });
      });
    });
  });
});