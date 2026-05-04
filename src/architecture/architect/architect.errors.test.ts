import {
  ArchitectInputOutputTypeResolutionError,
  ArchitectInvalidGruConfigurationError,
  ArchitectInvalidGruLayerArgumentsError,
  ArchitectInvalidLstmConfigurationError,
  ArchitectInvalidLstmLayerArgumentsError,
  ArchitectInvalidPerceptronConfigurationError,
  ArchitectZeroInputOutputNodesError,
} from './architect.errors';

describe('architect errors chapter', () => {
  describe('ArchitectInputOutputTypeResolutionError', () => {
    describe('given construction cannot resolve input and output node roles', () => {
      it('preserves the configured message, name, and cause', () => {
        // Arrange
        const cause = new Error('missing node roles');

        // Act
        const error = new ArchitectInputOutputTypeResolutionError(
          'failed to resolve input/output types',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'failed to resolve input/output types',
          name: 'ArchitectInputOutputTypeResolutionError',
          cause,
        });
      });
    });
  });

  describe('ArchitectZeroInputOutputNodesError', () => {
    describe('given construction produces no interface nodes', () => {
      it('preserves the configured message, name, and cause', () => {
        // Arrange
        const cause = new Error('empty interface');

        // Act
        const error = new ArchitectZeroInputOutputNodesError(
          'constructed network has zero interface nodes',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'constructed network has zero interface nodes',
          name: 'ArchitectZeroInputOutputNodesError',
          cause,
        });
      });
    });
  });

  describe('ArchitectInvalidPerceptronConfigurationError', () => {
    describe('given the perceptron builder receives too few layer sizes', () => {
      it('preserves the configured message and name', () => {
        // Arrange
        const cause = new Error('too few layers');

        // Act
        const error = new ArchitectInvalidPerceptronConfigurationError(
          'perceptron requires at least input and output sizes',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'perceptron requires at least input and output sizes',
          name: 'ArchitectInvalidPerceptronConfigurationError',
          cause,
        });
      });
    });
  });

  describe('ArchitectInvalidLstmLayerArgumentsError', () => {
    describe('given the LSTM builder receives invalid layer arguments', () => {
      it('preserves the configured message and name', () => {
        // Arrange
        const cause = new Error('invalid layer argument');

        // Act
        const error = new ArchitectInvalidLstmLayerArgumentsError(
          'lstm layer sizes must be positive integers',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'lstm layer sizes must be positive integers',
          name: 'ArchitectInvalidLstmLayerArgumentsError',
          cause,
        });
      });
    });
  });

  describe('ArchitectInvalidLstmConfigurationError', () => {
    describe('given the LSTM builder receives too few layer sizes', () => {
      it('preserves the configured message and name', () => {
        // Arrange
        const cause = new Error('missing output layer');

        // Act
        const error = new ArchitectInvalidLstmConfigurationError(
          'lstm requires at least input and output sizes',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'lstm requires at least input and output sizes',
          name: 'ArchitectInvalidLstmConfigurationError',
          cause,
        });
      });
    });
  });

  describe('ArchitectInvalidGruLayerArgumentsError', () => {
    describe('given the GRU builder receives invalid layer arguments', () => {
      it('preserves the configured message and name', () => {
        // Arrange
        const cause = new Error('invalid GRU layer argument');

        // Act
        const error = new ArchitectInvalidGruLayerArgumentsError(
          'gru layer sizes must be positive integers',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'gru layer sizes must be positive integers',
          name: 'ArchitectInvalidGruLayerArgumentsError',
          cause,
        });
      });
    });
  });

  describe('ArchitectInvalidGruConfigurationError', () => {
    describe('given the GRU builder receives too few layer sizes', () => {
      it('preserves the configured message and name', () => {
        // Arrange
        const cause = new Error('missing output layer');

        // Act
        const error = new ArchitectInvalidGruConfigurationError(
          'gru requires at least input and output sizes',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'gru requires at least input and output sizes',
          name: 'ArchitectInvalidGruConfigurationError',
          cause,
        });
      });
    });
  });
});
