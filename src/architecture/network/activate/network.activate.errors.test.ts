import {
  NetworkActivateBatchInputsCollectionError,
  NetworkActivateCorruptedStructureError,
  NetworkActivateInputSizeMismatchError,
} from './network.activate.errors';

describe('network activation chapter', () => {
  describe('activation errors', () => {
    describe('given activation input dimensionality mismatches the network', () => {
      describe('when the input-size error is constructed', () => {
        it('preserves the input-size error metadata', () => {
          // Arrange
          const cause = new Error('input vector too short');

          // Act
          const error = new NetworkActivateInputSizeMismatchError(
            'activation input size mismatch',
            { cause },
          );

          // Assert
          expect({
            cause: error.cause,
            message: error.message,
            name: error.name,
          }).toStrictEqual({
            cause,
            message: 'activation input size mismatch',
            name: 'NetworkActivateInputSizeMismatchError',
          });
        });
      });
    });

    describe('given activation encounters a corrupted node structure', () => {
      describe('when the structure error is constructed', () => {
        it('preserves the corrupted-structure error metadata', () => {
          // Arrange
          const cause = new Error('output node missing');

          // Act
          const error = new NetworkActivateCorruptedStructureError(
            'activation structure is corrupted',
            { cause },
          );

          // Assert
          expect({
            cause: error.cause,
            message: error.message,
            name: error.name,
          }).toStrictEqual({
            cause,
            message: 'activation structure is corrupted',
            name: 'NetworkActivateCorruptedStructureError',
          });
        });
      });
    });

    describe('given batch activation receives a non-array input collection', () => {
      describe('when the collection error is constructed', () => {
        it('preserves the batch-collection error metadata', () => {
          // Arrange
          const cause = new Error('batch input must be an array');

          // Act
          const error = new NetworkActivateBatchInputsCollectionError(
            'activation batch inputs must be an array',
            { cause },
          );

          // Assert
          expect({
            cause: error.cause,
            message: error.message,
            name: error.name,
          }).toStrictEqual({
            cause,
            message: 'activation batch inputs must be an array',
            name: 'NetworkActivateBatchInputsCollectionError',
          });
        });
      });
    });
  });
});
