import {
  NeatGenomeConversionError,
  NeatGenomeValidationError,
} from './genome.errors';
import type { NeatGenomeValidationIssue } from './genome.types';

describe('neat genome errors chapter', () => {
  describe('NeatGenomeConversionError', () => {
    describe('given malformed runtime state is projected into the genome contract', () => {
      it('preserves the configured message, name, and cause', () => {
        // Arrange
        const cause = { reason: 'invalid runtime payload' };

        // Act
        const error = new NeatGenomeConversionError(
          'failed to convert runtime network into genome',
          cause,
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'failed to convert runtime network into genome',
          name: 'NeatGenomeConversionError',
          cause,
        });
      });

      it('keeps the cause undefined when no cause is provided', () => {
        // Act
        const error = new NeatGenomeConversionError(
          'failed to convert runtime network into genome',
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'failed to convert runtime network into genome',
          name: 'NeatGenomeConversionError',
          cause: undefined,
        });
      });
    });
  });

  describe('NeatGenomeValidationError', () => {
    describe('given a strict genome contract fails validation', () => {
      it('preserves the configured message, name, and structured issues', () => {
        // Arrange
        const issues: NeatGenomeValidationIssue[] = [
          {
            code: 'duplicate-node-gene-id',
            message: 'node gene ids must be unique',
            path: 'nodeGenes[2].geneId',
          },
        ];

        // Act
        const error = new NeatGenomeValidationError(
          'strict genome validation failed',
          issues,
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          issues: error.issues,
        }).toEqual({
          message: 'strict genome validation failed',
          name: 'NeatGenomeValidationError',
          issues,
        });
      });
    });
  });
});