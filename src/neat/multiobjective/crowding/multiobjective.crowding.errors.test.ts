import { MultiobjectiveCrowdingGenomeIndexResolutionError } from './multiobjective.crowding.errors';

describe('multiobjective crowding errors chapter', () => {
  describe('MultiobjectiveCrowdingGenomeIndexResolutionError', () => {
    describe('given crowding helpers cannot resolve a genome to its source index', () => {
      it('preserves the configured message, name, and cause', () => {
        // Arrange
        const cause = new Error('missing genome reference');

        // Act
        const error = new MultiobjectiveCrowdingGenomeIndexResolutionError(
          'failed to resolve genome index for crowding',
          { cause },
        );

        // Assert
        expect({
          message: error.message,
          name: error.name,
          cause: error.cause,
        }).toEqual({
          message: 'failed to resolve genome index for crowding',
          name: 'MultiobjectiveCrowdingGenomeIndexResolutionError',
          cause,
        });
      });
    });
  });
});
