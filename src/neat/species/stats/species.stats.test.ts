import { getSpeciesStats } from './species.stats';

describe('species stats chapter', () => {
  describe('getSpeciesStats', () => {
    describe('given the controller has live species records with optional values missing', () => {
      it('projects compact snapshots with zero defaults for absent reporting fields', () => {
        // Arrange
        const speciesHost = {
          _species: [
            {
              id: 7,
              members: [{}, {}],
              bestScore: 9,
              lastImproved: 4,
            },
            {
              id: 8,
              members: undefined,
              bestScore: undefined,
              lastImproved: undefined,
            },
          ],
        };

        // Act
        const speciesStats = getSpeciesStats(speciesHost as never);

        // Assert
        expect(speciesStats).toEqual([
          {
            id: 7,
            size: 2,
            bestScore: 9,
            lastImproved: 4,
          },
          {
            id: 8,
            size: 0,
            bestScore: 0,
            lastImproved: 0,
          },
        ]);
      });
    });

    describe('given the controller has no live species registry yet', () => {
      it('returns an empty reporting array', () => {
        // Arrange
        const speciesHost = {};

        // Act
        const speciesStats = getSpeciesStats(speciesHost as never);

        // Assert
        expect(speciesStats).toEqual([]);
      });
    });
  });
});