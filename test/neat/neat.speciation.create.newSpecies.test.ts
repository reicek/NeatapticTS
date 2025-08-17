import { _speciate } from '../../src/neat/neat.speciation';

// Single expectation test: creates new species for each genome when all distances exceed threshold

describe('speciation - creation', () => {
  test('creates a new species per genome when none are compatible', () => {
    // Arrange
    const genomes = Array.from({ length: 3 }, (_, i) => ({
      nodes: [],
      connections: [],
      _id: i + 1,
    }));
    const ctx: any = {
      population: genomes,
      _species: [],
      _nextSpeciesId: 1,
      generation: 0,
      options: {
        speciation: true,
        targetSpecies: 0,
        compatibilityThreshold: 1,
      },
      _speciesCreated: new Map(),
      _prevSpeciesMembers: new Map(),
      _speciesLastStats: new Map(),
      _speciesHistory: [],
      _compatIntegral: 0,
      _getRNG: () => () => 0.5,
      _compatibilityDistance: () => 5, // always above threshold -> force new species
      _fallbackInnov: () => 1,
      _structuralEntropy: () => 0,
    };

    // Act
    _speciate.call(ctx);

    // Assert
    expect(ctx._species.length).toBe(3);
  });
});
