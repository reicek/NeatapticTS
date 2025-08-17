import { _speciate } from '../../src/neat/neat.speciation';

describe('speciation - assignment', () => {
  test('assigns second genome to first species when distance below threshold', () => {
    // Arrange
    const g1: any = { nodes: [], connections: [], _id: 1, score: 1 };
    const g2: any = { nodes: [], connections: [], _id: 2, score: 2 };
    const ctx: any = {
      population: [g1, g2],
      _species: [],
      _nextSpeciesId: 1,
      generation: 0,
      options: {
        speciation: true,
        targetSpecies: 0,
        compatibilityThreshold: 10,
      },
      _speciesCreated: new Map(),
      _prevSpeciesMembers: new Map(),
      _speciesLastStats: new Map(),
      _speciesHistory: [],
      _compatIntegral: 0,
      _getRNG: () => () => 0.5,
      _compatibilityDistance: (a: any, b: any) => 0, // always same -> below threshold
      _fallbackInnov: () => 1,
      _structuralEntropy: () => 0,
    };

    // Act
    _speciate.call(ctx);

    // Assert
    expect(ctx._species.length).toBe(1);
  });
});
