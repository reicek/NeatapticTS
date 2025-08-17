import { _speciate } from '../../src/neat/neat.speciation';

describe('speciation - age penalty', () => {
  test('applies age penalty after grace * 10 generations', () => {
    // Arrange
    const g: any = { nodes: [], connections: [], _id: 1, score: 10 };
    // Pre-create a species to ensure age calculation uses createdGen=0
    const ctx: any = {
      population: [g],
      _species: [
        {
          id: 1,
          members: [g],
          representative: g,
          lastImproved: 0,
          bestScore: 10,
        },
      ],
      _nextSpeciesId: 2,
      generation: 31, // > grace*10 when grace=3
      options: {
        speciation: true,
        targetSpecies: 0,
        compatibilityThreshold: 3,
        speciesAgeProtection: { grace: 3, oldPenalty: 0.5 },
      },
      _speciesCreated: new Map([[1, 0]]),
      _prevSpeciesMembers: new Map(),
      _speciesLastStats: new Map(),
      _speciesHistory: [],
      _compatIntegral: 0,
      _getRNG: () => () => 0.5,
      _compatibilityDistance: () => 0,
      _fallbackInnov: () => 1,
      _structuralEntropy: () => 0,
    };

    // Act: run speciation (age penalty applied in step 6)
    _speciate.call(ctx);

    // Assert: score halved by penalty
    expect(ctx._species[0].members[0].score).toBe(5);
  });
});
