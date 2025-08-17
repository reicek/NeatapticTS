import { _speciate } from '../../src/neat/neat.speciation';

describe('speciation - pid controller', () => {
  test('clips compatibility threshold at maximum and resets integral', () => {
    // Arrange
    const genome: any = { nodes: [], connections: [], _id: 1, score: 1 };
    const ctx: any = {
      population: [genome, { ...genome, _id: 2 }],
      _species: [],
      _nextSpeciesId: 1,
      generation: 0,
      options: {
        speciation: true,
        targetSpecies: 1, // drive large negative error
        compatibilityThreshold: 9.9,
        compatAdjust: {
          kp: 1,
          ki: 0.5,
          smoothingWindow: 1,
          minThreshold: 0.5,
          maxThreshold: 10,
          decay: 1,
        },
      },
      _speciesCreated: new Map(),
      _prevSpeciesMembers: new Map(),
      _speciesLastStats: new Map(),
      _speciesHistory: [],
      _compatIntegral: -100, // ensure reset when clipping top
      _getRNG: () => () => 0.5,
      _compatibilityDistance: () => 0, // one species
      _fallbackInnov: () => 1,
      _structuralEntropy: () => 0,
    };

    // Act
    _speciate.call(ctx);

    // Assert
    expect(ctx._compatIntegral).toBe(0);
  });
});
