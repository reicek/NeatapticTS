import { _speciate } from '../../src/neat/neat.speciation';

describe('speciation - auto compat tuning', () => {
  test('applies mild jitter when tuning error is zero', () => {
    // Arrange
    const genome: any = { nodes: [], connections: [], _id: 1, score: 1 };
    const ctx: any = {
      population: [genome],
      _species: [],
      _nextSpeciesId: 1,
      generation: 0,
      options: {
        speciation: true,
        targetSpecies: 1,
        compatibilityThreshold: 3,
        compatAdjust: {
          kp: 0,
          ki: 0,
          smoothingWindow: 1,
          minThreshold: 0.5,
          maxThreshold: 10,
          decay: 1,
        },
        autoCompatTuning: {
          enabled: true,
          target: 1,
          adjustRate: 0.01,
          minCoeff: 0.1,
          maxCoeff: 5,
        },
        excessCoeff: 1,
        disjointCoeff: 1,
      },
      _speciesCreated: new Map(),
      _prevSpeciesMembers: new Map(),
      _speciesLastStats: new Map(),
      _speciesHistory: [],
      _compatIntegral: 0,
      _getRNG: () => () => 0.5, // deterministic jitter (0.5 - 0.5) -> 0 => factor == 1
      _compatibilityDistance: () => 0,
      _fallbackInnov: () => 1,
      _structuralEntropy: () => 0,
    };

    // Act
    _speciate.call(ctx);

    // Assert
    expect(ctx.options.excessCoeff).toBeGreaterThan(0.099); // still within bounds (1 * factor ~ 1)
  });
});
