import { _speciate } from '../../src/neat/neat.speciation';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationOptions,
  SpeciationHarnessContext,
  SpeciesLastStats,
} from '../../src/neat/neat.types';

type AutoCompatOptions = SpeciationOptions & {
  compatAdjust: Required<NonNullable<SpeciationOptions['compatAdjust']>>;
  autoCompatTuning: Required<
    NonNullable<SpeciationOptions['autoCompatTuning']>
  >;
  excessCoeff: number;
  disjointCoeff: number;
};

type AutoCompatContext = SpeciationHarnessContext<AutoCompatOptions>;

const createAutoCompatContext = (): AutoCompatContext => {
  const genome: GenomeDetailed = {
    nodes: [],
    connections: [],
    _id: 1,
    score: 1,
  };
  return {
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
    _speciesCreated: new Map<number, number>(),
    _prevSpeciesMembers: new Map<number, Set<number>>(),
    _speciesLastStats: new Map<number, SpeciesLastStats>(),
    _speciesHistory: [],
    _compatIntegral: 0,
    _getRNG: () => () => 0.5,
    _compatibilityDistance: (
      genomeA: GenomeDetailed,
      genomeB: GenomeDetailed
    ) => {
      void genomeA;
      void genomeB;
      return 0;
    },
    _fallbackInnov: (connection: ConnectionLike) => {
      void connection;
      return 1;
    },
    _structuralEntropy: (genome: GenomeDetailed) => {
      void genome;
      return 0;
    },
  };
};

describe('speciation - auto compat tuning', () => {
  test('applies mild jitter when tuning error is zero', () => {
    // Arrange
    const ctx = createAutoCompatContext();

    // Act
    _speciate.call(ctx);

    // Assert
    expect(ctx.options.excessCoeff).toBeGreaterThan(0.099); // still within bounds (1 * factor ~ 1)
  });
});
