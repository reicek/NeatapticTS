import { _speciate } from '../../src/neat/speciation/speciation';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationOptions,
  SpeciationHarnessContext,
  SpeciesLastStats,
} from '../../src/neat/shared/neat.shared.types';

type SpeciationPidOptions = SpeciationOptions & {
  compatibilityThreshold: number;
  compatAdjust: Required<NonNullable<SpeciationOptions['compatAdjust']>>;
};

type SpeciationPidContext = SpeciationHarnessContext<SpeciationPidOptions>;

const buildPidContext = (
  population: GenomeDetailed[],
  overrides: Partial<SpeciationPidContext['options']>,
  compatIntegral: number,
): SpeciationPidContext => {
  const defaultOptions: SpeciationPidContext['options'] = {
    speciation: true,
    targetSpecies: 1,
    compatibilityThreshold: 9.9,
    compatAdjust: {
      kp: 1,
      ki: 0.5,
      smoothingWindow: 1,
      minThreshold: 0.5,
      maxThreshold: 10,
      decay: 1,
    },
  };
  return {
    population,
    _species: [],
    _nextSpeciesId: 1,
    generation: 0,
    options: { ...defaultOptions, ...overrides },
    _speciesCreated: new Map<number, number>(),
    _prevSpeciesMembers: new Map<number, Set<number>>(),
    _speciesLastStats: new Map<number, SpeciesLastStats>(),
    _speciesHistory: [],
    _compatIntegral: compatIntegral,
    _getRNG: () => () => 0.5,
    _compatibilityDistance: (
      genomeA: GenomeDetailed,
      genomeB: GenomeDetailed,
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

describe('speciation - pid controller', () => {
  test('clips compatibility threshold at maximum and resets integral', () => {
    // Arrange
    const genomes: GenomeDetailed[] = [
      { nodes: [], connections: [], _id: 1, score: 1 },
      { nodes: [], connections: [], _id: 2, score: 1 },
    ];
    const ctx = buildPidContext(genomes, {}, -100); // ensure reset when clipping top

    // Act
    _speciate.call(ctx);

    // Assert
    expect(ctx._compatIntegral).toBe(0);
  });
});
