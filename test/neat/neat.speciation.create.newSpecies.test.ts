import { _speciate } from '../../src/neat/speciation/speciation';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationOptions,
  SpeciationHarnessContext,
  SpeciesLastStats,
} from '../../src/neat/shared/neat.shared.types';

type SpeciationCreationOptions = SpeciationOptions & {
  compatibilityThreshold: number;
};

type SpeciationTestContext =
  SpeciationHarnessContext<SpeciationCreationOptions>;

// Single expectation test: creates new species for each genome when all distances exceed threshold

const buildContext = (populationSize: number): SpeciationTestContext => {
  const genomes: GenomeDetailed[] = Array.from(
    { length: populationSize },
    (_, index) => ({
      nodes: [],
      connections: [],
      _id: index + 1,
    }),
  );
  return {
    population: genomes,
    _species: [],
    _nextSpeciesId: 1,
    generation: 0,
    options: {
      speciation: true,
      targetSpecies: 0,
      compatibilityThreshold: 1,
    },
    _speciesCreated: new Map<number, number>(),
    _prevSpeciesMembers: new Map<number, Set<number>>(),
    _speciesLastStats: new Map<number, SpeciesLastStats>(),
    _speciesHistory: [],
    _compatIntegral: 0,
    _getRNG: () => () => 0.5,
    _compatibilityDistance: (
      genomeA: GenomeDetailed,
      genomeB: GenomeDetailed,
    ) => {
      void genomeA;
      void genomeB;
      return 5;
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

describe('speciation - creation', () => {
  test('creates a new species per genome when none are compatible', () => {
    // Arrange
    const ctx = buildContext(3);

    // Act
    _speciate.call(ctx);

    // Assert
    expect(ctx._species.length).toBe(3);
  });
});
