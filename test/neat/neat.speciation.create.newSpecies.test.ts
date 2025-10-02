import { _speciate } from '../../src/neat/neat.speciation';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationOptions,
  SpeciesLike,
} from '../../src/neat/neat.types';

type SpeciationTestContext = {
  population: GenomeDetailed[];
  _species: SpeciesLike[];
  _nextSpeciesId: number;
  generation: number;
  options: SpeciationOptions & { compatibilityThreshold: number };
  _speciesCreated: Map<number, number>;
  _prevSpeciesMembers: Map<number, Set<number>>;
  _speciesLastStats: Map<
    number,
    { meanNodes: number; meanConns: number; best: number }
  >;
  _speciesHistory: Array<Record<string, unknown>>;
  _compatIntegral: number;
  _getRNG: () => () => number;
  _compatibilityDistance: (
    genomeA: GenomeDetailed,
    genomeB: GenomeDetailed,
  ) => number;
  _fallbackInnov: (connection: ConnectionLike) => number;
  _structuralEntropy: (genome: GenomeDetailed) => number;
};

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
    _speciesLastStats: new Map<
      number,
      { meanNodes: number; meanConns: number; best: number }
    >(),
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
    _fallbackInnov: (_connection: ConnectionLike) => 1,
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
