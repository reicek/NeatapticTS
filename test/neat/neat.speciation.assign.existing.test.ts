import { _speciate } from '../../src/neat/speciation/speciation';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationOptions,
  SpeciationHarnessContext,
  SpeciesLastStats,
} from '../../src/neat/shared/neat.shared.types';

type AssignmentOptions = SpeciationOptions & {
  compatibilityThreshold: number;
};

type AssignmentContext = SpeciationHarnessContext<AssignmentOptions>;

const buildAssignmentContext = (): AssignmentContext => {
  const genomeOne: GenomeDetailed = {
    nodes: [],
    connections: [],
    _id: 1,
    score: 1,
  };
  const genomeTwo: GenomeDetailed = {
    nodes: [],
    connections: [],
    _id: 2,
    score: 2,
  };
  return {
    population: [genomeOne, genomeTwo],
    _species: [],
    _nextSpeciesId: 1,
    generation: 0,
    options: {
      speciation: true,
      targetSpecies: 0,
      compatibilityThreshold: 10,
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
      return 0;
    },
    _fallbackInnov: (connection: ConnectionLike) => {
      void connection;
      return 1;
    },
    _structuralEntropy: (genomeDetailed: GenomeDetailed) => {
      void genomeDetailed;
      return 0;
    },
  };
};

describe('speciation - assignment', () => {
  test('assigns second genome to first species when distance below threshold', () => {
    // Arrange
    const context = buildAssignmentContext();

    // Act
    _speciate.call(context);

    // Assert
    expect(context._species.length).toBe(1);
  });
});
