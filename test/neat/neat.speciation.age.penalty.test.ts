import { _speciate } from '../../src/neat/speciation/speciation';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationOptions,
  SpeciesLike,
  SpeciationHarnessContext,
  SpeciesLastStats,
} from '../../src/neat/shared/neat.shared.types';

type AgePenaltyContext = SpeciationHarnessContext<SpeciationOptions>;

const buildAgePenaltyContext = (): AgePenaltyContext => {
  const genome: GenomeDetailed = {
    nodes: [],
    connections: [],
    _id: 1,
    score: 10,
  };
  const species: SpeciesLike = {
    id: 1,
    members: [genome],
    representative: genome,
    lastImproved: 0,
    bestScore: 10,
  };
  return {
    population: [genome],
    _species: [species],
    _nextSpeciesId: 2,
    generation: 31, // > grace*10 when grace=3
    options: {
      speciation: true,
      targetSpecies: 0,
      compatibilityThreshold: 3,
      speciesAgeProtection: { grace: 3, oldPenalty: 0.5 },
    },
    _speciesCreated: new Map<number, number>([[1, 0]]),
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

describe('speciation - age penalty', () => {
  test('applies age penalty after grace * 10 generations', () => {
    // Arrange
    const context = buildAgePenaltyContext();

    // Act: run speciation (age penalty applied in step 6)
    _speciate.call(context);

    // Assert: score halved by penalty
    expect(context._species[0].members[0].score).toBe(5);
  });
});
