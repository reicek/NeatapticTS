import { applyAgeProtection } from './speciation.history.utils';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationHarnessContext,
  SpeciationOptions,
  SpeciesLastStats,
  SpeciesLike,
} from '../../shared/neat.shared.types';

type AgePenaltyContext = SpeciationHarnessContext<SpeciationOptions>;

function buildAgePenaltyContext(): AgePenaltyContext {
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
    generation: 31,
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
      leftGenome: GenomeDetailed,
      rightGenome: GenomeDetailed,
    ) => {
      void leftGenome;
      void rightGenome;
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
}

describe('neat speciation history chapter', () => {
  describe('applyAgeProtection', () => {
    describe('given a species older than the configured grace window', () => {
      it('applies the configured old-species score penalty', () => {
        // Arrange
        const speciationContext = buildAgePenaltyContext();

        // Act
        applyAgeProtection(speciationContext, speciationContext.options);

        // Assert
        expect(speciationContext._species[0].members[0].score).toBe(5);
      });
    });
  });
});
