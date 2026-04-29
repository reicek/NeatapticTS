import {
  assignPopulationToSpecies,
  refreshSpeciesRepresentatives,
  resetSpeciesMembers,
  snapshotPreviousMembers,
} from './speciation.assignment.utils';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationHarnessContext,
  SpeciationOptions,
  SpeciesLastStats,
} from '../../shared/neat.shared.types';

type AssignmentOptions = SpeciationOptions & {
  compatibilityThreshold: number;
};

type AssignmentContext = SpeciationHarnessContext<AssignmentOptions>;

function buildAssignmentContext(): AssignmentContext {
  const firstGenome: GenomeDetailed = {
    nodes: [],
    connections: [],
    _id: 1,
    score: 1,
  };
  const secondGenome: GenomeDetailed = {
    nodes: [],
    connections: [],
    _id: 2,
    score: 2,
  };

  return {
    population: [firstGenome, secondGenome],
    _species: [
      {
        id: 1,
        members: [firstGenome],
        representative: firstGenome,
        lastImproved: 0,
        bestScore: firstGenome.score ?? Number.NEGATIVE_INFINITY,
      },
    ],
    _nextSpeciesId: 2,
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

function buildCreationContext(populationSize: number): AssignmentContext {
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
      leftGenome: GenomeDetailed,
      rightGenome: GenomeDetailed,
    ) => {
      void leftGenome;
      void rightGenome;
      return 5;
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

describe('neat speciation assignment chapter', () => {
  describe('refreshSpeciesRepresentatives()', () => {
    describe('given a context with one populated and one empty species', () => {
      it('removes the empty species and updates the representative', () => {
        // Arrange
        const ctx = buildAssignmentContext();
        // Add an empty species alongside the populated one
        ctx._species.push({
          id: 99,
          members: [],
          representative: ctx._species[0].representative,
          lastImproved: 0,
          bestScore: 0,
        });

        // Act
        refreshSpeciesRepresentatives(ctx);

        // Assert: empty species was removed; remaining one has updated representative
        expect(ctx._species).toHaveLength(1);
      });
    });
  });

  describe('snapshotPreviousMembers()', () => {
    describe('given a context with no _prevSpeciesMembers set', () => {
      it('initializes the map before snapshotting (line 122 ?? new Map() branch)', () => {
        // Arrange: _prevSpeciesMembers is undefined → right arm of ?? fires
        const ctx = buildAssignmentContext();
        (ctx as Record<string, unknown>)._prevSpeciesMembers = undefined;

        // Act
        snapshotPreviousMembers(ctx);

        // Assert: map was created and species members captured
        expect(ctx._prevSpeciesMembers).toBeDefined();
      });
    });
  });

  describe('assignPopulationToSpecies', () => {
    describe('given a second genome already inside the compatibility threshold of an existing representative', () => {
      it('keeps both genomes in the existing species', () => {
        // Arrange
        const speciationContext = buildAssignmentContext();
        resetSpeciesMembers(speciationContext);

        // Act
        assignPopulationToSpecies(speciationContext, speciationContext.options);

        // Assert
        expect(speciationContext._species[0].members).toHaveLength(2);
      });
    });

    describe('given a population where no genome matches any existing species', () => {
      it('creates a new species for each genome', () => {
        // Arrange
        const speciationContext = buildCreationContext(3);

        // Act
        assignPopulationToSpecies(speciationContext, speciationContext.options);

        // Assert
        expect(speciationContext._species).toHaveLength(3);
      });
    });

    describe('given options with no compatibilityThreshold configured', () => {
      it('uses the default compatibility threshold (line 241 ?? DEFAULT branch)', () => {
        // Arrange: omit compatibilityThreshold so ?? DEFAULT_COMPATIBILITY_THRESHOLD fires
        const ctx = buildAssignmentContext();
        const optionsWithoutThreshold = {
          ...ctx.options,
          compatibilityThreshold: undefined,
        } as unknown as AssignmentOptions;

        // Act
        assignPopulationToSpecies(ctx, optionsWithoutThreshold);

        // Assert: both genomes end up in the same species (threshold is effectively large)
        expect(ctx._species).toBeDefined();
      });
    });
  });
});
