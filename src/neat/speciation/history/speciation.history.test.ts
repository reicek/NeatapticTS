import {
  applyAgeProtection,
  recordHistory,
  trimHistory,
} from './speciation.history.utils';
import { HISTORY_BUFFER_MAX_ENTRIES } from '../shared/speciation.shared';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationHarnessContext,
  SpeciationOptions,
  SpeciesLastStats,
  SpeciesLike,
} from '../../shared/neat.shared.types';

type AgePenaltyContext = SpeciationHarnessContext<SpeciationOptions>;

function createConnection(input: {
  fromGeneId: number;
  toGeneId: number;
  innovation?: number;
  enabled?: boolean;
}): ConnectionLike {
  return {
    innovation: input.innovation,
    enabled: input.enabled ?? true,
    from: { geneId: input.fromGeneId },
    to: { geneId: input.toGeneId },
  };
}

function createGenome(input: {
  genomeId: number;
  connections?: ConnectionLike[];
  score?: number;
  compatibilityMode?: 'allow-fallback';
}): GenomeDetailed {
  return {
    _id: input.genomeId,
    nodes: [],
    connections: input.connections ?? [],
    score: input.score,
    ...(input.compatibilityMode
      ? { _compatInnovationMode: input.compatibilityMode }
      : {}),
  };
}

function buildAgePenaltyContext(): AgePenaltyContext {
  const genome = createGenome({ genomeId: 1, score: 10 });
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

function buildExtendedHistoryContext(input: {
  members: GenomeDetailed[];
  fallbackInnov?: (connection: ConnectionLike) => number;
}): AgePenaltyContext {
  const species: SpeciesLike = {
    id: 1,
    members: input.members,
    representative: input.members[0],
    lastImproved: 0,
    bestScore: Math.max(
      ...input.members.map((member) => member.score ?? Number.NEGATIVE_INFINITY),
    ),
  };

  return {
    population: input.members,
    _species: [species],
    _nextSpeciesId: 2,
    generation: 7,
    options: {
      speciation: true,
      targetSpecies: 0,
      compatibilityThreshold: 3,
      speciesAllocation: { extendedHistory: true },
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
    _fallbackInnov:
      input.fallbackInnov ??
      ((connection: ConnectionLike) =>
        (((connection.from as { geneId?: number }).geneId ?? 0) * 100) +
        ((connection.to as { geneId?: number }).geneId ?? 0)),
    _structuralEntropy: (genomeDetailed: GenomeDetailed) => {
      void genomeDetailed;
      return 0;
    },
  };
}

function readLatestHistoryStat(
  speciationContext: AgePenaltyContext,
): Record<string, unknown> {
  recordHistory(speciationContext, speciationContext.options);
  return (
    speciationContext._speciesHistory[0] as {
      stats: Array<Record<string, unknown>>;
    }
  ).stats[0];
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

  describe('recordHistory', () => {
    describe('given identical native members with explicit innovations', () => {
      it('records zero innovation range for the species row', () => {
        // Arrange
        const speciationContext = buildExtendedHistoryContext({
          members: [
            createGenome({
              genomeId: 1,
              connections: [
                createConnection({ fromGeneId: 1, toGeneId: 2, innovation: 10 }),
              ],
            }),
            createGenome({
              genomeId: 2,
              connections: [
                createConnection({ fromGeneId: 1, toGeneId: 2, innovation: 10 }),
              ],
            }),
          ],
        });

        // Act
        const historyStat = readLatestHistoryStat(speciationContext);

        // Assert
        expect(historyStat.innovationRange).toBe(0);
      });
    });

    describe('given native members with a small structural innovation drift', () => {
      it('records the explicit small innovation span', () => {
        // Arrange
        const speciationContext = buildExtendedHistoryContext({
          members: [
            createGenome({
              genomeId: 1,
              connections: [
                createConnection({ fromGeneId: 1, toGeneId: 2, innovation: 10 }),
              ],
            }),
            createGenome({
              genomeId: 2,
              connections: [
                createConnection({ fromGeneId: 1, toGeneId: 2, innovation: 11 }),
              ],
            }),
          ],
        });

        // Act
        const historyStat = readLatestHistoryStat(speciationContext);

        // Assert
        expect(historyStat.innovationRange).toBe(1);
      });
    });

    describe('given native members with a larger topology divergence', () => {
      it('records the wider explicit innovation span', () => {
        // Arrange
        const speciationContext = buildExtendedHistoryContext({
          members: [
            createGenome({
              genomeId: 1,
              connections: [
                createConnection({ fromGeneId: 1, toGeneId: 2, innovation: 10 }),
              ],
            }),
            createGenome({
              genomeId: 2,
              connections: [
                createConnection({ fromGeneId: 1, toGeneId: 3, innovation: 40 }),
              ],
            }),
          ],
        });

        // Act
        const historyStat = readLatestHistoryStat(speciationContext);

        // Assert
        expect(historyStat.innovationRange).toBe(30);
      });
    });

    describe('given native members with explicit enabled and disabled innovations', () => {
      it('records the enabled ratio from explicit history only', () => {
        // Arrange
        const speciationContext = buildExtendedHistoryContext({
          members: [
            createGenome({
              genomeId: 1,
              connections: [
                createConnection({
                  fromGeneId: 1,
                  toGeneId: 2,
                  innovation: 10,
                  enabled: true,
                }),
                createConnection({
                  fromGeneId: 2,
                  toGeneId: 3,
                  innovation: 11,
                  enabled: false,
                }),
              ],
            }),
            createGenome({
              genomeId: 2,
              connections: [
                createConnection({
                  fromGeneId: 1,
                  toGeneId: 3,
                  innovation: 12,
                  enabled: true,
                }),
              ],
            }),
          ],
        });

        // Act
        const historyStat = readLatestHistoryStat(speciationContext);

        // Assert
        expect(historyStat.enabledRatio).toBeCloseTo(2 / 3);
      });
    });

    describe('given a native member missing its explicit innovation id', () => {
      it('throws instead of silently summarizing synthetic history ids', () => {
        // Arrange
        const speciationContext = buildExtendedHistoryContext({
          members: [
            createGenome({
              genomeId: 1,
              connections: [createConnection({ fromGeneId: 1, toGeneId: 2 })],
            }),
          ],
        });

        // Assert
        expect(() => recordHistory(speciationContext, speciationContext.options)).toThrow(
          /Species history requires explicit connection innovations/,
        );
      });
    });

    describe('given legacy or partial members that deliberately allow fallback innovations', () => {
      it('records the fallback-derived innovation span for the species row', () => {
        // Arrange
        const speciationContext = buildExtendedHistoryContext({
          members: [
            createGenome({
              genomeId: 1,
              compatibilityMode: 'allow-fallback',
              connections: [createConnection({ fromGeneId: 1, toGeneId: 2 })],
            }),
            createGenome({
              genomeId: 2,
              compatibilityMode: 'allow-fallback',
              connections: [createConnection({ fromGeneId: 2, toGeneId: 3 })],
            }),
          ],
        });

        // Act
        const historyStat = readLatestHistoryStat(speciationContext);

        // Assert
        expect(historyStat.innovationRange).toBe(101);
      });
    });
  });

  describe('trimHistory', () => {
    describe('given the history buffer exceeds the configured cap', () => {
      it('removes the oldest history entry', () => {
        // Arrange
        const speciationContext = buildAgePenaltyContext();
        speciationContext._speciesHistory = Array.from(
          { length: HISTORY_BUFFER_MAX_ENTRIES + 1 },
          (_, index) => ({
            generation: index,
            stats: [],
          }),
        );

        // Act
        trimHistory(speciationContext);

        // Assert
        expect(speciationContext._speciesHistory[0]?.generation).toBe(1);
      });
    });
  });
});
