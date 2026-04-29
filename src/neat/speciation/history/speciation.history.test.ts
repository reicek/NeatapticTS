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
      ...input.members.map(
        (member) => member.score ?? Number.NEGATIVE_INFINITY,
      ),
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
        ((connection.from as { geneId?: number }).geneId ?? 0) * 100 +
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

    describe('given options with no speciesAgeProtection configured', () => {
      it('uses the default grace and penalty object (line 117 ?? fallback)', () => {
        // Arrange: speciesAgeProtection absent → ?? { grace, oldPenalty } fires
        const ctx = buildAgePenaltyContext();
        ctx.options.speciesAgeProtection = undefined;
        const scoreBefore = (ctx._species[0].members[0] as GenomeDetailed)
          .score;

        // Act
        applyAgeProtection(ctx, ctx.options);

        // Assert: default penalty (0.5) applied → score halved
        expect((ctx._species[0].members[0] as GenomeDetailed).score).toBe(
          (scoreBefore ?? 0) * 0.5,
        );
      });
    });

    describe('given a species not found in the _speciesCreated map', () => {
      it('falls back to the current generation as createdGeneration (line 124 ?? fallback)', () => {
        // Arrange: species.id not in map → createdGeneration = generation → age = 0 → skip penalty
        const ctx = buildAgePenaltyContext();
        ctx._speciesCreated = new Map(); // empty — species not found
        const scoreBefore = (ctx._species[0].members[0] as GenomeDetailed)
          .score;

        // Act
        applyAgeProtection(ctx, ctx.options);

        // Assert: age = 0 → within grace → no penalty applied
        expect((ctx._species[0].members[0] as GenomeDetailed).score).toBe(
          scoreBefore,
        );
      });
    });

    describe('given age protection with grace undefined in the protection object', () => {
      it('uses DEFAULT_SPECIES_AGE_GRACE as fallback (line 128 ?? fallback)', () => {
        // Arrange: ageProtection object has no grace → ?? DEFAULT_SPECIES_AGE_GRACE
        const ctx = buildAgePenaltyContext();
        ctx.options.speciesAgeProtection = {
          oldPenalty: 0.5,
        } as typeof ctx.options.speciesAgeProtection;

        // Act
        applyAgeProtection(ctx, ctx.options);

        // Assert: penalty still applied (default grace = 3, age = 31 > 30)
        expect((ctx._species[0].members[0] as GenomeDetailed).score).toBe(5);
      });
    });

    describe('given age protection with oldPenalty undefined', () => {
      it('uses DEFAULT_SPECIES_OLD_PENALTY as fallback (line 131 ?? fallback)', () => {
        // Arrange: oldPenalty absent → ?? DEFAULT_SPECIES_OLD_PENALTY (0.5)
        const ctx = buildAgePenaltyContext();
        ctx.options.speciesAgeProtection = {
          grace: 3,
        } as typeof ctx.options.speciesAgeProtection;

        // Act
        applyAgeProtection(ctx, ctx.options);

        // Assert: default penalty 0.5 applied → 10 * 0.5 = 5
        expect((ctx._species[0].members[0] as GenomeDetailed).score).toBe(5);
      });
    });

    describe('given a penalty of exactly 1 (no-effect threshold)', () => {
      it('skips score scaling for the species (line 132 continue arm)', () => {
        // Arrange: oldPenalty = 1 >= PENALTY_NO_EFFECT_THRESHOLD → continue fires
        const ctx = buildAgePenaltyContext();
        ctx.options.speciesAgeProtection = { grace: 3, oldPenalty: 1 };
        const scoreBefore = (ctx._species[0].members[0] as GenomeDetailed)
          .score;

        // Act
        applyAgeProtection(ctx, ctx.options);

        // Assert: score unchanged (penalty >= threshold → skip)
        expect((ctx._species[0].members[0] as GenomeDetailed).score).toBe(
          scoreBefore,
        );
      });
    });

    describe('given a member with a non-number score', () => {
      it('skips that member when applying the penalty (line 134 false arm)', () => {
        // Arrange: member.score = undefined → typeof check fails → no multiply
        const ctx = buildAgePenaltyContext();
        (ctx._species[0].members[0] as GenomeDetailed).score = undefined;

        // Act
        applyAgeProtection(ctx, ctx.options);

        // Assert: score remains undefined (not multiplied)
        expect(
          (ctx._species[0].members[0] as GenomeDetailed).score,
        ).toBeUndefined();
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
                createConnection({
                  fromGeneId: 1,
                  toGeneId: 2,
                  innovation: 10,
                }),
              ],
            }),
            createGenome({
              genomeId: 2,
              connections: [
                createConnection({
                  fromGeneId: 1,
                  toGeneId: 2,
                  innovation: 10,
                }),
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
                createConnection({
                  fromGeneId: 1,
                  toGeneId: 2,
                  innovation: 10,
                }),
              ],
            }),
            createGenome({
              genomeId: 2,
              connections: [
                createConnection({
                  fromGeneId: 1,
                  toGeneId: 2,
                  innovation: 11,
                }),
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
                createConnection({
                  fromGeneId: 1,
                  toGeneId: 2,
                  innovation: 10,
                }),
              ],
            }),
            createGenome({
              genomeId: 2,
              connections: [
                createConnection({
                  fromGeneId: 1,
                  toGeneId: 3,
                  innovation: 40,
                }),
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

    describe('given a species with no members in extended history mode', () => {
      it('records zero averages for an empty member list (line 325 averageNumbers fallback)', () => {
        // Arrange: species has no members → averageNumbers([]) fires → DEFAULT_SCORE_FALLBACK
        const ctx = buildExtendedHistoryContext({ members: [] });
        ctx._species[0].members = [];

        // Act
        recordHistory(ctx, ctx.options);

        // Assert: meanNodes = 0 (fallback), history entry added
        const stat = (
          ctx._speciesHistory[0] as { stats: Array<Record<string, unknown>> }
        ).stats[0];
        expect(stat.meanNodes).toBe(0);
      });
    });

    describe('given a member with no connections in extended history mode', () => {
      it('uses the zero-connection fallback for enabledRatio (line 548 fallback)', () => {
        // Arrange: member has no connections → enabledTotal = 0 → DEFAULT_SCORE_FALLBACK
        const ctx = buildExtendedHistoryContext({
          members: [createGenome({ genomeId: 1, connections: [] })],
        });

        // Act
        const historyStat = readLatestHistoryStat(ctx);

        // Assert: enabledRatio = 0 (fallback for zero-connection member)
        expect(historyStat.enabledRatio).toBe(0);
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
        expect(() =>
          recordHistory(speciationContext, speciationContext.options),
        ).toThrow(/Species history requires explicit connection innovations/);
      });
    });

    describe('given a native member without _id and connection without geneIds', () => {
      it('records null in the error cause for missing genome and endpoint ids (lines 505-509 ?? null)', () => {
        // Arrange: no _id, no geneId on from/to → ?? null fires for all three cause fields
        const ctx = buildExtendedHistoryContext({
          members: [
            {
              nodes: [],
              connections: [
                { from: {}, to: {}, innovation: undefined, enabled: true },
              ] as unknown as ConnectionLike[],
              // _id intentionally absent → member._id ?? null → null
            } as unknown as GenomeDetailed,
          ],
        });

        // Act + Assert: error thrown with null cause fields (not a different error type)
        expect(() => recordHistory(ctx, ctx.options)).toThrow(
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

    describe('given compact history mode (no extendedHistory flag)', () => {
      it('records compact id, size, and best score row (lines 204-206 compact path)', () => {
        // Arrange: options without extendedHistory → compact path fires
        const genome = createGenome({ genomeId: 1, score: 7 });
        const species: SpeciesLike = {
          id: 42,
          members: [genome],
          representative: genome,
          lastImproved: 0,
          bestScore: 7,
        };
        const ctx: AgePenaltyContext = {
          ...buildAgePenaltyContext(),
          _species: [species],
          options: {
            speciation: true,
            targetSpecies: 0,
            compatibilityThreshold: 3,
          },
        };
        ctx._speciesHistory = [];

        // Act
        recordHistory(ctx, ctx.options);

        // Assert: compact row has id, size, best — no meanNodes etc.
        const stat = (
          ctx._speciesHistory[0] as { stats: Array<Record<string, unknown>> }
        ).stats[0];
        expect(stat.id).toBe(42);
      });
    });
  });

  describe('trimHistory', () => {
    describe('given a history buffer within the configured cap', () => {
      it('does not shift any entries (line 232 false arm of trim guard)', () => {
        // Arrange: history length <= max → shift never fires
        const ctx = buildAgePenaltyContext();
        ctx._speciesHistory = [{ generation: 0, stats: [] }];
        const lengthBefore = ctx._speciesHistory.length;

        // Act
        trimHistory(ctx);

        // Assert: history unchanged
        expect(ctx._speciesHistory.length).toBe(lengthBefore);
      });
    });

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
