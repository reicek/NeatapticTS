import { getSpeciesHistory } from './species.history.read';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciesHistoryEntry,
  SpeciesHistoryStatExtended,
  SpeciesLike,
} from '../../../shared/neat.shared.types';

type SpeciesHistoryReadHost = {
  options: { speciesAllocation: { extendedHistory: boolean } };
  _speciesHistory: SpeciesHistoryEntry[];
  _species: SpeciesLike[];
  _fallbackInnov?: (connection: ConnectionLike) => number;
};

function createGenomeMember(
  genomeId: number,
  connections: ConnectionLike[],
  compatibilityMode?: 'allow-fallback',
): GenomeDetailed {
  return {
    _id: genomeId,
    nodes: [],
    connections,
    ...(compatibilityMode ? { _compatInnovationMode: compatibilityMode } : {}),
  };
}

function createSpeciesHistoryHost(input?: {
  members?: GenomeDetailed[];
  historyStat?:
    SpeciesHistoryEntry['stats'][number] | SpeciesHistoryStatExtended;
  fallbackInnov?: (connection: ConnectionLike) => number;
}): SpeciesHistoryReadHost {
  const members = input?.members ?? [
    createGenomeMember(1, [
      {
        innovation: 2,
        enabled: true,
        from: { geneId: 1 },
        to: { geneId: 2 },
      },
      {
        innovation: 5,
        enabled: false,
        from: { geneId: 2 },
        to: { geneId: 3 },
      },
    ]),
    createGenomeMember(2, [
      {
        innovation: 7,
        enabled: true,
        from: { geneId: 1 },
        to: { geneId: 3 },
      },
    ]),
  ];

  const liveSpecies: SpeciesLike[] = [
    {
      id: 1,
      members,
      bestScore: 1,
      lastImproved: 0,
    },
  ];

  return {
    options: { speciesAllocation: { extendedHistory: true } },
    _speciesHistory: [
      {
        generation: 0,
        stats: [
          input?.historyStat ?? {
            id: 1,
            size: members.length,
            bestScore: 1,
            lastImproved: 0,
          },
        ],
      },
    ],
    _species: liveSpecies,
    _fallbackInnov: input?.fallbackInnov,
  };
}

describe('neat species history read chapter', () => {
  describe('getSpeciesHistory', () => {
    describe('given extended history is enabled and the recorded row lacks extended fields', () => {
      let historyEntries: SpeciesHistoryEntry[];

      beforeAll(() => {
        // Arrange
        const speciesHistoryHost = createSpeciesHistoryHost();

        // Act
        historyEntries = getSpeciesHistory(speciesHistoryHost);
      });

      describe('when innovation coverage is backfilled from live species members', () => {
        it('adds the current innovation range to the history row', () => {
          // Assert
          expect(
            (historyEntries[0].stats[0] as SpeciesHistoryStatExtended)
              .innovationRange,
          ).toBe(5);
        });
      });

      describe('when enabled connection coverage is backfilled from live species members', () => {
        it('adds the current enabled ratio to the history row', () => {
          // Assert
          expect(
            (historyEntries[0].stats[0] as SpeciesHistoryStatExtended)
              .enabledRatio,
          ).toBeCloseTo(2 / 3);
        });
      });
    });

    describe('given a recorded row that already has extended fields', () => {
      let historyEntries: SpeciesHistoryEntry[];

      beforeAll(() => {
        // Arrange
        const speciesHistoryHost = createSpeciesHistoryHost({
          historyStat: {
            id: 1,
            size: 2,
            bestScore: 1,
            lastImproved: 0,
            innovationRange: 123,
            enabledRatio: 0.5,
          },
        });

        // Act
        historyEntries = getSpeciesHistory(speciesHistoryHost);
      });

      describe('when the read path sees the extended metrics are already present', () => {
        it('preserves the existing innovation range instead of recomputing it', () => {
          // Assert
          expect(
            (historyEntries[0].stats[0] as SpeciesHistoryStatExtended)
              .innovationRange,
          ).toBe(123);
        });
      });
    });

    describe('given extended history needs backfill and native live species connections lack direct innovation ids', () => {
      it('throws instead of silently backfilling synthetic innovation ids', () => {
        // Arrange
        const speciesHistoryHost = createSpeciesHistoryHost({
          members: [
            createGenomeMember(1, [
              {
                enabled: true,
                from: { geneId: 1 },
                to: { geneId: 2 },
              },
            ]),
          ],
          fallbackInnov(connection) {
            return (
              ((connection.from as { geneId?: number }).geneId ?? 0) * 100 +
              ((connection.to as { geneId?: number }).geneId ?? 0)
            );
          },
        });

        // Assert
        expect(() => getSpeciesHistory(speciesHistoryHost)).toThrow(
          /Species history backfill requires explicit connection innovations/,
        );
      });
    });

    describe('given extended history needs backfill and legacy live species connections deliberately allow fallback ids', () => {
      let historyEntries: SpeciesHistoryEntry[];

      beforeAll(() => {
        // Arrange
        const speciesHistoryHost = createSpeciesHistoryHost({
          members: [
            createGenomeMember(
              1,
              [
                {
                  enabled: true,
                  from: { geneId: 1 },
                  to: { geneId: 2 },
                },
                {
                  enabled: false,
                  from: { geneId: 2 },
                  to: { geneId: 3 },
                },
              ],
              'allow-fallback',
            ),
            createGenomeMember(
              2,
              [
                {
                  enabled: true,
                  from: { geneId: 1 },
                  to: { geneId: 3 },
                },
              ],
              'allow-fallback',
            ),
          ],
          fallbackInnov(connection) {
            return (
              ((connection.from as { geneId?: number }).geneId ?? 0) * 100 +
              ((connection.to as { geneId?: number }).geneId ?? 0)
            );
          },
        });

        // Act
        historyEntries = getSpeciesHistory(speciesHistoryHost);
      });

      describe('when the read path summarizes legacy connections through the fallback resolver', () => {
        it('uses the fallback-derived innovation span in the backfilled history row', () => {
          // Assert
          expect(
            (historyEntries[0].stats[0] as SpeciesHistoryStatExtended)
              .innovationRange,
          ).toBe(101);
        });
      });
    });
  });
});
