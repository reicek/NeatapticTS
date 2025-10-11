/**
 * Tests for getSpeciesHistory fallback computations in species module.
 * Single expectation per test.
 */
import { getSpeciesHistory } from '../../src/neat/neat.species';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciesHistoryEntry,
  SpeciesHistoryStatExtended,
  SpeciesLike,
} from '../../src/neat/neat.types';

type SpeciesHistoryContext = {
  options: { speciesAllocation: { extendedHistory: boolean } } & Record<
    string,
    unknown
  >;
  _speciesHistory: SpeciesHistoryEntry[];
  _species: SpeciesLike[];
  _fallbackInnov?: (connection: ConnectionLike) => number;
};

/** Build a minimal context for species history fallback */
const ctxWithHistory = (): SpeciesHistoryContext => {
  /** two member genomes with connections containing innovations */
  const memberA: GenomeDetailed = {
    _id: 1,
    nodes: [],
    connections: [
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
    ],
  };
  const memberB: GenomeDetailed = {
    _id: 2,
    nodes: [],
    connections: [
      {
        innovation: 7,
        enabled: true,
        from: { geneId: 1 },
        to: { geneId: 3 },
      },
    ],
  };
  /** species present in current population */
  const species: SpeciesLike[] = [
    {
      id: 1,
      members: [memberA, memberB],
      bestScore: 1,
      lastImproved: 0,
    },
  ];
  /** species history snapshot lacking extended fields */
  const _speciesHistory: SpeciesHistoryEntry[] = [
    {
      generation: 0,
      stats: [{ id: 1, size: 2, bestScore: 1, lastImproved: 0 }],
    },
  ];
  /** context object implementing required members */
  return {
    options: { speciesAllocation: { extendedHistory: true } },
    _speciesHistory,
    _species: species,
  };
};

describe('Species history fallbacks', () => {
  test('augments stats with innovationRange when missing', () => {
    // Arrange
    const ctx = ctxWithHistory();
    // Act
    const hist = getSpeciesHistory.call(ctx);
    // Assert
    expect('innovationRange' in hist[0].stats[0]).toBe(true);
  });
  test('skips recomputation when fields present', () => {
    // Arrange
    const ctx = ctxWithHistory();
    const stat = ctx._speciesHistory[0].stats[0] as SpeciesHistoryStatExtended;
    stat.innovationRange = 123; // sentinel
    stat.enabledRatio = 0.5;
    // Act
    const hist = getSpeciesHistory.call(ctx);
    // Assert
    const entry = hist[0].stats[0] as SpeciesHistoryStatExtended;
    expect(entry.innovationRange).toBe(123);
  });
});
