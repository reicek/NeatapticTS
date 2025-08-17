/**
 * Tests for getSpeciesHistory fallback computations in species module.
 * Single expectation per test.
 */
import { getSpeciesHistory } from '../../src/neat/neat.species';

/** Build a minimal context for species history fallback */
function ctxWithHistory() {
  /** two member genomes with connections containing innovations */
  const memberA = {
    connections: [
      { innovation: 2, enabled: true },
      { innovation: 5, enabled: false },
    ],
  } as any;
  const memberB = { connections: [{ innovation: 7, enabled: true }] } as any;
  /** species present in current population */
  const species = [
    { id: 1, members: [memberA, memberB], bestScore: 1, lastImproved: 0 },
  ];
  /** species history snapshot lacking extended fields */
  const _speciesHistory = [
    {
      generation: 0,
      stats: [{ id: 1, size: 2, bestScore: 1, lastImproved: 0 }],
    },
  ];
  /** context object implementing required members */
  const ctx: any = {
    options: { speciesAllocation: { extendedHistory: true } },
    _speciesHistory,
    _species: species,
  };
  return ctx;
}

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
    const stat = ctx._speciesHistory[0].stats[0];
    stat.innovationRange = 123; // sentinel
    stat.enabledRatio = 0.5;
    // Act
    const hist = getSpeciesHistory.call(ctx);
    // Assert
    expect((hist[0].stats[0] as any).innovationRange).toBe(123);
  });
});
