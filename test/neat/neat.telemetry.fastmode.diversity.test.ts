/**
 * Tests for computeDiversityStats fast-mode tuning inside telemetry module.
 * Single expectation per test.
 */
import { computeDiversityStats as telemetryComputeDiversityStats } from '../../src/neat/neat.telemetry';

/** Create a stub Neat-like object with required fields for diversity stats */
function makeNeat(popSize: number) {
  /** population array of genome-like objects */
  const population = Array.from({ length: popSize }, (_, i) => ({
    nodes: new Array(3 + (i % 3)).fill(0).map((__, j) => ({ geneId: j })),
    connections: [{ from: { geneId: 0 }, to: { geneId: 1 }, enabled: true }],
    _depth: i % 5,
  }));
  /** stub instance providing options and helpers */
  let rngCounter = 0;
  const neat: any = {
    population,
    // Deterministic but cycling RNG to avoid infinite loops when sampling unique indices
    _getRNG: () => () => (rngCounter = (rngCounter + 1) % 97) / 97,
    _compatibilityDistance: () => 1,
    _structuralEntropy: () => 0.5,
    _lineageEnabled: true,
    options: {
      fastMode: true,
      diversityMetrics: { enabled: true },
      novelty: { enabled: true },
    },
  };
  return neat;
}

describe('Telemetry diversity fast-mode adjustments', () => {
  test('fast mode sets default pairSample', () => {
    // Arrange
    const neat = makeNeat(10);
    // Act
    telemetryComputeDiversityStats.call(neat);
    // Assert
    expect(neat.options.diversityMetrics.pairSample).toBe(20);
  });
});
