/**
 * Tests for computeDiversityStats fast-mode tuning inside telemetry module.
 * Single expectation per test.
 */
import { computeDiversityStats as telemetryComputeDiversityStats } from '../../src/neat/neat.telemetry';
import type { GenomeDetailed, TelemetryEntry } from '../../src/neat/neat.types';

type DiversityGenome = GenomeDetailed & {
  nodes: Array<{ geneId: number }>;
  connections: Array<{
    from: { geneId: number };
    to: { geneId: number };
    enabled: boolean;
  }>;
};

type DiversityContext = Record<string, unknown> & {
  population: DiversityGenome[];
  _getRNG: () => () => number;
  _compatibilityDistance: (a: DiversityGenome, b: DiversityGenome) => number;
  _structuralEntropy: (genome: DiversityGenome) => number;
  _lineageEnabled: boolean;
  options: {
    fastMode: boolean;
    diversityMetrics: {
      enabled: boolean;
      pairSample?: number;
      graphletSample?: number;
    };
    novelty: { enabled: boolean; k?: number };
  };
  _diversityStats?: TelemetryEntry['diversity'];
};

/** Create a stub Neat-like object with required fields for diversity stats */
const makeNeat = (popSize: number): DiversityContext => {
  /** population array of genome-like objects */
  const population: DiversityGenome[] = Array.from(
    { length: popSize },
    (_, index) => {
      const nodes = Array.from({ length: 3 + (index % 3) }, (_, nodeIndex) => ({
        geneId: nodeIndex,
      }));
      return {
        nodes,
        connections: [
          {
            from: nodes[0],
            to: nodes[1],
            enabled: true,
          },
        ],
        _depth: index % 5,
        _id: index + 1,
        score: 0,
      };
    },
  );
  /** stub instance providing options and helpers */
  let rngCounter = 0;
  return {
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
};

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
