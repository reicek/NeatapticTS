import {
  applyFastModeDefaults,
  computeCompatibilityStats,
  computeEntropyStats,
  computeGraphletEntropy,
  countEnabledEdges,
  pickDistinctIndices,
} from './telemetry.metrics.diversity';
import type { NeatOptions } from '../../shared/neat.shared.types';
import type { TelemetryDiversityOptions } from '../types/telemetry.types';
import type { TelemetryGenome } from '../types/telemetry.types';

function createTelemetryOptions(input: {
  pairSample?: number;
  graphletSample?: number;
  noveltyK?: number;
}): NeatOptions & TelemetryDiversityOptions {
  return {
    fastMode: true,
    diversityMetrics: {
      enabled: true,
      pairSample: input.pairSample,
      graphletSample: input.graphletSample,
    },
    novelty: {
      enabled: true,
      k: input.noveltyK,
    },
  };
}

describe('neat telemetry diversity metrics chapter', () => {
  describe('applyFastModeDefaults', () => {
    describe('given fast mode with unspecified sampling knobs', () => {
      it('fills the diversity and novelty defaults once', () => {
        // Arrange
        const telemetryContext: { _fastModeTuned?: boolean } = {};
        const telemetryOptions = createTelemetryOptions({});

        // Act
        applyFastModeDefaults(telemetryContext, telemetryOptions);

        // Assert
        expect({
          pairSample: telemetryOptions.diversityMetrics?.pairSample,
          graphletSample: telemetryOptions.diversityMetrics?.graphletSample,
          noveltyK: telemetryOptions.novelty?.k,
          fastModeTuned: telemetryContext._fastModeTuned,
        }).toEqual({
          pairSample: 20,
          graphletSample: 30,
          noveltyK: 5,
          fastModeTuned: true,
        });
      });
    });

    describe('given fast mode with explicit sampling knobs', () => {
      it('preserves the caller supplied diversity and novelty values', () => {
        // Arrange
        const telemetryContext: { _fastModeTuned?: boolean } = {};
        const telemetryOptions = createTelemetryOptions({
          pairSample: 50,
          graphletSample: 70,
          noveltyK: 11,
        });

        // Act
        applyFastModeDefaults(telemetryContext, telemetryOptions);

        // Assert
        expect({
          pairSample: telemetryOptions.diversityMetrics?.pairSample,
          graphletSample: telemetryOptions.diversityMetrics?.graphletSample,
          noveltyK: telemetryOptions.novelty?.k,
          fastModeTuned: telemetryContext._fastModeTuned,
        }).toEqual({
          pairSample: 50,
          graphletSample: 70,
          noveltyK: 11,
          fastModeTuned: true,
        });
      });
    });
  });

  describe('computeCompatibilityStats', () => {
    describe('given the sampled population is too small to form pairs', () => {
      it('returns zeroed compatibility statistics', () => {
        // Act
        const compatibilityStats = computeCompatibilityStats(
          [],
          0,
          3,
          createRngFactory([0.1]),
          () => 1,
        );

        // Assert
        expect(compatibilityStats).toEqual({ meanCompat: 0, varCompat: 0 });
      });
    });

    describe('given the compatibility distance is deterministic', () => {
      it('returns the repeated sampled mean with zero variance', () => {
        // Arrange
        const genomes = [
          createTelemetryGenome({ nodeCount: 2, connectionCount: 1 }),
          createTelemetryGenome({ nodeCount: 2, connectionCount: 3 }),
        ];

        // Act
        const compatibilityStats = computeCompatibilityStats(
          genomes,
          genomes.length,
          3,
          createRngFactory([0]),
          (firstGenome, secondGenome) =>
            Math.abs(
              firstGenome.connections.length - secondGenome.connections.length,
            ),
        );

        // Assert
        expect(compatibilityStats).toEqual({ meanCompat: 2, varCompat: 0 });
      });
    });
  });

  describe('computeEntropyStats', () => {
    describe('given the entropy function returns a simple connection count', () => {
      it('computes the mean and variance across the genomes', () => {
        // Arrange
        const genomes = [
          createTelemetryGenome({ nodeCount: 2, connectionCount: 1 }),
          createTelemetryGenome({ nodeCount: 2, connectionCount: 3 }),
        ];

        // Act
        const entropyStats = computeEntropyStats(
          genomes,
          (genome) => genome.connections.length,
        );

        // Assert
        expect(entropyStats).toEqual({ meanEntropy: 2, varEntropy: 1 });
      });
    });
  });

  describe('pickDistinctIndices', () => {
    describe('given the RNG repeats values before reaching the requested count', () => {
      it('continues until it collects the requested number of distinct indices', () => {
        // Arrange
        const rng = createRngFactory([0.1, 0.1, 0.4, 0.8])();

        // Act
        const selectedIndices = pickDistinctIndices(3, 3, rng);

        // Assert
        expect(selectedIndices).toEqual([0, 1, 2]);
      });
    });
  });

  describe('countEnabledEdges', () => {
    describe('given more than three enabled edges connect the selected nodes', () => {
      it('caps the returned graphlet edge count at three', () => {
        // Arrange
        const telemetryGenome = createTelemetryGenome({
          connections: [
            { fromIndex: 0, toIndex: 1 },
            { fromIndex: 1, toIndex: 2 },
            { fromIndex: 2, toIndex: 0 },
            { fromIndex: 0, toIndex: 2 },
          ],
          nodeCount: 3,
        });

        // Act
        const edgeCount = countEnabledEdges(
          telemetryGenome,
          telemetryGenome.nodes,
        );

        // Assert
        expect(edgeCount).toBe(3);
      });
    });
  });

  describe('computeGraphletEntropy', () => {
    describe('given repeated samples always land on the same fully connected motif', () => {
      it('returns zero entropy because every sample lands in one motif bucket', () => {
        // Arrange
        const genomes = [
          createTelemetryGenome({
            connections: [
              { fromIndex: 0, toIndex: 1 },
              { fromIndex: 1, toIndex: 2 },
              { fromIndex: 2, toIndex: 0 },
              { fromIndex: 0, toIndex: 2 },
            ],
            nodeCount: 3,
          }),
        ];

        // Act
        const graphletEntropy = computeGraphletEntropy(
          genomes,
          genomes.length,
          2,
          createRngFactory([0.1, 0.4, 0.8]),
        );

        // Assert
        expect(graphletEntropy).toBe(0);
      });
    });
  });
});

function createRngFactory(sequence: number[]): () => () => number {
  return () => {
    let sequenceIndex = 0;

    return () => {
      const value = sequence[sequenceIndex % sequence.length];
      sequenceIndex += 1;
      return value;
    };
  };
}

function createTelemetryGenome(input: {
  connectionCount?: number;
  connections?: Array<{
    enabled?: boolean;
    fromIndex: number;
    toIndex: number;
  }>;
  nodeCount: number;
}): TelemetryGenome {
  const nodes = Array.from({ length: input.nodeCount }, (_, nodeIndex) => ({
    geneId: nodeIndex,
  }));
  const generatedConnections: Array<{
    enabled?: boolean;
    fromIndex: number;
    toIndex: number;
  }> = input.connections
    ? input.connections
    : Array.from({ length: input.connectionCount ?? 0 }, (_, connectionIndex) => ({
        fromIndex: 0,
        toIndex: Math.min(connectionIndex + 1, input.nodeCount - 1),
      }));

  return {
    connections: generatedConnections.map((connection) => ({
      enabled: connection.enabled ?? true,
      from: nodes[connection.fromIndex],
      to: nodes[connection.toIndex],
    })),
    nodes,
  };
}
