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
    describe('given fast mode is disabled', () => {
      it('returns without changing any telemetry defaults', () => {
        // Arrange
        const telemetryContext: { _fastModeTuned?: boolean } = {};
        const telemetryOptions = {
          fastMode: false,
          diversityMetrics: {
            enabled: true,
            pairSample: 7,
            graphletSample: 9,
          },
          novelty: {
            enabled: true,
            k: 4,
          },
        } as NeatOptions & TelemetryDiversityOptions;

        // Act
        applyFastModeDefaults(telemetryContext, telemetryOptions);

        // Assert
        expect({
          pairSample: telemetryOptions.diversityMetrics?.pairSample,
          graphletSample: telemetryOptions.diversityMetrics?.graphletSample,
          noveltyK: telemetryOptions.novelty?.k,
          fastModeTuned: telemetryContext._fastModeTuned,
        }).toEqual({
          pairSample: 7,
          graphletSample: 9,
          noveltyK: 4,
          fastModeTuned: undefined,
        });
      });
    });

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

    describe('given fast mode with an already tuned context', () => {
      it('returns immediately without mutating the sampling knobs', () => {
        // Arrange
        const telemetryContext: { _fastModeTuned?: boolean } = {
          _fastModeTuned: true,
        };
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
          pairSample: undefined,
          graphletSample: undefined,
          noveltyK: undefined,
          fastModeTuned: true,
        });
      });
    });

    describe('given fast mode with no diversity metrics', () => {
      it('skips the sampling block while still setting novelty defaults', () => {
        // Arrange
        const telemetryContext: { _fastModeTuned?: boolean } = {};
        const telemetryOptions = {
          fastMode: true,
          diversityMetrics: undefined,
          novelty: {
            enabled: true,
          },
        } as NeatOptions & TelemetryDiversityOptions;

        // Act
        applyFastModeDefaults(telemetryContext, telemetryOptions);

        // Assert
        expect({
          diversityMetrics: telemetryOptions.diversityMetrics,
          noveltyK: telemetryOptions.novelty?.k,
          fastModeTuned: telemetryContext._fastModeTuned,
        }).toEqual({
          diversityMetrics: undefined,
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

    describe('given no compatibility distance callback is supplied', () => {
      it('falls back to zero-distance samples', () => {
        // Arrange
        const genomes = [
          createTelemetryGenome({ nodeCount: 2, connectionCount: 1 }),
          createTelemetryGenome({ nodeCount: 2, connectionCount: 1 }),
          createTelemetryGenome({ nodeCount: 2, connectionCount: 1 }),
        ];

        // Act
        const compatibilityStats = computeCompatibilityStats(
          genomes,
          genomes.length,
          1,
          createRngFactory([0.1, 0.8]),
          undefined,
        );

        // Assert
        expect(compatibilityStats).toEqual({ meanCompat: 0, varCompat: 0 });
      });
    });
  });

  describe('computeEntropyStats', () => {
    describe('given the population is empty', () => {
      it('returns zeroed entropy statistics', () => {
        // Act
        const entropyStats = computeEntropyStats([], () => 0);

        // Assert
        expect(entropyStats).toEqual({ meanEntropy: 0, varEntropy: 0 });
      });
    });

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

    describe('given a disabled edge and a partially selected edge', () => {
      it('skips disabled links and ignores edges that only touch one selected node', () => {
        // Arrange
        const telemetryGenome = createTelemetryGenome({
          connections: [
            { enabled: false, fromIndex: 0, toIndex: 0 },
            { enabled: true, fromIndex: 0, toIndex: 2 },
          ],
          nodeCount: 3,
        });
        const selectedNodes = [telemetryGenome.nodes[0]];

        // Act
        const edgeCount = countEnabledEdges(telemetryGenome, selectedNodes);

        // Assert
        expect(edgeCount).toBe(0);
      });
    });
  });

  describe('computeGraphletEntropy', () => {
    describe('given the sampled population is empty', () => {
      it('returns zero entropy without sampling anything', () => {
        // Act
        const graphletEntropy = computeGraphletEntropy(
          [],
          0,
          1,
          createRngFactory([0]),
        );

        // Assert
        expect(graphletEntropy).toBe(0);
      });
    });

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

    describe('given the sampled genome slot is empty', () => {
      it('stops the sampling loop without raising the entropy', () => {
        // Arrange
        const genomes = new Proxy([createTelemetryGenome({ nodeCount: 1 })], {
          get: (target, property) => {
            if (property === 'length') return 1;
            if (typeof property === 'string' && /^\d+$/.test(property))
              return undefined;
            return Reflect.get(target, property);
          },
        }) as TelemetryGenome[];

        // Act
        const graphletEntropy = computeGraphletEntropy(
          genomes,
          genomes.length,
          1,
          createRngFactory([0]),
        );

        // Assert
        expect(graphletEntropy).toBe(0);
      });
    });

    describe('given a sampled genome has fewer than three nodes', () => {
      it('skips the graphlet without counting a motif', () => {
        // Arrange
        const genomes = [createTelemetryGenome({ nodeCount: 2, connectionCount: 1 })];

        // Act
        const graphletEntropy = computeGraphletEntropy(
          genomes,
          genomes.length,
          1,
          createRngFactory([0]),
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
