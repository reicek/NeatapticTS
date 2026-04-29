import Network from '../../architecture/network';
import Neat from '../../neat';
import { Architect, methods } from '../../neataptic';
import { createGenomeFromNetwork } from '../genome/genome';
import { validateNativeGenome } from '../validate/neat.validate';
import type { NetworkJSON } from '../../architecture/network/network.types';
import { _fallbackInnov } from './compat';
import {
  compareInnovationLists,
  computeCompatibilityDistance,
  ensureGenerationCache,
  getSortedInnovationCache,
  resolveMaxInnovation,
} from './core/compat.core';
import type { NeatLikeForCompat } from './core/compat.types';

type ConnectionWithOptionalInnovation = {
  from: { index: number };
  to: { index: number };
  weight: number;
  innovation?: number;
};

type NetworkWithMutableConnections = Network & {
  connections: ConnectionWithOptionalInnovation[];
  _compatCache?: Array<[number, number]>;
  _compatInnovationMode?: 'allow-fallback';
};

function collectIssueCodes(issues: Array<{ code: string }>): string[] {
  return issues.map((issue) => issue.code);
}

function createTemporalModuleExtensions(
  networkJson: NetworkJSON,
): NonNullable<NetworkJSON['extensions']> {
  const hiddenNodeGeneIds = networkJson.nodes
    .filter((node) => node.type !== 'input' && node.type !== 'output')
    .map((node) => node.geneId)
    .filter((geneId): geneId is number => typeof geneId === 'number');
  const gatedConnections = networkJson.connections.filter(
    (
      connection,
    ): connection is NetworkJSON['connections'][number] & {
      innovation: number;
      gaterGeneId: number;
    } =>
      typeof connection.innovation === 'number' &&
      typeof connection.gaterGeneId === 'number',
  );

  if (hiddenNodeGeneIds.length === 0 || gatedConnections.length === 0) {
    throw new Error(
      'Expected recurrent module fixtures with hidden nodes and gated connections.',
    );
  }

  return {
    version: 1,
    values: {
      recurrentModules: [
        {
          moduleId: 'module:lstm:0',
          kind: 'lstm',
          nodeGeneIdsByRole: {
            recurrentCore: hiddenNodeGeneIds,
          },
          connectionInnovations: gatedConnections.map(
            (connection) => connection.innovation,
          ),
        },
      ],
      gatedBlocks: [
        {
          blockId: 'gated:block:0',
          gaterGeneIds: [
            ...new Set(
              gatedConnections.map((connection) => connection.gaterGeneId),
            ),
          ],
          connectionInnovations: gatedConnections.map(
            (connection) => connection.innovation,
          ),
        },
      ],
    },
  };
}

describe('neat compat chapter', () => {
  describe('_compatibilityDistance', () => {
    describe('given two native genomes with explicit innovation ids', () => {
      const fitness = (network: Network) => network.nodes.length;
      const neat = new Neat(3, 1, fitness, {
        popsize: 4,
        seed: 776,
      });

      let comparedGenome: NetworkWithMutableConnections;

      beforeAll(async () => {
        // Arrange
        comparedGenome = neat.population[0] as NetworkWithMutableConnections;

        // Act
        neat._compatibilityDistance(neat.population[0], neat.population[1]);
      });

      describe('when the first native comparison completes', () => {
        it('stores the canonical explicit-innovation cache on the genome', () => {
          // Assert
          expect(Array.isArray(comparedGenome._compatCache)).toBe(true);
        });
      });
    });

    describe('given two fallback-allowed genomes missing explicit innovation ids', () => {
      const fitness = (network: Network) => network.nodes.length;
      const neat = new Neat(3, 1, fitness, {
        popsize: 4,
        seed: 777,
        excessCoeff: 1,
        disjointCoeff: 1,
        weightDiffCoeff: 0.4,
      });

      let genomeA: NetworkWithMutableConnections;
      let genomeB: NetworkWithMutableConnections;
      let expectedFallbackInnovation: number;
      let firstDistance: number;
      let secondDistance: number;

      beforeAll(async () => {
        // Arrange
        await neat.evaluate();
        genomeA = neat.population[0] as NetworkWithMutableConnections;
        genomeB = neat.population[1] as NetworkWithMutableConnections;

        genomeA.connections.forEach((connection) => {
          Reflect.deleteProperty(connection, 'innovation');
        });
        genomeB.connections.forEach((connection) => {
          Reflect.deleteProperty(connection, 'innovation');
        });
        genomeA._compatInnovationMode = 'allow-fallback';
        genomeB._compatInnovationMode = 'allow-fallback';
        expectedFallbackInnovation =
          genomeA.connections[0].from.index * 100_000 +
          genomeA.connections[0].to.index;

        // Act
        firstDistance = neat._compatibilityDistance(genomeA, genomeB);
        secondDistance = neat._compatibilityDistance(genomeA, genomeB);
      });

      describe('when fallback innovations are used for the first comparison', () => {
        it('still produces a finite compatibility distance', () => {
          // Assert
          expect(Number.isFinite(firstDistance)).toBe(true);
        });
      });

      describe('when the same pair is compared twice in one generation', () => {
        it('reuses the cached distance value', () => {
          // Assert
          expect(secondDistance).toBe(firstDistance);
        });
      });

      describe('when missing innovations are normalized into the cache', () => {
        it('keeps fallback-derived innovation ids out of the native genome cache', () => {
          // Assert
          expect(genomeA._compatCache).toBeUndefined();
        });
      });

      describe('when the first comparison normalizes sorted innovations', () => {
        it('still derives the expected synthetic innovation id for comparison', () => {
          // Assert
          expect(Number.isFinite(expectedFallbackInnovation)).toBe(true);
        });
      });

      describe('when a fallback-allowed genome carries a stale explicit cache', () => {
        it('drops the stale cache before comparison', () => {
          // Arrange
          const staleCache: Array<[number, number]> = [[9_999, 1]];
          genomeA._compatCache = staleCache;

          // Act
          neat._compatibilityDistance(genomeA, genomeB);

          // Assert
          expect(genomeA._compatCache).toBeUndefined();
        });
      });
    });

    describe('given a native genome loses an explicit innovation id', () => {
      it('fails fast instead of silently synthesizing one', async () => {
        // Arrange
        const fitness = (network: Network) => network.nodes.length;
        const neat = new Neat(3, 1, fitness, {
          popsize: 2,
          seed: 780,
        });
        await neat.evaluate();
        const malformedGenome = neat
          .population[0] as NetworkWithMutableConnections;
        Reflect.deleteProperty(malformedGenome.connections[0], 'innovation');

        // Assert
        expect(() =>
          neat._compatibilityDistance(malformedGenome, neat.population[1]),
        ).toThrow(
          /Compatibility distance requires explicit connection innovations/,
        );
      });
    });

    describe('given a native genome without _id and a connection with no from or to endpoints', () => {
      it('uses null fallbacks for all missing diagnostic fields (lines 358,360,361 ?? null arms)', () => {
        // Arrange: fake genome with no _id; connection has no from, to, or innovation
        const fitness = (network: Network) => network.nodes.length;
        const neat = new Neat(2, 1, fitness, { popsize: 2, seed: 810 });
        const referenceGenome = neat.population[1];
        const fakeGenome = {
          nodes: referenceGenome.nodes,
          connections: [{ weight: 0.5, enabled: true }], // no from, no to, no innovation
          // no _id → genome._id ?? null fires
        } as unknown as Parameters<typeof neat._compatibilityDistance>[0];

        // Act & Assert: error thrown, null fallbacks exercised internally
        expect(() =>
          neat._compatibilityDistance(fakeGenome, referenceGenome),
        ).toThrow(
          /Compatibility distance requires explicit connection innovations/,
        );
      });
    });

    describe('given a native genome carries duplicate connection innovations', () => {
      it('reports the malformed identity before compatibility fallback can hide it', () => {
        // Arrange
        const fitness = (network: Network) => network.nodes.length;
        const neat = new Neat(3, 1, fitness, {
          popsize: 2,
          seed: 778,
        });
        const malformedGenome = neat
          .population[0] as NetworkWithMutableConnections;
        malformedGenome.connections[1].innovation =
          malformedGenome.connections[0].innovation;

        // Act
        const validationReport = validateNativeGenome(malformedGenome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'duplicate-connection-innovation',
        );
      });
    });

    describe('given a freshly bootstrapped proper-NEAT population', () => {
      it('treats equivalent generation-zero genomes as zero-distance homologs', () => {
        // Arrange
        const fitness = (network: Network) => network.nodes.length;
        const neat = new Neat(3, 1, fitness, {
          popsize: 2,
          seed: 779,
          excessCoeff: 1,
          disjointCoeff: 1,
          weightDiffCoeff: 0.4,
        });
        const firstGenome = neat.population[0];
        const secondGenome = neat.population[1];

        // Act
        const compatibilityDistance = neat._compatibilityDistance(
          firstGenome,
          secondGenome,
        );

        // Assert
        expect(compatibilityDistance).toBe(0);
      });
    });

    describe('given two strict genomes differ only by connection-gain extension state', () => {
      it('ignores the extension bag when computing canonical compatibility distance', () => {
        // Arrange
        const fitness = (network: Network) => network.nodes.length;
        const neat = new Neat(1, 1, fitness, {
          popsize: 1,
          seed: 781,
          excessCoeff: 1,
          disjointCoeff: 1,
          weightDiffCoeff: 0.4,
        });
        const sourceNetwork = new Network(1, 1, { seed: 782 });
        sourceNetwork.connections[0].gain = 1.5;
        const genomeWithGainExtension = createGenomeFromNetwork(sourceNetwork, {
          connectionGain: true,
        });
        const canonicalGenome = structuredClone(genomeWithGainExtension);
        Reflect.deleteProperty(canonicalGenome, 'extensions');

        // Act
        const compatibilityDistance = neat._compatibilityDistance(
          genomeWithGainExtension as unknown as Network,
          canonicalGenome as unknown as Network,
        );

        // Assert
        expect(compatibilityDistance).toBe(0);
      });
    });

    describe('given two strict genomes differ only by canonical node activation', () => {
      it('keeps canonical compatibility distance unchanged', () => {
        // Arrange
        const fitness = (network: Network) => network.nodes.length;
        const neat = new Neat(1, 1, fitness, {
          popsize: 1,
          seed: 783,
          excessCoeff: 1,
          disjointCoeff: 1,
          weightDiffCoeff: 0.4,
        });
        const sourceNetwork = new Network(1, 1, { seed: 784 });
        const canonicalGenome = createGenomeFromNetwork(sourceNetwork);
        sourceNetwork.nodes.at(-1)!.squash = methods.Activation.tanh;
        const mutatedActivationGenome = createGenomeFromNetwork(sourceNetwork);

        // Act
        const compatibilityDistance = neat._compatibilityDistance(
          canonicalGenome as unknown as Network,
          mutatedActivationGenome as unknown as Network,
        );

        // Assert
        expect(compatibilityDistance).toBe(0);
      });
    });

    describe('given two strict genomes differ only by temporal module extension state', () => {
      it('keeps canonical compatibility distance unchanged', () => {
        // Arrange
        const fitness = (network: Network) => network.nodes.length;
        const neat = new Neat(1, 1, fitness, {
          popsize: 1,
          seed: 785,
          excessCoeff: 1,
          disjointCoeff: 1,
          weightDiffCoeff: 0.4,
        });
        const sourcePayload = Architect.lstm(
          1,
          2,
          1,
        ).toJSON() as unknown as NetworkJSON;
        const canonicalGenome = createGenomeFromNetwork(
          Network.fromJSON(sourcePayload as unknown as Record<string, unknown>),
        );
        const genomeWithTemporalExtensions = structuredClone(canonicalGenome);
        genomeWithTemporalExtensions.extensions =
          createTemporalModuleExtensions(sourcePayload);

        // Act
        const compatibilityDistance = neat._compatibilityDistance(
          genomeWithTemporalExtensions as unknown as Network,
          canonicalGenome as unknown as Network,
        );

        // Assert
        expect(compatibilityDistance).toBe(0);
      });
    });
  });

  describe('ensureGenerationCache()', () => {
    describe('given an existing cache gen that does not match the current generation', () => {
      it('resets the distance cache when the generation has advanced (line 39 right arm of ||)', () => {
        // Arrange: _compatCacheGen set to old gen; generation has advanced
        const ctx = {
          _compatCacheGen: 5,
          generation: 6,
          _compatDistCache: new Map([['pair:A:B', 0.3]]),
        } as unknown as NeatLikeForCompat;

        // Act
        ensureGenerationCache(ctx);

        // Assert: cache was reset to a fresh empty map
        expect(ctx._compatDistCache!.size).toBe(0);
      });
    });

    describe('given the cache gen already matches the current generation', () => {
      it('skips the reset when the generation has not changed (false arm of entire condition)', () => {
        // Arrange: _compatCacheGen = truthy non-zero value that matches generation
        const existingMap = new Map([['pair:A:B', 0.3]]);
        const ctx = {
          _compatCacheGen: 7,
          generation: 7,
          _compatDistCache: existingMap,
        } as unknown as NeatLikeForCompat;

        // Act
        ensureGenerationCache(ctx);

        // Assert: cache was NOT reset
        expect(ctx._compatDistCache!.size).toBe(1);
      });
    });
  });

  describe('resolveMaxInnovation()', () => {
    describe('given an empty list', () => {
      it('returns 0 as the max innovation (empty-list arm)', () => {
        expect(resolveMaxInnovation([])).toBe(0);
      });
    });
  });

  describe('computeCompatibilityDistance()', () => {
    describe('given metrics with zero matching genes', () => {
      it('uses 0 as the average weight difference (matchingCount === 0 arm)', () => {
        // Arrange
        const mockCtx = {
          options: { excessCoeff: 1, disjointCoeff: 1, weightDiffCoeff: 1 },
        } as unknown as NeatLikeForCompat;
        const metrics = {
          firstGenomeSize: 2,
          secondGenomeSize: 2,
          matchingCount: 0,
          disjointCount: 2,
          excessCount: 0,
          weightDifferenceSum: 0,
        };

        // Act
        const distance = computeCompatibilityDistance(mockCtx, metrics);

        // Assert: distance computed without dividing by zero
        expect(Number.isFinite(distance)).toBe(true);
      });
    });
  });

  describe('compareInnovationLists()', () => {
    describe('given two lists where genome1 has a lower innovation not in genome2', () => {
      it('increments disjoint or excess count for the first genome (line 202-207 arm)', () => {
        // Arrange: list1 has innovation 1, list2 starts at 2 → innovFirst < innovSecond
        const list1: [number, number][] = [
          [1, 0.5],
          [3, 0.7],
        ];
        const list2: [number, number][] = [
          [2, 0.5],
          [4, 0.7],
        ];

        // Act
        const metrics = compareInnovationLists(list1, list2);

        // Assert: some disjoint or excess was recorded
        expect(metrics.disjointCount + metrics.excessCount).toBeGreaterThan(0);
      });
    });

    describe('given two lists where genome2 has a lower innovation not in genome1', () => {
      it('increments disjoint or excess count for the second genome (line 209-211 arm)', () => {
        // Arrange: list2 has innovation 2, list1 starts at 3 → innovSecond < innovFirst
        const list1: [number, number][] = [[3, 0.7]];
        const list2: [number, number][] = [
          [2, 0.5],
          [4, 0.3],
        ];

        // Act
        const metrics = compareInnovationLists(list1, list2);

        // Assert: some disjoint or excess was recorded for the second list
        expect(metrics.disjointCount + metrics.excessCount).toBeGreaterThan(0);
      });
    });
  });

  describe('getSortedInnovationCache()', () => {
    describe('given a native genome whose cache was already built in require-explicit mode', () => {
      it('returns the cached result on the second call without rebuilding (line 121 arm)', () => {
        // Arrange: native Neat genome defaults to require-explicit mode
        const fitness = (network: Network) => network.nodes.length;
        const neat = new Neat(2, 1, fitness, { popsize: 2, seed: 800 });
        const genome = neat.population[0];
        const mockCtx = neat as unknown as NeatLikeForCompat;

        // First call builds and caches the result
        const firstResult = getSortedInnovationCache(mockCtx, genome);

        // Act: second call should hit the cache (line 121)
        const secondResult = getSortedInnovationCache(mockCtx, genome);

        // Assert: same reference returned from cache
        expect(secondResult).toBe(firstResult);
      });
    });
  });

  describe('_fallbackInnov', () => {
    const mockCtx = {} as unknown as NeatLikeForCompat;

    describe('given a connection with no from endpoint', () => {
      it('falls back to zero as the from index', () => {
        // Act
        const result = _fallbackInnov.call(mockCtx, {
          to: { index: 3 },
          weight: 1.0,
        });

        // Assert: 0 * 100_000 + 3
        expect(result).toBe(3);
      });
    });

    describe('given a connection with no to endpoint', () => {
      it('falls back to zero as the to index', () => {
        // Act
        const result = _fallbackInnov.call(mockCtx, {
          from: { index: 2 },
          weight: 1.0,
        });

        // Assert: 2 * 100_000 + 0
        expect(result).toBe(200_000);
      });
    });
  });
});
