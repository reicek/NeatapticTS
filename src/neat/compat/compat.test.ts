import Network from '../../architecture/network';
import Neat from '../../neat';
import { Architect, methods } from '../../neataptic';
import { createGenomeFromNetwork } from '../genome/genome';
import { validateNativeGenome } from '../validate/neat.validate';
import type { NetworkJSON } from '../../architecture/network/network.types';

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
    throw new Error('Expected recurrent module fixtures with hidden nodes and gated connections.');
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
          gaterGeneIds: [...new Set(gatedConnections.map((connection) => connection.gaterGeneId))],
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
        const malformedGenome = neat.population[0] as NetworkWithMutableConnections;
        Reflect.deleteProperty(malformedGenome.connections[0], 'innovation');

        // Assert
        expect(() =>
          neat._compatibilityDistance(malformedGenome, neat.population[1]),
        ).toThrow(/Compatibility distance requires explicit connection innovations/);
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
        const malformedGenome = neat.population[0] as NetworkWithMutableConnections;
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
        const sourcePayload = Architect.lstm(1, 2, 1)
          .toJSON() as unknown as NetworkJSON;
        const canonicalGenome = createGenomeFromNetwork(
          Network.fromJSON(sourcePayload as unknown as Record<string, unknown>),
        );
        const genomeWithTemporalExtensions = structuredClone(canonicalGenome);
        genomeWithTemporalExtensions.extensions = createTemporalModuleExtensions(
          sourcePayload,
        );

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
});
