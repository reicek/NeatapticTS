import Network from '../../architecture/network';
import type { NetworkJSON } from '../../architecture/network/network.types';
import { methods } from '../../neataptic';
import {
  createCompatibilityGenomeView,
  createGenomeFromNetwork,
  createGenomeFromNetworkJson,
  createNetworkJsonFromGenome,
  validateGenomeContract,
} from './genome';
import type { NeatGenome } from './genome';

function collectIssueCodes(issues: Array<{ code: string }>): string[] {
  return issues.map((issue) => issue.code);
}

describe('neat genome utility coverage chapter', () => {
  describe('createGenomeFromNetworkJson', () => {
    describe('given a legacy-style payload omits strict identity fields and carries malformed values', () => {
      it('fails strict conversion after normalizing the fallback fields', () => {
        // Arrange
        const networkJson = new Network(1, 1, { seed: 1_450 })
          .toJSON() as unknown as NetworkJSON;

        networkJson.nodes = [
          {
            ...networkJson.nodes[0],
            geneId: undefined as unknown as number,
          },
          {
            ...networkJson.nodes[1],
            bias: Number.NaN,
            geneId: undefined as unknown as number,
            squash: '',
            type: 'mystery' as unknown as 'input',
          },
        ];
        networkJson.connections = [
          {
            ...networkJson.connections[0],
            from: 0,
            fromGeneId: undefined as unknown as number,
            gater: 99,
            gaterGeneId: undefined as unknown as number,
            innovation: undefined as unknown as number,
            to: 1,
            toGeneId: undefined as unknown as number,
            weight: Number.NaN,
          },
        ];

        // Assert
        expect(() => createGenomeFromNetworkJson(networkJson)).toThrow(
          'Strict genomes must assign a finite geneId to every node gene. (nodeGenes[0].geneId)',
        );
      });
    });

    describe('given a payload stores extension values in a non-plain container', () => {
      it('rejects the malformed extension bag during strict conversion', () => {
        // Arrange
        const networkJson = new Network(1, 1, { seed: 1_451 })
          .toJSON() as unknown as NetworkJSON;

        networkJson.extensions = {
          version: 1,
          values: [] as unknown as NonNullable<NetworkJSON['extensions']>['values'],
        };

        // Assert
        expect(() => createGenomeFromNetworkJson(networkJson)).toThrow(
          'Genome extension bags must carry a positive integer version and one plain object payload. (extensions)',
        );
      });
    });

    describe('given a payload stores temporal extension containers in non-array values', () => {
      it('rejects the malformed temporal extension containers during strict conversion', () => {
        // Arrange
        const networkJson = new Network(1, 1, { seed: 1_452 })
          .toJSON() as unknown as NetworkJSON;

        networkJson.extensions = {
          version: 1,
          values: {
            gatedBlocks: {} as unknown as NonNullable<NetworkJSON['extensions']>['values'],
            recurrentModules:
              {} as unknown as NonNullable<NetworkJSON['extensions']>['values'],
          },
        };

        // Assert
        expect(() => createGenomeFromNetworkJson(networkJson)).toThrow(
          'Recurrent-module extensions must be stored as one array of supported module descriptors. (extensions.values.recurrentModules)',
        );
      });
    });

    describe('given a legacy-style payload omits connection gene ids but still carries node gene ids by index', () => {
      it('reconstructs the endpoint gene ids from the indexed node lookup', () => {
        // Arrange
        const networkJson = new Network(1, 1, { seed: 1_459 })
          .toJSON() as unknown as NetworkJSON;
        const expectedConnection = networkJson.connections[0];

        networkJson.connections = [
          {
            ...expectedConnection,
            fromGeneId: undefined as unknown as number,
            toGeneId: undefined as unknown as number,
          },
        ];

        // Act
        const genome = createGenomeFromNetworkJson(networkJson);

        // Assert
        expect(genome.connectionGenes[0]).toEqual(
          expect.objectContaining({
            fromGeneId: networkJson.nodes[expectedConnection.from].geneId,
            toGeneId: networkJson.nodes[expectedConnection.to].geneId,
          }),
        );
      });
    });

    describe('given a legacy-style payload omits endpoint gene ids and points at unknown node indexes', () => {
      it('fails strict conversion after falling back to missing endpoint gene ids', () => {
        // Arrange
        const networkJson = new Network(1, 1, { seed: 1_462 })
          .toJSON() as unknown as NetworkJSON;

        networkJson.connections = [
          {
            ...networkJson.connections[0],
            from: 99,
            fromGeneId: undefined as unknown as number,
            to: 100,
            toGeneId: undefined as unknown as number,
          },
        ];

        // Assert
        expect(() => createGenomeFromNetworkJson(networkJson)).toThrow(
          'Connection endpoints must resolve to known node genes in the strict genome contract. (connectionGenes[0].fromGeneId)',
        );
      });
    });
  });

  describe('createNetworkJsonFromGenome', () => {
    describe('given one strict genome is materialized without runtime hints', () => {
      it('uses zero dropout and leaves the architecture hint undefined', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_453 }),
        );

        // Act
        const networkJson = createNetworkJsonFromGenome(genome);

        // Assert
        expect({
          architecture: networkJson.architecture,
          dropout: networkJson.dropout,
        }).toEqual({
          architecture: undefined,
          dropout: 0,
        });
      });
    });
  });

  describe('createGenomeFromNetwork', () => {
    describe('given runtime-only capture is requested while every captured value stays neutral', () => {
      it('keeps the strict genome extension bag undefined', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_454 });

        // Act
        const genome = createGenomeFromNetwork(sourceNetwork, {
          connectionGain: true,
          disabledConnectionReenableProbability: true,
          nodeResponse: true,
        });

        // Assert
        expect(genome.extensions).toBeUndefined();
      });
    });

    describe('given runtime-only capture requests only a disabled-connection re-enable probability', () => {
      it('creates a fresh extension bag for that runtime-only value', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_455 }) as Network & {
          _reenableProb?: number;
        };
        sourceNetwork._reenableProb = 0.4;

        // Act
        const genome = createGenomeFromNetwork(sourceNetwork, {
          disabledConnectionReenableProbability: true,
        });

        // Assert
        expect(genome.extensions).toEqual({
          values: {
            disabledConnectionReenableProbability: 0.4,
          },
          version: 1,
        });
      });
    });
  });

  describe('validateGenomeContract', () => {
    describe('given one strict genome breaks the public input and output size contract', () => {
      it('reports the size-related issues together', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_456 }),
        );

        genome.input = -1;
        genome.output = 2.5 as unknown as number;
        genome.nodeGenes = genome.nodeGenes.slice(0, 1);

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toEqual(
          expect.arrayContaining([
            'invalid-input-count',
            'invalid-output-count',
            'insufficient-node-count',
          ]),
        );
      });
    });

    describe('given one strict genome corrupts node ordering and connection identity together', () => {
      it('reports the node and connection integrity issues together', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_457 });
        sourceNetwork.mutate(methods.mutation.ADD_NODE);
        sourceNetwork.mutate(methods.mutation.ADD_NODE);
        const genome = createGenomeFromNetwork(sourceNetwork);

        if (genome.nodeGenes.length < 5 || genome.connectionGenes.length < 2) {
          throw new Error('Expected a multi-hidden genome fixture for integrity validation.');
        }

        genome.topologyIntent = 'feed-forward';
        genome.nodeGenes[0].type = 'mystery' as unknown as 'input';
        genome.nodeGenes[1].geneId = genome.nodeGenes[0].geneId;
        genome.nodeGenes[2].bias = Number.NaN;
        genome.nodeGenes[2].type = 'input';
        genome.nodeGenes[3].type = 'output';
        genome.nodeGenes[4].squash = '';
        genome.nodeGenes[4].type = 'hidden';

        genome.connectionGenes[0].innovation = genome.connectionGenes[1].innovation;
        genome.connectionGenes[0].weight = Number.NaN;
        genome.connectionGenes[0].fromGeneId = genome.nodeGenes[4].geneId;
        genome.connectionGenes[0].toGeneId = genome.nodeGenes[0].geneId;
        genome.connectionGenes[0].gaterGeneId = 999_999;
        genome.connectionGenes[1].fromGeneId = 999_998;

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toEqual(
          expect.arrayContaining([
            'duplicate-node-gene-id',
            'invalid-node-type',
            'invalid-node-bias',
            'invalid-node-squash',
            'input-node-order-mismatch',
            'output-node-order-mismatch',
            'duplicate-connection-innovation',
            'invalid-connection-weight',
            'feed-forward-recurrent-connection',
            'unknown-connection-endpoint',
            'unknown-gater-node',
          ]),
        );
      });
    });

    describe('given one strict genome carries malformed temporal descriptor entries', () => {
      it('reports both recurrent-module and gated-block descriptor issues', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_458 }),
        );
        const knownNodeGeneId = genome.nodeGenes[0].geneId;

        genome.extensions = {
          version: 1,
          values: {
            gatedBlocks: [
              null,
              {
                blockId: 'block:0',
                connectionInnovations: [Number.NaN],
                gaterGeneIds: [knownNodeGeneId],
              },
            ] as unknown as NonNullable<NeatGenome['extensions']>['values']['gatedBlocks'],
            recurrentModules: [
              null,
              {
                connectionInnovations: [Number.NaN],
                kind: 'lstm',
                moduleId: 'module:lstm:0',
                nodeGeneIdsByRole: {
                  recurrentCore: [knownNodeGeneId],
                },
              },
            ] as unknown as NonNullable<NeatGenome['extensions']>['values']['recurrentModules'],
          },
        };

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toEqual(
          expect.arrayContaining([
            'invalid-gated-block-extension',
            'invalid-recurrent-module-extension',
          ]),
        );
      });
    });
  });

  describe('createCompatibilityGenomeView', () => {
    describe('given a runtime-like source still carries one missing innovation id', () => {
      it('returns the original runtime-like source without creating a cached compatibility view', () => {
        // Arrange
        const runtimeLikeSource = {
          connections: [{ innovation: Number.NaN, weight: 1 }],
          nodes: [],
        } as Parameters<typeof createCompatibilityGenomeView>[0];

        // Act
        const compatibilityView = createCompatibilityGenomeView(runtimeLikeSource);

        // Assert
        expect(compatibilityView).toBe(runtimeLikeSource);
      });
    });

    describe('given a runtime source is converted into a cached compatibility view', () => {
      it('forwards runtime metadata, cache writes, and converted connection reads', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_460 }) as Network & {
          _compatCache?: Array<[number, number]>;
          _compatInnovationMode?: 'require-explicit' | 'allow-fallback';
          _id?: number;
        };
        sourceNetwork._compatInnovationMode = 'allow-fallback';
        sourceNetwork._id = 91;
        const compatibilityView = createCompatibilityGenomeView(sourceNetwork);

        compatibilityView._compatCache = [[1, 2]];

        // Assert
        expect({
          compatCache: compatibilityView._compatCache,
          compatInnovationMode: compatibilityView._compatInnovationMode,
          connections: compatibilityView.connections,
          id: compatibilityView._id,
          sourceCompatCache: sourceNetwork._compatCache,
        }).toEqual({
          compatCache: [[1, 2]],
          compatInnovationMode: 'allow-fallback',
          connections: sourceNetwork.connections.map((connection) => ({
            innovation: connection.innovation,
            weight: connection.weight,
          })),
          id: 91,
          sourceCompatCache: [[1, 2]],
        });
      });
    });

    describe('given a strict genome is converted into a cached compatibility view', () => {
      it('forwards strict-genome metadata, cache writes, and connection reads', () => {
        // Arrange
        const genome = createGenomeFromNetwork(new Network(2, 1, { seed: 1_461 })) as typeof createGenomeFromNetwork extends (...args: never[]) => infer T ? T & {
          _compatInnovationMode?: 'require-explicit' | 'allow-fallback';
          _id?: number;
        } : never;
        genome._compatInnovationMode = 'require-explicit';
        genome._id = 92;
        const compatibilityView = createCompatibilityGenomeView(genome);

        compatibilityView._compatCache = [[3, 4]];

        // Assert
        expect({
          compatCache: compatibilityView._compatCache,
          compatInnovationMode: compatibilityView._compatInnovationMode,
          connections: compatibilityView.connections,
          id: compatibilityView._id,
        }).toEqual({
          compatCache: [[3, 4]],
          compatInnovationMode: 'require-explicit',
          connections: genome.connectionGenes.map((connectionGene) => ({
            innovation: connectionGene.innovation,
            weight: connectionGene.weight,
          })),
          id: 92,
        });
      });
    });

    describe('given a connection-only compatibility object is not a runtime source or strict genome', () => {
      it('returns the original compatibility object unchanged', () => {
        // Arrange
        const compatibilitySource = {
          connections: [{ innovation: 7, weight: 0.5 }],
        } as Parameters<typeof createCompatibilityGenomeView>[0];

        // Act
        const compatibilityView = createCompatibilityGenomeView(compatibilitySource);

        // Assert
        expect(compatibilityView).toBe(compatibilitySource);
      });
    });
  });
});