import Node from '../../architecture/node';
import Network from '../../architecture/network';
import { Architect } from '../../neataptic';
import { createGenomeFromNetwork, type NeatGenome } from '../genome/genome';
import {
  NeatNativeGenomeValidationError,
  assertValidGenomeContract,
  assertValidNativeGenome,
  validateGenomeContract,
  type NativeGenomeValidationIssue,
  validateNativeGenome,
} from './neat.validate';
import type { NetworkJSON } from '../../architecture/network/network.types';

type MutableValidationNetwork = Network & {
  _compatCache?: unknown;
  _outputCache?: unknown;
};

function createValidationNetwork(): MutableValidationNetwork {
  return new Network(2, 1, { seed: 919 }) as MutableValidationNetwork;
}

function createCompatibilityCacheEntries(
  genome: MutableValidationNetwork,
): Array<[number, number]> {
  return [...genome.connections, ...genome.selfconns]
    .filter((connection) => Number.isFinite(connection.innovation))
    .map(
      (connection) =>
        [connection.innovation, connection.weight] as [number, number],
    )
    .toSorted(
      ([leftInnovation], [rightInnovation]) => leftInnovation - rightInnovation,
    );
}

function collectIssueCodes(issues: NativeGenomeValidationIssue[]): string[] {
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

function readFirstTemporalConnectionInnovation(
  extensions: NonNullable<NetworkJSON['extensions']>,
): number {
  const extensionValues = extensions.values as {
    gatedBlocks?: Array<{ connectionInnovations: number[] }>;
  };
  const connectionInnovation =
    extensionValues.gatedBlocks?.[0]?.connectionInnovations?.[0];

  if (typeof connectionInnovation !== 'number') {
    throw new Error('Expected one temporal gated-block connection innovation.');
  }

  return connectionInnovation;
}

describe('neat validate chapter', () => {
  describe('validateNativeGenome', () => {
    describe('given a native genome produced by the runtime network boundary', () => {
      it('returns a valid report', () => {
        // Arrange
        const genome = createValidationNetwork();

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(validationReport.isValid).toBe(true);
      });
    });

    describe('given two runtime nodes share the same gene id', () => {
      it('reports the duplicate node identity issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.nodes[1].geneId = genome.nodes[0].geneId;

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toEqual([
          'duplicate-node-gene-id',
        ]);
      });
    });

    describe('given one runtime node is missing its gene id', () => {
      it('reports the missing node identity issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.nodes[0].geneId = Number.NaN;

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'missing-node-gene-id',
        );
      });
    });

    describe('given two runtime connections share the same innovation id', () => {
      it('reports the duplicate connection innovation issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.connections[1].innovation = genome.connections[0].innovation;

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'duplicate-connection-innovation',
        );
      });
    });

    describe('given one runtime connection is missing its innovation id', () => {
      it('reports the missing connection innovation issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.connections[0].innovation = Number.NaN;

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'missing-connection-innovation',
        );
      });
    });

    describe('given one connection endpoint does not expose an integer node index', () => {
      it('reports the broken endpoint resolution issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.setTopologyIntent('unconstrained');
        genome.connections[0].from = {
          geneId: 77,
          index: 0.5,
        } as unknown as Node;

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'endpoint-resolution-failed',
        );
      });
    });

    describe('given one connection endpoint points at a detached runtime node', () => {
      it('reports the detached endpoint resolution issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.setTopologyIntent('unconstrained');
        const detachedNode = new Node('hidden');
        detachedNode.index = 99;
        detachedNode.geneId = 199;
        genome.connections[0].from = detachedNode;

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'endpoint-resolution-failed',
        );
      });
    });

    describe('given a gated connection points at a node outside the genome', () => {
      it('reports the broken gater resolution issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        const orphanGater = new Node('hidden');
        genome.connections[0].gater = orphanGater;
        genome.gates.push(genome.connections[0]);

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'gater-resolution-failed',
        );
      });
    });

    describe('given one gated connection does not expose an integer gater index', () => {
      it('reports the malformed gater resolution issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.connections[0].gater = {
          geneId: 88,
          index: 0.25,
        } as unknown as Node;
        genome.gates.push(genome.connections[0]);

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'gater-resolution-failed',
        );
      });
    });

    describe('given one runtime connection keeps a gater without gate registration', () => {
      it('reports the missing gated-connection registration issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.connections[0].gater = genome.nodes[1];

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'gated-connection-registration-mismatch',
        );
      });
    });

    describe('given one registered gate entry has no attached gater', () => {
      it('reports the gate registry mismatch issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.gates.push(genome.connections[0]);

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'gated-connection-registration-mismatch',
        );
      });
    });

    describe('given a feed-forward genome contains a backward edge', () => {
      it('reports the recurrent-edge violation', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.setTopologyIntent('feed-forward');
        genome.connections[0].from = genome.nodes.at(-1)!;
        genome.connections[0].to = genome.nodes[0];

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'feed-forward-recurrent-connection',
        );
      });
    });

    describe('given the public topology intent and runtime acyclic flag drift apart', () => {
      it('reports the topology intent mismatch issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        const runtimeGenome = genome as unknown as {
          _enforceAcyclic?: boolean;
        };
        genome.setTopologyIntent('feed-forward');
        runtimeGenome._enforceAcyclic = false;

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'topology-intent-mismatch',
        );
      });
    });

    describe('given the genome carries a stale compatibility cache', () => {
      it('reports the compatibility-cache mismatch issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome._compatCache = [[999, 0.25]];

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'compat-cache-mismatch',
        );
      });
    });

    describe('given the compatibility cache contains malformed tuple entries', () => {
      it('reports the malformed compatibility-cache issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome._compatCache = [genome.connections[0].innovation];

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'compat-cache-mismatch',
        );
      });
    });

    describe('given the compatibility cache length matches but one cached weight is stale', () => {
      it('reports the stale compatibility-cache content issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        const compatibilityCacheEntries =
          createCompatibilityCacheEntries(genome);
        genome._compatCache = compatibilityCacheEntries.with(0, [
          compatibilityCacheEntries[0][0],
          compatibilityCacheEntries[0][1] + 1,
        ]);

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'compat-cache-mismatch',
        );
      });
    });

    describe('given the compatibility cache matches the sorted runtime edges', () => {
      it('accepts the compatibility cache state', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome._compatCache = createCompatibilityCacheEntries(genome);

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).not.toContain(
          'compat-cache-mismatch',
        );
      });
    });

    describe('given the genome still carries a derived runtime cache field', () => {
      it('reports the stale derived-cache issue', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome._outputCache = [1, 0];

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'stale-derived-cache',
        );
      });
    });
  });

  describe('assertValidNativeGenome', () => {
    describe('given the genome already satisfies the native validation contract', () => {
      it('returns without throwing', () => {
        // Arrange
        const genome = createValidationNetwork();
        const assertGenome = () => assertValidNativeGenome(genome);

        // Assert
        expect(assertGenome).not.toThrow();
      });
    });

    describe('given the genome violates a native identity invariant', () => {
      it('throws the dedicated validation error', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.connections[1].innovation = genome.connections[0].innovation;
        const assertGenome = () => assertValidNativeGenome(genome);

        // Assert
        expect(assertGenome).toThrow(NeatNativeGenomeValidationError);
      });
    });
  });

  describe('assertValidGenomeContract', () => {
    describe('given a strict genome omits one connection innovation id', () => {
      it('throws the structural contract failure message', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          createValidationNetwork(),
        ) as NeatGenome & {
          connectionGenes: Array<
            NeatGenome['connectionGenes'][number] & {
              innovation?: number;
            }
          >;
        };
        Reflect.deleteProperty(genome.connectionGenes[0], 'innovation');
        const assertGenome = () => assertValidGenomeContract(genome);

        // Assert
        expect(assertGenome).toThrow(
          'Strict genomes must assign a finite innovation id to every connection gene. (connectionGenes[0].innovation)',
        );
      });
    });
  });

  describe('validateGenomeContract', () => {
    describe('given a strict genome keeps one temporal connection gene disabled', () => {
      it('still accepts the dormant temporal descriptor state', () => {
        // Arrange
        const sourcePayload = Architect.lstm(
          1,
          2,
          1,
        ).toJSON() as unknown as NetworkJSON;
        const genome = createGenomeFromNetwork(
          Network.fromJSON(sourcePayload as unknown as Record<string, unknown>),
        );
        genome.extensions = createTemporalModuleExtensions(sourcePayload);
        const moduleConnectionInnovation =
          readFirstTemporalConnectionInnovation(genome.extensions);
        const moduleConnectionGene = genome.connectionGenes.find(
          (connectionGene) =>
            connectionGene.innovation === moduleConnectionInnovation,
        );

        if (!moduleConnectionGene) {
          throw new Error(
            'Expected one module-owned connection gene for dormant-state validation.',
          );
        }

        moduleConnectionGene.enabled = false;

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(validationReport.isValid).toBe(true);
      });
    });

    describe('given a native runtime genome carries one self connection', () => {
      it('includes the self-connection path in the native validation pass', () => {
        // Arrange
        const genome = createValidationNetwork();
        genome.setTopologyIntent('unconstrained');
        const selfConnection = genome.nodes
          .at(-1)!
          .connect(genome.nodes.at(-1)!, 0.25)[0];
        selfConnection.innovation = 10_001;
        genome.selfconns = [selfConnection];

        // Act
        const validationReport = validateNativeGenome(genome);

        // Assert
        expect(validationReport.isValid).toBe(true);
      });
    });

    describe('given a strict genome carries a malformed recurrent-module descriptor', () => {
      it('reports the new recurrent-module extension issue through the validate facade', () => {
        // Arrange
        const sourcePayload = Architect.lstm(
          1,
          2,
          1,
        ).toJSON() as unknown as NetworkJSON;
        const genome = createGenomeFromNetwork(
          Network.fromJSON(sourcePayload as unknown as Record<string, unknown>),
        );
        genome.extensions = createTemporalModuleExtensions(sourcePayload);
        (
          genome.extensions.values.recurrentModules as Array<{
            nodeGeneIdsByRole: Record<string, number[]>;
          }>
        )[0].nodeGeneIdsByRole.recurrentCore = [999_999];

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(validationReport.issues.map((issue) => issue.code)).toContain(
          'invalid-recurrent-module-extension',
        );
      });
    });
  });
});
