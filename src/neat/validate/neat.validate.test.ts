import Node from '../../architecture/node';
import Network from '../../architecture/network';
import { Architect } from '../../neataptic';
import { createGenomeFromNetwork } from '../genome/genome';
import {
  assertValidNativeGenome,
  validateGenomeContract,
  type NativeGenomeValidationIssue,
  validateNativeGenome,
} from './neat.validate';
import { NeatNativeGenomeValidationError } from './neat.validate.errors';
import type { NetworkJSON } from '../../architecture/network/network.types';

type MutableValidationNetwork = Network & {
  _compatCache?: Array<[number, number]>;
  _outputCache?: unknown;
};

function createValidationNetwork(): MutableValidationNetwork {
  return new Network(2, 1, { seed: 919 }) as MutableValidationNetwork;
}

function collectIssueCodes(
  issues: NativeGenomeValidationIssue[],
): string[] {
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

  describe('validateGenomeContract', () => {
    describe('given a strict genome keeps one temporal connection gene disabled', () => {
      it('still accepts the dormant temporal descriptor state', () => {
        // Arrange
        const sourcePayload = Architect.lstm(1, 2, 1)
          .toJSON() as unknown as NetworkJSON;
        const genome = createGenomeFromNetwork(
          Network.fromJSON(sourcePayload as unknown as Record<string, unknown>),
        );
        genome.extensions = createTemporalModuleExtensions(sourcePayload);
        const moduleConnectionInnovation = readFirstTemporalConnectionInnovation(
          genome.extensions,
        );
        const moduleConnectionGene = genome.connectionGenes.find(
          (connectionGene) =>
            connectionGene.innovation === moduleConnectionInnovation,
        );

        if (!moduleConnectionGene) {
          throw new Error('Expected one module-owned connection gene for dormant-state validation.');
        }

        moduleConnectionGene.enabled = false;

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(validationReport.isValid).toBe(true);
      });
    });

    describe('given a strict genome carries a malformed recurrent-module descriptor', () => {
      it('reports the new recurrent-module extension issue through the validate facade', () => {
        // Arrange
        const sourcePayload = Architect.lstm(1, 2, 1)
          .toJSON() as unknown as NetworkJSON;
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