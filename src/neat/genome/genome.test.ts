import Network from '../../architecture/network';
import type { NetworkJSON } from '../../architecture/network/network.types';
import { Architect, methods } from '../../neataptic';
import {
  assertValidGenomeContract,
  createCompatibilityGenomeView,
  createGenomeFromNetwork,
  createNetworkFromGenome,
  validateGenomeContract,
  NeatGenomeConversionError,
  NeatGenomeValidationError,
} from './genome';
import { validateGenomeContract as validateGenomeContractFromValidate } from '../validate/neat.validate';
import type { NeatGenome } from './genome';

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

describe('neat genome chapter', () => {
  describe('createGenomeFromNetwork', () => {
    describe('given one native runtime genome', () => {
      it('preserves stable node-gene identity in contract order', () => {
        // Arrange
        const network = new Network(2, 1, { seed: 1_421 });

        // Act
        const genome = createGenomeFromNetwork(network);

        // Assert
        expect(genome.nodeGenes.map((nodeGene) => nodeGene.geneId)).toEqual(
          network.nodes.map((node) => node.geneId),
        );
      });
    });

    describe('given one runtime genome whose output node order drifted from the strict contract', () => {
      it('canonicalizes input-hidden-output order during conversion', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_425 });
        sourceNetwork.mutate(methods.mutation.ADD_NODE);
        const inputNodes = sourceNetwork.nodes.filter(
          (node) => node.type === 'input',
        );
        const hiddenNodes = sourceNetwork.nodes.filter(
          (node) => node.type === 'hidden',
        );
        const outputNodes = sourceNetwork.nodes.filter(
          (node) => node.type === 'output',
        );

        sourceNetwork.nodes = [...inputNodes, ...outputNodes, ...hiddenNodes];
        sourceNetwork.nodes.forEach((node, nodeIndex) => {
          node.index = nodeIndex;
        });

        // Act
        const genome = createGenomeFromNetwork(sourceNetwork);

        // Assert
        expect(genome.nodeGenes.map((nodeGene) => nodeGene.type)).toEqual([
          ...inputNodes.map(() => 'input'),
          ...hiddenNodes.map(() => 'hidden'),
          ...outputNodes.map(() => 'output'),
        ]);
      });
    });

    describe('given one runtime genome carries a non-neutral connection gain', () => {
      it('keeps the extension bag empty until gain capture is requested', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_426 });
        sourceNetwork.connections[0].gain = 1.5;

        // Act
        const genome = createGenomeFromNetwork(sourceNetwork);

        // Assert
        expect(genome.extensions).toBeUndefined();
      });

      it('captures the gain in the extension bag when the option is enabled', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_427 });
        sourceNetwork.connections[0].gain = 1.5;

        // Act
        const genome = createGenomeFromNetwork(sourceNetwork, {
          connectionGain: true,
        });

        // Assert
        expect(genome.extensions).toEqual({
          version: 1,
          values: {
            connectionGainByInnovation: {
              [String(sourceNetwork.connections[0].innovation)]: 1.5,
            },
          },
        });
      });
    });

    describe('given one runtime genome carries non-neutral response and re-enable semantics', () => {
      it('keeps the extension bag empty until capture is requested', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_430 }) as Network & {
          _reenableProb?: number;
        };
        sourceNetwork.nodes.at(-1)!.response = 1.5;
        sourceNetwork._reenableProb = 0.6;

        // Act
        const genome = createGenomeFromNetwork(sourceNetwork);

        // Assert
        expect(genome.extensions).toBeUndefined();
      });

      it('captures both traits in the extension bag when the options are enabled', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_431 }) as Network & {
          _reenableProb?: number;
        };
        const outputNode = sourceNetwork.nodes.at(-1);
        if (!outputNode) {
          throw new Error('Expected an output node for response capture.');
        }

        outputNode.response = 1.5;
        sourceNetwork._reenableProb = 0.6;

        // Act
        const genome = createGenomeFromNetwork(sourceNetwork, {
          nodeResponse: true,
          disabledConnectionReenableProbability: true,
        });

        // Assert
        expect(genome.extensions).toEqual({
          version: 1,
          values: {
            nodeResponseByGeneId: {
              [String(outputNode.geneId)]: 1.5,
            },
            disabledConnectionReenableProbability: 0.6,
          },
        });
      });
    });

    describe('given one runtime genome carries a non-default activation function', () => {
      it('stores the activation as canonical node-gene state instead of an extension', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_434 });
        const outputNode = sourceNetwork.nodes.at(-1);
        if (!outputNode) {
          throw new Error('Expected an output node for activation capture.');
        }

        outputNode.squash = methods.Activation.tanh;
        const sourcePayload = sourceNetwork.toJSON() as unknown as NetworkJSON;

        // Act
        const genome = createGenomeFromNetwork(sourceNetwork);

        // Assert
        expect({
          extensions: genome.extensions,
          squash: genome.nodeGenes.at(-1)?.squash ?? null,
        }).toEqual({
          extensions: undefined,
          squash: sourcePayload.nodes.at(-1)?.squash ?? null,
        });
      });
    });

    describe('given one runtime genome already carries temporal module descriptors in JSON extensions', () => {
      it('preserves the explicit extension bag during strict-genome capture', () => {
        // Arrange
        const sourcePayload = Architect.lstm(
          1,
          2,
          1,
        ).toJSON() as unknown as NetworkJSON;
        sourcePayload.extensions =
          createTemporalModuleExtensions(sourcePayload);
        const sourceNetwork = Network.fromJSON(
          sourcePayload as unknown as Record<string, unknown>,
        );

        // Act
        const genome = createGenomeFromNetwork(sourceNetwork);

        // Assert
        expect(genome.extensions).toEqual(sourcePayload.extensions);
      });
    });
  });

  describe('createNetworkFromGenome', () => {
    describe('given one non-trivial strict genome contract', () => {
      it('round-trips the structural network payload through the hard adapters', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_422 });
        sourceNetwork.mutate(methods.mutation.ADD_NODE);
        sourceNetwork.mutate(methods.mutation.ADD_CONN);
        const sourcePayload = sourceNetwork.toJSON() as unknown as NetworkJSON;
        const genome = createGenomeFromNetwork(sourceNetwork);

        // Act
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          dropout: sourcePayload.dropout,
          architecture: sourcePayload.architecture,
        });

        // Assert
        expect(rebuiltNetwork.toJSON()).toEqual(sourcePayload);
      });
    });

    describe('given one strict genome carries a connection-gain extension', () => {
      it('materializes the non-neutral gain onto the rebuilt runtime connection', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_428 });
        sourceNetwork.connections[0].gain = 1.5;
        const genome = createGenomeFromNetwork(sourceNetwork, {
          connectionGain: true,
        });

        // Act
        const rebuiltNetwork = createNetworkFromGenome(genome);

        // Assert
        expect(rebuiltNetwork.connections[0].gain).toBe(1.5);
      });
    });

    describe('given one strict genome carries response and re-enable extensions', () => {
      it('materializes both traits onto the rebuilt runtime genome', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_432 }) as Network & {
          _reenableProb?: number;
        };
        const outputNode = sourceNetwork.nodes.at(-1);
        if (!outputNode) {
          throw new Error(
            'Expected an output node for extension materialization.',
          );
        }

        outputNode.response = 1.5;
        sourceNetwork._reenableProb = 0.6;
        const genome = createGenomeFromNetwork(sourceNetwork, {
          nodeResponse: true,
          disabledConnectionReenableProbability: true,
        });

        // Act
        const rebuiltNetwork = createNetworkFromGenome(genome) as Network & {
          _reenableProb?: number;
        };

        // Assert
        expect({
          response: rebuiltNetwork.nodes.at(-1)?.response ?? null,
          reenableProb: rebuiltNetwork._reenableProb,
        }).toEqual({
          response: 1.5,
          reenableProb: 0.6,
        });
      });
    });

    describe('given one strict genome carries a canonical activation mutation', () => {
      it('materializes the stored squash function without using extension state', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_435 });
        const outputNode = sourceNetwork.nodes.at(-1);
        if (!outputNode) {
          throw new Error(
            'Expected an output node for activation materialization.',
          );
        }

        outputNode.squash = methods.Activation.tanh;
        const genome = createGenomeFromNetwork(sourceNetwork);

        // Act
        const rebuiltNetwork = createNetworkFromGenome(genome);

        // Assert
        expect({
          extensions: genome.extensions,
          squash: rebuiltNetwork.nodes.at(-1)?.squash ?? null,
        }).toEqual({
          extensions: undefined,
          squash: methods.Activation.tanh,
        });
      });
    });

    describe('given one strict genome carries temporal module descriptors in the extension bag', () => {
      it('preserves the descriptors when the runtime genome is rebuilt', () => {
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

        // Act
        const rebuiltNetwork = createNetworkFromGenome(genome);

        // Assert
        expect(
          (rebuiltNetwork.toJSON() as unknown as NetworkJSON).extensions,
        ).toEqual(genome.extensions);
      });

      it('keeps the descriptors when one referenced connection gene is disabled', () => {
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
            'Expected one module-owned connection gene for dormant-state coverage.',
          );
        }

        moduleConnectionGene.enabled = false;

        // Act
        const rebuiltNetwork = createNetworkFromGenome(genome);
        const rebuiltPayload =
          rebuiltNetwork.toJSON() as unknown as NetworkJSON;
        const rebuiltConnection = rebuiltPayload.connections.find(
          (connection) => connection.innovation === moduleConnectionInnovation,
        );

        // Assert
        expect({
          extensions: rebuiltPayload.extensions,
          enabled: rebuiltConnection?.enabled ?? null,
        }).toEqual({
          extensions: genome.extensions,
          enabled: false,
        });
      });
    });
  });

  describe('validateGenomeContract', () => {
    describe('given a strict genome loses one explicit connection innovation id', () => {
      it('reports the missing innovation issue', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_423 });
        const genome = createGenomeFromNetwork(sourceNetwork) as NeatGenome & {
          connectionGenes: Array<
            NeatGenome['connectionGenes'][number] & {
              innovation?: number;
            }
          >;
        };
        Reflect.deleteProperty(genome.connectionGenes[0], 'innovation');

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'missing-connection-innovation',
        );
      });
    });

    describe('given a strict genome assigns a connection-gain extension to a gated gene', () => {
      it('reports the invalid extension issue', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_429 });
        const genome = createGenomeFromNetwork(sourceNetwork);
        genome.connectionGenes[0].gaterGeneId = genome.nodeGenes[0].geneId;
        genome.extensions = {
          version: 1,
          values: {
            connectionGainByInnovation: {
              [String(genome.connectionGenes[0].innovation)]: 1.5,
            },
          },
        };

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'invalid-connection-gain-extension',
        );
      });
    });

    describe('given a strict genome stores extension maps in non-plain containers', () => {
      it('reports both malformed extension map issues', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_439 });
        const genome = createGenomeFromNetwork(sourceNetwork);
        genome.extensions = {
          version: 1,
          values: {
            connectionGainByInnovation: [] as unknown as Record<string, number>,
            nodeResponseByGeneId: [] as unknown as Record<string, number>,
          },
        };

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toEqual(
          expect.arrayContaining([
            'invalid-connection-gain-extension',
            'invalid-node-response-extension',
          ]),
        );
      });
    });

    describe('given a strict genome carries malformed temporal module descriptors', () => {
      it('reports both temporal extension issues', () => {
        // Arrange
        const sourcePayload = Architect.lstm(
          1,
          2,
          1,
        ).toJSON() as unknown as NetworkJSON;
        const genome = createGenomeFromNetwork(
          Network.fromJSON(sourcePayload as unknown as Record<string, unknown>),
        );
        genome.extensions = {
          version: 1,
          values: {
            recurrentModules: [
              {
                moduleId: 'module:lstm:0',
                kind: 'lstm',
                nodeGeneIdsByRole: {
                  recurrentCore: [999_999],
                },
                connectionInnovations: [genome.connectionGenes[0].innovation],
              },
            ],
            gatedBlocks: [
              {
                blockId: 'gated:block:0',
                gaterGeneIds: [999_999],
                connectionInnovations: [genome.connectionGenes[0].innovation],
              },
            ],
          },
        };

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toEqual(
          expect.arrayContaining([
            'invalid-recurrent-module-extension',
            'invalid-gated-block-extension',
          ]),
        );
      });
    });

    describe('given a strict genome carries malformed response and re-enable extensions', () => {
      it('reports both extension issues', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_433 });
        const genome = createGenomeFromNetwork(sourceNetwork);
        genome.extensions = {
          version: 1,
          values: {
            nodeResponseByGeneId: {
              9999: 1.5,
            },
            disabledConnectionReenableProbability: Number.NaN,
          },
        };

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toEqual(
          expect.arrayContaining([
            'invalid-node-response-extension',
            'invalid-connection-reenable-extension',
          ]),
        );
      });
    });
  });

  describe('assertValidGenomeContract', () => {
    describe('given one malformed strict genome contract', () => {
      it('throws the first validator issue with its path', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_436 });
        const genome = createGenomeFromNetwork(sourceNetwork) as NeatGenome & {
          connectionGenes: Array<
            NeatGenome['connectionGenes'][number] & {
              innovation?: number;
            }
          >;
        };
        Reflect.deleteProperty(genome.connectionGenes[0], 'innovation');

        // Assert
        expect(() => assertValidGenomeContract(genome)).toThrow(
          'Strict genomes must assign a finite innovation id to every connection gene. (connectionGenes[0].innovation)',
        );
      });
    });
  });

  describe('createCompatibilityGenomeView', () => {
    describe('given one strict genome contract is normalized twice', () => {
      it('reuses the cached compatibility view object', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_437 });
        const genome = createGenomeFromNetwork(sourceNetwork);
        const firstView = createCompatibilityGenomeView(genome);

        // Act
        const secondView = createCompatibilityGenomeView(genome);

        // Assert
        expect(secondView).toBe(firstView);
      });
    });

    describe('given one runtime genome is normalized twice', () => {
      it('reuses the cached runtime compatibility view object', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_438 });
        const firstView = createCompatibilityGenomeView(sourceNetwork);

        // Act
        const secondView = createCompatibilityGenomeView(sourceNetwork);

        // Assert
        expect(secondView).toBe(firstView);
      });
    });
  });

  describe('validate chapter pure genome seam', () => {
    describe('given one valid strict genome contract', () => {
      it('accepts the contract without requiring a live runtime phenotype', () => {
        // Arrange
        const sourceNetwork = new Network(2, 1, { seed: 1_424 });
        const genome = createGenomeFromNetwork(sourceNetwork);

        // Act
        const validationReport = validateGenomeContractFromValidate(genome);

        // Assert
        expect(validationReport.isValid).toBe(true);
      });
    });
  });

  describe('error class barrel re-exports', () => {
    describe('given NeatGenomeConversionError is imported through the genome barrel', () => {
      describe('when instantiated', () => {
        it('is an instance of Error', () => {
          expect(new NeatGenomeConversionError('test')).toBeInstanceOf(Error);
        });
      });
    });

    describe('given NeatGenomeValidationError is imported through the genome barrel', () => {
      describe('when instantiated', () => {
        it('is an instance of Error', () => {
          expect(new NeatGenomeValidationError('test', [])).toBeInstanceOf(
            Error,
          );
        });
      });
    });
  });
});
