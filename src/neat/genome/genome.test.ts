import Network from '../../architecture/network';
import type { NetworkJSON } from '../../architecture/network/network.types';
import { Architect, methods } from '../../neataptic';
import {
  NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE,
  NEAT_GENOME_EPISODIC_SLOT_EVICTION_POLICY_CATALOGUE,
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

type MaterializedNgePrimitiveModuleLike = {
  activate: (
    inputValues: number[],
    coordinates?: [number, number, number],
  ) => number[];
  activationThreshold?: number;
  archetypeId: string;
  broadcastRadius?: number;
  candidateZone?: string;
  clear?: () => void;
  computationType: string;
  costExempt?: boolean;
  decayRate?: number;
  evictionPolicy?: string;
  gatingMode?: 'topK' | 'threshold';
  heads?: number;
  hiddenDim?: number;
  inputSourceSpec?: {
    dimensionality: number;
  };
  isWithinBroadcastRadius?: (
    targetCoordinates: [number, number, number],
  ) => boolean;
  occupiedSlotCount?: number;
  outputDimensionality?: number;
  outputWidth?: number;
  parameterSchema?: Record<string, unknown>;
  position?: [number, number, number];
  receivesCoordinates?: boolean;
  retrieveSlot?: (
    queryValues: number[],
    coordinates?: [number, number, number],
  ) => number[] | null;
  residualStreamId?: string;
  slotCount?: number;
  slotStorage?: Float32Array;
  slotWidth?: number;
  topK?: number;
  weightSharedCohortId?: string;
  writeSlot?: (
    activationValues: number[],
    coordinates?: [number, number, number],
  ) => boolean;
};

type RuntimeNetworkWithNgePrimitiveModules = Network & {
  _ngePrimitiveModules?: MaterializedNgePrimitiveModuleLike[];
};

function createComputationMotifExtensions(
  overrides: Partial<NonNullable<NeatGenome['extensions']>['values']> = {},
): NonNullable<NeatGenome['extensions']> {
  return {
    version: 1,
    values: {
      moduleArchetypes: [
        {
          archetypeId: 'archetype:attention:0',
          computationType: 'AttentionHead',
          parameterSchema: {
            heads: 1,
            outputWidth: 5,
          },
          receivesCoordinates: true,
        },
        {
          archetypeId: 'archetype:recurrent:0',
          computationType: 'GatedRecurrentCell',
          parameterSchema: {
            decayRate: 1,
            hiddenDim: 1,
          },
        },
        {
          archetypeId: 'archetype:episodic:0',
          computationType: 'EpisodicSlot',
          parameterSchema: {
            slotCount: 2,
            evictionPolicy: 'lru',
          },
        },
        {
          archetypeId: 'archetype:dense:0',
          computationType: 'DenseFeedForward',
        },
      ],
      ...overrides,
    },
  };
}

function createAttentionHeadArchetype(): NonNullable<
  NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
>[number] {
  return createComputationMotifExtensions().values
    .moduleArchetypes?.[0] as NonNullable<
    NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
  >[number];
}

function createGatedRecurrentCellArchetype(): NonNullable<
  NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
>[number] {
  return createComputationMotifExtensions().values
    .moduleArchetypes?.[1] as NonNullable<
    NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
  >[number];
}

function createEpisodicSlotArchetype(): NonNullable<
  NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
>[number] {
  return createComputationMotifExtensions().values
    .moduleArchetypes?.[2] as NonNullable<
    NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
  >[number];
}

function createModulatorBroadcasterArchetype(): NonNullable<
  NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
>[number] {
  return {
    archetypeId: 'archetype:modulator:0',
    computationType: 'ModulatorBroadcaster',
    position: [1, -2, 0.5],
    broadcastRadius: 1.75,
    inputSourceSpec: {
      dimensionality: 2,
    },
    outputDimensionality: 2,
  } as NonNullable<
    NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
  >[number];
}

function createGatingRouterArchetype(): NonNullable<
  NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
>[number] {
  return {
    archetypeId: 'archetype:router:0',
    computationType: 'GatingRouter',
    candidateZone: 'zone:basal-ganglia:0',
    topK: 2,
  } as NonNullable<
    NonNullable<NeatGenome['extensions']>['values']['moduleArchetypes']
  >[number];
}

function roundNumericArray(values: number[]): number[] {
  return values.map((value) => Number(value.toFixed(6)));
}

describe('neat genome chapter', () => {
  describe('public barrel exports', () => {
    it('re-exports the public NGE catalogue constants from the chapter barrel', () => {
      // Assert
      expect({
        computationTypes: NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE,
        evictionPolicies: NEAT_GENOME_EPISODIC_SLOT_EVICTION_POLICY_CATALOGUE,
      }).toEqual({
        computationTypes: [
          'DenseFeedForward',
          'AttentionHead',
          'GatedRecurrentCell',
          'EpisodicSlot',
          'ModulatorBroadcaster',
          'GatingRouter',
        ],
        evictionPolicies: ['lru', 'fifo'],
      });
    });
  });

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

    describe('given one strict genome carries NGE module archetypes but motif materialization is disabled', () => {
      it('does not attach opt-in primitive modules to the rebuilt runtime genome', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_465 }),
        );
        genome.extensions = createComputationMotifExtensions();

        // Act
        const rebuiltNetwork = createNetworkFromGenome(
          genome,
        ) as RuntimeNetworkWithNgePrimitiveModules;

        // Assert
        expect(rebuiltNetwork._ngePrimitiveModules).toBeUndefined();
      });

      it('attaches an empty primitive shelf when the NGE lane is enabled without module archetypes', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_465_5 }),
        );

        // Act
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;

        // Assert
        expect(rebuiltNetwork._ngePrimitiveModules).toEqual([]);
      });

      it('materializes only the implemented primitive module types when the NGE lane is enabled', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_466 }),
        );
        genome.extensions = createComputationMotifExtensions();

        // Act
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;

        // Assert
        expect(
          rebuiltNetwork._ngePrimitiveModules?.map((primitiveModule) => ({
            archetypeId: primitiveModule.archetypeId,
            computationType: primitiveModule.computationType,
            decayRate: primitiveModule.decayRate,
            evictionPolicy: primitiveModule.evictionPolicy,
            hasClear: typeof primitiveModule.clear === 'function',
            heads: primitiveModule.heads,
            hiddenDim: primitiveModule.hiddenDim,
            occupiedSlotCount: primitiveModule.occupiedSlotCount,
            outputWidth: primitiveModule.outputWidth,
            receivesCoordinates: primitiveModule.receivesCoordinates,
            slotCount: primitiveModule.slotCount,
            slotStorageLength: primitiveModule.slotStorage?.length,
            slotWidth: primitiveModule.slotWidth,
          })),
        ).toEqual([
          {
            archetypeId: 'archetype:attention:0',
            computationType: 'AttentionHead',
            decayRate: undefined,
            evictionPolicy: undefined,
            hasClear: false,
            heads: 1,
            hiddenDim: undefined,
            occupiedSlotCount: undefined,
            outputWidth: 5,
            receivesCoordinates: true,
            slotCount: undefined,
            slotStorageLength: undefined,
            slotWidth: undefined,
          },
          {
            archetypeId: 'archetype:recurrent:0',
            computationType: 'GatedRecurrentCell',
            decayRate: 1,
            evictionPolicy: undefined,
            hasClear: true,
            heads: undefined,
            hiddenDim: 1,
            occupiedSlotCount: undefined,
            outputWidth: undefined,
            receivesCoordinates: false,
            slotCount: undefined,
            slotStorageLength: undefined,
            slotWidth: undefined,
          },
          {
            archetypeId: 'archetype:episodic:0',
            computationType: 'EpisodicSlot',
            decayRate: undefined,
            evictionPolicy: 'lru',
            hasClear: false,
            heads: undefined,
            hiddenDim: undefined,
            occupiedSlotCount: 0,
            outputWidth: undefined,
            receivesCoordinates: false,
            slotCount: 2,
            slotStorageLength: 0,
            slotWidth: 0,
          },
        ]);
      });

      it('routes attention activations differently for different inputs without collapsing to a uniform pattern', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_467 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createAttentionHeadArchetype()],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const attentionModule = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (!attentionModule) {
          throw new Error(
            'Expected one materialized attention primitive module.',
          );
        }

        // Act
        const firstRouting = roundNumericArray(
          attentionModule.activate([2, 0], [0, 0, 0]),
        );
        const secondRouting = roundNumericArray(
          attentionModule.activate([0, 2], [0, 0, 0]),
        );

        // Assert
        expect({
          firstIsNonUniform: new Set(firstRouting).size > 1,
          firstRouting,
          secondIsNonUniform: new Set(secondRouting).size > 1,
          secondRouting,
        }).toEqual({
          firstIsNonUniform: true,
          firstRouting: [1.297571, 0, 0, 0, 0],
          secondIsNonUniform: true,
          secondRouting: [0, 1.297571, 0, 0, 0],
        });
      });

      it('uses default primitive parameters and preserves governance ids when archetypes omit optional schema fields', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_467_5 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [
            {
              archetypeId: 'archetype:attention:defaults',
              computationType: 'AttentionHead',
              receivesCoordinates: true,
              residualStreamId: 'stream:attention',
              weightSharedCohortId: 'cohort:attention',
            },
            {
              archetypeId: 'archetype:recurrent:defaults',
              computationType: 'GatedRecurrentCell',
              residualStreamId: 'stream:recurrent',
              weightSharedCohortId: 'cohort:recurrent',
            },
          ],
          residualStreams: [
            {
              streamId: 'stream:attention',
              width: 1,
            },
            {
              streamId: 'stream:recurrent',
              width: 1,
            },
          ],
          weightSharedCohorts: [
            {
              cohortId: 'cohort:attention',
            },
            {
              cohortId: 'cohort:recurrent',
            },
          ],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const attentionModule = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (!attentionModule) {
          throw new Error(
            'Expected one default-configured attention primitive module.',
          );
        }

        // Assert
        expect({
          attentionDefaultRouting: roundNumericArray(
            attentionModule.activate([2, 0]),
          ),
          modules: rebuiltNetwork._ngePrimitiveModules?.map(
            (primitiveModule) => ({
              archetypeId: primitiveModule.archetypeId,
              computationType: primitiveModule.computationType,
              decayRate: primitiveModule.decayRate,
              heads: primitiveModule.heads,
              hiddenDim: primitiveModule.hiddenDim,
              outputWidth: primitiveModule.outputWidth,
              parameterSchema: primitiveModule.parameterSchema,
              receivesCoordinates: primitiveModule.receivesCoordinates,
              residualStreamId: primitiveModule.residualStreamId,
              weightSharedCohortId: primitiveModule.weightSharedCohortId,
            }),
          ),
        }).toEqual({
          attentionDefaultRouting: [1.761594],
          modules: [
            {
              archetypeId: 'archetype:attention:defaults',
              computationType: 'AttentionHead',
              decayRate: undefined,
              heads: 1,
              hiddenDim: undefined,
              outputWidth: 1,
              parameterSchema: {},
              receivesCoordinates: true,
              residualStreamId: 'stream:attention',
              weightSharedCohortId: 'cohort:attention',
            },
            {
              archetypeId: 'archetype:recurrent:defaults',
              computationType: 'GatedRecurrentCell',
              decayRate: 0.5,
              heads: undefined,
              hiddenDim: 1,
              outputWidth: undefined,
              parameterSchema: {},
              receivesCoordinates: false,
              residualStreamId: 'stream:recurrent',
              weightSharedCohortId: 'cohort:recurrent',
            },
          ],
        });
      });

      it('keeps gated recurrent state within an episode and resets it after clear', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createGatedRecurrentCellArchetype()],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const gatedRecurrentCell = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !gatedRecurrentCell ||
          typeof gatedRecurrentCell.clear !== 'function'
        ) {
          throw new Error(
            'Expected one materialized gated recurrent primitive module.',
          );
        }

        // Act
        const firstActivation = roundNumericArray(
          gatedRecurrentCell.activate([1]),
        );
        const secondActivation = roundNumericArray(
          gatedRecurrentCell.activate([0]),
        );
        gatedRecurrentCell.clear();
        const resetActivation = roundNumericArray(
          gatedRecurrentCell.activate([0]),
        );

        // Assert
        expect({
          firstActivation,
          preservedStateAfterZeroInput: secondActivation[0] > 0,
          resetActivation,
        }).toEqual({
          firstActivation: [0.642015],
          preservedStateAfterZeroInput: true,
          resetActivation: [0],
        });
      });

      it('initializes episodic slot storage as an empty float32 shelf with default LRU eviction', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_5 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [
            {
              archetypeId: 'archetype:episodic:defaults',
              computationType: 'EpisodicSlot',
            },
          ],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const episodicSlot = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !episodicSlot ||
          typeof episodicSlot.retrieveSlot !== 'function' ||
          typeof episodicSlot.writeSlot !== 'function' ||
          !(episodicSlot.slotStorage instanceof Float32Array)
        ) {
          throw new Error(
            'Expected one materialized episodic-slot primitive module.',
          );
        }

        // Assert
        expect({
          computationType: episodicSlot.computationType,
          emptyActivation: episodicSlot.activate([1, 0]),
          emptyRetrieve: episodicSlot.retrieveSlot([1, 0]),
          emptyWrite: episodicSlot.writeSlot([]),
          evictionPolicy: episodicSlot.evictionPolicy,
          occupiedSlotCount: episodicSlot.occupiedSlotCount,
          slotCount: episodicSlot.slotCount,
          slotStorageLength: episodicSlot.slotStorage.length,
          slotStorageType: episodicSlot.slotStorage.constructor.name,
          slotWidth: episodicSlot.slotWidth,
        }).toEqual({
          computationType: 'EpisodicSlot',
          emptyActivation: [],
          emptyRetrieve: null,
          emptyWrite: false,
          evictionPolicy: 'lru',
          occupiedSlotCount: 0,
          slotCount: 1,
          slotStorageLength: 0,
          slotStorageType: 'Float32Array',
          slotWidth: 0,
        });
      });

      it('writes only novel activations and grows until the configured slot cap', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_6 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createEpisodicSlotArchetype()],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const episodicSlot = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !episodicSlot ||
          typeof episodicSlot.writeSlot !== 'function' ||
          !(episodicSlot.slotStorage instanceof Float32Array)
        ) {
          throw new Error(
            'Expected one materialized episodic-slot primitive module.',
          );
        }

        // Act
        const firstWrite = episodicSlot.writeSlot([1, 0]);
        const duplicateWrite = episodicSlot.writeSlot([1, 0]);
        const secondNovelWrite = episodicSlot.writeSlot([0, 1]);

        // Assert
        expect({
          duplicateWrite,
          firstWrite,
          occupiedSlotCount: episodicSlot.occupiedSlotCount,
          secondNovelWrite,
          slotStorage: Array.from(episodicSlot.slotStorage),
          slotWidth: episodicSlot.slotWidth,
        }).toEqual({
          duplicateWrite: false,
          firstWrite: true,
          occupiedSlotCount: 2,
          secondNovelWrite: true,
          slotStorage: [1, 0, 0, 1],
          slotWidth: 2,
        });
      });

      it('treats non-zero activations as novel when the best stored slot is still all zeros', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_65 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createEpisodicSlotArchetype()],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const episodicSlot = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !episodicSlot ||
          typeof episodicSlot.writeSlot !== 'function' ||
          !(episodicSlot.slotStorage instanceof Float32Array)
        ) {
          throw new Error(
            'Expected one materialized episodic-slot primitive module.',
          );
        }

        // Act
        const firstZeroWrite = episodicSlot.writeSlot([0, 0]);
        const duplicateZeroWrite = episodicSlot.writeSlot([0, 0]);
        const novelNonZeroWrite = episodicSlot.writeSlot([1, 0]);

        // Assert
        expect({
          duplicateZeroWrite,
          firstZeroWrite,
          novelNonZeroWrite,
          occupiedSlotCount: episodicSlot.occupiedSlotCount,
          slotStorage: Array.from(episodicSlot.slotStorage),
        }).toEqual({
          duplicateZeroWrite: false,
          firstZeroWrite: true,
          novelNonZeroWrite: true,
          occupiedSlotCount: 2,
          slotStorage: [0, 0, 1, 0],
        });
      });

      it('retrieves the argmax dot-product slot match from stored episodic activations', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_7 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [
            {
              ...createEpisodicSlotArchetype(),
              parameterSchema: {
                slotCount: 3,
                evictionPolicy: 'lru',
              },
            },
          ],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const episodicSlot = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !episodicSlot ||
          typeof episodicSlot.retrieveSlot !== 'function' ||
          typeof episodicSlot.writeSlot !== 'function'
        ) {
          throw new Error(
            'Expected one materialized episodic-slot primitive module.',
          );
        }

        episodicSlot.writeSlot([1, 0]);
        episodicSlot.writeSlot([0, 2]);
        episodicSlot.writeSlot([1, 1]);

        // Assert
        expect(episodicSlot.retrieveSlot([0, 1.5])).toEqual([0, 2]);
      });

      it('breaks equal dot-product retrieval ties by choosing the higher cosine-similarity slot', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_75 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createEpisodicSlotArchetype()],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const episodicSlot = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !episodicSlot ||
          typeof episodicSlot.retrieveSlot !== 'function' ||
          typeof episodicSlot.writeSlot !== 'function'
        ) {
          throw new Error(
            'Expected one materialized episodic-slot primitive module.',
          );
        }

        episodicSlot.writeSlot([1, 1]);
        episodicSlot.writeSlot([1, 0]);

        // Assert
        expect(episodicSlot.retrieveSlot([1, 0])).toEqual([1, 0]);
      });

      it('uses default episodic parameters and preserves governance ids when the archetype omits schema fields', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_76 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [
            {
              archetypeId: 'archetype:episodic:defaults',
              computationType: 'EpisodicSlot',
              residualStreamId: 'stream:episodic',
              weightSharedCohortId: 'cohort:episodic',
            },
          ],
          residualStreams: [
            {
              streamId: 'stream:episodic',
              width: 1,
            },
          ],
          weightSharedCohorts: [
            {
              cohortId: 'cohort:episodic',
            },
          ],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const episodicSlot = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !episodicSlot ||
          !(episodicSlot.slotStorage instanceof Float32Array)
        ) {
          throw new Error(
            'Expected one materialized episodic-slot primitive module.',
          );
        }

        // Assert
        expect({
          computationType: episodicSlot.computationType,
          evictionPolicy: episodicSlot.evictionPolicy,
          parameterSchema: episodicSlot.parameterSchema,
          residualStreamId: episodicSlot.residualStreamId,
          slotCount: episodicSlot.slotCount,
          slotStorageLength: episodicSlot.slotStorage.length,
          weightSharedCohortId: episodicSlot.weightSharedCohortId,
        }).toEqual({
          computationType: 'EpisodicSlot',
          evictionPolicy: 'lru',
          parameterSchema: {},
          residualStreamId: 'stream:episodic',
          slotCount: 1,
          slotStorageLength: 0,
          weightSharedCohortId: 'cohort:episodic',
        });
      });

      it('materializes broadcaster governance fields and deterministic modulation when NGE is enabled', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_8 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createModulatorBroadcasterArchetype()],
        });
        const firstNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const secondNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const firstBroadcaster = firstNetwork._ngePrimitiveModules?.[0];
        const secondBroadcaster = secondNetwork._ngePrimitiveModules?.[0];

        if (
          !firstBroadcaster ||
          !secondBroadcaster ||
          typeof firstBroadcaster.isWithinBroadcastRadius !== 'function' ||
          typeof secondBroadcaster.isWithinBroadcastRadius !== 'function'
        ) {
          throw new Error(
            'Expected deterministic ModulatorBroadcaster primitive materialization.',
          );
        }

        // Act
        const firstVector = roundNumericArray(
          firstBroadcaster.activate([2, -1]),
        );
        const secondVector = roundNumericArray(
          secondBroadcaster.activate([2, -1]),
        );

        // Assert
        expect({
          archetypeId: firstBroadcaster.archetypeId,
          broadcastRadius: firstBroadcaster.broadcastRadius,
          costExempt: firstBroadcaster.costExempt,
          inRadius: firstBroadcaster.isWithinBroadcastRadius([1.5, -1, 0.5]),
          inputDimensionality: firstBroadcaster.inputSourceSpec?.dimensionality,
          modulationVectorLength: firstVector.length,
          nonZeroVector: firstVector.some((value) => value !== 0),
          outOfRadius: firstBroadcaster.isWithinBroadcastRadius([4, -2, 0.5]),
          outputDimensionality: firstBroadcaster.outputDimensionality,
          position: firstBroadcaster.position,
          repeatVectorMatches:
            JSON.stringify(firstVector) === JSON.stringify(secondVector),
        }).toEqual({
          archetypeId: 'archetype:modulator:0',
          broadcastRadius: 1.75,
          costExempt: true,
          inRadius: true,
          inputDimensionality: 2,
          modulationVectorLength: 2,
          nonZeroVector: true,
          outOfRadius: false,
          outputDimensionality: 2,
          position: [1, -2, 0.5],
          repeatVectorMatches: true,
        });
      });

      it('does not materialize broadcaster primitives when the NGE lane is disabled', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_81 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createModulatorBroadcasterArchetype()],
        });

        // Act
        const rebuiltNetwork = createNetworkFromGenome(
          genome,
        ) as RuntimeNetworkWithNgePrimitiveModules;

        // Assert
        expect(rebuiltNetwork._ngePrimitiveModules).toBeUndefined();
      });

      it('pads broadcaster inputs with zeros and preserves governance ids when declared', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_82 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [
            {
              ...createModulatorBroadcasterArchetype(),
              broadcastRadius: 0,
              inputSourceSpec: {
                dimensionality: 3,
              },
              outputDimensionality: 3,
              position: [0, 0, 0],
              residualStreamId: 'stream:modulator',
              weightSharedCohortId: 'cohort:modulator',
            } as NonNullable<
              NonNullable<
                NeatGenome['extensions']
              >['values']['moduleArchetypes']
            >[number],
          ],
          residualStreams: [
            {
              streamId: 'stream:modulator',
              width: 1,
            },
          ],
          weightSharedCohorts: [
            {
              cohortId: 'cohort:modulator',
            },
          ],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const broadcaster = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (!broadcaster) {
          throw new Error(
            'Expected one broadcaster primitive module with governance ids.',
          );
        }

        // Assert
        expect({
          activation: roundNumericArray(broadcaster.activate([])),
          inputDimensionality: broadcaster.inputSourceSpec?.dimensionality,
          residualStreamId: broadcaster.residualStreamId,
          weightSharedCohortId: broadcaster.weightSharedCohortId,
        }).toEqual({
          activation: [0, 0, 0],
          inputDimensionality: 3,
          residualStreamId: 'stream:modulator',
          weightSharedCohortId: 'cohort:modulator',
        });
      });

      it('materializes router governance fields, default top-k mode, and deterministic routing when NGE is enabled', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_83 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createGatingRouterArchetype()],
        });
        const firstNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const secondNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const firstRouter = firstNetwork._ngePrimitiveModules?.[0];
        const secondRouter = secondNetwork._ngePrimitiveModules?.[0];

        if (!firstRouter || !secondRouter) {
          throw new Error(
            'Expected deterministic GatingRouter primitive materialization.',
          );
        }

        // Act
        const firstRouting = roundNumericArray(
          firstRouter.activate([0.1, 0.9, 0.5, 0.8]),
        );
        const secondRouting = roundNumericArray(
          secondRouter.activate([0.1, 0.9, 0.5, 0.8]),
        );

        // Assert
        expect({
          activationThreshold: firstRouter.activationThreshold,
          archetypeId: firstRouter.archetypeId,
          candidateZone: firstRouter.candidateZone,
          gatingMode: firstRouter.gatingMode,
          repeatRoutingMatches:
            JSON.stringify(firstRouting) === JSON.stringify(secondRouting),
          routing: firstRouting,
          topK: firstRouter.topK,
        }).toEqual({
          activationThreshold: 0.5,
          archetypeId: 'archetype:router:0',
          candidateZone: 'zone:basal-ganglia:0',
          gatingMode: 'topK',
          repeatRoutingMatches: true,
          routing: [0, 0.9, 0, 0.8],
          topK: 2,
        });
      });

      it('stores threshold routing governance and filters candidates below the activation floor', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_84 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [
            {
              ...createGatingRouterArchetype(),
              gatingMode: {
                type: 'threshold',
                activationThreshold: 0.6,
              },
              residualStreamId: 'stream:router',
              weightSharedCohortId: 'cohort:router',
            } as NonNullable<
              NonNullable<
                NeatGenome['extensions']
              >['values']['moduleArchetypes']
            >[number],
          ],
          residualStreams: [
            {
              streamId: 'stream:router',
              width: 1,
            },
          ],
          weightSharedCohorts: [
            {
              cohortId: 'cohort:router',
            },
          ],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const gatingRouter = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (!gatingRouter) {
          throw new Error(
            'Expected one threshold-governed GatingRouter primitive module.',
          );
        }

        // Assert
        expect({
          activationThreshold: gatingRouter.activationThreshold,
          gatingMode: gatingRouter.gatingMode,
          residualStreamId: gatingRouter.residualStreamId,
          routing: roundNumericArray(
            gatingRouter.activate([0.59, 0.6, 0.95, 0.2]),
          ),
          weightSharedCohortId: gatingRouter.weightSharedCohortId,
        }).toEqual({
          activationThreshold: 0.6,
          gatingMode: 'threshold',
          residualStreamId: 'stream:router',
          routing: [0, 0.6, 0.95, 0],
          weightSharedCohortId: 'cohort:router',
        });
      });

      it('does not materialize router primitives when the NGE lane is disabled', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_85 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createGatingRouterArchetype()],
        });

        // Act
        const rebuiltNetwork = createNetworkFromGenome(
          genome,
        ) as RuntimeNetworkWithNgePrimitiveModules;

        // Assert
        expect(rebuiltNetwork._ngePrimitiveModules).toBeUndefined();
      });

      it('evicts the least recently used slot when the episodic shelf is full', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_8 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createEpisodicSlotArchetype()],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const episodicSlot = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !episodicSlot ||
          typeof episodicSlot.retrieveSlot !== 'function' ||
          typeof episodicSlot.writeSlot !== 'function' ||
          !(episodicSlot.slotStorage instanceof Float32Array)
        ) {
          throw new Error(
            'Expected one materialized episodic-slot primitive module.',
          );
        }

        episodicSlot.writeSlot([1, 0]);
        episodicSlot.writeSlot([0, 1]);
        episodicSlot.retrieveSlot([1, 0]);
        episodicSlot.writeSlot([2, 2]);

        // Assert
        expect({
          occupiedSlotCount: episodicSlot.occupiedSlotCount,
          retrievedNewest: episodicSlot.retrieveSlot([2, 2]),
          slotStorage: Array.from(episodicSlot.slotStorage),
        }).toEqual({
          occupiedSlotCount: 2,
          retrievedNewest: [2, 2],
          slotStorage: [1, 0, 2, 2],
        });
      });

      it('evicts the earliest written slot when FIFO episodic eviction is selected', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_9 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [
            {
              ...createEpisodicSlotArchetype(),
              parameterSchema: {
                slotCount: 2,
                evictionPolicy: 'fifo',
              },
            },
          ],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const episodicSlot = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !episodicSlot ||
          typeof episodicSlot.retrieveSlot !== 'function' ||
          typeof episodicSlot.writeSlot !== 'function' ||
          !(episodicSlot.slotStorage instanceof Float32Array)
        ) {
          throw new Error(
            'Expected one materialized episodic-slot primitive module.',
          );
        }

        episodicSlot.writeSlot([1, 0]);
        episodicSlot.writeSlot([0, 1]);
        episodicSlot.retrieveSlot([1, 0]);
        episodicSlot.writeSlot([2, 2]);

        // Assert
        expect({
          evictionPolicy: episodicSlot.evictionPolicy,
          occupiedSlotCount: episodicSlot.occupiedSlotCount,
          retrievedNewest: episodicSlot.retrieveSlot([2, 2]),
          slotStorage: Array.from(episodicSlot.slotStorage),
        }).toEqual({
          evictionPolicy: 'fifo',
          occupiedSlotCount: 2,
          retrievedNewest: [2, 2],
          slotStorage: [2, 2, 0, 1],
        });
      });

      it('falls back to slot zero if occupancy metadata loses track of a full episodic shelf', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_95 }),
        );
        genome.extensions = createComputationMotifExtensions({
          moduleArchetypes: [createEpisodicSlotArchetype()],
        });
        const rebuiltNetwork = createNetworkFromGenome(genome, {
          ngeEnabled: true,
        }) as RuntimeNetworkWithNgePrimitiveModules;
        const episodicSlot = rebuiltNetwork._ngePrimitiveModules?.[0];

        if (
          !episodicSlot ||
          typeof episodicSlot.writeSlot !== 'function' ||
          !(episodicSlot.slotStorage instanceof Float32Array)
        ) {
          throw new Error(
            'Expected one materialized episodic-slot primitive module.',
          );
        }

        episodicSlot.writeSlot([1, 0]);
        episodicSlot.writeSlot([0, 1]);
        episodicSlot.occupiedSlotCount = 1;
        episodicSlot.writeSlot([2, 2]);

        // Assert
        expect({
          occupiedSlotCount: episodicSlot.occupiedSlotCount,
          slotStorage: Array.from(episodicSlot.slotStorage),
        }).toEqual({
          occupiedSlotCount: 1,
          slotStorage: [2, 2, 0, 1],
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

    describe('given a strict genome carries malformed GatingRouter governance', () => {
      it('reports the invalid module-archetype extension issue', () => {
        // Arrange
        const genome = createGenomeFromNetwork(
          new Network(1, 1, { seed: 1_468_86 }),
        );
        genome.extensions = {
          version: 1,
          values: {
            moduleArchetypes: [
              {
                archetypeId: 'archetype:router:invalid',
                computationType: 'GatingRouter',
                candidateZone: '',
                topK: 0,
                gatingMode: {
                  type: 'threshold',
                },
              } as unknown as NonNullable<
                NonNullable<
                  NeatGenome['extensions']
                >['values']['moduleArchetypes']
              >[number],
            ],
          },
        };

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'invalid-module-archetype-extension',
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

    describe('given a strict genome carries one malformed modulator broadcaster archetype', () => {
      it('reports the invalid module-archetype issue', () => {
        // Arrange
        const sourceNetwork = new Network(1, 1, { seed: 1_439_5 });
        const genome = createGenomeFromNetwork(sourceNetwork);
        genome.extensions = {
          version: 1,
          values: {
            moduleArchetypes: [
              {
                archetypeId: 'archetype:modulator:invalid',
                computationType: 'ModulatorBroadcaster',
              } as NonNullable<
                NonNullable<
                  NeatGenome['extensions']
                >['values']['moduleArchetypes']
              >[number],
            ],
          },
        };

        // Act
        const validationReport = validateGenomeContract(genome);

        // Assert
        expect(collectIssueCodes(validationReport.issues)).toContain(
          'invalid-module-archetype-extension',
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
