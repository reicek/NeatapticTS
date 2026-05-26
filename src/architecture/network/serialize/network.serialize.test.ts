import { Architect, methods } from '../../../neataptic';
import Connection from '../../connection';
import Group from '../../group';
import Layer from '../../layer';
import Node from '../../node';
import { validateNativeGenome } from '../../../neat/validate/neat.validate';
import Network from '../network';
import type { NetworkJSON } from '../network.types';
import {
  createParameterLayoutV1,
  deserializeCompressedArchive,
  deserializeCompressedArchiveAsync,
  deserializeCompressedArchiveAsyncWithMetrics,
  deserializeCompressedArchiveWithMetrics,
  deserializeCompressed,
  fromParameterVector,
  serializeCompressedArchive,
  serializeCompressedArchiveAsync,
  serializeCompressedArchiveAsyncWithMetrics,
  serializeCompressedArchiveWithMetrics,
  serializeCompressed,
  toParameterVector,
} from './network.serialize.utils';

function createSerializableNetwork(seed: number): Network {
  return new Network(2, 1, { seed });
}

function createSingleValueSerializableNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

function outputsMatchWithinTolerance(
  actualOutput: number[],
  expectedOutput: number[],
): boolean {
  return (
    actualOutput.length === expectedOutput.length &&
    actualOutput.every(
      (outputValue, outputIndex) =>
        Math.abs(outputValue - expectedOutput[outputIndex]!) <= Number.EPSILON,
    )
  );
}

type ConstructedSerializationScenario = {
  network: Network;
  activationInputValues: number[];
};

function createConstructedSerializationScenario(): ConstructedSerializationScenario {
  const leftSensor = new Node('input');
  const rightSensor = new Node('input');
  const hiddenStage = new Group(2);
  const readoutLayer = Layer.dense(2, 'output');
  const primaryReadout = readoutLayer.nodes[0];
  const secondaryReadout = readoutLayer.nodes[1];

  leftSensor.describe({ label: 'leftSensor' });
  rightSensor.describe({ label: 'rightSensor' });
  primaryReadout.describe({ label: 'primaryReadout' });
  secondaryReadout.describe({ label: 'secondaryReadout' });

  leftSensor.connect(hiddenStage);
  rightSensor.connect(hiddenStage);
  hiddenStage.connect(readoutLayer);

  return {
    network: Network.construct(
      [hiddenStage, rightSensor, readoutLayer, leftSensor],
      {
        inputNodes: ['rightSensor', 'leftSensor'],
        outputNodes: ['secondaryReadout', 'primaryReadout'],
      },
    ).network,
    activationInputValues: [0.8, 0.2],
  };
}

type JsonRoundTripScenario = {
  architectureName: string;
  createNetwork: () => Network;
};

const JSON_ROUND_TRIP_SCENARIOS: JsonRoundTripScenario[] = [
  {
    architectureName: 'Perceptron',
    createNetwork: () => Architect.perceptron(2, 3, 1),
  },
  {
    architectureName: 'Basic Network',
    createNetwork: () => new Network(3, 2),
  },
  {
    architectureName: 'LSTM',
    createNetwork: () => Architect.lstm(2, 3, 1),
  },
  {
    architectureName: 'GRU',
    createNetwork: () => Architect.gru(2, 3, 2, 1),
  },
  {
    architectureName: 'Random',
    createNetwork: () => Architect.random(2, 5, 1),
  },
  {
    architectureName: 'RandomSparse',
    createNetwork: () =>
      Architect.randomSparse(2, 5, 1, {
        connections: 10,
        seed: 701,
      }),
  },
  {
    architectureName: 'NARX',
    createNetwork: () => Architect.narx(2, 2, 2, 1, 1),
  },
  {
    architectureName: 'Hopfield',
    createNetwork: () => Architect.hopfield(3),
  },
];

function createDeterministicInputValues(inputCount: number): number[] {
  return Array.from({ length: inputCount }, (_, inputIndex) =>
    Number(((inputIndex + 1) / 10).toFixed(2)),
  );
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

function summarizeTemporalExtensionBag(networkJson: NetworkJSON): {
  recurrentModuleCount: number;
  gatedBlockCount: number;
  recurrentKinds: string[];
} {
  const extensionValues = networkJson.extensions?.values as
    | {
        recurrentModules?: Array<{ kind?: string }>;
        gatedBlocks?: Array<unknown>;
      }
    | undefined;
  const recurrentModules = Array.isArray(extensionValues?.recurrentModules)
    ? extensionValues.recurrentModules
    : [];
  const gatedBlocks = Array.isArray(extensionValues?.gatedBlocks)
    ? extensionValues.gatedBlocks
    : [];

  return {
    recurrentModuleCount: recurrentModules.length,
    gatedBlockCount: gatedBlocks.length,
    recurrentKinds: recurrentModules
      .map((recurrentModule) => recurrentModule.kind)
      .filter((kind): kind is string => typeof kind === 'string')
      .toSorted(),
  };
}

function readFirstTemporalConnectionInnovation(
  extensions: NonNullable<NetworkJSON['extensions']> | undefined,
): number {
  const extensionValues = extensions?.values as
    | {
        gatedBlocks?: Array<{ connectionInnovations: number[] }>;
      }
    | undefined;
  const connectionInnovation =
    extensionValues?.gatedBlocks?.[0]?.connectionInnovations?.[0];

  if (typeof connectionInnovation !== 'number') {
    throw new Error('Expected one temporal gated-block connection innovation.');
  }

  return connectionInnovation;
}

type ParameterLayoutOrderingSummary = {
  version: number;
  kindOrder: Array<'bias' | 'weight'>;
  biasNodeIds: number[];
  weightDescriptorKeys: string[];
};

type ParameterLayoutEntryLike = {
  kind: 'bias' | 'weight';
  nodeId?: number;
  from?: number;
  innovation?: number;
  to?: number;
};

type ParameterLayoutLike = {
  version: number;
  entries: ParameterLayoutEntryLike[];
};

type ParameterRuntimeStateSummary = {
  descriptorValues: Array<{
    descriptorKey: string;
    value: number;
  }>;
  layoutVersion: number;
};

type ParameterVectorPayloadLike = {
  layout: {
    version: number;
    entries: ParameterLayoutEntryLike[];
  };
  values: Float64Array;
};

type ParameterVectorRoundTripScenario = {
  activationInputValues: number[];
  sourceNetwork: Network;
  targetNetwork: Network;
};

type WeightDescriptorIdentity = {
  descriptorKey: string;
  fromGeneId: number;
  innovation: number | null;
  toGeneId: number;
};

type ParameterLayoutExpectedErrorScenario = {
  expectedErrorMessage: string;
  network: Network;
};

type MissingInnovationOrderingScenario = {
  expectedWeightDescriptorKeys: string[];
  network: Network;
};

function createParameterLayoutOrderingNetwork(): Network {
  return createConstructedSerializationScenario().network;
}

function createMissingInnovationOrderingScenario(): MissingInnovationOrderingScenario {
  const network = new Network(2, 1, { seed: 421 });
  const fallbackConnection = readRequiredLayoutConnection(
    network,
    0,
    'fallback',
  );
  const innovationConnection = readRequiredLayoutConnection(
    network,
    1,
    'innovation',
  );

  fallbackConnection.innovation = Number.NaN;
  innovationConnection.innovation = 1;

  return {
    expectedWeightDescriptorKeys: [
      createConnectionDescriptorKey(innovationConnection),
      createConnectionDescriptorKey(fallbackConnection),
    ],
    network,
  };
}

function createDuplicateFallbackWeightIdentityScenario(): ParameterLayoutExpectedErrorScenario {
  const network = createSingleValueSerializableNetwork(431);
  const originalConnection = readRequiredLayoutConnection(
    network,
    0,
    'original',
  );
  const duplicateConnection = new Connection(
    originalConnection.from,
    originalConnection.to,
    0.25,
  );

  network.connections.push(duplicateConnection);
  network.connections.forEach((connection) => {
    connection.innovation = Number.NaN;
  });

  return {
    expectedErrorMessage: `ParameterLayoutV1 requires unique stable weight identities. Duplicate identity fallback:${createConnectionDescriptorKey(originalConnection)} is ambiguous.`,
    network,
  };
}

function createDuplicateBiasNodeIdScenario(): ParameterLayoutExpectedErrorScenario {
  const network = createSingleValueSerializableNetwork(432);
  const originalBiasNode = network.nodes.at(0);
  const duplicateBiasNode = network.nodes.at(1);

  if (originalBiasNode === undefined || duplicateBiasNode === undefined) {
    throw new Error('Expected two nodes for duplicate-bias layout case.');
  }

  const duplicateNodeId = readRequiredGeneId(
    originalBiasNode,
    'duplicate bias',
  );

  Object.defineProperty(duplicateBiasNode, 'geneId', {
    configurable: true,
    value: duplicateNodeId,
    writable: true,
  });

  return {
    expectedErrorMessage: `ParameterLayoutV1 requires unique bias node ids. Duplicate node id ${duplicateNodeId} is ambiguous.`,
    network,
  };
}

function createDuplicateInnovationWeightIdentityScenario(): ParameterLayoutExpectedErrorScenario {
  const network = createParameterLayoutOrderingNetwork();
  const duplicateInnovation = 9_001;
  const [primaryConnection, secondaryConnection] =
    readDistinctSourceLayoutConnections(network);

  primaryConnection.innovation = duplicateInnovation;
  secondaryConnection.innovation = duplicateInnovation;

  return {
    expectedErrorMessage: `ParameterLayoutV1 requires unique stable weight identities. Duplicate identity innovation:${duplicateInnovation} is ambiguous.`,
    network,
  };
}

function createMissingWeightSourceGeneIdScenario(): ParameterLayoutExpectedErrorScenario {
  const network = createSingleValueSerializableNetwork(433);
  const sourceNode = new Node('hidden');
  const targetNode = network.nodes.at(-1);

  if (targetNode === undefined) {
    throw new Error(
      'Expected one target node for missing-gene-id layout case.',
    );
  }

  Object.defineProperty(sourceNode, 'geneId', {
    configurable: true,
    value: undefined,
    writable: true,
  });
  network.connections.push(new Connection(sourceNode, targetNode, 0.125));

  return {
    expectedErrorMessage:
      'ParameterLayoutV1 requires a stable weight source node gene id.',
    network,
  };
}

function summarizeParameterLayoutOrdering(
  network: Network,
): ParameterLayoutOrderingSummary {
  const parameterLayout = createParameterLayoutV1(
    network,
  ) as ParameterLayoutLike;

  return {
    version: parameterLayout.version,
    kindOrder: parameterLayout.entries.map(
      (entry: ParameterLayoutEntryLike) => entry.kind,
    ),
    biasNodeIds: parameterLayout.entries
      .filter(
        (
          entry: ParameterLayoutEntryLike,
        ): entry is {
          kind: 'bias';
          nodeId?: number;
        } => entry.kind === 'bias',
      )
      .map((entry: { kind: 'bias'; nodeId?: number }) => entry.nodeId)
      .filter(
        (nodeId: number | undefined): nodeId is number =>
          typeof nodeId === 'number',
      ),
    weightDescriptorKeys: parameterLayout.entries
      .filter(
        (
          entry: ParameterLayoutEntryLike,
        ): entry is {
          kind: 'weight';
          from?: number;
          to?: number;
        } => entry.kind === 'weight',
      )
      .map((entry: { kind: 'weight'; from?: number; to?: number }) =>
        createWeightDescriptorKey(entry.from, entry.to),
      ),
  };
}

function createExpectedParameterLayoutOrderingSummary(
  summary: ParameterLayoutOrderingSummary,
  expectedWeightDescriptorKeys: string[],
): ParameterLayoutOrderingSummary {
  return {
    version: 1,
    kindOrder: createExpectedKindOrder(summary),
    biasNodeIds: summary.biasNodeIds.toSorted(
      (leftNodeId, rightNodeId) => leftNodeId - rightNodeId,
    ),
    weightDescriptorKeys: expectedWeightDescriptorKeys,
  };
}

function createExpectedKindOrder(
  summary: ParameterLayoutOrderingSummary,
): Array<'bias' | 'weight'> {
  return [
    ...Array.from(
      { length: summary.biasNodeIds.length },
      () => 'bias' as const,
    ),
    ...Array.from(
      { length: summary.weightDescriptorKeys.length },
      () => 'weight' as const,
    ),
  ];
}

function collectExpectedWeightDescriptorKeys(network: Network): string[] {
  return network.connections
    .map((connection) => {
      const fromGeneId = readRequiredGeneId(connection.from, 'source');
      const toGeneId = readRequiredGeneId(connection.to, 'target');

      return {
        descriptorKey: createWeightDescriptorKey(fromGeneId, toGeneId),
        fromGeneId,
        innovation:
          typeof connection.innovation === 'number'
            ? connection.innovation
            : null,
        toGeneId,
      };
    })
    .toSorted(compareWeightDescriptorIdentity)
    .map((weightDescriptorIdentity) => weightDescriptorIdentity.descriptorKey);
}

function compareWeightDescriptorIdentity(
  leftWeightDescriptorIdentity: WeightDescriptorIdentity,
  rightWeightDescriptorIdentity: WeightDescriptorIdentity,
): number {
  const leftHasInnovation = leftWeightDescriptorIdentity.innovation !== null;
  const rightHasInnovation = rightWeightDescriptorIdentity.innovation !== null;

  if (leftHasInnovation && rightHasInnovation) {
    const leftInnovation = leftWeightDescriptorIdentity.innovation as number;
    const rightInnovation = rightWeightDescriptorIdentity.innovation as number;
    const innovationDifference = leftInnovation - rightInnovation;

    if (innovationDifference !== 0) {
      return innovationDifference;
    }
  }

  if (leftHasInnovation !== rightHasInnovation) {
    return leftHasInnovation ? -1 : 1;
  }

  if (
    leftWeightDescriptorIdentity.fromGeneId !==
    rightWeightDescriptorIdentity.fromGeneId
  ) {
    return (
      leftWeightDescriptorIdentity.fromGeneId -
      rightWeightDescriptorIdentity.fromGeneId
    );
  }

  return (
    leftWeightDescriptorIdentity.toGeneId -
    rightWeightDescriptorIdentity.toGeneId
  );
}

function createWeightDescriptorKey(
  fromGeneId: number | undefined,
  toGeneId: number | undefined,
): string {
  return `${fromGeneId ?? 'missing'}->${toGeneId ?? 'missing'}`;
}

function readRequiredGeneId(node: Node, endpointLabel: string): number {
  if (typeof node.geneId !== 'number') {
    throw new Error(`Expected one ${endpointLabel} node gene id.`);
  }

  return node.geneId;
}

function readRequiredLayoutConnection(
  network: Network,
  connectionIndex: number,
  connectionRoleLabel: string,
): Connection {
  const connection = network.connections.at(connectionIndex);

  if (connection === undefined) {
    throw new Error(
      `Expected one ${connectionRoleLabel} connection for layout ordering.`,
    );
  }

  return connection;
}

function readDistinctSourceLayoutConnections(
  network: Network,
): [Connection, Connection] {
  for (const primaryConnection of network.connections) {
    const primarySourceGeneId = readRequiredGeneId(
      primaryConnection.from,
      'source',
    );
    const secondaryConnection = network.connections.find(
      (candidateConnection) =>
        candidateConnection !== primaryConnection &&
        readRequiredGeneId(candidateConnection.from, 'source') !==
          primarySourceGeneId,
    );

    if (secondaryConnection !== undefined) {
      return [primaryConnection, secondaryConnection];
    }
  }

  throw new Error(
    'Expected two layout-ordering connections with distinct source gene ids.',
  );
}

function createConnectionDescriptorKey(connection: Connection): string {
  return createWeightDescriptorKey(
    readRequiredGeneId(connection.from, 'source'),
    readRequiredGeneId(connection.to, 'target'),
  );
}

function createParameterVectorRoundTripScenario(): ParameterVectorRoundTripScenario {
  const sourceNetwork = new Network(2, 1, { seed: 551 });

  return {
    activationInputValues: [0.25, 0.75],
    sourceNetwork,
    targetNetwork: Network.fromJSON(sourceNetwork.toJSON()),
  };
}

function perturbRuntimeParameterState(network: Network): void {
  network.nodes.forEach((node, nodeIndex) => {
    node.bias += (nodeIndex + 1) * 2.5;
  });

  network.connections.forEach((connection, connectionIndex) => {
    connection.weight -= (connectionIndex + 1) * 1.75;
  });

  network.selfconns.forEach((connection, connectionIndex) => {
    connection.weight += (connectionIndex + 1) * 1.25;
  });
}

function summarizeRuntimeParameterState(
  network: Network,
): ParameterRuntimeStateSummary {
  const parameterLayout = createParameterLayoutV1(
    network,
  ) as ParameterLayoutLike;

  return {
    descriptorValues: parameterLayout.entries.map((layoutEntry) => ({
      descriptorKey: createParameterRuntimeDescriptorKey(layoutEntry),
      value: readParameterValueForLayoutEntry(network, layoutEntry),
    })),
    layoutVersion: parameterLayout.version,
  };
}

function createParameterRuntimeDescriptorKey(
  layoutEntry: ParameterLayoutEntryLike,
): string {
  if (layoutEntry.kind === 'bias') {
    return `bias:${layoutEntry.nodeId}`;
  }

  return typeof layoutEntry.innovation === 'number'
    ? `weight:innovation:${layoutEntry.innovation}`
    : `weight:fallback:${layoutEntry.from}->${layoutEntry.to}`;
}

function readParameterValueForLayoutEntry(
  network: Network,
  layoutEntry: ParameterLayoutEntryLike,
): number {
  if (layoutEntry.kind === 'bias') {
    const matchingNode = network.nodes.find(
      (candidateNode) => candidateNode.geneId === layoutEntry.nodeId,
    );

    if (matchingNode === undefined) {
      throw new Error('Expected one bias node while reading parameter state.');
    }

    return matchingNode.bias;
  }

  const matchingConnection = [
    ...network.connections,
    ...network.selfconns,
  ].find((candidateConnection) =>
    typeof layoutEntry.innovation === 'number'
      ? candidateConnection.innovation === layoutEntry.innovation
      : readRequiredGeneId(candidateConnection.from, 'source') ===
          layoutEntry.from &&
        readRequiredGeneId(candidateConnection.to, 'target') === layoutEntry.to,
  );

  if (matchingConnection === undefined) {
    throw new Error(
      'Expected one weight connection while reading parameter state.',
    );
  }

  return matchingConnection.weight;
}

function attemptParameterImportAndCaptureFailure(
  targetNetwork: Network,
  parameterVector: ParameterVectorPayloadLike,
): {
  errorMessage: string;
  targetParameterState: ParameterRuntimeStateSummary;
} {
  let errorMessage = '';

  try {
    fromParameterVector(targetNetwork, parameterVector as never);
  } catch (error) {
    errorMessage = (error as Error).message;
  }

  return {
    errorMessage,
    targetParameterState: summarizeRuntimeParameterState(targetNetwork),
  };
}

function rebuildEquivalentNetworkFromReorderedJson(network: Network): Network {
  const serializedJson = network.toJSON() as unknown as NetworkJSON;
  const reorderedNodeEntries = [
    ...collectNodeEntriesByType(serializedJson, 'input'),
    ...collectNodeEntriesByType(serializedJson, 'hidden').toReversed(),
    ...collectNodeEntriesByType(serializedJson, 'output'),
  ];
  const remappedIndexByOriginalIndex = new Map(
    reorderedNodeEntries.map((nodeEntry, reorderedIndex) => [
      nodeEntry.originalIndex,
      reorderedIndex,
    ]),
  );
  const reorderedNodes = reorderedNodeEntries.map(
    ({ nodeJsonEntry }, reorderedIndex) => ({
      ...nodeJsonEntry,
      index: reorderedIndex,
    }),
  );
  const reorderedConnections = serializedJson.connections
    .toReversed()
    .map((connectionJsonEntry) => ({
      ...connectionJsonEntry,
      from: remapConnectionEndpointIndex(
        remappedIndexByOriginalIndex,
        connectionJsonEntry.from,
      ),
      to: remapConnectionEndpointIndex(
        remappedIndexByOriginalIndex,
        connectionJsonEntry.to,
      ),
      gater:
        connectionJsonEntry.gater == null
          ? null
          : remapConnectionEndpointIndex(
              remappedIndexByOriginalIndex,
              connectionJsonEntry.gater,
            ),
    }));

  return Network.fromJSON({
    ...serializedJson,
    connections: reorderedConnections,
    nodes: reorderedNodes,
  });
}

function collectNodeEntriesByType(
  serializedJson: NetworkJSON,
  nodeType: NetworkJSON['nodes'][number]['type'],
): Array<{
  nodeJsonEntry: NetworkJSON['nodes'][number];
  originalIndex: number;
}> {
  return serializedJson.nodes
    .map((nodeJsonEntry, originalIndex) => ({
      nodeJsonEntry,
      originalIndex,
    }))
    .filter(({ nodeJsonEntry }) => nodeJsonEntry.type === nodeType);
}

function remapConnectionEndpointIndex(
  remappedIndexByOriginalIndex: Map<number, number>,
  originalIndex: number,
): number {
  const remappedIndex = remappedIndexByOriginalIndex.get(originalIndex);

  if (typeof remappedIndex !== 'number') {
    throw new Error('Expected one remapped node index for layout ordering.');
  }

  return remappedIndex;
}

describe('network serialize chapter', () => {
  describe('Network.serializeCompressedArchiveWithMetrics()', () => {
    describe('given one recurrent runtime is archived with the default codec', () => {
      it('reports encode metrics while preserving the exact next activation output', () => {
        // Arrange
        const network = Architect.lstm(2, 3, 1);
        const historyInputs = [
          [0.1, 0.2],
          [0.3, 0.4],
        ];
        const nextInput = [0.5, 0.6];

        historyInputs.forEach((inputValues) => {
          network.activate(inputValues);
        });

        // Act
        const archiveWithMetrics =
          serializeCompressedArchiveWithMetrics.call(network);
        const expectedNextOutput = network.activate(nextInput);
        const rebuiltNetwork = deserializeCompressedArchive(
          archiveWithMetrics.archive,
        );
        const rebuiltNextOutput = rebuiltNetwork.activate(nextInput);

        // Assert
        expect({
          compressionRatioMatches:
            archiveWithMetrics.metrics.compressionRatio ===
            archiveWithMetrics.metrics.compressedByteLength /
              archiveWithMetrics.metrics.uncompressedByteLength,
          compressedByteLengthPositive:
            archiveWithMetrics.metrics.compressedByteLength > 0,
          encodeTimeMsFinite:
            Number.isFinite(archiveWithMetrics.metrics.encodeTimeMs) &&
            archiveWithMetrics.metrics.encodeTimeMs >= 0,
          rebuiltNextOutputWithinTolerance: outputsMatchWithinTolerance(
            rebuiltNextOutput,
            expectedNextOutput,
          ),
          uncompressedByteLengthPositive:
            archiveWithMetrics.metrics.uncompressedByteLength > 0,
        }).toEqual({
          compressionRatioMatches: true,
          compressedByteLengthPositive: true,
          encodeTimeMsFinite: true,
          rebuiltNextOutputWithinTolerance: true,
          uncompressedByteLengthPositive: true,
        });
      });
    });
  });

  describe('Network.serializeCompressedArchiveAsyncWithMetrics()', () => {
    describe('given browser archive compression is used for one recurrent runtime', () => {
      it('reports encode metrics while preserving the exact next activation output', async () => {
        // Arrange
        const network = Architect.lstm(2, 3, 1);
        const historyInputs = [
          [0.1, 0.2],
          [0.3, 0.4],
        ];
        const nextInput = [0.5, 0.6];
        const originalProcess = globalThis.process;

        historyInputs.forEach((inputValues) => {
          network.activate(inputValues);
        });

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          const archiveWithMetrics =
            await serializeCompressedArchiveAsyncWithMetrics.call(network);
          const expectedNextOutput = network.activate(nextInput);
          const rebuiltNetwork = await deserializeCompressedArchiveAsync(
            archiveWithMetrics.archive,
          );
          const rebuiltNextOutput = rebuiltNetwork.activate(nextInput);

          // Assert
          expect({
            compressionRatioMatches:
              archiveWithMetrics.metrics.compressionRatio ===
              archiveWithMetrics.metrics.compressedByteLength /
                archiveWithMetrics.metrics.uncompressedByteLength,
            compressedByteLengthPositive:
              archiveWithMetrics.metrics.compressedByteLength > 0,
            encodeTimeMsFinite:
              Number.isFinite(archiveWithMetrics.metrics.encodeTimeMs) &&
              archiveWithMetrics.metrics.encodeTimeMs >= 0,
            rebuiltNextOutputWithinTolerance: outputsMatchWithinTolerance(
              rebuiltNextOutput,
              expectedNextOutput,
            ),
            uncompressedByteLengthPositive:
              archiveWithMetrics.metrics.uncompressedByteLength > 0,
          }).toEqual({
            compressionRatioMatches: true,
            compressedByteLengthPositive: true,
            encodeTimeMsFinite: true,
            rebuiltNextOutputWithinTolerance: true,
            uncompressedByteLengthPositive: true,
          });
        } finally {
          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: originalProcess,
          });
        }
      });
    });
  });

  describe('Network.deserializeCompressedArchiveWithMetrics()', () => {
    describe('given one recurrent runtime is archived with the default codec', () => {
      it('reports decode metrics while preserving the exact next activation output', () => {
        // Arrange
        const network = Architect.lstm(2, 3, 1);
        const historyInputs = [
          [0.1, 0.2],
          [0.3, 0.4],
        ];
        const nextInput = [0.5, 0.6];

        historyInputs.forEach((inputValues) => {
          network.activate(inputValues);
        });

        const compressedArchive = serializeCompressedArchive.call(network);
        const expectedNextOutput = network.activate(nextInput);

        // Act
        const rebuiltNetworkWithMetrics =
          deserializeCompressedArchiveWithMetrics(compressedArchive);
        const rebuiltNextOutput =
          rebuiltNetworkWithMetrics.value.activate(nextInput);

        // Assert
        expect({
          compressionRatioMatches:
            rebuiltNetworkWithMetrics.metrics.compressionRatio ===
            rebuiltNetworkWithMetrics.metrics.compressedByteLength /
              rebuiltNetworkWithMetrics.metrics.uncompressedByteLength,
          compressedByteLengthPositive:
            rebuiltNetworkWithMetrics.metrics.compressedByteLength > 0,
          decodeTimeMsFinite:
            Number.isFinite(rebuiltNetworkWithMetrics.metrics.decodeTimeMs) &&
            rebuiltNetworkWithMetrics.metrics.decodeTimeMs >= 0,
          rebuiltNextOutputWithinTolerance: outputsMatchWithinTolerance(
            rebuiltNextOutput,
            expectedNextOutput,
          ),
          uncompressedByteLengthPositive:
            rebuiltNetworkWithMetrics.metrics.uncompressedByteLength > 0,
        }).toEqual({
          compressionRatioMatches: true,
          compressedByteLengthPositive: true,
          decodeTimeMsFinite: true,
          rebuiltNextOutputWithinTolerance: true,
          uncompressedByteLengthPositive: true,
        });
      });
    });
  });

  describe('Network.deserializeCompressedArchiveAsyncWithMetrics()', () => {
    describe('given optional decode callbacks are omitted', () => {
      it('rebuilds a runnable network through the default async metrics path', async () => {
        // Arrange
        const network = Architect.lstm(2, 3, 1);
        const historyInputs = [
          [0.1, 0.2],
          [0.3, 0.4],
        ];
        const nextInput = [0.5, 0.6];

        historyInputs.forEach((inputValues) => {
          network.activate(inputValues);
        });

        const compressedArchive =
          await serializeCompressedArchiveAsync.call(network);
        const expectedNextOutput = network.activate(nextInput);

        // Act
        const rebuiltNetworkWithMetrics =
          await deserializeCompressedArchiveAsyncWithMetrics(compressedArchive);
        const rebuiltNextOutput =
          rebuiltNetworkWithMetrics.value.activate(nextInput);

        // Assert
        expect({
          decodeTimeMsFinite:
            Number.isFinite(rebuiltNetworkWithMetrics.metrics.decodeTimeMs) &&
            rebuiltNetworkWithMetrics.metrics.decodeTimeMs >= 0,
          rebuiltNextOutputWithinTolerance: outputsMatchWithinTolerance(
            rebuiltNextOutput,
            expectedNextOutput,
          ),
          uncompressedByteLengthPositive:
            rebuiltNetworkWithMetrics.metrics.uncompressedByteLength > 0,
        }).toEqual({
          decodeTimeMsFinite: true,
          rebuiltNextOutputWithinTolerance: true,
          uncompressedByteLengthPositive: true,
        });
      });
    });

    describe('given browser archive compression is used for one recurrent runtime', () => {
      it('reports decode metrics while preserving progress snapshots and the exact next activation output', async () => {
        // Arrange
        const network = Architect.lstm(2, 3, 1);
        const historyInputs = [
          [0.1, 0.2],
          [0.3, 0.4],
        ];
        const nextInput = [0.5, 0.6];
        const originalProcess = globalThis.process;
        const progressSnapshots: Array<{ done: boolean }> = [];

        historyInputs.forEach((inputValues) => {
          network.activate(inputValues);
        });

        const compressedArchive =
          await serializeCompressedArchiveAsync.call(network);

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });

        try {
          const expectedNextOutput = network.activate(nextInput);

          // Act
          const rebuiltNetworkWithMetrics =
            await deserializeCompressedArchiveAsyncWithMetrics(
              compressedArchive,
              undefined,
              undefined,
              {
                onProgress(progressUpdate) {
                  progressSnapshots.push({
                    done: progressUpdate.done,
                  });
                },
              },
            );
          const rebuiltNextOutput =
            rebuiltNetworkWithMetrics.value.activate(nextInput);

          // Assert
          expect({
            compressionRatioMatches:
              rebuiltNetworkWithMetrics.metrics.compressionRatio ===
              rebuiltNetworkWithMetrics.metrics.compressedByteLength /
                rebuiltNetworkWithMetrics.metrics.uncompressedByteLength,
            decodeTimeMsFinite:
              Number.isFinite(rebuiltNetworkWithMetrics.metrics.decodeTimeMs) &&
              rebuiltNetworkWithMetrics.metrics.decodeTimeMs >= 0,
            hasProgressSnapshots: progressSnapshots.length > 0,
            lastProgressDone: progressSnapshots.at(-1)?.done ?? false,
            rebuiltNextOutputWithinTolerance: outputsMatchWithinTolerance(
              rebuiltNextOutput,
              expectedNextOutput,
            ),
          }).toEqual({
            compressionRatioMatches: true,
            decodeTimeMsFinite: true,
            hasProgressSnapshots: true,
            lastProgressDone: true,
            rebuiltNextOutputWithinTolerance: true,
          });
        } finally {
          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: originalProcess,
          });
        }
      });
    });
  });

  describe('Network.deserializeCompressedArchiveAsync()', () => {
    describe('given browser archive compression is used for one recurrent runtime', () => {
      describe('when incremental decode progress is requested', () => {
        it('forwards progress snapshots while preserving the exact next activation output', async () => {
          // Arrange
          const network = Architect.lstm(2, 3, 1);
          const historyInputs = [
            [0.1, 0.2],
            [0.3, 0.4],
          ];
          const nextInput = [0.5, 0.6];
          const originalProcess = globalThis.process;
          const progressSnapshots: Array<{ done: boolean }> = [];

          historyInputs.forEach((inputValues) => {
            network.activate(inputValues);
          });

          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: undefined,
          });

          try {
            const compressedArchive =
              await serializeCompressedArchiveAsync.call(network);
            const expectedNextOutput = network.activate(nextInput);

            // Act
            const rebuiltNetwork = await deserializeCompressedArchiveAsync(
              compressedArchive,
              undefined,
              undefined,
              {
                onProgress(progressUpdate) {
                  progressSnapshots.push({
                    done: progressUpdate.done,
                  });
                },
              },
            );
            const rebuiltNextOutput = rebuiltNetwork.activate(nextInput);

            // Assert
            expect({
              hasProgressSnapshots: progressSnapshots.length > 0,
              lastProgressDone: progressSnapshots.at(-1)?.done ?? false,
              rebuiltNextOutputWithinTolerance:
                rebuiltNextOutput.length === expectedNextOutput.length &&
                rebuiltNextOutput.every(
                  (outputValue, outputIndex) =>
                    Math.abs(outputValue - expectedNextOutput[outputIndex]!) <=
                    Number.EPSILON,
                ),
            }).toEqual({
              hasProgressSnapshots: true,
              lastProgressDone: true,
              rebuiltNextOutputWithinTolerance: true,
            });
          } finally {
            Object.defineProperty(globalThis, 'process', {
              configurable: true,
              value: originalProcess,
            });
          }
        });
      });

      describe('when the archived payload is rebuilt and activated again', () => {
        it('preserves the next activation output within floating-point tolerance', async () => {
          // Arrange
          const network = Architect.lstm(2, 3, 1);
          const historyInputs = [
            [0.1, 0.2],
            [0.3, 0.4],
          ];
          const nextInput = [0.5, 0.6];
          const originalProcess = globalThis.process;

          historyInputs.forEach((inputValues) => {
            network.activate(inputValues);
          });

          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: undefined,
          });

          try {
            const compressedArchive =
              await serializeCompressedArchiveAsync.call(network);
            const expectedNextOutput = network.activate(nextInput);

            // Act
            const rebuiltNetwork =
              await deserializeCompressedArchiveAsync(compressedArchive);
            const rebuiltNextOutput = rebuiltNetwork.activate(nextInput);

            // Assert
            expect(
              outputsMatchWithinTolerance(
                rebuiltNextOutput,
                expectedNextOutput,
              ),
            ).toBe(true);
          } finally {
            Object.defineProperty(globalThis, 'process', {
              configurable: true,
              value: originalProcess,
            });
          }
        });
      });
    });
  });

  describe('Network.deserializeCompressedArchive()', () => {
    describe('given one recurrent runtime is archived with the default gzip codec', () => {
      describe('when the archived payload is rebuilt and activated again', () => {
        it('preserves the next activation output within floating-point tolerance', () => {
          // Arrange
          const network = Architect.lstm(2, 3, 1);
          const historyInputs = [
            [0.1, 0.2],
            [0.3, 0.4],
          ];
          const nextInput = [0.5, 0.6];

          historyInputs.forEach((inputValues) => {
            network.activate(inputValues);
          });

          const compressedArchive = serializeCompressedArchive.call(network);
          const expectedNextOutput = network.activate(nextInput);

          // Act
          const rebuiltNetwork =
            deserializeCompressedArchive(compressedArchive);
          const rebuiltNextOutput = rebuiltNetwork.activate(nextInput);

          // Assert
          expect(
            outputsMatchWithinTolerance(rebuiltNextOutput, expectedNextOutput),
          ).toBe(true);
        });
      });
    });

    describe('given one recurrent runtime is archived with the zstd codec', () => {
      describe('when the archived payload is rebuilt and activated again', () => {
        it('preserves the next activation output within floating-point tolerance', () => {
          // Arrange
          const network = Architect.lstm(2, 3, 1);
          const historyInputs = [
            [0.1, 0.2],
            [0.3, 0.4],
          ];
          const nextInput = [0.5, 0.6];

          historyInputs.forEach((inputValues) => {
            network.activate(inputValues);
          });

          const compressedArchive = serializeCompressedArchive.call(network, {
            compression: 'zstd',
          });
          const expectedNextOutput = network.activate(nextInput);

          // Act
          const rebuiltNetwork =
            deserializeCompressedArchive(compressedArchive);
          const rebuiltNextOutput = rebuiltNetwork.activate(nextInput);

          // Assert
          expect(
            outputsMatchWithinTolerance(rebuiltNextOutput, expectedNextOutput),
          ).toBe(true);
        });
      });
    });
  });

  describe('Network.deserializeCompressed()', () => {
    describe('given one recurrent runtime is serialized after carrying state forward', () => {
      describe('when the compressed payload is rebuilt and activated again', () => {
        it('preserves the next activation output within floating-point tolerance', () => {
          // Arrange
          const network = Architect.lstm(2, 3, 1);
          const historyInputs = [
            [0.1, 0.2],
            [0.3, 0.4],
          ];
          const nextInput = [0.5, 0.6];

          historyInputs.forEach((inputValues) => {
            network.activate(inputValues);
          });

          const compressedPayload = serializeCompressed.call(network);
          const expectedNextOutput = network.activate(nextInput);

          // Act
          const rebuiltNetwork = deserializeCompressed(compressedPayload);
          const rebuiltNextOutput = rebuiltNetwork.activate(nextInput);

          // Assert
          expect(
            outputsMatchWithinTolerance(rebuiltNextOutput, expectedNextOutput),
          ).toBe(true);
        });
      });
    });

    describe('given the compressed payload uses an unknown format tag', () => {
      describe('when the payload is rebuilt', () => {
        it('throws an invalid-format error', () => {
          // Arrange
          let errorMessage = '';

          // Act
          try {
            deserializeCompressed({
              format: 'unsupported-compressed-format',
            } as never);
          } catch (error) {
            errorMessage = (error as Error).message;
          }

          // Assert
          expect(errorMessage).toBe(
            'Invalid compressed network payload format.',
          );
        });
      });
    });

    describe('given explicit size overrides are supplied', () => {
      describe('when the compressed payload is rebuilt', () => {
        it('uses the override branch without changing the output width', () => {
          // Arrange
          const network = createSerializableNetwork(390);
          const compressedPayload = serializeCompressed.call(network);

          // Act
          const rebuiltNetwork = deserializeCompressed(
            compressedPayload,
            network.input,
            network.output,
          );

          // Assert
          expect(rebuiltNetwork.output).toBe(network.output);
        });
      });
    });

    describe('given the compressed runtime-state arrays are truncated', () => {
      describe('when the payload is rebuilt', () => {
        it('throws an invalid-runtime-state-length error', () => {
          // Arrange
          const network = createSerializableNetwork(391);
          const compressedPayload = serializeCompressed.call(network);
          let errorMessage = '';

          compressedPayload.states = compressedPayload.states.slice(1);

          // Act
          try {
            deserializeCompressed(compressedPayload);
          } catch (error) {
            errorMessage = (error as Error).message;
          }

          // Assert
          expect(errorMessage).toBe(
            'Compressed runtime state length is invalid.',
          );
        });
      });
    });
  });

  describe('ParameterLayoutV1 ordering', () => {
    describe('given one live runtime is inspected twice', () => {
      it('keeps the same descriptor ordering between repeated reads', () => {
        // Arrange
        const network = createParameterLayoutOrderingNetwork();
        const expectedWeightDescriptorKeys =
          collectExpectedWeightDescriptorKeys(network);

        // Act
        const firstLayoutSummary = summarizeParameterLayoutOrdering(network);
        const secondLayoutSummary = summarizeParameterLayoutOrdering(network);
        const expectedLayoutSummary =
          createExpectedParameterLayoutOrderingSummary(
            firstLayoutSummary,
            expectedWeightDescriptorKeys,
          );

        // Assert
        expect({
          firstLayoutSummary,
          secondLayoutSummary,
        }).toEqual({
          firstLayoutSummary: expectedLayoutSummary,
          secondLayoutSummary: expectedLayoutSummary,
        });
      });
    });

    describe('given equivalent restore paths perturb incidental payload order', () => {
      it('keeps the same descriptor ordering across rebuilds', () => {
        // Arrange
        const network = createParameterLayoutOrderingNetwork();
        const expectedWeightDescriptorKeys =
          collectExpectedWeightDescriptorKeys(network);
        const rebuiltFromReorderedJson =
          rebuildEquivalentNetworkFromReorderedJson(network);
        const rebuiltFromCompactPayload = Network.deserialize(
          network.serialize(),
          network.input,
          network.output,
        );

        // Act
        const liveLayoutSummary = summarizeParameterLayoutOrdering(network);
        const reorderedJsonLayoutSummary = summarizeParameterLayoutOrdering(
          rebuiltFromReorderedJson,
        );
        const compactLayoutSummary = summarizeParameterLayoutOrdering(
          rebuiltFromCompactPayload,
        );
        const expectedLayoutSummary =
          createExpectedParameterLayoutOrderingSummary(
            liveLayoutSummary,
            expectedWeightDescriptorKeys,
          );

        // Assert
        expect({
          compactLayoutSummary,
          liveLayoutSummary,
          reorderedJsonLayoutSummary,
        }).toEqual({
          compactLayoutSummary: expectedLayoutSummary,
          liveLayoutSummary: expectedLayoutSummary,
          reorderedJsonLayoutSummary: expectedLayoutSummary,
        });
      });
    });

    describe('given one weight is missing an innovation id', () => {
      it('places finite innovations before fallback endpoint identities', () => {
        // Arrange
        const { expectedWeightDescriptorKeys, network } =
          createMissingInnovationOrderingScenario();

        // Act
        const layoutSummary = summarizeParameterLayoutOrdering(network);

        // Assert
        expect(layoutSummary.weightDescriptorKeys).toEqual(
          expectedWeightDescriptorKeys,
        );
      });
    });

    describe('given one fallback weight follows a finite innovation in runtime order', () => {
      it('still places finite innovations before fallback endpoint identities', () => {
        // Arrange
        const network = new Network(2, 1, { seed: 434 });
        const innovationConnection = readRequiredLayoutConnection(
          network,
          0,
          'innovation',
        );
        const fallbackConnection = readRequiredLayoutConnection(
          network,
          1,
          'fallback',
        );
        innovationConnection.innovation = 1;
        fallbackConnection.innovation = Number.NaN;
        const expectedWeightDescriptorKeys = [
          createConnectionDescriptorKey(innovationConnection),
          createConnectionDescriptorKey(fallbackConnection),
        ];

        // Act
        const layoutSummary = summarizeParameterLayoutOrdering(network);

        // Assert
        expect(layoutSummary.weightDescriptorKeys).toEqual(
          expectedWeightDescriptorKeys,
        );
      });
    });

    describe('given fallback weight identities are ambiguous', () => {
      it('throws instead of inheriting connection array order', () => {
        // Arrange
        const { expectedErrorMessage, network } =
          createDuplicateFallbackWeightIdentityScenario();
        let errorMessage = '';

        // Act
        try {
          createParameterLayoutV1(network);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(expectedErrorMessage);
      });
    });

    describe('given bias node ids are ambiguous', () => {
      it('throws instead of inheriting node array order', () => {
        // Arrange
        const { expectedErrorMessage, network } =
          createDuplicateBiasNodeIdScenario();
        let errorMessage = '';

        // Act
        try {
          createParameterLayoutV1(network);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(expectedErrorMessage);
      });
    });

    describe('given equal innovations collide across distinct source nodes', () => {
      it('throws instead of treating the secondary source-order tie break as identity', () => {
        // Arrange
        const { expectedErrorMessage, network } =
          createDuplicateInnovationWeightIdentityScenario();
        let errorMessage = '';

        // Act
        try {
          createParameterLayoutV1(network);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(expectedErrorMessage);
      });
    });

    describe('given a weight endpoint is missing a stable gene id', () => {
      it('throws instead of emitting an unstable descriptor', () => {
        // Arrange
        const { expectedErrorMessage, network } =
          createMissingWeightSourceGeneIdScenario();
        let errorMessage = '';

        // Act
        try {
          createParameterLayoutV1(network);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(expectedErrorMessage);
      });
    });
  });

  describe('ParameterVector v1 roundtrip', () => {
    describe('given one compatible runtime clone is perturbed before import', () => {
      it('restores the same-runtime inference outputs for the same topology', () => {
        // Arrange
        const { activationInputValues, sourceNetwork, targetNetwork } =
          createParameterVectorRoundTripScenario();
        const sourceParameterState =
          summarizeRuntimeParameterState(sourceNetwork);
        const sourceOutput = sourceNetwork.activate(activationInputValues);
        const parameterVector = toParameterVector(sourceNetwork);

        perturbRuntimeParameterState(targetNetwork);

        const perturbedTargetOutput = targetNetwork.activate(
          activationInputValues,
        );

        // Act
        fromParameterVector(targetNetwork, parameterVector);

        const importedTargetOutput = targetNetwork.activate(
          activationInputValues,
        );
        const importedTargetState =
          summarizeRuntimeParameterState(targetNetwork);

        // Assert
        expect({
          importedTargetOutputWithinTolerance: outputsMatchWithinTolerance(
            importedTargetOutput,
            sourceOutput,
          ),
          importedTargetState,
          perturbedTargetOutputWithinTolerance: outputsMatchWithinTolerance(
            perturbedTargetOutput,
            sourceOutput,
          ),
        }).toEqual({
          importedTargetOutputWithinTolerance: true,
          importedTargetState: sourceParameterState,
          perturbedTargetOutputWithinTolerance: false,
        });
      });
    });

    describe('given an imported payload is incompatible with the target layout', () => {
      it('rejects a layout version mismatch before mutating the target network', () => {
        // Arrange
        const { sourceNetwork, targetNetwork } =
          createParameterVectorRoundTripScenario();
        const baselineParameterState =
          summarizeRuntimeParameterState(targetNetwork);
        const baseParameterVector: ParameterVectorPayloadLike =
          toParameterVector(sourceNetwork);
        const versionMismatchVector: ParameterVectorPayloadLike = {
          ...baseParameterVector,
          layout: {
            ...baseParameterVector.layout,
            version: 2 as 1,
          },
        };

        // Act
        const importAttempt = attemptParameterImportAndCaptureFailure(
          targetNetwork,
          versionMismatchVector,
        );

        // Assert
        expect({
          errorMessageIncludesVersion:
            importAttempt.errorMessage.includes('version'),
          targetParameterState: importAttempt.targetParameterState,
        }).toEqual({
          errorMessageIncludesVersion: true,
          targetParameterState: baselineParameterState,
        });
      });

      it('rejects a layout entry-count mismatch before mutating the target network', () => {
        // Arrange
        const { sourceNetwork, targetNetwork } =
          createParameterVectorRoundTripScenario();
        const baselineParameterState =
          summarizeRuntimeParameterState(targetNetwork);
        const baseParameterVector: ParameterVectorPayloadLike =
          toParameterVector(sourceNetwork);
        const entryCountMismatchVector: ParameterVectorPayloadLike = {
          ...baseParameterVector,
          layout: {
            ...baseParameterVector.layout,
            entries: baseParameterVector.layout.entries.slice(1),
          },
          values: baseParameterVector.values.slice(1),
        };

        // Act
        const importAttempt = attemptParameterImportAndCaptureFailure(
          targetNetwork,
          entryCountMismatchVector,
        );

        // Assert
        expect({
          errorMessageIncludesEntryCount:
            importAttempt.errorMessage.includes('entry count'),
          targetParameterState: importAttempt.targetParameterState,
        }).toEqual({
          errorMessageIncludesEntryCount: true,
          targetParameterState: baselineParameterState,
        });
      });

      it('rejects a values-length mismatch before mutating the target network', () => {
        // Arrange
        const { sourceNetwork, targetNetwork } =
          createParameterVectorRoundTripScenario();
        const baselineParameterState =
          summarizeRuntimeParameterState(targetNetwork);
        const baseParameterVector: ParameterVectorPayloadLike =
          toParameterVector(sourceNetwork);
        const valuesLengthMismatchVector: ParameterVectorPayloadLike = {
          ...baseParameterVector,
          values: baseParameterVector.values.slice(1),
        };

        // Act
        const importAttempt = attemptParameterImportAndCaptureFailure(
          targetNetwork,
          valuesLengthMismatchVector,
        );

        // Assert
        expect({
          errorMessageIncludesValuesLength:
            importAttempt.errorMessage.includes('values length'),
          targetParameterState: importAttempt.targetParameterState,
        }).toEqual({
          errorMessageIncludesValuesLength: true,
          targetParameterState: baselineParameterState,
        });
      });

      it('rejects an ordered descriptor mismatch before mutating the target network', () => {
        // Arrange
        const { sourceNetwork, targetNetwork } =
          createParameterVectorRoundTripScenario();
        const baselineParameterState =
          summarizeRuntimeParameterState(targetNetwork);
        const baseParameterVector: ParameterVectorPayloadLike =
          toParameterVector(sourceNetwork);
        const firstEntry = baseParameterVector.layout.entries.at(0);
        const secondEntry = baseParameterVector.layout.entries.at(1);

        if (firstEntry === undefined || secondEntry === undefined) {
          throw new Error(
            'Expected two layout entries for descriptor mismatch coverage.',
          );
        }

        const descriptorMismatchVector: ParameterVectorPayloadLike = {
          ...baseParameterVector,
          layout: {
            ...baseParameterVector.layout,
            entries: baseParameterVector.layout.entries
              .with(0, secondEntry)
              .with(1, firstEntry),
          },
        };

        // Act
        const importAttempt = attemptParameterImportAndCaptureFailure(
          targetNetwork,
          descriptorMismatchVector,
        );

        // Assert
        expect({
          errorMessageIncludesDescriptorMismatch:
            importAttempt.errorMessage.includes('descriptor mismatch'),
          targetParameterState: importAttempt.targetParameterState,
        }).toEqual({
          errorMessageIncludesDescriptorMismatch: true,
          targetParameterState: baselineParameterState,
        });
      });
    });

    describe('given one source runtime has a non-neutral node response', () => {
      it('rejects ParameterVector export explicitly', () => {
        // Arrange
        const network = createSingleValueSerializableNetwork(552);
        const responseNode = network.nodes.at(0);

        if (responseNode === undefined) {
          throw new Error(
            'Expected one node for unsupported node.response coverage.',
          );
        }

        responseNode.response = 0.5;
        let errorMessage = '';

        // Act
        try {
          toParameterVector(network);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage.includes('node.response')).toBe(true);
      });
    });

    describe('given one exported weight lacks an innovation id', () => {
      it('uses fallback endpoint ids in the ParameterVector layout', () => {
        // Arrange
        const { network } = createMissingInnovationOrderingScenario();
        const fallbackConnection = network.connections.find(
          (connection) => !Number.isFinite(connection.innovation),
        );

        if (fallbackConnection === undefined) {
          throw new Error(
            'Expected one fallback connection for ParameterVector coverage.',
          );
        }

        const expectedFallbackDescriptor = {
          kind: 'weight' as const,
          from: readRequiredGeneId(fallbackConnection.from, 'source'),
          to: readRequiredGeneId(fallbackConnection.to, 'target'),
        };

        // Act
        const parameterVector = toParameterVector(network);
        const fallbackDescriptor = parameterVector.layout.entries.find(
          (layoutEntry) =>
            layoutEntry.kind === 'weight' &&
            typeof layoutEntry.innovation !== 'number',
        );

        // Assert
        expect(fallbackDescriptor).toEqual(expectedFallbackDescriptor);
      });
    });

    describe('given one target runtime has a non-neutral connection gain', () => {
      it('rejects ParameterVector import before mutating the target runtime', () => {
        // Arrange
        const { sourceNetwork, targetNetwork } =
          createParameterVectorRoundTripScenario();
        const parameterVector = toParameterVector(sourceNetwork);
        const baselineParameterState =
          summarizeRuntimeParameterState(targetNetwork);
        const targetConnection = targetNetwork.connections.at(0);

        if (targetConnection === undefined) {
          throw new Error(
            'Expected one connection for unsupported connection.gain coverage.',
          );
        }

        targetConnection.gain = 0.5;

        // Act
        const importAttempt = attemptParameterImportAndCaptureFailure(
          targetNetwork,
          parameterVector,
        );

        // Assert
        expect({
          errorMessageIncludesConnectionGain:
            importAttempt.errorMessage.includes('connection.gain'),
          targetParameterState: importAttempt.targetParameterState,
        }).toEqual({
          errorMessageIncludesConnectionGain: true,
          targetParameterState: baselineParameterState,
        });
      });
    });
  });

  describe('Network.deserialize()', () => {
    describe('given one compact payload omits topology intent metadata', () => {
      describe('when the payload is rebuilt', () => {
        it('rebuilds the network without requiring the optional topology slot', () => {
          // Arrange
          const network = createSerializableNetwork(359);
          const serializedNetwork = network.serialize();
          const compactPayloadWithoutTopologyIntent = serializedNetwork.slice(
            0,
            7,
          ) as never;

          // Act
          const deserialized = Network.deserialize(
            compactPayloadWithoutTopologyIntent,
            network.input,
            network.output,
          );

          // Assert
          expect(deserialized.output).toBe(network.output);
        });
      });
    });

    describe('given one construct-built runtime uses explicit public IO ordering', () => {
      describe('when the compact payload is rebuilt', () => {
        it('preserves the ordered IO ids and topology intent', () => {
          // Arrange
          const { network } = createConstructedSerializationScenario();
          const serializedNetwork = network.serialize();
          const expectedSignature = {
            inputNodeIds: network.inputNodeIds,
            outputNodeIds: network.outputNodeIds,
            topologyIntent: network.getTopologyIntent(),
          };

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );
          const actualSignature = {
            inputNodeIds: deserialized.inputNodeIds,
            outputNodeIds: deserialized.outputNodeIds,
            topologyIntent: deserialized.getTopologyIntent(),
          };

          // Assert
          expect(actualSignature).toEqual(expectedSignature);
        });
      });
    });

    describe('given a compact payload is rebuilt without explicit size overrides', () => {
      describe('when the source network shape is compared to the rebuilt network', () => {
        it('preserves the node count', () => {
          // Arrange
          const network = createSerializableNetwork(360);
          network.activate([0.5, 0.5]);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(serializedNetwork);
          const deserializedNodeCount = deserialized.nodes.length;

          // Assert
          expect(deserializedNodeCount).toBe(network.nodes.length);
        });
      });

      describe('when the rebuilt node collection is inspected', () => {
        it('returns a nodes array', () => {
          // Arrange
          const network = createSerializableNetwork(361);
          network.activate([0.5, 0.5]);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(serializedNetwork);
          const hasNodesArray = Array.isArray(deserialized.nodes);

          // Assert
          expect(hasNodesArray).toBe(true);
        });
      });

      describe('when the rebuilt network is activated with the original input', () => {
        it('preserves the output width', () => {
          // Arrange
          const network = createSerializableNetwork(362);
          const input = [0.2, 0.8];
          const originalOutput = network.activate(input);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );
          const deserializedOutput = deserialized.activate(input);

          // Assert
          expect(deserializedOutput.length).toBe(originalOutput.length);
        });
      });

      describe('when the rebuilt network output is compared to the original output', () => {
        it('stays numerically close', () => {
          // Arrange
          const network = createSerializableNetwork(363);
          const input = [0.2, 0.8];
          const originalOutput = network.activate(input);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );
          const deserializedOutput = deserialized.activate(input);
          const epsilon = 0.05;
          const allOutputsStayClose = deserializedOutput.every(
            (outputValue, outputIndex) =>
              Math.abs(outputValue - originalOutput[outputIndex]) < epsilon,
          );

          // Assert
          expect(allOutputsStayClose).toBe(true);
        });
      });

      describe('when historical identity is compared after rebuild', () => {
        it('preserves node gene ids, connection innovations, and topology intent', () => {
          // Arrange
          const network = createSerializableNetwork(3631);
          network.setTopologyIntent('unconstrained');
          network.mutate(methods.mutation.ADD_NODE);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );

          // Assert
          expect({
            nodeGeneIds: deserialized.nodes.map((node) => node.geneId),
            connectionInnovations: deserialized.connections.map(
              (connection) => connection.innovation,
            ),
            topologyIntent: deserialized.getTopologyIntent(),
          }).toEqual({
            nodeGeneIds: network.nodes.map((node) => node.geneId),
            connectionInnovations: network.connections.map(
              (connection) => connection.innovation,
            ),
            topologyIntent: network.getTopologyIntent(),
          });
        });
      });

      describe('when the rebuilt network is validated as a native genome', () => {
        it('passes the proper-NEAT validator', () => {
          // Arrange
          const network = createSerializableNetwork(3632);
          network.setTopologyIntent('unconstrained');
          network.mutate(methods.mutation.ADD_NODE);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );
          const validationReport = validateNativeGenome(deserialized);

          // Assert
          expect(validationReport.isValid).toBe(true);
        });
      });
    });

    describe('given the compact payload has no connections', () => {
      describe('when deserialize() is called', () => {
        it('rebuilds a network with no connections', () => {
          // Arrange
          const network = createSerializableNetwork(364);
          network.connections = [];
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );
          const connectionCount = deserialized.connections.length;

          // Assert
          expect(connectionCount).toBe(0);
        });
      });
    });

    describe('given the compact payload contains invalid connection indices', () => {
      describe('when deserialize() rebuilds the payload', () => {
        it('skips the invalid connection entries', () => {
          // Arrange
          const network = createSerializableNetwork(365);
          const serializedNetwork = network.serialize();
          serializedNetwork[3][0].from = 999;
          serializedNetwork[3][0].to = 999;

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );

          // Assert
          expect(deserialized.connections.length).toBeLessThanOrEqual(
            network.connections.length,
          );
        });
      });

      describe('when warnings are observed during rebuild', () => {
        it('warns about the invalid connection indices', () => {
          // Arrange
          const network = createSerializableNetwork(366);
          const serializedNetwork = network.serialize();
          serializedNetwork[3][0].from = 999;
          serializedNetwork[3][0].to = 999;
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            Network.deserialize(
              serializedNetwork,
              network.input,
              network.output,
            );
            const warnedAboutInvalidConnectionIndices = warnSpy.mock.calls.some(
              (call) =>
                typeof call[0] === 'string' &&
                call[0].includes('Invalid connection indices'),
            );

            // Assert
            expect(warnedAboutInvalidConnectionIndices).toBe(true);
          } finally {
            warnSpy.mockRestore();
          }
        });
      });
    });

    describe('given the compact payload contains an invalid gater index', () => {
      describe('when deserialize() rebuilds the payload', () => {
        it('skips the invalid gater assignment', () => {
          // Arrange
          const network = createSerializableNetwork(367);
          const serializedNetwork = network.serialize();
          serializedNetwork[3][0].gater = 999;

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );

          // Assert
          expect(deserialized.gates.length).toBeLessThanOrEqual(
            network.gates.length,
          );
        });
      });

      describe('when warnings are observed during rebuild', () => {
        it('warns about the invalid gater index', () => {
          // Arrange
          const network = createSerializableNetwork(368);
          const serializedNetwork = network.serialize();
          serializedNetwork[3][0].gater = 999;
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            Network.deserialize(
              serializedNetwork,
              network.input,
              network.output,
            );
            const warnedAboutInvalidGaterIndex = warnSpy.mock.calls.some(
              (call) =>
                typeof call[0] === 'string' &&
                call[0].includes('Invalid gater index'),
            );

            // Assert
            expect(warnedAboutInvalidGaterIndex).toBe(true);
          } finally {
            warnSpy.mockRestore();
          }
        });
      });
    });

    describe('given the compact payload contains an unknown squash key', () => {
      describe('when deserialize() rebuilds the payload', () => {
        it('falls back to identity activation', () => {
          // Arrange
          const network = createSerializableNetwork(369);
          const serializedNetwork = network.serialize();
          serializedNetwork[2][0] = 'notARealSquashFn' as never;

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );

          // Assert
          expect(deserialized.nodes[0].squash).toBe(
            methods.Activation.identity,
          );
        });
      });

      describe('when warnings are observed during rebuild', () => {
        it('warns about the unknown squash key', () => {
          // Arrange
          const network = createSerializableNetwork(370);
          const serializedNetwork = network.serialize();
          serializedNetwork[2][0] = 'notARealSquashFn' as never;
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            Network.deserialize(
              serializedNetwork,
              network.input,
              network.output,
            );
            const warnedAboutUnknownSquash = warnSpy.mock.calls.some(
              (call) =>
                typeof call[0] === 'string' &&
                call[0].includes('Unknown squash function'),
            );

            // Assert
            expect(warnedAboutUnknownSquash).toBe(true);
          } finally {
            warnSpy.mockRestore();
          }
        });
      });
    });
  });

  describe('Network.fromJSON()', () => {
    describe('given a live runtime carries a stale forward connection reference', () => {
      describe('when the JSON snapshot is built', () => {
        it('skips the stale connection instead of exporting invalid endpoint indices', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(3719);
          const staleNode = new Node('hidden');
          const staleConnection = new Connection(
            staleNode,
            network.nodes.at(-1) ?? network.nodes[0],
            0.5,
          );

          (staleNode as unknown as { index: number }).index = 0;
          network.connections.push(staleConnection);

          // Act
          const serializedJson = network.toJSON() as {
            connections: Array<Record<string, unknown>>;
          };

          // Assert
          expect(serializedJson.connections.length).toBe(
            network.connections.length - 1,
          );
        });
      });
    });

    describe('given one construct-built runtime is serialized to JSON', () => {
      describe('when the rebuilt network is activated with the same input vector', () => {
        it('preserves the activation output values exactly', () => {
          // Arrange
          const { network, activationInputValues } =
            createConstructedSerializationScenario();
          const expectedOutputValues = network.activate(activationInputValues);
          const serializedJson = network.toJSON();

          // Act
          const rebuiltNetwork = Network.fromJSON(serializedJson);
          const actualOutputValues = rebuiltNetwork.activate(
            activationInputValues,
          );

          // Assert
          expect(actualOutputValues).toEqual(expectedOutputValues);
        });
      });

      describe('when the rebuilt network architecture is inspected', () => {
        it('preserves the public architecture descriptor', () => {
          // Arrange
          const { network } = createConstructedSerializationScenario();
          const expectedDescriptor = network.describeArchitecture();
          const serializedJson = network.toJSON();

          // Act
          const rebuiltNetwork = Network.fromJSON(serializedJson);
          const actualDescriptor = rebuiltNetwork.describeArchitecture();

          // Assert
          expect(actualDescriptor).toEqual(expectedDescriptor);
        });
      });
    });

    describe('given a supported architecture is serialized to JSON', () => {
      for (const jsonRoundTripScenario of JSON_ROUND_TRIP_SCENARIOS) {
        describe(`${jsonRoundTripScenario.architectureName}`, () => {
          let originalNetwork: Network;
          let deserializedNetwork: Network;
          let inputValues: number[];
          let originalOutputValues: number[];
          let deserializedOutputValues: number[];

          beforeAll(() => {
            // Arrange
            originalNetwork = jsonRoundTripScenario.createNetwork();
            const serializedJson = originalNetwork.toJSON();
            deserializedNetwork = Network.fromJSON(serializedJson);
            inputValues = createDeterministicInputValues(originalNetwork.input);
            originalOutputValues = originalNetwork.activate(inputValues);
            deserializedOutputValues =
              deserializedNetwork.activate(inputValues);
          });

          describe('when the rebuilt output width is compared to the original', () => {
            it('preserves the output length', () => {
              // Assert
              expect(deserializedOutputValues.length).toBe(
                originalOutputValues.length,
              );
            });
          });

          describe('when the rebuilt output values are compared to the original', () => {
            it('stays numerically close', () => {
              // Arrange
              const outputsStayClose = deserializedOutputValues.every(
                (outputValue, outputIndex) =>
                  Math.abs(outputValue - originalOutputValues[outputIndex]) <
                  1e-9,
              );

              // Assert
              expect(outputsStayClose).toBe(true);
            });
          });

          describe('when the nodes field is removed from the JSON payload', () => {
            it('throws', () => {
              // Arrange
              const serializedJson = originalNetwork.toJSON() as {
                nodes?: unknown;
              } & Record<string, unknown>;
              delete serializedJson.nodes;

              // Act
              const deserializeWithoutNodes = () =>
                Network.fromJSON(serializedJson);

              // Assert
              expect(deserializeWithoutNodes).toThrow();
            });
          });

          describe('when the connections field is removed from the JSON payload', () => {
            it('throws', () => {
              // Arrange
              const serializedJson = originalNetwork.toJSON() as {
                connections?: unknown;
              } & Record<string, unknown>;
              delete serializedJson.connections;

              // Act
              const deserializeWithoutConnections = () =>
                Network.fromJSON(serializedJson);

              // Assert
              expect(deserializeWithoutConnections).toThrow();
            });
          });

          describe('when the JSON payload root is null', () => {
            it('throws an invalid JSON error', () => {
              // Arrange
              const nullJsonPayload = null as unknown as Record<
                string,
                unknown
              >;

              // Act
              const deserializeNullRoot = () =>
                Network.fromJSON(nullJsonPayload);

              // Assert
              expect(deserializeNullRoot).toThrow();
            });
          });

          describe('when one connection entry is not an object', () => {
            it('skips the malformed connection entry', () => {
              // Arrange
              const serializedJson = originalNetwork.toJSON() as {
                connections: unknown[];
              };
              serializedJson.connections = [{}];

              // Act
              const deserializedNetwork = Network.fromJSON(
                serializedJson as unknown as Record<string, unknown>,
              );
              const connectionCount = deserializedNetwork.connections.length;

              // Assert
              expect(connectionCount).toBe(0);
            });
          });
        });
      }
    });

    describe('given a network snapshot uses historical identity fields', () => {
      describe('when the payload is inspected before restore', () => {
        it('includes connection innovations and endpoint gene ids', () => {
          // Arrange
          const network = createSerializableNetwork(3721);
          const serializedJson = network.toJSON() as {
            connections: Array<{
              innovation?: number;
              fromGeneId?: number;
              toGeneId?: number;
            }>;
          };

          // Act
          const connectionIdentity = serializedJson.connections.map(
            ({ innovation, fromGeneId, toGeneId }) => ({
              innovation,
              fromGeneId,
              toGeneId,
            }),
          );

          // Assert
          expect(connectionIdentity).toEqual(
            network.connections.map((connection) => ({
              innovation: connection.innovation,
              fromGeneId: connection.from.geneId,
              toGeneId: connection.to.geneId,
            })),
          );
        });
      });

      describe('when clone() rebuilds the network through JSON', () => {
        it('preserves node ids, node responses, connection innovations, and enabled flags', () => {
          // Arrange
          const network = createSerializableNetwork(3722);
          network.connections[0].enabled = false;
          network.nodes.at(-1)!.response = 1.5;
          network.mutate(methods.mutation.ADD_NODE);

          // Act
          const cloned = network.clone();

          // Assert
          expect({
            nodeGeneIds: cloned.nodes.map((node) => node.geneId),
            nodeResponses: cloned.nodes.map((node) => node.response),
            connectionIdentity: cloned.connections.map((connection) => ({
              innovation: connection.innovation,
              enabled: connection.enabled,
            })),
          }).toEqual({
            nodeGeneIds: network.nodes.map((node) => node.geneId),
            nodeResponses: network.nodes.map((node) => node.response),
            connectionIdentity: network.connections.map((connection) => ({
              innovation: connection.innovation,
              enabled: connection.enabled,
            })),
          });
        });
      });

      describe('when a high-id payload is restored into a fresh process', () => {
        it('advances node and connection counters past the restored maxima', () => {
          // Arrange
          const network = createSerializableNetwork(3723);
          const serializedJson = network.toJSON() as {
            nodes: Array<{ geneId?: number }>;
            connections: Array<{
              innovation?: number;
              fromGeneId?: number;
              toGeneId?: number;
            }>;
          };
          serializedJson.nodes.forEach((node, nodeIndex) => {
            node.geneId = 900 + nodeIndex;
          });
          serializedJson.connections.forEach((connection, connectionIndex) => {
            connection.innovation = 1_200 + connectionIndex;
            connection.fromGeneId = serializedJson.nodes[0].geneId;
            connection.toGeneId = serializedJson.nodes.at(-1)?.geneId;
          });
          (Node as unknown as { _nextGeneId: number })._nextGeneId = 1;
          (
            Connection as unknown as { _nextInnovation: number }
          )._nextInnovation = 1;

          // Act
          const restored = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );
          const nextNode = new Node('hidden');
          const nextConnection = new Connection(
            restored.nodes[0],
            restored.nodes.at(-1) ?? restored.nodes[0],
            0.25,
          );

          // Assert
          expect({
            nodeCounterAdvanced:
              nextNode.geneId > 900 + restored.nodes.length - 1,
            connectionCounterAdvanced:
              nextConnection.innovation >
              1_200 + restored.connections.length - 1,
          }).toEqual({
            nodeCounterAdvanced: true,
            connectionCounterAdvanced: true,
          });
        });
      });
    });

    describe('given the JSON payload contains an unknown squash key', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('falls back to identity activation', () => {
          // Arrange
          const network = new Network(1, 1);
          const serializedJson = network.toJSON() as {
            nodes: Array<Record<string, unknown>>;
          };
          serializedJson.nodes[0].squash = 'UNKNOWN_FUNCTION';

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.nodes[0].squash).toBe(
            methods.Activation.identity,
          );
        });
      });
    });

    describe('given the JSON payload contains invalid connection indices', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('skips the invalid connection row', () => {
          // Arrange
          const network = createSerializableNetwork(371);
          const inputNodeIndex = network.nodes.findIndex(
            (node) => node.type === 'input',
          );
          const outputNodeIndex = network.nodes.findIndex(
            (node) => node.type === 'output',
          );
          const serializedJson: Record<string, unknown> = {
            formatVersion: 1,
            nodes: network.nodes.map((node) => ({
              bias: node.bias,
              type: node.type,
              squash: 'LOGISTIC',
            })),
            connections: [
              { from: inputNodeIndex, to: outputNodeIndex, weight: 0.5 },
              { from: 999, to: outputNodeIndex, weight: 0.5 },
            ],
            input: network.input,
            output: network.output,
          };

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.connections.length).toBe(1);
        });
      });
    });

    describe('given the JSON payload contains an invalid gater index', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('skips the invalid gater assignment', () => {
          // Arrange
          const network = createSerializableNetwork(372);
          const inputNodeIndex = network.nodes.findIndex(
            (node) => node.type === 'input',
          );
          const outputNodeIndex = network.nodes.findIndex(
            (node) => node.type === 'output',
          );
          const serializedJson: Record<string, unknown> = {
            formatVersion: 1,
            nodes: network.nodes.map((node) => ({
              bias: node.bias,
              type: node.type,
              squash: 'LOGISTIC',
            })),
            connections: [
              {
                from: inputNodeIndex,
                to: outputNodeIndex,
                weight: 0.5,
                gater: 999,
              },
            ],
            input: network.input,
            output: network.output,
          };

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.gates.length).toBe(0);
        });
      });
    });

    describe('given the JSON payload contains extra unexpected fields', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('ignores the extra fields', () => {
          // Arrange
          const network = createSerializableNetwork(373);
          const serializedJson = network.toJSON() as Record<string, unknown>;
          serializedJson.extraField = 'shouldBeIgnored';

          // Act
          const deserialized = Network.fromJSON(serializedJson);

          // Assert
          expect(deserialized).toBeInstanceOf(Network);
        });
      });
    });

    describe('given the JSON payload omits optional squash and gater fields', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('uses the fallback squash function', () => {
          // Arrange
          const network = createSerializableNetwork(374);
          const serializedJson = network.toJSON() as {
            nodes: Array<Record<string, unknown>>;
            connections: Array<Record<string, unknown>>;
          };
          delete serializedJson.nodes[0].squash;
          delete serializedJson.connections[0].gater;

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.nodes[0].squash).toBeDefined();
        });
      });

      describe('when the rebuilt connection metadata is inspected', () => {
        it('defaults the gater to null', () => {
          // Arrange
          const network = createSerializableNetwork(375);
          const serializedJson = network.toJSON() as {
            nodes: Array<Record<string, unknown>>;
            connections: Array<Record<string, unknown>>;
          };
          delete serializedJson.nodes[0].squash;
          delete serializedJson.connections[0].gater;

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.connections[0].gater).toBeNull();
        });
      });
    });

    describe('given a serialized connection uses a non-neutral gain', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('preserves the connection gain', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(3811);
          network.connections[0].gain = 1.5;
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);

          // Assert
          expect(deserialized.connections[0].gain).toBe(1.5);
        });
      });
    });

    describe('given the JSON payload carries an explicit temporal-module extension bag', () => {
      describe('when fromJSON() rebuilds the payload and it is serialized again', () => {
        it('preserves the extension bag across the runtime round-trip', () => {
          // Arrange
          const serializedJson = Architect.lstm(
            1,
            2,
            1,
          ).toJSON() as unknown as NetworkJSON;
          serializedJson.extensions =
            createTemporalModuleExtensions(serializedJson);

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );
          const reserialized = deserialized.toJSON() as unknown as NetworkJSON;

          // Assert
          expect(reserialized.extensions).toEqual(serializedJson.extensions);
        });
      });
    });

    describe('given the JSON payload carries an invalid architecture descriptor shape', () => {
      describe('when hiddenLayerSizes is not an array during fromJSON()', () => {
        it('ignores the invalid descriptor and keeps a valid runtime architecture descriptor', () => {
          // Arrange
          const serializedJson = createSerializableNetwork(
            3812,
          ).toJSON() as unknown as NetworkJSON;
          serializedJson.architecture = {
            source: 'layer-metadata',
            hiddenLayerSizes: 'invalid-shape',
          } as unknown as NetworkJSON['architecture'];

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );
          const hasArrayHiddenLayerSizes = Array.isArray(
            deserialized.describeArchitecture().hiddenLayerSizes,
          );

          // Assert
          expect(hasArrayHiddenLayerSizes).toBe(true);
        });
      });
    });

    describe('given the JSON payload carries an invalid generic extension bag shape', () => {
      describe('when extension values are not a plain object during fromJSON()', () => {
        it('does not preserve the invalid extension bag in the next JSON snapshot', () => {
          // Arrange
          const serializedJson = createSerializableNetwork(
            3813,
          ).toJSON() as unknown as NetworkJSON;
          serializedJson.extensions = {
            version: 1,
            values: ['invalid-values-shape'] as unknown as Record<
              string,
              unknown
            >,
          };

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );
          const reserialized = deserialized.toJSON() as unknown as NetworkJSON;

          // Assert
          expect(reserialized.extensions).toBeUndefined();
        });
      });
    });

    describe('given a recurrent builder emits deliberate temporal descriptors', () => {
      describe('when LSTM JSON is serialized without manual extension tagging', () => {
        it('includes one LSTM recurrent module and one gated block', () => {
          // Arrange
          const serializedJson = Architect.lstm(
            1,
            2,
            1,
          ).toJSON() as unknown as NetworkJSON;

          // Act
          const temporalSummary = summarizeTemporalExtensionBag(serializedJson);

          // Assert
          expect(temporalSummary).toEqual({
            recurrentModuleCount: 1,
            gatedBlockCount: 1,
            recurrentKinds: ['lstm'],
          });
        });
      });

      describe('when GRU JSON is serialized without manual extension tagging', () => {
        it('includes one GRU recurrent module and one gated block', () => {
          // Arrange
          const serializedJson = Architect.gru(
            1,
            2,
            1,
          ).toJSON() as unknown as NetworkJSON;

          // Act
          const temporalSummary = summarizeTemporalExtensionBag(serializedJson);

          // Assert
          expect(temporalSummary).toEqual({
            recurrentModuleCount: 1,
            gatedBlockCount: 1,
            recurrentKinds: ['gru'],
          });
        });
      });

      describe('when NARX JSON is serialized without manual extension tagging', () => {
        it('includes one memory-module descriptor per delay line and no gated blocks', () => {
          // Arrange
          const serializedJson = Architect.narx(
            2,
            2,
            1,
            2,
            1,
          ).toJSON() as unknown as NetworkJSON;

          // Act
          const temporalSummary = summarizeTemporalExtensionBag(serializedJson);

          // Assert
          expect(temporalSummary).toEqual({
            recurrentModuleCount: 2,
            gatedBlockCount: 0,
            recurrentKinds: ['narx-memory', 'narx-memory'],
          });
        });
      });
    });

    describe('given one recurrent builder loses a module-owned connection after construction', () => {
      describe('when the edited network is serialized again', () => {
        it('drops the stale temporal descriptors instead of emitting invalid module metadata', () => {
          // Arrange
          const network = Architect.lstm(1, 2, 1);
          const serializedBeforeEdit =
            network.toJSON() as unknown as NetworkJSON;
          const extensionValues = serializedBeforeEdit.extensions?.values as {
            gatedBlocks?: Array<{ connectionInnovations: number[] }>;
          };
          const moduleConnectionInnovation =
            extensionValues.gatedBlocks?.[0]?.connectionInnovations?.[0];
          const moduleConnection = [
            ...network.connections,
            ...network.selfconns,
          ].find(
            (connection) =>
              connection.innovation === moduleConnectionInnovation,
          );

          if (!moduleConnection) {
            throw new Error(
              'Expected one module-owned connection to exist for invalidation coverage.',
            );
          }

          network.disconnect(moduleConnection.from, moduleConnection.to);

          // Act
          const serializedAfterEdit =
            network.toJSON() as unknown as NetworkJSON;
          const temporalSummary =
            summarizeTemporalExtensionBag(serializedAfterEdit);

          // Assert
          expect(temporalSummary).toEqual({
            recurrentModuleCount: 0,
            gatedBlockCount: 0,
            recurrentKinds: [],
          });
        });
      });
    });

    describe('given one recurrent builder disables a module-owned connection after construction', () => {
      describe('when the edited network is serialized again', () => {
        it('keeps the temporal descriptors because disabled genes still count as dormant structure', () => {
          // Arrange
          const network = Architect.lstm(1, 2, 1);
          const serializedBeforeEdit =
            network.toJSON() as unknown as NetworkJSON;
          const moduleConnectionInnovation =
            readFirstTemporalConnectionInnovation(
              serializedBeforeEdit.extensions,
            );
          const moduleConnection = [
            ...network.connections,
            ...network.selfconns,
          ].find(
            (connection) =>
              connection.innovation === moduleConnectionInnovation,
          );

          if (!moduleConnection) {
            throw new Error(
              'Expected one module-owned connection to exist for dormant-state coverage.',
            );
          }

          moduleConnection.enabled = false;

          // Act
          const serializedAfterEdit =
            network.toJSON() as unknown as NetworkJSON;
          const temporalSummary =
            summarizeTemporalExtensionBag(serializedAfterEdit);
          const disabledConnection = serializedAfterEdit.connections.find(
            (connection) =>
              connection.innovation === moduleConnectionInnovation,
          );

          // Assert
          expect({
            temporalSummary,
            enabled: disabledConnection?.enabled ?? null,
          }).toEqual({
            temporalSummary: {
              recurrentModuleCount: 1,
              gatedBlockCount: 1,
              recurrentKinds: ['lstm'],
            },
            enabled: false,
          });
        });
      });
    });

    describe('given the JSON payload contains empty node and connection arrays', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('returns a network with zero nodes', () => {
          // Arrange
          const serializedJson = {
            nodes: [],
            connections: [],
            input: 1,
            output: 1,
          } as Record<string, unknown>;

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.nodes.length).toBe(0);
        });
      });

      describe('when the rebuilt connection collection is inspected', () => {
        it('returns zero connections', () => {
          // Arrange
          const serializedJson = {
            nodes: [],
            connections: [],
            input: 1,
            output: 1,
          } as Record<string, unknown>;

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.connections.length).toBe(0);
        });
      });
    });

    describe('given a serialized network used a custom activation function', () => {
      describe('when the rebuilt squash reference is compared to the original', () => {
        it('does not preserve the original custom function', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(376);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * 100;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);

          // Assert
          expect(deserialized.nodes[0].squash).not.toBe(customSquash);
        });
      });

      describe('when the rebuilt squash output is compared to the original custom output', () => {
        it('changes the functional behavior', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(377);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * 100;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const rebuiltOutputChanged = deserialized.nodes[0].squash(0.5) !== 50;

          // Assert
          expect(rebuiltOutputChanged).toBe(true);
        });
      });

      describe('when the rebuilt network is activated', () => {
        it('remains usable after deserialization', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(378);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * 100;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const activateDeserializedNetwork = () =>
            deserialized.activate([0.5]);

          // Assert
          expect(activateDeserializedNetwork).not.toThrow();
        });
      });
    });

    describe('given a second custom activation function was serialized', () => {
      describe('when the rebuilt squash is inspected', () => {
        it('still exposes a callable squash function', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(379);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * value;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const rebuiltSquashIsCallable =
            typeof deserialized.nodes[0].squash === 'function';

          // Assert
          expect(rebuiltSquashIsCallable).toBe(true);
        });
      });

      describe('when the rebuilt output is compared to the original custom output', () => {
        it('does not preserve the custom activation semantics', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(380);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * value;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();
          const testValue = 0.5;

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const rebuiltOutputChanged =
            deserialized.nodes[0].squash(testValue) !== customSquash(testValue);

          // Assert
          expect(rebuiltOutputChanged).toBe(true);
        });
      });

      describe('when identity is selected as the fallback squash function', () => {
        it('keeps the identity behavior intact', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(381);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * value;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();
          const testValue = 0.5;

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const identityFallbackStayedCorrect =
            deserialized.nodes[0].squash !== methods.Activation.identity ||
            deserialized.nodes[0].squash(testValue) === testValue;

          // Assert
          expect(identityFallbackStayedCorrect).toBe(true);
        });
      });
    });

    describe('given a mutated network is serialized to JSON', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('preserves the expanded node count', () => {
          // Arrange
          const network = createSerializableNetwork(382);
          network.mutate(methods.mutation.ADD_NODE);
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);

          // Assert
          expect(deserialized.nodes.length).toBeGreaterThan(2);
        });
      });
    });
  });
});
