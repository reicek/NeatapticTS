import type Network from '../../architecture/network/network';
import type {
  NetworkJSON,
  NetworkJSONConnection,
  NetworkJSONNode,
  NetworkTopologyIntent,
} from '../../architecture/network/network.types';
import { fromJSONImpl, toJSONImpl } from '../../architecture/network/serialize/network.serialize.utils';
import { validateNetworkJsonOrThrow } from '../../architecture/network/serialize/network.serialize.json.utils';
import { NETWORK_JSON_FORMAT_VERSION } from '../../architecture/network/serialize/network.serialize.utils.types';
import type { ConnectionLike, GenomeLike } from '../compat/core/compat.types';
import {
  NeatGenomeConversionError,
  NeatGenomeValidationError,
} from './genome.errors';
import type {
  GenomeMaterializationRuntimeHints,
  NeatGenome,
  NeatGenomeCaptureOptions,
  NeatGenomeConnectionGene,
  NeatGenomeExtensionValues,
  NeatGenomeGatedBlockDescriptor,
  NeatGenomeExtensions,
  NeatGenomeNodeGene,
  NeatGenomeNodeType,
  NeatGenomeRecurrentModuleDescriptor,
  NeatGenomeRecurrentModuleKind,
  NeatGenomeValidationIssue,
  NeatGenomeValidationIssueCode,
  NeatGenomeValidationReport,
} from './genome.types';

type RuntimeCompatibilitySource = GenomeLike & {
  input?: number;
  output?: number;
  nodes?: Array<{ geneId?: number }>;
  getTopologyIntent?: () => NetworkTopologyIntent;
};

type GenomeCompatibilitySource = NeatGenome & {
  _id?: number;
  _compatInnovationMode?: 'require-explicit' | 'allow-fallback';
};

const runtimeCompatibilityViewCache = new WeakMap<object, GenomeLike>();
const genomeCompatibilityViewCache = new WeakMap<object, GenomeLike>();
const NEAT_GENOME_EXTENSIONS_VERSION = 1;
const NEUTRAL_CONNECTION_GAIN = 1;
const NEUTRAL_NODE_RESPONSE = 1;
const SUPPORTED_RECURRENT_MODULE_KINDS = new Set<NeatGenomeRecurrentModuleKind>([
  'lstm',
  'gru',
  'narx-memory',
]);

/**
 * Convert one executable phenotype into the strict NEAT genome contract.
 *
 * This is the phenotype-to-genome adapter introduced in Step 7.1. It strips
 * runtime-only state and keeps only structural identity plus portable gene
 * attributes.
 *
 * @param network - Executable phenotype.
 * @param captureOptions - Optional opt-in runtime-to-genome extension capture settings.
 * @returns Strict structural genome contract.
 */
export function createGenomeFromNetwork(
  network: Network,
  captureOptions: NeatGenomeCaptureOptions = {},
): NeatGenome {
  const runtimePayload = toJSONImpl.call(network);
  const strictGenome = createGenomeFromNetworkJson(runtimePayload, captureOptions);
  const mergedExtensions = mergeRuntimeOnlyGenomeExtensions(
    strictGenome.extensions,
    network,
    captureOptions,
  );

  if (!mergedExtensions) {
    return strictGenome;
  }

  const runtimeAwareGenome: NeatGenome = {
    ...strictGenome,
    extensions: mergedExtensions,
  };

  assertValidGenomeContract(runtimeAwareGenome);
  return runtimeAwareGenome;
}

/**
 * Convert one versioned network JSON payload into the strict NEAT genome
 * contract.
 *
 * @param networkJson - Versioned phenotype JSON payload.
 * @param captureOptions - Optional opt-in runtime-to-genome extension capture settings.
 * @returns Strict structural genome contract.
 */
export function createGenomeFromNetworkJson(
  networkJson: NetworkJSON,
  captureOptions: NeatGenomeCaptureOptions = {},
): NeatGenome {
  validateNetworkJsonOrThrow(networkJson);

  const runtimeOrderedNodeEntries = networkJson.nodes.toSorted(
    (leftNode, rightNode) => leftNode.index - rightNode.index,
  );
  const nodeGenes = createCanonicalNodeGenes(runtimeOrderedNodeEntries);
  const geneIdsByNodeIndex = createGeneIdLookupByNodeIndex(
    runtimeOrderedNodeEntries,
  );
  const connectionGenes = networkJson.connections.map((connectionJsonEntry) =>
    createConnectionGeneFromJsonConnection(connectionJsonEntry, geneIdsByNodeIndex),
  );
  const extensions = resolveGenomeExtensions(networkJson, captureOptions);

  const genome: NeatGenome = {
    input: networkJson.input,
    output: networkJson.output,
    topologyIntent: networkJson.topologyIntent,
    nodeGenes,
    connectionGenes,
    ...(extensions ? { extensions } : {}),
  };

  assertValidGenomeContract(genome);
  return genome;
}

/**
 * Convert one strict genome contract into the versioned network JSON payload
 * understood by the runtime phenotype serializer.
 *
 * @param genome - Strict structural genome contract.
 * @param runtimeHints - Optional phenotype-only metadata to preserve.
 * @returns Versioned network JSON payload.
 */
export function createNetworkJsonFromGenome(
  genome: NeatGenome,
  runtimeHints: GenomeMaterializationRuntimeHints = {},
): NetworkJSON {
  assertValidGenomeContract(genome);

  const nodeIndexesByGeneId = createNodeIndexLookupByGeneId(genome.nodeGenes);
  const connectionGainByInnovation = readConnectionGainByInnovation(
    genome.extensions,
  );
  const nodeResponseByGeneId = readNodeResponseByGeneId(genome.extensions);
  const nodeJsonEntries = genome.nodeGenes.map(
    (nodeGene, nodeIndex): NetworkJSONNode => ({
      type: nodeGene.type,
      bias: nodeGene.bias,
      ...(typeof nodeResponseByGeneId?.[String(nodeGene.geneId)] === 'number'
        ? { response: nodeResponseByGeneId[String(nodeGene.geneId)] }
        : {}),
      squash: nodeGene.squash,
      index: nodeIndex,
      geneId: nodeGene.geneId,
    }),
  );
  const connectionJsonEntries = genome.connectionGenes.map(
    (connectionGene): NetworkJSONConnection => {
      const resolvedConnectionGain = connectionGainByInnovation?.[
        String(connectionGene.innovation)
      ];

      return {
        from: resolveNodeIndexForGeneId(
          nodeIndexesByGeneId,
          connectionGene.fromGeneId,
          'fromGeneId',
        ),
        to: resolveNodeIndexForGeneId(
          nodeIndexesByGeneId,
          connectionGene.toGeneId,
          'toGeneId',
        ),
        weight: connectionGene.weight,
        ...(typeof resolvedConnectionGain === 'number'
          ? { gain: resolvedConnectionGain }
          : {}),
        gater:
          typeof connectionGene.gaterGeneId === 'number'
            ? resolveNodeIndexForGeneId(
                nodeIndexesByGeneId,
                connectionGene.gaterGeneId,
                'gaterGeneId',
              )
            : null,
        enabled: connectionGene.enabled,
        innovation: connectionGene.innovation,
        fromGeneId: connectionGene.fromGeneId,
        toGeneId: connectionGene.toGeneId,
        gaterGeneId: connectionGene.gaterGeneId,
      };
    },
  );

  return {
    formatVersion: NETWORK_JSON_FORMAT_VERSION,
    input: genome.input,
    output: genome.output,
    dropout: runtimeHints.dropout ?? 0,
    topologyIntent: genome.topologyIntent,
    nodes: nodeJsonEntries,
    connections: connectionJsonEntries,
    ...(genome.extensions
      ? { extensions: structuredClone(genome.extensions) }
      : {}),
    architecture: runtimeHints.architecture,
  };
}

/**
 * Materialize one executable phenotype from the strict genome contract.
 *
 * @param genome - Strict structural genome contract.
 * @param runtimeHints - Optional phenotype-only metadata to preserve.
 * @returns Executable runtime phenotype.
 */
export function createNetworkFromGenome(
  genome: NeatGenome,
  runtimeHints: GenomeMaterializationRuntimeHints = {},
): Network {
  const materializedNetwork = fromJSONImpl(
    createNetworkJsonFromGenome(genome, runtimeHints),
  );
  const disabledConnectionReenableProbability =
    readDisabledConnectionReenableProbability(genome.extensions);

  if (typeof disabledConnectionReenableProbability === 'number') {
    (
      materializedNetwork as Network & { _reenableProb?: number }
    )._reenableProb = disabledConnectionReenableProbability;
  }

  return materializedNetwork;
}

/**
 * Validate one strict genome contract.
 *
 * @param genome - Strict structural genome contract.
 * @returns Structured validation report.
 */
export function validateGenomeContract(
  genome: NeatGenome,
): NeatGenomeValidationReport {
  const issues: NeatGenomeValidationIssue[] = [];

  validateGenomeSize(genome, issues);
  validateNodeGenes(genome, issues);
  validateConnectionGenes(genome, issues);
  validateExtensions(genome, issues);

  return {
    isValid: issues.length === 0,
    input: genome.input,
    output: genome.output,
    topologyIntent: genome.topologyIntent,
    nodeCount: genome.nodeGenes.length,
    connectionCount: genome.connectionGenes.length,
    issues,
  };
}

/**
 * Assert that one strict genome contract is valid.
 *
 * @param genome - Strict structural genome contract.
 * @returns Nothing.
 * @throws {NeatGenomeValidationError} When the contract is malformed.
 */
export function assertValidGenomeContract(genome: NeatGenome): void {
  const validationReport = validateGenomeContract(genome);
  if (validationReport.isValid) {
    return;
  }

  throw new NeatGenomeValidationError(
    buildGenomeValidationFailureMessage(validationReport),
    validationReport.issues,
  );
}

/**
 * Create the compatibility-layer view for a runtime phenotype or strict genome.
 *
 * Native explicit-innovation flows are normalized through the strict genome
 * contract. Deliberate fallback-innovation flows remain on the legacy runtime
 * edge path so compatibility can keep using endpoint-derived synthetic ids.
 *
 * @param source - Runtime genome or strict genome contract.
 * @returns Compatibility-layer genome view.
 */
export function createCompatibilityGenomeView(
  source: GenomeLike | NeatGenome | RuntimeCompatibilitySource,
): GenomeLike {
  if (isNeatGenome(source)) {
    return getOrCreateGenomeCompatibilityView(source);
  }

  if (isRuntimeCompatibilitySource(source)) {
    const runtimeSource = source;
    const hasMissingInnovations = runtimeSource.connections.some(
      (connection) => !Number.isFinite(connection.innovation),
    );

    if (hasMissingInnovations) {
      return runtimeSource;
    }

    return getOrCreateRuntimeCompatibilityView(runtimeSource);
  }

  return source as GenomeLike;
}

function createNodeGeneFromJsonNode(nodeJsonEntry: NetworkJSONNode): NeatGenomeNodeGene {
  return {
    geneId:
      typeof nodeJsonEntry.geneId === 'number'
        ? nodeJsonEntry.geneId
        : Number.NaN,
    type: resolveCanonicalGenomeNodeType(nodeJsonEntry.type),
    bias: nodeJsonEntry.bias,
    squash: nodeJsonEntry.squash,
  };
}

function resolveCanonicalGenomeNodeType(nodeType: string): NeatGenomeNodeType {
  if (nodeType === 'input' || nodeType === 'output') {
    return nodeType;
  }

  return 'hidden';
}

function createGeneIdLookupByNodeIndex(
  sortedNodeEntries: NetworkJSONNode[],
): Map<number, number> {
  return new Map(
    sortedNodeEntries.map((nodeJsonEntry) => [
      nodeJsonEntry.index,
      typeof nodeJsonEntry.geneId === 'number'
        ? nodeJsonEntry.geneId
        : Number.NaN,
    ]),
  );
}

function createConnectionGeneFromJsonConnection(
  connectionJsonEntry: NetworkJSONConnection,
  geneIdsByNodeIndex: Map<number, number>,
): NeatGenomeConnectionGene {
  return {
    innovation:
      typeof connectionJsonEntry.innovation === 'number'
        ? connectionJsonEntry.innovation
        : Number.NaN,
    fromGeneId:
      typeof connectionJsonEntry.fromGeneId === 'number'
        ? connectionJsonEntry.fromGeneId
        : (geneIdsByNodeIndex.get(connectionJsonEntry.from) ?? Number.NaN),
    toGeneId:
      typeof connectionJsonEntry.toGeneId === 'number'
        ? connectionJsonEntry.toGeneId
        : (geneIdsByNodeIndex.get(connectionJsonEntry.to) ?? Number.NaN),
    weight: connectionJsonEntry.weight,
    enabled: connectionJsonEntry.enabled !== false,
    gaterGeneId:
      connectionJsonEntry.gater == null
        ? null
        : typeof connectionJsonEntry.gaterGeneId === 'number'
          ? connectionJsonEntry.gaterGeneId
          : (geneIdsByNodeIndex.get(connectionJsonEntry.gater) ?? Number.NaN),
  };
}

function resolveGenomeExtensions(
  networkJson: NetworkJSON,
  captureOptions: NeatGenomeCaptureOptions,
): NeatGenomeExtensions | undefined {
  const explicitExtensions = cloneExplicitExtensions(networkJson.extensions);
  if (explicitExtensions) {
    return explicitExtensions;
  }

  const extensionValues: NeatGenomeExtensionValues = {};

  if (captureOptions.connectionGain === true) {
    const connectionGainByInnovation = collectConnectionGainByInnovation(
      networkJson.connections,
    );
    if (connectionGainByInnovation) {
      extensionValues.connectionGainByInnovation = connectionGainByInnovation;
    }
  }

  if (captureOptions.nodeResponse === true) {
    const nodeResponseByGeneId = collectNodeResponseByGeneId(networkJson.nodes);
    if (nodeResponseByGeneId) {
      extensionValues.nodeResponseByGeneId = nodeResponseByGeneId;
    }
  }

  if (Object.keys(extensionValues).length === 0) {
    return undefined;
  }

  return {
    version: NEAT_GENOME_EXTENSIONS_VERSION,
    values: extensionValues,
  };
}

function mergeRuntimeOnlyGenomeExtensions(
  existingExtensions: NeatGenomeExtensions | undefined,
  network: Network,
  captureOptions: NeatGenomeCaptureOptions,
): NeatGenomeExtensions | undefined {
  if (captureOptions.disabledConnectionReenableProbability !== true) {
    return existingExtensions;
  }

  const disabledConnectionReenableProbability =
    readRuntimeDisabledConnectionReenableProbability(network);
  if (typeof disabledConnectionReenableProbability === 'undefined') {
    return existingExtensions;
  }

  return {
    version:
      existingExtensions?.version ?? NEAT_GENOME_EXTENSIONS_VERSION,
    values: {
      ...(existingExtensions?.values ?? {}),
      disabledConnectionReenableProbability,
    },
  };
}

function cloneExplicitExtensions(
  extensions: NetworkJSON['extensions'],
): NeatGenomeExtensions | undefined {
  if (!extensions) {
    return undefined;
  }

  return {
    version: extensions.version,
    values: isPlainObjectRecord(extensions.values)
      ? { ...extensions.values }
      : (extensions.values as NeatGenomeExtensionValues),
  };
}

function collectConnectionGainByInnovation(
  connectionJsonEntries: NetworkJSONConnection[],
): Record<string, number> | undefined {
  const connectionGainByInnovation = Object.fromEntries(
    connectionJsonEntries
      .filter(isEligibleConnectionGainExtensionEntry)
      .map((connectionJsonEntry) => [
        String(connectionJsonEntry.innovation),
        connectionJsonEntry.gain as number,
      ]),
  );

  return Object.keys(connectionGainByInnovation).length > 0
    ? connectionGainByInnovation
    : undefined;
}

function collectNodeResponseByGeneId(
  nodeJsonEntries: NetworkJSONNode[],
): Record<string, number> | undefined {
  const nodeResponseByGeneId = Object.fromEntries(
    nodeJsonEntries
      .filter(isEligibleNodeResponseExtensionEntry)
      .map((nodeJsonEntry) => [
        String(nodeJsonEntry.geneId),
        nodeJsonEntry.response as number,
      ]),
  );

  return Object.keys(nodeResponseByGeneId).length > 0
    ? nodeResponseByGeneId
    : undefined;
}

function isEligibleConnectionGainExtensionEntry(
  connectionJsonEntry: NetworkJSONConnection,
): boolean {
  return (
    connectionJsonEntry.gater == null &&
    Number.isFinite(connectionJsonEntry.innovation) &&
    Number.isFinite(connectionJsonEntry.gain) &&
    connectionJsonEntry.gain !== NEUTRAL_CONNECTION_GAIN
  );
}

function isEligibleNodeResponseExtensionEntry(
  nodeJsonEntry: NetworkJSONNode,
): boolean {
  return (
    Number.isFinite(nodeJsonEntry.geneId) &&
    Number.isFinite(nodeJsonEntry.response) &&
    nodeJsonEntry.response !== NEUTRAL_NODE_RESPONSE
  );
}

function readConnectionGainByInnovation(
  extensions: NeatGenomeExtensions | undefined,
): Record<string, number> | undefined {
  const connectionGainByInnovation = (
    extensions?.values as NeatGenomeExtensionValues | undefined
  )?.connectionGainByInnovation;

  return isPlainObjectRecord(connectionGainByInnovation)
    ? (connectionGainByInnovation as Record<string, number>)
    : undefined;
}

function readNodeResponseByGeneId(
  extensions: NeatGenomeExtensions | undefined,
): Record<string, number> | undefined {
  const nodeResponseByGeneId = (
    extensions?.values as NeatGenomeExtensionValues | undefined
  )?.nodeResponseByGeneId;

  return isPlainObjectRecord(nodeResponseByGeneId)
    ? (nodeResponseByGeneId as Record<string, number>)
    : undefined;
}

function readDisabledConnectionReenableProbability(
  extensions: NeatGenomeExtensions | undefined,
): number | undefined {
  const disabledConnectionReenableProbability = (
    extensions?.values as NeatGenomeExtensionValues | undefined
  )?.disabledConnectionReenableProbability;

  return typeof disabledConnectionReenableProbability === 'number' &&
    Number.isFinite(disabledConnectionReenableProbability)
    ? disabledConnectionReenableProbability
    : undefined;
}

function readRuntimeDisabledConnectionReenableProbability(
  network: Network,
): number | undefined {
  const runtimeNetwork = network as Network & { _reenableProb?: number };

  return typeof runtimeNetwork._reenableProb === 'number'
    ? runtimeNetwork._reenableProb
    : undefined;
}

function createNodeIndexLookupByGeneId(
  nodeGenes: NeatGenomeNodeGene[],
): Map<number, number> {
  return new Map(
    nodeGenes.map((nodeGene, nodeIndex) => [nodeGene.geneId, nodeIndex]),
  );
}

function resolveNodeIndexForGeneId(
  nodeIndexesByGeneId: Map<number, number>,
  geneId: number,
  label: 'fromGeneId' | 'toGeneId' | 'gaterGeneId',
): number {
  const resolvedIndex = nodeIndexesByGeneId.get(geneId);
  return resolvedIndex as number;
}

function validateGenomeSize(
  genome: NeatGenome,
  issues: NeatGenomeValidationIssue[],
): void {
  if (!Number.isInteger(genome.input) || genome.input < 0) {
    issues.push(
      createIssue(
        'invalid-input-count',
        'input',
        'Strict genomes must carry a non-negative integer input count.',
      ),
    );
  }

  if (!Number.isInteger(genome.output) || genome.output < 0) {
    issues.push(
      createIssue(
        'invalid-output-count',
        'output',
        'Strict genomes must carry a non-negative integer output count.',
      ),
    );
  }

  if (genome.nodeGenes.length < genome.input + genome.output) {
    issues.push(
      createIssue(
        'insufficient-node-count',
        'nodeGenes',
        'Strict genomes must carry at least enough nodes to satisfy the public input/output contract.',
        {
          nodeCount: genome.nodeGenes.length,
          requiredNodeCount: genome.input + genome.output,
        },
      ),
    );
  }
}

function validateNodeGenes(
  genome: NeatGenome,
  issues: NeatGenomeValidationIssue[],
): void {
  const nodePathsByGeneId = new Map<number, string>();

  for (let nodeIndex = 0; nodeIndex < genome.nodeGenes.length; nodeIndex++) {
    const nodeGene = genome.nodeGenes[nodeIndex];
    const nodePath = `nodeGenes[${nodeIndex}]`;

    if (!Number.isFinite(nodeGene.geneId)) {
      issues.push(
        createIssue(
          'missing-node-gene-id',
          `${nodePath}.geneId`,
          'Strict genomes must assign a finite geneId to every node gene.',
        ),
      );
    } else {
      const firstPath = nodePathsByGeneId.get(nodeGene.geneId);
      if (firstPath) {
        issues.push(
          createIssue(
            'duplicate-node-gene-id',
            `${nodePath}.geneId`,
            'Node gene ids must stay unique across one strict genome.',
            {
              duplicateGeneId: nodeGene.geneId,
              firstPath,
            },
          ),
        );
      } else {
        nodePathsByGeneId.set(nodeGene.geneId, `${nodePath}.geneId`);
      }
    }

    if (!isNodeType(nodeGene.type)) {
      issues.push(
        createIssue(
          'invalid-node-type',
          `${nodePath}.type`,
          'Node genes must use one of the supported canonical node-role literals.',
        ),
      );
    }

    if (!Number.isFinite(nodeGene.bias)) {
      issues.push(
        createIssue(
          'invalid-node-bias',
          `${nodePath}.bias`,
          'Node-gene bias values must stay finite in the strict genome contract.',
        ),
      );
    }

    if (typeof nodeGene.squash !== 'string' || nodeGene.squash.length === 0) {
      issues.push(
        createIssue(
          'invalid-node-squash',
          `${nodePath}.squash`,
          'Node genes must carry one stable activation identifier.',
        ),
      );
    }

    if (nodeIndex < genome.input && nodeGene.type !== 'input') {
      issues.push(
        createIssue(
          'input-node-order-mismatch',
          `${nodePath}.type`,
          'Input-region node order must stay at the front of the strict genome contract.',
        ),
      );
    }

    if (
      nodeIndex >= genome.nodeGenes.length - genome.output &&
      nodeGene.type !== 'output'
    ) {
      issues.push(
        createIssue(
          'output-node-order-mismatch',
          `${nodePath}.type`,
          'Output-region node order must stay at the end of the strict genome contract.',
        ),
      );
    }

    if (
      nodeIndex >= genome.input &&
      nodeIndex < genome.nodeGenes.length - genome.output &&
      nodeGene.type !== 'hidden'
    ) {
      issues.push(
        createIssue(
          nodeGene.type === 'input'
            ? 'input-node-order-mismatch'
            : 'output-node-order-mismatch',
          `${nodePath}.type`,
          'Only hidden node genes may occupy the interior node-order region.',
        ),
      );
    }
  }
}

function validateConnectionGenes(
  genome: NeatGenome,
  issues: NeatGenomeValidationIssue[],
): void {
  const connectionPathsByInnovation = new Map<number, string>();
  const nodeIndexesByGeneId = createNodeIndexLookupByGeneId(genome.nodeGenes);

  for (
    let connectionIndex = 0;
    connectionIndex < genome.connectionGenes.length;
    connectionIndex++
  ) {
    const connectionGene = genome.connectionGenes[connectionIndex];
    const connectionPath = `connectionGenes[${connectionIndex}]`;

    if (!Number.isFinite(connectionGene.innovation)) {
      issues.push(
        createIssue(
          'missing-connection-innovation',
          `${connectionPath}.innovation`,
          'Strict genomes must assign a finite innovation id to every connection gene.',
        ),
      );
    } else {
      const firstPath = connectionPathsByInnovation.get(connectionGene.innovation);
      if (firstPath) {
        issues.push(
          createIssue(
            'duplicate-connection-innovation',
            `${connectionPath}.innovation`,
            'Connection innovations must stay unique across one strict genome.',
            {
              duplicateInnovation: connectionGene.innovation,
              firstPath,
            },
          ),
        );
      } else {
        connectionPathsByInnovation.set(
          connectionGene.innovation,
          `${connectionPath}.innovation`,
        );
      }
    }

    if (!Number.isFinite(connectionGene.weight)) {
      issues.push(
        createIssue(
          'invalid-connection-weight',
          `${connectionPath}.weight`,
          'Connection-gene weights must stay finite in the strict genome contract.',
        ),
      );
    }

    validateResolvedConnectionGeneEndpoint(
      connectionGene.fromGeneId,
      `${connectionPath}.fromGeneId`,
      nodeIndexesByGeneId,
      issues,
    );
    validateResolvedConnectionGeneEndpoint(
      connectionGene.toGeneId,
      `${connectionPath}.toGeneId`,
      nodeIndexesByGeneId,
      issues,
    );

    if (connectionGene.gaterGeneId !== null) {
      validateResolvedConnectionGeneEndpoint(
        connectionGene.gaterGeneId,
        `${connectionPath}.gaterGeneId`,
        nodeIndexesByGeneId,
        issues,
        'unknown-gater-node',
      );
    }

    if (genome.topologyIntent === 'feed-forward') {
      const sourceIndex = nodeIndexesByGeneId.get(connectionGene.fromGeneId);
      const targetIndex = nodeIndexesByGeneId.get(connectionGene.toGeneId);

      if (
        typeof sourceIndex === 'number' &&
        typeof targetIndex === 'number' &&
        sourceIndex >= targetIndex
      ) {
        issues.push(
          createIssue(
            'feed-forward-recurrent-connection',
            connectionPath,
            'Feed-forward genome contracts must not carry backward or self connection genes.',
            {
              fromGeneId: connectionGene.fromGeneId,
              toGeneId: connectionGene.toGeneId,
            },
          ),
        );
      }
    }
  }
}

function validateResolvedConnectionGeneEndpoint(
  geneId: number,
  path: string,
  nodeIndexesByGeneId: Map<number, number>,
  issues: NeatGenomeValidationIssue[],
  code: Extract<
    NeatGenomeValidationIssueCode,
    'unknown-connection-endpoint' | 'unknown-gater-node'
  > = 'unknown-connection-endpoint',
): void {
  if (!Number.isFinite(geneId) || !nodeIndexesByGeneId.has(geneId)) {
    issues.push(
      createIssue(
        code,
        path,
        code === 'unknown-gater-node'
          ? 'Gater node genes must resolve to one known node gene in the strict genome contract.'
          : 'Connection endpoints must resolve to known node genes in the strict genome contract.',
        { geneId },
      ),
    );
  }
}

function validateExtensions(
  genome: NeatGenome,
  issues: NeatGenomeValidationIssue[],
): void {
  const extensions = genome.extensions;
  if (!extensions) {
    return;
  }

  const hasValidVersion = Number.isInteger(extensions.version) && extensions.version > 0;
  const hasValidValues = isPlainObjectRecord(extensions.values);

  if (!hasValidVersion || !hasValidValues) {
    issues.push(
      createIssue(
        'invalid-extensions-bag',
        'extensions',
        'Genome extension bags must carry a positive integer version and one plain object payload.',
      ),
    );
    return;
  }

  validateConnectionGainExtension(
    (extensions.values as NeatGenomeExtensionValues).connectionGainByInnovation,
    genome.connectionGenes,
    issues,
  );
  validateNodeResponseExtension(
    (extensions.values as NeatGenomeExtensionValues).nodeResponseByGeneId,
    genome.nodeGenes,
    issues,
  );
  validateConnectionReenableExtension(
    (extensions.values as NeatGenomeExtensionValues)
      .disabledConnectionReenableProbability,
    issues,
  );
  validateRecurrentModuleExtension(
    (extensions.values as NeatGenomeExtensionValues).recurrentModules,
    genome.nodeGenes,
    genome.connectionGenes,
    issues,
  );
  validateGatedBlockExtension(
    (extensions.values as NeatGenomeExtensionValues).gatedBlocks,
    genome.nodeGenes,
    genome.connectionGenes,
    issues,
  );
}

function validateConnectionGainExtension(
  connectionGainByInnovation:
    | NeatGenomeExtensionValues['connectionGainByInnovation']
    | undefined,
  connectionGenes: NeatGenomeConnectionGene[],
  issues: NeatGenomeValidationIssue[],
): void {
  if (typeof connectionGainByInnovation === 'undefined') {
    return;
  }

  if (!isPlainObjectRecord(connectionGainByInnovation)) {
    issues.push(
      createIssue(
        'invalid-connection-gain-extension',
        'extensions.values.connectionGainByInnovation',
        'Connection-gain extensions must be stored as one plain object keyed by connection innovation.',
      ),
    );
    return;
  }

  const connectionGenesByInnovation = new Map(
    connectionGenes.map((connectionGene) => [
      connectionGene.innovation,
      connectionGene,
    ]),
  );

  for (const [innovationKey, gainValue] of Object.entries(
    connectionGainByInnovation,
  )) {
    const parsedInnovation = Number(innovationKey);
    const matchedConnectionGene = connectionGenesByInnovation.get(
      parsedInnovation,
    );
    const hasValidGain =
      typeof gainValue === 'number' &&
      Number.isFinite(gainValue) &&
      gainValue !== NEUTRAL_CONNECTION_GAIN;

    if (
      !Number.isFinite(parsedInnovation) ||
      !matchedConnectionGene ||
      matchedConnectionGene.gaterGeneId !== null ||
      !hasValidGain
    ) {
      issues.push(
        createIssue(
          'invalid-connection-gain-extension',
          `extensions.values.connectionGainByInnovation.${innovationKey}`,
          'Connection-gain extensions must target known ungated connection innovations and carry finite non-neutral gain values.',
          {
            innovation: innovationKey,
            gain: gainValue,
          },
        ),
      );
    }
  }
}

function validateNodeResponseExtension(
  nodeResponseByGeneId:
    | NeatGenomeExtensionValues['nodeResponseByGeneId']
    | undefined,
  nodeGenes: NeatGenomeNodeGene[],
  issues: NeatGenomeValidationIssue[],
): void {
  if (typeof nodeResponseByGeneId === 'undefined') {
    return;
  }

  if (!isPlainObjectRecord(nodeResponseByGeneId)) {
    issues.push(
      createIssue(
        'invalid-node-response-extension',
        'extensions.values.nodeResponseByGeneId',
        'Node-response extensions must be stored as one plain object keyed by node gene id.',
      ),
    );
    return;
  }

  const nodeGenesById = new Map(
    nodeGenes.map((nodeGene) => [nodeGene.geneId, nodeGene]),
  );

  for (const [geneIdKey, responseValue] of Object.entries(nodeResponseByGeneId)) {
    const parsedGeneId = Number(geneIdKey);
    const matchedNodeGene = nodeGenesById.get(parsedGeneId);
    const hasValidResponse =
      typeof responseValue === 'number' &&
      Number.isFinite(responseValue) &&
      responseValue !== NEUTRAL_NODE_RESPONSE;

    if (!Number.isFinite(parsedGeneId) || !matchedNodeGene || !hasValidResponse) {
      issues.push(
        createIssue(
          'invalid-node-response-extension',
          `extensions.values.nodeResponseByGeneId.${geneIdKey}`,
          'Node-response extensions must target known node gene ids and carry finite non-neutral response values.',
          {
            geneId: geneIdKey,
            response: responseValue,
          },
        ),
      );
    }
  }
}

function validateConnectionReenableExtension(
  disabledConnectionReenableProbability:
    | NeatGenomeExtensionValues['disabledConnectionReenableProbability']
    | undefined,
  issues: NeatGenomeValidationIssue[],
): void {
  if (typeof disabledConnectionReenableProbability === 'undefined') {
    return;
  }

  if (
    typeof disabledConnectionReenableProbability !== 'number' ||
    !Number.isFinite(disabledConnectionReenableProbability) ||
    disabledConnectionReenableProbability < 0 ||
    disabledConnectionReenableProbability > 1
  ) {
    issues.push(
      createIssue(
        'invalid-connection-reenable-extension',
        'extensions.values.disabledConnectionReenableProbability',
        'Disabled-connection re-enable extensions must carry one finite probability between 0 and 1.',
        {
          reenableProbability: disabledConnectionReenableProbability,
        },
      ),
    );
  }
}

function validateRecurrentModuleExtension(
  recurrentModules:
    | NeatGenomeExtensionValues['recurrentModules']
    | undefined,
  nodeGenes: NeatGenomeNodeGene[],
  connectionGenes: NeatGenomeConnectionGene[],
  issues: NeatGenomeValidationIssue[],
): void {
  if (typeof recurrentModules === 'undefined') {
    return;
  }

  if (!Array.isArray(recurrentModules)) {
    issues.push(
      createIssue(
        'invalid-recurrent-module-extension',
        'extensions.values.recurrentModules',
        'Recurrent-module extensions must be stored as one array of supported module descriptors.',
      ),
    );
    return;
  }

  const knownNodeGeneIds = new Set(nodeGenes.map((nodeGene) => nodeGene.geneId));
  const knownConnectionInnovations = new Set(
    connectionGenes.map((connectionGene) => connectionGene.innovation),
  );

  recurrentModules.forEach((recurrentModule, moduleIndex) => {
    if (
      !isValidRecurrentModuleDescriptor(
        recurrentModule,
        knownNodeGeneIds,
        knownConnectionInnovations,
      )
    ) {
      issues.push(
        createIssue(
          'invalid-recurrent-module-extension',
          `extensions.values.recurrentModules[${moduleIndex}]`,
          'Recurrent-module extensions must declare a supported kind, one non-empty role map of known node gene ids, and known connection innovations.',
          { moduleIndex },
        ),
      );
    }
  });
}

function validateGatedBlockExtension(
  gatedBlocks: NeatGenomeExtensionValues['gatedBlocks'] | undefined,
  nodeGenes: NeatGenomeNodeGene[],
  connectionGenes: NeatGenomeConnectionGene[],
  issues: NeatGenomeValidationIssue[],
): void {
  if (typeof gatedBlocks === 'undefined') {
    return;
  }

  if (!Array.isArray(gatedBlocks)) {
    issues.push(
      createIssue(
        'invalid-gated-block-extension',
        'extensions.values.gatedBlocks',
        'Gated-block extensions must be stored as one array of block descriptors.',
      ),
    );
    return;
  }

  const knownNodeGeneIds = new Set(nodeGenes.map((nodeGene) => nodeGene.geneId));
  const connectionGenesByInnovation = new Map(
    connectionGenes.map((connectionGene) => [
      connectionGene.innovation,
      connectionGene,
    ]),
  );

  gatedBlocks.forEach((gatedBlock, blockIndex) => {
    if (
      !isValidGatedBlockDescriptor(
        gatedBlock,
        knownNodeGeneIds,
        connectionGenesByInnovation,
      )
    ) {
      issues.push(
        createIssue(
          'invalid-gated-block-extension',
          `extensions.values.gatedBlocks[${blockIndex}]`,
          'Gated-block extensions must reference known gater node gene ids and known gated connection innovations.',
          { blockIndex },
        ),
      );
    }
  });
}

function isValidRecurrentModuleDescriptor(
  recurrentModule: unknown,
  knownNodeGeneIds: Set<number>,
  knownConnectionInnovations: Set<number>,
): recurrentModule is NeatGenomeRecurrentModuleDescriptor {
  if (!isPlainObjectRecord(recurrentModule)) {
    return false;
  }

  const moduleId = recurrentModule.moduleId;
  const kind = recurrentModule.kind;
  const nodeGeneIdsByRole = recurrentModule.nodeGeneIdsByRole;
  const connectionInnovations = recurrentModule.connectionInnovations;

  return (
    typeof moduleId === 'string' &&
    moduleId.length > 0 &&
    typeof kind === 'string' &&
    SUPPORTED_RECURRENT_MODULE_KINDS.has(kind as NeatGenomeRecurrentModuleKind) &&
    isPlainObjectRecord(nodeGeneIdsByRole) &&
    Object.keys(nodeGeneIdsByRole).length > 0 &&
    Object.values(nodeGeneIdsByRole).every((roleNodeGeneIds) =>
      isKnownNonEmptyNumberArray(roleNodeGeneIds, knownNodeGeneIds),
    ) &&
    isKnownNonEmptyNumberArray(
      connectionInnovations,
      knownConnectionInnovations,
    )
  );
}

function isValidGatedBlockDescriptor(
  gatedBlock: unknown,
  knownNodeGeneIds: Set<number>,
  connectionGenesByInnovation: Map<number, NeatGenomeConnectionGene>,
): gatedBlock is NeatGenomeGatedBlockDescriptor {
  if (!isPlainObjectRecord(gatedBlock)) {
    return false;
  }

  const blockId = gatedBlock.blockId;
  const gaterGeneIds = gatedBlock.gaterGeneIds;
  const connectionInnovations = gatedBlock.connectionInnovations;

  if (
    typeof blockId !== 'string' ||
    blockId.length === 0 ||
    !isKnownNonEmptyNumberArray(gaterGeneIds, knownNodeGeneIds) ||
    !Array.isArray(connectionInnovations) ||
    connectionInnovations.length === 0
  ) {
    return false;
  }

  const gaterGeneIdSet = new Set(gaterGeneIds as number[]);
  return connectionInnovations.every((connectionInnovation) => {
    if (
      typeof connectionInnovation !== 'number' ||
      !Number.isFinite(connectionInnovation)
    ) {
      return false;
    }

    const matchedConnectionGene = connectionGenesByInnovation.get(
      connectionInnovation,
    );

    return (
      !!matchedConnectionGene &&
      typeof matchedConnectionGene.gaterGeneId === 'number' &&
      gaterGeneIdSet.has(matchedConnectionGene.gaterGeneId)
    );
  });
}

function isKnownNonEmptyNumberArray(
  value: unknown,
  knownNumbers: Set<number>,
): value is number[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (entry) =>
        typeof entry === 'number' &&
        Number.isFinite(entry) &&
        knownNumbers.has(entry),
    )
  );
}

function isPlainObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function createIssue(
  code: NeatGenomeValidationIssueCode,
  path: string,
  message: string,
  detail?: Record<string, unknown>,
): NeatGenomeValidationIssue {
  return { code, path, message, detail };
}

function buildGenomeValidationFailureMessage(
  validationReport: NeatGenomeValidationReport,
): string {
  const firstIssue = validationReport.issues[0]!;
  return `${firstIssue.message} (${firstIssue.path})`;
}

function isNodeType(nodeType: string): nodeType is NeatGenomeNodeType {
  return nodeType === 'input' || nodeType === 'hidden' || nodeType === 'output';
}

function isNeatGenome(value: unknown): value is GenomeCompatibilitySource {
  return (
    !!value &&
    typeof value === 'object' &&
    Array.isArray((value as NeatGenome).nodeGenes) &&
    Array.isArray((value as NeatGenome).connectionGenes)
  );
}

function isRuntimeCompatibilitySource(
  value: unknown,
): value is RuntimeCompatibilitySource {
  return (
    !!value &&
    typeof value === 'object' &&
    Array.isArray((value as RuntimeCompatibilitySource).connections) &&
    Array.isArray((value as RuntimeCompatibilitySource).nodes)
  );
}

function getOrCreateRuntimeCompatibilityView(
  source: RuntimeCompatibilitySource,
): GenomeLike {
  const cachedView = runtimeCompatibilityViewCache.get(source as object);
  if (cachedView) {
    return cachedView;
  }

  const compatibilityView = {
    get _id() {
      return source._id;
    },
    get _compatInnovationMode() {
      return source._compatInnovationMode;
    },
    get _compatCache() {
      return source._compatCache;
    },
    set _compatCache(cacheEntries: Array<[number, number]> | undefined) {
      source._compatCache = cacheEntries;
    },
    get connections(): ConnectionLike[] {
      return createGenomeFromNetwork(source as unknown as Network).connectionGenes.map(
        (connectionGene) => ({
          innovation: connectionGene.innovation,
          weight: connectionGene.weight,
        }),
      );
    },
  } satisfies GenomeLike;

  runtimeCompatibilityViewCache.set(source as object, compatibilityView);
  return compatibilityView;
}

function getOrCreateGenomeCompatibilityView(
  source: GenomeCompatibilitySource,
): GenomeLike {
  const cachedView = genomeCompatibilityViewCache.get(source as object);
  if (cachedView) {
    return cachedView;
  }

  let cachedInnovationView: Array<[number, number]> | undefined;
  const compatibilityView = {
    get _id() {
      return source._id;
    },
    get _compatInnovationMode() {
      return source._compatInnovationMode;
    },
    get _compatCache() {
      return cachedInnovationView;
    },
    set _compatCache(cacheEntries: Array<[number, number]> | undefined) {
      cachedInnovationView = cacheEntries;
    },
    get connections(): ConnectionLike[] {
      return source.connectionGenes.map((connectionGene) => ({
        innovation: connectionGene.innovation,
        weight: connectionGene.weight,
      }));
    },
  } satisfies GenomeLike;

  genomeCompatibilityViewCache.set(source as object, compatibilityView);
  return compatibilityView;
}

function createCanonicalNodeGenes(
  runtimeOrderedNodeEntries: NetworkJSONNode[],
): NeatGenomeNodeGene[] {
  return runtimeOrderedNodeEntries
    .toSorted(compareCanonicalNodeEntries)
    .map(createNodeGeneFromJsonNode);
}

function compareCanonicalNodeEntries(
  leftNode: NetworkJSONNode,
  rightNode: NetworkJSONNode,
): number {
  const typeRankDelta =
    resolveCanonicalNodeTypeRank(leftNode.type) -
    resolveCanonicalNodeTypeRank(rightNode.type);
  if (typeRankDelta !== 0) {
    return typeRankDelta;
  }

  return leftNode.index - rightNode.index;
}

function resolveCanonicalNodeTypeRank(nodeType: string): number {
  if (nodeType === 'input') {
    return 0;
  }

  if (nodeType === 'output') {
    return 2;
  }

  return 1;
}