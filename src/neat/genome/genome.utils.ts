import type Network from '../../architecture/network/network';
import type {
  NetworkJSON,
  NetworkJSONConnection,
  NetworkJSONNode,
  NetworkTopologyIntent,
} from '../../architecture/network/network.types';
import {
  fromJSONImpl,
  toJSONImpl,
} from '../../architecture/network/serialize/network.serialize.utils';
import { validateNetworkJsonOrThrow } from '../../architecture/network/serialize/network.serialize.json.utils';
import { NETWORK_JSON_FORMAT_VERSION } from '../../architecture/network/serialize/network.serialize.utils.types';
import type { ConnectionLike, GenomeLike } from '../compat/core/compat.types';
import { NeatGenomeValidationError } from './genome.errors';
import {
  NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE,
  NEAT_GENOME_EPISODIC_SLOT_EVICTION_POLICY_CATALOGUE,
} from './genome.types';
import type {
  GenomeMaterializationRuntimeHints,
  NeatGenome,
  NeatGenomeComputationType,
  NeatGenomeCaptureOptions,
  NeatGenomeConnectionGene,
  NeatGenomeEpisodicSlotEvictionPolicy,
  NeatGenomeExtensionValues,
  NeatGenomeGatingRouterArchetypeDescriptor,
  NeatGenomeGatingRouterMode,
  NeatGenomeGatedBlockDescriptor,
  NeatGenomeExtensions,
  NeatGenomeModulatorBroadcasterArchetypeDescriptor,
  NeatGenomeModulatorBroadcasterInputSourceSpec,
  NeatGenomeModuleArchetypeDescriptor,
  NeatGenomeNodeGene,
  NeatGenomeNodeType,
  NeatGenomeRecurrentModuleDescriptor,
  NeatGenomeRecurrentModuleKind,
  NeatGenomeResidualStreamDescriptor,
  NeatGenomeSubstrateCoordinate,
  NeatGenomeValidationIssue,
  NeatGenomeValidationIssueCode,
  NeatGenomeValidationReport,
  NeatGenomeWeightSharedCohortDescriptor,
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

type NgePrimitiveActivationCoordinates = NeatGenomeSubstrateCoordinate;

type MaterializedAttentionHeadPrimitiveModule = {
  activate: (
    inputValues: number[],
    coordinates?: NgePrimitiveActivationCoordinates,
  ) => number[];
  archetypeId: string;
  computationType: 'AttentionHead';
  heads: number;
  outputWidth: number;
  parameterSchema: Record<string, unknown>;
  receivesCoordinates: boolean;
  residualStreamId?: string;
  weightSharedCohortId?: string;
};

type MaterializedGatedRecurrentCellPrimitiveModule = {
  activate: (
    inputValues: number[],
    coordinates?: NgePrimitiveActivationCoordinates,
  ) => number[];
  archetypeId: string;
  clear: () => void;
  computationType: 'GatedRecurrentCell';
  decayRate: number;
  hiddenDim: number;
  parameterSchema: Record<string, unknown>;
  receivesCoordinates: boolean;
  residualStreamId?: string;
  state: number[];
  weightSharedCohortId?: string;
};

type MaterializedEpisodicSlotPrimitiveModule = {
  activate: (
    inputValues: number[],
    coordinates?: NgePrimitiveActivationCoordinates,
  ) => number[];
  archetypeId: string;
  computationType: 'EpisodicSlot';
  evictionPolicy: NeatGenomeEpisodicSlotEvictionPolicy;
  occupiedSlotCount: number;
  parameterSchema: Record<string, unknown>;
  receivesCoordinates: boolean;
  residualStreamId?: string;
  retrieveSlot: (
    queryValues: number[],
    coordinates?: NgePrimitiveActivationCoordinates,
  ) => number[] | null;
  slotCount: number;
  slotStorage: Float32Array;
  slotWidth: number;
  weightSharedCohortId?: string;
  writeSlot: (
    activationValues: number[],
    coordinates?: NgePrimitiveActivationCoordinates,
  ) => boolean;
};

type MaterializedModulatorBroadcasterPrimitiveModule = {
  activate: (
    inputValues: number[],
    coordinates?: NgePrimitiveActivationCoordinates,
  ) => number[];
  archetypeId: string;
  broadcastRadius: number;
  computationType: 'ModulatorBroadcaster';
  costExempt: true;
  inputSourceSpec: NeatGenomeModulatorBroadcasterInputSourceSpec;
  isWithinBroadcastRadius: (
    targetCoordinates: NgePrimitiveActivationCoordinates,
  ) => boolean;
  outputDimensionality: number;
  position: NgePrimitiveActivationCoordinates;
  residualStreamId?: string;
  weightSharedCohortId?: string;
};

type MaterializedGatingRouterPrimitiveModule = {
  activate: (
    inputValues: number[],
    coordinates?: NgePrimitiveActivationCoordinates,
  ) => number[];
  activationThreshold: number;
  archetypeId: string;
  candidateZone: string;
  computationType: 'GatingRouter';
  gatingMode: NeatGenomeGatingRouterMode['type'];
  receivesCoordinates: boolean;
  residualStreamId?: string;
  topK: number;
  weightSharedCohortId?: string;
};

type MaterializedNgePrimitiveModule =
  | MaterializedAttentionHeadPrimitiveModule
  | MaterializedGatedRecurrentCellPrimitiveModule
  | MaterializedEpisodicSlotPrimitiveModule
  | MaterializedModulatorBroadcasterPrimitiveModule
  | MaterializedGatingRouterPrimitiveModule;

type EpisodicSlotMatch = {
  cosineSimilarity: number;
  dotProduct: number;
  slotIndex: number;
};

type RuntimeNetworkWithNgePrimitiveModules = Network & {
  _ngePrimitiveModules?: MaterializedNgePrimitiveModule[];
};

const runtimeCompatibilityViewCache = new WeakMap<object, GenomeLike>();
const genomeCompatibilityViewCache = new WeakMap<object, GenomeLike>();
const NEAT_GENOME_EXTENSIONS_VERSION = 1;
const NEUTRAL_CONNECTION_GAIN = 1;
const NEUTRAL_NODE_RESPONSE = 1;
const DEFAULT_EPISODIC_SLOT_COUNT = 1;
const DEFAULT_EPISODIC_SLOT_EVICTION_POLICY: NeatGenomeEpisodicSlotEvictionPolicy =
  'lru';
const DEFAULT_GATING_ROUTER_ACTIVATION_THRESHOLD = 0.5;
const EPISODIC_SLOT_NOVELTY_THRESHOLD = 0.999;
const SUPPORTED_RECURRENT_MODULE_KINDS = new Set<NeatGenomeRecurrentModuleKind>(
  ['lstm', 'gru', 'narx-memory'],
);
const SUPPORTED_NGE_COMPUTATION_TYPES = new Set<NeatGenomeComputationType>(
  NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE,
);
const SUPPORTED_EPISODIC_SLOT_EVICTION_POLICIES =
  new Set<NeatGenomeEpisodicSlotEvictionPolicy>(
    NEAT_GENOME_EPISODIC_SLOT_EVICTION_POLICY_CATALOGUE,
  );

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
  const strictGenome = createGenomeFromNetworkJson(
    runtimePayload,
    captureOptions,
  );
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
 * Convert one versioned network JSON payload into the strict NEAT genome contract with canonical node ordering and validated edge identities.
 * This conversion isolates phenotype serialization details from genome-native heredity and compatibility workflows.
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
    createConnectionGeneFromJsonConnection(
      connectionJsonEntry,
      geneIdsByNodeIndex,
    ),
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
 * Convert one strict genome contract into the versioned network JSON payload understood by the runtime phenotype serializer.
 * The mapping preserves historical identifiers so roundtrips remain deterministic for replay and checkpoint lanes.
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
      const resolvedConnectionGain =
        connectionGainByInnovation?.[String(connectionGene.innovation)];

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
 * Materialize one executable phenotype from the strict genome contract after validation and JSON reconstruction.
 * Runtime-only hints and optional extension-derived knobs are applied after structural materialization completes.
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

  applyNgePrimitiveModuleMaterialization(
    materializedNetwork as RuntimeNetworkWithNgePrimitiveModules,
    genome.extensions,
    runtimeHints,
  );

  return materializedNetwork;
}

/**
 * Validate one strict genome contract and return a structured report covering size, node, connection, and extension invariants.
 * Callers can use the report for diagnostics-first flows without throwing on first failure.
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
 * Assert that one strict genome contract is valid and throw a rich validation error when any invariant fails.
 * This guard keeps downstream genome operators free from repetitive defensive contract checks.
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

function createNodeGeneFromJsonNode(
  nodeJsonEntry: NetworkJSONNode,
): NeatGenomeNodeGene {
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
    version: existingExtensions?.version ?? NEAT_GENOME_EXTENSIONS_VERSION,
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

function applyNgePrimitiveModuleMaterialization(
  runtimeNetwork: RuntimeNetworkWithNgePrimitiveModules,
  extensions: NeatGenomeExtensions | undefined,
  runtimeHints: GenomeMaterializationRuntimeHints,
): void {
  if (runtimeHints.ngeEnabled !== true) {
    Reflect.deleteProperty(runtimeNetwork, '_ngePrimitiveModules');
    return;
  }

  runtimeNetwork._ngePrimitiveModules = materializeNgePrimitiveModules(
    readModuleArchetypes(extensions),
  );
}

function readModuleArchetypes(
  extensions: NeatGenomeExtensions | undefined,
): NeatGenomeModuleArchetypeDescriptor[] {
  const moduleArchetypes = (
    extensions?.values as NeatGenomeExtensionValues | undefined
  )?.moduleArchetypes;

  return Array.isArray(moduleArchetypes) ? [...moduleArchetypes] : [];
}

function materializeNgePrimitiveModules(
  moduleArchetypes: readonly NeatGenomeModuleArchetypeDescriptor[],
): MaterializedNgePrimitiveModule[] {
  return moduleArchetypes.reduce<MaterializedNgePrimitiveModule[]>(
    (materializedPrimitiveModules, moduleArchetype) => {
      switch (moduleArchetype.computationType) {
        case 'AttentionHead':
          materializedPrimitiveModules.push(
            createAttentionHeadPrimitiveModule(moduleArchetype),
          );
          break;
        case 'GatedRecurrentCell':
          materializedPrimitiveModules.push(
            createGatedRecurrentCellPrimitiveModule(moduleArchetype),
          );
          break;
        case 'EpisodicSlot':
          materializedPrimitiveModules.push(
            createEpisodicSlotPrimitiveModule(moduleArchetype),
          );
          break;
        case 'ModulatorBroadcaster':
          materializedPrimitiveModules.push(
            createModulatorBroadcasterPrimitiveModule(moduleArchetype),
          );
          break;
        case 'GatingRouter':
          materializedPrimitiveModules.push(
            createGatingRouterPrimitiveModule(moduleArchetype),
          );
          break;
        default:
          break;
      }

      return materializedPrimitiveModules;
    },
    [],
  );
}

function createAttentionHeadPrimitiveModule(
  moduleArchetype: NeatGenomeModuleArchetypeDescriptor,
): MaterializedAttentionHeadPrimitiveModule {
  const parameterSchema = resolvePrimitiveParameterSchema(
    moduleArchetype.parameterSchema,
  );
  const heads = Math.max(1, Math.trunc(Number(parameterSchema.heads) || 1));
  const outputWidth = Math.max(
    1,
    Math.trunc(Number(parameterSchema.outputWidth) || 1),
  );
  const attentionHeadPrimitiveModule: MaterializedAttentionHeadPrimitiveModule =
    {
      activate: (
        inputValues: number[],
        coordinates?: NgePrimitiveActivationCoordinates,
      ): number[] => {
        const routedValues = routeAttentionCandidateValues(
          resolvePrimitiveInputValues(
            inputValues,
            attentionHeadPrimitiveModule.receivesCoordinates,
            coordinates,
          ),
          attentionHeadPrimitiveModule.heads,
        );

        return Array.from(
          { length: attentionHeadPrimitiveModule.outputWidth },
          (_unusedValue, outputIndex) =>
            routedValues[outputIndex % routedValues.length],
        );
      },
      archetypeId: moduleArchetype.archetypeId,
      computationType: 'AttentionHead',
      heads,
      outputWidth,
      parameterSchema,
      receivesCoordinates: moduleArchetype.receivesCoordinates === true,
      ...(typeof moduleArchetype.residualStreamId === 'string'
        ? { residualStreamId: moduleArchetype.residualStreamId }
        : {}),
      ...(typeof moduleArchetype.weightSharedCohortId === 'string'
        ? { weightSharedCohortId: moduleArchetype.weightSharedCohortId }
        : {}),
    };

  return attentionHeadPrimitiveModule;
}

function createGatedRecurrentCellPrimitiveModule(
  moduleArchetype: NeatGenomeModuleArchetypeDescriptor,
): MaterializedGatedRecurrentCellPrimitiveModule {
  const parameterSchema = resolvePrimitiveParameterSchema(
    moduleArchetype.parameterSchema,
  );
  const hiddenDim = Math.max(
    1,
    Math.trunc(Number(parameterSchema.hiddenDim) || 1),
  );
  const decayRate = Math.min(
    1,
    Math.max(0, Number(parameterSchema.decayRate) || 0.5),
  );
  const gatedRecurrentCellPrimitiveModule: MaterializedGatedRecurrentCellPrimitiveModule =
    {
      activate: (
        inputValues: number[],
        coordinates?: NgePrimitiveActivationCoordinates,
      ): number[] => {
        const primitiveInputValues = resolvePrimitiveInputValues(
          inputValues,
          gatedRecurrentCellPrimitiveModule.receivesCoordinates,
          coordinates,
        );

        gatedRecurrentCellPrimitiveModule.state = Array.from(
          { length: gatedRecurrentCellPrimitiveModule.hiddenDim },
          (_unusedValue, stateIndex) => {
            const previousStateValue =
              gatedRecurrentCellPrimitiveModule.state[stateIndex];
            const currentInputValue =
              primitiveInputValues[stateIndex % primitiveInputValues.length];
            const gateValue =
              1 / (1 + Math.exp(-(currentInputValue + previousStateValue)));

            return Math.tanh(
              previousStateValue *
                gatedRecurrentCellPrimitiveModule.decayRate *
                gateValue +
                Math.tanh(currentInputValue),
            );
          },
        );

        return [...gatedRecurrentCellPrimitiveModule.state];
      },
      archetypeId: moduleArchetype.archetypeId,
      clear: (): void => {
        gatedRecurrentCellPrimitiveModule.state = Array(
          gatedRecurrentCellPrimitiveModule.hiddenDim,
        ).fill(0);
      },
      computationType: 'GatedRecurrentCell',
      decayRate,
      hiddenDim,
      parameterSchema,
      receivesCoordinates: moduleArchetype.receivesCoordinates === true,
      ...(typeof moduleArchetype.residualStreamId === 'string'
        ? { residualStreamId: moduleArchetype.residualStreamId }
        : {}),
      state: Array(hiddenDim).fill(0),
      ...(typeof moduleArchetype.weightSharedCohortId === 'string'
        ? { weightSharedCohortId: moduleArchetype.weightSharedCohortId }
        : {}),
    };

  return gatedRecurrentCellPrimitiveModule;
}

function createEpisodicSlotPrimitiveModule(
  moduleArchetype: NeatGenomeModuleArchetypeDescriptor,
): MaterializedEpisodicSlotPrimitiveModule {
  const parameterSchema = resolvePrimitiveParameterSchema(
    moduleArchetype.parameterSchema,
  );
  const slotCount = Math.max(
    DEFAULT_EPISODIC_SLOT_COUNT,
    Math.trunc(
      Number(parameterSchema.slotCount) || DEFAULT_EPISODIC_SLOT_COUNT,
    ),
  );
  const evictionPolicy = resolveEpisodicSlotEvictionPolicy(
    parameterSchema.evictionPolicy,
  );
  const slotAccessSequence = new Uint32Array(slotCount);
  const slotOccupancy = new Uint8Array(slotCount);
  const slotWriteSequence = new Uint32Array(slotCount);
  let sequenceValue = 0;

  const episodicSlotPrimitiveModule: MaterializedEpisodicSlotPrimitiveModule = {
    activate: (
      inputValues: number[],
      coordinates?: NgePrimitiveActivationCoordinates,
    ): number[] =>
      episodicSlotPrimitiveModule.retrieveSlot(inputValues, coordinates) ?? [],
    archetypeId: moduleArchetype.archetypeId,
    computationType: 'EpisodicSlot',
    evictionPolicy,
    occupiedSlotCount: 0,
    parameterSchema,
    receivesCoordinates: moduleArchetype.receivesCoordinates === true,
    retrieveSlot: (
      queryValues: number[],
      coordinates?: NgePrimitiveActivationCoordinates,
    ): number[] | null => {
      if (
        episodicSlotPrimitiveModule.occupiedSlotCount === 0 ||
        episodicSlotPrimitiveModule.slotWidth === 0
      ) {
        return null;
      }

      const resolvedQueryVector = resolveEpisodicSlotVector(
        resolvePrimitiveInputValues(
          queryValues,
          episodicSlotPrimitiveModule.receivesCoordinates,
          coordinates,
        ),
        episodicSlotPrimitiveModule.slotWidth,
      );
      const bestMatch = findBestMatchingEpisodicSlot(
        episodicSlotPrimitiveModule.slotStorage,
        slotOccupancy,
        episodicSlotPrimitiveModule.slotWidth,
        resolvedQueryVector,
      ) as EpisodicSlotMatch;

      sequenceValue += 1;
      slotAccessSequence[bestMatch.slotIndex] = sequenceValue;

      return Array.from(
        readEpisodicSlotVector(
          episodicSlotPrimitiveModule.slotStorage,
          episodicSlotPrimitiveModule.slotWidth,
          bestMatch.slotIndex,
        ),
      );
    },
    ...(typeof moduleArchetype.residualStreamId === 'string'
      ? { residualStreamId: moduleArchetype.residualStreamId }
      : {}),
    slotCount,
    slotStorage: new Float32Array(0),
    slotWidth: 0,
    ...(typeof moduleArchetype.weightSharedCohortId === 'string'
      ? { weightSharedCohortId: moduleArchetype.weightSharedCohortId }
      : {}),
    writeSlot: (
      activationValues: number[],
      coordinates?: NgePrimitiveActivationCoordinates,
    ): boolean => {
      const primitiveInputValues = resolvePrimitiveInputValues(
        activationValues,
        episodicSlotPrimitiveModule.receivesCoordinates,
        coordinates,
      );

      if (primitiveInputValues.length === 0) {
        return false;
      }

      ensureEpisodicSlotStorageWidth(
        episodicSlotPrimitiveModule,
        primitiveInputValues.length,
      );

      const resolvedSlotVector = resolveEpisodicSlotVector(
        primitiveInputValues,
        episodicSlotPrimitiveModule.slotWidth,
      );
      const bestMatch = findBestMatchingEpisodicSlot(
        episodicSlotPrimitiveModule.slotStorage,
        slotOccupancy,
        episodicSlotPrimitiveModule.slotWidth,
        resolvedSlotVector,
      );

      if (
        bestMatch &&
        bestMatch.cosineSimilarity >= EPISODIC_SLOT_NOVELTY_THRESHOLD
      ) {
        sequenceValue += 1;
        slotAccessSequence[bestMatch.slotIndex] = sequenceValue;
        return false;
      }

      const targetSlotIndex = selectEpisodicSlotWriteIndex({
        evictionPolicy: episodicSlotPrimitiveModule.evictionPolicy,
        occupiedSlotCount: episodicSlotPrimitiveModule.occupiedSlotCount,
        slotAccessSequence,
        slotCount: episodicSlotPrimitiveModule.slotCount,
        slotOccupancy,
        slotWriteSequence,
      });

      writeEpisodicSlotVector(
        episodicSlotPrimitiveModule.slotStorage,
        episodicSlotPrimitiveModule.slotWidth,
        targetSlotIndex,
        resolvedSlotVector,
      );

      if (slotOccupancy[targetSlotIndex] !== 1) {
        slotOccupancy[targetSlotIndex] = 1;
        episodicSlotPrimitiveModule.occupiedSlotCount += 1;
      }

      sequenceValue += 1;
      slotAccessSequence[targetSlotIndex] = sequenceValue;
      slotWriteSequence[targetSlotIndex] = sequenceValue;

      return true;
    },
  };

  return episodicSlotPrimitiveModule;
}

function createModulatorBroadcasterPrimitiveModule(
  moduleArchetype: NeatGenomeModuleArchetypeDescriptor,
): MaterializedModulatorBroadcasterPrimitiveModule {
  const modulatorBroadcasterArchetype =
    moduleArchetype as NeatGenomeModulatorBroadcasterArchetypeDescriptor;
  const broadcastRadius = Math.max(
    0,
    modulatorBroadcasterArchetype.broadcastRadius,
  );
  const inputSourceSpec = structuredClone(
    modulatorBroadcasterArchetype.inputSourceSpec,
  );
  const outputDimensionality = Math.max(
    1,
    Math.trunc(modulatorBroadcasterArchetype.outputDimensionality),
  );
  const position = [
    ...modulatorBroadcasterArchetype.position,
  ] as NgePrimitiveActivationCoordinates;
  const positionMean =
    position.reduce(
      (coordinateSum, coordinateValue) => coordinateSum + coordinateValue,
      0,
    ) / position.length;
  const radiusNormalizer = broadcastRadius + 1;
  const modulatorBroadcasterPrimitiveModule: MaterializedModulatorBroadcasterPrimitiveModule =
    {
      activate: (inputValues: number[]): number[] => {
        const normalizedInputValues = resolveModulatorBroadcasterInputValues(
          inputValues,
          inputSourceSpec.dimensionality,
        );

        return Array.from(
          { length: outputDimensionality },
          (_unusedValue, outputIndex) => {
            const inputValue =
              normalizedInputValues[outputIndex % normalizedInputValues.length];
            const axisBias =
              position[outputIndex % position.length] / radiusNormalizer;

            return Math.tanh(inputValue + positionMean + axisBias);
          },
        );
      },
      archetypeId: moduleArchetype.archetypeId,
      broadcastRadius,
      computationType: 'ModulatorBroadcaster',
      costExempt: true,
      inputSourceSpec,
      isWithinBroadcastRadius: (
        targetCoordinates: NgePrimitiveActivationCoordinates,
      ): boolean =>
        calculateSquaredCoordinateDistance(position, targetCoordinates) <=
        broadcastRadius * broadcastRadius,
      outputDimensionality,
      position,
      ...(typeof moduleArchetype.residualStreamId === 'string'
        ? { residualStreamId: moduleArchetype.residualStreamId }
        : {}),
      ...(typeof moduleArchetype.weightSharedCohortId === 'string'
        ? { weightSharedCohortId: moduleArchetype.weightSharedCohortId }
        : {}),
    };

  return modulatorBroadcasterPrimitiveModule;
}

function createGatingRouterPrimitiveModule(
  moduleArchetype: NeatGenomeModuleArchetypeDescriptor,
): MaterializedGatingRouterPrimitiveModule {
  const gatingRouterArchetype =
    moduleArchetype as NeatGenomeGatingRouterArchetypeDescriptor;
  const topK = Math.max(1, Math.trunc(gatingRouterArchetype.topK));
  const resolvedMode = resolveGatingRouterMode(
    gatingRouterArchetype.gatingMode,
  );
  const gatingRouterPrimitiveModule: MaterializedGatingRouterPrimitiveModule = {
    activate: (
      inputValues: number[],
      coordinates?: NgePrimitiveActivationCoordinates,
    ): number[] => {
      const primitiveInputValues = resolvePrimitiveInputValues(
        inputValues,
        gatingRouterPrimitiveModule.receivesCoordinates,
        coordinates,
      );
      const selectedCandidateIndices = new Set(
        selectGatingRouterCandidateIndices({
          activationThreshold: gatingRouterPrimitiveModule.activationThreshold,
          gatingMode: gatingRouterPrimitiveModule.gatingMode,
          inputValues: primitiveInputValues,
          topK: gatingRouterPrimitiveModule.topK,
        }),
      );

      return primitiveInputValues.map((activationValue, candidateIndex) =>
        selectedCandidateIndices.has(candidateIndex) ? activationValue : 0,
      );
    },
    activationThreshold: resolvedMode.activationThreshold,
    archetypeId: moduleArchetype.archetypeId,
    candidateZone: gatingRouterArchetype.candidateZone,
    computationType: 'GatingRouter',
    gatingMode: resolvedMode.gatingMode,
    receivesCoordinates: moduleArchetype.receivesCoordinates === true,
    ...(typeof moduleArchetype.residualStreamId === 'string'
      ? { residualStreamId: moduleArchetype.residualStreamId }
      : {}),
    topK,
    ...(typeof moduleArchetype.weightSharedCohortId === 'string'
      ? { weightSharedCohortId: moduleArchetype.weightSharedCohortId }
      : {}),
  };

  return gatingRouterPrimitiveModule;
}

function resolvePrimitiveParameterSchema(
  parameterSchema: Record<string, unknown> | undefined,
): Record<string, unknown> {
  return isPlainObjectRecord(parameterSchema)
    ? structuredClone(parameterSchema)
    : {};
}

function resolveGatingRouterMode(
  gatingMode: NeatGenomeGatingRouterMode | undefined,
): {
  activationThreshold: number;
  gatingMode: NeatGenomeGatingRouterMode['type'];
} {
  if (gatingMode?.type === 'threshold') {
    return {
      activationThreshold: gatingMode.activationThreshold,
      gatingMode: 'threshold',
    };
  }

  return {
    activationThreshold: DEFAULT_GATING_ROUTER_ACTIVATION_THRESHOLD,
    gatingMode: 'topK',
  };
}

function selectGatingRouterCandidateIndices(context: {
  activationThreshold: number;
  gatingMode: NeatGenomeGatingRouterMode['type'];
  inputValues: number[];
  topK: number;
}): number[] {
  return context.inputValues
    .map((activationValue, candidateIndex) => ({
      activationValue,
      candidateIndex,
    }))
    .filter(
      (candidate) =>
        context.gatingMode !== 'threshold' ||
        candidate.activationValue >= context.activationThreshold,
    )
    .toSorted(
      (leftCandidate, rightCandidate) =>
        rightCandidate.activationValue - leftCandidate.activationValue ||
        leftCandidate.candidateIndex - rightCandidate.candidateIndex,
    )
    .slice(0, context.topK)
    .map((candidate) => candidate.candidateIndex);
}

function resolveModulatorBroadcasterInputValues(
  inputValues: number[],
  inputDimensionality: number,
): number[] {
  const resolvedInputDimensionality = Math.max(
    1,
    Math.trunc(inputDimensionality),
  );

  return Array.from(
    { length: resolvedInputDimensionality },
    (_unusedValue, inputIndex) => inputValues[inputIndex] ?? 0,
  );
}

function calculateSquaredCoordinateDistance(
  sourceCoordinates: NgePrimitiveActivationCoordinates,
  targetCoordinates: NgePrimitiveActivationCoordinates,
): number {
  return sourceCoordinates.reduce(
    (squaredDistance, coordinateValue, coordinateIndex) => {
      const axisDelta = coordinateValue - targetCoordinates[coordinateIndex];

      return squaredDistance + axisDelta * axisDelta;
    },
    0,
  );
}

function resolvePrimitiveInputValues(
  inputValues: number[],
  receivesCoordinates: boolean,
  coordinates?: NgePrimitiveActivationCoordinates,
): number[] {
  return [
    ...inputValues,
    ...(receivesCoordinates ? [...(coordinates ?? [])] : []),
  ];
}

function resolveEpisodicSlotEvictionPolicy(
  evictionPolicyValue: unknown,
): NeatGenomeEpisodicSlotEvictionPolicy {
  return typeof evictionPolicyValue === 'string' &&
    SUPPORTED_EPISODIC_SLOT_EVICTION_POLICIES.has(
      evictionPolicyValue as NeatGenomeEpisodicSlotEvictionPolicy,
    )
    ? (evictionPolicyValue as NeatGenomeEpisodicSlotEvictionPolicy)
    : DEFAULT_EPISODIC_SLOT_EVICTION_POLICY;
}

function ensureEpisodicSlotStorageWidth(
  episodicSlotPrimitiveModule: MaterializedEpisodicSlotPrimitiveModule,
  inputWidth: number,
): void {
  if (episodicSlotPrimitiveModule.slotWidth !== 0 || inputWidth <= 0) {
    return;
  }

  episodicSlotPrimitiveModule.slotWidth = inputWidth;
  episodicSlotPrimitiveModule.slotStorage = new Float32Array(
    episodicSlotPrimitiveModule.slotCount * inputWidth,
  );
}

function resolveEpisodicSlotVector(
  candidateValues: number[],
  slotWidth: number,
): Float32Array {
  const alignedValues = new Float32Array(slotWidth);
  alignedValues.set(candidateValues.slice(0, slotWidth));
  return alignedValues;
}

function findBestMatchingEpisodicSlot(
  slotStorage: Float32Array,
  slotOccupancy: Uint8Array,
  slotWidth: number,
  queryVector: Float32Array,
): EpisodicSlotMatch | undefined {
  let bestMatch: EpisodicSlotMatch | undefined;

  for (let slotIndex = 0; slotIndex < slotOccupancy.length; slotIndex++) {
    if (slotOccupancy[slotIndex] !== 1) {
      continue;
    }

    const storedVector = readEpisodicSlotVector(
      slotStorage,
      slotWidth,
      slotIndex,
    );
    const dotProduct = computeVectorDotProduct(queryVector, storedVector);
    const cosineSimilarity = computeVectorCosineSimilarity(
      queryVector,
      storedVector,
    );

    if (
      shouldReplaceBestEpisodicSlotMatch(
        bestMatch,
        dotProduct,
        cosineSimilarity,
        slotIndex,
      )
    ) {
      bestMatch = {
        cosineSimilarity,
        dotProduct,
        slotIndex,
      };
    }
  }

  return bestMatch;
}

function readEpisodicSlotVector(
  slotStorage: Float32Array,
  slotWidth: number,
  slotIndex: number,
): Float32Array {
  const slotStartIndex = slotIndex * slotWidth;
  return slotStorage.subarray(slotStartIndex, slotStartIndex + slotWidth);
}

function computeVectorDotProduct(
  leftValues: ArrayLike<number>,
  rightValues: ArrayLike<number>,
): number {
  let dotProduct = 0;

  for (let valueIndex = 0; valueIndex < leftValues.length; valueIndex++) {
    dotProduct += leftValues[valueIndex] * rightValues[valueIndex];
  }

  return dotProduct;
}

function computeVectorCosineSimilarity(
  leftValues: ArrayLike<number>,
  rightValues: ArrayLike<number>,
): number {
  let leftMagnitudeSquared = 0;
  let rightMagnitudeSquared = 0;

  for (let valueIndex = 0; valueIndex < leftValues.length; valueIndex++) {
    leftMagnitudeSquared += leftValues[valueIndex] ** 2;
    rightMagnitudeSquared += rightValues[valueIndex] ** 2;
  }

  if (leftMagnitudeSquared === 0 && rightMagnitudeSquared === 0) {
    return 1;
  }

  if (leftMagnitudeSquared === 0 || rightMagnitudeSquared === 0) {
    return 0;
  }

  return (
    computeVectorDotProduct(leftValues, rightValues) /
    Math.sqrt(leftMagnitudeSquared * rightMagnitudeSquared)
  );
}

function shouldReplaceBestEpisodicSlotMatch(
  bestMatch: EpisodicSlotMatch | undefined,
  dotProduct: number,
  cosineSimilarity: number,
  slotIndex: number,
): boolean {
  if (!bestMatch) {
    return true;
  }

  if (dotProduct !== bestMatch.dotProduct) {
    return dotProduct > bestMatch.dotProduct;
  }

  if (cosineSimilarity !== bestMatch.cosineSimilarity) {
    return cosineSimilarity > bestMatch.cosineSimilarity;
  }

  return slotIndex < bestMatch.slotIndex;
}

function selectEpisodicSlotWriteIndex(context: {
  evictionPolicy: NeatGenomeEpisodicSlotEvictionPolicy;
  occupiedSlotCount: number;
  slotAccessSequence: Uint32Array;
  slotCount: number;
  slotOccupancy: Uint8Array;
  slotWriteSequence: Uint32Array;
}): number {
  if (context.occupiedSlotCount < context.slotCount) {
    return findFirstEmptyEpisodicSlotIndex(context.slotOccupancy);
  }

  return context.evictionPolicy === 'fifo'
    ? findOldestEpisodicSlotIndex(context.slotWriteSequence)
    : findOldestEpisodicSlotIndex(context.slotAccessSequence);
}

function findFirstEmptyEpisodicSlotIndex(slotOccupancy: Uint8Array): number {
  for (let slotIndex = 0; slotIndex < slotOccupancy.length; slotIndex++) {
    if (slotOccupancy[slotIndex] !== 1) {
      return slotIndex;
    }
  }

  return 0;
}

function findOldestEpisodicSlotIndex(sequenceValues: Uint32Array): number {
  let selectedIndex = 0;

  for (
    let sequenceIndex = 1;
    sequenceIndex < sequenceValues.length;
    sequenceIndex++
  ) {
    const selectedSequenceValue = sequenceValues[selectedIndex];
    const candidateSequenceValue = sequenceValues[sequenceIndex];

    if (candidateSequenceValue < selectedSequenceValue) {
      selectedIndex = sequenceIndex;
    }
  }

  return selectedIndex;
}

function writeEpisodicSlotVector(
  slotStorage: Float32Array,
  slotWidth: number,
  slotIndex: number,
  slotVector: Float32Array,
): void {
  const slotStartIndex = slotIndex * slotWidth;
  slotStorage.fill(0, slotStartIndex, slotStartIndex + slotWidth);
  slotStorage.set(slotVector.subarray(0, slotWidth), slotStartIndex);
}

function routeAttentionCandidateValues(
  candidateValues: number[],
  heads: number,
): number[] {
  return candidateValues.map(
    (candidateValue, candidateIndex, allCandidates) => {
      const headWidth = Math.max(1, Math.ceil(allCandidates.length / heads));
      const groupStartIndex =
        Math.floor(candidateIndex / headWidth) * headWidth;
      const groupCandidates = allCandidates.slice(
        groupStartIndex,
        groupStartIndex + headWidth,
      );
      const groupDenominator = groupCandidates.reduce(
        (sum, groupCandidateValue) => sum + Math.exp(groupCandidateValue),
        0,
      );

      return candidateValue * (Math.exp(candidateValue) / groupDenominator);
    },
  );
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
  void label;
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
      const firstPath = connectionPathsByInnovation.get(
        connectionGene.innovation,
      );
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

  const hasValidVersion =
    Number.isInteger(extensions.version) && extensions.version > 0;
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

  const knownResidualStreamIds = validateResidualStreamExtension(
    (extensions.values as NeatGenomeExtensionValues).residualStreams,
    issues,
  );
  const knownWeightSharedCohortIds = validateWeightSharedCohortExtension(
    (extensions.values as NeatGenomeExtensionValues).weightSharedCohorts,
    issues,
  );

  validateModuleArchetypeExtension(
    (extensions.values as NeatGenomeExtensionValues).moduleArchetypes,
    knownResidualStreamIds,
    knownWeightSharedCohortIds,
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
    const matchedConnectionGene =
      connectionGenesByInnovation.get(parsedInnovation);
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

  for (const [geneIdKey, responseValue] of Object.entries(
    nodeResponseByGeneId,
  )) {
    const parsedGeneId = Number(geneIdKey);
    const matchedNodeGene = nodeGenesById.get(parsedGeneId);
    const hasValidResponse =
      typeof responseValue === 'number' &&
      Number.isFinite(responseValue) &&
      responseValue !== NEUTRAL_NODE_RESPONSE;

    if (
      !Number.isFinite(parsedGeneId) ||
      !matchedNodeGene ||
      !hasValidResponse
    ) {
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
  recurrentModules: NeatGenomeExtensionValues['recurrentModules'] | undefined,
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

  const knownNodeGeneIds = new Set(
    nodeGenes.map((nodeGene) => nodeGene.geneId),
  );
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

  const knownNodeGeneIds = new Set(
    nodeGenes.map((nodeGene) => nodeGene.geneId),
  );
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
    SUPPORTED_RECURRENT_MODULE_KINDS.has(
      kind as NeatGenomeRecurrentModuleKind,
    ) &&
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

    const matchedConnectionGene =
      connectionGenesByInnovation.get(connectionInnovation);

    return (
      !!matchedConnectionGene &&
      typeof matchedConnectionGene.gaterGeneId === 'number' &&
      gaterGeneIdSet.has(matchedConnectionGene.gaterGeneId)
    );
  });
}

function validateResidualStreamExtension(
  residualStreams: NeatGenomeExtensionValues['residualStreams'] | undefined,
  issues: NeatGenomeValidationIssue[],
): Set<string> {
  const knownResidualStreamIds = new Set<string>();

  if (typeof residualStreams === 'undefined') {
    return knownResidualStreamIds;
  }

  if (!Array.isArray(residualStreams)) {
    issues.push(
      createIssue(
        'invalid-residual-stream-extension',
        'extensions.values.residualStreams',
        'Residual-stream extensions must be stored as one array of supported stream descriptors.',
      ),
    );
    return knownResidualStreamIds;
  }

  residualStreams.forEach((residualStream, streamIndex) => {
    if (!isValidResidualStreamDescriptor(residualStream)) {
      issues.push(
        createIssue(
          'invalid-residual-stream-extension',
          `extensions.values.residualStreams[${streamIndex}]`,
          'Residual-stream extensions must declare one stable stream id and one positive integer width.',
          { streamIndex },
        ),
      );
      return;
    }

    knownResidualStreamIds.add(residualStream.streamId);
  });

  return knownResidualStreamIds;
}

function validateWeightSharedCohortExtension(
  weightSharedCohorts:
    | NeatGenomeExtensionValues['weightSharedCohorts']
    | undefined,
  issues: NeatGenomeValidationIssue[],
): Set<string> {
  const knownWeightSharedCohortIds = new Set<string>();

  if (typeof weightSharedCohorts === 'undefined') {
    return knownWeightSharedCohortIds;
  }

  if (!Array.isArray(weightSharedCohorts)) {
    issues.push(
      createIssue(
        'invalid-weight-shared-cohort-extension',
        'extensions.values.weightSharedCohorts',
        'Weight-shared cohort extensions must be stored as one array of supported cohort descriptors.',
      ),
    );
    return knownWeightSharedCohortIds;
  }

  weightSharedCohorts.forEach((weightSharedCohort, cohortIndex) => {
    if (!isValidWeightSharedCohortDescriptor(weightSharedCohort)) {
      issues.push(
        createIssue(
          'invalid-weight-shared-cohort-extension',
          `extensions.values.weightSharedCohorts[${cohortIndex}]`,
          'Weight-shared cohort extensions must declare one stable cohort id and an optional plain-object shared parameter schema.',
          { cohortIndex },
        ),
      );
      return;
    }

    knownWeightSharedCohortIds.add(weightSharedCohort.cohortId);
  });

  return knownWeightSharedCohortIds;
}

function validateModuleArchetypeExtension(
  moduleArchetypes: NeatGenomeExtensionValues['moduleArchetypes'] | undefined,
  knownResidualStreamIds: Set<string>,
  knownWeightSharedCohortIds: Set<string>,
  issues: NeatGenomeValidationIssue[],
): void {
  if (typeof moduleArchetypes === 'undefined') {
    return;
  }

  if (!Array.isArray(moduleArchetypes)) {
    issues.push(
      createIssue(
        'invalid-module-archetype-extension',
        'extensions.values.moduleArchetypes',
        'Module-archetype extensions must be stored as one array of supported archetype descriptors.',
      ),
    );
    return;
  }

  moduleArchetypes.forEach((moduleArchetype, archetypeIndex) => {
    if (
      !isValidModuleArchetypeDescriptor(
        moduleArchetype,
        knownResidualStreamIds,
        knownWeightSharedCohortIds,
      )
    ) {
      issues.push(
        createIssue(
          'invalid-module-archetype-extension',
          `extensions.values.moduleArchetypes[${archetypeIndex}]`,
          'Module-archetype extensions must declare one stable archetype id, a supported computationType, optional plain-object parameter schema, optional boolean coordinate injection, only known residual-stream or weight-shared cohort references, and any ModulatorBroadcaster or GatingRouter entries must also provide their required governance fields.',
          { archetypeIndex },
        ),
      );
    }
  });
}

function isValidResidualStreamDescriptor(
  residualStream: unknown,
): residualStream is NeatGenomeResidualStreamDescriptor {
  if (!isPlainObjectRecord(residualStream)) {
    return false;
  }

  const streamId = residualStream.streamId;
  const width = residualStream.width;

  return (
    typeof streamId === 'string' &&
    streamId.length > 0 &&
    typeof width === 'number' &&
    Number.isInteger(width) &&
    width > 0
  );
}

function isValidWeightSharedCohortDescriptor(
  weightSharedCohort: unknown,
): weightSharedCohort is NeatGenomeWeightSharedCohortDescriptor {
  if (!isPlainObjectRecord(weightSharedCohort)) {
    return false;
  }

  const cohortId = weightSharedCohort.cohortId;
  const sharedParameterSchema = weightSharedCohort.sharedParameterSchema;

  return (
    typeof cohortId === 'string' &&
    cohortId.length > 0 &&
    (typeof sharedParameterSchema === 'undefined' ||
      isPlainObjectRecord(sharedParameterSchema))
  );
}

function isValidModuleArchetypeDescriptor(
  moduleArchetype: unknown,
  knownResidualStreamIds: Set<string>,
  knownWeightSharedCohortIds: Set<string>,
): moduleArchetype is NeatGenomeModuleArchetypeDescriptor {
  if (!isPlainObjectRecord(moduleArchetype)) {
    return false;
  }

  if (
    !hasValidModuleArchetypeBaseFields(
      moduleArchetype,
      knownResidualStreamIds,
      knownWeightSharedCohortIds,
    )
  ) {
    return false;
  }

  if (moduleArchetype.computationType === 'ModulatorBroadcaster') {
    return hasValidModulatorBroadcasterGovernance(moduleArchetype);
  }

  if (moduleArchetype.computationType === 'GatingRouter') {
    return hasValidGatingRouterGovernance(moduleArchetype);
  }

  return true;
}

function hasValidModuleArchetypeBaseFields(
  moduleArchetype: Record<string, unknown>,
  knownResidualStreamIds: Set<string>,
  knownWeightSharedCohortIds: Set<string>,
): boolean {
  const archetypeId = moduleArchetype.archetypeId;
  const computationType = moduleArchetype.computationType;
  const parameterSchema = moduleArchetype.parameterSchema;
  const receivesCoordinates = moduleArchetype.receivesCoordinates;
  const residualStreamId = moduleArchetype.residualStreamId;
  const weightSharedCohortId = moduleArchetype.weightSharedCohortId;

  return (
    typeof archetypeId === 'string' &&
    archetypeId.length > 0 &&
    typeof computationType === 'string' &&
    SUPPORTED_NGE_COMPUTATION_TYPES.has(
      computationType as NeatGenomeComputationType,
    ) &&
    (typeof parameterSchema === 'undefined' ||
      isPlainObjectRecord(parameterSchema)) &&
    (typeof receivesCoordinates === 'undefined' ||
      typeof receivesCoordinates === 'boolean') &&
    (typeof residualStreamId === 'undefined' ||
      (typeof residualStreamId === 'string' &&
        residualStreamId.length > 0 &&
        knownResidualStreamIds.has(residualStreamId))) &&
    (typeof weightSharedCohortId === 'undefined' ||
      (typeof weightSharedCohortId === 'string' &&
        weightSharedCohortId.length > 0 &&
        knownWeightSharedCohortIds.has(weightSharedCohortId)))
  );
}

function hasValidModulatorBroadcasterGovernance(
  moduleArchetype: Record<string, unknown>,
): boolean {
  const position = moduleArchetype.position;
  const broadcastRadius = moduleArchetype.broadcastRadius;
  const inputSourceSpec = moduleArchetype.inputSourceSpec;
  const outputDimensionality = moduleArchetype.outputDimensionality;

  return (
    isValidSubstrateCoordinate(position) &&
    typeof broadcastRadius === 'number' &&
    Number.isFinite(broadcastRadius) &&
    broadcastRadius >= 0 &&
    isValidModulatorBroadcasterInputSourceSpec(inputSourceSpec) &&
    typeof outputDimensionality === 'number' &&
    Number.isInteger(outputDimensionality) &&
    outputDimensionality > 0
  );
}

function hasValidGatingRouterGovernance(
  moduleArchetype: Record<string, unknown>,
): boolean {
  const candidateZone = moduleArchetype.candidateZone;
  const topK = moduleArchetype.topK;
  const gatingMode = moduleArchetype.gatingMode;

  return (
    typeof candidateZone === 'string' &&
    candidateZone.length > 0 &&
    typeof topK === 'number' &&
    Number.isInteger(topK) &&
    topK > 0 &&
    (typeof gatingMode === 'undefined' || isValidGatingRouterMode(gatingMode))
  );
}

function isValidSubstrateCoordinate(
  value: unknown,
): value is NeatGenomeSubstrateCoordinate {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every(
      (coordinateValue) =>
        typeof coordinateValue === 'number' && Number.isFinite(coordinateValue),
    )
  );
}

function isValidModulatorBroadcasterInputSourceSpec(
  value: unknown,
): value is NeatGenomeModulatorBroadcasterInputSourceSpec {
  return (
    isPlainObjectRecord(value) &&
    typeof value.dimensionality === 'number' &&
    Number.isInteger(value.dimensionality) &&
    value.dimensionality > 0
  );
}

function isValidGatingRouterMode(
  value: unknown,
): value is NeatGenomeGatingRouterMode {
  if (!isPlainObjectRecord(value)) {
    return false;
  }

  if (value.type === 'topK') {
    return true;
  }

  return (
    value.type === 'threshold' &&
    typeof value.activationThreshold === 'number' &&
    Number.isFinite(value.activationThreshold)
  );
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
      return createGenomeFromNetwork(
        source as unknown as Network,
      ).connectionGenes.map((connectionGene) => ({
        innovation: connectionGene.innovation,
        weight: connectionGene.weight,
      }));
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
