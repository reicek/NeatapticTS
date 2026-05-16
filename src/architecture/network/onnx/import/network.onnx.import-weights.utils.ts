import type Network from '../../network';
import Connection from '../../../connection';
import type NeatapticNode from '../../../node';
import type {
  Conv2DMapping,
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';
import type {
  OnnxImportAggregatedLayerAssignmentContext,
  OnnxImportAggregatedNeuronAssignmentContext,
  OnnxImportConvCoordinateAssignmentContext,
  OnnxImportConvLayerContextBuildParams,
  OnnxImportConvKernelAssignmentContext,
  OnnxImportConvLayerContext,
  OnnxImportConvMetadata,
  OnnxImportConvNodeSlices,
  OnnxImportConvOutputCoordinate,
  OnnxImportConvSourceLayout,
  OnnxImportConvTensorContext,
  OnnxImportHiddenSizeDerivationContext,
  OnnxImportInboundConnectionMap,
  OnnxImportLayerNodePair,
  OnnxImportLayerNodePairBuildParams,
  OnnxImportLayerTensorNames,
  OnnxImportLayerWeightBucket,
  OnnxImportPerNeuronAssignmentContext,
  OnnxImportPerNeuronLayerAssignmentContext,
  OnnxImportWeightAssignmentContext,
  OnnxImportWeightAssignmentBuildParams,
} from './network.onnx.import-weights.types';
import type {
  NodeInternals,
  OnnxConvKernelCoordinate,
} from '../network.onnx.utils.types';
import { readOnnxTensorFloatData } from '../schema/network.onnx.schema.tensor-data.utils';

const METADATA_KEY_LAYER_SIZES = 'layer_sizes';
const METADATA_KEY_CONV2D_LAYERS = 'conv2d_layers';
const METADATA_KEY_CONV2D_SPECS = 'conv2d_specs';
const METADATA_KEY_POOL2D_SPECS = 'pool2d_specs';
const METADATA_KEY_SHARED_INITIALIZER_ALIASES = 'shared_initializer_aliases';
const NODE_TYPE_INPUT = 'input';
const NODE_TYPE_HIDDEN = 'hidden';
const NODE_TYPE_OUTPUT = 'output';
const WEIGHT_TENSOR_PREFIX = 'W';
const BIAS_TENSOR_PREFIX = 'B';
const CONV_WEIGHT_TENSOR_PREFIX = 'ConvW';
const CONV_BIAS_TENSOR_PREFIX = 'ConvB';
const PER_NEURON_TENSOR_SEGMENT = '_n';
const LAYER_INDEX_OFFSET = 1;
const FIRST_INDEX = 0;
const ZERO_VALUE = 0;
const FIRST_BIAS_INDEX = 0;
const WEIGHT_TENSOR_PATTERN = /^W(\d+)(?:_n(\d+))?$/i;

/**
 * Extract hidden layer sizes from ONNX initializers (weight tensors).
 *
 * @param initializers ONNX initializer tensors.
 * @param metadataProps Optional ONNX metadata properties.
 * @returns Hidden layer sizes in order.
 */
export function deriveHiddenLayerSizes(
  initializers: OnnxTensor[],
  metadataProps?: OnnxMetadataProperty[],
): number[] {
  // Step 1: Build a normalized hidden-size derivation context.
  const derivationContext: OnnxImportHiddenSizeDerivationContext = {
    initializers,
    metadataProps: metadataProps ?? [],
  };

  // Step 2: Prefer explicit metadata-driven layer sizes when available.
  const metadataLayerSizes = parseMetadataLayerSizes(
    derivationContext.metadataProps,
  );
  if (metadataLayerSizes) return metadataLayerSizes;

  // Step 3: Bucket ONNX weight tensors by export layer index.
  const layerWeightBuckets = collectLayerWeightBuckets(
    derivationContext.initializers,
  );

  // Step 4: Resolve sorted layer indices and fold hidden sizes.
  const sortedLayerIndices = collectSortedLayerIndices(layerWeightBuckets);
  return buildHiddenLayerSizesFromBuckets(
    layerWeightBuckets,
    sortedLayerIndices,
  );
}

/**
 * Assign weights and biases from ONNX initializers to a newly created network.
 *
 * @param network Target network to mutate.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @param metadataProps Optional ONNX metadata properties.
 * @returns Nothing.
 */
export function assignWeightsAndBiases(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadataProps?: OnnxMetadataProperty[],
): void {
  // Step 1: Build assignment context from ONNX graph + network state.
  const assignmentContext = buildWeightAssignmentContext({
    network,
    onnx,
    hiddenLayerSizes,
    metadataProps,
  });

  // Step 2: Assign dense/per-neuron weights for all imported layer tensors.
  applyDenseWeightAssignments(assignmentContext);

  // Step 3: Apply optional Conv2D reconstruction metadata pass.
  applyOptionalConvReconstruction(assignmentContext);
}

/**
 * Parse explicit metadata-driven hidden layer sizes.
 *
 * @param metadataProps ONNX metadata properties.
 * @returns Parsed hidden layer sizes when available.
 */
function parseMetadataLayerSizes(
  metadataProps: OnnxMetadataProperty[],
): number[] | null {
  // Step 1: Resolve the metadata payload for explicit layer sizes.
  const metadataLayerSizes = metadataProps.find(
    (property) => property.key === METADATA_KEY_LAYER_SIZES,
  );
  if (!metadataLayerSizes) return null;

  // Step 2: Parse and normalize JSON payload into numeric layer sizes.
  try {
    const parsedLayerSizes = JSON.parse(metadataLayerSizes.value);
    return Array.isArray(parsedLayerSizes)
      ? (parsedLayerSizes as number[])
      : null;
  } catch {
    return null;
  }
}

/**
 * Collect ONNX weight tensor buckets grouped by export layer index.
 *
 * @param initializers ONNX initializer tensors.
 * @returns Layer-weight buckets keyed by export layer index.
 */
function collectLayerWeightBuckets(
  initializers: OnnxTensor[],
): Record<string, OnnxImportLayerWeightBucket> {
  // Step 1: Collect only dense/per-neuron weight tensors.
  const weightTensors = initializers.filter((tensor) =>
    tensor.name.startsWith(WEIGHT_TENSOR_PREFIX),
  );

  // Step 2: Fold tensors into layer-index buckets.
  return weightTensors.reduce<Record<string, OnnxImportLayerWeightBucket>>(
    (layerBuckets, tensor) => {
      const parsedWeightName = parseWeightTensorName(tensor.name);
      if (!parsedWeightName) return layerBuckets;

      const layerBucket = layerBuckets[parsedWeightName.layerIndex] ?? {
        perNeuron: [],
      };
      if (parsedWeightName.neuronIndex !== null) {
        layerBucket.perNeuron.push(tensor);
      } else {
        layerBucket.aggregated = tensor;
      }

      layerBuckets[parsedWeightName.layerIndex] = layerBucket;
      return layerBuckets;
    },
    {},
  );
}

/**
 * Collect sorted layer indices from weight buckets.
 *
 * @param layerWeightBuckets Layer-weight buckets.
 * @returns Ascending export layer indices.
 */
function collectSortedLayerIndices(
  layerWeightBuckets: Record<string, OnnxImportLayerWeightBucket>,
): number[] {
  return Object.keys(layerWeightBuckets)
    .map(Number)
    .toSorted((leftIndex, rightIndex) => leftIndex - rightIndex);
}

/**
 * Build hidden-layer sizes from weight buckets while excluding output layer.
 *
 * @param layerWeightBuckets Layer-weight buckets.
 * @param sortedLayerIndices Ascending layer indices.
 * @returns Hidden-layer sizes.
 */
function buildHiddenLayerSizesFromBuckets(
  layerWeightBuckets: Record<string, OnnxImportLayerWeightBucket>,
  sortedLayerIndices: number[],
): number[] {
  // Step 1: Exit early when no layer tensors were discovered.
  if (sortedLayerIndices.length === ZERO_VALUE) return [];

  // Step 2: Fold all non-output layer entries into hidden sizes.
  return sortedLayerIndices
    .slice(FIRST_INDEX, -1)
    .map((layerIndex) =>
      resolveLayerHiddenSize(layerWeightBuckets, layerIndex),
    );
}

/**
 * Resolve one hidden-layer size from its weight bucket.
 *
 * @param layerWeightBuckets Layer-weight buckets.
 * @param layerIndex Export layer index.
 * @returns Hidden-layer size.
 */
function resolveLayerHiddenSize(
  layerWeightBuckets: Record<string, OnnxImportLayerWeightBucket>,
  layerIndex: number,
): number {
  const layerBucket = layerWeightBuckets[String(layerIndex)];
  if (layerBucket.aggregated) return layerBucket.aggregated.dims[FIRST_INDEX];
  return layerBucket.perNeuron.length;
}

/**
 * Build the shared assignment context for import weight restoration.
 *
 * @param params Assignment context input params.
 * @returns Shared assignment context.
 */
function buildWeightAssignmentContext(
  params: OnnxImportWeightAssignmentBuildParams,
): OnnxImportWeightAssignmentContext {
  // Step 1: Build initializer lookup map keyed by tensor name.
  const metadataProps = params.metadataProps ?? [];
  const initializerMap = buildInitializerMap(
    params.onnx.graph.initializer,
    metadataProps,
  );

  // Step 2: Resolve sorted dense layer indices from weight tensor names.
  const sortedLayerIndices = collectSortedUniqueLayerIndices(initializerMap);

  // Step 3: Pre-collect stable input/hidden/output node groups.
  const inputNodes = collectNodesByType(params.network.nodes, NODE_TYPE_INPUT);
  const hiddenNodes = collectNodesByType(
    params.network.nodes,
    NODE_TYPE_HIDDEN,
  );
  const outputNodes = collectNodesByType(
    params.network.nodes,
    NODE_TYPE_OUTPUT,
  );

  // Step 4: Fold all values into the assignment context.
  return {
    onnx: params.onnx,
    hiddenLayerSizes: params.hiddenLayerSizes,
    metadataProps,
    initializerMap,
    sortedLayerIndices,
    inputNodes,
    hiddenNodes,
    outputNodes,
  };
}

/**
 * Parse layer index from dense/per-neuron weight tensor name.
 *
 * @param tensorName Tensor name.
 * @returns Parsed layer index or null.
 */
function parseLayerIndexFromWeightTensor(tensorName: string): number | null {
  const match = WEIGHT_TENSOR_PATTERN.exec(tensorName);
  return match ? Number(match[1]) : null;
}

/**
 * Parse layer/neuron components from a weight tensor name.
 *
 * @param tensorName Tensor name.
 * @returns Parsed layer+neuron components when matched.
 */
function parseWeightTensorName(
  tensorName: string,
): { layerIndex: string; neuronIndex: number | null } | null {
  const match = WEIGHT_TENSOR_PATTERN.exec(tensorName);
  if (!match) return null;
  return {
    layerIndex: match[1],
    neuronIndex: match[2] !== undefined ? Number(match[2]) : null,
  };
}

/**
 * Build ONNX initializer map keyed by tensor name.
 *
 * @param initializers ONNX initializer list.
 * @returns Tensor map by name.
 */
function buildInitializerMap(
  initializers: OnnxTensor[],
  metadataProps: OnnxMetadataProperty[],
): Record<string, OnnxTensor> {
  const initializerMap = initializers.reduce<Record<string, OnnxTensor>>(
    (mapByName, tensor) => {
      mapByName[tensor.name] = tensor;
      return mapByName;
    },
    {},
  );

  hydrateSharedInitializerAliases(initializerMap, metadataProps);
  return initializerMap;
}

/** Hydrate alias tensor names back into the initializer map for metadata-backed shared initializers. */
function hydrateSharedInitializerAliases(
  initializerMap: Record<string, OnnxTensor>,
  metadataProps: OnnxMetadataProperty[],
): void {
  const sharedInitializerAliases = parseSharedInitializerAliases(metadataProps);
  sharedInitializerAliases.forEach((initializerAlias) => {
    const canonicalTensor =
      initializerMap[initializerAlias.canonicalTensorName];
    if (!canonicalTensor) {
      return;
    }
    initializerMap[initializerAlias.aliasTensorName] = canonicalTensor;
  });
}

/** Parse valid shared-initializer alias metadata records from ONNX metadata. */
function parseSharedInitializerAliases(
  metadataProps: OnnxMetadataProperty[],
): Array<{ aliasTensorName: string; canonicalTensorName: string }> {
  const metadataProperty = metadataProps.find(
    (property) => property.key === METADATA_KEY_SHARED_INITIALIZER_ALIASES,
  );
  if (!metadataProperty) {
    return [];
  }

  try {
    const parsedAliases = JSON.parse(metadataProperty.value);
    if (!Array.isArray(parsedAliases)) {
      return [];
    }

    return parsedAliases.filter(
      (
        initializerAlias,
      ): initializerAlias is {
        aliasTensorName: string;
        canonicalTensorName: string;
      } =>
        Boolean(initializerAlias) &&
        typeof initializerAlias === 'object' &&
        typeof initializerAlias.aliasTensorName === 'string' &&
        typeof initializerAlias.canonicalTensorName === 'string',
    );
  } catch {
    return [];
  }
}

/**
 * Collect unique sorted layer indices from initializer weight tensors.
 *
 * @param initializerMap Initializer map keyed by tensor name.
 * @returns Unique sorted layer indices.
 */
function collectSortedUniqueLayerIndices(
  initializerMap: Record<string, OnnxTensor>,
): number[] {
  const uniqueLayerIndices = Object.keys(initializerMap).reduce<Set<number>>(
    (layerIndexSet, tensorName) => {
      const layerIndex = parseLayerIndexFromWeightTensor(tensorName);
      if (layerIndex === null) return layerIndexSet;
      layerIndexSet.add(layerIndex);
      return layerIndexSet;
    },
    new Set<number>(),
  );

  return Array.from(uniqueLayerIndices).toSorted(
    (leftIndex, rightIndex) => leftIndex - rightIndex,
  );
}

/**
 * Collect nodes by runtime node type discriminator.
 *
 * @param nodes Network nodes.
 * @param nodeType Runtime node type.
 * @returns Filtered nodes.
 */
function collectNodesByType(
  nodes: NeatapticNode[],
  nodeType: 'input' | 'hidden' | 'output',
): NeatapticNode[] {
  return nodes.filter((node) => node.type === nodeType);
}

/**
 * Apply dense/per-neuron assignments for all sorted layer indices.
 *
 * @param assignmentContext Shared assignment context.
 * @returns Nothing.
 */
function applyDenseWeightAssignments(
  assignmentContext: OnnxImportWeightAssignmentContext,
): void {
  // Step 1: Traverse all sorted ONNX layer indices.
  assignmentContext.sortedLayerIndices.forEach((layerIndex) => {
    const nodePair = buildLayerNodePair(assignmentContext, {
      layerIndex,
      sequentialIndex: layerIndex,
    });
    assignLayerWeights(assignmentContext.initializerMap, nodePair);
  });
}

/**
 * Build current/previous node slices for one sequential import layer pass.
 *
 * @param assignmentContext Shared assignment context.
 * @param params Sequential traversal params.
 * @returns Layer node pair.
 */
function buildLayerNodePair(
  assignmentContext: OnnxImportWeightAssignmentContext,
  params: OnnxImportLayerNodePairBuildParams,
): OnnxImportLayerNodePair {
  // Step 1: Resolve target current layer nodes.
  const currentLayerNodes = resolveCurrentLayerNodes(assignmentContext, params);

  // Step 2: Resolve source previous layer nodes.
  const previousLayerNodes = resolvePreviousLayerNodes(
    assignmentContext,
    params,
  );

  // Step 3: Fold slices into a typed node-pair payload.
  return {
    sequentialIndex: params.sequentialIndex,
    layerIndex: params.layerIndex,
    currentLayerNodes,
    previousLayerNodes,
  };
}

/**
 * Resolve current layer nodes for one sequential layer assignment pass.
 *
 * @param assignmentContext Shared assignment context.
 * @param params Sequential traversal params.
 * @returns Current layer nodes.
 */
function resolveCurrentLayerNodes(
  assignmentContext: OnnxImportWeightAssignmentContext,
  params: { layerIndex: number },
): NeatapticNode[] {
  const layerPosition = params.layerIndex;
  const isHiddenLayer = layerPosition < assignmentContext.hiddenLayerSizes.length;
  if (!isHiddenLayer) return assignmentContext.outputNodes;

  const layerStart = sumHiddenSizesToIndex(
    assignmentContext.hiddenLayerSizes,
    layerPosition,
  );
  const layerEnd = sumHiddenSizesToIndex(
    assignmentContext.hiddenLayerSizes,
    layerPosition + LAYER_INDEX_OFFSET,
  );
  return assignmentContext.hiddenNodes.slice(layerStart, layerEnd);
}

/**
 * Resolve previous layer nodes for one sequential layer assignment pass.
 *
 * @param assignmentContext Shared assignment context.
 * @param params Sequential traversal params.
 * @returns Previous layer nodes.
 */
function resolvePreviousLayerNodes(
  assignmentContext: OnnxImportWeightAssignmentContext,
  params: { layerIndex: number },
): NeatapticNode[] {
  const layerPosition = params.layerIndex;
  if (layerPosition === ZERO_VALUE)
    return assignmentContext.inputNodes;

  const previousLayerStart = sumHiddenSizesToIndex(
    assignmentContext.hiddenLayerSizes,
    layerPosition - LAYER_INDEX_OFFSET,
  );
  const previousLayerEnd = sumHiddenSizesToIndex(
    assignmentContext.hiddenLayerSizes,
    layerPosition,
  );
  return assignmentContext.hiddenNodes.slice(
    previousLayerStart,
    previousLayerEnd,
  );
}

/**
 * Sum hidden-layer sizes from index `0` to `exclusiveEndIndex`.
 *
 * @param hiddenLayerSizes Hidden-layer size list.
 * @param exclusiveEndIndex Exclusive end index.
 * @returns Prefix sum.
 */
function sumHiddenSizesToIndex(
  hiddenLayerSizes: number[],
  exclusiveEndIndex: number,
): number {
  return hiddenLayerSizes
    .slice(FIRST_INDEX, exclusiveEndIndex)
    .reduce((sum, value) => sum + value, ZERO_VALUE);
}

/**
 * Assign one layer's weights using aggregated or per-neuron tensors.
 *
 * @param initializerMap ONNX initializer map.
 * @param nodePair Current/previous node slices.
 * @returns Nothing.
 */
function assignLayerWeights(
  initializerMap: Record<string, OnnxTensor>,
  nodePair: OnnxImportLayerNodePair,
): void {
  // Step 1: Resolve aggregated tensor path for the layer.
  const aggregatedContext: OnnxImportAggregatedLayerAssignmentContext = {
    initializerMap,
    nodePair,
  };
  if (hasAggregatedLayerWeights(aggregatedContext)) {
    applyAggregatedLayerWeights(aggregatedContext);
    return;
  }

  // Step 2: Fallback to per-neuron weight tensors.
  const perNeuronContext: OnnxImportPerNeuronLayerAssignmentContext = {
    initializerMap,
    nodePair,
  };
  applyPerNeuronLayerWeights(perNeuronContext);
}

/**
 * Determine whether the layer has aggregated weight tensor data.
 *
 * @param aggregatedContext Aggregated assignment context.
 * @returns True when aggregated tensor exists.
 */
function hasAggregatedLayerWeights(
  aggregatedContext: OnnxImportAggregatedLayerAssignmentContext,
): boolean {
  const tensorNames = buildLayerTensorNames(
    aggregatedContext.nodePair.layerIndex,
  );
  return Boolean(
    aggregatedContext.initializerMap[tensorNames.weightTensorName],
  );
}

/**
 * Build dense weight/bias tensor names for one layer index.
 *
 * @param layerIndex Export layer index.
 * @returns Layer tensor names.
 */
function buildLayerTensorNames(layerIndex: number): OnnxImportLayerTensorNames {
  return {
    weightTensorName: WEIGHT_TENSOR_PREFIX + String(layerIndex),
    biasTensorName: BIAS_TENSOR_PREFIX + String(layerIndex),
  };
}

/**
 * Apply aggregated dense tensor assignments for one layer.
 *
 * @param aggregatedContext Aggregated assignment context.
 * @returns Nothing.
 */
function applyAggregatedLayerWeights(
  aggregatedContext: OnnxImportAggregatedLayerAssignmentContext,
): void {
  // Step 1: Resolve aggregated tensors for this layer.
  const tensorNames = buildLayerTensorNames(
    aggregatedContext.nodePair.layerIndex,
  );
  const aggregatedWeights =
    aggregatedContext.initializerMap[tensorNames.weightTensorName];
  const biasTensor =
    aggregatedContext.initializerMap[tensorNames.biasTensorName];
  if (!aggregatedWeights || !biasTensor) return;

  // Step 2: Assign one target neuron row at a time.
  aggregatedContext.nodePair.currentLayerNodes.forEach(
    (targetNode, targetNodeIndex) => {
      applyAggregatedNeuronAssignment({
        previousLayerNodes: aggregatedContext.nodePair.previousLayerNodes,
        targetNode,
        targetNodeIndex,
        aggregatedWeights,
        biasTensor,
      });
    },
  );
}

/**
 * Apply aggregated dense row weights and bias for one target neuron.
 *
 * @param neuronContext Aggregated neuron assignment context.
 * @returns Nothing.
 */
function applyAggregatedNeuronAssignment(
  neuronContext: OnnxImportAggregatedNeuronAssignmentContext,
): void {
  // Step 1: Resolve target node internals for bias assignment.
  const targetNodeInternal = neuronContext.targetNode as NodeInternals;
  const aggregatedWeightValues = readOnnxTensorFloatData(
    neuronContext.aggregatedWeights,
  );
  const biasValues = readOnnxTensorFloatData(neuronContext.biasTensor);

  // Step 2: Assign incoming connection weights from previous layer.
  neuronContext.previousLayerNodes.forEach((sourceNode, sourceNodeIndex) => {
    const sourceNodeInternal = sourceNode as NodeInternals;
    const connection = sourceNodeInternal.connections.out.find(
      (candidate) => candidate.to === neuronContext.targetNode,
    );
    if (!connection) return;
    const weightIndex =
      neuronContext.targetNodeIndex * neuronContext.previousLayerNodes.length +
      sourceNodeIndex;
    connection.weight = aggregatedWeightValues[weightIndex];
  });

  // Step 3: Assign target neuron bias value.
  targetNodeInternal.bias = biasValues[neuronContext.targetNodeIndex];
}

/**
 * Apply per-neuron tensor assignments for one layer.
 *
 * @param perNeuronContext Per-neuron layer assignment context.
 * @returns Nothing.
 */
function applyPerNeuronLayerWeights(
  perNeuronContext: OnnxImportPerNeuronLayerAssignmentContext,
): void {
  // Step 1: Traverse current layer nodes and resolve per-neuron tensors.
  perNeuronContext.nodePair.currentLayerNodes.forEach(
    (targetNode, neuronIndex) => {
      const tensorNames = buildPerNeuronTensorNames(
        perNeuronContext.nodePair.layerIndex,
        neuronIndex,
      );
      const weightTensor =
        perNeuronContext.initializerMap[tensorNames.weightTensorName];
      const biasTensor =
        perNeuronContext.initializerMap[tensorNames.biasTensorName];
      if (!weightTensor || !biasTensor) return;

      applyPerNeuronAssignment({
        previousLayerNodes: perNeuronContext.nodePair.previousLayerNodes,
        targetNode,
        weightTensor,
        biasTensor,
      });
    },
  );
}

/**
 * Build per-neuron tensor names for one layer and neuron index.
 *
 * @param layerIndex Export layer index.
 * @param neuronIndex Neuron index in layer.
 * @returns Per-neuron tensor names.
 */
function buildPerNeuronTensorNames(
  layerIndex: number,
  neuronIndex: number,
): OnnxImportLayerTensorNames {
  const layerPrefix =
    String(layerIndex) + PER_NEURON_TENSOR_SEGMENT + String(neuronIndex);
  return {
    weightTensorName: WEIGHT_TENSOR_PREFIX + layerPrefix,
    biasTensorName: BIAS_TENSOR_PREFIX + layerPrefix,
  };
}

/**
 * Apply one per-neuron weight vector and bias assignment.
 *
 * @param perNeuronAssignmentContext Per-neuron assignment context.
 * @returns Nothing.
 */
function applyPerNeuronAssignment(
  perNeuronAssignmentContext: OnnxImportPerNeuronAssignmentContext,
): void {
  // Step 1: Resolve target node internals.
  const targetNodeInternal =
    perNeuronAssignmentContext.targetNode as NodeInternals;
  const weightValues = readOnnxTensorFloatData(
    perNeuronAssignmentContext.weightTensor,
  );
  const biasValues = readOnnxTensorFloatData(
    perNeuronAssignmentContext.biasTensor,
  );

  // Step 2: Assign incoming connection weights from per-neuron vector.
  perNeuronAssignmentContext.previousLayerNodes.forEach(
    (sourceNode, sourceNodeIndex) => {
      const sourceNodeInternal = sourceNode as NodeInternals;
      const connection = sourceNodeInternal.connections.out.find(
        (candidate) => candidate.to === perNeuronAssignmentContext.targetNode,
      );
      if (!connection) return;
      connection.weight = weightValues[sourceNodeIndex];
    },
  );

  // Step 3: Assign per-neuron scalar bias.
  targetNodeInternal.bias = biasValues[FIRST_BIAS_INDEX];
}

/**
 * Apply optional Conv2D reconstruction pass from metadata payloads.
 *
 * @param assignmentContext Shared assignment context.
 * @returns Nothing.
 */
function applyOptionalConvReconstruction(
  assignmentContext: OnnxImportWeightAssignmentContext,
): void {
  // Step 1: Parse Conv metadata payload.
  const convMetadata = parseConvMetadata(assignmentContext.metadataProps);
  if (!convMetadata) return;

  // Step 2: Apply Conv reconstruction for each declared layer.
  try {
    convMetadata.convLayers.forEach((layerExportIndex) => {
      const layerContext = buildConvLayerContext({
        assignmentContext,
        convMetadata,
        layerExportIndex,
      });
      if (!layerContext) return;
      applyConvLayerReconstruction(layerContext);
    });
  } catch {
    /* Swallow conv reconstruction errors (experimental). */
  }
}

/**
 * Parse Conv reconstruction metadata payload.
 *
 * @param metadataProps ONNX metadata properties.
 * @returns Parsed Conv metadata.
 */
function parseConvMetadata(
  metadataProps: OnnxMetadataProperty[],
): OnnxImportConvMetadata | null {
  // Step 1: Resolve required Conv metadata entries.
  const convLayersMetadata = metadataProps.find(
    (property) => property.key === METADATA_KEY_CONV2D_LAYERS,
  );
  const convSpecsMetadata = metadataProps.find(
    (property) => property.key === METADATA_KEY_CONV2D_SPECS,
  );
  if (!convLayersMetadata || !convSpecsMetadata) return null;

  // Step 2: Parse Conv layer indices and mapping specs.
  try {
    const convLayers = JSON.parse(convLayersMetadata.value) as number[];
    const convSpecs = JSON.parse(convSpecsMetadata.value) as Conv2DMapping[];
    return { convLayers, convSpecs };
  } catch {
    return null;
  }
}

/**
 * Build one Conv layer reconstruction context.
 *
 * @param params Conv context input params.
 * @returns Conv layer context when valid.
 */
function buildConvLayerContext(
  params: OnnxImportConvLayerContextBuildParams,
): OnnxImportConvLayerContext | null {
  // Step 1: Resolve Conv mapping for requested export layer.
  const convSpec = params.convMetadata.convSpecs.find(
    (specification) => specification.layerIndex === params.layerExportIndex,
  );
  if (!convSpec) return null;

  // Step 2: Validate hidden-layer index range.
  const hiddenLayerIndex = params.layerExportIndex - LAYER_INDEX_OFFSET;
  const isOutOfBounds =
    hiddenLayerIndex < ZERO_VALUE ||
    hiddenLayerIndex >= params.assignmentContext.hiddenLayerSizes.length;
  if (isOutOfBounds) return null;

  // Step 3: Fold validated context payload.
  return {
    onnx: params.assignmentContext.onnx,
    hiddenLayerSizes: params.assignmentContext.hiddenLayerSizes,
    hiddenNodes: params.assignmentContext.hiddenNodes,
    inputNodes: params.assignmentContext.inputNodes,
    layerExportIndex: params.layerExportIndex,
    convSpec,
    convSpecs: params.convMetadata.convSpecs,
    poolingSpecs: parsePoolingSpecs(params.assignmentContext.metadataProps),
  };
}

function parsePoolingSpecs(
  metadataProps: OnnxMetadataProperty[],
): Pool2DMapping[] {
  const poolingSpecsMetadata = metadataProps.find(
    (property) => property.key === METADATA_KEY_POOL2D_SPECS,
  );
  if (!poolingSpecsMetadata) {
    return [];
  }

  try {
    const poolingSpecs = JSON.parse(poolingSpecsMetadata.value);
    return Array.isArray(poolingSpecs)
      ? (poolingSpecs as Pool2DMapping[])
      : [];
  } catch {
    return [];
  }
}

/**
 * Apply Conv reconstruction for one validated Conv layer context.
 *
 * @param layerContext Conv layer context.
 * @returns Nothing.
 */
function applyConvLayerReconstruction(
  layerContext: OnnxImportConvLayerContext,
): void {
  // Step 1: Resolve layer and previous-layer node slices.
  const nodeSlices = buildConvNodeSlices(layerContext);

  // Step 2: Resolve and validate Conv initializer tensors.
  const tensorContext = buildConvTensorContext(layerContext);
  if (!tensorContext) return;

  // Step 3: Precompute shared kernel coordinates for all output positions.
  const kernelCoordinates = collectConvKernelCoordinates(
    tensorContext.inChannels,
    tensorContext.kernelHeight,
    tensorContext.kernelWidth,
  );
  const sourceLayout = resolveConvSourceLayout(layerContext);

  // Step 4: Traverse all output coordinates and assign weights.
  const outputCoordinates = collectConvOutputCoordinates(
    layerContext.convSpec,
    tensorContext.outChannels,
  );
  outputCoordinates.forEach((coordinate) => {
    applyConvCoordinateAssignment({
      coordinate,
      convSpec: layerContext.convSpec,
      sourceLayout,
      tensorContext,
      kernelCoordinates,
      layerNodes: nodeSlices.layerNodes,
      previousLayerNodes: nodeSlices.previousLayerNodes,
    });
  });
}

function resolveConvSourceLayout(
  layerContext: OnnxImportConvLayerContext,
): OnnxImportConvSourceLayout {
  const defaultLayout = {
    channelStride:
      layerContext.convSpec.inHeight * layerContext.convSpec.inWidth,
    sourceHeight: layerContext.convSpec.inHeight,
    sourceWidth: layerContext.convSpec.inWidth,
  };
  const upstreamPoolingSpec = layerContext.poolingSpecs.find(
    (poolingSpec) =>
      poolingSpec.afterLayerIndex === layerContext.layerExportIndex - 1,
  );
  const upstreamConvSpec = layerContext.convSpecs.find(
    (convSpec) => convSpec.layerIndex === layerContext.layerExportIndex - 1,
  );
  if (!upstreamPoolingSpec || !upstreamConvSpec) {
    return defaultLayout;
  }

  const pooledHeight = calculateSpatialOutputSize(
    upstreamConvSpec.outHeight,
    upstreamPoolingSpec.kernelHeight,
    upstreamPoolingSpec.strideHeight,
    upstreamPoolingSpec.padTop ?? ZERO_VALUE,
    upstreamPoolingSpec.padBottom ?? ZERO_VALUE,
  );
  const pooledWidth = calculateSpatialOutputSize(
    upstreamConvSpec.outWidth,
    upstreamPoolingSpec.kernelWidth,
    upstreamPoolingSpec.strideWidth,
    upstreamPoolingSpec.padLeft ?? ZERO_VALUE,
    upstreamPoolingSpec.padRight ?? ZERO_VALUE,
  );
  const matchesDerivedPooledShape =
    pooledHeight === layerContext.convSpec.inHeight &&
    pooledWidth === layerContext.convSpec.inWidth &&
    upstreamConvSpec.outChannels === layerContext.convSpec.inChannels;
  if (!matchesDerivedPooledShape) {
    return defaultLayout;
  }

  return {
    channelStride: upstreamConvSpec.outHeight * upstreamConvSpec.outWidth,
    sourceHeight: layerContext.convSpec.inHeight,
    sourceWidth: layerContext.convSpec.inWidth,
  };
}

function calculateSpatialOutputSize(
  inputSize: number,
  kernelSize: number,
  strideSize: number,
  leadingPadding: number,
  trailingPadding: number,
): number {
  if (inputSize <= ZERO_VALUE || kernelSize <= ZERO_VALUE || strideSize <= ZERO_VALUE) {
    return ZERO_VALUE;
  }

  return (
    Math.floor(
      (inputSize + leadingPadding + trailingPadding - kernelSize) /
        strideSize,
    ) + 1
  );
}

/**
 * Build Conv current/previous node slices for one layer context.
 *
 * @param layerContext Conv layer context.
 * @returns Node slice payload.
 */
function buildConvNodeSlices(
  layerContext: OnnxImportConvLayerContext,
): OnnxImportConvNodeSlices {
  // Step 1: Resolve hidden layer boundaries.
  const hiddenLayerIndex = layerContext.layerExportIndex - LAYER_INDEX_OFFSET;
  const layerStart = sumHiddenSizesToIndex(
    layerContext.hiddenLayerSizes,
    hiddenLayerIndex,
  );
  const layerEnd = layerStart + layerContext.hiddenLayerSizes[hiddenLayerIndex];

  // Step 2: Resolve current and previous layer node slices.
  const layerNodes = layerContext.hiddenNodes.slice(layerStart, layerEnd);
  const previousLayerNodes =
    hiddenLayerIndex === ZERO_VALUE
      ? layerContext.inputNodes
      : layerContext.hiddenNodes.slice(
          sumHiddenSizesToIndex(
            layerContext.hiddenLayerSizes,
            hiddenLayerIndex - LAYER_INDEX_OFFSET,
          ),
          sumHiddenSizesToIndex(
            layerContext.hiddenLayerSizes,
            hiddenLayerIndex,
          ),
        );

  // Step 3: Fold slice payload.
  return { layerNodes, previousLayerNodes };
}

/**
 * Build validated Conv tensor context for one layer.
 *
 * @param layerContext Conv layer context.
 * @returns Conv tensor context when valid.
 */
function buildConvTensorContext(
  layerContext: OnnxImportConvLayerContext,
): OnnxImportConvTensorContext | null {
  // Step 1: Resolve Conv weight and bias tensor names.
  const tensorLayerIndex = layerContext.layerExportIndex - LAYER_INDEX_OFFSET;
  const convWeightTensorName =
    CONV_WEIGHT_TENSOR_PREFIX + String(tensorLayerIndex);
  const convBiasTensorName = CONV_BIAS_TENSOR_PREFIX + String(tensorLayerIndex);

  // Step 2: Resolve tensors from ONNX initializers.
  const convWeightTensor = layerContext.onnx.graph.initializer.find(
    (tensor) => tensor.name === convWeightTensorName,
  );
  const convBiasTensor = layerContext.onnx.graph.initializer.find(
    (tensor) => tensor.name === convBiasTensorName,
  );
  if (!convWeightTensor || !convBiasTensor) return null;

  // Step 3: Validate Conv kernel dimensions against metadata mapping.
  const [outChannels, inChannels, kernelHeight, kernelWidth] =
    convWeightTensor.dims as [number, number, number, number];
  const dimensionsMatch =
    outChannels === layerContext.convSpec.outChannels &&
    inChannels === layerContext.convSpec.inChannels &&
    kernelHeight === layerContext.convSpec.kernelHeight &&
    kernelWidth === layerContext.convSpec.kernelWidth;
  if (!dimensionsMatch) return null;

  // Step 4: Fold validated tensor payload.
  return {
    convWeightTensor,
    convBiasTensor,
    outChannels,
    inChannels,
    kernelHeight,
    kernelWidth,
  };
}

/**
 * Collect all output traversal coordinates for one Conv layer.
 *
 * @param convSpec Conv mapping spec.
 * @param outChannels Output channel count.
 * @returns Output traversal coordinates.
 */
function collectConvOutputCoordinates(
  convSpec: Conv2DMapping,
  outChannels: number,
): OnnxImportConvOutputCoordinate[] {
  return Array.from({ length: outChannels }, (_unusedOut, outChannelIndex) =>
    Array.from({ length: convSpec.outHeight }, (_unusedRow, outRowIndex) =>
      Array.from(
        { length: convSpec.outWidth },
        (_unusedColumn, outColumnIndex) => ({
          outChannelIndex,
          outRowIndex,
          outColumnIndex,
        }),
      ),
    ).flat(),
  ).flat();
}

/**
 * Apply Conv bias and kernel weights for one output coordinate.
 *
 * @param coordinateContext Conv coordinate assignment context.
 * @returns Nothing.
 */
function applyConvCoordinateAssignment(
  coordinateContext: OnnxImportConvCoordinateAssignmentContext,
): void {
  // Step 1: Resolve target neuron for this output coordinate.
  const neuronLinearIndex = buildConvNeuronLinearIndex(
    coordinateContext.coordinate,
    coordinateContext.convSpec,
  );
  const neuron = coordinateContext.layerNodes[neuronLinearIndex];
  if (!neuron) return;

  // Step 2: Resolve neuron internals and assign channel bias.
  const neuronInternal = neuron as NodeInternals;
  const convBiasValues = readOnnxTensorFloatData(
    coordinateContext.tensorContext.convBiasTensor,
  );
  neuronInternal.bias = convBiasValues[coordinateContext.coordinate.outChannelIndex];

  // Step 3: Reset all inbound weights so non-receptive positions stay zero.
  resetInboundConnectionWeights(neuronInternal);

  // Step 4: Build inbound map once for this target neuron.
  const inboundConnectionMap = buildInboundConnectionMap(neuronInternal);

  // Step 5: Assign inbound kernel-based connection weights.
  coordinateContext.kernelCoordinates.forEach((kernelCoordinate) => {
    assignConvKernelWeight({
      tensorContext: coordinateContext.tensorContext,
      convSpec: coordinateContext.convSpec,
      sourceLayout: coordinateContext.sourceLayout,
      coordinate: coordinateContext.coordinate,
      inChannelIndex: kernelCoordinate.inChannelIndex,
      kernelRowIndex: kernelCoordinate.kernelRowIndex,
      kernelColumnIndex: kernelCoordinate.kernelColumnIndex,
      inboundConnectionMap,
      previousLayerNodes: coordinateContext.previousLayerNodes,
    });
  });
}

/**
 * Reset all inbound weights so Conv reconstruction can write only receptive edges.
 *
 * @param neuronInternal Target neuron internals.
 * @returns Nothing.
 */
function resetInboundConnectionWeights(neuronInternal: NodeInternals): void {
  neuronInternal.connections.in.forEach((connection) => {
    connection.weight = ZERO_VALUE;
  });
}

/**
 * Build flattened linear index for one Conv output coordinate.
 *
 * @param coordinate Conv output coordinate.
 * @param convSpec Conv mapping spec.
 * @returns Linear neuron index.
 */
function buildConvNeuronLinearIndex(
  coordinate: OnnxImportConvOutputCoordinate,
  convSpec: Conv2DMapping,
): number {
  return (
    coordinate.outChannelIndex * (convSpec.outHeight * convSpec.outWidth) +
    coordinate.outRowIndex * convSpec.outWidth +
    coordinate.outColumnIndex
  );
}

/**
 * Collect all kernel traversal coordinates for one Conv output position.
 *
 * @param inChannels Input channel count.
 * @param kernelHeight Kernel height.
 * @param kernelWidth Kernel width.
 * @returns Kernel traversal coordinates.
 */
function collectConvKernelCoordinates(
  inChannels: number,
  kernelHeight: number,
  kernelWidth: number,
): OnnxConvKernelCoordinate[] {
  return Array.from({ length: inChannels }, (_unusedChannel, inChannelIndex) =>
    Array.from({ length: kernelHeight }, (_unusedRow, kernelRowIndex) =>
      Array.from(
        { length: kernelWidth },
        (_unusedColumn, kernelColumnIndex) => ({
          inChannelIndex,
          kernelRowIndex,
          kernelColumnIndex,
        }),
      ),
    ).flat(),
  ).flat();
}

/**
 * Assign one Conv kernel weight to the matching inbound neuron connection.
 *
 * @param kernelAssignmentContext Conv kernel assignment context.
 * @returns Nothing.
 */
function assignConvKernelWeight(
  kernelAssignmentContext: OnnxImportConvKernelAssignmentContext,
): void {
  // Step 1: Resolve input feature-space coordinate.
  const inputCoordinate = buildInputCoordinate(kernelAssignmentContext);
  if (!inputCoordinate) return;

  // Step 2: Resolve source node index and outbound source node.
  const sourceNodeIndex = buildInputFeatureLinearIndex(
    kernelAssignmentContext.sourceLayout,
    kernelAssignmentContext.inChannelIndex,
    inputCoordinate.inputRow,
    inputCoordinate.inputColumn,
  );
  const sourceNode =
    kernelAssignmentContext.previousLayerNodes[sourceNodeIndex];
  if (!sourceNode) return;

  // Step 3: Resolve inbound connection and assign kernel weight.
  const connection =
    kernelAssignmentContext.inboundConnectionMap.get(sourceNode);
  if (!connection) return;
  connection.weight = readConvKernelWeight(kernelAssignmentContext);
}

/**
 * Build input-space coordinate for one Conv kernel element.
 *
 * @param kernelAssignmentContext Conv kernel assignment context.
 * @returns Input coordinate when in bounds.
 */
function buildInputCoordinate(
  kernelAssignmentContext: OnnxImportConvKernelAssignmentContext,
): { inputRow: number; inputColumn: number } | null {
  const inputRowBase =
    kernelAssignmentContext.coordinate.outRowIndex *
      kernelAssignmentContext.convSpec.strideHeight -
    (kernelAssignmentContext.convSpec.padTop ?? ZERO_VALUE);
  const inputColumnBase =
    kernelAssignmentContext.coordinate.outColumnIndex *
      kernelAssignmentContext.convSpec.strideWidth -
    (kernelAssignmentContext.convSpec.padLeft ?? ZERO_VALUE);
  const inputRow = inputRowBase + kernelAssignmentContext.kernelRowIndex;
  const inputColumn =
    inputColumnBase + kernelAssignmentContext.kernelColumnIndex;

  const isOutOfInputBounds =
    inputRow < ZERO_VALUE ||
    inputRow >= kernelAssignmentContext.convSpec.inHeight ||
    inputColumn < ZERO_VALUE ||
    inputColumn >= kernelAssignmentContext.convSpec.inWidth;
  if (isOutOfInputBounds) return null;
  return { inputRow, inputColumn };
}

/**
 * Build linear feature index in input feature space.
 *
 * @param convSpec Conv mapping spec.
 * @param inChannelIndex Input channel index.
 * @param inputRow Input row index.
 * @param inputColumn Input column index.
 * @returns Linear input feature index.
 */
function buildInputFeatureLinearIndex(
  sourceLayout: OnnxImportConvSourceLayout,
  inChannelIndex: number,
  inputRow: number,
  inputColumn: number,
): number {
  return (
    inChannelIndex * sourceLayout.channelStride +
    inputRow * sourceLayout.sourceWidth +
    inputColumn
  );
}

/**
 * Build inbound connection lookup map for one neuron.
 *
 * @param neuronInternal Neuron internals.
 * @returns Inbound connection map keyed by source node.
 */
function buildInboundConnectionMap(
  neuronInternal: NodeInternals,
): OnnxImportInboundConnectionMap {
  return neuronInternal.connections.in.reduce(
    (connectionMap, connection) =>
      connectionMap.set(connection.from as NeatapticNode, connection),
    new Map<NeatapticNode, Connection>(),
  );
}

/**
 * Read one Conv kernel weight from flattened ONNX tensor payload.
 *
 * @param kernelAssignmentContext Conv kernel assignment context.
 * @returns Kernel weight.
 */
function readConvKernelWeight(
  kernelAssignmentContext: OnnxImportConvKernelAssignmentContext,
): number {
  const convWeightValues = readOnnxTensorFloatData(
    kernelAssignmentContext.tensorContext.convWeightTensor,
  );
  const weightIndex =
    ((kernelAssignmentContext.coordinate.outChannelIndex *
      kernelAssignmentContext.tensorContext.inChannels +
      kernelAssignmentContext.inChannelIndex) *
      kernelAssignmentContext.tensorContext.kernelHeight +
      kernelAssignmentContext.kernelRowIndex) *
      kernelAssignmentContext.tensorContext.kernelWidth +
    kernelAssignmentContext.kernelColumnIndex;
  return convWeightValues[weightIndex];
}
