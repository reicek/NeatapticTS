import type NeatapticNode from '../../../node';
import type {
  OnnxGraph,
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
} from '../schema/network.onnx.schema.types';
import type {
  ConvLayerPairContext,
  ConvKernelConsistencyContext,
  ConvOutputCoordinate,
  ConvRepresentativeKernelContext,
  ConvSharingValidationContext,
  ConvSharingValidationResult,
  FusedRecurrentEmissionExecutionContext,
  FusedRecurrentGraphNames,
  FusedRecurrentInitializerNames,
  GruEmissionContext,
  HiddenLayerHeuristicContext,
  LstmEmissionContext,
  OnnxExportOptions,
  RecurrentGateBlockCollectionContext,
  RecurrentGateParameterCollectionResult,
  RecurrentGateRow,
  RecurrentGateRowCollectionContext,
  RecurrentHeuristicEmissionContext,
  WeightToleranceComparisonContext,
} from './network.onnx.export.types';
import type {
  NodeInternals,
  OnnxConvKernelCoordinate,
} from '../network.onnx.utils.types';

/** Minimum hidden-node count for LSTM heuristic eligibility. */
const LSTM_MIN_SIZE = 10;

/** Hidden-node divisor for the LSTM heuristic gate layout (5 slices). */
const LSTM_UNIT_DIVISOR = 5;

/** Minimum hidden-node count for GRU heuristic eligibility. */
const GRU_MIN_SIZE = 8;

/** Hidden-node divisor for the GRU heuristic gate layout (4 slices). */
const GRU_UNIT_DIVISOR = 4;

/** Lower-bound inclusive fallback threshold used for ambiguous recurrent sizing metadata. */
const RECURRENT_FALLBACK_MIN_SIZE = 8;

/** Upper-bound exclusive fallback threshold used for ambiguous recurrent sizing metadata. */
const RECURRENT_FALLBACK_MAX_SIZE = 10;

/** Numeric tolerance used for Conv2D sharing consistency checks. */
const CONV_WEIGHT_SHARING_TOLERANCE = 1e-9;

/** ONNX tensor float data type enum value for FLOAT tensors. */
const ONNX_FLOAT_DATA_TYPE = 1;

/** ONNX attribute type literal for integer attributes. */
const ONNX_ATTRIBUTE_TYPE_INT = 'INT';

/** ONNX hidden-size attribute key for recurrent nodes. */
const ONNX_ATTRIBUTE_HIDDEN_SIZE_KEY = 'hidden_size';

/** ONNX layout attribute key for recurrent nodes. */
const ONNX_ATTRIBUTE_LAYOUT_KEY = 'layout';

/** Default ONNX recurrent layout attribute value. */
const ONNX_LAYOUT_DEFAULT = 0;

/** Metadata key for hidden-layer size collection. */
const METADATA_KEY_LAYER_SIZES = 'layer_sizes';

/** Metadata key for recurrent single-step layers. */
const METADATA_KEY_RECURRENT_SINGLE_STEP = 'recurrent_single_step';

/** Metadata key for ambiguous recurrent-size fallback annotation. */
const METADATA_KEY_RNN_PATTERN_FALLBACK = 'rnn_pattern_fallback';

/** Metadata key for Conv sharing verified layer indices. */
const METADATA_KEY_CONV2D_SHARING_VERIFIED = 'conv2d_sharing_verified';

/** Metadata key for Conv sharing mismatch layer indices. */
const METADATA_KEY_CONV2D_SHARING_MISMATCH = 'conv2d_sharing_mismatch';

/** Metadata key for emitted LSTM heuristic layers. */
const METADATA_KEY_LSTM_EMITTED_LAYERS = 'lstm_emitted_layers';

/** Metadata key for emitted GRU heuristic layers. */
const METADATA_KEY_GRU_EMITTED_LAYERS = 'gru_emitted_layers';

/** Metadata key describing dense-family initializer aliases reused during export. */
const METADATA_KEY_SHARED_INITIALIZER_ALIASES = 'shared_initializer_aliases';

/** Metadata fallback reason for in-between GRU/LSTM size thresholds. */
const METADATA_REASON_SIZE_BETWEEN_GRU_LSTM_THRESHOLDS =
  'size_between_gru_lstm_thresholds';

/** ONNX operator label for LSTM. */
const ONNX_OPERATOR_LSTM = 'LSTM';

/** ONNX operator label for GRU. */
const ONNX_OPERATOR_GRU = 'GRU';

/** Node naming prefix for LSTM layers. */
const LSTM_NODE_PREFIX = 'lstm';

/** Node naming prefix for GRU layers. */
const GRU_NODE_PREFIX = 'gru';

/** Output naming suffix for LSTM hidden tensors. */
const LSTM_OUTPUT_SUFFIX = 'lstm_hidden';

/** Output naming suffix for GRU hidden tensors. */
const GRU_OUTPUT_SUFFIX = 'gru_hidden';

/** Prefix for generated layer output tensor names. */
const GENERATED_LAYER_OUTPUT_PREFIX = 'Layer';

/** Base graph input tensor name for first recurrent layer. */
const GRAPH_INPUT_NAME = 'input';

/** Recurrent diagonal gate index for LSTM (cell gate). */
const LSTM_DIAGONAL_GATE_INDEX = 2;

/** Recurrent diagonal gate index for GRU (candidate gate). */
const GRU_DIAGONAL_GATE_INDEX = 2;

type SharedInitializerAliasRecord = {
  aliasTensorName: string;
  canonicalTensorName: string;
  initializerKind: 'dense_weight' | 'dense_bias' | 'per_neuron_weight' | 'per_neuron_bias';
};

/**
 * Emit heuristic fused recurrent operators (LSTM/GRU) when recurrent export is enabled.
 *
 * @param model Target ONNX model.
 * @param layers Layered network nodes.
 * @param allowRecurrent Whether recurrent export is enabled.
 * @param previousOutputName Current graph output name (kept for backward-compatible emission semantics).
 * @returns Nothing.
 */
export function emitFusedRecurrentHeuristics(
  model: OnnxModel,
  layers: NeatapticNode[][],
  allowRecurrent: boolean | undefined,
  previousOutputName: string,
): void {
  if (!allowRecurrent) {
    return;
  }

  // Step 1: Build traversal context for hidden-layer heuristic emission.
  const emissionContext = buildRecurrentHeuristicEmissionContext(
    model,
    layers,
    previousOutputName,
  );

  // Step 2: Traverse hidden layers and emit tiny focused recurrent heuristics.
  const hiddenLayerIndices = collectHiddenLayerIndices(emissionContext.layers);
  for (const layerIndex of hiddenLayerIndices) {
    const hiddenLayerContext = buildHiddenLayerHeuristicContext(
      emissionContext,
      layerIndex,
    );
    emitFallbackRecurrentPatternMetadata(hiddenLayerContext);
    tryEmitFusedLstm(hiddenLayerContext);
    tryEmitFusedGru(hiddenLayerContext);
  }
}

/**
 * Finalize export metadata and optional conv-sharing validation.
 *
 * @param model Target ONNX model.
 * @param layers Layered network nodes.
 * @param options Export options.
 * @param includeMetadata Whether metadata emission is enabled.
 * @param hiddenSizesMetadata Hidden-layer sizes collected during emission.
 * @param recurrentLayerIndices Recurrent layer indices.
 * @returns Nothing.
 */
export function finalizeExportMetadata(
  model: OnnxModel,
  layers: NeatapticNode[][],
  options: OnnxExportOptions,
  includeMetadata: boolean,
  hiddenSizesMetadata: number[],
  recurrentLayerIndices: number[],
): void {
  if (!includeMetadata) {
    return;
  }

  // Step 1: Reuse exact dense-family initializer aliases before metadata finalization.
  const sharedInitializerAliases = reuseSharedInitializers(model);
  if (sharedInitializerAliases.length) {
    appendMetadataProperty(
      model,
      buildMetadataProperty(
        METADATA_KEY_SHARED_INITIALIZER_ALIASES,
        sharedInitializerAliases,
      ),
    );
  }

  // Step 2: Append baseline export metadata.
  appendMetadataProperty(
    model,
    buildMetadataProperty(METADATA_KEY_LAYER_SIZES, hiddenSizesMetadata),
  );
  appendRecurrentSingleStepMetadata(model, recurrentLayerIndices);

  // Step 3: Optionally evaluate Conv2D sharing and append summary metadata.
  if (!shouldValidateConvSharing(options)) {
    return;
  }
  const convSharingResult = validateConvSharingAcrossMappings({
    layers,
    mappings: options.conv2dMappings || [],
  });
  appendConvSharingMetadata(model, convSharingResult);
}

/** Reuse exact dense-family initializers and rewrite later node inputs to the canonical tensors. */
function reuseSharedInitializers(
  model: OnnxModel,
): SharedInitializerAliasRecord[] {
  const signatureToCanonicalTensorName = new Map<string, string>();
  const initializerAliases: SharedInitializerAliasRecord[] = [];
  const aliasTensorNameByRemovedTensorName = new Map<string, string>();

  model.graph.initializer = model.graph.initializer.filter((initializerEntry) => {
    const initializerKind = classifySharedInitializerKind(initializerEntry.name);
    if (!initializerKind) {
      return true;
    }

    const initializerSignature = buildSharedInitializerSignature(
      initializerEntry,
      initializerKind,
    );
    const canonicalTensorName =
      signatureToCanonicalTensorName.get(initializerSignature);
    if (!canonicalTensorName) {
      signatureToCanonicalTensorName.set(
        initializerSignature,
        initializerEntry.name,
      );
      return true;
    }

    aliasTensorNameByRemovedTensorName.set(
      initializerEntry.name,
      canonicalTensorName,
    );
    initializerAliases.push({
      aliasTensorName: initializerEntry.name,
      canonicalTensorName,
      initializerKind,
    });
    return false;
  });

  if (!initializerAliases.length) {
    return [];
  }

  rewriteInitializerInputs(model.graph, aliasTensorNameByRemovedTensorName);
  return initializerAliases;
}

/** Classify the dense-family initializer kinds supported by the Phase 5B alias subset. */
function classifySharedInitializerKind(
  initializerName: string,
): SharedInitializerAliasRecord['initializerKind'] | null {
  if (/^W\d+$/.test(initializerName)) {
    return 'dense_weight';
  }
  if (/^B\d+$/.test(initializerName)) {
    return 'dense_bias';
  }
  if (/^W\d+_n\d+$/.test(initializerName)) {
    return 'per_neuron_weight';
  }
  if (/^B\d+_n\d+$/.test(initializerName)) {
    return 'per_neuron_bias';
  }
  return null;
}

/** Build an exact-match signature for dense-family alias reuse. */
function buildSharedInitializerSignature(
  initializerEntry: OnnxTensor,
  initializerKind: SharedInitializerAliasRecord['initializerKind'],
): string {
  return JSON.stringify({
    initializerKind,
    dims: initializerEntry.dims,
    floatData: initializerEntry.float_data,
  });
}

/** Rewrite graph-node initializer inputs after later aliases collapse into one canonical tensor. */
function rewriteInitializerInputs(
  graph: OnnxGraph,
  aliasTensorNameByRemovedTensorName: Map<string, string>,
): void {
  graph.node.forEach((nodeEntry) => {
    nodeEntry.input = nodeEntry.input.map(
      (inputName) =>
        aliasTensorNameByRemovedTensorName.get(inputName) ?? inputName,
    );
  });
}

/**
 * Try emitting heuristic fused LSTM node and metadata.
 */
function tryEmitFusedLstm(context: HiddenLayerHeuristicContext): void {
  if (!isEligibleForLstmHeuristic(context.currentSize)) {
    return;
  }

  // Step 1: Build focused emission context.
  const lstmContext = buildLstmEmissionContext(context);
  const executionContext = buildFusedLstmExecutionContext(lstmContext);

  // Step 2: Emit fused LSTM payload.
  emitFusedRecurrentLayer(executionContext);
}

/** Build shared fused-recurrent execution context for LSTM. */
function buildFusedLstmExecutionContext(
  context: LstmEmissionContext,
): FusedRecurrentEmissionExecutionContext {
  return {
    model: context.model,
    layerIndex: context.layerIndex,
    previousOutputName: context.previousOutputName,
    previousLayerNodes: context.previousLayerNodes,
    gateNodeGroups: collectLstmGateNodeGroups(context),
    unitSize: context.unitSize,
    diagonalGateIndex: LSTM_DIAGONAL_GATE_INDEX,
    operatorType: ONNX_OPERATOR_LSTM,
    metadataKey: METADATA_KEY_LSTM_EMITTED_LAYERS,
    nodePrefix: LSTM_NODE_PREFIX,
    outputSuffix: LSTM_OUTPUT_SUFFIX,
  };
}

/**
 * Try emitting heuristic fused GRU node and metadata.
 */
function tryEmitFusedGru(context: HiddenLayerHeuristicContext): void {
  if (!isEligibleForGruHeuristic(context.currentSize)) {
    return;
  }

  // Step 1: Build focused emission context.
  const gruContext = buildGruEmissionContext(context);
  const executionContext = buildFusedGruExecutionContext(gruContext);

  // Step 2: Emit fused GRU payload.
  emitFusedRecurrentLayer(executionContext);
}

/** Build shared fused-recurrent execution context for GRU. */
function buildFusedGruExecutionContext(
  context: GruEmissionContext,
): FusedRecurrentEmissionExecutionContext {
  return {
    model: context.model,
    layerIndex: context.layerIndex,
    previousOutputName: resolveGruPreviousOutputName(context.layerIndex),
    previousLayerNodes: context.previousLayerNodes,
    gateNodeGroups: collectGruGateNodeGroups(context),
    unitSize: context.unitSize,
    diagonalGateIndex: GRU_DIAGONAL_GATE_INDEX,
    operatorType: ONNX_OPERATOR_GRU,
    metadataKey: METADATA_KEY_GRU_EMITTED_LAYERS,
    nodePrefix: GRU_NODE_PREFIX,
    outputSuffix: GRU_OUTPUT_SUFFIX,
  };
}

/** Emit shared fused recurrent payload (initializers, node, metadata). */
function emitFusedRecurrentLayer(
  context: FusedRecurrentEmissionExecutionContext,
): void {
  // Step 1: Collect and fold gate parameters.
  const gateParameterBlocks = context.gateNodeGroups.map(
    (gateNodes, gateIndex) =>
      collectRecurrentGateBlockParameters({
        gateNodes,
        previousLayerNodes: context.previousLayerNodes,
        unitSize: context.unitSize,
        useDiagonalSelfWeights: gateIndex === context.diagonalGateIndex,
      }),
  );
  const fusedParameters = foldRecurrentGateBlocks(gateParameterBlocks);

  // Step 2: Build graph names and append ONNX payloads.
  const initializerNames = buildFusedRecurrentInitializerNames(
    context.operatorType,
    context.layerIndex,
  );
  const graphNames = buildFusedRecurrentGraphNames(
    context.nodePrefix,
    context.outputSuffix,
    context.layerIndex,
  );
  appendFusedRecurrentInitializers(
    context.model,
    initializerNames,
    fusedParameters,
    context.gateNodeGroups.length,
    context.unitSize,
    context.previousLayerNodes.length,
  );
  appendFusedRecurrentNode(
    context.model.graph,
    context.operatorType,
    context.previousOutputName,
    initializerNames,
    graphNames,
    context.unitSize,
  );
  appendIndexMetadata(context.model, context.metadataKey, context.layerIndex);
}

/**
 * Append a unique layer index to metadata array key.
 */
function appendIndexMetadata(
  model: OnnxModel,
  key: string,
  layerIndex: number,
): void {
  const metadataProperties = ensureMetadataProps(model);
  const metadataIndex = findMetadataPropertyIndex(metadataProperties, key);
  if (metadataIndex >= 0) {
    upsertLayerIndexMetadataValue(
      metadataProperties,
      metadataIndex,
      layerIndex,
    );
    return;
  }
  metadataProperties.push({ key, value: JSON.stringify([layerIndex]) });
}

/** Find metadata property index by key. */
function findMetadataPropertyIndex(
  metadataProperties: OnnxMetadataProperty[],
  key: string,
): number {
  return metadataProperties.findIndex((property) => property.key === key);
}

/** Upsert one layer index into metadata array-like JSON value. */
function upsertLayerIndexMetadataValue(
  metadataProperties: OnnxMetadataProperty[],
  metadataIndex: number,
  layerIndex: number,
): void {
  const existingLayerIndices = parseMetadataLayerIndices(
    metadataProperties[metadataIndex].value,
  );
  if (existingLayerIndices.includes(layerIndex)) {
    return;
  }
  metadataProperties[metadataIndex].value = JSON.stringify([
    ...existingLayerIndices,
    layerIndex,
  ]);
}

/** Parse metadata JSON value into a numeric layer-index array. */
function parseMetadataLayerIndices(metadataValue: string): number[] {
  try {
    const parsedValue = JSON.parse(metadataValue);
    return Array.isArray(parsedValue)
      ? parsedValue.filter(
          (entry): entry is number => typeof entry === 'number',
        )
      : [];
  } catch {
    return [];
  }
}

/** Build reusable context for recurrent heuristic traversal. */
function buildRecurrentHeuristicEmissionContext(
  model: OnnxModel,
  layers: NeatapticNode[][],
  previousOutputName: string,
): RecurrentHeuristicEmissionContext {
  return { model, layers, previousOutputName };
}

/** Collect hidden-layer indices for recurrent traversal. */
function collectHiddenLayerIndices(layers: NeatapticNode[][]): number[] {
  const hiddenLayerCount = Math.max(layers.length - 2, 0);
  return Array.from(
    { length: hiddenLayerCount },
    (_unused, offsetIndex) => offsetIndex + 1,
  );
}

/** Build one hidden-layer traversal context. */
function buildHiddenLayerHeuristicContext(
  context: RecurrentHeuristicEmissionContext,
  layerIndex: number,
): HiddenLayerHeuristicContext {
  const currentLayerNodes = context.layers[layerIndex] || [];
  return {
    model: context.model,
    layers: context.layers,
    layerIndex,
    previousOutputName: context.previousOutputName,
    currentLayerNodes,
    currentSize: currentLayerNodes.length,
  };
}

/** Emit fallback metadata for recurrent-size ambiguity. */
function emitFallbackRecurrentPatternMetadata(
  context: HiddenLayerHeuristicContext,
): void {
  if (!isFallbackRecurrentPatternSize(context.currentSize)) {
    return;
  }
  appendMetadataProperty(
    context.model,
    buildMetadataProperty(METADATA_KEY_RNN_PATTERN_FALLBACK, {
      layer: context.layerIndex,
      reason: METADATA_REASON_SIZE_BETWEEN_GRU_LSTM_THRESHOLDS,
    }),
  );
}

/** Check whether hidden size should emit recurrent fallback metadata. */
function isFallbackRecurrentPatternSize(currentSize: number): boolean {
  return (
    currentSize >= RECURRENT_FALLBACK_MIN_SIZE &&
    currentSize < RECURRENT_FALLBACK_MAX_SIZE
  );
}

/** Check LSTM heuristic eligibility by size and gate divisibility. */
function isEligibleForLstmHeuristic(currentSize: number): boolean {
  return currentSize >= LSTM_MIN_SIZE && currentSize % LSTM_UNIT_DIVISOR === 0;
}

/** Build LSTM emission context from one hidden-layer traversal record. */
function buildLstmEmissionContext(
  context: HiddenLayerHeuristicContext,
): LstmEmissionContext {
  return {
    model: context.model,
    layerIndex: context.layerIndex,
    previousOutputName: context.previousOutputName,
    previousLayerNodes: context.layers[context.layerIndex - 1] || [],
    currentLayerNodes: context.currentLayerNodes,
    unitSize: context.currentSize / LSTM_UNIT_DIVISOR,
  };
}

/** Collect LSTM gate node groups in canonical export order. */
function collectLstmGateNodeGroups(
  context: LstmEmissionContext,
): NeatapticNode[][] {
  const unitSize = context.unitSize;
  const inputGateNodes = context.currentLayerNodes.slice(0, unitSize);
  const forgetGateNodes = context.currentLayerNodes.slice(
    unitSize,
    unitSize * 2,
  );
  const cellGateNodes = context.currentLayerNodes.slice(
    unitSize * 2,
    unitSize * 3,
  );
  const outputGateNodes = context.currentLayerNodes.slice(
    unitSize * 3,
    unitSize * 4,
  );
  return [inputGateNodes, forgetGateNodes, cellGateNodes, outputGateNodes];
}

/** Check GRU heuristic eligibility by size and gate divisibility. */
function isEligibleForGruHeuristic(currentSize: number): boolean {
  return currentSize >= GRU_MIN_SIZE && currentSize % GRU_UNIT_DIVISOR === 0;
}

/** Build GRU emission context from one hidden-layer traversal record. */
function buildGruEmissionContext(
  context: HiddenLayerHeuristicContext,
): GruEmissionContext {
  return {
    model: context.model,
    layerIndex: context.layerIndex,
    previousLayerNodes: context.layers[context.layerIndex - 1] || [],
    currentLayerNodes: context.currentLayerNodes,
    unitSize: context.currentSize / GRU_UNIT_DIVISOR,
  };
}

/** Collect GRU gate node groups in canonical export order. */
function collectGruGateNodeGroups(
  context: GruEmissionContext,
): NeatapticNode[][] {
  const unitSize = context.unitSize;
  const updateGateNodes = context.currentLayerNodes.slice(0, unitSize);
  const resetGateNodes = context.currentLayerNodes.slice(
    unitSize,
    unitSize * 2,
  );
  const candidateGateNodes = context.currentLayerNodes.slice(
    unitSize * 2,
    unitSize * 3,
  );
  return [updateGateNodes, resetGateNodes, candidateGateNodes];
}

/** Collect flattened parameter vectors for one gate node block. */
function collectRecurrentGateBlockParameters(
  context: RecurrentGateBlockCollectionContext,
): RecurrentGateParameterCollectionResult {
  const gateRows = context.gateNodes.map((gateNode, rowIndex) =>
    collectRecurrentGateRow({
      previousLayerNodes: context.previousLayerNodes,
      targetNodeInternal: asNodeInternals(gateNode),
      rowIndex,
      unitSize: context.unitSize,
      useDiagonalSelfWeights: context.useDiagonalSelfWeights,
    }),
  );
  return foldRecurrentGateRows(gateRows);
}

/** Collect one recurrent gate row payload (inputs, recurrent slice, and bias). */
function collectRecurrentGateRow(
  context: RecurrentGateRowCollectionContext,
): RecurrentGateRow {
  const inputWeights = context.previousLayerNodes.map((sourceNode) =>
    resolveIncomingWeight(context.targetNodeInternal, sourceNode),
  );
  const recurrentWeights = Array.from(
    { length: context.unitSize },
    (_unused, columnIndex) => resolveRecurrentRowWeight(context, columnIndex),
  );
  return {
    inputWeights,
    recurrentWeights,
    bias: context.targetNodeInternal.bias,
  };
}

/** Resolve one recurrent row value at the requested column. */
function resolveRecurrentRowWeight(
  context: RecurrentGateRowCollectionContext,
  columnIndex: number,
): number {
  if (!context.useDiagonalSelfWeights) {
    return 0;
  }
  if (columnIndex !== context.rowIndex) {
    return 0;
  }
  return resolveSelfConnectionWeight(context.targetNodeInternal);
}

/** Fold recurrent gate rows into flattened ONNX initializer vectors. */
function foldRecurrentGateRows(
  gateRows: RecurrentGateRow[],
): RecurrentGateParameterCollectionResult {
  return gateRows.reduce<RecurrentGateParameterCollectionResult>(
    (result, gateRow) => ({
      inputWeights: [...result.inputWeights, ...gateRow.inputWeights],
      recurrentWeights: [
        ...result.recurrentWeights,
        ...gateRow.recurrentWeights,
      ],
      biases: [...result.biases, gateRow.bias],
    }),
    { inputWeights: [], recurrentWeights: [], biases: [] },
  );
}

/** Fold gate blocks into a single fused parameter payload. */
function foldRecurrentGateBlocks(
  gateParameterBlocks: RecurrentGateParameterCollectionResult[],
): RecurrentGateParameterCollectionResult {
  return gateParameterBlocks.reduce<RecurrentGateParameterCollectionResult>(
    (result, gateBlock) => ({
      inputWeights: [...result.inputWeights, ...gateBlock.inputWeights],
      recurrentWeights: [
        ...result.recurrentWeights,
        ...gateBlock.recurrentWeights,
      ],
      biases: [...result.biases, ...gateBlock.biases],
    }),
    { inputWeights: [], recurrentWeights: [], biases: [] },
  );
}

/** Build fused recurrent initializer names for the current layer. */
function buildFusedRecurrentInitializerNames(
  operatorType: 'LSTM' | 'GRU',
  layerIndex: number,
): FusedRecurrentInitializerNames {
  const layerOffset = layerIndex - 1;
  return {
    weightName: `${operatorType}_W${layerOffset}`,
    recurrentWeightName: `${operatorType}_R${layerOffset}`,
    biasName: `${operatorType}_B${layerOffset}`,
  };
}

/** Build fused recurrent graph names for node and output. */
function buildFusedRecurrentGraphNames(
  nodePrefix: string,
  outputSuffix: string,
  layerIndex: number,
): FusedRecurrentGraphNames {
  return {
    nodeName: `${nodePrefix}_l${layerIndex}`,
    outputName: `${GENERATED_LAYER_OUTPUT_PREFIX}_${layerIndex}_${outputSuffix}`,
  };
}

/** Append fused recurrent initializer tensors to the ONNX graph. */
function appendFusedRecurrentInitializers(
  model: OnnxModel,
  initializerNames: FusedRecurrentInitializerNames,
  parameters: RecurrentGateParameterCollectionResult,
  gateCount: number,
  unitSize: number,
  previousSize: number,
): void {
  model.graph.initializer.push({
    name: initializerNames.weightName,
    data_type: ONNX_FLOAT_DATA_TYPE,
    dims: [gateCount * unitSize, previousSize],
    float_data: parameters.inputWeights,
  });
  model.graph.initializer.push({
    name: initializerNames.recurrentWeightName,
    data_type: ONNX_FLOAT_DATA_TYPE,
    dims: [gateCount * unitSize, unitSize],
    float_data: parameters.recurrentWeights,
  });
  model.graph.initializer.push({
    name: initializerNames.biasName,
    data_type: ONNX_FLOAT_DATA_TYPE,
    dims: [gateCount * unitSize],
    float_data: parameters.biases,
  });
}

/** Append fused recurrent operator node to the ONNX graph. */
function appendFusedRecurrentNode(
  graph: OnnxGraph,
  operatorType: 'LSTM' | 'GRU',
  previousOutputName: string,
  initializerNames: FusedRecurrentInitializerNames,
  graphNames: FusedRecurrentGraphNames,
  unitSize: number,
): void {
  graph.node.push({
    op_type: operatorType,
    input: [
      previousOutputName,
      initializerNames.weightName,
      initializerNames.recurrentWeightName,
      initializerNames.biasName,
    ],
    output: [graphNames.outputName],
    name: graphNames.nodeName,
    attributes: [
      {
        name: ONNX_ATTRIBUTE_HIDDEN_SIZE_KEY,
        type: ONNX_ATTRIBUTE_TYPE_INT,
        i: unitSize,
      },
      {
        name: ONNX_ATTRIBUTE_LAYOUT_KEY,
        type: ONNX_ATTRIBUTE_TYPE_INT,
        i: ONNX_LAYOUT_DEFAULT,
      },
    ],
  });
}

/** Resolve previous output naming semantics for GRU heuristic emission. */
function resolveGruPreviousOutputName(layerIndex: number): string {
  return layerIndex === 1
    ? GRAPH_INPUT_NAME
    : `${GENERATED_LAYER_OUTPUT_PREFIX}_${layerIndex - 1}`;
}

/** Append recurrent single-step metadata when recurrent layers exist. */
function appendRecurrentSingleStepMetadata(
  model: OnnxModel,
  recurrentLayerIndices: number[],
): void {
  if (!recurrentLayerIndices.length) {
    return;
  }
  appendMetadataProperty(
    model,
    buildMetadataProperty(
      METADATA_KEY_RECURRENT_SINGLE_STEP,
      recurrentLayerIndices,
    ),
  );
}

/** Determine whether Conv2D sharing validation is enabled and configured. */
function shouldValidateConvSharing(options: OnnxExportOptions): boolean {
  return Boolean(
    options.validateConvSharing &&
    options.conv2dMappings &&
    options.conv2dMappings.length,
  );
}

/**
 * Determine whether one Conv mapping behaves like a shared kernel layer.
 *
 * @param layers Layered network nodes.
 * @param convSpec Conv mapping to evaluate.
 * @returns True when representative kernels stay consistent across outputs.
 */
export function isConvMappingWeightShared(
  layers: NeatapticNode[][],
  convSpec: ConvLayerPairContext['convSpec'],
  options?: OnnxExportOptions,
): boolean {
  const layerPair = resolveConvLayerPairContext(
    layers,
    convSpec.layerIndex,
    convSpec,
  );
  if (!layerPair) {
    return false;
  }
  return isConvLayerPairConsistent(layerPair, options);
}

/** Validate Conv2D sharing across all declared Conv mappings. */
function validateConvSharingAcrossMappings(
  context: ConvSharingValidationContext,
): ConvSharingValidationResult {
  const validationResult: ConvSharingValidationResult = {
    verifiedLayers: [],
    mismatchedLayers: [],
  };

  for (const convSpec of context.mappings) {
    const layerPair = resolveConvLayerPairContext(
      context.layers,
      convSpec.layerIndex,
      convSpec,
    );
    if (!layerPair) {
      continue;
    }
    const isConsistent = isConvLayerPairConsistent(layerPair);
    appendConvLayerValidationResult(
      validationResult,
      convSpec.layerIndex,
      isConsistent,
    );
  }

  return validationResult;
}

/** Resolve one Conv mapping layer pair or return undefined for invalid layout. */
function resolveConvLayerPairContext(
  layers: NeatapticNode[][],
  layerIndex: number,
  convSpec: ConvLayerPairContext['convSpec'],
): ConvLayerPairContext | undefined {
  const previousLayerNodes = layers[layerIndex - 1];
  const currentLayerNodes = layers[layerIndex];
  if (!previousLayerNodes || !currentLayerNodes) {
    return undefined;
  }
  return { convSpec, previousLayerNodes, currentLayerNodes };
}

/** Validate one Conv layer pair against representative kernel sharing. */
function isConvLayerPairConsistent(
  context: ConvLayerPairContext,
  options?: OnnxExportOptions,
): boolean {
  const sourceLayout = resolveConvSourceLayout(context, options);
  const representativeKernels = collectRepresentativeKernels(
    context,
    sourceLayout,
  );
  const outputCoordinates = collectConvOutputCoordinates(context.convSpec);
  return (
    outputCoordinates.every((outputCoordinate) =>
      isOutputCoordinateConsistent(
        context,
        outputCoordinate,
        representativeKernels,
        CONV_WEIGHT_SHARING_TOLERANCE,
        sourceLayout,
      ),
    ) && hasNoIgnoredSourceWeights(context, sourceLayout)
  );
}

type ResolvedConvSourceLayout = {
  sourceHeight: number;
  sourceWidth: number;
  channelStride: number;
};

function resolveConvSourceLayout(
  context: ConvLayerPairContext,
  options?: OnnxExportOptions,
): ResolvedConvSourceLayout {
  const defaultSourceLayout = {
    sourceHeight: context.convSpec.inHeight,
    sourceWidth: context.convSpec.inWidth,
    channelStride: context.convSpec.inHeight * context.convSpec.inWidth,
  };

  const upstreamPoolingSpec = options?.pool2dMappings?.find(
    (poolingSpec) => poolingSpec.afterLayerIndex === context.convSpec.layerIndex - 1,
  );
  const upstreamConvSpec = options?.conv2dMappings?.find(
    (mapping) => mapping.layerIndex === context.convSpec.layerIndex - 1,
  );
  if (!upstreamPoolingSpec || !upstreamConvSpec) {
    return defaultSourceLayout;
  }

  const derivedInputHeight = calculateSpatialOutputSize(
    upstreamConvSpec.outHeight,
    upstreamPoolingSpec.kernelHeight,
    upstreamPoolingSpec.strideHeight,
    upstreamPoolingSpec.padTop ?? 0,
    upstreamPoolingSpec.padBottom ?? 0,
  );
  const derivedInputWidth = calculateSpatialOutputSize(
    upstreamConvSpec.outWidth,
    upstreamPoolingSpec.kernelWidth,
    upstreamPoolingSpec.strideWidth,
    upstreamPoolingSpec.padLeft ?? 0,
    upstreamPoolingSpec.padRight ?? 0,
  );
  const matchesDerivedPooledShape =
    derivedInputHeight === context.convSpec.inHeight &&
    derivedInputWidth === context.convSpec.inWidth &&
    upstreamConvSpec.outChannels === context.convSpec.inChannels;
  if (!matchesDerivedPooledShape) {
    return defaultSourceLayout;
  }

  return {
    sourceHeight: upstreamConvSpec.outHeight,
    sourceWidth: context.convSpec.inWidth,
    channelStride: upstreamConvSpec.outHeight * upstreamConvSpec.outWidth,
  };
}

function calculateSpatialOutputSize(
  inputSize: number,
  kernelSize: number,
  strideSize: number,
  leadingPadding: number,
  trailingPadding: number,
): number {
  if (inputSize <= 0 || kernelSize <= 0 || strideSize <= 0) {
    return 0;
  }

  return (
    Math.floor(
      (inputSize + leadingPadding + trailingPadding - kernelSize) /
        strideSize,
    ) + 1
  );
}

/**
 * Ensure weights outside the Conv-addressable source slice remain zero.
 *
 * @param context Conv layer pair context.
 * @returns True when ignored dense source nodes carry no extra weight.
 */
function hasNoIgnoredSourceWeights(
  context: ConvLayerPairContext,
  sourceLayout: ResolvedConvSourceLayout,
): boolean {
  const addressedSourceIndices = collectAddressedSourceIndices(
    context.convSpec,
    sourceLayout,
  );
  if (context.previousLayerNodes.length <= addressedSourceIndices.size) {
    return true;
  }

  const ignoredSourceNodes = context.previousLayerNodes.filter(
    (_sourceNode, sourceIndex) => !addressedSourceIndices.has(sourceIndex),
  );
  return context.currentLayerNodes.every((currentNode) => {
    const currentNodeInternal = asNodeInternals(currentNode);
    return ignoredSourceNodes.every((ignoredSourceNode) =>
      areWeightsWithinTolerance({
        leftWeight: resolveIncomingWeight(currentNodeInternal, ignoredSourceNode),
        rightWeight: 0,
        tolerance: CONV_WEIGHT_SHARING_TOLERANCE,
      }),
    );
  });
}

function collectAddressedSourceIndices(
  convSpec: ConvLayerPairContext['convSpec'],
  sourceLayout: ResolvedConvSourceLayout,
): Set<number> {
  return new Set(
    Array.from({ length: convSpec.inChannels }, (_unusedChannel, inChannelIndex) =>
      Array.from({ length: convSpec.inHeight }, (_unusedRow, inputRow) =>
        Array.from({ length: convSpec.inWidth }, (_unusedColumn, inputColumn) =>
          buildConvSourceIndex(
            sourceLayout,
            inChannelIndex,
            inputRow,
            inputColumn,
          ),
        ),
      ).flat(),
    ).flat(),
  );
}

/** Append one Conv-layer validation outcome and optional warning. */
function appendConvLayerValidationResult(
  result: ConvSharingValidationResult,
  layerIndex: number,
  isConsistent: boolean,
): void {
  if (isConsistent) {
    result.verifiedLayers.push(layerIndex);
    return;
  }
  result.mismatchedLayers.push(layerIndex);
  console.warn(
    `Conv2D weight sharing mismatch detected in layer ${layerIndex}`,
  );
}

/** Append Conv-sharing validation metadata arrays. */
function appendConvSharingMetadata(
  model: OnnxModel,
  result: ConvSharingValidationResult,
): void {
  if (result.verifiedLayers.length) {
    appendMetadataProperty(
      model,
      buildMetadataProperty(
        METADATA_KEY_CONV2D_SHARING_VERIFIED,
        result.verifiedLayers,
      ),
    );
  }
  if (result.mismatchedLayers.length) {
    appendMetadataProperty(
      model,
      buildMetadataProperty(
        METADATA_KEY_CONV2D_SHARING_MISMATCH,
        result.mismatchedLayers,
      ),
    );
  }
}

/** Collect representative kernels for each output channel. */
function collectRepresentativeKernels(
  context: ConvLayerPairContext,
  sourceLayout: ResolvedConvSourceLayout,
): number[][] {
  const outChannelIndices = Array.from(
    { length: context.convSpec.outChannels },
    (_unused, outChannelIndex) => outChannelIndex,
  );
  return outChannelIndices.map((outChannelIndex) =>
    collectRepresentativeKernelForChannel({
      convSpec: context.convSpec,
      previousLayerNodes: context.previousLayerNodes,
      currentLayerNodes: context.currentLayerNodes,
      outChannelIndex,
      sourceLayout,
    }),
  );
}

/** Collect one representative kernel by reading the first output position for a channel. */
function collectRepresentativeKernelForChannel(
  context: ConvRepresentativeKernelContext & {
    sourceLayout: ResolvedConvSourceLayout;
  },
): number[] {
  const representativeNeuronIndex =
    context.outChannelIndex *
    (context.convSpec.outHeight * context.convSpec.outWidth);
  const representativeNeuron =
    context.currentLayerNodes[representativeNeuronIndex];
  if (!representativeNeuron) {
    return [];
  }
  const representativeInternal = asNodeInternals(representativeNeuron);
  const kernelCoordinates = collectConvKernelCoordinates(context.convSpec);
  return kernelCoordinates.map((kernelCoordinate) =>
    collectRepresentativeKernelWeight(
      context.convSpec,
      context.previousLayerNodes,
      representativeInternal,
      kernelCoordinate,
      context.sourceLayout,
    ),
  );
}

/** Collect output coordinates for full Conv traversal. */
function collectConvOutputCoordinates(
  convSpec: ConvLayerPairContext['convSpec'],
): ConvOutputCoordinate[] {
  const outChannelIndices = Array.from(
    { length: convSpec.outChannels },
    (_unused, outChannelIndex) => outChannelIndex,
  );
  const outRowIndices = Array.from(
    { length: convSpec.outHeight },
    (_unused, outRowIndex) => outRowIndex,
  );
  const outColumnIndices = Array.from(
    { length: convSpec.outWidth },
    (_unused, outColumnIndex) => outColumnIndex,
  );
  return outChannelIndices.flatMap((outChannelIndex) =>
    outRowIndices.flatMap((outRowIndex) =>
      outColumnIndices.map((outColumnIndex) => ({
        outChannelIndex,
        outRowIndex,
        outColumnIndex,
      })),
    ),
  );
}

/** Collect kernel coordinates for one Conv kernel traversal. */
function collectConvKernelCoordinates(
  convSpec: ConvLayerPairContext['convSpec'],
): OnnxConvKernelCoordinate[] {
  const inChannelIndices = Array.from(
    { length: convSpec.inChannels },
    (_unused, inChannelIndex) => inChannelIndex,
  );
  const kernelRowIndices = Array.from(
    { length: convSpec.kernelHeight },
    (_unused, kernelRowIndex) => kernelRowIndex,
  );
  const kernelColumnIndices = Array.from(
    { length: convSpec.kernelWidth },
    (_unused, kernelColumnIndex) => kernelColumnIndex,
  );
  return inChannelIndices.flatMap((inChannelIndex) =>
    kernelRowIndices.flatMap((kernelRowIndex) =>
      kernelColumnIndices.map((kernelColumnIndex) => ({
        inChannelIndex,
        kernelRowIndex,
        kernelColumnIndex,
      })),
    ),
  );
}

/** Validate one output coordinate against channel representative kernel weights. */
function isOutputCoordinateConsistent(
  context: ConvLayerPairContext,
  outputCoordinate: ConvOutputCoordinate,
  representativeKernels: number[][],
  tolerance: number,
  sourceLayout: ResolvedConvSourceLayout,
): boolean {
  const neuronInternal = resolveNeuronInternalAtOutputCoordinate(
    context,
    outputCoordinate,
  );
  if (!neuronInternal) {
    return true;
  }
  const representativeKernelWeights =
    representativeKernels[outputCoordinate.outChannelIndex];
  const kernelCoordinates = collectConvKernelCoordinates(context.convSpec);
  return kernelCoordinates.every((kernelCoordinate, kernelPointer) =>
    isKernelCoordinateConsistent({
      convSpec: context.convSpec,
      previousLayerNodes: context.previousLayerNodes,
      neuronInternal,
      outputCoordinate,
      kernelCoordinate,
      representativeKernelWeights,
      kernelPointer,
      tolerance,
      sourceLayout,
    }),
  );
}

/** Resolve runtime internals for output coordinate neuron, if present. */
function resolveNeuronInternalAtOutputCoordinate(
  context: ConvLayerPairContext,
  outputCoordinate: ConvOutputCoordinate,
): NodeInternals | undefined {
  const neuronIndex =
    outputCoordinate.outChannelIndex *
      (context.convSpec.outHeight * context.convSpec.outWidth) +
    outputCoordinate.outRowIndex * context.convSpec.outWidth +
    outputCoordinate.outColumnIndex;
  const neuron = context.currentLayerNodes[neuronIndex];
  return neuron ? asNodeInternals(neuron) : undefined;
}

/** Validate one kernel coordinate against its representative channel value. */
function isKernelCoordinateConsistent(
  context: ConvKernelConsistencyContext & {
    sourceLayout: ResolvedConvSourceLayout;
  },
): boolean {
  const inputPosition = resolveInputPosition(context);
  if (
    !isInputPositionInsideBounds(
      context.convSpec,
      inputPosition.inputRow,
      inputPosition.inputColumn,
    )
  ) {
    return true;
  }

  const sourceNode = resolveSourceNodeAtInputPosition(
    context.convSpec,
    context.previousLayerNodes,
    context.kernelCoordinate.inChannelIndex,
    inputPosition.inputRow,
    inputPosition.inputColumn,
      context.sourceLayout,
  );
  const currentWeight = sourceNode
    ? resolveIncomingWeight(context.neuronInternal, sourceNode)
    : 0;
  const representativeWeight =
    context.representativeKernelWeights[context.kernelPointer] ?? 0;
  return areWeightsWithinTolerance({
    leftWeight: currentWeight,
    rightWeight: representativeWeight,
    tolerance: context.tolerance,
  });
}

/** Resolve input row/column projected by output and kernel coordinates. */
function resolveInputPosition(context: ConvKernelConsistencyContext): {
  inputRow: number;
  inputColumn: number;
} {
  const inputRowBase =
    context.outputCoordinate.outRowIndex * context.convSpec.strideHeight -
    (context.convSpec.padTop || 0);
  const inputColumnBase =
    context.outputCoordinate.outColumnIndex * context.convSpec.strideWidth -
    (context.convSpec.padLeft || 0);
  return {
    inputRow: inputRowBase + context.kernelCoordinate.kernelRowIndex,
    inputColumn: inputColumnBase + context.kernelCoordinate.kernelColumnIndex,
  };
}

/** Check whether input row/column falls inside Conv input bounds. */
function isInputPositionInsideBounds(
  convSpec: ConvLayerPairContext['convSpec'],
  inputRow: number,
  inputColumn: number,
): boolean {
  return (
    inputRow >= 0 &&
    inputRow < convSpec.inHeight &&
    inputColumn >= 0 &&
    inputColumn < convSpec.inWidth
  );
}

/** Resolve source node by Conv input position coordinates. */
function resolveSourceNodeAtInputPosition(
  convSpec: ConvLayerPairContext['convSpec'],
  previousLayerNodes: NeatapticNode[],
  inChannelIndex: number,
  inputRow: number,
  inputColumn: number,
  sourceLayout: ResolvedConvSourceLayout,
): NeatapticNode | undefined {
  const inputFeatureIndex = buildConvSourceIndex(
    sourceLayout,
    inChannelIndex,
    inputRow,
    inputColumn,
  );
  return previousLayerNodes[inputFeatureIndex];
}

/** Collect representative kernel value using top-left receptive field indexing. */
function collectRepresentativeKernelWeight(
  convSpec: ConvLayerPairContext['convSpec'],
  previousLayerNodes: NeatapticNode[],
  representativeInternal: NodeInternals,
  kernelCoordinate: OnnxConvKernelCoordinate,
  sourceLayout: ResolvedConvSourceLayout,
): number {
  const inputFeatureIndex = buildConvSourceIndex(
    sourceLayout,
    kernelCoordinate.inChannelIndex,
    kernelCoordinate.kernelRowIndex,
    kernelCoordinate.kernelColumnIndex,
  );
  const sourceNode = previousLayerNodes[inputFeatureIndex];
  if (!sourceNode) {
    return 0;
  }
  return resolveIncomingWeight(representativeInternal, sourceNode);
}

function buildConvSourceIndex(
  sourceLayout: ResolvedConvSourceLayout,
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

/** Compare two scalar weights using configured tolerance. */
function areWeightsWithinTolerance(
  context: WeightToleranceComparisonContext,
): boolean {
  return (
    Math.abs(context.leftWeight - context.rightWeight) <= context.tolerance
  );
}

/** Resolve runtime node internals in one typed helper. */
function asNodeInternals(node: NeatapticNode): NodeInternals {
  return node as unknown as NodeInternals;
}

/** Resolve incoming connection weight from a specific source node. */
function resolveIncomingWeight(
  targetNodeInternal: NodeInternals,
  sourceNode: NeatapticNode,
): number {
  const connection = targetNodeInternal.connections.in.find(
    (candidate) => candidate.from === sourceNode,
  );
  return connection ? connection.weight : 0;
}

/** Resolve self-connection weight for diagonal recurrent matrix entries. */
function resolveSelfConnectionWeight(
  targetNodeInternal: NodeInternals,
): number {
  const selfConnection = targetNodeInternal.connections.self[0];
  return selfConnection ? selfConnection.weight : 0;
}

/** Build a metadata key/value property with JSON string serialization. */
function buildMetadataProperty(
  key: string,
  value: unknown,
): OnnxMetadataProperty {
  return {
    key,
    value: JSON.stringify(value),
  };
}

/** Append metadata property to model metadata_props list. */
function appendMetadataProperty(
  model: OnnxModel,
  metadataProperty: OnnxMetadataProperty,
): void {
  const metadataProperties = ensureMetadataProps(model);
  metadataProperties.push(metadataProperty);
}

/** Ensure metadata_props array exists and return it. */
function ensureMetadataProps(model: OnnxModel): OnnxMetadataProperty[] {
  model.metadata_props = model.metadata_props || [];
  return model.metadata_props;
}
