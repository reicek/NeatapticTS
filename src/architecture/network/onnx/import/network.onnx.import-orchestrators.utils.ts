import Connection from '../../../connection';
import type Network from '../../network';
import type NeatapticNode from '../../../node';
import { deriveHiddenLayerSizes } from './network.onnx.import-weights.utils';
import { reconstructFusedRecurrentLayers } from './network.onnx.import-fused-recurrent.utils';
import type {
  Conv2DMapping,
  OnnxMetadataProperty,
  OnnxModel,
  OnnxTensor,
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';
import type { NodeInternals } from '../network.onnx.utils.types';
import type {
  NetworkWithOnnxImportAdvancedGraph,
  OnnxImportAttentionBlock,
  OnnxImportAdvancedGraphCrossLayerConnection,
  OnnxImportAdvancedGraphMetadata,
  OnnxImportResidualAdd,
  OnnxImportSharedInitializerAlias,
  NetworkWithOnnxImportPooling,
  OnnxImportFlattenConsistencyAudit,
  OnnxImportArchitectureContext,
  OnnxImportArchitectureResult,
  OnnxImportDimensionRecord,
  OnnxImportHiddenLayerSpan,
  OnnxImportLayerConnectionContext,
  OnnxImportPoolingMetadata,
  OnnxImportPoolingVirtualShape,
  OnnxImportRecurrentRestorationContext,
  OnnxImportSelfConnectionUpsertContext,
} from './network.onnx.import-orchestrators.types';
import { readOnnxTensorFloatData } from '../schema/network.onnx.schema.tensor-data.utils';

/**
 * Re-export fused recurrent layer reconstruction so import orchestration can expose one stable entrypoint for advanced recurrent restoration workflows.
 * Keeping the re-export documented here helps callers discover the fused path alongside the surrounding ONNX import orchestration utilities.
 */
export { reconstructFusedRecurrentLayers };

const METADATA_KEY_RECURRENT_SINGLE_STEP = 'recurrent_single_step';
const METADATA_KEY_POOL2D_LAYERS = 'pool2d_layers';
const METADATA_KEY_POOL2D_SPECS = 'pool2d_specs';
const METADATA_KEY_CONV2D_SPECS = 'conv2d_specs';
const METADATA_KEY_CONV2D_INFERRED_SPECS = 'conv2d_inferred_specs';
const METADATA_KEY_FLATTEN_LAYERS = 'flatten_layers';
const METADATA_KEY_LAYER_SIZES = 'layer_sizes';
const METADATA_KEY_ADVANCED_GRAPH_CROSS_LAYER_CONNECTIONS =
  'advanced_graph_cross_layer_connections';
const METADATA_KEY_ADVANCED_GRAPH_RESIDUAL_ADDS =
  'advanced_graph_residual_adds';
const METADATA_KEY_SHARED_INITIALIZER_ALIASES = 'shared_initializer_aliases';
const METADATA_KEY_ADVANCED_GRAPH_ATTENTION_BLOCKS =
  'advanced_graph_attention_blocks';
const NODE_TYPE_INPUT = 'input';
const NODE_TYPE_HIDDEN = 'hidden';
const NODE_TYPE_OUTPUT = 'output';
const ONNX_DIM_VALUE_FIELD = 'dim_value';
const RECURRENT_TENSOR_PREFIX = 'R';
const LSTM_RECURRENT_TENSOR_PREFIX = 'LSTM_R';
const GRU_RECURRENT_TENSOR_PREFIX = 'GRU_R';
const RECURRENT_TENSOR_NAME_PATTERN = /^(?:R|LSTM_R|GRU_R)(\d+)$/;
const FUSED_RECURRENT_GATE_GROUP_INDEX = 2;
const HIDDEN_LAYER_NUMBER_OFFSET = 1;
const DEFAULT_NON_MATCHING_LAYER_INDEX = 0;
const FIRST_SELF_CONNECTION_INDEX = 0;
const ZERO_LENGTH = 0;
const MINIMUM_SPATIAL_OUTPUT_SIZE = 1;
const EMPTY_POOLING_SPECS: Pool2DMapping[] = [];

/**
 * Extract input/output counts and hidden layer sizes from ONNX model.
 * This architecture probe normalizes graph terminal dimensions and initializer-derived hidden spans into one deterministic result contract used by all downstream reconstruction passes.
 *
 * @param onnx Source ONNX model.
 * @returns Parsed architecture dimensions.
 */
export function extractOnnxArchitecture(
  onnx: OnnxModel,
): OnnxImportArchitectureResult {
  // Step 1: Build a normalized read-context from the ONNX model payload.
  const architectureContext = buildArchitectureContext(onnx);

  // Step 2: Resolve input/output terminal dimension values.
  const inputCount = readLastDimensionValue(
    architectureContext.inputShapeDimensions,
  );
  const outputCount = readLastDimensionValue(
    architectureContext.outputShapeDimensions,
  );

  // Step 3: Derive hidden-layer sizes from initializers + metadata.
  const hiddenLayerSizes = deriveHiddenLayerSizes(
    architectureContext.initializers,
    architectureContext.metadata,
  );

  // Step 4: Fold parsed architecture dimensions into result payload.
  return { inputCount, outputCount, hiddenLayerSizes };
}

/**
 * Remove placeholder hidden nodes that arise from single-layer perceptron imports.
 *
 * @param network Target network.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @returns Nothing.
 */
export function pruneSingleLayerHiddenPlaceholders(
  network: Network,
  hiddenLayerSizes: number[],
): void {
  // Step 1: Exit unless this import has no hidden layers.
  if (!isSingleLayerPerceptronImport(hiddenLayerSizes)) return;

  // Step 2: Fold the retained node groups back into network state.
  network.nodes = collectPerceptronBoundaryNodes(network.nodes);
}

/**
 * Restore recurrent self-connections from recurrent metadata and R tensors.
 * Restoration uses metadata-gated span resolution and diagonal tensor extraction so imported recurrent units recover their self-feedback semantics without guessing hidden-node layout.
 *
 * @param network Target network.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @param metadata Parsed metadata properties.
 * @returns Nothing.
 */
/**
 * Contract for restoreRecurrentSelfConnections.
 */
export function restoreRecurrentSelfConnections(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadata: OnnxMetadataProperty[],
): void {
  // Step 1: Build the recurrent restoration execution context.
  const restorationContext: OnnxImportRecurrentRestorationContext = {
    hiddenLayerSizes,
    metadata,
    onnx,
  };

  // Step 2: Resolve recurrent-target hidden-layer spans.
  const recurrentLayerSpans = collectRecurrentLayerSpans(restorationContext);

  // Step 3: Collect hidden nodes once for deterministic span slicing.
  const hiddenNodes = collectNodesByType(network.nodes, NODE_TYPE_HIDDEN);

  // Step 4: Apply recurrent diagonal self-weights per resolved span.
  recurrentLayerSpans.forEach((span) => {
    applyLayerSelfConnections({
      onnx,
      hiddenNodes,
      span,
    });
  });
}

/**
 * Attach optional pooling metadata from ONNX model to network instance.
 * The importer keeps pooling metadata as additive diagnostics state so later runtime or visualization tooling can reason about spatial stages without modifying core graph wiring.
 *
 * @param network Target network.
 * @param metadata ONNX metadata.
 * @returns Nothing.
 */
export function attachOnnxPoolingMetadata(
  network: Network,
  metadata: OnnxMetadataProperty[],
): void {
  // Step 1: Parse optional pooling metadata payload from model metadata.
  const poolingMetadata = parsePoolingMetadata(network, metadata);

  // Step 2: Exit for absent or invalid pooling metadata payloads.
  if (!poolingMetadata) return;

  // Step 3: Attach parsed pooling metadata to the imported network.
  attachParsedPoolingMetadata(network, poolingMetadata);
}

/**
 * Attach optional advanced-graph audit metadata from ONNX model metadata.
 *
 * Phase 5 starts with honest fallback: import keeps rebuilding the layered
 * baseline, but it can still preserve the exact cross-layer feed-forward edges
 * the exporter detected so later residual, concat, and attention passes have a
 * deterministic seam to reuse.
 *
 * @param network Target network.
 * @param metadata ONNX metadata.
 * @returns Nothing.
 */
export function attachOnnxAdvancedGraphMetadata(
  network: Network,
  metadata: OnnxMetadataProperty[],
  onnx?: OnnxModel,
): void {
  // Step 1: Parse optional advanced-graph audit metadata.
  const advancedGraphMetadata = parseAdvancedGraphMetadata(metadata, onnx);

  // Step 2: Exit when the metadata is absent or invalid.
  if (!advancedGraphMetadata) return;

  // Step 3: Attach the parsed audit payload without changing runtime wiring.
  const networkWithOnnxAdvancedGraph =
    network as NetworkWithOnnxImportAdvancedGraph;
  networkWithOnnxAdvancedGraph._onnxAdvancedGraph = advancedGraphMetadata;
}

/**
 * Parse recurrent layer indices metadata.
 *
 * @param rawMetadataValue Raw metadata JSON string.
 * @returns Normalized recurrent layer indices.
 */
function parseRecurrentLayerIndices(rawMetadataValue: string): number[] {
  try {
    const parsedMetadataValue = JSON.parse(rawMetadataValue);
    return normalizeRecurrentLayerIndices(parsedMetadataValue);
  } catch {
    return [DEFAULT_NON_MATCHING_LAYER_INDEX];
  }
}

/**
 * Apply one hidden layer diagonal recurrent self-weights.
 *
 * @param layerConnectionContext Layer connection context.
 * @returns Nothing.
 */
function applyLayerSelfConnections(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): void {
  // Step 1: Resolve the recurrent tensor for the target hidden-layer span.
  const genericRecurrentInitializer = findRecurrentInitializer(
    layerConnectionContext,
  );

  // Step 2: Slice the target hidden-layer node segment for the span.
  const layerHiddenNodes = sliceLayerHiddenNodes(layerConnectionContext);

  if (genericRecurrentInitializer) {
    // Step 3: Build diagonal recurrent self-weights for this layer.
    const recurrentDiagonalWeights = collectDiagonalRecurrentWeights(
      readOnnxTensorFloatData(genericRecurrentInitializer),
      layerConnectionContext.span.hiddenLayerSize,
    );

    // Step 4: Upsert one self-connection per hidden unit.
    layerHiddenNodes.forEach((hiddenNode, hiddenUnitIndex) => {
      upsertSelfConnection({
        node: hiddenNode,
        recurrentWeight: recurrentDiagonalWeights[hiddenUnitIndex],
      });
    });
    return;
  }

  const fusedRecurrentInitializer = findFusedRecurrentInitializer(
    layerConnectionContext,
  );
  if (!fusedRecurrentInitializer) return;

  // Step 3: Restore the recurrent gate slice from the fused-family tensor.
  const fusedUnitSize = fusedRecurrentInitializer.dims.at(-1);
  if (!fusedUnitSize || fusedUnitSize <= ZERO_LENGTH) return;

  const recurrentGateStart = fusedUnitSize * FUSED_RECURRENT_GATE_GROUP_INDEX;
  const recurrentGateNodes = layerHiddenNodes.slice(
    recurrentGateStart,
    recurrentGateStart + fusedUnitSize,
  );
  if (recurrentGateNodes.length !== fusedUnitSize) return;

  const recurrentGateWeights = collectFusedRecurrentGateWeights(
    readOnnxTensorFloatData(fusedRecurrentInitializer),
    fusedUnitSize,
  );

  recurrentGateNodes.forEach((hiddenNode, hiddenUnitIndex) => {
    upsertSelfConnection({
      node: hiddenNode,
      recurrentWeight: recurrentGateWeights[hiddenUnitIndex],
    });
  });
}

/**
 * Restore supported one-hop residual-add skip connections from ONNX metadata.
 *
 * The import side stays conservative: it only rehydrates skip edges when the
 * exporter recorded both the residual merge intent and the exact cross-layer
 * edge list, and when the residual branch tensor is still present.
 *
 * @param network Target network.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes from architecture extraction.
 * @param metadata ONNX metadata payload.
 * @returns Nothing.
 */
export function restoreResidualAddConnections(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadata: OnnxMetadataProperty[],
): void {
  const residualAdds = parseResidualAddMetadata(metadata);
  const crossLayerConnections =
    parseAdvancedGraphCrossLayerConnections(metadata);
  if (!residualAdds || !crossLayerConnections) {
    return;
  }

  const importedLayers = buildImportedLayers(network, hiddenLayerSizes);
  residualAdds.forEach((residualAdd) => {
    restoreSingleResidualAddConnections(
      network,
      onnx,
      importedLayers,
      crossLayerConnections,
      residualAdd,
    );
  });
}

/**
 * Build architecture extraction context from ONNX graph state.
 *
 * @param onnx Source ONNX model.
 * @returns Normalized architecture extraction context.
 */
function buildArchitectureContext(
  onnx: OnnxModel,
): OnnxImportArchitectureContext {
  return {
    inputShapeDimensions: onnx.graph.inputs[0].type.tensor_type.shape.dim,
    outputShapeDimensions: onnx.graph.outputs[0].type.tensor_type.shape.dim,
    initializers: onnx.graph.initializer,
    metadata: onnx.metadata_props ?? [],
  };
}

/**
 * Read the terminal ONNX shape dimension value from one shape array.
 *
 * @param dimensions ONNX shape dimensions.
 * @returns Terminal `dim_value` payload.
 */
function readLastDimensionValue(dimensions: { dim_value?: number }[]): number {
  const lastDimension = dimensions.at(-1) as OnnxImportDimensionRecord;
  return lastDimension[ONNX_DIM_VALUE_FIELD];
}

/**
 * Determine whether import shape corresponds to a single-layer perceptron.
 *
 * @param hiddenLayerSizes Hidden-layer size list.
 * @returns True when no hidden layers exist.
 */
function isSingleLayerPerceptronImport(hiddenLayerSizes: number[]): boolean {
  return hiddenLayerSizes.length === ZERO_LENGTH;
}

/**
 * Collect input and output boundary nodes for perceptron imports.
 *
 * @param nodes Full network node list.
 * @returns Input/output-only node list.
 */
function collectPerceptronBoundaryNodes(
  nodes: NeatapticNode[],
): NeatapticNode[] {
  const inputNodes = collectNodesByType(nodes, NODE_TYPE_INPUT);
  const outputNodes = collectNodesByType(nodes, NODE_TYPE_OUTPUT);
  return [...inputNodes, ...outputNodes];
}

/**
 * Collect nodes matching one runtime node-type discriminator.
 *
 * @param nodes Node list.
 * @param nodeType Runtime node type.
 * @returns Filtered node list.
 */
function collectNodesByType(
  nodes: NeatapticNode[],
  nodeType: 'input' | 'hidden' | 'output',
): NeatapticNode[] {
  return nodes.filter((nodeItem) => nodeItem.type === nodeType);
}

/**
 * Resolve recurrent-target hidden-layer spans from metadata + hidden sizes.
 *
 * @param restorationContext Recurrent restoration context.
 * @returns Hidden-layer spans requiring recurrent restoration.
 */
function collectRecurrentLayerSpans(
  restorationContext: OnnxImportRecurrentRestorationContext,
): OnnxImportHiddenLayerSpan[] {
  const recurrentLayerIndices = resolveRecurrentLayerIndices(
    restorationContext.hiddenLayerSizes,
    restorationContext.metadata,
    restorationContext.onnx,
  );
  const hiddenLayerSpans = buildHiddenLayerSpans(
    restorationContext.hiddenLayerSizes,
  );
  const recurrentLayerSet = new Set(recurrentLayerIndices);
  return hiddenLayerSpans.filter((span) =>
    recurrentLayerSet.has(span.layerNumber),
  );
}

/**
 * Resolve recurrent layer indices from ONNX metadata.
 *
 * @param metadata ONNX metadata payload.
 * @returns Parsed recurrent layer indices.
 */
function resolveRecurrentLayerIndices(
  hiddenLayerSizes: number[],
  metadata: OnnxMetadataProperty[],
  onnx: OnnxModel,
): number[] {
  const recurrentMetadata = findMetadataProperty(
    metadata,
    METADATA_KEY_RECURRENT_SINGLE_STEP,
  );
  if (!recurrentMetadata) {
    return inferRecurrentLayerIndicesFromInitializers(hiddenLayerSizes, onnx);
  }

  return parseRecurrentLayerIndices(recurrentMetadata.value);
}

/**
 * Infer recurrent layer indices from plain recurrent tensors when metadata is absent.
 *
 * @param hiddenLayerSizes Hidden-layer size list used to bound valid layer indices.
 * @param onnx Source ONNX model.
 * @returns One-based recurrent layer indices inferred from `Rk` tensors.
 */
function inferRecurrentLayerIndicesFromInitializers(
  hiddenLayerSizes: number[],
  onnx: OnnxModel,
): number[] {
  const maximumHiddenLayerNumber = hiddenLayerSizes.length;
  const inferredLayerNumbers = onnx.graph.initializer
    .map((tensor) => tensor.name.match(RECURRENT_TENSOR_NAME_PATTERN))
    .flatMap((matchResult) => {
      const rawLayerIndex = matchResult?.[1];
      if (rawLayerIndex === undefined) {
        return [];
      }

      const hiddenLayerNumber =
        Number(rawLayerIndex) + HIDDEN_LAYER_NUMBER_OFFSET;
      return [hiddenLayerNumber];
    })
    .filter(
      (hiddenLayerNumber) =>
        hiddenLayerNumber >= HIDDEN_LAYER_NUMBER_OFFSET &&
        hiddenLayerNumber <= maximumHiddenLayerNumber,
    );

  return Array.from(new Set(inferredLayerNumbers)).toSorted(
    (leftLayerNumber, rightLayerNumber) => leftLayerNumber - rightLayerNumber,
  );
}

/**
 * Find one ONNX metadata property by key.
 *
 * @param metadata ONNX metadata array.
 * @param metadataKey Metadata key.
 * @returns Matching metadata property when present.
 */
function findMetadataProperty(
  metadata: OnnxMetadataProperty[],
  metadataKey: string,
): OnnxMetadataProperty | undefined {
  return metadata.find((property) => property.key === metadataKey);
}

/**
 * Build hidden-layer spans with one-based layer numbering and global offsets.
 *
 * @param hiddenLayerSizes Hidden-layer size list.
 * @returns Hidden-layer span payload list.
 */
function buildHiddenLayerSpans(
  hiddenLayerSizes: number[],
): OnnxImportHiddenLayerSpan[] {
  return hiddenLayerSizes.reduce(
    (state, hiddenLayerSize, hiddenLayerIndex) => {
      const layerNumber = hiddenLayerIndex + HIDDEN_LAYER_NUMBER_OFFSET;
      const span: OnnxImportHiddenLayerSpan = {
        layerNumber,
        hiddenLayerSize,
        hiddenStart: state.hiddenStart,
      };
      return {
        spans: [...state.spans, span],
        hiddenStart: state.hiddenStart + hiddenLayerSize,
      };
    },
    {
      spans: [] as OnnxImportHiddenLayerSpan[],
      hiddenStart: 0,
    },
  ).spans;
}

/**
 * Normalize recurrent layer indices parsed from metadata JSON.
 *
 * @param parsedMetadataValue Parsed metadata JSON value.
 * @returns Recurrent layer indices.
 */
function normalizeRecurrentLayerIndices(
  parsedMetadataValue:
    number[] | number | string | boolean | null | Record<string, number>,
): number[] {
  if (Array.isArray(parsedMetadataValue)) {
    return parsedMetadataValue as number[];
  }
  return [DEFAULT_NON_MATCHING_LAYER_INDEX];
}

/**
 * Resolve recurrent initializer tensor for one hidden-layer span.
 *
 * @param layerConnectionContext Layer connection context.
 * @returns Recurrent initializer tensor when available.
 */
function findRecurrentInitializer(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): OnnxTensor | undefined {
  const recurrentTensorName =
    RECURRENT_TENSOR_PREFIX +
    String(
      layerConnectionContext.span.layerNumber - HIDDEN_LAYER_NUMBER_OFFSET,
    );
  return layerConnectionContext.onnx.graph.initializer.find(
    (tensor) => tensor.name === recurrentTensorName,
  );
}

/**
 * Resolve a fused recurrent tensor for one hidden-layer span when generic `Rk` is absent.
 *
 * @param layerConnectionContext Layer connection context.
 * @returns Matching fused recurrent tensor when present.
 */
function findFusedRecurrentInitializer(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): OnnxTensor | undefined {
  const hiddenLayerIndex =
    layerConnectionContext.span.layerNumber - HIDDEN_LAYER_NUMBER_OFFSET;
  const recurrentTensorNames = [
    `${LSTM_RECURRENT_TENSOR_PREFIX}${hiddenLayerIndex}`,
    `${GRU_RECURRENT_TENSOR_PREFIX}${hiddenLayerIndex}`,
  ];

  return layerConnectionContext.onnx.graph.initializer.find((tensor) =>
    recurrentTensorNames.includes(tensor.name),
  );
}

/**
 * Slice hidden nodes for one hidden-layer span.
 *
 * @param layerConnectionContext Layer connection context.
 * @returns Hidden nodes belonging to the span.
 */
function sliceLayerHiddenNodes(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): NeatapticNode[] {
  const { hiddenStart, hiddenLayerSize } = layerConnectionContext.span;
  return layerConnectionContext.hiddenNodes.slice(
    hiddenStart,
    hiddenStart + hiddenLayerSize,
  );
}

/**
 * Collect diagonal recurrent weights from flattened layer tensor data.
 *
 * @param recurrentTensorWeights Flattened recurrent tensor weights.
 * @param hiddenLayerSize Hidden-layer width.
 * @returns Diagonal recurrent self-weights.
 */
function collectDiagonalRecurrentWeights(
  recurrentTensorWeights: number[],
  hiddenLayerSize: number,
): number[] {
  return Array.from({ length: hiddenLayerSize }, (_unused, hiddenUnitIndex) => {
    const recurrentIndex = hiddenUnitIndex * hiddenLayerSize + hiddenUnitIndex;
    return recurrentTensorWeights[recurrentIndex];
  });
}

/**
 * Collect diagonal recurrent weights from the recurrent gate block inside a fused tensor.
 *
 * @param recurrentTensorWeights Flattened fused recurrent tensor weights.
 * @param unitSize Fused recurrent unit size.
 * @returns Diagonal recurrent self-weights for the recurrent gate slice.
 */
function collectFusedRecurrentGateWeights(
  recurrentTensorWeights: number[],
  unitSize: number,
): number[] {
  const recurrentGateStart = unitSize * FUSED_RECURRENT_GATE_GROUP_INDEX;

  return Array.from({ length: unitSize }, (_unused, hiddenUnitIndex) => {
    const recurrentRowIndex = recurrentGateStart + hiddenUnitIndex;
    const recurrentIndex = recurrentRowIndex * unitSize + hiddenUnitIndex;
    return recurrentTensorWeights[recurrentIndex];
  });
}

/**
 * Upsert one node self-connection for recurrent import restoration.
 *
 * @param selfConnectionContext Self-connection upsert context.
 * @returns Nothing.
 */
function upsertSelfConnection(
  selfConnectionContext: OnnxImportSelfConnectionUpsertContext,
): void {
  const { node, recurrentWeight } = selfConnectionContext;
  const nodeInternal = node as NodeInternals;
  const existingSelfConnection =
    nodeInternal.connections.self[FIRST_SELF_CONNECTION_INDEX];

  if (!existingSelfConnection) {
    const acquiredSelfConnection = Connection.acquire(
      node,
      node,
      recurrentWeight,
    );
    nodeInternal.connections.self.push(acquiredSelfConnection);
    node.connections.in.push(acquiredSelfConnection);
    node.connections.out.push(acquiredSelfConnection);
    return;
  }

  existingSelfConnection.weight = recurrentWeight;
}

/**
 * Parse pooling metadata payload from ONNX metadata.
 *
 * @param metadata ONNX metadata entries.
 * @returns Parsed pooling metadata payload.
 */
function parsePoolingMetadata(
  network: Network,
  metadata: OnnxMetadataProperty[],
): OnnxImportPoolingMetadata | null {
  const layersMetadata = findMetadataProperty(
    metadata,
    METADATA_KEY_POOL2D_LAYERS,
  );
  if (!layersMetadata) return null;

  const specsMetadata = findMetadataProperty(
    metadata,
    METADATA_KEY_POOL2D_SPECS,
  );
  try {
    const layers = JSON.parse(layersMetadata.value) as number[];
    const specs = specsMetadata
      ? (JSON.parse(specsMetadata.value) as Pool2DMapping[])
      : EMPTY_POOLING_SPECS;
    const flattenLayers = parseOptionalLayerIndicesMetadata(
      metadata,
      METADATA_KEY_FLATTEN_LAYERS,
    );
    const hiddenLayerSizes = parseOptionalLayerIndicesMetadata(
      metadata,
      METADATA_KEY_LAYER_SIZES,
    );
    const convSpecs = collectAvailableConvSpecs(metadata);
    const virtualShapes = collectVirtualPoolingShapes(
      specs,
      convSpecs,
      flattenLayers,
    );
    const flattenConsistency = collectFlattenConsistencyAudit(
      virtualShapes,
      hiddenLayerSizes,
      collectNodesByType(network.nodes, NODE_TYPE_OUTPUT).length,
    );

    return {
      layers,
      specs,
      flattenLayers,
      virtualShapes,
      ...(flattenConsistency.length === ZERO_LENGTH
        ? {}
        : { flattenConsistency }),
    };
  } catch {
    return null;
  }
}

/**
 * Parse advanced-graph cross-layer metadata from ONNX metadata.
 *
 * @param metadata ONNX metadata entries.
 * @returns Parsed advanced-graph metadata, or null when absent or invalid.
 */
function parseAdvancedGraphMetadata(
  metadata: OnnxMetadataProperty[],
  onnx?: OnnxModel,
): OnnxImportAdvancedGraphMetadata | null {
  const crossLayerConnections =
    parseAdvancedGraphCrossLayerConnections(metadata);
  const residualAdds = parseResidualAddMetadata(metadata);
  const sharedInitializerAliases = parseSharedInitializerAliases(metadata);
  const attentionBlocks = parseAttentionBlockMetadata(metadata, onnx);

  if (
    !crossLayerConnections &&
    !residualAdds &&
    !sharedInitializerAliases &&
    !attentionBlocks
  ) {
    return null;
  }

  return {
    ...(crossLayerConnections ? { crossLayerConnections } : {}),
    ...(residualAdds ? { residualAdds } : {}),
    ...(sharedInitializerAliases ? { sharedInitializerAliases } : {}),
    ...(attentionBlocks ? { attentionBlocks } : {}),
  };
}

/** Parse valid fixed-width self-attention audit metadata. */
function parseAttentionBlockMetadata(
  metadata: OnnxMetadataProperty[],
  onnx?: OnnxModel,
): OnnxImportAttentionBlock[] | null {
  if (!onnx) {
    return null;
  }

  const metadataProperty = findMetadataProperty(
    metadata,
    METADATA_KEY_ADVANCED_GRAPH_ATTENTION_BLOCKS,
  );
  if (!metadataProperty) {
    return null;
  }

  try {
    const parsedAttentionBlocks = JSON.parse(metadataProperty.value);
    if (!Array.isArray(parsedAttentionBlocks)) {
      return null;
    }

    const attentionBlocks = parsedAttentionBlocks.filter(isAttentionBlock);
    if (attentionBlocks.length !== parsedAttentionBlocks.length) {
      return null;
    }

    return attentionBlocks.every((attentionBlock) =>
      matchesExportedAttentionShadowSubset(onnx, attentionBlock),
    )
      ? attentionBlocks
      : null;
  } catch {
    return null;
  }
}

/** Parse valid one-hop residual-add metadata. */
function parseResidualAddMetadata(
  metadata: OnnxMetadataProperty[],
): OnnxImportResidualAdd[] | null {
  const metadataProperty = findMetadataProperty(
    metadata,
    METADATA_KEY_ADVANCED_GRAPH_RESIDUAL_ADDS,
  );
  if (!metadataProperty) {
    return null;
  }

  try {
    const parsedResidualAdds = JSON.parse(metadataProperty.value);
    if (!Array.isArray(parsedResidualAdds)) {
      return null;
    }

    const residualAdds = parsedResidualAdds.filter(isResidualAdd);
    if (residualAdds.length !== parsedResidualAdds.length) {
      return null;
    }

    return residualAdds;
  } catch {
    return null;
  }
}

/** Parse valid cross-layer feed-forward audit metadata. */
function parseAdvancedGraphCrossLayerConnections(
  metadata: OnnxMetadataProperty[],
): OnnxImportAdvancedGraphCrossLayerConnection[] | null {
  const metadataProperty = findMetadataProperty(
    metadata,
    METADATA_KEY_ADVANCED_GRAPH_CROSS_LAYER_CONNECTIONS,
  );
  if (!metadataProperty) {
    return null;
  }

  try {
    const parsedConnections = JSON.parse(metadataProperty.value);
    if (!Array.isArray(parsedConnections)) {
      return null;
    }

    const crossLayerConnections = parsedConnections.filter(
      isAdvancedGraphCrossLayerConnection,
    );
    if (crossLayerConnections.length !== parsedConnections.length) {
      return null;
    }

    return crossLayerConnections;
  } catch {
    return null;
  }
}

/** Parse valid shared-initializer alias audit metadata. */
function parseSharedInitializerAliases(
  metadata: OnnxMetadataProperty[],
): OnnxImportSharedInitializerAlias[] | null {
  const metadataProperty = findMetadataProperty(
    metadata,
    METADATA_KEY_SHARED_INITIALIZER_ALIASES,
  );
  if (!metadataProperty) {
    return null;
  }

  try {
    const parsedAliases = JSON.parse(metadataProperty.value);
    if (!Array.isArray(parsedAliases)) {
      return null;
    }

    const sharedInitializerAliases = parsedAliases.filter(
      isSharedInitializerAlias,
    );
    if (sharedInitializerAliases.length !== parsedAliases.length) {
      return null;
    }

    return sharedInitializerAliases;
  } catch {
    return null;
  }
}

/**
 * Validate one parsed cross-layer audit record.
 *
 * @param value Parsed JSON value.
 * @returns Whether the value matches the expected metadata shape.
 */
function isAdvancedGraphCrossLayerConnection(
  value: unknown,
): value is OnnxImportAdvancedGraphCrossLayerConnection {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate =
    value as Partial<OnnxImportAdvancedGraphCrossLayerConnection>;
  return (
    typeof candidate.sourceNodeIndex === 'number' &&
    typeof candidate.sourceLayerIndex === 'number' &&
    typeof candidate.targetNodeIndex === 'number' &&
    typeof candidate.targetLayerIndex === 'number' &&
    typeof candidate.branchTensorName === 'string'
  );
}

/** Validate one parsed shared-initializer alias audit record. */
function isSharedInitializerAlias(
  value: unknown,
): value is OnnxImportSharedInitializerAlias {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<OnnxImportSharedInitializerAlias>;
  return (
    typeof candidate.aliasTensorName === 'string' &&
    typeof candidate.canonicalTensorName === 'string' &&
    typeof candidate.initializerKind === 'string'
  );
}

/** Validate one parsed fixed-width self-attention audit record. */
function isAttentionBlock(value: unknown): value is OnnxImportAttentionBlock {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<OnnxImportAttentionBlock>;
  return (
    typeof candidate.sourceLayerIndex === 'number' &&
    typeof candidate.targetLayerIndex === 'number' &&
    typeof candidate.sequenceLength === 'number' &&
    typeof candidate.modelWidth === 'number' &&
    typeof candidate.heads === 'number' &&
    typeof candidate.shadowOutputName === 'string'
  );
}

function matchesExportedAttentionShadowSubset(
  onnx: OnnxModel,
  attentionBlock: OnnxImportAttentionBlock,
): boolean {
  const expectedOutputWeightName = `W${attentionBlock.targetLayerIndex - 1}`;
  const expectedOutputBiasName = `B${attentionBlock.targetLayerIndex - 1}`;
  const expectedShadowOutputName = attentionBlock.shadowOutputName;
  const expectedSoftmaxNodeName = `attention_l${attentionBlock.targetLayerIndex}_softmax`;
  const expectedProjectionNodeName = `attention_l${attentionBlock.targetLayerIndex}_output_projection`;

  if (
    attentionBlock.sourceLayerIndex !== attentionBlock.targetLayerIndex - 1 ||
    attentionBlock.targetLayerIndex <= ZERO_LENGTH ||
    attentionBlock.sequenceLength <= ZERO_LENGTH ||
    attentionBlock.modelWidth <= ZERO_LENGTH ||
    attentionBlock.heads <= ZERO_LENGTH ||
    attentionBlock.modelWidth % attentionBlock.heads !== ZERO_LENGTH
  ) {
    return false;
  }

  const queryWeights = findInitializer(
    onnx,
    `AttentionQW_l${attentionBlock.targetLayerIndex}`,
  );
  const keyWeights = findInitializer(
    onnx,
    `AttentionKW_l${attentionBlock.targetLayerIndex}`,
  );
  const valueWeights = findInitializer(
    onnx,
    `AttentionVW_l${attentionBlock.targetLayerIndex}`,
  );
  const outputWeights = findInitializer(onnx, expectedOutputWeightName);
  const outputBias = findInitializer(onnx, expectedOutputBiasName);
  const softmaxNode = findNodeByName(onnx, expectedSoftmaxNodeName);
  const outputProjectionNode = findNodeByName(onnx, expectedProjectionNodeName);

  return Boolean(
    queryWeights &&
    keyWeights &&
    valueWeights &&
    outputWeights &&
    outputBias &&
    softmaxNode?.op_type === 'Softmax' &&
    hasSoftmaxAxisMinusOne(softmaxNode) &&
    outputProjectionNode?.op_type === 'Gemm' &&
    outputProjectionNode.input[1] === expectedOutputWeightName &&
    outputProjectionNode.input[2] === expectedOutputBiasName &&
    outputProjectionNode.output[0] === expectedShadowOutputName &&
    hasSquareProjectionShape(queryWeights, attentionBlock.modelWidth) &&
    hasSquareProjectionShape(keyWeights, attentionBlock.modelWidth) &&
    hasSquareProjectionShape(valueWeights, attentionBlock.modelWidth) &&
    outputWeights.dims.at(-1) ===
      attentionBlock.sequenceLength * attentionBlock.modelWidth,
  );
}

function findInitializer(
  onnx: OnnxModel,
  tensorName: string,
): OnnxTensor | undefined {
  return onnx.graph.initializer.find(
    (initializerEntry) => initializerEntry.name === tensorName,
  );
}

function findNodeByName(onnx: OnnxModel, nodeName: string) {
  return onnx.graph.node.find((nodeEntry) => nodeEntry.name === nodeName);
}

function hasSoftmaxAxisMinusOne(
  node: OnnxModel['graph']['node'][number],
): boolean {
  return (
    node.attributes?.some(
      (attributeEntry) =>
        attributeEntry.name === 'axis' && attributeEntry.i === -1,
    ) ?? false
  );
}

function hasSquareProjectionShape(
  initializer: OnnxTensor,
  modelWidth: number,
): boolean {
  return (
    initializer.dims.length === 2 &&
    initializer.dims[0] === modelWidth &&
    initializer.dims[1] === modelWidth
  );
}

/** Validate one parsed residual-add record. */
function isResidualAdd(value: unknown): value is OnnxImportResidualAdd {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<OnnxImportResidualAdd>;
  return (
    typeof candidate.sourceLayerIndex === 'number' &&
    typeof candidate.targetLayerIndex === 'number' &&
    typeof candidate.branchTensorName === 'string' &&
    typeof candidate.mergeNodeName === 'string' &&
    typeof candidate.mergeOutputName === 'string'
  );
}

/** Build the imported layer ordering from runtime nodes and hidden-layer widths. */
function buildImportedLayers(
  network: Network,
  hiddenLayerSizes: number[],
): NeatapticNode[][] {
  const inputNodes = collectNodesByType(network.nodes, NODE_TYPE_INPUT);
  const hiddenNodes = collectNodesByType(network.nodes, NODE_TYPE_HIDDEN);
  const outputNodes = collectNodesByType(network.nodes, NODE_TYPE_OUTPUT);
  const hiddenLayers = hiddenLayerSizes.reduce(
    (state, hiddenLayerSize) => {
      const nextOffset = state.hiddenOffset + hiddenLayerSize;
      return {
        hiddenLayers: [
          ...state.hiddenLayers,
          hiddenNodes.slice(state.hiddenOffset, nextOffset),
        ],
        hiddenOffset: nextOffset,
      };
    },
    {
      hiddenLayers: [] as NeatapticNode[][],
      hiddenOffset: 0,
    },
  ).hiddenLayers;

  return [inputNodes, ...hiddenLayers, outputNodes];
}

/** Restore one residual-add branch using the recorded edge list and residual weight tensor. */
function restoreSingleResidualAddConnections(
  network: Network,
  onnx: OnnxModel,
  importedLayers: NeatapticNode[][],
  crossLayerConnections: OnnxImportAdvancedGraphCrossLayerConnection[],
  residualAdd: OnnxImportResidualAdd,
): void {
  const sourceLayerNodes = importedLayers[residualAdd.sourceLayerIndex];
  const targetLayerNodes = importedLayers[residualAdd.targetLayerIndex];
  const residualWeightTensor = onnx.graph.initializer.find(
    (tensor) => tensor.name === `ResidualW_l${residualAdd.targetLayerIndex}`,
  );
  if (!sourceLayerNodes || !targetLayerNodes || !residualWeightTensor) {
    return;
  }

  const sourceLayerIndexByNode = new Map(
    sourceLayerNodes.map((nodeEntry, sourceLayerIndex) => [
      nodeEntry,
      sourceLayerIndex,
    ]),
  );
  const targetLayerIndexByNode = new Map(
    targetLayerNodes.map((nodeEntry, targetLayerIndex) => [
      nodeEntry,
      targetLayerIndex,
    ]),
  );
  const matchingCrossLayerConnections = crossLayerConnections.filter(
    (crossLayerConnection) =>
      crossLayerConnection.sourceLayerIndex === residualAdd.sourceLayerIndex &&
      crossLayerConnection.targetLayerIndex === residualAdd.targetLayerIndex,
  );

  matchingCrossLayerConnections.forEach((crossLayerConnection) => {
    const sourceNode = network.nodes[crossLayerConnection.sourceNodeIndex];
    const targetNode = network.nodes[crossLayerConnection.targetNodeIndex];
    if (!sourceNode || !targetNode) {
      return;
    }

    const sourceLayerIndex = sourceLayerIndexByNode.get(sourceNode);
    const targetLayerIndex = targetLayerIndexByNode.get(targetNode);
    if (sourceLayerIndex === undefined || targetLayerIndex === undefined) {
      return;
    }

    const residualWeight = resolveResidualWeight(
      readOnnxTensorFloatData(residualWeightTensor),
      sourceLayerNodes.length,
      sourceLayerIndex,
      targetLayerIndex,
    );
    upsertFeedForwardConnection(sourceNode, targetNode, residualWeight);
  });
}

/** Resolve one residual branch weight from a row-major target-by-source matrix. */
function resolveResidualWeight(
  residualWeights: number[],
  sourceLayerWidth: number,
  sourceLayerIndex: number,
  targetLayerIndex: number,
): number {
  return (
    residualWeights[targetLayerIndex * sourceLayerWidth + sourceLayerIndex] ?? 0
  );
}

/** Upsert one feed-forward connection between two runtime nodes. */
function upsertFeedForwardConnection(
  sourceNode: NeatapticNode,
  targetNode: NeatapticNode,
  weight: number,
): void {
  const targetNodeInternal = targetNode as NodeInternals;
  const existingConnection = targetNodeInternal.connections.in.find(
    (connection) => connection.from === sourceNode,
  );
  if (existingConnection) {
    existingConnection.weight = weight;
    return;
  }

  const connection = Connection.acquire(sourceNode, targetNode, weight);
  sourceNode.connections.out.push(connection);
  targetNode.connections.in.push(connection);
}

/**
 * Parse one optional layer-index metadata field.
 *
 * @param metadata ONNX metadata entries.
 * @param metadataKey Metadata key carrying a JSON array of layer indices.
 * @returns Parsed layer indices or an empty list when absent/invalid.
 */
function parseOptionalLayerIndicesMetadata(
  metadata: OnnxMetadataProperty[],
  metadataKey: string,
): number[] {
  const metadataProperty = findMetadataProperty(metadata, metadataKey);
  if (!metadataProperty) return [];

  try {
    const parsedLayerIndices = JSON.parse(metadataProperty.value);
    return Array.isArray(parsedLayerIndices)
      ? (parsedLayerIndices as number[])
      : [];
  } catch {
    return [];
  }
}

/**
 * Collect explicit and inferred Conv specs that can anchor pooling shape simulation.
 *
 * @param metadata ONNX metadata entries.
 * @returns Layer-indexed Conv specs, preferring explicit specs over inferred ones.
 */
function collectAvailableConvSpecs(
  metadata: OnnxMetadataProperty[],
): Conv2DMapping[] {
  const inferredConvSpecs = parseOptionalConvSpecs(
    metadata,
    METADATA_KEY_CONV2D_INFERRED_SPECS,
  );
  const explicitConvSpecs = parseOptionalConvSpecs(
    metadata,
    METADATA_KEY_CONV2D_SPECS,
  );
  const convSpecsByLayerIndex = [
    ...inferredConvSpecs,
    ...explicitConvSpecs,
  ].reduce(
    (currentSpecMap, convSpec) =>
      currentSpecMap.set(convSpec.layerIndex, convSpec),
    new Map<number, Conv2DMapping>(),
  );

  return Array.from(convSpecsByLayerIndex.values()).toSorted(
    (leftSpec, rightSpec) => leftSpec.layerIndex - rightSpec.layerIndex,
  );
}

/**
 * Parse one optional Conv spec metadata field.
 *
 * @param metadata ONNX metadata entries.
 * @param metadataKey Metadata key carrying Conv spec JSON.
 * @returns Parsed Conv specs or an empty list when absent/invalid.
 */
function parseOptionalConvSpecs(
  metadata: OnnxMetadataProperty[],
  metadataKey: string,
): Conv2DMapping[] {
  const metadataProperty = findMetadataProperty(metadata, metadataKey);
  if (!metadataProperty) return [];

  try {
    const parsedConvSpecs = JSON.parse(metadataProperty.value);
    return Array.isArray(parsedConvSpecs)
      ? (parsedConvSpecs as Conv2DMapping[])
      : [];
  } catch {
    return [];
  }
}

/**
 * Derive virtual pooled shapes from Conv and Pool metadata without changing weights.
 *
 * @param poolingSpecs Imported pooling metadata specs.
 * @param convSpecs Available explicit/inferred Conv specs.
 * @param flattenLayers Layer indices marked with flatten-after-pool metadata.
 * @returns Derived virtual pooled shapes for future consistency checks.
 */
function collectVirtualPoolingShapes(
  poolingSpecs: Pool2DMapping[],
  convSpecs: Conv2DMapping[],
  flattenLayers: number[],
): OnnxImportPoolingVirtualShape[] {
  const convSpecsByLayerIndex = new Map(
    convSpecs.map((convSpec) => [convSpec.layerIndex, convSpec]),
  );
  const flattenLayerSet = new Set(flattenLayers);

  return poolingSpecs.flatMap((poolingSpec) => {
    const convSpec = convSpecsByLayerIndex.get(poolingSpec.afterLayerIndex);
    if (!convSpec) return [];

    const virtualShape = buildVirtualPoolingShape(
      poolingSpec,
      convSpec,
      flattenLayerSet,
    );
    return virtualShape ? [virtualShape] : [];
  });
}

/**
 * Collect flatten-consistency audit records for virtual pooled shapes with flatten metadata.
 *
 * @param virtualShapes Derived virtual pooled shapes.
 * @param hiddenLayerSizes Hidden-layer widths from ONNX metadata.
 * @param outputCount Imported output width.
 * @returns Metadata-only flatten-consistency audit records.
 */
function collectFlattenConsistencyAudit(
  virtualShapes: OnnxImportPoolingVirtualShape[],
  hiddenLayerSizes: number[],
  outputCount: number,
): OnnxImportFlattenConsistencyAudit[] {
  return virtualShapes.flatMap((virtualShape) => {
    if (virtualShape.flattenedSize === undefined) return [];

    const consumerLayerIndex =
      virtualShape.afterLayerIndex + HIDDEN_LAYER_NUMBER_OFFSET;
    const consumerWidth = resolveConsumerLayerWidth(
      consumerLayerIndex,
      hiddenLayerSizes,
      outputCount,
    );
    if (consumerWidth === undefined) return [];

    return [
      {
        afterLayerIndex: virtualShape.afterLayerIndex,
        consumerLayerIndex,
        consumerWidth,
        flattenedSize: virtualShape.flattenedSize,
        matches: consumerWidth === virtualShape.flattenedSize,
      },
    ];
  });
}

/**
 * Resolve the width of the next dense consumer after one flattened pooling site.
 *
 * @param consumerLayerIndex One-based export-layer index for the next dense consumer.
 * @param hiddenLayerSizes Hidden-layer widths from ONNX metadata.
 * @param outputCount Imported output width.
 * @returns Consumer width when a dense consumer exists.
 */
function resolveConsumerLayerWidth(
  consumerLayerIndex: number,
  hiddenLayerSizes: number[],
  outputCount: number,
): number | undefined {
  if (
    consumerLayerIndex >= HIDDEN_LAYER_NUMBER_OFFSET &&
    consumerLayerIndex <= hiddenLayerSizes.length
  ) {
    return hiddenLayerSizes[consumerLayerIndex - HIDDEN_LAYER_NUMBER_OFFSET];
  }

  return consumerLayerIndex ===
    hiddenLayerSizes.length + HIDDEN_LAYER_NUMBER_OFFSET
    ? outputCount
    : undefined;
}

/**
 * Build one virtual pooled shape from the pre-pool Conv output shape.
 *
 * @param poolingSpec Pool metadata describing the virtual pooling step.
 * @param convSpec Conv metadata describing the pre-pool spatial shape.
 * @param flattenLayerSet Layer indices marked with flatten-after-pool metadata.
 * @returns Virtual pooled shape when the metadata is usable.
 */
function buildVirtualPoolingShape(
  poolingSpec: Pool2DMapping,
  convSpec: Conv2DMapping,
  flattenLayerSet: Set<number>,
): OnnxImportPoolingVirtualShape | null {
  const outputHeight = calculateSpatialOutputSize(
    convSpec.outHeight,
    poolingSpec.kernelHeight,
    poolingSpec.strideHeight,
    poolingSpec.padTop ?? ZERO_LENGTH,
    poolingSpec.padBottom ?? ZERO_LENGTH,
  );
  const outputWidth = calculateSpatialOutputSize(
    convSpec.outWidth,
    poolingSpec.kernelWidth,
    poolingSpec.strideWidth,
    poolingSpec.padLeft ?? ZERO_LENGTH,
    poolingSpec.padRight ?? ZERO_LENGTH,
  );
  if (
    outputHeight < MINIMUM_SPATIAL_OUTPUT_SIZE ||
    outputWidth < MINIMUM_SPATIAL_OUTPUT_SIZE
  ) {
    return null;
  }

  const flattenedSize = flattenLayerSet.has(poolingSpec.afterLayerIndex)
    ? outputHeight * outputWidth * convSpec.outChannels
    : undefined;

  return {
    afterLayerIndex: poolingSpec.afterLayerIndex,
    inputChannels: convSpec.outChannels,
    inputHeight: convSpec.outHeight,
    inputWidth: convSpec.outWidth,
    outputChannels: convSpec.outChannels,
    outputHeight,
    outputWidth,
    ...(flattenedSize === undefined ? {} : { flattenedSize }),
  };
}

/**
 * Calculate one pooled spatial output size from kernel, stride, and padding metadata.
 *
 * @param inputSize Pre-pool spatial size.
 * @param kernelSize Pool kernel size.
 * @param strideSize Pool stride size.
 * @param leadingPadding Leading padding value.
 * @param trailingPadding Trailing padding value.
 * @returns Derived output size, or zero when the metadata is unusable.
 */
function calculateSpatialOutputSize(
  inputSize: number,
  kernelSize: number,
  strideSize: number,
  leadingPadding: number,
  trailingPadding: number,
): number {
  if (
    inputSize <= ZERO_LENGTH ||
    kernelSize <= ZERO_LENGTH ||
    strideSize <= ZERO_LENGTH
  ) {
    return ZERO_LENGTH;
  }

  return (
    Math.floor(
      (inputSize + leadingPadding + trailingPadding - kernelSize) / strideSize,
    ) + MINIMUM_SPATIAL_OUTPUT_SIZE
  );
}

/**
 * Attach parsed pooling metadata to imported network instance.
 *
 * @param network Target network.
 * @param poolingMetadata Parsed pooling metadata payload.
 * @returns Nothing.
 */
function attachParsedPoolingMetadata(
  network: Network,
  poolingMetadata: OnnxImportPoolingMetadata,
): void {
  const networkWithOnnxPooling = network as NetworkWithOnnxImportPooling;
  networkWithOnnxPooling._onnxPooling = poolingMetadata;
}
