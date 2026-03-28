import Connection from '../../../connection';
import type Network from '../../network';
import type NeatapticNode from '../../../node';
import { deriveHiddenLayerSizes } from './network.onnx.import-weights.utils';
import { reconstructFusedRecurrentLayers } from './network.onnx.import-fused-recurrent.utils';
import type {
  OnnxMetadataProperty,
  OnnxModel,
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';
import type { NodeInternals } from '../network.onnx.utils.types';
import type {
  NetworkWithOnnxImportPooling,
  OnnxImportArchitectureContext,
  OnnxImportArchitectureResult,
  OnnxImportDimensionRecord,
  OnnxImportHiddenLayerSpan,
  OnnxImportLayerConnectionContext,
  OnnxImportPoolingMetadata,
  OnnxImportRecurrentRestorationContext,
  OnnxImportSelfConnectionUpsertContext,
} from './network.onnx.import-orchestrators.types';

export { reconstructFusedRecurrentLayers };

const METADATA_KEY_RECURRENT_SINGLE_STEP = 'recurrent_single_step';
const METADATA_KEY_POOL2D_LAYERS = 'pool2d_layers';
const METADATA_KEY_POOL2D_SPECS = 'pool2d_specs';
const NODE_TYPE_INPUT = 'input';
const NODE_TYPE_HIDDEN = 'hidden';
const NODE_TYPE_OUTPUT = 'output';
const ONNX_DIM_VALUE_FIELD = 'dim_value';
const RECURRENT_TENSOR_PREFIX = 'R';
const HIDDEN_LAYER_NUMBER_OFFSET = 1;
const DEFAULT_NON_MATCHING_LAYER_INDEX = 0;
const FIRST_SELF_CONNECTION_INDEX = 0;
const ZERO_LENGTH = 0;
const EMPTY_POOLING_SPECS: Pool2DMapping[] = [];

/**
 * Extract input/output counts and hidden layer sizes from ONNX model.
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
 * Remove placeholder hidden nodes for single-layer perceptron imports.
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
 *
 * @param network Target network.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer sizes.
 * @param metadata Parsed metadata properties.
 * @returns Nothing.
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
  const poolingMetadata = parsePoolingMetadata(metadata);

  // Step 2: Exit for absent or invalid pooling metadata payloads.
  if (!poolingMetadata) return;

  // Step 3: Attach parsed pooling metadata to the imported network.
  attachParsedPoolingMetadata(network, poolingMetadata);
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
  const recurrentInitializer = findRecurrentInitializer(layerConnectionContext);
  if (!recurrentInitializer) return;

  // Step 2: Slice the target hidden-layer node segment for the span.
  const layerHiddenNodes = sliceLayerHiddenNodes(layerConnectionContext);

  // Step 3: Build diagonal recurrent self-weights for this layer.
  const recurrentDiagonalWeights = collectDiagonalRecurrentWeights(
    recurrentInitializer.float_data,
    layerConnectionContext.span.hiddenLayerSize,
  );

  // Step 4: Upsert one self-connection per hidden unit.
  layerHiddenNodes.forEach((hiddenNode, hiddenUnitIndex) => {
    upsertSelfConnection({
      node: hiddenNode,
      recurrentWeight: recurrentDiagonalWeights[hiddenUnitIndex],
    });
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
    restorationContext.metadata,
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
  metadata: OnnxMetadataProperty[],
): number[] {
  const recurrentMetadata = findMetadataProperty(
    metadata,
    METADATA_KEY_RECURRENT_SINGLE_STEP,
  );
  if (!recurrentMetadata) return [];
  return parseRecurrentLayerIndices(recurrentMetadata.value);
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
    | number[]
    | number
    | string
    | boolean
    | null
    | Record<string, number>,
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
): { name: string; float_data: number[] } | undefined {
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
    return { layers, specs };
  } catch {
    return null;
  }
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
