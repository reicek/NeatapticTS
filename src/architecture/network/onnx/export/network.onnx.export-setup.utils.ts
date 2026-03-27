import type NeatapticNode from '../../../node';
import type {
  OnnxDimension,
  OnnxModel,
  OnnxValueInfo,
} from '../schema/network.onnx.schema.types';
import type {
  OnnxBaseModelBuildContext,
  OnnxGraphDimensionBuildContext,
  OnnxGraphDimensions,
  OnnxModelMetadataContext,
  OnnxRecurrentCollectionContext,
  OnnxRecurrentInputValueInfoContext,
  OnnxRecurrentLayerProcessingContext,
  OnnxRecurrentLayerTraversalContext,
} from './network.onnx.export.types';
import type { NodeInternals } from '../network.onnx.utils.types';

const FLOAT_TENSOR_ELEMENT_TYPE = 1;
const ONNX_IR_VERSION = 9;
const DEFAULT_MODEL_INPUT_NAME = 'input';
const DEFAULT_MODEL_OUTPUT_NAME = 'output';
const SYMBOLIC_BATCH_DIMENSION_NAME = 'N';
const DEFAULT_METADATA_OPSET_DOMAIN = '';
const DEFAULT_METADATA_PRODUCER_VERSION = '0.0.0';
const DEFAULT_METADATA_DOC_STRING =
  'Exported from NeatapticTS ONNX exporter (phases 1-2 baseline)';
const FIRST_HIDDEN_LAYER_INDEX = 1;
const RECURRENT_PREVIOUS_STATE_PRIMARY_INPUT_NAME = 'hidden_prev';
const RECURRENT_PREVIOUS_STATE_LAYER_PREFIX = 'hidden_prev_l';

/**
 * Build tensor dimensions for model input and output, optionally with symbolic batch dimension.
 *
 * @param context Dimension construction context.
 * @returns Input and output dimension arrays for ONNX value info.
 */
export function createGraphDimensions(
  context: OnnxGraphDimensionBuildContext,
): OnnxGraphDimensions {
  // Step 1: Build dimensions for input and output graph boundaries.
  const inputDims = createTensorDimensions(
    context.inputWidth,
    context.batchDimension,
  );
  const outputDims = createTensorDimensions(
    context.outputWidth,
    context.batchDimension,
  );

  // Step 2: Fold both sides into a typed graph dimensions payload.
  return { inputDims, outputDims };
}

/**
 * Create the base ONNX model shell with graph input/output declarations.
 *
 * @param context Base model build context.
 * @returns Initialized ONNX model with empty initializer/node lists.
 */
export function createBaseModel(context: OnnxBaseModelBuildContext): OnnxModel {
  // Step 1: Construct graph input/output value info payloads.
  const graphInputs = [
    createGraphValueInfo(DEFAULT_MODEL_INPUT_NAME, context.inputDims),
  ];
  const graphOutputs = [
    createGraphValueInfo(DEFAULT_MODEL_OUTPUT_NAME, context.outputDims),
  ];

  // Step 2: Return base model shell with empty node and initializer arrays.
  return {
    graph: {
      inputs: graphInputs,
      outputs: graphOutputs,
      initializer: [],
      node: [],
    },
  };
}

/**
 * Attach producer and opset metadata to a model when metadata emission is enabled.
 *
 * @param context Metadata application context.
 * @returns Nothing.
 */
export function applyModelMetadata(context: OnnxModelMetadataContext): void {
  // Step 1: Exit when metadata emission is disabled.
  if (!context.includeMetadata) return;

  // Step 2: Resolve stable metadata fallbacks.
  const resolvedProducerVersion =
    context.producerVersion ?? DEFAULT_METADATA_PRODUCER_VERSION;
  const resolvedDocString = context.docString ?? DEFAULT_METADATA_DOC_STRING;

  // Step 3: Apply metadata fields to the model payload.
  context.model.ir_version = ONNX_IR_VERSION;
  context.model.opset_import = [
    { version: context.opset, domain: DEFAULT_METADATA_OPSET_DOMAIN },
  ];
  context.model.producer_name = context.producerName;
  context.model.producer_version = resolvedProducerVersion;
  context.model.doc_string = resolvedDocString;
}

/**
 * Detect hidden layers with self-recurrence and add matching previous-state graph inputs.
 *
 * @param context Recurrent collection context.
 * @returns Export-layer indices with recurrent self-connections.
 */
export function collectRecurrentLayerIndices(
  context: OnnxRecurrentCollectionContext,
): number[] {
  // Step 1: Exit when recurrent collection is not enabled.
  const recurrentLayerIndices: number[] = [];
  if (!isRecurrentCollectionEnabled(context)) return recurrentLayerIndices;

  // Step 2: Traverse hidden layers and append recurrent graph inputs where needed.
  const hiddenLayerTraversalContexts =
    createHiddenLayerTraversalContexts(context);
  hiddenLayerTraversalContexts.forEach((traversalContext) => {
    processHiddenLayerRecurrence({
      model: context.model,
      traversalContext,
      recurrentLayerIndices,
    });
  });

  // Step 3: Return discovered recurrent hidden-layer indices.
  return recurrentLayerIndices;
}

/**
 * Build one tensor shape dimension payload for dense vectors.
 *
 * @param width Vector width.
 * @param batchDimension Whether symbolic batch dimension is enabled.
 * @returns ONNX dimensions for the vector payload.
 */
function createTensorDimensions(
  width: number,
  batchDimension: boolean,
): OnnxDimension[] {
  // Step 1: Create symbolic batch prefix when enabled.
  const leadingDimensions = batchDimension
    ? [{ dim_param: SYMBOLIC_BATCH_DIMENSION_NAME }]
    : [];

  // Step 2: Append concrete feature width dimension.
  return [...leadingDimensions, { dim_value: width }];
}

/**
 * Create ONNX value info payload for one graph boundary tensor.
 *
 * @param valueName Tensor value name.
 * @param dimensions Tensor dimensions.
 * @returns ONNX value info payload.
 */
function createGraphValueInfo(
  valueName: string,
  dimensions: OnnxDimension[],
): OnnxValueInfo {
  // Step 1: Return strongly-shaped ONNX value info for the provided tensor.
  return {
    name: valueName,
    type: {
      tensor_type: {
        elem_type: FLOAT_TENSOR_ELEMENT_TYPE,
        shape: { dim: dimensions },
      },
    },
  };
}

/**
 * Determine whether recurrent layer collection should execute.
 *
 * @param context Recurrent collection context.
 * @returns True when recurrent collection is enabled.
 */
function isRecurrentCollectionEnabled(
  context: OnnxRecurrentCollectionContext,
): boolean {
  // Step 1: Require both recurrent export and single-step mode toggles.
  return (
    !!context.options.allowRecurrent && !!context.options.recurrentSingleStep
  );
}

/**
 * Build traversal contexts for all hidden layers.
 *
 * @param context Recurrent collection context.
 * @returns Hidden layer traversal contexts.
 */
function createHiddenLayerTraversalContexts(
  context: OnnxRecurrentCollectionContext,
): OnnxRecurrentLayerTraversalContext[] {
  // Step 1: Build index range for hidden layers only.
  const hiddenLayerIndices = createHiddenLayerIndices(context.layers.length);

  // Step 2: Map hidden layer indices to typed traversal payloads.
  return hiddenLayerIndices.map((layerIndex) => ({
    layerIndex,
    hiddenLayerNodes: context.layers[layerIndex],
    batchDimension: context.batchDimension,
  }));
}

/**
 * Build hidden layer indices excluding input and output layers.
 *
 * @param totalLayerCount Total number of network layers.
 * @returns Hidden layer indices.
 */
function createHiddenLayerIndices(totalLayerCount: number): number[] {
  // Step 1: Compute hidden layer count from total layer count.
  const hiddenLayerCount = Math.max(totalLayerCount - 2, 0);

  // Step 2: Return ordered hidden layer indices.
  return Array.from(
    { length: hiddenLayerCount },
    (_, hiddenLayerOffset) => FIRST_HIDDEN_LAYER_INDEX + hiddenLayerOffset,
  );
}

/**
 * Process one hidden layer for recurrent self-connections.
 *
 * @param context Hidden layer recurrent processing context.
 * @returns Nothing.
 */
function processHiddenLayerRecurrence(
  context: OnnxRecurrentLayerProcessingContext,
): void {
  // Step 1: Exit when layer does not contain self-recurrent nodes.
  if (!hasLayerSelfRecurrence(context.traversalContext.hiddenLayerNodes))
    return;

  // Step 2: Record recurrent layer index for metadata and downstream graph logic.
  appendRecurrentLayerIndex(
    context.recurrentLayerIndices,
    context.traversalContext,
  );

  // Step 3: Append recurrent previous-state input to model graph.
  appendRecurrentGraphInput(context.model, context.traversalContext);
}

/**
 * Append one recurrent layer index to the collected index list.
 *
 * @param recurrentLayerIndices Collected recurrent layer indices.
 * @param traversalContext Hidden layer traversal context.
 * @returns Nothing.
 */
function appendRecurrentLayerIndex(
  recurrentLayerIndices: number[],
  traversalContext: OnnxRecurrentLayerTraversalContext,
): void {
  // Step 1: Push the current hidden layer index into collected recurrent indices.
  recurrentLayerIndices.push(traversalContext.layerIndex);
}

/**
 * Append one recurrent previous-state graph input for a hidden layer.
 *
 * @param model Target ONNX model.
 * @param traversalContext Hidden layer traversal context.
 * @returns Nothing.
 */
function appendRecurrentGraphInput(
  model: OnnxModel,
  traversalContext: OnnxRecurrentLayerTraversalContext,
): void {
  // Step 1: Build recurrent input payload context for the hidden layer.
  const recurrentInputContext =
    createRecurrentInputValueInfoContext(traversalContext);

  // Step 2: Append generated recurrent value info to graph inputs.
  model.graph.inputs.push(createRecurrentInputValueInfo(recurrentInputContext));
}

/**
 * Detect whether a hidden layer contains at least one self-recurrent node.
 *
 * @param hiddenLayerNodes Hidden layer nodes.
 * @returns True when a node has a self-connection.
 */
function hasLayerSelfRecurrence(hiddenLayerNodes: NeatapticNode[]): boolean {
  // Step 1: Inspect each node for one or more self-connections.
  return hiddenLayerNodes.some(
    (hiddenLayerNode) =>
      (hiddenLayerNode as unknown as NodeInternals).connections.self.length > 0,
  );
}

/**
 * Build recurrent input context for one hidden recurrent layer.
 *
 * @param traversalContext Hidden layer traversal context.
 * @returns Recurrent input value-info context.
 */
function createRecurrentInputValueInfoContext(
  traversalContext: OnnxRecurrentLayerTraversalContext,
): OnnxRecurrentInputValueInfoContext {
  // Step 1: Resolve stable input name for this recurrent layer.
  const previousStateInputName =
    traversalContext.layerIndex === FIRST_HIDDEN_LAYER_INDEX
      ? RECURRENT_PREVIOUS_STATE_PRIMARY_INPUT_NAME
      : `${RECURRENT_PREVIOUS_STATE_LAYER_PREFIX}${traversalContext.layerIndex}`;

  // Step 2: Return typed recurrent input context.
  return {
    previousStateInputName,
    hiddenLayerWidth: traversalContext.hiddenLayerNodes.length,
    batchDimension: traversalContext.batchDimension,
  };
}

/**
 * Build one recurrent previous-state graph input payload.
 *
 * @param context Recurrent input value-info context.
 * @returns ONNX value info payload for recurrent state input.
 */
function createRecurrentInputValueInfo(
  context: OnnxRecurrentInputValueInfoContext,
): OnnxValueInfo {
  // Step 1: Reuse shared value-info builder with recurrent width dimensions.
  return createGraphValueInfo(
    context.previousStateInputName,
    createTensorDimensions(context.hiddenLayerWidth, context.batchDimension),
  );
}
