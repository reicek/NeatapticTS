import type NeatapticNode from '../../../../node';
import type { OnnxModel } from '../../schema/network.onnx.schema.types';
import type {
  RecurrentActivationEmissionContext,
  RecurrentGemmEmissionContext,
  RecurrentGraphNames,
  RecurrentInitializerEmissionContext,
  RecurrentInitializerNames,
  RecurrentInitializerValues,
  RecurrentLayerEmissionContext,
  RecurrentLayerEmissionParams,
} from '../network.onnx.export.types';
import type { NodeInternals } from '../../network.onnx.utils.types';
import {
  buildDenseWeightsAndBiases,
  buildDiagonalRecurrentWeights,
} from './network.onnx.export-layer-common.utils';
import { resolveOnnxActivationNodeConfig } from '../../network.onnx.layer-analysis.utils';

/** ONNX float tensor data type id. */
const ONNX_FLOAT_DATA_TYPE = 1;
/** ONNX Gemm alpha default coefficient. */
const GEMM_ALPHA = 1;
/** ONNX Gemm beta default coefficient. */
const GEMM_BETA = 1;
/** ONNX Gemm transB flag for row-major source weight layout. */
const GEMM_TRANSPOSE_B = 1;
/** First hidden layer export index used by recurrent single-step path. */
const FIRST_RECURRENT_LAYER_INDEX = 1;

/** Tensor-name prefixes for recurrent single-step emission. */
const WEIGHT_TENSOR_PREFIX = 'W';
const BIAS_TENSOR_PREFIX = 'B';
const RECURRENT_TENSOR_PREFIX = 'R';

/** Shared ONNX op names used by recurrent single-step emission. */
const GEMM_OP_TYPE = 'Gemm';
const ADD_OP_TYPE = 'Add';

/** Shared ONNX attribute names and types. */
const ATTRIBUTE_ALPHA = 'alpha';
const ATTRIBUTE_BETA = 'beta';
const ATTRIBUTE_TRANSPOSE_B = 'transB';
const ATTRIBUTE_TYPE_FLOAT = 'FLOAT';
const ATTRIBUTE_TYPE_INT = 'INT';

/** Shared recurrent tensor-name prefixes. */
const INPUT_GEMM_OUTPUT_PREFIX = 'Gemm_in_';
const RECURRENT_GEMM_OUTPUT_PREFIX = 'Gemm_rec_';
const RECURRENT_SUM_OUTPUT_PREFIX = 'RecurrentSum_';
const LAYER_OUTPUT_PREFIX = 'Layer_';
const INPUT_GEMM_NODE_PREFIX = 'gemm_in_l';
const RECURRENT_GEMM_NODE_PREFIX = 'gemm_rec_l';
const RECURRENT_ADD_NODE_PREFIX = 'add_recurrent_l';
const ACTIVATION_NODE_PREFIX = 'act_l';

/** Hidden-state input naming defaults for recurrent branch emission. */
const BASE_PREVIOUS_HIDDEN_INPUT_NAME = 'hidden_prev';
const PREVIOUS_HIDDEN_LAYER_INPUT_PREFIX = 'hidden_prev_l';

/**
 * Emit the constrained recurrent single-step export path for one hidden layer.
 *
 * This boundary models recurrence with two parallel Gemm branches:
 * one for the feed-forward input and one for the previous hidden state. The
 * recurrent branch uses a diagonal matrix derived from self-connections only,
 * which keeps the exported shape simple and matches the importer's current
 * reconstruction contract.
 *
 * Hidden-state inputs are named `hidden_prev` for the first recurrent layer and
 * `hidden_prev_l{n}` for later recurrent layers. Mixed activations are not
 * supported on this path because the single activation node is applied after
 * the input and recurrent branches are summed.
 *
 * @param params Recurrent emission parameters.
 * @returns Output tensor name.
 * @example
 * ```ts
 * const outputName = emitRecurrentLayer({
 *   model,
 *   layerIndex: 1,
 *   previousOutputName: 'input',
 *   previousLayerNodes,
 *   currentLayerNodes,
 * });
 * ```
 */
export function emitRecurrentLayer(
  params: RecurrentLayerEmissionParams,
): string {
  // Step 1: Derive stable execution context and deterministic names.
  const emissionContext = buildRecurrentLayerEmissionContext(params);
  const initializerNames = buildRecurrentInitializerNames(emissionContext);
  const graphNames = buildRecurrentGraphNames(emissionContext);

  // Step 2: Collect reusable initializer payloads for dense + recurrent branches.
  const initializerValues = collectRecurrentInitializerValues(emissionContext);

  // Step 3: Emit initializer tensors and branch Gemm nodes.
  emitRecurrentInitializers({
    model: emissionContext.model,
    previousLayerWidth: emissionContext.previousLayerWidth,
    currentLayerWidth: emissionContext.currentLayerWidth,
    names: initializerNames,
    values: initializerValues,
  });
  emitRecurrentGemmNode(
    buildInputBranchGemmEmissionContext(
      emissionContext,
      initializerNames,
      graphNames,
    ),
  );
  emitRecurrentGemmNode(
    buildRecurrentBranchGemmEmissionContext(
      emissionContext,
      initializerNames,
      graphNames,
    ),
  );

  // Step 4: Fold branch outputs and apply layer activation.
  emitRecurrentAddNode(emissionContext.model, graphNames);
  emitRecurrentActivationNode({
    model: emissionContext.model,
    currentLayerNodes: emissionContext.currentLayerNodes,
    recurrentSumOutputName: graphNames.recurrentSumOutputName,
    layerOutputName: graphNames.layerOutputName,
    activationNodeName: graphNames.activationNodeName,
    opset: emissionContext.opset,
  });

  // Step 5: Return the canonical layer output tensor name.
  return graphNames.layerOutputName;
}

/**
 * Build derived recurrent-layer context from input params.
 *
 * @param params User-provided recurrent layer params.
 * @returns Derived context with cached dimensions and layer slot.
 */
function buildRecurrentLayerEmissionContext(
  params: RecurrentLayerEmissionParams,
): RecurrentLayerEmissionContext {
  return {
    ...params,
    layerSlot: params.layerIndex - 1,
    previousLayerWidth: params.previousLayerNodes.length,
    currentLayerWidth: params.currentLayerNodes.length,
  };
}

/**
 * Build deterministic tensor names for recurrent initializer emission.
 *
 * @param context Recurrent layer execution context.
 * @returns Tensor-name group for initializer emission.
 */
function buildRecurrentInitializerNames(
  context: RecurrentLayerEmissionContext,
): RecurrentInitializerNames {
  return {
    weightTensorName: `${WEIGHT_TENSOR_PREFIX}${context.layerSlot}`,
    biasTensorName: `${BIAS_TENSOR_PREFIX}${context.layerSlot}`,
    recurrentTensorName: `${RECURRENT_TENSOR_PREFIX}${context.layerSlot}`,
  };
}

/**
 * Build deterministic graph names for recurrent-node emission.
 *
 * @param context Recurrent layer execution context.
 * @returns Graph-name group for branch and activation nodes.
 */
function buildRecurrentGraphNames(
  context: RecurrentLayerEmissionContext,
): RecurrentGraphNames {
  return {
    inputGemmOutputName: `${INPUT_GEMM_OUTPUT_PREFIX}${context.layerIndex}`,
    recurrentGemmOutputName: `${RECURRENT_GEMM_OUTPUT_PREFIX}${context.layerIndex}`,
    recurrentSumOutputName: `${RECURRENT_SUM_OUTPUT_PREFIX}${context.layerIndex}`,
    layerOutputName: `${LAYER_OUTPUT_PREFIX}${context.layerIndex}`,
    inputGemmNodeName: `${INPUT_GEMM_NODE_PREFIX}${context.layerIndex}`,
    recurrentGemmNodeName: `${RECURRENT_GEMM_NODE_PREFIX}${context.layerIndex}`,
    recurrentAddNodeName: `${RECURRENT_ADD_NODE_PREFIX}${context.layerIndex}`,
    activationNodeName: `${ACTIVATION_NODE_PREFIX}${context.layerIndex}`,
  };
}

/**
 * Collect recurrent initializer vectors for one layer.
 *
 * @param context Recurrent layer execution context.
 * @returns Dense and recurrent initializer vectors.
 */
function collectRecurrentInitializerValues(
  context: RecurrentLayerEmissionContext,
): RecurrentInitializerValues {
  const { weightMatrixValues, biasVector } = buildDenseWeightsAndBiases(
    context.previousLayerNodes,
    context.currentLayerNodes,
  );
  const recurrentWeights = buildDiagonalRecurrentWeights(
    context.currentLayerNodes,
  );
  return {
    weightMatrixValues,
    biasVector,
    recurrentWeights,
  };
}

/**
 * Emit dense and recurrent initializer tensors.
 *
 * @param context Initializer emission context.
 * @returns Nothing.
 */
function emitRecurrentInitializers(
  context: RecurrentInitializerEmissionContext,
): void {
  context.model.graph.initializer.push({
    name: context.names.weightTensorName,
    data_type: ONNX_FLOAT_DATA_TYPE,
    dims: [context.currentLayerWidth, context.previousLayerWidth],
    float_data: context.values.weightMatrixValues,
  });
  context.model.graph.initializer.push({
    name: context.names.biasTensorName,
    data_type: ONNX_FLOAT_DATA_TYPE,
    dims: [context.currentLayerWidth],
    float_data: context.values.biasVector,
  });
  context.model.graph.initializer.push({
    name: context.names.recurrentTensorName,
    data_type: ONNX_FLOAT_DATA_TYPE,
    dims: [context.currentLayerWidth, context.currentLayerWidth],
    float_data: context.values.recurrentWeights,
  });
}

/**
 * Build Gemm emission context for the feed-forward branch.
 *
 * @param context Recurrent layer execution context.
 * @param initializerNames Recurrent initializer names.
 * @param graphNames Recurrent graph names.
 * @returns Gemm emission context.
 */
function buildInputBranchGemmEmissionContext(
  context: RecurrentLayerEmissionContext,
  initializerNames: RecurrentInitializerNames,
  graphNames: RecurrentGraphNames,
): RecurrentGemmEmissionContext {
  return {
    model: context.model,
    inputNames: [
      context.previousOutputName,
      initializerNames.weightTensorName,
      initializerNames.biasTensorName,
    ],
    outputName: graphNames.inputGemmOutputName,
    nodeName: graphNames.inputGemmNodeName,
  };
}

/**
 * Build Gemm emission context for the recurrent hidden-state branch.
 *
 * @param context Recurrent layer execution context.
 * @param initializerNames Recurrent initializer names.
 * @param graphNames Recurrent graph names.
 * @returns Gemm emission context.
 */
function buildRecurrentBranchGemmEmissionContext(
  context: RecurrentLayerEmissionContext,
  initializerNames: RecurrentInitializerNames,
  graphNames: RecurrentGraphNames,
): RecurrentGemmEmissionContext {
  return {
    model: context.model,
    inputNames: [
      resolvePreviousHiddenInputName(context.layerIndex),
      initializerNames.recurrentTensorName,
    ],
    outputName: graphNames.recurrentGemmOutputName,
    nodeName: graphNames.recurrentGemmNodeName,
  };
}

/**
 * Resolve recurrent branch hidden-state input for one layer.
 *
 * @param layerIndex Current recurrent layer index.
 * @returns Hidden-state tensor input name.
 */
function resolvePreviousHiddenInputName(layerIndex: number): string {
  if (layerIndex === FIRST_RECURRENT_LAYER_INDEX) {
    return BASE_PREVIOUS_HIDDEN_INPUT_NAME;
  }
  return `${PREVIOUS_HIDDEN_LAYER_INPUT_PREFIX}${layerIndex}`;
}

/**
 * Emit one recurrent Gemm node with shared ONNX attributes.
 *
 * @param context Gemm emission context.
 * @returns Nothing.
 */
function emitRecurrentGemmNode(context: RecurrentGemmEmissionContext): void {
  context.model.graph.node.push({
    op_type: GEMM_OP_TYPE,
    input: context.inputNames,
    output: [context.outputName],
    name: context.nodeName,
    attributes: buildDefaultGemmAttributes(),
  });
}

/**
 * Emit Add node that fuses feed-forward and recurrent branch outputs.
 *
 * @param model Target ONNX model.
 * @param graphNames Deterministic graph names for this layer.
 * @returns Nothing.
 */
function emitRecurrentAddNode(
  model: OnnxModel,
  graphNames: RecurrentGraphNames,
): void {
  model.graph.node.push({
    op_type: ADD_OP_TYPE,
    input: [graphNames.inputGemmOutputName, graphNames.recurrentGemmOutputName],
    output: [graphNames.recurrentSumOutputName],
    name: graphNames.recurrentAddNodeName,
  });
}

/**
 * Emit activation node for recurrent branch sum output.
 *
 * @param context Activation emission context.
 * @returns Nothing.
 */
function emitRecurrentActivationNode(
  context: RecurrentActivationEmissionContext,
): void {
  const activationConfig = resolveOnnxActivationNodeConfig(
    readNodeInternals(context.currentLayerNodes[0]).squash,
    context.opset,
  );

  context.model.graph.node.push({
    op_type: activationConfig.operation,
    input: [context.recurrentSumOutputName],
    output: [context.layerOutputName],
    name: context.activationNodeName,
    attributes: activationConfig.attributes,
  });
}

/**
 * Normalize runtime node shape to recurrent-export internals contract.
 *
 * @param node Runtime node instance.
 * @returns Node internals used by ONNX emission helpers.
 */
function readNodeInternals(node: NeatapticNode): NodeInternals {
  return node as NodeInternals;
}

/**
 * Build the shared attribute list for ONNX Gemm node payloads.
 *
 * @returns Gemm attribute payload list.
 */
function buildDefaultGemmAttributes(): {
  name: string;
  type: string;
  f?: number;
  i?: number;
}[] {
  return [
    { name: ATTRIBUTE_ALPHA, type: ATTRIBUTE_TYPE_FLOAT, f: GEMM_ALPHA },
    { name: ATTRIBUTE_BETA, type: ATTRIBUTE_TYPE_FLOAT, f: GEMM_BETA },
    {
      name: ATTRIBUTE_TRANSPOSE_B,
      type: ATTRIBUTE_TYPE_INT,
      i: GEMM_TRANSPOSE_B,
    },
  ];
}
