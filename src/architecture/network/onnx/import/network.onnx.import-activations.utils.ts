import * as methods from '../../../../methods/methods';
import type Network from '../../network';
import type { OnnxModel } from '../schema/network.onnx.schema.types';
import type {
  ActivationFunction,
  HiddenLayerActivationTraversalContext,
  NodeInternals,
  OnnxActivationAssignmentContext,
  OnnxActivationLayerOperations,
  OnnxActivationOperation,
  OnnxActivationOperationResolutionContext,
  OnnxActivationParseResult,
  OutputLayerActivationContext,
} from '../network.onnx.utils.types';

/** Empty string literal for safe node-name parsing defaults. */
const EMPTY_NODE_NAME = '';

/** Runtime node type discriminator for hidden nodes. */
const HIDDEN_NODE_TYPE = 'hidden';

/** Runtime node type discriminator for output nodes. */
const OUTPUT_NODE_TYPE = 'output';

/** ONNX export layers are 1-indexed. */
const EXPORT_LAYER_INDEX_OFFSET = 1;

/** Output layer follows the final hidden layer in ONNX export indexing. */
const OUTPUT_LAYER_INDEX_OFFSET = 1;

/** Fallback operation index used when a per-neuron activation is absent. */
const DEFAULT_LAYER_OPERATION_INDEX = 0;

/** Initial hidden-node offset for first hidden-layer traversal context. */
const INITIAL_HIDDEN_OFFSET = 0;

/** Empty traversal-context sentinel length. */
const NO_HIDDEN_TRAVERSAL_CONTEXTS = 0;

/** ONNX activation node-name pattern emitted by ONNX export. */
const ACTIVATION_NODE_NAME_PATTERN = /^act(?:_conv)?_l(\d+)(?:_n(\d+))?$/i;

/** Default activation operation when no explicit mapping exists. */
const DEFAULT_ACTIVATION_OPERATION: OnnxActivationOperation = 'Identity';

/** Supported ONNX activation operator names for import. */
const SUPPORTED_ACTIVATION_OPERATIONS = new Set<OnnxActivationOperation>([
  'Tanh',
  'Sigmoid',
  'Logistic',
  'Relu',
  'Identity',
  'Softplus',
  'Softsign',
  'Selu',
  'Mish',
  'Gelu',
]);

/** Mapping from ONNX op types to runtime activation functions. */
const ACTIVATION_OPERATION_TO_FUNCTION = new Map<
  OnnxActivationOperation,
  ActivationFunction
>([
  ['Tanh', methods.Activation.tanh],
  ['Sigmoid', methods.Activation.sigmoid],
  ['Logistic', methods.Activation.sigmoid],
  ['Relu', methods.Activation.relu],
  ['Identity', methods.Activation.identity],
  ['Softplus', methods.Activation.softplus],
  ['Softsign', methods.Activation.softsign],
  ['Selu', methods.Activation.selu],
  ['Mish', methods.Activation.mish],
  ['Gelu', methods.Activation.gelu],
]);

/**
 * Assign runtime node activation functions from ONNX activation graph operations.
 *
 * @param network Target network to mutate.
 * @param onnx Source ONNX model.
 * @param hiddenLayerSizes Hidden layer size list.
 * @returns Nothing.
 */
export function assignActivationFunctions(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
): void {
  // Step 1: Build one immutable context for hidden and output assignment.
  const assignmentContext = buildActivationAssignmentContext(
    network,
    onnx,
    hiddenLayerSizes,
  );

  // Step 2: Apply hidden-layer operations with tiny traversal contexts.
  applyHiddenLayerActivations(assignmentContext);
  // Step 3: Apply output-layer operation using the same operation lookup.
  applyOutputLayerActivation(assignmentContext);
  return;
}

/**
 * Build immutable assignment context for hidden/output activation import.
 *
 * @param sourceNetwork Target network to mutate.
 * @param sourceOnnx Source ONNX model.
 * @param sourceHiddenLayerSizes Hidden layer widths in export order.
 * @returns Prepared assignment context.
 */
function buildActivationAssignmentContext(
  sourceNetwork: Network,
  sourceOnnx: OnnxModel,
  sourceHiddenLayerSizes: number[],
): OnnxActivationAssignmentContext {
  return {
    hiddenLayerSizes: sourceHiddenLayerSizes,
    hiddenNodes: collectNodeInternalsByType(sourceNetwork, HIDDEN_NODE_TYPE),
    outputNodes: collectNodeInternalsByType(sourceNetwork, OUTPUT_NODE_TYPE),
    operationsByLayer: collectOperationsByLayer(sourceOnnx),
  };
}

/**
 * Collect node internals by runtime node type.
 *
 * @param sourceNetwork Network with runtime nodes.
 * @param targetType Runtime node type to collect.
 * @returns Runtime node internals matching the requested type.
 */
function collectNodeInternalsByType(
  sourceNetwork: Network,
  targetType: string,
): NodeInternals[] {
  return sourceNetwork.nodes
    .filter((node) => node.type === targetType)
    .map(asNodeInternals);
}

/**
 * Collect ONNX activation operations grouped by export-layer index.
 *
 * @param sourceOnnx Source ONNX model.
 * @returns Layer-indexed activation operation lookup.
 */
function collectOperationsByLayer(
  sourceOnnx: OnnxModel,
): OnnxActivationLayerOperations {
  return sourceOnnx.graph.node.reduce<OnnxActivationLayerOperations>(
    (operationsByLayer, graphNode) => {
      const parseResult = parseActivationNode(
        graphNode.name ?? EMPTY_NODE_NAME,
      );
      const operation = asSupportedActivationOperation(graphNode.op_type);
      if (!parseResult || !operation) return operationsByLayer;

      appendOperationToLayer(
        operationsByLayer,
        parseResult.layerIndex,
        operation,
      );
      return operationsByLayer;
    },
    {},
  );
}

/**
 * Parse one ONNX activation node name.
 *
 * @param nodeName ONNX graph node name.
 * @returns Parsed layer/neuron metadata or null when not a supported activation node.
 */
function parseActivationNode(
  nodeName: string,
): OnnxActivationParseResult | null {
  const match = ACTIVATION_NODE_NAME_PATTERN.exec(nodeName);
  if (!match) return null;
  return {
    layerIndex: Number(match[1]),
    neuronIndex: match[2] !== undefined ? Number(match[2]) : undefined,
  };
}

/**
 * Convert an ONNX op type to a supported activation operation.
 *
 * @param operationName ONNX graph node operation type.
 * @returns Supported activation operation or null when unsupported.
 */
function asSupportedActivationOperation(
  operationName: string,
): OnnxActivationOperation | null {
  if (
    !SUPPORTED_ACTIVATION_OPERATIONS.has(
      operationName as OnnxActivationOperation,
    )
  ) {
    return null;
  }
  return operationName as OnnxActivationOperation;
}

/**
 * Append one operation to the lookup bucket for a layer.
 *
 * @param operationsByLayer Layer-indexed operation lookup.
 * @param layerIndex Export-layer index.
 * @param operation Supported activation operation.
 * @returns Nothing.
 */
function appendOperationToLayer(
  operationsByLayer: OnnxActivationLayerOperations,
  layerIndex: number,
  operation: OnnxActivationOperation,
): void {
  const existingOperations = operationsByLayer[layerIndex] ?? [];
  operationsByLayer[layerIndex] = [...existingOperations, operation];
}

/**
 * Apply imported activation operations to hidden layer nodes.
 *
 * @param context Shared assignment context.
 * @returns Nothing.
 */
function applyHiddenLayerActivations(
  context: OnnxActivationAssignmentContext,
): void {
  const hiddenLayerTraversalContexts =
    buildHiddenLayerTraversalContexts(context);
  if (hiddenLayerTraversalContexts.length === NO_HIDDEN_TRAVERSAL_CONTEXTS) {
    return;
  }

  applyHiddenLayerTraversalContexts(hiddenLayerTraversalContexts);
}

/**
 * Build traversal contexts for each hidden layer.
 *
 * @param context Shared assignment context.
 * @returns Ordered hidden-layer traversal contexts.
 */
function buildHiddenLayerTraversalContexts(
  context: OnnxActivationAssignmentContext,
): HiddenLayerActivationTraversalContext[] {
  return context.hiddenLayerSizes.reduce<
    HiddenLayerActivationTraversalContext[]
  >((traversalContexts, hiddenLayerSize, hiddenLayerIndex) => {
    const previousContext = traversalContexts.at(-1);
    const hiddenOffset =
      previousContext === undefined
        ? INITIAL_HIDDEN_OFFSET
        : previousContext.hiddenOffset + previousContext.hiddenLayerSize;
    return [
      ...traversalContexts,
      {
        hiddenLayerIndex,
        hiddenLayerSize,
        hiddenOffset,
        hiddenNodes: context.hiddenNodes,
        operationsByLayer: context.operationsByLayer,
      },
    ];
  }, []);
}

/**
 * Apply hidden-layer activation assignment for each traversal context.
 *
 * @param traversalContexts Ordered hidden-layer traversal contexts.
 * @returns Nothing.
 */
function applyHiddenLayerTraversalContexts(
  traversalContexts: HiddenLayerActivationTraversalContext[],
): void {
  traversalContexts.forEach(applyHiddenLayerTraversalActivation);
}

/**
 * Apply activation operations for one hidden-layer traversal context.
 *
 * @param traversalContext Hidden-layer traversal context.
 * @returns Nothing.
 */
function applyHiddenLayerTraversalActivation(
  traversalContext: HiddenLayerActivationTraversalContext,
): void {
  buildHiddenNeuronIndices(traversalContext.hiddenLayerSize).forEach(
    (neuronIndex) => {
      applyHiddenNeuronActivation(traversalContext, neuronIndex);
    },
  );
}

/**
 * Build contiguous hidden-neuron index list for one layer.
 *
 * @param hiddenLayerSize Hidden-layer width.
 * @returns Ordered neuron indices.
 */
function buildHiddenNeuronIndices(hiddenLayerSize: number): number[] {
  return Array.from(
    { length: hiddenLayerSize },
    (_, hiddenNeuronIndex) => hiddenNeuronIndex,
  );
}

/**
 * Apply one hidden neuron activation if the target node exists.
 *
 * @param traversalContext Hidden-layer traversal context.
 * @param neuronIndex Neuron index in current hidden layer.
 * @returns Nothing.
 */
function applyHiddenNeuronActivation(
  traversalContext: HiddenLayerActivationTraversalContext,
  neuronIndex: number,
): void {
  const hiddenNode = resolveHiddenNode(traversalContext, neuronIndex);
  if (!hiddenNode) return;

  hiddenNode.squash = resolveActivationFunction({
    operations: resolveLayerOperations(traversalContext),
    neuronIndex,
  });
}

/**
 * Resolve one hidden node by traversal/offset metadata.
 *
 * @param traversalContext Hidden-layer traversal context.
 * @param neuronIndex Neuron index in current hidden layer.
 * @returns Hidden node internals when present.
 */
function resolveHiddenNode(
  traversalContext: HiddenLayerActivationTraversalContext,
  neuronIndex: number,
): NodeInternals | undefined {
  const absoluteHiddenNodeIndex = traversalContext.hiddenOffset + neuronIndex;
  return traversalContext.hiddenNodes[absoluteHiddenNodeIndex];
}

/**
 * Resolve operations list for one hidden layer.
 *
 * @param traversalContext Hidden-layer traversal context.
 * @returns Ordered operations for target export layer.
 */
function resolveLayerOperations(
  traversalContext: HiddenLayerActivationTraversalContext,
): OnnxActivationOperation[] {
  const exportLayerIndex =
    traversalContext.hiddenLayerIndex + EXPORT_LAYER_INDEX_OFFSET;
  return traversalContext.operationsByLayer[exportLayerIndex] ?? [];
}

/**
 * Apply imported activation to all output nodes.
 *
 * @param context Shared assignment context.
 * @returns Nothing.
 */
function applyOutputLayerActivation(
  context: OnnxActivationAssignmentContext,
): void {
  const outputLayerContext: OutputLayerActivationContext = {
    outputNodes: context.outputNodes,
    operationsByLayer: context.operationsByLayer,
    outputLayerIndex:
      context.hiddenLayerSizes.length + OUTPUT_LAYER_INDEX_OFFSET,
  };
  const outputActivationFunction = resolveActivationFunction({
    operations: resolveOutputOperations(outputLayerContext),
    neuronIndex: DEFAULT_LAYER_OPERATION_INDEX,
  });

  outputLayerContext.outputNodes.forEach((outputNode) => {
    outputNode.squash = outputActivationFunction;
  });
}

/**
 * Resolve output-layer operations from shared lookup.
 *
 * @param context Output-layer assignment context.
 * @returns Ordered output-layer operations.
 */
function resolveOutputOperations(
  context: OutputLayerActivationContext,
): OnnxActivationOperation[] {
  return context.operationsByLayer[context.outputLayerIndex] ?? [];
}

/**
 * Resolve runtime activation function from operation context.
 *
 * @param context Operation-resolution context.
 * @returns Runtime activation function.
 */
function resolveActivationFunction(
  context: OnnxActivationOperationResolutionContext,
): ActivationFunction {
  const operation = resolveOperationByPriority(context);
  return ACTIVATION_OPERATION_TO_FUNCTION.get(operation)!;
}

/**
 * Resolve activation operation by neuron-first then layer-default fallback.
 *
 * @param context Operation-resolution context.
 * @returns Supported activation operation.
 */
function resolveOperationByPriority(
  context: OnnxActivationOperationResolutionContext,
): OnnxActivationOperation {
  return (
    context.operations[context.neuronIndex] ??
    context.operations[DEFAULT_LAYER_OPERATION_INDEX] ??
    DEFAULT_ACTIVATION_OPERATION
  );
}

/**
 * Cast one public node instance to runtime node internals.
 *
 * @param node Source node object.
 * @returns Runtime node internals.
 */
function asNodeInternals(node: unknown): NodeInternals {
  return node as NodeInternals;
}
