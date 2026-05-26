import type Network from '../../network/network';
import Connection from '../../connection';
import type NeatapticNode from '../../node';
import * as methods from '../../../methods/methods';
import type { OnnxAttribute } from './schema/network.onnx.schema.types';
import type {
  ActivationFunction,
  NodeInternals,
  OnnxActivationOperation,
  OnnxExportOptions,
  LayerActivationValidationContext,
  LayerConnectivityValidationContext,
  LayerOrderingNodeGroups,
  LayerOrderingResolutionContext,
  LayerValidationTraversalContext,
} from './network.onnx.utils.types';
import {
  NetworkOnnxLayerOrderingUnresolvableError,
  NetworkOnnxMixedActivationsUnsupportedError,
  NetworkOnnxPartialConnectivityUnsupportedError,
} from './network.onnx.errors';

const NODE_TYPE_INPUT = 'input';
const NODE_TYPE_HIDDEN = 'hidden';
const NODE_TYPE_OUTPUT = 'output';

const EMPTY_STRING = '';

const ACTIVATION_TOKEN_TANH = 'TANH';
const ACTIVATION_TOKEN_LOGISTIC = 'LOGISTIC';
const ACTIVATION_TOKEN_SIGMOID = 'SIGMOID';
const ACTIVATION_TOKEN_RELU = 'RELU';
const DEFAULT_ONNX_OPSET = 18;
const GELU_MINIMUM_ONNX_OPSET = 20;
const MISH_MINIMUM_ONNX_OPSET = 18;
const SELU_ALPHA = 1.6732632423543772;
const SELU_GAMMA = 1.0507009873554805;

const ONNX_ACTIVATION_TANH: OnnxActivationOperation = 'Tanh';
const ONNX_ACTIVATION_SIGMOID: OnnxActivationOperation = 'Sigmoid';
const ONNX_ACTIVATION_RELU: OnnxActivationOperation = 'Relu';
const ONNX_ACTIVATION_IDENTITY: OnnxActivationOperation = 'Identity';
const ONNX_ACTIVATION_SOFTPLUS: OnnxActivationOperation = 'Softplus';
const ONNX_ACTIVATION_SOFTSIGN: OnnxActivationOperation = 'Softsign';
const ONNX_ACTIVATION_SELU: OnnxActivationOperation = 'Selu';
const ONNX_ACTIVATION_MISH: OnnxActivationOperation = 'Mish';
const ONNX_ACTIVATION_GELU: OnnxActivationOperation = 'Gelu';

const FIRST_NON_INPUT_LAYER_INDEX = 1;

const ERROR_LAYER_ORDERING_UNRESOLVABLE =
  'Invalid network structure for ONNX export: cannot resolve layered ordering.';
const ERROR_MIXED_ACTIVATIONS_PREFIX =
  'ONNX export error: Mixed activation functions detected in layer';
const ERROR_MIXED_ACTIVATIONS_SUFFIX =
  '(enable allowMixedActivations to decompose layer)';
const ERROR_PARTIAL_CONNECTIVITY_PREFIX =
  'ONNX export currently only supports simple MLPs. ONNX export error: Missing connection from node';
const ERROR_PARTIAL_CONNECTIVITY_SUFFIX = '(enable allowPartialConnectivity)';

const WARNING_MIXED_ACTIVATIONS_PREFIX = 'Warning: Mixed activations in layer';
const WARNING_MIXED_ACTIVATIONS_SUFFIX =
  'exporting per-neuron Gemm + Activation (+Concat) baseline.';
const WARNING_UNSUPPORTED_ACTIVATION_PREFIX = 'Unsupported activation function';
const WARNING_UNSUPPORTED_ACTIVATION_SUFFIX =
  'for ONNX export, defaulting to Identity.';

/**
 * Rebuild the network's flat connections array from each node's outgoing list.
 * Rehydrating this cache from node-owned adjacency shelves keeps exporter traversal deterministic after structural edits that may leave the flat cache stale.
 *
 * @param networkLike Network-like instance to mutate.
 * @returns Nothing.
 */
export function rebuildConnectionsLocal(networkLike: Network): void {
  // Step 1: Collect unique outgoing edges from all nodes.
  const rebuiltConnections = collectUniqueOutgoingConnections(
    networkLike.nodes,
  );
  // Step 2: Replace flat connection cache with the rebuilt set.
  networkLike.connections = rebuiltConnections;
}

/**
 * Map an internal activation function (squash) to an ONNX op_type.
 * Mapping flows through the exporter activation resolver so opset-gated operators and identity fallbacks stay centralized in one compatibility decision path.
 *
 * @param squash Activation function reference.
 * @returns ONNX activation operator name.
 */
export function mapActivationToOnnx(
  squash: ActivationFunction,
  opset: number = DEFAULT_ONNX_OPSET,
): OnnxActivationOperation {
  return resolveOnnxActivationNodeConfig(squash, opset).operation;
}

/**
 * Resolve the ONNX activation node payload for one runtime activation.
 * The payload includes both the resolved operator and any mandatory attributes, allowing downstream graph emission to stay declarative and free of activation-specific branching.
 *
 * @param squash Activation function reference.
 * @param opset Target ONNX opset.
 * @returns Activation operator plus any required ONNX attributes.
 */
export function resolveOnnxActivationNodeConfig(
  squash: ActivationFunction,
  opset: number = DEFAULT_ONNX_OPSET,
): {
  operation: OnnxActivationOperation;
  attributes?: OnnxAttribute[];
} {
  // Step 1: Resolve the best exporter-owned activation mapping.
  const activationResolution = resolveOnnxActivationOperation(squash, opset);

  // Step 2: Warn only when the exporter had to fall back.
  warnWhenActivationFallbackIsUsed({
    squash,
    didUseFallback: activationResolution.didUseFallback,
  });

  // Step 3: Return the caller-facing activation payload.
  return {
    operation: activationResolution.operation,
    attributes: activationResolution.attributes,
  };
}

/**
 * Infer a strictly layered node ordering from an analyzed network structure.
 *
 * @param network Source network.
 * @returns Ordered layers: input, hidden..., output.
 */
export function inferLayerOrdering(network: Network): NeatapticNode[][] {
  // Step 1: Partition nodes by role.
  const nodeGroups = collectLayerOrderingNodeGroups(network);

  if (hasNoHiddenNodes(nodeGroups)) {
    // Step 2a: Return direct input->output ordering for no-hidden networks.
    return finalizeOrderingWithoutHiddenNodes(nodeGroups);
  }

  // Step 2b: Initialize and resolve hidden layers incrementally.
  const initialResolutionContext =
    initializeLayerOrderingResolutionContext(nodeGroups);
  const resolvedLayerOrderingContext = resolveAllHiddenLayers(
    initialResolutionContext,
  );

  // Step 3: Append output layer to produce final order.
  return finalizeOrderingWithOutputLayer({
    orderedLayers: resolvedLayerOrderingContext.orderedLayers,
    outputNodes: nodeGroups.outputNodes,
  });
}

/**
 * Validate connectivity and activation homogeneity constraints per layer.
 * Validation enforces exporter baseline assumptions before node emission so unsupported mixed-activation or sparse connectivity cases are surfaced with actionable errors.
 *
 * @param layers Layered node arrays.
 * @param network Source network (reserved for compatibility).
 * @param options Export options.
 * @returns Nothing.
 */
export function validateLayerHomogeneityAndConnectivity(
  layers: NeatapticNode[][],
  network: Network,
  options: OnnxExportOptions,
): void {
  void network;
  // Step 1: Build layer-to-layer validation units.
  const layerValidationContexts = buildLayerValidationContexts(layers, options);
  // Step 2: Validate activation consistency and connectivity per unit.
  layerValidationContexts.forEach((layerValidationContext) => {
    validateSingleLayer(layerValidationContext);
  });
}

/**
 * Collect unique outgoing connections across a node list.
 *
 * @param nodes Nodes to traverse.
 * @returns Stable array of unique connections.
 */
function collectUniqueOutgoingConnections(
  nodes: NeatapticNode[],
): Connection[] {
  return Array.from(
    nodes.reduce((connectionSet, node) => {
      // Step 1: Merge this node's outgoing links into the shared set.
      node.connections?.out.forEach((connection) => {
        connectionSet.add(connection);
      });
      return connectionSet;
    }, new Set<Connection>()),
  );
}

/**
 * Normalize activation function name to uppercase for token matching.
 *
 * @param squash Runtime activation function reference.
 * @returns Uppercased activation name or empty string.
 */
function normalizeActivationName(squash: ActivationFunction): string {
  return (squash?.name ?? EMPTY_STRING).toUpperCase();
}

/**
 * Resolve ONNX activation op from a normalized activation name token.
 *
 * @param normalizedActivationName Uppercased activation name.
 * @returns ONNX activation operation.
 */
function resolveOnnxActivationOperation(
  squash: ActivationFunction,
  opset: number,
): {
  operation: OnnxActivationOperation;
  attributes?: OnnxAttribute[];
  didUseFallback: boolean;
} {
  if (!squash) {
    return { operation: ONNX_ACTIVATION_IDENTITY, didUseFallback: false };
  }

  if (squash === methods.Activation.identity) {
    return { operation: ONNX_ACTIVATION_IDENTITY, didUseFallback: false };
  }

  if (squash === methods.Activation.softplus) {
    return { operation: ONNX_ACTIVATION_SOFTPLUS, didUseFallback: false };
  }

  if (squash === methods.Activation.softsign) {
    return { operation: ONNX_ACTIVATION_SOFTSIGN, didUseFallback: false };
  }

  if (squash === methods.Activation.selu) {
    return {
      operation: ONNX_ACTIVATION_SELU,
      didUseFallback: false,
      attributes: [
        { name: 'alpha', type: 'FLOAT', f: SELU_ALPHA },
        { name: 'gamma', type: 'FLOAT', f: SELU_GAMMA },
      ],
    };
  }

  if (squash === methods.Activation.mish) {
    if (opset >= MISH_MINIMUM_ONNX_OPSET) {
      return { operation: ONNX_ACTIVATION_MISH, didUseFallback: false };
    }

    return { operation: ONNX_ACTIVATION_IDENTITY, didUseFallback: true };
  }

  if (squash === methods.Activation.gelu) {
    if (opset >= GELU_MINIMUM_ONNX_OPSET) {
      return {
        operation: ONNX_ACTIVATION_GELU,
        didUseFallback: false,
        attributes: [{ name: 'approximate', type: 'STRING', s: 'tanh' }],
      };
    }

    return { operation: ONNX_ACTIVATION_IDENTITY, didUseFallback: true };
  }

  const normalizedActivationName = normalizeActivationName(squash);
  if (normalizedActivationName.includes(ACTIVATION_TOKEN_TANH)) {
    return { operation: ONNX_ACTIVATION_TANH, didUseFallback: false };
  }

  if (
    normalizedActivationName.includes(ACTIVATION_TOKEN_LOGISTIC) ||
    normalizedActivationName.includes(ACTIVATION_TOKEN_SIGMOID)
  ) {
    return { operation: ONNX_ACTIVATION_SIGMOID, didUseFallback: false };
  }

  if (normalizedActivationName.includes(ACTIVATION_TOKEN_RELU)) {
    return { operation: ONNX_ACTIVATION_RELU, didUseFallback: false };
  }

  return { operation: ONNX_ACTIVATION_IDENTITY, didUseFallback: true };
}

/**
 * Emit a warning when activation export falls back to Identity.
 *
 * @param context Activation fallback evaluation context.
 * @returns Nothing.
 */
function warnWhenActivationFallbackIsUsed(context: {
  squash: ActivationFunction;
  didUseFallback: boolean;
}): void {
  if (!context.squash) {
    return;
  }

  if (!context.didUseFallback) {
    return;
  }

  console.warn(
    `${WARNING_UNSUPPORTED_ACTIVATION_PREFIX} ${context.squash.name} ${WARNING_UNSUPPORTED_ACTIVATION_SUFFIX}`,
  );
}

/**
 * Partition all network nodes into input/hidden/output groups.
 *
 * @param network Source network.
 * @returns Node groups used by layered-ordering inference.
 */
function collectLayerOrderingNodeGroups(
  network: Network,
): LayerOrderingNodeGroups {
  return {
    inputNodes: filterNodesByType(network.nodes, NODE_TYPE_INPUT),
    hiddenNodes: filterNodesByType(network.nodes, NODE_TYPE_HIDDEN),
    outputNodes: filterNodesByType(network.nodes, NODE_TYPE_OUTPUT),
  };
}

/**
 * Filter nodes by one expected node type.
 *
 * @param nodes Candidate node list.
 * @param nodeType Expected node type.
 * @returns Matching nodes.
 */
function filterNodesByType(
  nodes: NeatapticNode[],
  nodeType: string,
): NeatapticNode[] {
  return nodes.filter((node) => node.type === nodeType);
}

/**
 * Check whether the layer groups contain no hidden nodes.
 *
 * @param nodeGroups Partitioned node groups.
 * @returns True when hidden layer traversal can be skipped.
 */
function hasNoHiddenNodes(nodeGroups: LayerOrderingNodeGroups): boolean {
  return nodeGroups.hiddenNodes.length === 0;
}

/**
 * Finalize ordering for networks without hidden layers.
 *
 * @param nodeGroups Partitioned node groups.
 * @returns Input and output layers only.
 */
function finalizeOrderingWithoutHiddenNodes(
  nodeGroups: LayerOrderingNodeGroups,
): NeatapticNode[][] {
  return [nodeGroups.inputNodes, nodeGroups.outputNodes];
}

/**
 * Create initial hidden-layer resolution context.
 *
 * @param nodeGroups Partitioned node groups.
 * @returns Initial mutable state for hidden-layer resolution.
 */
function initializeLayerOrderingResolutionContext(
  nodeGroups: LayerOrderingNodeGroups,
): LayerOrderingResolutionContext {
  return {
    remainingHiddenNodes: [...nodeGroups.hiddenNodes],
    previousLayerNodes: nodeGroups.inputNodes,
    orderedLayers: [],
  };
}

/**
 * Resolve all hidden layers in dependency order.
 *
 * @param initialContext Starting hidden-layer resolution context.
 * @returns Final resolved layer-ordering context.
 */
function resolveAllHiddenLayers(
  initialContext: LayerOrderingResolutionContext,
): LayerOrderingResolutionContext {
  let resolutionContext = initialContext;

  // Step 1: Iteratively resolve one hidden layer at a time.
  while (resolutionContext.remainingHiddenNodes.length > 0) {
    resolutionContext = resolveNextHiddenLayer(resolutionContext);
  }

  // Step 2: Persist the last resolved hidden layer.
  return appendLastResolvedLayer(resolutionContext);
}

/**
 * Resolve the next hidden layer from unresolved candidates.
 *
 * @param resolutionContext Current resolution state.
 * @returns Updated resolution state.
 */
function resolveNextHiddenLayer(
  resolutionContext: LayerOrderingResolutionContext,
): LayerOrderingResolutionContext {
  // Step 1: Select hidden nodes fully driven by previous layer outputs.
  const currentLayerNodes =
    collectCurrentResolvableHiddenLayer(resolutionContext);
  // Step 2: Guard against cyclic or unsupported topology.
  ensureLayerWasResolved(currentLayerNodes);

  // Step 3: Advance traversal state with current layer removal.
  return {
    orderedLayers: [
      ...resolutionContext.orderedLayers,
      resolutionContext.previousLayerNodes,
    ],
    previousLayerNodes: currentLayerNodes,
    remainingHiddenNodes: filterUnresolvedHiddenNodes({
      remainingHiddenNodes: resolutionContext.remainingHiddenNodes,
      currentLayerNodes,
    }),
  };
}

/**
 * Collect unresolved hidden nodes that can be placed in the next layer.
 *
 * @param resolutionContext Current hidden-layer resolution context.
 * @returns Hidden nodes that are resolvable in this pass.
 */
function collectCurrentResolvableHiddenLayer(
  resolutionContext: LayerOrderingResolutionContext,
): NeatapticNode[] {
  return resolutionContext.remainingHiddenNodes.filter((hiddenNode) =>
    hasAllIncomingConnectionsFromPreviousLayer({
      hiddenNode,
      previousLayerNodes: resolutionContext.previousLayerNodes,
    }),
  );
}

/**
 * Check whether a hidden node receives all inputs from the previous layer.
 *
 * @param context Hidden-node connectivity check context.
 * @returns True when the hidden node is layer-resolvable.
 */
function hasAllIncomingConnectionsFromPreviousLayer(context: {
  hiddenNode: NeatapticNode;
  previousLayerNodes: NeatapticNode[];
}): boolean {
  const hiddenNodeInternals = context.hiddenNode as unknown as NodeInternals;
  return hiddenNodeInternals.connections.in.every((connection) =>
    context.previousLayerNodes.includes(connection.from),
  );
}

/**
 * Ensure current hidden-layer resolution pass produced at least one node.
 *
 * @param currentLayerNodes Nodes resolved for current layer.
 * @returns Nothing.
 * @throws Error When no hidden nodes can be resolved.
 */
function ensureLayerWasResolved(currentLayerNodes: NeatapticNode[]): void {
  if (currentLayerNodes.length > 0) {
    return;
  }

  throw new NetworkOnnxLayerOrderingUnresolvableError(
    ERROR_LAYER_ORDERING_UNRESOLVABLE,
  );
}

/**
 * Remove just-resolved hidden nodes from unresolved candidates.
 *
 * @param context Remaining/just-resolved hidden node context.
 * @returns Hidden nodes still unresolved.
 */
function filterUnresolvedHiddenNodes(context: {
  remainingHiddenNodes: NeatapticNode[];
  currentLayerNodes: NeatapticNode[];
}): NeatapticNode[] {
  return context.remainingHiddenNodes.filter(
    (hiddenNode) => !context.currentLayerNodes.includes(hiddenNode),
  );
}

/**
 * Append the final resolved hidden layer into ordered layer output.
 *
 * @param resolutionContext Final traversal state before append.
 * @returns Traversal state with last hidden layer persisted.
 */
function appendLastResolvedLayer(
  resolutionContext: LayerOrderingResolutionContext,
): LayerOrderingResolutionContext {
  return {
    ...resolutionContext,
    orderedLayers: [
      ...resolutionContext.orderedLayers,
      resolutionContext.previousLayerNodes,
    ],
  };
}

/**
 * Append output layer to resolved input/hidden ordering.
 *
 * @param context Final ordering context.
 * @returns Full layer ordering including output layer.
 */
function finalizeOrderingWithOutputLayer(context: {
  orderedLayers: NeatapticNode[][];
  outputNodes: NeatapticNode[];
}): NeatapticNode[][] {
  return [...context.orderedLayers, context.outputNodes];
}

/**
 * Build per-layer validation contexts for all non-input layers.
 *
 * @param layers Ordered network layers.
 * @param options ONNX export options.
 * @returns Traversal contexts used by layer validators.
 */
function buildLayerValidationContexts(
  layers: NeatapticNode[][],
  options: OnnxExportOptions,
): LayerValidationTraversalContext[] {
  return layers.slice(FIRST_NON_INPUT_LAYER_INDEX).map(
    (currentLayerNodes, layerOffset): LayerValidationTraversalContext => ({
      layerIndex: layerOffset + FIRST_NON_INPUT_LAYER_INDEX,
      previousLayerNodes: layers[layerOffset],
      currentLayerNodes,
      options,
    }),
  );
}

/**
 * Validate one current layer against activation/connectivity constraints.
 *
 * @param layerValidationContext Layer validation context.
 * @returns Nothing.
 */
function validateSingleLayer(
  layerValidationContext: LayerValidationTraversalContext,
): void {
  // Step 1: Build a focused activation-check context.
  const activationValidationContext = createLayerActivationValidationContext(
    layerValidationContext,
  );

  // Step 2: Validate activation consistency, then full connectivity.
  validateLayerActivationHomogeneity(activationValidationContext);
  validateLayerConnectivity(layerValidationContext);
}

/**
 * Create activation validation context from one layer traversal context.
 *
 * @param layerValidationContext Layer validation context.
 * @returns Activation validation context.
 */
function createLayerActivationValidationContext(
  layerValidationContext: LayerValidationTraversalContext,
): LayerActivationValidationContext {
  return {
    layerIndex: layerValidationContext.layerIndex,
    activationNames: layerValidationContext.currentLayerNodes.map((node) => {
      const nodeInternal = node as unknown as NodeInternals;
      return nodeInternal.squash?.name;
    }),
    allowMixedActivations:
      layerValidationContext.options.allowMixedActivations ?? false,
  };
}

/**
 * Validate that a layer has homogeneous activation unless explicitly allowed.
 *
 * @param activationValidationContext Activation validation context.
 * @returns Nothing.
 * @throws Error When mixed activations are disallowed.
 */
function validateLayerActivationHomogeneity(
  activationValidationContext: LayerActivationValidationContext,
): void {
  const activationNameSet = new Set(
    activationValidationContext.activationNames,
  );
  const hasMixedActivations = activationNameSet.size > 1;

  if (!hasMixedActivations) {
    return;
  }

  if (!activationValidationContext.allowMixedActivations) {
    throw new NetworkOnnxMixedActivationsUnsupportedError(
      `${ERROR_MIXED_ACTIVATIONS_PREFIX} ${activationValidationContext.layerIndex}. ${ERROR_MIXED_ACTIVATIONS_SUFFIX}`,
    );
  }

  console.warn(
    `${WARNING_MIXED_ACTIVATIONS_PREFIX} ${activationValidationContext.layerIndex}; ${WARNING_MIXED_ACTIVATIONS_SUFFIX}`,
  );
}

/**
 * Validate that each current-layer node has required incoming connectivity.
 *
 * @param layerValidationContext Layer connectivity traversal context.
 * @returns Nothing.
 */
function validateLayerConnectivity(
  layerValidationContext: LayerValidationTraversalContext,
): void {
  layerValidationContext.currentLayerNodes.forEach((targetNode) => {
    validateTargetNodeConnectivity({
      targetNode,
      previousLayerNodes: layerValidationContext.previousLayerNodes,
      layerIndex: layerValidationContext.layerIndex,
      allowPartialConnectivity:
        layerValidationContext.options.allowPartialConnectivity ?? false,
    });
  });
}

/**
 * Validate full source coverage for one target node.
 *
 * @param context Target-node connectivity context.
 * @returns Nothing.
 */
function validateTargetNodeConnectivity(context: {
  targetNode: NeatapticNode;
  previousLayerNodes: NeatapticNode[];
  layerIndex: number;
  allowPartialConnectivity: boolean;
}): void {
  context.previousLayerNodes.forEach((sourceNode) => {
    validateSourceToTargetConnectivity({
      layerIndex: context.layerIndex,
      sourceNode,
      targetNode: context.targetNode,
      allowPartialConnectivity: context.allowPartialConnectivity,
    });
  });
}

/**
 * Validate one source->target connection pair under export constraints.
 *
 * @param connectivityValidationContext Source/target connectivity context.
 * @returns Nothing.
 * @throws Error When required source->target edge is missing.
 */
function validateSourceToTargetConnectivity(
  connectivityValidationContext: LayerConnectivityValidationContext,
): void {
  if (connectivityValidationContext.allowPartialConnectivity) {
    return;
  }

  const targetNodeInternal =
    connectivityValidationContext.targetNode as unknown as NodeInternals;
  const hasSourceConnection = targetNodeInternal.connections.in.some(
    (connection) =>
      connection.from === connectivityValidationContext.sourceNode,
  );

  if (hasSourceConnection) {
    return;
  }

  throw new NetworkOnnxPartialConnectivityUnsupportedError(
    `${ERROR_PARTIAL_CONNECTIVITY_PREFIX} ${connectivityValidationContext.sourceNode.index} to node ${connectivityValidationContext.targetNode.index} in layer ${connectivityValidationContext.layerIndex}. ${ERROR_PARTIAL_CONNECTIVITY_SUFFIX}`,
  );
}
