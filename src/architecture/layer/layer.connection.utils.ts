import Connection from '../connection/connection';
import Group from '../group';
import * as methods from '../../methods/methods';
import Node from '../node';
import type { LayerConnectionContext, LayerLike } from './layer.utils.types';

const LAYER_OUTPUT_CONNECT_ERROR =
  'Layer output is not defined. Cannot connect from this layer.';
const LAYER_OUTPUT_GATE_ERROR =
  'Layer output is not defined. Cannot gate from this layer.';
const LAYER_OUTPUT_INPUT_TARGET_ERROR =
  'Layer output (acting as input target) is not defined.';
const LAYER_OUTPUT_INPUT_SOURCE_ERROR =
  'Layer output (acting as input source) is not defined.';
const NODE_INDEX_START = 0;
const LOOP_INDEX_STEP = 1;
const REVERSE_INDEX_OFFSET = 1;
const REVERSE_INDEX_STEP = 1;

/**
 * Connects a layer's output group to a target.
 *
 * Conceptually, this is the "forward wiring" helper: it takes whatever this
 * layer exposes as its `output` group and connects it to another structure.
 *
 * Target handling:
 * - **Layer-like target**: calls `target.input(layer, method, weight)` so the
 *   target can decide how to interpret the source.
 * - **Group/Node target**: calls `output.connect(target, method, weight)`.
 *
 * Throws when `context.output` is `null` (some layer types may not expose an
 * output group).
 *
 * Example:
 *
 * ```ts
 * connectLayer(layerAContext, layerBLike);
 * connectLayer(layerAContext, someGroup, methods.groupConnection.ALL_TO_ALL);
 * ```
 *
 * @param context - The layer state needed for connections.
 * @param target - The layer, group, or node to connect to.
 * @param method - Optional connection method override.
 * @param weight - Optional fixed weight to apply.
 * @returns The created connection list.
 */
export function connectLayer(
  context: LayerConnectionContext,
  target: Group | Node | LayerLike,
  method?: unknown,
  weight?: number,
): Connection[] {
  const { output, isLayer, layer } = context;

  // Step 1: Ensure the output group exists before connecting.
  if (!output) {
    throw new Error(LAYER_OUTPUT_CONNECT_ERROR);
  }

  // Step 2: Delegate connection creation based on the target type.
  if (isLayer(target)) {
    return target.input(layer, method, weight);
  }

  if (target instanceof Group || target instanceof Node) {
    return output.connect(target, method, weight);
  }

  return [];
}

/**
 * Applies gating to the provided connections using the layer output group.
 *
 * Gating lets a third party (here, this layer's output group) *modulate* a set
 * of connections. This is used in recurrent architectures (e.g. LSTM/GRU) to
 * implement gate-controlled flow.
 *
 * Throws when `context.output` is `null`.
 *
 * Example:
 *
 * ```ts
 * gateLayer(layerContext, connections, methods.gating.OUTPUT);
 * ```
 *
 * @param context - The layer state needed for gating.
 * @param connections - The connections to gate.
 * @param method - The gating method.
 */
export function gateLayer(
  context: LayerConnectionContext,
  connections: Connection[],
  method: unknown,
): void {
  const { output } = context;

  // Step 1: Ensure the output group exists before gating.
  if (!output) {
    throw new Error(LAYER_OUTPUT_GATE_ERROR);
  }

  // Step 2: Delegate gating to the output group.
  output.gate(connections, method);
}

/**
 * Connects a source group or layer to this layer's input target.
 *
 * This is the inverse of `connectLayer(...)`: instead of using *this* layer as
 * the source, we treat this layer as the *target* and connect a provided
 * source into `context.output`.
 *
 * If `from` is a layer-like object, its `output` group is used as the source.
 * If no `method` is provided, this defaults to `ALL_TO_ALL` wiring.
 *
 * Throws when either the resolved source group or this layer's `context.output`
 * is `null`.
 *
 * Example:
 *
 * ```ts
 * inputLayer(thisLayerContext, previousLayerLike);
 * inputLayer(thisLayerContext, someGroup, methods.groupConnection.ONE_TO_ONE);
 * ```
 *
 * @param context - The layer state needed for input wiring.
 * @param from - The source layer or group.
 * @param method - Optional connection method override.
 * @param weight - Optional fixed weight to apply.
 * @returns The created connection list.
 */
export function inputLayer(
  context: LayerConnectionContext,
  from: LayerLike | Group,
  method?: unknown,
  weight?: number,
): Connection[] {
  const { output, isLayer } = context;

  // Step 1: Resolve the source group when a layer is provided.
  let sourceGroup: Group | null = from as Group;
  if (isLayer(from)) {
    sourceGroup = from.output;
  }

  // Step 2: Establish a default connection method when not provided.
  const resolvedMethod = method ?? methods.groupConnection.ALL_TO_ALL;

  // Step 3: Ensure the output group exists before connecting.
  if (!output) {
    throw new Error(LAYER_OUTPUT_INPUT_TARGET_ERROR);
  }

  if (!sourceGroup) {
    throw new Error(LAYER_OUTPUT_INPUT_SOURCE_ERROR);
  }

  // Step 4: Connect the source group to the output group.
  return sourceGroup.connect(output, resolvedMethod, weight);
}

/**
 * Disconnects nodes in this layer from a target group or node.
 *
 * This iterates through this layer's nodes and calls `node.disconnect(...)`.
 * It also updates the layer's tracked `connections.in/out` arrays so they
 * remain consistent with the underlying node graph.
 *
 * Example:
 *
 * ```ts
 * disconnectLayer(layerContext, someNode, false);
 * ```
 *
 * @param context - The layer state needed for disconnecting.
 * @param target - The group or node to disconnect.
 * @param twoSided - Whether to remove reciprocal connections as well.
 */
export function disconnectLayer(
  context: LayerConnectionContext,
  target: Group | Node,
  twoSided: boolean = false,
): void {
  const { nodes, connections } = context;

  // Step 1: Disconnect each node from the target and clean internal tracking.
  if (target instanceof Group) {
    disconnectFromGroup(nodes, target, connections, twoSided);
    return;
  }

  disconnectFromNode(nodes, target, connections, twoSided);
}

/**
 * Disconnects all layer nodes from a target group.
 *
 * This is a "cartesian disconnect": every node in this layer is disconnected
 * from every node in the target group.
 *
 * @param layerNodes - Nodes in the layer.
 * @param targetGroup - Group to disconnect from.
 * @param layerConnections - Connection tracking for the layer.
 * @param removeTwoSided - Whether to remove reciprocal connections as well.
 */
export function disconnectFromGroup(
  layerNodes: Node[],
  targetGroup: Group,
  layerConnections: LayerConnectionContext['connections'],
  removeTwoSided: boolean,
): void {
  for (
    let sourceNodeIndex = NODE_INDEX_START;
    sourceNodeIndex < layerNodes.length;
    sourceNodeIndex += LOOP_INDEX_STEP
  ) {
    const sourceNode = layerNodes[sourceNodeIndex];
    for (
      let targetNodeIndex = NODE_INDEX_START;
      targetNodeIndex < targetGroup.nodes.length;
      targetNodeIndex += LOOP_INDEX_STEP
    ) {
      const targetNode = targetGroup.nodes[targetNodeIndex];

      sourceNode.disconnect(targetNode, removeTwoSided);
      removeOutgoingConnection(layerConnections, sourceNode, targetNode);
      if (removeTwoSided) {
        removeIncomingConnection(layerConnections, targetNode, sourceNode);
      }
    }
  }
}

/**
 * Disconnects all layer nodes from a target node.
 *
 * @param layerNodes - Nodes in the layer.
 * @param targetNode - Node to disconnect from.
 * @param layerConnections - Connection tracking for the layer.
 * @param removeTwoSided - Whether to remove reciprocal connections as well.
 */
export function disconnectFromNode(
  layerNodes: Node[],
  targetNode: Node,
  layerConnections: LayerConnectionContext['connections'],
  removeTwoSided: boolean,
): void {
  for (
    let sourceNodeIndex = NODE_INDEX_START;
    sourceNodeIndex < layerNodes.length;
    sourceNodeIndex += LOOP_INDEX_STEP
  ) {
    const sourceNode = layerNodes[sourceNodeIndex];
    sourceNode.disconnect(targetNode, removeTwoSided);
    removeOutgoingConnection(layerConnections, sourceNode, targetNode);

    if (removeTwoSided) {
      removeIncomingConnection(layerConnections, targetNode, sourceNode);
    }
  }
}

/**
 * Removes an outgoing connection from layer tracking.
 *
 * This scans in reverse so we can `splice(...)` safely while iterating.
 *
 * @param layerConnections - Connection tracking for the layer.
 * @param sourceNode - Source node for the connection.
 * @param targetNode - Target node for the connection.
 */
export function removeOutgoingConnection(
  layerConnections: LayerConnectionContext['connections'],
  sourceNode: Node,
  targetNode: Node,
): void {
  for (
    let connectionIndex = layerConnections.out.length - REVERSE_INDEX_OFFSET;
    connectionIndex >= NODE_INDEX_START;
    connectionIndex -= REVERSE_INDEX_STEP
  ) {
    const connection = layerConnections.out[connectionIndex];
    if (connection.from === sourceNode && connection.to === targetNode) {
      layerConnections.out.splice(connectionIndex, 1);
      break;
    }
  }
}

/**
 * Removes an incoming connection from layer tracking.
 *
 * This scans in reverse so we can `splice(...)` safely while iterating.
 *
 * @param layerConnections - Connection tracking for the layer.
 * @param sourceNode - Source node for the connection.
 * @param targetNode - Target node for the connection.
 */
export function removeIncomingConnection(
  layerConnections: LayerConnectionContext['connections'],
  sourceNode: Node,
  targetNode: Node,
): void {
  for (
    let connectionIndex = layerConnections.in.length - REVERSE_INDEX_OFFSET;
    connectionIndex >= NODE_INDEX_START;
    connectionIndex -= REVERSE_INDEX_STEP
  ) {
    const connection = layerConnections.in[connectionIndex];
    if (connection.from === sourceNode && connection.to === targetNode) {
      layerConnections.in.splice(connectionIndex, 1);
      break;
    }
  }
}

/**
 * Clears activation state for all nodes in a layer.
 *
 * Use this when you want to reset per-node transient state between runs
 * (especially helpful in recurrent networks that keep state across timesteps).
 *
 * Example:
 *
 * ```ts
 * clearLayer(layerContext);
 * ```
 *
 * @param context - The layer state needed to reset nodes.
 */
export function clearLayer(context: LayerConnectionContext): void {
  const { nodes } = context;

  // Step 1: Reset each node's internal activation state.
  for (
    let nodeIndex = NODE_INDEX_START;
    nodeIndex < nodes.length;
    nodeIndex += LOOP_INDEX_STEP
  ) {
    nodes[nodeIndex].clear();
  }
}
