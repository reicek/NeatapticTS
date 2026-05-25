import type Network from '../../network/network';
import type Node from '../../node';
import type { ActivationPrecision } from '../../../config';
import {
  ACTIVATION_PRECISION_F32,
  NO_OUTPUT_NODES_ERROR,
  OUTPUT_NODE_TYPE,
} from './network.standalone.utils.types';
import {
  resolveActivationTraversalNodes,
  resolveOrderedOutputNodes,
} from '../activate/network.activate.schedule.utils';
import type {
  NetworkStandaloneProps,
  NodeWithIndex,
  StandaloneGenerationContext as GenerationContext,
} from '../network.types';
import { NetworkStandaloneNoOutputNodesError } from './network.standalone.errors';

/**
 * Reinterpret the runtime `Network` instance as the internal standalone
 * property surface used by generator setup utilities.
 *
 * @param net Network instance to cast.
 * @returns Internal network properties used by the standalone generator.
 */
export function asStandaloneProps(net: Network): NetworkStandaloneProps {
  return net as unknown as NetworkStandaloneProps;
}

/**
 * Enforce the standalone precondition that at least one output node exists
 * before source generation proceeds.
 *
 * @param standaloneProps Internal standalone network view.
 * @returns Void.
 * @throws If no output node exists.
 */
export function ensureOutputNodesExist(
  standaloneProps: NetworkStandaloneProps,
): void {
  const hasOutputNode = standaloneProps.nodes.some(function hasOutput(
    node: Node,
  ) {
    return node.type === OUTPUT_NODE_TYPE;
  });

  if (!hasOutputNode) {
    throw new NetworkStandaloneNoOutputNodesError(NO_OUTPUT_NODES_ERROR);
  }
}

/**
 * Allocate a fresh emit-pass context that accumulates node indexes, cached
 * activation sources, and generated body lines.
 *
 * @param standaloneProps Internal standalone network view.
 * @returns Generation context with precision, index buffers, and emission state.
 */
export function createGenerationContext(
  standaloneProps: NetworkStandaloneProps,
): GenerationContext {
  return {
    standaloneProps,
    resolvedActivationPrecision:
      resolveStandaloneActivationPrecision(standaloneProps),
    inputNodeIndexes: [],
    activationNodeIndexes: [],
    outputNodeIndexes: [],
    emittedActivationSource: {},
    activationFunctionSources: [],
    activationFunctionIndexMap: {},
    nextActivationFunctionIndex: 0,
    initialActivations: [],
    initialStates: [],
    bodyLines: [],
  };
}

/**
 * Resolve standalone numeric precision by reconciling shared precision config
 * with the legacy `_activationPrecision` override.
 *
 * @param standaloneProps Internal standalone network view.
 * @returns Resolved activation precision for generated storage.
 */
function resolveStandaloneActivationPrecision(
  standaloneProps: NetworkStandaloneProps,
): ActivationPrecision | undefined {
  const explicitActivationPrecision = standaloneProps._activationPrecision;
  const sharedActivationPrecision =
    standaloneProps._precisionConfig?.activationPrecision;

  if (
    explicitActivationPrecision === ACTIVATION_PRECISION_F32 &&
    explicitActivationPrecision !== sharedActivationPrecision
  ) {
    return explicitActivationPrecision;
  }

  return sharedActivationPrecision ?? explicitActivationPrecision;
}

/**
 * Stamp stable per-node indexes and snapshot initial activation/state buffers
 * used by emitted standalone runtime state.
 *
 * @param generationContext Mutable generation context.
 * @returns Void.
 */
export function seedNodeIndexesAndState(
  generationContext: GenerationContext,
): void {
  const allNodes = generationContext.standaloneProps.nodes;

  for (
    let nodeTraversalIndex = 0;
    nodeTraversalIndex < allNodes.length;
    nodeTraversalIndex++
  ) {
    const currentNode = allNodes[nodeTraversalIndex] as NodeWithIndex;
    currentNode.index = nodeTraversalIndex;
    generationContext.initialActivations.push(currentNode.activation);
    generationContext.initialStates.push(currentNode.state);
  }
}

type StandaloneScheduleAwareNetwork = {
  _topoDirty?: boolean;
  _activationSchedule?: unknown;
  _topoOrder?: unknown;
  _computeTopoOrder: () => void;
};

/**
 * Resolve standalone execution metadata from the runtime activation contract.
 *
 * The standalone generator should honor the same traversal order and public
 * input/output role order that runtime activation uses. That matters for delay
 * lines such as NARX memory blocks, where raw node storage order can differ
 * from the dependency order used during activation.
 *
 * @param network - Runtime network being snapshotted.
 * @param generationContext - Mutable generation context receiving index metadata.
 * @returns Void.
 */
export function resolveStandaloneExecutionMetadata(
  network: Network,
  generationContext: GenerationContext,
): void {
  const scheduleAwareNetwork =
    network as unknown as StandaloneScheduleAwareNetwork;

  // Step 1: Refresh activation scheduling before reading traversal helpers.
  if (
    scheduleAwareNetwork._topoDirty ||
    (!scheduleAwareNetwork._activationSchedule &&
      !scheduleAwareNetwork._topoOrder)
  ) {
    scheduleAwareNetwork._computeTopoOrder();
  }

  // Step 2: Resolve the authoritative runtime execution order.
  generationContext.inputNodeIndexes = resolveInputNodeIndexes(network);
  generationContext.activationNodeIndexes = resolveActivationTraversalNodes(
    network,
  ).flatMap((node) => {
    if (node.type === 'input') {
      return [];
    }

    const indexedNode = node as Partial<NodeWithIndex>;
    return typeof indexedNode.index === 'number' ? [indexedNode.index] : [];
  });
  generationContext.outputNodeIndexes = resolveOrderedOutputNodes(
    network,
  ).flatMap((node) => {
    const indexedNode = node as Partial<NodeWithIndex>;
    return typeof indexedNode.index === 'number' ? [indexedNode.index] : [];
  });
}

/**
 * Resolve input-node indexes in public input-vector order.
 *
 * @param network - Runtime network being snapshotted.
 * @returns Input-node indexes used when seeding generated activation buffers.
 */
function resolveInputNodeIndexes(network: Network): number[] {
  const nodesByGeneId = new Map(
    network.nodes.map((node) => [node.geneId, node] as const),
  );
  const explicitInputNodeIndexes = network.inputNodeIds.flatMap((nodeId) => {
    const node = nodesByGeneId.get(nodeId) as
      | Partial<NodeWithIndex>
      | undefined;
    return typeof node?.index === 'number' ? [node.index] : [];
  });

  if (explicitInputNodeIndexes.length === network.input && network.input > 0) {
    return explicitInputNodeIndexes;
  }

  return network.nodes.flatMap((node) => {
    if (node.type !== 'input') {
      return [];
    }

    const indexedNode = node as Partial<NodeWithIndex>;
    return typeof indexedNode.index === 'number' ? [indexedNode.index] : [];
  });
}
