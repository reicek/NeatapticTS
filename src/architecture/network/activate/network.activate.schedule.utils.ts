import type Network from '../../network/network';
import type Node from '../../node';
import type { ActivationSchedule } from '../network.types';
import {
  INPUT_NODE_TYPE,
  OUTPUT_NODE_TYPE,
} from './network.activate.utils.types';

const DEFAULT_SCHEDULE_ITERATIONS = 1;

type ScheduleAwareRuntime = {
  _activationSchedule?: ActivationSchedule | null;
  _topoOrder?: Node[] | null;
};

/**
 * Resolve the node traversal order for one activation pass.
 *
 * The compiled activation schedule takes priority, then the legacy acyclic
 * `_topoOrder` cache, then the raw `nodes` array as a final compatibility
 * fallback for older or partially initialized runtimes.
 *
 * @param network - Target runtime network.
 * @returns Deterministic activation-node traversal list.
 */
export function resolveActivationTraversalNodes(network: Network): Node[] {
  const scheduleAwareNetwork = network as unknown as ScheduleAwareRuntime;
  const nodesByGeneId = createNodesByGeneId(network.nodes);
  const scheduledNodes = resolveNodesFromCompiledSchedule(
    scheduleAwareNetwork._activationSchedule,
    nodesByGeneId,
  );

  if (scheduledNodes) {
    return scheduledNodes;
  }

  if (
    Array.isArray(scheduleAwareNetwork._topoOrder) &&
    scheduleAwareNetwork._topoOrder.length > 0
  ) {
    return [...scheduleAwareNetwork._topoOrder];
  }

  return [...network.nodes];
}

/**
 * Resolve one input-value lookup keyed by stable input-node gene id.
 *
 * Explicit `inputNodeIds` are used when available so callers can reorder the
 * runtime `nodes` array without changing public input-vector semantics.
 *
 * @param network - Target runtime network.
 * @param inputVector - Input vector supplied by the caller.
 * @returns Stable input-value lookup for this activation pass.
 */
export function resolveInputValuesByNodeId(
  network: Network,
  inputVector: number[],
): Map<number, number> {
  const inputValuesByNodeId = new Map<number, number>();
  const nodesByGeneId = createNodesByGeneId(network.nodes);
  const hasExplicitInputRoles =
    network.inputNodeIds.length === inputVector.length &&
    network.inputNodeIds.every((nodeId) => nodesByGeneId.has(nodeId));

  if (hasExplicitInputRoles) {
    network.inputNodeIds.forEach((nodeId, inputIndex) => {
      inputValuesByNodeId.set(nodeId, inputVector[inputIndex]);
    });

    return inputValuesByNodeId;
  }

  let inputIndex = 0;
  for (const node of network.nodes) {
    if (node.type !== INPUT_NODE_TYPE) {
      continue;
    }

    inputValuesByNodeId.set(node.geneId, inputVector[inputIndex]);
    inputIndex += 1;

    if (inputIndex >= inputVector.length) {
      break;
    }
  }

  return inputValuesByNodeId;
}

/**
 * Resolve output nodes in the public output-vector order.
 *
 * Explicit `outputNodeIds` keep output readout stable even when traversal order
 * or storage order changes. Older runtimes fall back to raw output-node order.
 *
 * @param network - Target runtime network.
 * @returns Output nodes in public vector order.
 */
export function resolveOrderedOutputNodes(network: Network): Node[] {
  const scheduleAwareNetwork = network as unknown as ScheduleAwareRuntime;
  const nodesByGeneId = createNodesByGeneId(network.nodes);
  const orderedOutputNodeIds: readonly number[] = scheduleAwareNetwork
    ._activationSchedule?.outputNodeIds.length
    ? scheduleAwareNetwork._activationSchedule.outputNodeIds
    : network.outputNodeIds;
  const explicitOutputNodes = orderedOutputNodeIds.flatMap((nodeId) => {
    const outputNode = nodesByGeneId.get(nodeId);

    if (!outputNode || outputNode.type !== OUTPUT_NODE_TYPE) {
      return [];
    }

    return [outputNode];
  });

  if (explicitOutputNodes.length === network.output && network.output > 0) {
    return explicitOutputNodes;
  }

  return network.nodes.filter((node) => node.type === OUTPUT_NODE_TYPE);
}

/**
 * Resolve activation nodes from the compiled schedule when it is present and valid.
 *
 * @param activationSchedule - Cached compiled schedule.
 * @param nodesByGeneId - Stable node lookup by gene id.
 * @returns Flattened activation-node order or null when the schedule is absent or stale.
 */
function resolveNodesFromCompiledSchedule(
  activationSchedule: ActivationSchedule | null | undefined,
  nodesByGeneId: ReadonlyMap<number, Node>,
): Node[] | null {
  if (!activationSchedule || activationSchedule.steps.length === 0) {
    return null;
  }

  const scheduledNodes: Node[] = [];

  for (const step of activationSchedule.steps) {
    const iterationCount =
      step.kind === 'recurrent-component'
        ? (step.iterations ?? DEFAULT_SCHEDULE_ITERATIONS)
        : DEFAULT_SCHEDULE_ITERATIONS;

    for (
      let iterationIndex = 0;
      iterationIndex < iterationCount;
      iterationIndex++
    ) {
      for (const nodeId of step.nodeIds) {
        const scheduledNode = nodesByGeneId.get(nodeId);

        if (!scheduledNode) {
          return null;
        }

        scheduledNodes.push(scheduledNode);
      }
    }
  }

  return scheduledNodes;
}

/**
 * Create a stable node lookup by gene id.
 *
 * @param nodes - Runtime node collection.
 * @returns Stable node lookup map.
 */
function createNodesByGeneId(nodes: readonly Node[]): Map<number, Node> {
  const nodesByGeneId = new Map<number, Node>();

  for (const node of nodes) {
    nodesByGeneId.set(node.geneId, node);
  }

  return nodesByGeneId;
}
