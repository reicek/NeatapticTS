import type {
  ActivationSchedule,
  ActivationSchedulingDiagnostics,
  ActivationScheduleStep,
  TopologyBuildContext,
  TopologyNetwork,
  TopologyNode,
  TopologyNetworkProps,
} from './network.topology.utils.types';
import {
  IN_DEGREE_DECREMENT,
  ZERO_COUNT,
} from './network.topology.utils.types';
import {
  resolveStableNodeTieBreakValue,
  sortNodesByStableTieBreak,
} from './network.topology.loop.utils';

/**
 * Cast network to internal topology props view.
 *
 * @param network Network instance.
 * @returns Internal topology props view.
 */
export function asTopologyProps(
  network: TopologyNetwork,
): TopologyNetworkProps {
  return network as unknown as TopologyNetworkProps;
}

/**
 * Determine whether recurrent scheduling should be used.
 *
 * @param internalTopologyProps Internal topology props view.
 * @returns True when acyclic mode is disabled.
 */
export function shouldBuildRecurrentSchedule(
  internalTopologyProps: TopologyNetworkProps,
): boolean {
  return !internalTopologyProps._enforceAcyclic;
}

/**
 * Build and cache the deterministic recurrent schedule.
 *
 * @param network Network instance.
 * @param internalTopologyProps Internal topology props view.
 * @returns Void.
 */
export function finalizeRecurrentSchedule(
  network: TopologyNetwork,
  internalTopologyProps: TopologyNetworkProps,
): void {
  internalTopologyProps._activationSchedule =
    resolveRecurrentActivationSchedule(network);
  internalTopologyProps._activationSchedulingDiagnostics =
    resolveCompiledSchedulingDiagnostics(
      network,
      internalTopologyProps._activationSchedule,
    );
  internalTopologyProps._topoOrder = null;
  internalTopologyProps._topoDirty = false;
}

/**
 * Clear cached topological order state.
 *
 * @param internalTopologyProps Internal topology props view.
 * @returns Void.
 */
export function clearCachedTopoOrder(
  internalTopologyProps: TopologyNetworkProps,
): void {
  internalTopologyProps._activationSchedulingDiagnostics = null;
  internalTopologyProps._activationSchedule = null;
  internalTopologyProps._topoOrder = null;
  internalTopologyProps._topoDirty = false;
}

/**
 * Create mutable build context for Kahn traversal.
 *
 * @param network Network instance.
 * @param internalTopologyProps Internal topology props view.
 * @returns Initialized build context.
 */
export function createTopologyBuildContext(
  network: TopologyNetwork,
  internalTopologyProps: TopologyNetworkProps,
): TopologyBuildContext {
  return {
    network,
    internalTopologyProps,
    inDegreeByNode: new Map(),
    processingQueue: [],
    activationSteps: [],
    topoOrder: [],
  };
}

/**
 * Initialize all nodes with zero in-degree.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
export function initializeAllNodeInDegreeCounts(
  buildContext: TopologyBuildContext,
): void {
  for (const node of buildContext.network.nodes) {
    buildContext.inDegreeByNode.set(node, ZERO_COUNT);
  }
}

/**
 * Apply in-degree increments from non-self connections.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
export function applyIncomingEdgeCounts(
  buildContext: TopologyBuildContext,
): void {
  for (const connection of buildContext.network.connections) {
    if (isSelfConnection(connection.from, connection.to)) {
      continue;
    }

    incrementNodeInDegree(buildContext, connection.to);
  }
}

/**
 * Finalize cached order, falling back to raw node order on cycle detection.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
export function finalizeTopoOrder(buildContext: TopologyBuildContext): void {
  buildContext.internalTopologyProps._activationSchedule =
    resolveFinalActivationSchedule(buildContext);
  buildContext.internalTopologyProps._topoOrder =
    resolveFinalOrder(buildContext);
  buildContext.internalTopologyProps._activationSchedulingDiagnostics =
    resolveFinalSchedulingDiagnostics(buildContext);
  buildContext.internalTopologyProps._topoDirty = false;
}

/**
 * Resolve the final deterministic activation schedule when the graph is acyclic.
 *
 * @param buildContext Mutable build context.
 * @returns Cached activation schedule or null when a complete acyclic order was not found.
 */
function resolveFinalActivationSchedule(
  buildContext: TopologyBuildContext,
): ActivationSchedule | null {
  const isCompleteOrder =
    buildContext.topoOrder.length === buildContext.network.nodes.length;

  if (!isCompleteOrder) {
    return null;
  }

  return {
    mode: 'acyclic',
    steps: buildContext.activationSteps.map((activationStep) => ({
      kind: 'wave',
      nodeIds: [...activationStep],
    })),
    outputNodeIds: buildContext.network.outputNodeIds,
  };
}

/**
 * Resolve final human-friendly scheduling diagnostics for acyclic mode.
 *
 * @param buildContext Mutable build context.
 * @returns Scheduling diagnostics snapshot.
 */
function resolveFinalSchedulingDiagnostics(
  buildContext: TopologyBuildContext,
): ActivationSchedulingDiagnostics {
  const activationSchedule =
    buildContext.internalTopologyProps._activationSchedule;

  if (activationSchedule) {
    return resolveCompiledSchedulingDiagnostics(
      buildContext.network,
      activationSchedule,
    );
  }

  const cycleNodeIds = resolveCycleNodeIds(buildContext);

  return {
    topologyIntent: buildContext.network.getTopologyIntent(),
    requestedMode: 'acyclic',
    topologyDirty: false,
    executionPath: 'cycle-fallback-order',
    issue: 'cycle-detected',
    message:
      'Acyclic scheduling detected a cycle, so activation falls back to raw node order until the cycle is removed or recurrent mode is enabled.',
    inputNodeIds: buildContext.network.inputNodeIds,
    outputNodeIds: buildContext.network.outputNodeIds,
    stepCount: ZERO_COUNT,
    recurrentComponentCount: ZERO_COUNT,
    stateSemantics: null,
    cycleNodeIds,
    suggestions: [
      'Remove the reported cycle or back-connection if this network should stay feed-forward.',
      'If recurrent behavior is intentional, switch the topology intent to unconstrained or disable acyclic enforcement.',
    ],
  };
}

/**
 * Resolve the deterministic recurrent activation schedule.
 *
 * The schedule is based on the SCC condensation graph so recurrent structure is
 * explicit before activation-path integration consumes it.
 *
 * @param network Network instance.
 * @returns Deterministic recurrent activation schedule.
 */
function resolveRecurrentActivationSchedule(
  network: TopologyNetwork,
): ActivationSchedule {
  const stronglyConnectedComponents = collectStronglyConnectedComponents(
    network.nodes,
  );
  const componentIndexByNode = createComponentIndexByNode(
    stronglyConnectedComponents,
  );
  const condensationContext = createCondensationContext(
    network,
    stronglyConnectedComponents,
    componentIndexByNode,
  );
  const steps = buildRecurrentScheduleSteps(
    stronglyConnectedComponents,
    condensationContext,
  );

  return {
    mode: 'recurrent',
    steps,
    outputNodeIds: network.outputNodeIds,
    stateSemantics: 'carry',
  };
}

/**
 * Resolve the standard diagnostics payload for a compiled schedule.
 *
 * @param network Network instance.
 * @param activationSchedule Compiled activation schedule.
 * @returns Scheduling diagnostics snapshot.
 */
function resolveCompiledSchedulingDiagnostics(
  network: TopologyNetwork,
  activationSchedule: ActivationSchedule,
): ActivationSchedulingDiagnostics {
  const recurrentComponentCount = activationSchedule.steps.filter(
    (activationStep) => activationStep.kind === 'recurrent-component',
  ).length;

  return {
    topologyIntent: network.getTopologyIntent(),
    requestedMode: activationSchedule.mode,
    topologyDirty: false,
    executionPath: 'compiled-schedule',
    issue: null,
    message:
      activationSchedule.mode === 'acyclic'
        ? 'Activation is using the compiled acyclic schedule with stable wave ordering.'
        : 'Activation is using the compiled recurrent schedule with explicit recurrent-component steps and carried recurrent state.',
    inputNodeIds: network.inputNodeIds,
    outputNodeIds: activationSchedule.outputNodeIds,
    stepCount: activationSchedule.steps.length,
    recurrentComponentCount,
    stateSemantics: activationSchedule.stateSemantics ?? null,
    cycleNodeIds: [],
    suggestions:
      activationSchedule.mode === 'recurrent'
        ? [
            'Call clear() before a new independent sequence when carried recurrent state should reset.',
          ]
        : [],
  };
}

/**
 * Resolve stable node ids that remained unscheduled after acyclic traversal.
 *
 * @param buildContext Mutable build context.
 * @returns Stable node ids implicated in the cycle fallback.
 */
function resolveCycleNodeIds(buildContext: TopologyBuildContext): number[] {
  const scheduledNodes = new Set(buildContext.topoOrder);
  const unscheduledNodes = buildContext.network.nodes.filter(
    (node) => !scheduledNodes.has(node),
  );

  return sortNodesByStableTieBreak(unscheduledNodes).map((node) => node.geneId);
}

/**
 * Collect strongly-connected components using Tarjan traversal.
 *
 * @param nodes Candidate graph nodes.
 * @returns Stable SCC list.
 */
function collectStronglyConnectedComponents(
  nodes: readonly TopologyNode[],
): TopologyNode[][] {
  const sortedNodes = sortNodesByStableTieBreak([...nodes]);
  const indexByNode = new Map<TopologyNode, number>();
  const lowLinkByNode = new Map<TopologyNode, number>();
  const nodeStack: TopologyNode[] = [];
  const stackedNodes = new Set<TopologyNode>();
  const stronglyConnectedComponents: TopologyNode[][] = [];
  let currentIndex = ZERO_COUNT;

  for (const node of sortedNodes) {
    if (indexByNode.has(node)) {
      continue;
    }

    visitNode(node);
  }

  return stronglyConnectedComponents;

  function visitNode(node: TopologyNode): void {
    indexByNode.set(node, currentIndex);
    lowLinkByNode.set(node, currentIndex);
    currentIndex += IN_DEGREE_DECREMENT;
    nodeStack.push(node);
    stackedNodes.add(node);

    for (const neighbor of resolveOutgoingNeighbors(node)) {
      if (!indexByNode.has(neighbor)) {
        visitNode(neighbor);
        lowLinkByNode.set(
          node,
          Math.min(lowLinkByNode.get(node)!, lowLinkByNode.get(neighbor)!),
        );
        continue;
      }

      if (!stackedNodes.has(neighbor)) {
        continue;
      }

      lowLinkByNode.set(
        node,
        Math.min(lowLinkByNode.get(node)!, indexByNode.get(neighbor)!),
      );
    }

    const nodeLowLink = lowLinkByNode.get(node);
    const nodeIndex = indexByNode.get(node);
    if (nodeLowLink !== nodeIndex) {
      return;
    }

    const componentNodes: TopologyNode[] = [];
    while (nodeStack.length > ZERO_COUNT) {
      const stackedNode = nodeStack.pop()!;
      stackedNodes.delete(stackedNode);
      componentNodes.push(stackedNode);
      if (stackedNode === node) {
        break;
      }
    }

    stronglyConnectedComponents.push(sortNodesByStableTieBreak(componentNodes));
  }
}

/**
 * Resolve one node's outgoing neighbors for SCC traversal.
 *
 * Self-loops are excluded from traversal because they do not change SCC
 * membership, but singleton self-loops are still classified as recurrent later.
 *
 * @param node Candidate node.
 * @returns Deterministic outgoing neighbors.
 */
function resolveOutgoingNeighbors(node: TopologyNode): TopologyNode[] {
  const outgoingNeighbors = node.connections.out
    .map((connection) => connection.to)
    .filter((neighbor) => neighbor !== node);

  return sortNodesByStableTieBreak(outgoingNeighbors);
}

/**
 * Build a reverse lookup from node to SCC index.
 *
 * @param stronglyConnectedComponents Stable SCC list.
 * @returns Node-to-component lookup map.
 */
function createComponentIndexByNode(
  stronglyConnectedComponents: readonly TopologyNode[][],
): Map<TopologyNode, number> {
  const componentIndexByNode = new Map<TopologyNode, number>();

  stronglyConnectedComponents.forEach((componentNodes, componentIndex) => {
    for (const node of componentNodes) {
      componentIndexByNode.set(node, componentIndex);
    }
  });

  return componentIndexByNode;
}

interface CondensationContext {
  componentInDegree: number[];
  outgoingComponentsByIndex: Array<Set<number>>;
}

/**
 * Build the SCC condensation graph.
 *
 * @param network Network instance.
 * @param stronglyConnectedComponents Stable SCC list.
 * @param componentIndexByNode Node-to-component lookup.
 * @returns Condensation graph context.
 */
function createCondensationContext(
  network: TopologyNetwork,
  stronglyConnectedComponents: readonly TopologyNode[][],
  componentIndexByNode: ReadonlyMap<TopologyNode, number>,
): CondensationContext {
  const componentInDegree = stronglyConnectedComponents.map(() => ZERO_COUNT);
  const outgoingComponentsByIndex = stronglyConnectedComponents.map(
    () => new Set<number>(),
  );

  for (const connection of network.connections) {
    const fromComponentIndex = componentIndexByNode.get(connection.from);
    const toComponentIndex = componentIndexByNode.get(connection.to);

    if (
      typeof fromComponentIndex !== 'number' ||
      typeof toComponentIndex !== 'number' ||
      fromComponentIndex === toComponentIndex
    ) {
      continue;
    }

    const outgoingComponents = outgoingComponentsByIndex[fromComponentIndex];
    if (outgoingComponents.has(toComponentIndex)) {
      continue;
    }

    outgoingComponents.add(toComponentIndex);
    componentInDegree[toComponentIndex] += IN_DEGREE_DECREMENT;
  }

  return {
    componentInDegree,
    outgoingComponentsByIndex,
  };
}

/**
 * Build deterministic recurrent schedule steps from the condensation graph.
 *
 * @param stronglyConnectedComponents Stable SCC list.
 * @param condensationContext Condensation graph context.
 * @returns Structured recurrent schedule steps.
 */
function buildRecurrentScheduleSteps(
  stronglyConnectedComponents: readonly TopologyNode[][],
  condensationContext: CondensationContext,
): ActivationScheduleStep[] {
  const steps: ActivationScheduleStep[] = [];
  const queuedComponentIndexes = new Set<number>();
  const processedComponentIndexes = new Set<number>();
  let processingQueue = seedCondensationQueue(
    stronglyConnectedComponents,
    condensationContext.componentInDegree,
    queuedComponentIndexes,
  );

  while (processingQueue.length > ZERO_COUNT) {
    const currentWave = processingQueue.splice(0, processingQueue.length);
    const nextQueue: number[] = [];
    let currentWaveNodeIds: number[] = [];

    for (const componentIndex of currentWave) {
      processedComponentIndexes.add(componentIndex);
      const componentNodes = stronglyConnectedComponents[componentIndex];
      const componentNodeIds = componentNodes.map((node) => node.geneId);

      if (isRecurrentComponent(componentNodes)) {
        if (currentWaveNodeIds.length > ZERO_COUNT) {
          steps.push({
            kind: 'wave',
            nodeIds: currentWaveNodeIds,
          });
          currentWaveNodeIds = [];
        }

        steps.push({
          kind: 'recurrent-component',
          nodeIds: componentNodeIds,
          iterations: 1,
        });
      } else {
        currentWaveNodeIds = [...currentWaveNodeIds, ...componentNodeIds];
      }

      for (const nextComponentIndex of condensationContext
        .outgoingComponentsByIndex[componentIndex]) {
        condensationContext.componentInDegree[nextComponentIndex] -=
          IN_DEGREE_DECREMENT;

        if (
          condensationContext.componentInDegree[nextComponentIndex] ===
            ZERO_COUNT &&
          !queuedComponentIndexes.has(nextComponentIndex) &&
          !processedComponentIndexes.has(nextComponentIndex)
        ) {
          nextQueue.push(nextComponentIndex);
          queuedComponentIndexes.add(nextComponentIndex);
        }
      }
    }

    if (currentWaveNodeIds.length > ZERO_COUNT) {
      steps.push({
        kind: 'wave',
        nodeIds: currentWaveNodeIds,
      });
    }

    processingQueue = sortComponentIndexesByTieBreak(
      nextQueue,
      stronglyConnectedComponents,
    );
  }

  return steps;
}

/**
 * Seed the condensation queue with zero-indegree or input-owning components.
 *
 * @param stronglyConnectedComponents Stable SCC list.
 * @param componentInDegree Component indegree counts.
 * @param queuedComponentIndexes Mutable set of already queued components.
 * @returns Initial deterministic queue.
 */
function seedCondensationQueue(
  stronglyConnectedComponents: readonly TopologyNode[][],
  componentInDegree: readonly number[],
  queuedComponentIndexes: Set<number>,
): number[] {
  const seededComponentIndexes = stronglyConnectedComponents.flatMap(
    (componentNodes, componentIndex) => {
      const shouldSeed =
        componentInDegree[componentIndex] === ZERO_COUNT ||
        componentNodes.some((node) => node.type === 'input');

      if (!shouldSeed) {
        return [];
      }

      queuedComponentIndexes.add(componentIndex);
      return [componentIndex];
    },
  );

  return sortComponentIndexesByTieBreak(
    seededComponentIndexes,
    stronglyConnectedComponents,
  );
}

/**
 * Sort component indexes by the deterministic node tie-break of each SCC root.
 *
 * @param componentIndexes Candidate component indexes.
 * @param stronglyConnectedComponents Stable SCC list.
 * @returns Sorted component indexes.
 */
function sortComponentIndexesByTieBreak(
  componentIndexes: readonly number[],
  stronglyConnectedComponents: readonly TopologyNode[][],
): number[] {
  return [...componentIndexes].toSorted(
    (leftComponentIndex, rightComponentIndex) =>
      resolveComponentTieBreakValue(
        stronglyConnectedComponents[leftComponentIndex],
      ) -
      resolveComponentTieBreakValue(
        stronglyConnectedComponents[rightComponentIndex],
      ),
  );
}

/**
 * Resolve one SCC tie-break value from its first stable node.
 *
 * @param componentNodes Stable SCC node list.
 * @returns Deterministic component sort scalar.
 */
function resolveComponentTieBreakValue(
  componentNodes: readonly TopologyNode[],
): number {
  const firstNode = componentNodes[0]!;
  return resolveStableNodeTieBreakValue(firstNode);
}

/**
 * Check whether one SCC should be treated as a recurrent execution boundary.
 *
 * @param componentNodes Stable SCC node list.
 * @returns True when the component is cyclic or carries a self-loop.
 */
function isRecurrentComponent(
  componentNodes: readonly TopologyNode[],
): boolean {
  if (componentNodes.length > IN_DEGREE_DECREMENT) {
    return true;
  }

  return componentNodes.some(
    (node) => node.connections.self.length > ZERO_COUNT,
  );
}

/**
 * Resolve final topological order with cycle fallback.
 *
 * @param buildContext Mutable build context.
 * @returns Fully valid topological order or raw node order fallback.
 */
function resolveFinalOrder(buildContext: TopologyBuildContext) {
  const isCompleteOrder =
    buildContext.topoOrder.length === buildContext.network.nodes.length;

  if (isCompleteOrder) {
    return buildContext.topoOrder;
  }

  return [...buildContext.network.nodes];
}

/**
 * Test whether a connection is a self-loop.
 *
 * @param from Source node.
 * @param to Target node.
 * @returns True when source and target are the same node.
 */
function isSelfConnection(
  from: TopologyBuildContext['network']['nodes'][number],
  to: TopologyBuildContext['network']['nodes'][number],
): boolean {
  return from === to;
}

/**
 * Increment in-degree for a node in the tally map.
 *
 * @param buildContext Mutable build context.
 * @param node Target node.
 * @returns Void.
 */
function incrementNodeInDegree(
  buildContext: TopologyBuildContext,
  node: TopologyBuildContext['network']['nodes'][number],
): void {
  const currentInDegree = buildContext.inDegreeByNode.get(node) ?? ZERO_COUNT;
  buildContext.inDegreeByNode.set(node, currentInDegree + IN_DEGREE_DECREMENT);
}
