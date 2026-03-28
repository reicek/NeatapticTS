import type { EngineState } from './engineState.types';
import type { NetworkConnection, NetworkNode } from './evolutionEngine.types';

/**
 * Collect node indices of one requested type into the shared engine scratch buffer.
 *
 * This helper keeps the facade free of buffer-growth details while preserving
 * the existing allocation-light behaviour used by the evolution loop.
 *
 * @param state - Shared engine state that owns the reusable node-index buffer.
 * @param nodes - Candidate nodes to scan.
 * @param type - Node type to collect, such as `input`, `hidden`, or `output`.
 * @returns Number of matching indices written into the shared scratch buffer.
 *
 * @example
 * const outputCount = collectEvolutionEngineNodeIndicesByType(state, nodes, 'output');
 */
export const collectEvolutionEngineNodeIndicesByType = (
  state: EngineState,
  nodes: NetworkNode[] | undefined,
  type: string,
): number => {
  if (!Array.isArray(nodes) || nodes.length === 0) {
    return 0;
  }

  let writeCount = 0;
  let scratchBuffer = state.scratch.nodeIndexBuffer;

  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    const nodeReference = nodes[nodeIndex];
    if (!nodeReference || nodeReference.type !== type) {
      continue;
    }

    if (writeCount >= scratchBuffer.length) {
      const nextCapacity = 1 << Math.ceil(Math.log2(writeCount + 1));
      const grownBuffer = new Int32Array(nextCapacity);
      grownBuffer.set(scratchBuffer);
      state.scratch.nodeIndexBuffer = grownBuffer;
      scratchBuffer = grownBuffer;
    }

    scratchBuffer[writeCount++] = nodeIndex;
  }

  return writeCount;
};

/**
 * Collect enabled outgoing connections from one hidden node into the shared scratch array.
 *
 * The evolution loop reuses this helper during network inspection to avoid
 * allocating transient arrays while still keeping the public facade compact.
 *
 * @param state - Shared engine state that owns the reusable scratch arrays.
 * @param hiddenNode - Hidden node whose outgoing edges should be inspected.
 * @param nodes - Full node list aligned with the scratch index buffer.
 * @param outputCount - Number of output-node indices already staged in scratch.
 * @returns Shared scratch array containing enabled hidden-to-output connections.
 *
 * @example
 * const connections = collectEvolutionEngineHiddenToOutputConnections(
 *   state,
 *   hiddenNode,
 *   nodes,
 *   outputCount,
 * );
 */
export const collectEvolutionEngineHiddenToOutputConnections = (
  state: EngineState,
  hiddenNode: NetworkNode,
  nodes: NetworkNode[],
  outputCount: number,
): NetworkConnection[] => {
  if (!hiddenNode?.connections || !Array.isArray(nodes) || outputCount <= 0) {
    return [];
  }

  const maxScratchCount = state.scratch.nodeIndexBuffer.length;
  const effectiveOutputCount = Math.min(
    outputCount | 0,
    maxScratchCount,
    nodes.length,
  );

  if (effectiveOutputCount <= 0) {
    return [];
  }

  const hiddenToOutputConnections = state.scratch.hiddenToOutputConnections;
  hiddenToOutputConnections.length = 0;

  const outgoingConnections = hiddenNode.connections.out ?? [];
  for (
    let connectionIndex = 0;
    connectionIndex < outgoingConnections.length;
    connectionIndex++
  ) {
    const candidateConnection = outgoingConnections[
      connectionIndex
    ] as unknown as NetworkConnection;

    if (!candidateConnection || candidateConnection.enabled === false) {
      continue;
    }

    for (
      let outputIndex = 0;
      outputIndex < effectiveOutputCount;
      outputIndex++
    ) {
      const targetNodeIndex = state.scratch.nodeIndexBuffer[outputIndex];
      const targetNode = nodes[targetNodeIndex];
      if (candidateConnection.to === targetNode) {
        hiddenToOutputConnections.push(candidateConnection);
        break;
      }
    }
  }

  return hiddenToOutputConnections;
};
