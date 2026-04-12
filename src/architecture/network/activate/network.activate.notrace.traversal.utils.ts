import {
  INPUT_NODE_TYPE,
  OUTPUT_NODE_TYPE,
  type NoTraceNodeTraversalContext,
  type SingleNodeNoTraceActivationContext,
} from './network.activate.utils.types';
import {
  resolveActivationTraversalNodes,
  resolveInputValuesByNodeId,
  resolveOrderedOutputNodes,
} from './network.activate.schedule.utils';

/**
 * Traverse nodes in activation order and write output activations into pooled storage.
 *
 * This helper isolates traversal concerns from no-trace orchestration so the main flow
 * can remain focused on high-level activation phases.
 *
 * @param traversalContext - Inputs required to process each node and collect outputs.
 * @returns Nothing.
 */
export function populatePooledOutputBufferFromNodes(
  traversalContext: NoTraceNodeTraversalContext,
): void {
  const activationNodes = resolveActivationTraversalNodes(
    traversalContext.network,
  );
  const inputValuesByNodeId = resolveInputValuesByNodeId(
    traversalContext.network,
    traversalContext.inputVector,
  );
  const orderedOutputNodes = resolveOrderedOutputNodes(traversalContext.network);

  activationNodes.forEach(function processNode(networkNode): void {
    activateSingleNodeWithoutTrace({
      inputValuesByNodeId,
      networkNode,
    });
  });

  orderedOutputNodes.forEach(function writeOutputNodeActivation(
    outputNode,
    outputIndex,
  ): void {
    traversalContext.pooledOutputBuffer[outputIndex] = outputNode.activation;
  });
}

/**
 * Activate one node and return the next output write index.
 *
 * @param activationContext - Node-specific activation state.
 * @returns Updated output write index.
 */
function activateSingleNodeWithoutTrace(
  activationContext: SingleNodeNoTraceActivationContext,
): void {
  if (isInputNode(activationContext.networkNode)) {
    activateInputNode(activationContext);
    return;
  }

  if (isOutputNode(activationContext.networkNode)) {
    activationContext.networkNode.noTraceActivate();
    return;
  }

  activateHiddenNode(activationContext.networkNode);
}

/**
 * Determine whether a node is an input-role node.
 *
 * @param networkNode - Node under traversal.
 * @returns True when node role is input.
 */
function isInputNode(
  networkNode: SingleNodeNoTraceActivationContext['networkNode'],
): boolean {
  return networkNode.type === INPUT_NODE_TYPE;
}

/**
 * Determine whether a node is an output-role node.
 *
 * @param networkNode - Node under traversal.
 * @returns True when node role is output.
 */
function isOutputNode(
  networkNode: SingleNodeNoTraceActivationContext['networkNode'],
): boolean {
  return networkNode.type === OUTPUT_NODE_TYPE;
}

/**
 * Activate an input node using the matching input vector value.
 *
 * @param activationContext - Node-specific activation state.
 * @returns Nothing.
 */
function activateInputNode(
  activationContext: SingleNodeNoTraceActivationContext,
): void {
  activationContext.networkNode.noTraceActivate(
    activationContext.inputValuesByNodeId.get(
      activationContext.networkNode.geneId,
    ),
  );
}

/**
 * Activate a hidden node without trace bookkeeping.
 *
 * @param networkNode - Hidden-role node to activate.
 * @returns Nothing.
 */
function activateHiddenNode(
  networkNode: SingleNodeNoTraceActivationContext['networkNode'],
): void {
  networkNode.noTraceActivate();
}
