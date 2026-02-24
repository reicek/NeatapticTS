import {
  INITIAL_OUTPUT_WRITE_INDEX,
  INPUT_NODE_TYPE,
  OUTPUT_WRITE_INDEX_INCREMENT,
  OUTPUT_NODE_TYPE,
  type NoTraceNodeTraversalContext,
  type SingleNodeNoTraceActivationContext,
} from './network.activate.utils.types';

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
  let outputWriteIndex = INITIAL_OUTPUT_WRITE_INDEX;

  traversalContext.networkNodes.forEach(
    function processNodeAtIndex(networkNode, nodeIndex): void {
      outputWriteIndex = activateSingleNodeWithoutTrace({
        networkNode,
        nodeIndex,
        inputVector: traversalContext.inputVector,
        pooledOutputBuffer: traversalContext.pooledOutputBuffer,
        outputWriteIndex,
      });
    },
  );
}

/**
 * Activate one node and return the next output write index.
 *
 * @param activationContext - Node-specific activation state.
 * @returns Updated output write index.
 */
function activateSingleNodeWithoutTrace(
  activationContext: SingleNodeNoTraceActivationContext,
): number {
  if (isInputNode(activationContext.networkNode)) {
    activateInputNode(activationContext);
    return activationContext.outputWriteIndex;
  }

  if (isOutputNode(activationContext.networkNode)) {
    return activateOutputNodeAndAdvanceIndex(activationContext);
  }

  activateHiddenNode(activationContext.networkNode);
  return activationContext.outputWriteIndex;
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
    activationContext.inputVector[activationContext.nodeIndex],
  );
}

/**
 * Activate an output node, write the activation value, and advance the output index.
 *
 * @param activationContext - Node-specific activation state.
 * @returns Next output write index.
 */
function activateOutputNodeAndAdvanceIndex(
  activationContext: SingleNodeNoTraceActivationContext,
): number {
  activationContext.pooledOutputBuffer[activationContext.outputWriteIndex] =
    activationContext.networkNode.noTraceActivate();
  return activationContext.outputWriteIndex + OUTPUT_WRITE_INDEX_INCREMENT;
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
