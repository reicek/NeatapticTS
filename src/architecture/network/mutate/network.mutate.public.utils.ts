import { config } from '../../../config';
import Node from '../../node/node';
import { acquireNode } from '../../nodePool/nodePool';
import type Network from '../network';
import type { NetworkMutationProps } from '../network.types';

/**
 * Public structural-mutation helpers that stay outside the mutation dispatch table.
 *
 * This file owns small graph-editing methods that are exposed directly on
 * `Network` for callers who want one specific structural operation without
 * going through the broader mutation-method dispatch flow.
 */

/**
 * Split one randomly selected connection by inserting a hidden node.
 *
 * This preserves the long-standing public `addNodeBetween()` behavior:
 * - it does not opt into `ADD_NODE` deterministic-chain policy,
 * - it preserves the original source-edge weight on the first new connection,
 * - it uses `1` for the hidden-to-target edge to keep the split easy to reason about.
 *
 * @param this Target network instance.
 * @returns Nothing.
 */
export function addNodeBetweenImpl(this: Network): void {
  const mutationProps = this as unknown as NetworkMutationProps;
  if (this.connections.length === 0) {
    return;
  }

  const selectedConnectionIndex = Math.floor(
    mutationProps._rand() * this.connections.length,
  );
  const selectedConnection = this.connections[selectedConnectionIndex];
  if (!selectedConnection) {
    return;
  }

  // Step 1: Remove the original edge so the new hidden node fully owns the split path.
  this.disconnect(selectedConnection.from, selectedConnection.to);

  // Step 2: Create the replacement hidden node using the same pooling policy as the runtime.
  const insertedNode = config.enableNodePooling
    ? acquireNode({ type: 'hidden', rng: mutationProps._rand })
    : new Node('hidden', undefined, mutationProps._rand);
  this.nodes.push(insertedNode);

  // Step 3: Rebuild the path while preserving the original edge weight contract.
  this.connect(
    selectedConnection.from,
    insertedNode,
    selectedConnection.weight,
  );
  this.connect(insertedNode, selectedConnection.to, 1);

  // Step 4: Mark topology-derived caches dirty so later activation paths rebuild coherent state.
  mutationProps._topoDirty = true;
  mutationProps._nodeIndexDirty = true;
}
