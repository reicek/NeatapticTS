import type { LayerPropagationContext } from './layer.utils.types';

const SIZE_MISMATCH_ERROR_MESSAGE =
  'Array with values should be same as the amount of nodes!';
const REVERSE_INDEX_OFFSET = 1;
const PROPAGATE_NODE_INDEX_FLOOR = 0;
const REVERSE_INDEX_STEP = 1;
const PROPAGATE_TRAVERSE = true;
const PROPAGATE_PROJECTED_ERROR = 0;

/**
 * Ensures target values align with the node count.
 *
 * In backpropagation, a `targets` array is only meaningful for layers that are
 * acting as an output layer (supervised training). Hidden layers typically
 * propagate without explicit targets.
 *
 * Example:
 *
 * ```ts
 * assertTargetInputSize(2, [1, 0]);
 * ```
 *
 * @param nodeCount - Number of nodes in the layer.
 * @param inputTargets - Optional target values provided by the caller.
 */
export function assertTargetInputSize(
  nodeCount: number,
  inputTargets?: number[],
): void {
  if (inputTargets !== undefined && inputTargets.length !== nodeCount) {
    throw new Error(SIZE_MISMATCH_ERROR_MESSAGE);
  }
}

/**
 * Propagates errors through all nodes in reverse order.
 *
 * Reverse iteration matches the historical ordering used by Neataptic-style
 * implementations, and can matter when a node's propagation uses state that is
 * mutated as you traverse.
 *
 * Target handling:
 * - When `targets` is omitted, each node propagates based on its accumulated
 *   error from downstream connections (hidden layer behavior).
 * - When `targets` is provided, each node receives a corresponding target value
 *   (output layer behavior).
 *
 * Examples:
 *
 * ```ts
 * // Hidden layer:
 * propagateNodesInReverse({ nodes }, 0.3, 0.1);
 *
 * // Output layer:
 * propagateNodesInReverse({ nodes }, 0.3, 0.1, [1, 0, 0]);
 * ```
 *
 * @param context - The layer state needed for propagation.
 * @param rate - The learning rate for weight updates.
 * @param momentum - The momentum factor for smoothing updates.
 * @param targets - Optional target values for output layers.
 */
export function propagateNodesInReverse(
  context: LayerPropagationContext,
  rate: number,
  momentum: number,
  targets?: number[],
): void {
  const { nodes } = context;

  for (
    let nodeIndex = nodes.length - REVERSE_INDEX_OFFSET;
    nodeIndex >= PROPAGATE_NODE_INDEX_FLOOR;
    nodeIndex -= REVERSE_INDEX_STEP
  ) {
    const node = nodes[nodeIndex];

    if (targets === undefined) {
      node.propagate(
        rate,
        momentum,
        PROPAGATE_TRAVERSE,
        PROPAGATE_PROJECTED_ERROR,
      );
    } else {
      node.propagate(
        rate,
        momentum,
        PROPAGATE_TRAVERSE,
        PROPAGATE_PROJECTED_ERROR,
        targets[nodeIndex],
      );
    }
  }
}
