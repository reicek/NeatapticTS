import { activationArrayPool } from '../activationArrayPool/activationArrayPool';
import type { LayerActivationContext } from './layer.utils.types';
import { LayerSizeMismatchError } from './layer.errors';

const SIZE_MISMATCH_ERROR_MESSAGE =
  'Array with values should be same as the amount of nodes!';
const DROPOUT_DISABLED_THRESHOLD = 0;
const MASK_ENABLED = 1;
const MASK_DISABLED = 0;
const NODE_INDEX_START = 0;
const NODE_INDEX_STEP = 1;

/**
 * Ensures optional activation inputs align 1:1 with layer nodes.
 *
 * This guard prevents silent index skew where one node might accidentally
 * reuse another node's value.
 *
 * In practice, this enables two safe activation modes:
 * - **Implicit activation**: omit `inputValues` and let each node compute its
 *   activation from its inbound connections.
 * - **Explicit activation**: pass a `number[]` with exactly one value per node.
 *
 * This function exists because a mismatched array length is almost always a
 * caller bug, and failing fast is easier to debug than producing subtly wrong
 * activations.
 *
 * Throws when `inputValues.length !== nodeCount`.
 *
 * Example:
 *
 * ```ts
 * assertActivationInputSize(3, [0.1, 0.2, 0.3]);
 * assertActivationInputSize(3, [0.1, 0.2]); // throws
 * ```
 *
 * @param nodeCount Number of nodes in the layer.
 * @param inputValues Optional activation values provided by the caller.
 */
export function assertActivationInputSize(
  nodeCount: number,
  inputValues?: number[],
): void {
  if (inputValues !== undefined && inputValues.length !== nodeCount) {
    throw new LayerSizeMismatchError(SIZE_MISMATCH_ERROR_MESSAGE);
  }
}

/**
 * Resolves a shared dropout mask for the full layer.
 *
 * In this layer-level dropout model, all nodes receive the same mask per call,
 * which keeps activation behavior synchronized for grouped layer semantics.
 *
 * Notes:
 * - Dropout is only applied when `isTraining` is true.
 * - A return value of `1` means "keep" and `0` means "drop".
 * - This helper intentionally does **not** rescale activations (some dropout
 *   implementations divide by $(1 - p)$ during training). In this codebase the
 *   mask is a simple on/off switch.
 *
 * Example:
 *
 * ```ts
 * resolveLayerMask(0.5, false); // => 1 (dropout disabled)
 * resolveLayerMask(0.5, true); // => 0 or 1
 * ```
 *
 * @param layerDropout The dropout rate configured for the layer.
 * @param isTraining Whether the layer is running in training mode.
 * @returns A mask value of 1 or 0 for all nodes in the layer.
 */
export function resolveLayerMask(
  layerDropout: number,
  isTraining: boolean,
): number {
  if (!isTraining || layerDropout <= DROPOUT_DISABLED_THRESHOLD) {
    return MASK_ENABLED;
  }

  return Math.random() >= layerDropout ? MASK_ENABLED : MASK_DISABLED;
}

/**
 * Applies one mask value to every node in the layer.
 *
 * In this library, a node-level `mask` is used as a lightweight dropout control.
 * A mask of `0` effectively disables the node for the current activation step.
 *
 * Example:
 *
 * ```ts
 * applyLayerMask(layer.nodes, 1);
 * ```
 *
 * @param nodeList The layer nodes to update.
 * @param mask The mask value to apply.
 */
export function applyLayerMask(
  nodeList: LayerActivationContext['nodes'],
  mask: number,
): void {
  nodeList.forEach((node) => {
    node.mask = mask;
  });
}

/**
 * Acquires a pooled output buffer sized for the current activation call.
 *
 * Pooling avoids frequent temporary allocations in hot activation paths.
 *
 * The returned array is owned by the pool. Treat it as **temporary**:
 * - Fill it.
 * - Clone it (if you need a stable output).
 * - Release it back to the pool.
 *
 * Example (typical pattern):
 *
 * ```ts
 * const pooled = acquireActivationOutput(nodes.length);
 * fillActivationOutput(nodes, values, pooled);
 * const output = cloneActivationOutput(pooled);
 * releaseActivationOutput(pooled);
 * ```
 *
 * @param nodeCount Number of nodes in the layer.
 * @returns A pooled output array.
 */
export function acquireActivationOutput(nodeCount: number): number[] {
  return activationArrayPool.acquire(nodeCount) as number[];
}

/**
 * Releases a pooled activation output buffer back to the pool.
 *
 * Call this after cloning/consuming the buffer to keep memory reuse effective.
 *
 * Important: do not keep using `output` after releasing it.
 *
 * @param output The pooled output array to release.
 */
export function releaseActivationOutput(output: number[]): void {
  activationArrayPool.release(output);
}

/**
 * Clones pooled output into a stable caller-owned array.
 *
 * This is the "escape hatch" that turns a pooled scratch buffer into a normal
 * array you can safely return from APIs.
 *
 * Example:
 *
 * ```ts
 * const stable = cloneActivationOutput(pooled);
 * ```
 *
 * @param output The pooled output array to clone.
 * @returns A cloned output array.
 */
export function cloneActivationOutput(output: number[]): number[] {
  return Array.from(output);
}

/**
 * Activates each node and writes outputs into the provided buffer.
 *
 * When `inputValues` is provided, each node receives the corresponding input
 * value. Otherwise each node self-activates from incoming state.
 *
 * This function is deliberately low-level: it does not allocate and it does not
 * return anything. That makes it ideal for hot paths where you want to reuse a
 * buffer (typically a pooled one).
 *
 * Example:
 *
 * ```ts
 * fillActivationOutput(layer.nodes, [0.2, 0.4], pooled);
 * ```
 *
 * @param nodeList Nodes to activate.
 * @param inputValues Optional activation values for each node.
 * @param output Output buffer to populate.
 */
export function fillActivationOutput(
  nodeList: LayerActivationContext['nodes'],
  inputValues: number[] | undefined,
  output: number[],
): void {
  for (
    let nodeIndex = NODE_INDEX_START;
    nodeIndex < nodeList.length;
    nodeIndex += NODE_INDEX_STEP
  ) {
    const node = nodeList[nodeIndex];
    let activationValue: number;

    if (inputValues === undefined) {
      activationValue = node.activate();
    } else {
      activationValue = node.activate(inputValues[nodeIndex]);
    }

    output[nodeIndex] = activationValue;
  }
}
