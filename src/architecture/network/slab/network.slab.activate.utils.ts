import type Network from '../../network/network';
import {
  _tryFastSlabFallbackForGating,
  _tryFastSlabFallbackForMissingPrerequisites,
  _prepareFastSlabRuntime,
  _resolveFastTopoOrder,
  _ensureFastSlabBuffers,
  _seedFastInputLayer,
  _propagateFastSlabActivations,
  _collectFastSlabOutput,
} from './network.slab.fast-path.helpers.utils';
import { _reindexNodes } from './network.slab.shared.helpers.utils';
import type { NetworkSlabProps } from './network.slab.utils.types';
import { SLAB_ZERO } from './network.slab.utils.types';

/**
 * Executes fast slab activation once slab and adjacency prerequisites are prepared.
 *
 * @param network - Target network.
 * @param input - Input activation vector.
 * @returns Output activation array.
 */
export function _activateFastSlab(network: Network, input: number[]): number[] {
  // Step 1: Resolve internal slab runtime and apply fallback guards.
  const internalNet = network as unknown as NetworkSlabProps;
  const gatedFallback = _tryFastSlabFallbackForGating(network, input);
  if (gatedFallback) {
    return gatedFallback;
  }

  const missingPrerequisiteFallback =
    _tryFastSlabFallbackForMissingPrerequisites(network, internalNet, input);
  if (missingPrerequisiteFallback) {
    return missingPrerequisiteFallback;
  }

  // Step 2: Prepare runtime caches and buffers.
  _prepareFastSlabRuntime(network, internalNet, (currentNetwork) => {
    _reindexNodes(currentNetwork);
  });
  const topoOrder = _resolveFastTopoOrder(network, internalNet);
  const nodeCount = network.nodes.length;
  _ensureFastSlabBuffers(internalNet, nodeCount);
  const activationBuffer = internalNet._fastA as Float32Array | Float64Array;
  const stateBuffer = internalNet._fastS as Float32Array | Float64Array;
  stateBuffer.fill(SLAB_ZERO);

  // Step 3: Seed inputs, propagate, and collect output values.
  _seedFastInputLayer(network, input, activationBuffer);
  _propagateFastSlabActivations(
    network,
    internalNet,
    topoOrder,
    activationBuffer,
    stateBuffer,
  );
  return _collectFastSlabOutput(network, activationBuffer, nodeCount);
}
