import type Network from '../../network/network';
import type {
  ConnectionSlabView,
  NetworkSlabProps,
} from './network.slab.utils.types';
import { SLAB_ONE, SLAB_ZERO } from './network.slab.utils.types';

/**
 * Creates a read-oriented packed slab view from current network internals.
 *
 * @param network - Target network.
 * @returns Packed connection slab view.
 */
export function _createConnectionSlabView(
  network: Network,
): ConnectionSlabView {
  // Step 1: Resolve mutable slab internals from network runtime.
  const internalNet = network as unknown as NetworkSlabProps;
  const capacity = _resolveConnectionSlabCapacity(internalNet);
  const gain = _resolveConnectionGainView(internalNet, capacity);

  // Step 2: Return current slab arrays and metadata as one view object.
  return {
    weights: internalNet._connWeights!,
    from: internalNet._connFrom!,
    to: internalNet._connTo!,
    flags: internalNet._connFlags!,
    gain,
    plastic: internalNet._connPlastic || null,
    version: internalNet._slabVersion || SLAB_ZERO,
    used: internalNet._connCount || SLAB_ZERO,
    capacity,
  };
}

/**
 * Reads the current monotonic slab version counter from network internals.
 *
 * @param network - Target network.
 * @returns Non-negative slab version counter.
 */
export function _readSlabVersion(network: Network): number {
  // Step 1: Read monotonic slab version with zero fallback.
  const internalNet = network as unknown as NetworkSlabProps;
  return internalNet._slabVersion || SLAB_ZERO;
}

/**
 * Resolves effective slab capacity using explicit capacity first.
 *
 * @param internalNet - Internal slab runtime shape.
 * @returns Effective capacity value.
 */
function _resolveConnectionSlabCapacity(internalNet: NetworkSlabProps): number {
  // Step 1: Prefer tracked capacity and fall back to weight slab length.
  return (
    internalNet._connCapacity ||
    (internalNet._connWeights && internalNet._connWeights.length) ||
    SLAB_ZERO
  );
}

/**
 * Resolves gain slab view, synthesizing neutral gain values when omitted.
 *
 * @param internalNet - Internal slab runtime shape.
 * @param capacity - Resolved slab capacity.
 * @returns Gain array view.
 */
function _resolveConnectionGainView(
  internalNet: NetworkSlabProps,
  capacity: number,
): Float32Array | Float64Array | null {
  // Step 1: Return retained gain slab when present.
  if (internalNet._connGain) {
    return internalNet._connGain;
  }

  // Step 2: Synthesize neutral gain view while preserving omission semantics.
  const gain = internalNet._useFloat32Weights
    ? new Float32Array(capacity)
    : new Float64Array(capacity);
  for (
    let connectionIndex = SLAB_ZERO;
    connectionIndex < (internalNet._connCount || SLAB_ZERO);
    connectionIndex++
  ) {
    gain[connectionIndex] = SLAB_ONE;
  }
  return gain;
}
