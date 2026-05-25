import type Network from '../../network/network';
import type { ActivateNetworkInternals as NetworkInternals } from '../network.types';
import type {
  BatchActivationContext,
  NoTraceActivationContext,
  RawActivationContext,
} from './network.activate.utils.types';

/**
 * Create the immutable context consumed by no-trace activation helpers.
 *
 * This context snapshots the caller input and expected input width while exposing
 * the internal network surface required by low-level activation utilities.
 *
 * @param network - Network instance bound to the activation call.
 * @param inputVector - Input activation vector supplied by the caller.
 * @returns Fully populated no-trace activation context.
 * @example
 * ```ts
 * const context = createNoTraceActivationContext(network, [0.2, 0.8]);
 * // context.expectedInputSize mirrors network.input
 * ```
 */
export function createNoTraceActivationContext(
  network: Network,
  inputVector: number[],
): NoTraceActivationContext {
  return {
    network,
    networkInternal: toNetworkInternals(network),
    inputVector,
    expectedInputSize: network.input,
  };
}

/**
 * Create the context consumed by raw activation helpers.
 *
 * Raw activation may preserve node traces for training and uses an explicit
 * depth limit to prevent runaway recurrent propagation.
 *
 * @param network - Network instance bound to the activation call.
 * @param inputVector - Input activation vector supplied by the caller.
 * @param isTraining - Whether activation should retain training traces.
 * @param maximumActivationDepth - Guard against runaway activation depth.
 * @returns Fully populated raw activation context.
 * @example
 * ```ts
 * const context = createRawActivationContext(network, [1, 0], true, 64);
 * // context.maximumActivationDepth bounds propagation depth
 * ```
 */
export function createRawActivationContext(
  network: Network,
  inputVector: number[],
  isTraining: boolean,
  maximumActivationDepth: number,
): RawActivationContext {
  return {
    networkInternal: toNetworkInternals(network),
    inputVector,
    isTraining,
    maximumActivationDepth,
  };
}

/**
 * Create the context consumed by batch activation helpers.
 *
 * Batch mode carries the full input matrix, expected input width, and training
 * trace policy in one object so downstream helpers can stay orchestration-only.
 *
 * @param network - Network instance bound to the activation call.
 * @param batchInputs - Input matrix supplied by the caller.
 * @param isTraining - Whether activation should retain training traces.
 * @returns Fully populated batch activation context.
 * @example
 * ```ts
 * const context = createBatchActivationContext(network, [[0, 1], [1, 0]], false);
 * // context.batchInputs can be iterated row-by-row by activation helpers
 * ```
 */
export function createBatchActivationContext(
  network: Network,
  batchInputs: number[][],
  isTraining: boolean,
): BatchActivationContext {
  return {
    networkInternal: toNetworkInternals(network),
    batchInputs,
    expectedInputSize: network.input,
    isTraining,
  };
}

/**
 * Convert a network instance into the activation internals interface used by helper modules.
 *
 * @param network - Runtime network instance.
 * @returns Network internals view used by activation helper modules.
 */
function toNetworkInternals(network: Network): NetworkInternals {
  return network as unknown as NetworkInternals;
}
