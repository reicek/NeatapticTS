import type Network from '../../network/network';
import type { ActivateNetworkInternals as NetworkInternals } from '../network.types';
import type {
  BatchActivationContext,
  NoTraceActivationContext,
  RawActivationContext,
} from './network.activate.utils.types';

/**
 * Build shared no-trace activation context for helper orchestration.
 *
 * @param network - Network instance bound to the activation call.
 * @param inputVector - Input activation vector supplied by the caller.
 * @returns Fully populated no-trace activation context.
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
 * Build shared raw activation context for helper orchestration.
 *
 * @param network - Network instance bound to the activation call.
 * @param inputVector - Input activation vector supplied by the caller.
 * @param isTraining - Whether activation should retain training traces.
 * @param maximumActivationDepth - Guard against runaway activation depth.
 * @returns Fully populated raw activation context.
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
 * Build shared batch activation context for helper orchestration.
 *
 * @param network - Network instance bound to the activation call.
 * @param batchInputs - Input matrix supplied by the caller.
 * @param isTraining - Whether activation should retain training traces.
 * @returns Fully populated batch activation context.
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
