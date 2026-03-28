import { type RawActivationContext } from './network.activate.utils.types';

/**
 * Execute raw activation through the network delegate using a compact orchestration flow.
 *
 * This helper keeps the exported activation method focused on context creation while this
 * module owns the execution path and future branching behavior.
 *
 * @param activationContext - Shared raw activation state.
 * @returns Activation output vector from the network delegate.
 */
export function executeRawActivation(
  activationContext: RawActivationContext,
): number[] {
  // Step 1: Select the activation path for the current reuse configuration.
  return activateWithSelectedReusePath(activationContext);
}

/**
 * Select the raw activation execution path based on runtime reuse configuration.
 *
 * @param activationContext - Shared raw activation state.
 * @returns Activation output vector.
 */
function activateWithSelectedReusePath(
  activationContext: RawActivationContext,
): number[] {
  if (activationContext.networkInternal._reuseActivationArrays) {
    return activateViaNetworkDelegate(activationContext);
  }

  return activateViaNetworkDelegate(activationContext);
}

/**
 * Delegate raw activation to the core network activation implementation.
 *
 * @param activationContext - Shared raw activation state.
 * @returns Activation output vector.
 */
function activateViaNetworkDelegate(
  activationContext: RawActivationContext,
): number[] {
  return activationContext.networkInternal.activate(
    activationContext.inputVector,
    activationContext.isTraining,
    activationContext.maximumActivationDepth,
  );
}
