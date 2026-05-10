import type { ActivationPrecision, PrecisionConfig } from '../../../config';
import type { ActivationArray } from '../../activationArrayPool/activationArrayPool';
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
): ActivationArray {
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
): ActivationArray {
  if (activationContext.networkInternal._reuseActivationArrays) {
    return activateWithReusableOutputBuffer(activationContext);
  }

  return activateViaNetworkDelegate(activationContext);
}

/**
 * Reuse the network-owned activation buffer for raw activation output when enabled.
 *
 * @param activationContext - Shared raw activation state.
 * @returns Reused typed buffer or a detached plain array, depending on runtime flags.
 */
function activateWithReusableOutputBuffer(
  activationContext: RawActivationContext,
): ActivationArray {
  const rawActivationResult = activateViaNetworkDelegate(activationContext);
  const reusableOutputBuffer = ensureReusableActivationOutputBuffer(
    activationContext,
    rawActivationResult.length,
  );

  copyActivationResultIntoReusableBuffer(rawActivationResult, reusableOutputBuffer);

  if (shouldReturnTypedActivations(activationContext)) {
    return reusableOutputBuffer;
  }

  return detachReusableOutputBuffer(reusableOutputBuffer);
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

/**
 * Ensure the network owns a reusable typed activation output buffer of the requested size.
 *
 * @param activationContext - Shared raw activation state.
 * @param outputSize - Required output width for the current activation pass.
 * @returns Reusable typed activation output buffer.
 */
function ensureReusableActivationOutputBuffer(
  activationContext: RawActivationContext,
  outputSize: number,
): Float32Array | Float64Array {
  const runtimeNetwork = activationContext.networkInternal as RawActivationRuntimeProps;
  const useFloat32Activation =
    resolveRawActivationPrecision(runtimeNetwork) === 'f32';

  if (
    !runtimeNetwork._activationPool ||
    runtimeNetwork._activationPool.length !== outputSize ||
    requiresTypedActivationPoolReplacement(
      runtimeNetwork._activationPool,
      useFloat32Activation,
    )
  ) {
    runtimeNetwork._activationPool = useFloat32Activation
      ? new Float32Array(outputSize)
      : new Float64Array(outputSize);
  }

  return runtimeNetwork._activationPool;
}

/**
 * Copy plain activation output values into the reusable typed buffer.
 *
 * @param activationResult - Detached activation result from the main activation path.
 * @param reusableOutputBuffer - Network-owned typed output buffer.
 * @returns Nothing.
 */
function copyActivationResultIntoReusableBuffer(
  activationResult: number[],
  reusableOutputBuffer: Float32Array | Float64Array,
): void {
  for (
    let outputIndex = 0;
    outputIndex < activationResult.length;
    outputIndex++
  ) {
    reusableOutputBuffer[outputIndex] = activationResult[outputIndex];
  }
}

/**
 * Decide whether raw activation may return the reusable typed buffer directly.
 *
 * @param activationContext - Shared raw activation state.
 * @returns True when typed activations may escape to the caller.
 */
function shouldReturnTypedActivations(
  activationContext: RawActivationContext,
): boolean {
  return (activationContext.networkInternal as RawActivationRuntimeProps)
    ._returnTypedActivations;
}

/**
 * Detach the reusable typed output buffer into a plain array for compatibility callers.
 *
 * @param reusableOutputBuffer - Network-owned typed output buffer.
 * @returns Detached plain activation output array.
 */
function detachReusableOutputBuffer(
  reusableOutputBuffer: Float32Array | Float64Array,
): number[] {
  return Array.from(reusableOutputBuffer);
}

/**
 * Read the resolved runtime activation precision for raw typed-output reuse.
 *
 * @param runtimeNetwork - Raw activation runtime view.
 * @returns Active activation precision for reusable raw output.
 */
function resolveRawActivationPrecision(
  runtimeNetwork: RawActivationRuntimeProps,
): ActivationPrecision {
  if (
    runtimeNetwork._activationPrecision === 'f32' &&
    runtimeNetwork._activationPrecision !==
      runtimeNetwork._precisionConfig?.activationPrecision
  ) {
    return runtimeNetwork._activationPrecision;
  }

  return (
    runtimeNetwork._precisionConfig?.activationPrecision ??
    runtimeNetwork._activationPrecision ??
    'f64'
  );
}

/**
 * Check whether the current reusable buffer must be replaced for the requested precision.
 *
 * @param activationPool - Existing reusable activation pool buffer.
 * @param useFloat32Activation - True when the caller needs Float32 output.
 * @returns True when the current buffer type does not match the requested precision.
 */
function requiresTypedActivationPoolReplacement(
  activationPool: Float32Array | Float64Array,
  useFloat32Activation: boolean,
): boolean {
  return useFloat32Activation
    ? !(activationPool instanceof Float32Array)
    : !(activationPool instanceof Float64Array);
}

type RawActivationRuntimeProps = RawActivationContext['networkInternal'] & {
  _activationPool?: Float32Array | Float64Array;
  _precisionConfig?: PrecisionConfig;
  _activationPrecision?: ActivationPrecision;
  _returnTypedActivations: boolean;
};
