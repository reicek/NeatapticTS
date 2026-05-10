import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import type { ActivationPrecision, PrecisionConfig } from '../../../config';
import {
  NO_TRACE_FAST_SLAB_TRAINING_FLAG,
  UNDEFINED_INPUT_LENGTH_TEXT,
  type NoTraceActivationContext,
} from './network.activate.utils.types';
import { populatePooledOutputBufferFromNodes } from './network.activate.notrace.traversal.utils';
import { NetworkActivateInputSizeMismatchError } from './network.activate.errors';

/**
 * Execute no-trace activation with a fast-path attempt and deterministic fallback traversal.
 *
 * The orchestration follows a strict sequence: refresh order guarantees, validate input shape,
 * try fast slab inference, then compute outputs through node traversal when needed.
 *
 * @param activationContext - Shared no-trace activation state.
 * @returns Output activation vector detached from pooled storage.
 */
export function executeNoTraceActivation(
  activationContext: NoTraceActivationContext,
): number[] {
  // Step 1: Keep ordering guarantees current when acyclic constraints are enabled.
  refreshTopologicalOrderWhenRequired(activationContext);

  // Step 2: Fail fast on invalid input dimensionality.
  assertInputMatchesNetworkInputSize(activationContext);

  // Step 3: Opportunistically use fast slab inference when available.
  const fastSlabResult = tryActivateWithFastSlab(activationContext);
  if (fastSlabResult !== null) return fastSlabResult;

  // Step 4: Fall back to deterministic node-by-node activation.
  return activateWithoutTraceUsingNodeIteration(activationContext);
}

/**
 * Refresh compiled activation scheduling when topology is marked dirty.
 *
 * @param activationContext - Shared no-trace activation state.
 * @returns Nothing.
 */
function refreshTopologicalOrderWhenRequired(
  activationContext: NoTraceActivationContext,
): void {
  if (activationContext.networkInternal._topoDirty) {
    activationContext.networkInternal._computeTopoOrder();
  }
}

/**
 * Validate that the input vector length matches expected network input dimensionality.
 *
 * @param activationContext - Shared no-trace activation state.
 * @returns Nothing.
 */
function assertInputMatchesNetworkInputSize(
  activationContext: NoTraceActivationContext,
): void {
  if (!isInputVectorLengthValid(activationContext)) {
    throw new NetworkActivateInputSizeMismatchError(
      buildInputSizeMismatchMessage(activationContext),
    );
  }
}

/**
 * Check whether the input vector has a valid length for activation.
 *
 * @param activationContext - Shared no-trace activation state.
 * @returns True when the input vector is an array with expected length.
 */
function isInputVectorLengthValid(
  activationContext: NoTraceActivationContext,
): boolean {
  return (
    Array.isArray(activationContext.inputVector) &&
    activationContext.inputVector.length === activationContext.expectedInputSize
  );
}

/**
 * Build a descriptive input mismatch message for activation validation errors.
 *
 * @param activationContext - Shared no-trace activation state.
 * @returns Formatted mismatch error message.
 */
function buildInputSizeMismatchMessage(
  activationContext: NoTraceActivationContext,
): string {
  const receivedInputLength = formatInputLengthForMessage(
    activationContext.inputVector,
  );

  return `Input size mismatch: expected ${activationContext.expectedInputSize}, got ${receivedInputLength}`;
}

/**
 * Convert input length into a display-safe string for error messaging.
 *
 * @param inputVector - Candidate activation input vector.
 * @returns Numeric length as string or predefined undefined text.
 */
function formatInputLengthForMessage(inputVector: number[]): string {
  if (inputVector) return `${inputVector.length}`;
  return UNDEFINED_INPUT_LENGTH_TEXT;
}

/**
 * Attempt fast slab activation and return null when slab execution is unavailable or fails.
 *
 * @param activationContext - Shared no-trace activation state.
 * @returns Fast slab output when successful, otherwise null.
 */
function tryActivateWithFastSlab(
  activationContext: NoTraceActivationContext,
): number[] | null {
  if (!canUseNoTraceFastSlab(activationContext)) return null;

  try {
    return activationContext.networkInternal._fastSlabActivate(
      activationContext.inputVector,
    );
  } catch {
    return null;
  }
}

/**
 * Determine whether fast slab activation is available for no-trace execution mode.
 *
 * @param activationContext - Shared no-trace activation state.
 * @returns True when slab execution is available for inference mode.
 */
function canUseNoTraceFastSlab(
  activationContext: NoTraceActivationContext,
): boolean {
  return activationContext.networkInternal._canUseFastSlab(
    NO_TRACE_FAST_SLAB_TRAINING_FLAG,
  );
}

/**
 * Execute no-trace activation through node traversal and pooled output collection.
 *
 * @param activationContext - Shared no-trace activation state.
 * @returns Detached output activation vector.
 */
function activateWithoutTraceUsingNodeIteration(
  activationContext: NoTraceActivationContext,
): number[] {
  const pooledOutputBuffer = activationArrayPool.acquire(
    activationContext.network.output,
    resolveNoTraceActivationPrecision(activationContext),
  );

  try {
    populatePooledOutputBufferFromNodes({
      network: activationContext.network,
      inputVector: activationContext.inputVector,
      pooledOutputBuffer,
    });

    return detachPooledOutputBuffer(pooledOutputBuffer);
  } finally {
    activationArrayPool.release(pooledOutputBuffer);
  }
}

/**
 * Clone pooled output storage into a detached plain array.
 *
 * @param pooledOutputBuffer - Pooled activation output storage.
 * @returns Detached output activation vector.
 */
function detachPooledOutputBuffer(
  pooledOutputBuffer: ReturnType<typeof activationArrayPool.acquire>,
): number[] {
  return Array.from(pooledOutputBuffer) as number[];
}

/**
 * Read the resolved runtime activation precision for no-trace output pooling.
 *
 * @param activationContext Shared no-trace activation state.
 * @returns Active per-network activation precision when present.
 */
function resolveNoTraceActivationPrecision(
  activationContext: NoTraceActivationContext,
): ActivationPrecision | undefined {
  const precisionCarrier = activationContext.networkInternal as NoTraceActivationContext['networkInternal'] & {
    _precisionConfig?: PrecisionConfig;
    _activationPrecision?: ActivationPrecision;
  };

  if (
    precisionCarrier._activationPrecision === 'f32' &&
    precisionCarrier._activationPrecision !==
      precisionCarrier._precisionConfig?.activationPrecision
  ) {
    return precisionCarrier._activationPrecision;
  }

  return (
    precisionCarrier._precisionConfig?.activationPrecision ??
    precisionCarrier._activationPrecision
  );
}
