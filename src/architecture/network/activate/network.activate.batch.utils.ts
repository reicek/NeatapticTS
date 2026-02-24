import {
  BATCH_INPUTS_COLLECTION_ERROR_MESSAGE,
  UNDEFINED_INPUT_LENGTH_TEXT,
  type BatchActivationContext,
  type BatchRowActivationContext,
} from './network.activate.utils.types';

/**
 * Execute mini-batch activation with top-level shape validation and per-row checks.
 *
 * The orchestration keeps behavior deterministic by validating the container first,
 * then validating each row before delegating to the core network activation function.
 *
 * @param activationContext - Shared batch activation state.
 * @returns Matrix of activation outputs.
 */
export function executeBatchActivation(
  activationContext: BatchActivationContext,
): number[][] {
  // Step 1: Validate the top-level batch container shape.
  assertBatchInputCollection(activationContext.batchInputs);

  // Step 2: Activate each input row with per-row validation.
  return activateValidatedBatchRows(activationContext);
}

/**
 * Validate that the batch input collection is an array of rows.
 *
 * @param batchInputs - Candidate batch input collection.
 * @returns Nothing.
 */
function assertBatchInputCollection(batchInputs: number[][]): void {
  if (!Array.isArray(batchInputs)) {
    throw new Error(BATCH_INPUTS_COLLECTION_ERROR_MESSAGE);
  }
}

/**
 * Activate each row in a validated batch matrix.
 *
 * @param activationContext - Shared batch activation state.
 * @returns Matrix of activation outputs.
 */
function activateValidatedBatchRows(
  activationContext: BatchActivationContext,
): number[][] {
  return activationContext.batchInputs.map(
    function activateBatchRow(inputVector, batchIndex): number[] {
      return activateSingleBatchRow({
        networkInternal: activationContext.networkInternal,
        inputVector,
        batchIndex,
        expectedInputSize: activationContext.expectedInputSize,
        isTraining: activationContext.isTraining,
      });
    },
  );
}

/**
 * Validate and activate one batch row.
 *
 * @param rowActivationContext - Shared state for one batch-row activation.
 * @returns Activation output vector for the row.
 */
function activateSingleBatchRow(
  rowActivationContext: BatchRowActivationContext,
): number[] {
  assertBatchRowInputSize(rowActivationContext);
  return rowActivationContext.networkInternal.activate(
    rowActivationContext.inputVector,
    rowActivationContext.isTraining,
  );
}

/**
 * Validate one batch row dimensionality.
 *
 * @param rowActivationContext - Shared state for one batch-row activation.
 * @returns Nothing.
 */
function assertBatchRowInputSize(
  rowActivationContext: BatchRowActivationContext,
): void {
  if (isBatchRowInputSizeValid(rowActivationContext)) return;

  throw new Error(buildBatchRowInputSizeMismatchMessage(rowActivationContext));
}

/**
 * Determine whether one batch row matches the expected input dimensionality.
 *
 * @param rowActivationContext - Shared state for one batch-row activation.
 * @returns True when row size is valid.
 */
function isBatchRowInputSizeValid(
  rowActivationContext: BatchRowActivationContext,
): boolean {
  return (
    Array.isArray(rowActivationContext.inputVector) &&
    rowActivationContext.inputVector.length ===
      rowActivationContext.expectedInputSize
  );
}

/**
 * Build a descriptive mismatch message for invalid batch row input dimensions.
 *
 * @param rowActivationContext - Shared state for one batch-row activation.
 * @returns Formatted error message for invalid row dimensionality.
 */
function buildBatchRowInputSizeMismatchMessage(
  rowActivationContext: BatchRowActivationContext,
): string {
  const receivedInputLength = formatInputLengthForMessage(
    rowActivationContext.inputVector,
  );

  return `Input[${rowActivationContext.batchIndex}] size mismatch: expected ${rowActivationContext.expectedInputSize}, got ${receivedInputLength}`;
}

/**
 * Convert input length into a display-safe string for error messaging.
 *
 * @param inputVector - Candidate batch row input vector.
 * @returns Numeric length as string or predefined undefined text.
 */
function formatInputLengthForMessage(inputVector: number[]): string {
  if (inputVector) return `${inputVector.length}`;
  return UNDEFINED_INPUT_LENGTH_TEXT;
}
