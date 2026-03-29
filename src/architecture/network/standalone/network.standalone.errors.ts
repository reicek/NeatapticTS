/**
 * Raised when standalone generation is requested for a network without output nodes.
 */
export class NetworkStandaloneNoOutputNodesError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkStandaloneNoOutputNodesError';
  }
}

/**
 * Stable error name emitted into generated standalone input guards.
 */
export const NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME =
  'NetworkStandaloneInputSizeMismatchError';

/**
 * Build the named-error factory source emitted into generated standalone functions.
 *
 * @returns Deterministic JavaScript source for a named input-size mismatch error factory.
 */
export function buildStandaloneInputSizeMismatchErrorFactorySource(): string {
  return `function ${NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME}(message){ var error = new Error(message); error.name = '${NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME}'; return error; }`;
}
