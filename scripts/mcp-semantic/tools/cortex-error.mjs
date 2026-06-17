/**
 * @module cortex-error
 * @description Shared error taxonomy for Repo Cortex MCP tools.
 *
 * All corpus tools prefix their error messages with a stable code so callers
 * and tests can identify failure modes without parsing free-form text.
 */

/** Known error codes emitted by Repo Cortex MCP tools. */
export const ErrorCodes = Object.freeze({
  CORPUS_NOT_FOUND: 'CORPUS_NOT_FOUND',
  CORTEX_TIMEOUT_PARTIAL: 'CORTEX_TIMEOUT_PARTIAL',
  EMPTY_QUERY: 'EMPTY_QUERY',
  INVALID_ALPHA: 'INVALID_ALPHA',
  INVALID_BUDGET: 'INVALID_BUDGET',
  INVALID_LIMIT: 'INVALID_LIMIT',
  INVALID_MAX_HOPS: 'INVALID_MAX_HOPS',
  INVALID_METADATA_FILTER: 'INVALID_METADATA_FILTER',
  INVALID_QUERY_CLASS: 'INVALID_QUERY_CLASS',
  INVALID_SIGNAL_TYPE: 'INVALID_SIGNAL_TYPE',
  MISSING_CHUNK_ID: 'MISSING_CHUNK_ID',
  SEED_REQUIRED: 'SEED_REQUIRED',
});

/**
 * Build an Error whose message starts with the requested taxonomy code.
 *
 * @param {string} code - One of {@link ErrorCodes}.
 * @param {string} message - Human-readable details after the code.
 * @returns {Error} Throwable error carrying the taxonomy code.
 */
export function cortexError(code, message) {
  return new Error(`${code}: ${message}`);
}

/**
 * Check whether an error carries a given taxonomy code in its message.
 *
 * @param {unknown} error - Caught value.
 * @param {string} code - Error code to match.
 * @returns {boolean} True when `error` is an Error and its message starts with `code:`.
 */
export function isCortexError(error, code) {
  return (
    error instanceof Error &&
    typeof error.message === 'string' &&
    error.message.startsWith(`${code}:`)
  );
}
