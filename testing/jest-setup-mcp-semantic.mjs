/* global process, afterAll */

/**
 * @module jest-setup-mcp-semantic
 * @description Shared setup for the mcp-semantic-mjs Jest project.
 *
 * Forces the Repo Cortex dense and reranker ONNX subsystems to cold so ESM
 * contract tests exercise graceful degradation without loading
 * onnxruntime-node native bindings, which can crash during Jest teardown.
 */

const originalDenseForceState = process.env.DENSE_FORCE_STATE;
const originalRerankerForceState = process.env.RERANKER_FORCE_STATE;

process.env.DENSE_FORCE_STATE = 'cold';
process.env.RERANKER_FORCE_STATE = 'cold';

afterAll(() => {
  if (originalDenseForceState === undefined) {
    delete process.env.DENSE_FORCE_STATE;
  } else {
    process.env.DENSE_FORCE_STATE = originalDenseForceState;
  }
  if (originalRerankerForceState === undefined) {
    delete process.env.RERANKER_FORCE_STATE;
  } else {
    process.env.RERANKER_FORCE_STATE = originalRerankerForceState;
  }
});
