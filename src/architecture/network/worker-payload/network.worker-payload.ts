/**
 * Worker-friendly inference payload primitives for `Network` forward passes.
 *
 * This module starts the transport ladder with the one artifact every later
 * strategy needs: a deterministic, plain-data inference IR. The IR keeps the
 * runtime execution order, activation identities, recurrent self-loop metadata,
 * and output ordering, but it deliberately leaves out training-time objects and
 * host-specific worker mechanics.
 *
 * Read this boundary in two passes:
 *
 * 1. `extractNetworkInferenceIR(...)` when you want a stable snapshot of a
 *    live network forward pass.
 * 2. `INFERENCE_ACTIVATION_TABLE` when you need the exact activation-id shelf
 *    that worker payloads reference.
 *
 * @example
 * ```ts
 * const inferenceIr = extractNetworkInferenceIR(network);
 * const outputNodeIndexes = inferenceIr.outputNodeIndices;
 * ```
 */
export {
  detectInferenceWorkerCapabilities,
  resolveAutoInferenceTransport,
} from './network.worker-payload.capabilities';
export { resolveBrowserWorkerAssetUrl } from './network.worker-payload.browser-url';
export {
  evaluateInWorkers,
} from './network.worker-payload.batch';
export {
  createNeatParallelPopulationEvaluator,
} from './network.worker-payload.neat';
export { ParallelInferencePool } from './network.worker-payload.pool';
export type {
  AutoInferenceTransport,
  InferenceWorkerCapabilities,
  InferenceWorkerCapabilityOptions,
} from './network.worker-payload.capabilities';
export type { BrowserWorkerAssetUrlOptions } from './network.worker-payload.browser-url';
export type {
  BatchEvaluationResult,
  EvaluateInWorkersOptions,
} from './network.worker-payload.batch';
export type { NeatParallelPopulationEvaluatorOptions } from './network.worker-payload.neat';
export type {
  ParallelInferencePoolOptions,
  ParallelInferenceWorkerLike,
} from './network.worker-payload.pool';
export {
  createInferencePredictor,
  exportTransferableInferencePayload,
  extractNetworkInferenceIR,
  getTransferList,
  exportPortableInferencePayload,
  INFERENCE_ACTIVATION_TABLE,
} from './network.worker-payload.utils';
export { openInferenceChannel } from './network.worker-payload.channel';
export {
  openSharedInferenceWorker,
  SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION,
} from './network.worker-payload.shared';
export type {
  InferenceChannel,
  InferenceChannelOptions,
  InferencePredictor,
  NetworkInferenceIR,
  NetworkInferenceIREdge,
  NetworkInferenceIRNode,
  PortableInferencePayload,
  PortableInferencePayloadEdge,
  PortableInferencePayloadNode,
  SharedInferenceWorker,
  SharedInferenceWorkerOptions,
  TransferableInferencePayload,
  TransferableInferencePayloadOptions,
} from './network.worker-payload.types';
