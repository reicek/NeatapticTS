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
export { evaluateInWorkers } from './network.worker-payload.batch';
export { createNeatParallelPopulationEvaluator } from './network.worker-payload.neat';
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
import {
  createInferencePredictor as createInferencePredictorImpl,
  exportPortableInferencePayload as exportPortableInferencePayloadImpl,
  exportTransferableInferencePayload as exportTransferableInferencePayloadImpl,
  extractNetworkInferenceIR as extractNetworkInferenceIRImpl,
} from './network.worker-payload.utils';
import type {
  InferenceChannelOptions as InferenceChannelOptionsType,
  NetworkInferenceIREdge as NetworkInferenceIREdgeType,
  PortableInferencePayloadEdge as PortableInferencePayloadEdgeType,
  TransferableInferencePayloadOptions as TransferableInferencePayloadOptionsType,
} from './network.worker-payload.types';

/**
 * Create an inference predictor from a portable or transferable payload so repeated forward evaluations can run without reconstructing a mutable network instance.
 */
export const createInferencePredictor = createInferencePredictorImpl;

/**
 * Extract a deterministic inference intermediate representation from a live network so transport and payload-export helpers can share one stable graph contract.
 */
export const extractNetworkInferenceIR = extractNetworkInferenceIRImpl;

/**
 * Export a portable inference payload with structured-clone-safe data so browser and node runtimes can persist and reload predictor artifacts reliably.
 */
export const exportPortableInferencePayload =
  exportPortableInferencePayloadImpl;

/**
 * Export a transferable inference payload so typed-array-heavy artifacts can cross worker boundaries with explicit ownership transfer and reduced copy overhead.
 */
export const exportTransferableInferencePayload =
  exportTransferableInferencePayloadImpl;

/**
 * Channel options that configure request batching, queueing, and lifecycle behavior for asynchronous inference transport endpoints across browser and node worker runtimes.
 * These options define how inference requests are buffered, dispatched, and finalized over long-lived channel sessions.
 */
export type InferenceChannelOptions = InferenceChannelOptionsType;

/**
 * Directed edge record in the inference graph representation used by payload export, replay, and worker predictor execution pipelines.
 * The edge contract preserves stable source-target linkage and weight metadata during transport and reconstruction.
 */
export type NetworkInferenceIREdge = NetworkInferenceIREdgeType;

/**
 * Portable payload edge schema preserving source-target linkage and connection metadata in runtime-agnostic JSON-style artifacts.
 * This shape is intentionally serialization-friendly so offline tooling can inspect and replay predictor graphs deterministically.
 */
export type PortableInferencePayloadEdge = PortableInferencePayloadEdgeType;

/**
 * Transfer options that control buffer packing and transfer-list behavior for worker-friendly inference payload publication.
 * They allow callers to tune ownership and copy semantics for high-throughput cross-thread prediction workloads.
 */
export type TransferableInferencePayloadOptions =
  TransferableInferencePayloadOptionsType;

export {
  getTransferList,
  INFERENCE_ACTIVATION_TABLE,
} from './network.worker-payload.utils';
export { openInferenceChannel } from './network.worker-payload.channel';
export {
  openSharedInferenceWorker,
  SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION,
} from './network.worker-payload.shared';
export type {
  InferenceChannel,
  InferencePredictor,
  NetworkInferenceIR,
  NetworkInferenceIRNode,
  PortableInferencePayload,
  PortableInferencePayloadNode,
  SharedInferenceWorker,
  SharedInferenceWorkerOptions,
  TransferableInferencePayload,
} from './network.worker-payload.types';
