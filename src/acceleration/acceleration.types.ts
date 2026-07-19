/**
 * Core type definitions for the generic network acceleration layer.
 *
 * This module provides the portable configuration and status vocabulary used by
 * the acceleration layer. It intentionally avoids importing implementation
 * details (GPU detection, worker pool logic) so it can be imported by both
 * browser and Node entry points without side effects.
 */

/**
 * Runtime acceleration mode selected after resolving capabilities and config.
 */
export type AccelerationMode = 'cpu' | 'gpu' | 'worker';

/**
 * Explicit backend preference that callers may request.
 *
 * - `auto` lets the library choose based on network size and environment.
 * - `gpu` prefers the WebGPU backend when eligible.
 * - `worker` prefers the worker-thread backend when eligible.
 * - `cpu` forces the CPU backend.
 */
export type BackendMode = 'auto' | 'gpu' | 'worker' | 'cpu';

/**
 * GPU capability report returned by environment detection.
 *
 * Reports whether the WebGPU backend can be selected, and gives a human-readable
 * explanation when it cannot (for example, missing adapter, disabled flag, or
 * network that is too small to benefit).
 */
export interface GPUStatus {
  /** True when a GPU backend is available and selected. */
  available: boolean;

  /** Human-readable explanation of the availability decision. */
  reason: string;

  /**
   * Optional resolved WebGPU device when one has been requested and acquired.
   *
   * This field carries the live device handle after a successful auto-enable
   * attempt so callers can assign it to a network or inference context. It is
   * optional because detection itself is synchronous and does not acquire a
   * device; device acquisition happens later on the auto-enable path.
   */
  device?: GPUDevice | null;
}

/**
 * Worker capability report returned by environment detection.
 *
 * Reports whether worker-thread evaluation can be selected, how many workers the
 * runtime intends to use, and why the worker path was accepted or rejected.
 */
export interface WorkerStatus {
  /** True when worker evaluation is available and selected. */
  available: boolean;

  /** Number of workers the runtime intends to use. */
  count: number;

  /** Human-readable explanation of the availability decision. */
  reason: string;
}

/**
 * Combined capability report used by both configuration and runtime status.
 */
export interface AccelerationCapabilities {
  /** GPU availability report. */
  gpu: GPUStatus;

  /** Worker availability report. */
  worker: WorkerStatus;
}

/**
 * User-supplied acceleration configuration.
 *
 * All fields are optional; {@link resolveAccelerationConfig} fills in the
 * defaults and resolves the final backend choice.
 */
export interface AccelerationConfig {
  /** Minimum node count before the GPU backend is considered. */
  gpuNodeThreshold?: number;

  /** Minimum parallel batch size before GPU batching is preferred. */
  gpuBatchParallelThreshold?: number;

  /** Minimum logical cores required before worker evaluation is considered. */
  workerMinCores?: number;

  /** Maximum number of workers to spawn when workers are enabled. */
  maxWorkers?: number;

  /** When true, the GPU backend is never selected even if available. */
  disableGPU?: boolean;

  /** When true, worker evaluation is never selected even if available. */
  disableWorkers?: boolean;

  /** Preferred backend mode; defaults to `auto`. */
  backend?: BackendMode;

  /** True when an active worker pool currently owns execution. */
  hasActiveWorker?: boolean;

  /**
   * Number of weight variants to evaluate concurrently in a single batch.
   *
   * Defaults to `16`. A value of `1` evaluates variants sequentially; larger
   * values dispatch that many variants in parallel, restoring each connection's
   * original weight before the next batch begins.
   */
  parallelVariantCount?: number;

  /**
   * Optional per-lifecycle-stage variant counts forwarded to NGE grow/stabilize
   * evaluation.
   *
   * When provided, a stage-specific count overrides the default number of
   * weight variants generated for that NGE lifecycle stage, ensuring the
   * generated variant pool matches the configured parallel evaluation width.
   * Consumers such as {@link evaluateNgeWeightVariants} read these counts from
   * the acceleration config and fall back to their own stage defaults when a
   * stage is omitted.
   */
  stageVariantCounts?: {
    /** Number of weight variants generated for baby-stage networks. */
    baby?: number;
    /** Number of weight variants generated for juvenile-stage networks. */
    juvenile?: number;
    /** Number of weight variants generated for adult-stage networks. */
    adult?: number;
  };

  /**
   * Regression guard micro-benchmark configuration.
   *
   * Controls how the paired CPU vs GPU benchmark decides whether the current
   * environment's GPU path is actually faster than CPU for small networks.
   */
  benchmark?: {
    /** Number of samples collected per backend during regression guard. */
    samples?: number;

    /**
     * Ratio threshold at which GPU is considered slower than CPU.
     *
     * If `gpuMedianMs / cpuMedianMs` exceeds this value, GPU is blacklisted.
     */
    gpuSlowerRatioThreshold?: number;

    /** Time-to-live for the GPU blacklist, in milliseconds. */
    blacklistTtlMs?: number;
  };
}

/**
 * CPU capability report returned by environment detection.
 *
 * CPU is always treated as an available fallback; this report records that
 * invariant and can carry an optional reason for diagnostics.
 */
export interface CPUStatus {
  /** True when the CPU backend is available as a fallback. */
  available: boolean;

  /** Optional human-readable explanation of the CPU fallback decision. */
  reason?: string;
}

/**
 * Runtime acceleration status snapshot.
 *
 * Combines the resolved capability reports with the selected acceleration mode so
 * callers can inspect why a particular backend is active. The optional
 * `gapReasons` field collects human-readable explanations for every backend
 * that could not be enabled.
 */
export interface AccelerationStatus extends AccelerationCapabilities {
  /** Selected acceleration mode after resolution. */
  mode: AccelerationMode;

  /** CPU fallback report; CPU is always available. */
  cpu?: CPUStatus;

  /** Human-readable reasons why each unavailable backend was disabled. */
  gapReasons?: string[];
}

/**
 * Workload description used to size a GPU buffer-set pool cap.
 *
 * The pool cap scales with the network size and variant count. Only
 * `nodeCount` is used by the current heuristic; `variantCount` and
 * `connectionCount` are kept for future motif-specific sizing and for
 * type-level documentation of the inputs that matter for buffer planning.
 *
 * @property nodeCount - Number of nodes in the evaluated network.
 * @property variantCount - Number of weight variants being evaluated in parallel.
 * @property connectionCount - Number of connections in the evaluated network.
 */
export interface BufferPoolWorkload {
  /** Number of nodes in the evaluated network. */
  nodeCount?: number;

  /** Number of weight variants being evaluated in parallel. */
  variantCount?: number;

  /** Number of connections in the evaluated network. */
  connectionCount?: number;
}

/**
 * Options that override the dynamically computed buffer-pool cap.
 *
 * Callers can either pin the cap to an exact byte budget with
 * `maxPooledBytes`, or supply a device limit so the computed cap is clamped
 * to one quarter of `maxBufferSize`. The remaining fields override the
 * heuristic defaults for benchmarking or unusual network motifs.
 *
 * @property maxPooledBytes - Exact caller cap; takes precedence over any computed value.
 * @property maxBufferSize - Device `maxBufferSize` used to clamp the computed cap.
 * @property avgDegree - Average in-degree used to estimate connection buffer needs.
 * @property bufferCount - Number of independent buffers retained per topology/variant entry.
 * @property float32Bytes - Bytes per Float32 element.
 * @property safetyFactor - Multiplier over the raw byte estimate to leave headroom.
 * @property minBytes - Floor below which the cap is never allowed to fall.
 */
export interface BufferPoolMaxPooledBytesOptions {
  /** Exact caller cap; takes precedence over any computed value. */
  maxPooledBytes?: number;

  /** Device `maxBufferSize` used to clamp the computed cap. */
  maxBufferSize?: number;

  /** Average in-degree used to estimate connection buffer needs. */
  avgDegree?: number;

  /** Number of independent buffers retained per topology/variant entry. */
  bufferCount?: number;

  /** Bytes per Float32 element. */
  float32Bytes?: number;

  /** Multiplier over the raw byte estimate to leave headroom. */
  safetyFactor?: number;

  /** Floor below which the cap is never allowed to fall. */
  minBytes?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Observer telemetry types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Typed fallback event emitted when the requested backend cannot be used and
 * the runtime falls back to a different backend.
 */
export interface AccelerationFallbackEvent {
  /** Backend that was originally requested. */
  requested: AccelerationMode;

  /** Backend that was actually selected. */
  chosen: AccelerationMode;

  /** Human-readable reason for the fallback. */
  reason: string;

  /** Timestamp from `performance.now()` or `Date.now()` when the event occurred. */
  timestamp: number;
}

/**
 * Typed telemetry event emitted after an inference call completes.
 *
 * Carries the active backend, inference duration, and optional resource counters
 * such as worker queue depth or pooled GPU buffer size.
 */
export interface AccelerationTelemetryEvent {
  /** Backend that ran the inference. */
  backend: AccelerationMode;

  /** Duration of the inference call in milliseconds. */
  inferenceMs: number;

  /** Optional worker queue depth at the time of the event. */
  queueDepth?: number;

  /** Optional number of bytes currently held in pooled GPU buffers. */
  bytesPooled?: number;
}

/**
 * Backend transition event emitted when the active acceleration backend changes.
 */
export interface AccelerationBackendChangeEvent {
  /** Previous active backend, or `null` on first selection. */
  previous: AccelerationMode | null;

  /** New active backend. */
  current: AccelerationMode;

  /** Optional reason for the transition. */
  reason?: string;

  /** Timestamp from `performance.now()` or `Date.now()` when the change occurred. */
  timestamp: number;
}

/**
 * Injectable callback surface for acceleration lifecycle events.
 *
 * All callbacks are optional. A consumer can subscribe to just fallback
 * diagnostics, just telemetry, or just backend transitions without needing to
 * implement the full surface.
 *
 * @example
 * ```ts
 * const observer: AccelerationObserver = {
 *   onFallback: (event) => console.warn('Fallback:', event.reason),
 *   onTelemetry: (event) => console.log(event.backend, event.inferenceMs),
 * };
 * ```
 */
export interface AccelerationObserver {
  /** Called whenever the runtime falls back from a requested backend. */
  onFallback?: (event: AccelerationFallbackEvent) => void;

  /** Called after an inference call with timing and queue metadata. */
  onTelemetry?: (event: AccelerationTelemetryEvent) => void;

  /** Called when the active acceleration backend changes. */
  onBackendChange?: (event: AccelerationBackendChangeEvent) => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight-variant evaluation types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * One candidate weight perturbation applied to a network connection.
 *
 * Variants are intentionally lightweight: they reference a connection by index
 * in the network's connection list and carry a signed delta. The evaluator
 * applies the delta, runs the network on the provided inputs, scores the
 * outputs, and restores the original weight before moving to the next variant.
 */
export interface WeightVariant {
  /** Index into `network.connections` identifying the connection to perturb. */
  weightIndex: number;

  /** Signed delta added to the connection's current weight during evaluation. */
  delta: number;
}

/**
 * Input batch used to score weight variants.
 *
 * Each inner array is one input vector passed to the network's `activate`
 * method. The same batch is evaluated for every variant so scores are
 * comparable.
 */
export type WeightVariantInputs = number[][];

/**
 * Target output vector used by the default scorer.
 *
 * The default scorer treats the first `target.length` outputs as the
 * prediction and computes negative mean squared error. Callers that supply a
 * custom scorer can interpret the target however they choose.
 */
export type WeightVariantTarget = number[];

/**
 * Scoring function that compares a stack of network outputs to a target vector.
 *
 * Higher scores are better. The default scorer returns negative mean squared
 * error so that a perfect prediction yields a score of zero and worse
 * predictions yield increasingly negative scores.
 */
export type VariantScorer = (
  outputs: readonly number[][],
  target: readonly number[],
) => number;

/**
 * Minimal network surface required by the variant evaluator.
 *
 * This interface deliberately avoids importing `Network` from
 * `src/architecture/` so the acceleration layer stays independent of the
 * architecture module. Any object with `nodes`, a connection list whose
 * entries expose a `weight`, and an `activate` method can be evaluated.
 */
export interface VariantEvaluationNetwork {
  /** Opaque node list; only its length matters for backend selection today. */
  nodes: readonly unknown[];

  /**
   * Connection list whose entries expose at least a mutable `weight` property.
   * Extra fields are ignored by the evaluator.
   */
  connections: readonly { weight: number }[];

  /**
   * Optional WebGPU device bound to this network surface.
   *
   * When present, callers may opt into the GPU activation path via
   * `{ useGPU: true }`. This field is optional so test doubles that only
   * exercise the CPU path remain structurally assignable.
   */
  gpuDevice?: GPUDevice | null;

  /**
   * Run one forward pass.
   *
   * Accepts the same input shapes as {@link Network.activate} so a live
   * `Network` instance is structurally assignable without importing from
   * `src/architecture/`. The second argument is ignored by the evaluator;
   * it is included only to satisfy the overload surface of real networks.
   *
   * May be synchronous or asynchronous; the evaluator normalizes the result
   * with `Promise.resolve`.
   *
   * @param input - Input vector or typed array.
   * @param trainingOrOptions - Optional training flag or activation options;
   *   ignored by the evaluator.
   */
  activate(
    input: number[] | Float32Array,
    trainingOrOptions?: boolean | { training?: boolean; useGPU?: boolean },
  ): number[] | Promise<number[] | Float32Array>;
}

/**
 * Result of evaluating a set of weight variants against a fixed input batch.
 */
export interface WeightVariantResult {
  /** Index of the highest-scoring variant in the input `variants` array. */
  bestIndex: number;

  /** Score of the best variant (higher is better). */
  bestScore: number;

  /** Per-variant scores in the same order as the input `variants` array. */
  scores: number[];

  /** Diagnostics about how the evaluation was performed. */
  metadata: {
    /** Backend that ran the evaluation, e.g. `'cpu'`. */
    backend: string;

    /** Number of variants that were evaluated. */
    variantCount: number;

    /** Largest absolute delta in the variant set, or `1` when none was given. */
    scaleDivisor: number;

    /** Either `'default'` for the negative-MSE scorer or `'custom'`. */
    scorer: string;
  };
}

/**
 * Async weight-variant evaluator signature.
 *
 * Implementations receive a network surface, a list of variants, an input batch,
 * a target vector, and optional scoring/config/observer hooks, and return a
 * promise of per-variant scores plus backend metadata.
 */
export type VariantEvaluator = (
  network: VariantEvaluationNetwork,
  variants: readonly WeightVariant[],
  inputs: WeightVariantInputs,
  target: WeightVariantTarget,
  scoreFn?: VariantScorer,
  seed?: number,
  config?: AccelerationConfig,
  observer?: AccelerationObserver,
) => Promise<WeightVariantResult>;
