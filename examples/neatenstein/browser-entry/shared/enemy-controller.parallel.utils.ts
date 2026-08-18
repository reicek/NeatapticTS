/**
 * Parallel inference infrastructure for enemy AI.
 *
 * Provides SAB-backed weight pools, tiered inference strategy resolution,
 * deterministic parallel dispatch, and barrier synchronization for
 * async enemy inference.
 *
 * ## Determinism Contract
 *
 * Each inference result is tagged with `(simTick, enemyIndex)`. The sim tick
 * does NOT advance until ALL enemy results for that tick are collected
 * (barrier). Results are applied in `enemyIndex` order, not completion order.
 * This ensures the same seed produces the same simulation regardless of
 * inference latency.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Minimum pool size for SAB-backed parallel inference.
 *
 * With SharedArrayBuffer-backed weight slots, the enemy count cap is raised
 * from 8 to 16-32. This constant defines the minimum pool capacity.
 */
export const ENEMY_INFERENCE_POOL_SIZE = 16;

/**
 * Threshold below which the InferenceChannel fallback is preferred over SAB.
 *
 * For small N (≤8), per-enemy InferenceChannel workers are simpler and
 * sufficient. Above this threshold, the SAB pool is primary.
 */
const INFERENCE_CHANNEL_THRESHOLD = 8;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Inference strategy tier.
 *
 * - `sab`: SharedArrayBuffer-backed weight slots in a SharedInferenceWorker.
 *   Primary path for 16-32 enemies.
 * - `channel`: Per-enemy InferenceChannel. Fallback for small N (≤8) or
 *   when SharedArrayBuffer is unavailable.
 * - `inline`: Synchronous inline activation. Final fallback when neither
 *   SAB nor InferenceChannel is available.
 */
export type InferenceStrategy = 'sab' | 'channel' | 'inline';

/**
 * A single inference result tagged with deterministic ordering metadata.
 *
 * The `(simTick, enemyIndex)` tag ensures results can be collected and
 * applied in `enemyIndex` order regardless of async completion order.
 */
export interface InferenceResult {
  /** Sim tick when this inference was dispatched. */
  simTick: number;
  /** Enemy index this result corresponds to. */
  enemyIndex: number;
  /** Inference output vector (e.g. MLP activations). */
  output: Float32Array;
}

/**
 * Pending inference dispatch state for barrier synchronization.
 *
 * Tracks which enemy results have been collected for the current sim tick.
 * The sim tick does not advance until all expected results arrive.
 */
export interface InferenceBarrierState {
  /** Current sim tick (barrier sync point). */
  simTick: number;
  /** Total enemies expected this tick. */
  expectedCount: number;
  /** Results collected so far, keyed by enemyIndex. */
  collected: Map<number, InferenceResult>;
}

/**
 * SAB-backed enemy inference pool.
 *
 * Allocates a SharedArrayBuffer (when available) for per-enemy weight slots.
 * Each enemy owns a weight slot; weights are copied in at respawn time.
 */
export interface EnemyInferencePool {
  /** Pool capacity (number of weight slots). */
  poolSize: number;
  /** Per-slot weight byte length. */
  weightByteLength: number;
  /**
   * SharedArrayBuffer backing the weight slots, or `null` when SAB is
   * unavailable (cross-origin isolation not configured).
   */
  sharedBuffer: SharedArrayBuffer | null;
  /**
   * Per-slot Float32Array views into the SAB. When SAB is unavailable,
   * these are standalone Float32Array instances.
   */
  slots: Float32Array[];
}

// ---------------------------------------------------------------------------
// Pool creation and weight loading
// ---------------------------------------------------------------------------

/**
 * Create a SAB-backed enemy inference pool.
 *
 * Allocates a {@link SharedArrayBuffer} when available (cross-origin
 * isolation configured) and creates per-slot `Float32Array` views. When SAB
 * is unavailable, falls back to standalone `Float32Array` instances.
 *
 * @param poolSize - Number of weight slots (defaults to
 *   {@link ENEMY_INFERENCE_POOL_SIZE}).
 * @param weightSize - Number of floats per weight vector (defaults to 128).
 * @returns A new {@link EnemyInferencePool}.
 */
export function createEnemyInferencePool(
  poolSize: number = ENEMY_INFERENCE_POOL_SIZE,
  weightSize: number = 128,
): EnemyInferencePool {
  const weightByteLength = weightSize * Float32Array.BYTES_PER_ELEMENT;
  const totalByteLength = poolSize * weightByteLength;

  let sharedBuffer: SharedArrayBuffer | null = null;
  const slots: Float32Array[] = [];

  // Attempt SharedArrayBuffer allocation (requires cross-origin isolation).
  if (
    typeof SharedArrayBuffer !== 'undefined'
  ) {
    try {
      sharedBuffer = new SharedArrayBuffer(totalByteLength);
      for (let i = 0; i < poolSize; i += 1) {
        slots.push(
          new Float32Array(
            sharedBuffer,
            i * weightByteLength,
            weightSize,
          ),
        );
      }
    } catch {
      // SAB allocation failed — fall back to standalone arrays.
      sharedBuffer = null;
    }
  }

  // Fallback: standalone Float32Array instances (no shared memory).
  if (slots.length === 0) {
    for (let i = 0; i < poolSize; i += 1) {
      slots.push(new Float32Array(weightSize));
    }
  }

  return {
    poolSize,
    weightByteLength,
    sharedBuffer,
    slots,
  };
}

/**
 * Load per-enemy weights into SAB pool slots.
 *
 * Copies each enemy's weight vector into the corresponding pool slot at
 * respawn time. The SAB slot is then read by the SharedInferenceWorker
 * during parallel activation.
 *
 * @param pool - The inference pool to load into.
 * @param perEnemyWeights - One Float32Array per enemy (from A4's selectVariant).
 */
export function loadEnemyWeightSlots(
  pool: EnemyInferencePool,
  perEnemyWeights: Float32Array[],
): void {
  const maxSlots = Math.min(perEnemyWeights.length, pool.poolSize);
  for (let i = 0; i < maxSlots; i += 1) {
    const source = perEnemyWeights[i];
    const slot = pool.slots[i];
    if (source.length <= slot.length) {
      slot.set(source);
    } else {
      // Source is larger than slot — copy first N elements.
      slot.set(source.subarray(0, slot.length));
    }
  }
}

// ---------------------------------------------------------------------------
// Strategy resolution
// ---------------------------------------------------------------------------

/**
 * Resolve the inference strategy based on enemy count and SAB availability.
 *
 * Tiered approach:
 * 1. **SAB** — primary for 16-32 enemies when SharedArrayBuffer is available.
 * 2. **Channel** — fallback for small N (≤8) or when SAB is unavailable.
 * 3. **Inline** — final fallback when neither SAB nor channel is available.
 *
 * @param enemyCount - Number of enemies to infer this tick.
 * @param sabAvailable - Whether SharedArrayBuffer is available.
 * @returns The resolved inference strategy tier.
 */
export function resolveInferenceStrategy(
  enemyCount: number,
  sabAvailable: boolean,
): InferenceStrategy {
  if (sabAvailable && enemyCount > INFERENCE_CHANNEL_THRESHOLD) {
    return 'sab';
  }
  if (enemyCount > 0) {
    return 'channel';
  }
  return 'inline';
}

// ---------------------------------------------------------------------------
// Parallel dispatch and barrier
// ---------------------------------------------------------------------------

/**
 * Per-enemy inference callback used by the inline strategy.
 *
 * Receives the enemy index and returns the inference output vector.
 * In the inline strategy, all inference is computed synchronously within
 * {@link dispatchParallelInference}, so the barrier is satisfied immediately.
 */
export type InferenceCallback = (enemyIndex: number) => Float32Array;

/**
 * Dispatch parallel inference for all enemies in a single tick.
 *
 * Each inference request is tagged with `(simTick, enemyIndex)` so results
 * can be collected and applied deterministically regardless of completion
 * order. The returned barrier state tracks pending results.
 *
 * For the `inline` strategy, when `inferenceFn` is provided, all enemy
 * results are computed synchronously and the `collected` map is fully
 * populated before returning. For `sab` and `channel` strategies, results
 * arrive asynchronously and must be collected via
 * {@link collectInferenceResult} before calling {@link awaitInferenceBarrier}.
 *
 * @param simTick - Current simulation tick.
 * @param enemyCount - Number of enemies to dispatch.
 * @param strategy - Resolved inference strategy.
 * @param inferenceFn - Optional inline inference callback (required for the
 *   `inline` strategy to populate results synchronously).
 * @returns Barrier state for collecting results.
 */
export function dispatchParallelInference(
  simTick: number,
  enemyCount: number,
  strategy: InferenceStrategy,
  inferenceFn?: InferenceCallback,
): InferenceBarrierState {
  const barrier: InferenceBarrierState = {
    simTick,
    expectedCount: enemyCount,
    collected: new Map<number, InferenceResult>(),
  };

  if (strategy === 'inline' && inferenceFn) {
    // Synchronous inline inference — compute all results immediately.
    for (let i = 0; i < enemyCount; i += 1) {
      const output = inferenceFn(i);
      barrier.collected.set(i, {
        simTick,
        enemyIndex: i,
        output,
      });
    }
  }
  // For 'sab' and 'channel' strategies, results are populated asynchronously
  // by worker callbacks that call collectInferenceResult on the barrier.

  return barrier;
}

/**
 * Collect a single asynchronous inference result into the barrier.
 *
 * Called by worker message handlers when an enemy inference result arrives.
 * Stale results from a previous sim tick are silently dropped. Results for
 * the current tick are stored keyed by `enemyIndex`.
 *
 * @param barrier - The barrier state to collect into.
 * @param result - The inference result from a worker.
 */
export function collectInferenceResult(
  barrier: InferenceBarrierState,
  result: InferenceResult,
): void {
  if (result.simTick !== barrier.simTick) {
    return; // stale result from a previous tick — drop
  }
  barrier.collected.set(result.enemyIndex, result);
}

/**
 * Await the inference barrier — collect all enemy results before advancing.
 *
 * The sim tick does NOT advance until ALL enemy inference results for that
 * tick are collected. Results are applied in `enemyIndex` order, not
 * completion order, preserving determinism.
 *
 * Throws when `collected.size < expectedCount` to prevent silent data loss.
 *
 * @param barrier - The barrier state from {@link dispatchParallelInference}.
 * @returns Results ordered by `enemyIndex`.
 * @throws {Error} When not all expected results have been collected.
 */
export function awaitInferenceBarrier(
  barrier: InferenceBarrierState,
): InferenceResult[] {
  if (barrier.collected.size < barrier.expectedCount) {
    throw new Error(
      `Inference barrier not satisfied: collected ${barrier.collected.size} of ${barrier.expectedCount} results for sim tick ${barrier.simTick}`,
    );
  }

  const results: InferenceResult[] = [];
  for (let i = 0; i < barrier.expectedCount; i += 1) {
    const result = barrier.collected.get(i);
    if (!result) {
      throw new Error(
        `Missing inference result for enemy index ${i} at sim tick ${barrier.simTick}`,
      );
    }
    results.push(result);
  }
  return results;
}