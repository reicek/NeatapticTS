/**
 * Minimal worker resource contract required by the generic inference pool.
 *
 * The pool owns worker lifecycle, not transport internals. Any persistent
 * predictor or worker can participate as long as it can release its resources
 * deterministically.
 */
export interface ParallelInferenceWorkerLike {
  /** Release the worker-side resources owned by this slot. */
  release(): Promise<void>;
}

/**
 * Configuration for one bounded parallel inference pool.
 *
 * The opener stays caller-owned so the pool can schedule shared-memory
 * workers, inference channels, or future worker-backed predictors without
 * coupling this boundary to one transport strategy.
 *
 * @typeParam TPayload - Payload or network shelf item assigned to one slot.
 * @typeParam TWorker - Persistent worker or predictor resource used by the slot.
 */
export interface ParallelInferencePoolOptions<
  TPayload,
  TWorker extends ParallelInferenceWorkerLike,
> {
  /** Maximum number of simultaneously active workers. */
  readonly workerCount?: number;
  /** Opens one persistent worker resource for the given payload. */
  readonly openWorker: (payload: TPayload) => Promise<TWorker> | TWorker;
}

type ParallelInferencePoolSlot<
  TPayload,
  TWorker extends ParallelInferenceWorkerLike,
> = {
  assignedPayload?: TPayload;
  worker?: TWorker;
};

const DEFAULT_PARALLEL_INFERENCE_WORKER_COUNT = 4;

/**
 * Generic bounded worker pool for ordered parallel inference tasks.
 *
 * The pool keeps persistent predictors warm, caps concurrent worker usage, and
 * rebuilds results in the caller's original order even when individual jobs
 * finish out of order.
 *
 * @typeParam TPayload - Payload or network shelf item assigned to one slot.
 * @typeParam TWorker - Persistent worker or predictor resource used by the slot.
 *
 * @example
 * ```ts
 * const pool = new ParallelInferencePool({
 *   openWorker: (payload) => openSharedInferenceWorker(payload),
 *   workerCount: 4,
 * });
 * await pool.initialize(payloads);
 * const results = await pool.evaluateOrderedBatch(payloads, async (worker) => {
 *   return worker.infer([0.25, 0.75]);
 * });
 * await pool.dispose();
 * ```
 */
export class ParallelInferencePool<
  TPayload,
  TWorker extends ParallelInferenceWorkerLike,
> {
  readonly #openWorker: (payload: TPayload) => Promise<TWorker> | TWorker;
  readonly #workerCount: number;
  #slots: Array<ParallelInferencePoolSlot<TPayload, TWorker>> = [];

  /**
   * Creates one bounded worker pool for ordered parallel inference tasks.
   *
   * @param options - Worker opener plus optional concurrency bound.
   */
  constructor(
    options: ParallelInferencePoolOptions<TPayload, TWorker>,
  ) {
    this.#openWorker = options.openWorker;
    this.#workerCount = Math.max(
      1,
      options.workerCount ?? resolveDefaultParallelInferenceWorkerCount(),
    );
  }

  /**
   * Prepares empty slot state for the next payload shelf.
   *
   * Existing workers are released before the new shelf becomes active so slot
   * reuse stays deterministic across generations or evaluation batches.
   *
   * @param payloads - Ordered payload shelf that may be evaluated next.
   * @returns Nothing.
   */
  async initialize(payloads: readonly TPayload[]): Promise<void> {
    await this.dispose();
    this.#slots = Array.from(
      { length: Math.min(this.#workerCount, Math.max(payloads.length, 1)) },
      () => ({}),
    );
  }

  /**
   * Evaluates a payload shelf through the bounded worker slots.
   *
   * The pool schedules work FIFO, lets each slot consume tasks until the queue
   * is empty, and returns results in the same order as the input payloads.
   *
   * @typeParam TResult - Result produced by one worker evaluation.
   * @param payloads - Ordered payload shelf to evaluate.
   * @param evaluator - Caller-owned evaluation logic for one warm worker slot.
   * @returns Results in the caller's original payload order.
   */
  async evaluateOrderedBatch<TResult>(
    payloads: readonly TPayload[],
    evaluator: (
      worker: TWorker,
      payload: TPayload,
      payloadIndex: number,
    ) => Promise<TResult>,
  ): Promise<TResult[]> {
    if (payloads.length === 0) {
      return [];
    }

    if (this.#needsInitialization(payloads.length)) {
      await this.initialize(payloads);
    }

    const orderedResults = new Array<TResult>(payloads.length);
    let nextPayloadIndex = 0;
    const activeSlots = this.#slots.slice(0, Math.min(this.#slots.length, payloads.length));

    await Promise.all(
      activeSlots.map(async (slot) => {
        while (nextPayloadIndex < payloads.length) {
          const payloadIndex = nextPayloadIndex;
          nextPayloadIndex += 1;
          const payload = payloads[payloadIndex];

          if (!payload) {
            continue;
          }

          const worker = await this.#prepareSlotForPayload(slot, payload);
          orderedResults[payloadIndex] = await evaluator(
            worker,
            payload,
            payloadIndex,
          );
        }
      }),
    );

    return orderedResults;
  }

  /**
   * Releases every active worker and clears the slot shelf.
   *
   * @returns Nothing.
   */
  async dispose(): Promise<void> {
    const activeSlots = this.#slots;
    this.#slots = [];

    await Promise.all(
      activeSlots.map(async (slot) => {
        if (slot.worker) {
          await slot.worker.release();
        }
      }),
    );
  }

  #needsInitialization(payloadCount: number): boolean {
    const targetSlotCount = Math.min(
      this.#workerCount,
      Math.max(payloadCount, 1),
    );

    return this.#slots.length !== targetSlotCount;
  }

  async #prepareSlotForPayload(
    slot: ParallelInferencePoolSlot<TPayload, TWorker>,
    payload: TPayload,
  ): Promise<TWorker> {
    if (slot.assignedPayload === payload && slot.worker) {
      return slot.worker;
    }

    if (slot.worker) {
      await slot.worker.release();
    }

    const worker = await this.#openWorker(payload);
    slot.assignedPayload = payload;
    slot.worker = worker;
    return worker;
  }
}

function resolveDefaultParallelInferenceWorkerCount(): number {
  return (
    globalThis.navigator?.hardwareConcurrency ??
    DEFAULT_PARALLEL_INFERENCE_WORKER_COUNT
  );
}