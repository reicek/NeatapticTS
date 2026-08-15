import {
  ParallelInferencePool,
  type ParallelInferencePoolOptions,
  type ParallelInferenceWorkerLike,
} from './network.worker-payload.pool';

/**
 * Ordered result envelope returned by `evaluateInWorkers(...)`.
 *
 * The helper preserves input order even when individual worker tasks finish out
 * of order, and it reports stable task ids so callers can correlate results
 * with their original batch shelf without reconstructing indexes manually.
 *
 * @typeParam TResult - Result type produced for each input task.
 */
export interface BatchEvaluationResult<TResult> {
  /** Total elapsed wall-clock time for the batch helper call. */
  readonly elapsedMs: number;
  /** Whether the batch ran through workers or the explicit local fallback. */
  readonly mode: 'parallel' | 'single-thread';
  /** Ordered results aligned to the original input shelf. */
  readonly results: TResult[];
  /** Stable task ids aligned to the original input indexes. */
  readonly taskIds: number[];
}

/**
 * Generic ordered batch-evaluation options for worker and fallback execution.
 *
 * Keep the transport substrate caller-owned: this helper schedules and assembles
 * deterministic results, but the caller still decides how a worker is opened,
 * how a task is scored, and what the single-thread fallback should do.
 *
 * @typeParam TInput - Logical task item supplied by the caller.
 * @typeParam TPayload - Worker payload prepared from one logical input.
 * @typeParam TWorker - Warm worker resource used during the parallel path.
 * @typeParam TResult - Result type produced for each input task.
 */
export interface EvaluateInWorkersOptions<
  TInput,
  TPayload,
  TWorker extends ParallelInferenceWorkerLike,
  TResult,
> {
  /** Ordered task shelf to evaluate. */
  readonly inputs: readonly TInput[];
  /** Optional local fallback that preserves semantics when workers are unavailable. */
  readonly evaluateLocally?: (
    input: TInput,
    inputIndex: number,
  ) => Promise<TResult> | TResult;
  /** Parallel worker evaluator used when a pool or opener is available. */
  readonly evaluateWithWorker?: (
    worker: TWorker,
    input: TInput,
    inputIndex: number,
    payload: TPayload,
  ) => Promise<TResult>;
  /** Opens one warm worker resource when the helper creates its own transient pool. */
  readonly openWorker?: ParallelInferencePoolOptions<
    TPayload,
    TWorker
  >['openWorker'];
  /** Optional prepared worker pool to reuse across repeated batch calls. */
  readonly workerPool?: ParallelInferencePool<TPayload, TWorker>;
  /** Optional worker count used only when the helper creates a transient pool. */
  readonly workerCount?: number;
  /** Resolves the worker payload for one logical input. Defaults to the input itself. */
  readonly resolvePayload?: (input: TInput, inputIndex: number) => TPayload;
}

/**
 * Evaluates an ordered batch through workers when available, otherwise locally.
 *
 * This helper keeps the honest async boundary visible: callers provide worker
 * opening plus task execution when they want parallelism, and they also provide
 * a local fallback when they need the same semantics in worker-less hosts.
 *
 * @typeParam TInput - Logical task item supplied by the caller.
 * @typeParam TPayload - Worker payload prepared from one logical input.
 * @typeParam TWorker - Warm worker resource used during the parallel path.
 * @typeParam TResult - Result type produced for each input task.
 * @param options - Ordered inputs plus worker and fallback execution hooks.
 * @returns Ordered batch results with elapsed time and stable task ids.
 * @throws {Error} When neither worker execution nor a local fallback evaluator is available.
 *
 * @example
 * ```ts
 * const batchResult = await evaluateInWorkers({
 *   inputs: payloads,
 *   openWorker: (payload) => openSharedInferenceWorker(payload),
 *   evaluateWithWorker: (worker, _payload) => worker.infer([0.25, 0.75]),
 * });
 * ```
 */
export async function evaluateInWorkers<
  TInput,
  TPayload,
  TWorker extends ParallelInferenceWorkerLike,
  TResult,
>(
  options: EvaluateInWorkersOptions<TInput, TPayload, TWorker, TResult>,
): Promise<BatchEvaluationResult<TResult>> {
  const taskIds = options.inputs.map((_input, inputIndex) => inputIndex);
  const startedAt = resolveElapsedTimeStart();

  if (canUseParallelEvaluation(options)) {
    const payloads = options.inputs.map((input, inputIndex) =>
      resolveBatchPayload(options, input, inputIndex),
    );
    const openWorker = options.openWorker;
    const workerPool =
      options.workerPool ??
      new ParallelInferencePool<TPayload, TWorker>({
        openWorker: openWorker as ParallelInferencePoolOptions<
          TPayload,
          TWorker
        >['openWorker'],
        workerCount: options.workerCount,
      });

    try {
      const results = await workerPool.evaluateOrderedBatch(
        payloads,
        async (worker, payload, payloadIndex) =>
          options.evaluateWithWorker(
            worker,
            options.inputs[payloadIndex] as TInput,
            payloadIndex,
            payload,
          ),
      );

      return {
        elapsedMs: resolveElapsedTimeDuration(startedAt),
        mode: 'parallel',
        results,
        taskIds,
      };
    } finally {
      if (!options.workerPool) {
        await workerPool.dispose();
      }
    }
  }

  if (!options.evaluateLocally) {
    throw new Error(
      'evaluateInWorkers requires either worker execution or a local fallback evaluator.',
    );
  }

  const results = await Promise.all(
    options.inputs.map(async (input, inputIndex) =>
      options.evaluateLocally!(input, inputIndex),
    ),
  );

  return {
    elapsedMs: resolveElapsedTimeDuration(startedAt),
    mode: 'single-thread',
    results,
    taskIds,
  };
}

function canUseParallelEvaluation<
  TInput,
  TPayload,
  TWorker extends ParallelInferenceWorkerLike,
  TResult,
>(
  options: EvaluateInWorkersOptions<TInput, TPayload, TWorker, TResult>,
): options is
  | (EvaluateInWorkersOptions<TInput, TPayload, TWorker, TResult> & {
      evaluateWithWorker: NonNullable<
        EvaluateInWorkersOptions<
          TInput,
          TPayload,
          TWorker,
          TResult
        >['evaluateWithWorker']
      >;
      openWorker: NonNullable<
        EvaluateInWorkersOptions<
          TInput,
          TPayload,
          TWorker,
          TResult
        >['openWorker']
      >;
    })
  | (EvaluateInWorkersOptions<TInput, TPayload, TWorker, TResult> & {
      evaluateWithWorker: NonNullable<
        EvaluateInWorkersOptions<
          TInput,
          TPayload,
          TWorker,
          TResult
        >['evaluateWithWorker']
      >;
      workerPool: NonNullable<
        EvaluateInWorkersOptions<
          TInput,
          TPayload,
          TWorker,
          TResult
        >['workerPool']
      >;
    }) {
  return Boolean(
    options.evaluateWithWorker && (options.workerPool || options.openWorker),
  );
}

function resolveBatchPayload<
  TInput,
  TPayload,
  TWorker extends ParallelInferenceWorkerLike,
  TResult,
>(
  options: EvaluateInWorkersOptions<TInput, TPayload, TWorker, TResult>,
  input: TInput,
  inputIndex: number,
): TPayload {
  if (options.resolvePayload) {
    return options.resolvePayload(input, inputIndex);
  }

  return input as unknown as TPayload;
}

function resolveElapsedTimeStart(): number {
  return globalThis.performance?.now?.() ?? Date.now();
}

function resolveElapsedTimeDuration(startedAt: number): number {
  return (globalThis.performance?.now?.() ?? Date.now()) - startedAt;
}
