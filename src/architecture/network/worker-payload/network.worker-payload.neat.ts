import { evaluateInWorkers } from './network.worker-payload.batch';
import type { BatchEvaluationResult } from './network.worker-payload.batch';
import type { EvaluateInWorkersOptions } from './network.worker-payload.batch';
import type { ParallelInferencePool } from './network.worker-payload.pool';
import type { ParallelInferenceWorkerLike } from './network.worker-payload.pool';

type ScoreableGenome = {
  score?: number;
};

/**
 * Low-friction NEAT population-evaluation options built on top of `evaluateInWorkers(...)`.
 *
 * The factory keeps the public surface focused on population evaluation rather
 * than transport details. Callers can opt into parallel execution with
 * `parallel: true`, keep the local scoring delegate as the honest fallback,
 * and add worker-specific overrides only when they need them.
 *
 * @typeParam TGenome - Population genome type consumed by the NEAT fitness delegate.
 * @typeParam TPayload - Worker payload resolved for one genome.
 * @typeParam TWorker - Warm worker resource used during the parallel path.
 * @typeParam TResult - Result type produced for each genome evaluation.
 */
export interface NeatParallelPopulationEvaluatorOptions<
  TGenome extends ScoreableGenome,
  TPayload,
  TWorker extends ParallelInferenceWorkerLike,
  TResult,
> {
  /** Attempt worker-backed evaluation when true. */
  readonly parallel?: boolean;
  /** Local genome evaluator used for the honest fallback path. */
  readonly evaluateGenome: (
    genome: TGenome,
    genomeIndex: number,
  ) => Promise<TResult> | TResult;
  /** Optional ordered result assignment override. */
  readonly assignResult?: (
    genome: TGenome,
    result: TResult,
    genomeIndex: number,
  ) => void;
  /** Parallel worker evaluator used when worker execution is enabled. */
  readonly evaluateWithWorker?: EvaluateInWorkersOptions<
    TGenome,
    TPayload,
    TWorker,
    TResult
  >['evaluateWithWorker'];
  /** Opens one warm worker resource when the helper creates its own transient pool. */
  readonly openWorker?: EvaluateInWorkersOptions<
    TGenome,
    TPayload,
    TWorker,
    TResult
  >['openWorker'];
  /** Resolves the worker payload for one genome. Defaults to the genome itself. */
  readonly resolvePayload?: EvaluateInWorkersOptions<
    TGenome,
    TPayload,
    TWorker,
    TResult
  >['resolvePayload'];
  /** Optional prepared worker pool to reuse across repeated population runs. */
  readonly workerPool?: ParallelInferencePool<TPayload, TWorker>;
  /** Optional worker count used only when the helper creates a transient pool. */
  readonly workerCount?: number;
}

/**
 * Create a NEAT-compatible population fitness delegate on top of `evaluateInWorkers(...)`.
 *
 * The returned function matches `fitnessPopulation: true` evaluation mode: it
 * scores the whole population in one async call and writes ordered results back
 * onto the genomes in place.
 *
 * @typeParam TGenome - Population genome type consumed by the NEAT fitness delegate.
 * @typeParam TPayload - Worker payload resolved for one genome.
 * @typeParam TWorker - Warm worker resource used during the parallel path.
 * @typeParam TResult - Result type produced for each genome evaluation.
 * @param options - Boolean-first population-evaluation options plus advanced worker overrides.
 * @returns Population fitness delegate compatible with `fitnessPopulation: true`.
 *
 * @example
 * ```ts
 * const evaluatePopulation = createNeatParallelPopulationEvaluator({
 *   parallel: true,
 *   evaluateGenome: async (genome) => genome.score ?? 0,
 *   openWorker: (payload) => openSharedInferenceWorker(payload),
 *   evaluateWithWorker: async (worker, genome) => worker.infer(genome.activate([0, 1])),
 * });
 * ```
 */
export function createNeatParallelPopulationEvaluator<
  TGenome extends ScoreableGenome,
  TPayload = TGenome,
  TWorker extends ParallelInferenceWorkerLike = ParallelInferenceWorkerLike,
  TResult = number,
>(
  options: NeatParallelPopulationEvaluatorOptions<
    TGenome,
    TPayload,
    TWorker,
    TResult
  >,
): (population: TGenome[]) => Promise<void> {
  return async (population: TGenome[]): Promise<void> => {
    const batchResult = await evaluateInWorkers({
      inputs: population,
      evaluateLocally: (genome, genomeIndex) =>
        options.evaluateGenome(genome, genomeIndex),
      evaluateWithWorker: options.parallel
        ? options.evaluateWithWorker
        : undefined,
      openWorker: options.parallel ? options.openWorker : undefined,
      resolvePayload: options.resolvePayload,
      workerCount: options.parallel ? options.workerCount : undefined,
      workerPool: options.parallel ? options.workerPool : undefined,
    });

    applyPopulationEvaluationResults(population, batchResult, options.assignResult);
  };
}

function applyPopulationEvaluationResults<
  TGenome extends ScoreableGenome,
  TResult,
>(
  population: readonly TGenome[],
  batchResult: BatchEvaluationResult<TResult>,
  assignResult:
    | ((genome: TGenome, result: TResult, genomeIndex: number) => void)
    | undefined,
): void {
  population.forEach((genome, genomeIndex) => {
    const result = resolveRequiredPopulationEvaluationResult(
      batchResult.results,
      genomeIndex,
    );

    if (assignResult) {
      assignResult(genome, result, genomeIndex);
      return;
    }

    assignNumericGenomeScore(genome, result, genomeIndex);
  });
}

function resolveRequiredPopulationEvaluationResult<TResult>(
  results: readonly TResult[],
  genomeIndex: number,
): TResult {
  const result = results[genomeIndex];

  if (typeof result === 'undefined') {
    throw new Error(
      `Expected NEAT population evaluation result for genome ${String(genomeIndex)}.`,
    );
  }

  return result;
}

function assignNumericGenomeScore<TResult>(
  genome: ScoreableGenome,
  result: TResult,
  genomeIndex: number,
): void {
  if (typeof result !== 'number') {
    throw new Error(
      `createNeatParallelPopulationEvaluator requires assignResult when genome ${String(genomeIndex)} returns a non-numeric result.`,
    );
  }

  genome.score = result;
}