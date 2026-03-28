import Network from '../../network/network';
import { config } from '../../../config';
import Multi from '../../../multithreading/multi';
import type {
  TestWorkerInstance,
  TestWorkerConstructor,
} from '../../../multithreading/types';
import type {
  CostFunctionOrRef,
  EvolveCostFunction as CostFunction,
  FitnessSetup,
  PopulationFitnessFunction,
  PopulationWorkerEvaluationContext,
  SingleGenomeFitnessFunction,
  TrainingSample,
  WorkerTraversalContext,
} from '../network.types';
import {
  DEFAULT_THREAD_COUNT,
  type GenomeStructureCounts,
} from './network.evolve.utils.types';

/**
 * Cache for complexity penalty computations keyed by genome reference.
 */
const complexityCache: WeakMap<
  Network,
  { nodes: number; conns: number; gates: number; value: number }
> = new WeakMap();

/**
 * Compute structural complexity penalty scaled by growth.
 *
 * @param genome - Candidate network whose complexity to measure.
 * @param growth - Positive scalar controlling parsimony pressure.
 * @returns Complexity penalty.
 */
export function computeComplexityPenalty(
  genome: Network,
  growth: number,
): number {
  const structureCounts = getGenomeStructureCounts(genome);
  const cachedComplexity = getCachedComplexityBase(genome, structureCounts);
  const complexityBase =
    cachedComplexity ?? computeComplexityBase(genome, structureCounts);

  if (cachedComplexity === null)
    cacheComplexityBase(genome, structureCounts, complexityBase);

  return complexityBase * growth;
}

/**
 * Build a single-threaded genome fitness evaluator.
 *
 * @param set - Dataset of training samples.
 * @param cost - Cost function reference.
 * @param amount - Number of repeated evaluations.
 * @param growth - Complexity penalty scalar.
 * @returns Single-genome fitness function.
 */
export function buildSingleThreadFitness(
  set: TrainingSample[],
  cost: CostFunction,
  amount: number,
  growth: number,
): SingleGenomeFitnessFunction {
  return function evaluateSingleGenomeFitness(genome: Network): number {
    const accumulatedNegativeError = evaluateGenomeAmountTimes(
      genome,
      set,
      cost,
      amount,
    );
    if (accumulatedNegativeError === -Infinity) return -Infinity;

    const penalizedScore =
      accumulatedNegativeError - computeComplexityPenalty(genome, growth);
    const normalizedScore = penalizedScore / amount;
    return Number.isNaN(normalizedScore) ? -Infinity : normalizedScore;
  };
}

/**
 * Build worker-based population fitness setup.
 *
 * @param set - Dataset.
 * @param cost - Cost function or reference.
 * @param amount - Repetition count.
 * @param growth - Complexity penalty scalar.
 * @param threads - Desired worker count.
 * @param options - Evolution options object.
 * @returns Population fitness setup.
 */
export async function buildMultiThreadFitness(
  set: TrainingSample[],
  cost: CostFunctionOrRef,
  amount: number,
  growth: number,
  threads: number,
  options: Record<string, unknown>,
): Promise<FitnessSetup> {
  const serializedSet = Multi.serializeDataSet(set);
  const workerConstructor = await resolveTestWorkerConstructor();
  if (!workerConstructor)
    return createSingleThreadFallbackFitness(set, cost, amount, growth);

  const workers = spawnTestWorkers(
    workerConstructor,
    serializedSet,
    cost,
    threads,
  );
  const fitnessFunction = buildPopulationWorkerFitnessFunction(workers, growth);

  options.fitnessPopulation = true;
  installWorkerTerminationHook(options, workers);
  return { fitnessFunction, threads };
}

/**
 * Evaluate one genome with a worker and assign penalized score.
 *
 * @param worker - Worker instance.
 * @param genome - Genome under evaluation.
 * @param growth - Complexity penalty scalar.
 * @returns Promise resolving when score assignment completes.
 */
export async function evaluateGenomeWithWorker(
  worker: TestWorkerInstance,
  genome: Network,
  growth: number,
): Promise<void> {
  const rawResult = await Promise.resolve(
    worker.evaluate(
      genome as unknown as import('../../../multithreading/types').SerializableNetwork,
    ),
  );

  if (typeof rawResult !== 'number') return;

  const penalizedScore = -rawResult - computeComplexityPenalty(genome, growth);
  genome.score = Number.isNaN(rawResult) ? -Infinity : penalizedScore;
}

/**
 * Register worker termination hook onto options object.
 *
 * @param options - Evolve options object.
 * @param workers - Spawned worker instances.
 * @returns Nothing.
 */
export function installWorkerTerminationHook(
  options: Record<string, unknown>,
  workers: TestWorkerInstance[],
): void {
  (options as Record<string, unknown>)._workerTerminators =
    function terminateWorkers(): void {
      workers.forEach((worker) => {
        try {
          worker.terminate?.();
        } catch {
          // Ignore termination errors
        }
      });
    };
}

/**
 * Evaluate one genome repeatedly and accumulate negative error.
 *
 * @param genome - Genome under evaluation.
 * @param set - Dataset used for evaluation.
 * @param cost - Cost function reference.
 * @param amount - Number of repeated evaluations.
 * @returns Accumulated negative error or -Infinity on failure.
 */
function evaluateGenomeAmountTimes(
  genome: Network,
  set: TrainingSample[],
  cost: CostFunction,
  amount: number,
): number {
  let accumulatedNegativeError = 0;

  for (
    let evaluationIndex = 0;
    evaluationIndex < amount;
    evaluationIndex += 1
  ) {
    const evaluationError = evaluateGenomeErrorSafely(genome, set, cost);
    if (evaluationError === null) return -Infinity;
    accumulatedNegativeError -= evaluationError;
  }

  return accumulatedNegativeError;
}

/**
 * Evaluate one genome and return error, with warning-protected failure handling.
 *
 * @param genome - Genome under evaluation.
 * @param set - Dataset used for evaluation.
 * @param cost - Cost function reference.
 * @returns Error value, or null when evaluation fails.
 */
function evaluateGenomeErrorSafely(
  genome: Network,
  set: TrainingSample[],
  cost: CostFunction,
): number | null {
  try {
    return genome.test(set, cost).error;
  } catch (error: unknown) {
    warnGenomeEvaluationFailure(error);
    return null;
  }
}

/**
 * Emit warning when genome evaluation fails.
 *
 * @param error - Unknown evaluation error.
 * @returns Nothing.
 */
function warnGenomeEvaluationFailure(error: unknown): void {
  if (!config.warnings) return;
  console.warn(
    `Genome evaluation failed: ${
      (error && (error as Error).message) || error
    }. Penalizing with -Infinity fitness.`,
  );
}

/**
 * Resolve worker constructor for current runtime environment.
 *
 * @returns Worker constructor or null when unavailable.
 */
async function resolveTestWorkerConstructor(): Promise<TestWorkerConstructor | null> {
  try {
    const isNodeRuntime =
      typeof process !== 'undefined' &&
      !!(process.versions as Record<string, unknown> | undefined)?.node;
    if (isNodeRuntime && Multi.workers?.getNodeTestWorker)
      return await Multi.workers.getNodeTestWorker();
    if (!isNodeRuntime && Multi.workers?.getBrowserTestWorker)
      return await Multi.workers.getBrowserTestWorker();
    return null;
  } catch (error: unknown) {
    if (config.warnings)
      console.warn(
        'Failed to load worker class; falling back to single-thread path:',
        (error as Error)?.message || error,
      );
    return null;
  }
}

/**
 * Build single-thread fallback fitness setup.
 *
 * @param set - Dataset.
 * @param cost - Cost function.
 * @param amount - Repetition count.
 * @param growth - Complexity penalty scalar.
 * @returns Single-thread fitness setup.
 */
function createSingleThreadFallbackFitness(
  set: TrainingSample[],
  cost: CostFunctionOrRef,
  amount: number,
  growth: number,
): FitnessSetup {
  return {
    fitnessFunction: buildSingleThreadFitness(
      set,
      cost as CostFunction,
      amount,
      growth,
    ),
    threads: DEFAULT_THREAD_COUNT,
  };
}

/**
 * Spawn worker instances up to requested thread count.
 *
 * @param workerConstructor - Worker constructor resolved for runtime.
 * @param serializedSet - Serialized dataset shared with workers.
 * @param cost - Cost function or cost reference.
 * @param threads - Requested worker count.
 * @returns Spawned worker instances.
 */
function spawnTestWorkers(
  workerConstructor: TestWorkerConstructor,
  serializedSet: ReturnType<typeof Multi.serializeDataSet>,
  cost: CostFunctionOrRef,
  threads: number,
): TestWorkerInstance[] {
  const spawnedWorkers: TestWorkerInstance[] = [];
  const costName = resolveCostName(cost);

  for (let threadIndex = 0; threadIndex < threads; threadIndex += 1) {
    try {
      spawnedWorkers.push(
        new workerConstructor(serializedSet, { name: costName }),
      );
    } catch (error: unknown) {
      if (config.warnings) console.warn('Worker spawn failed', error);
    }
  }

  return spawnedWorkers;
}

/**
 * Resolve serializable cost name for worker payload.
 *
 * @param cost - Cost function or cost reference.
 * @returns Cost name string.
 */
function resolveCostName(cost: CostFunctionOrRef): string {
  return (
    (typeof cost === 'function' ? cost.name : cost.name) ||
    cost.toString?.() ||
    'cost'
  );
}

/**
 * Build population-level fitness function powered by worker queue.
 *
 * @param workers - Spawned worker instances.
 * @param growth - Complexity penalty scalar.
 * @returns Population-level fitness function.
 */
function buildPopulationWorkerFitnessFunction(
  workers: TestWorkerInstance[],
  growth: number,
): PopulationFitnessFunction {
  return function evaluatePopulationWithWorkers(
    population: Network[],
  ): Promise<void> {
    return new Promise<void>((resolveEvaluationPromise) => {
      const evaluationContext = createPopulationWorkerEvaluationContext(
        workers,
        population,
        growth,
        resolveEvaluationPromise,
      );
      if (hasNoWorkers(evaluationContext)) {
        resolveEvaluation(evaluationContext);
        return;
      }

      startWorkerTraversal(evaluationContext);
    });
  };
}

/**
 * Creates the shared evaluation context for one population run.
 *
 * @param sourceWorkers - Worker pool.
 * @param sourcePopulation - Population to evaluate.
 * @param sourceGrowth - Complexity penalty scalar.
 * @param sourceResolve - Promise resolver.
 * @returns Population evaluation context.
 */
function createPopulationWorkerEvaluationContext(
  sourceWorkers: TestWorkerInstance[],
  sourcePopulation: Network[],
  sourceGrowth: number,
  sourceResolve: () => void,
): PopulationWorkerEvaluationContext {
  return {
    workers: sourceWorkers,
    population: sourcePopulation,
    nextGenomeIndex: 0,
    activeWorkerCount: sourceWorkers.length,
    growth: sourceGrowth,
    resolve: sourceResolve,
  };
}

/**
 * Checks whether there are workers available to process genomes.
 *
 * @param context - Population evaluation context.
 * @returns True when worker pool is empty.
 */
function hasNoWorkers(context: PopulationWorkerEvaluationContext): boolean {
  return context.workers.length === 0;
}

/**
 * Starts traversal loops for all workers.
 *
 * @param context - Population evaluation context.
 * @returns Nothing.
 */
function startWorkerTraversal(
  context: PopulationWorkerEvaluationContext,
): void {
  context.workers.forEach((worker) => {
    const traversalContext = createWorkerTraversalContext(context, worker);
    runWorkerTraversalStep(traversalContext);
  });
}

/**
 * Creates traversal context for one worker.
 *
 * @param context - Population evaluation context.
 * @param worker - Worker instance.
 * @returns Worker traversal context.
 */
function createWorkerTraversalContext(
  context: PopulationWorkerEvaluationContext,
  worker: TestWorkerInstance,
): WorkerTraversalContext {
  return { evaluationContext: context, worker };
}

/**
 * Runs one asynchronous traversal step for a worker.
 *
 * @param traversalContext - Worker traversal context.
 * @returns Nothing.
 */
function runWorkerTraversalStep(
  traversalContext: WorkerTraversalContext,
): void {
  const queuedGenome = claimNextGenome(traversalContext.evaluationContext);
  if (!queuedGenome) {
    finalizeWorker(traversalContext.evaluationContext);
    return;
  }

  evaluateGenomeWithWorker(
    traversalContext.worker,
    queuedGenome,
    traversalContext.evaluationContext.growth,
  )
    .then(() => runWorkerTraversalStep(traversalContext))
    .catch(() => runWorkerTraversalStep(traversalContext));
}

/**
 * Claims the next genome index from shared queue state.
 *
 * @param context - Population evaluation context.
 * @returns Next genome, or undefined when queue is exhausted.
 */
function claimNextGenome(
  context: PopulationWorkerEvaluationContext,
): Network | undefined {
  const queuedGenome = context.population.at(context.nextGenomeIndex);
  if (!queuedGenome) {
    return undefined;
  }

  context.nextGenomeIndex += 1;
  return queuedGenome;
}

/**
 * Marks one worker as completed and resolves when all workers finish.
 *
 * @param context - Population evaluation context.
 * @returns Nothing.
 */
function finalizeWorker(context: PopulationWorkerEvaluationContext): void {
  context.activeWorkerCount -= 1;
  if (context.activeWorkerCount === 0) {
    resolveEvaluation(context);
  }
}

/**
 * Resolves the population evaluation promise.
 *
 * @param context - Population evaluation context.
 * @returns Nothing.
 */
function resolveEvaluation(context: PopulationWorkerEvaluationContext): void {
  context.resolve();
}

/**
 * Get structural counts used by complexity heuristic.
 *
 * @param genome - Candidate network whose structure is being measured.
 * @returns Structural counts used for complexity computation.
 */
function getGenomeStructureCounts(genome: Network): GenomeStructureCounts {
  return {
    nodeCount: genome.nodes.length,
    connectionCount: genome.connections.length,
    gateCount: genome.gates.length,
  };
}

/**
 * Retrieve cached complexity base if cached structure counts still match.
 *
 * @param genome - Candidate network whose cached complexity is queried.
 * @param structureCounts - Current structural counts.
 * @returns Cached complexity base or null when cache miss occurs.
 */
function getCachedComplexityBase(
  genome: Network,
  structureCounts: GenomeStructureCounts,
): number | null {
  const cachedComplexity = complexityCache.get(genome);
  if (!cachedComplexity) return null;

  const matchesNodeCount = cachedComplexity.nodes === structureCounts.nodeCount;
  const matchesConnectionCount =
    cachedComplexity.conns === structureCounts.connectionCount;
  const matchesGateCount = cachedComplexity.gates === structureCounts.gateCount;
  if (!matchesNodeCount || !matchesConnectionCount || !matchesGateCount)
    return null;

  return cachedComplexity.value;
}

/**
 * Compute non-scaled complexity base from structural counts.
 *
 * @param genome - Candidate network whose complexity to compute.
 * @param structureCounts - Current structural counts.
 * @returns Base complexity value before growth scaling.
 */
function computeComplexityBase(
  genome: Network,
  structureCounts: GenomeStructureCounts,
): number {
  const hiddenNodeCount =
    structureCounts.nodeCount - genome.input - genome.output;
  return (
    hiddenNodeCount +
    structureCounts.connectionCount +
    structureCounts.gateCount
  );
}

/**
 * Store complexity base cache entry for future reuse.
 *
 * @param genome - Candidate network used as cache key.
 * @param structureCounts - Current structural counts.
 * @param complexityBase - Computed base complexity value.
 * @returns Nothing.
 */
function cacheComplexityBase(
  genome: Network,
  structureCounts: GenomeStructureCounts,
  complexityBase: number,
): void {
  complexityCache.set(genome, {
    nodes: structureCounts.nodeCount,
    conns: structureCounts.connectionCount,
    gates: structureCounts.gateCount,
    value: complexityBase,
  });
}
