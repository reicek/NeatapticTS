import Network from '../../network';
import * as methods from '../../../methods/methods';
import { config } from '../../../config';
import Multi from '../../../multithreading/multi';
import type {
  TestWorkerInstance,
  TestWorkerConstructor,
} from '../../../multithreading/types';
import type {
  CostFunctionOrRef,
  EvolutionConfig,
  EvolutionFitnessFunction,
  EvolutionLoopState,
  EvolutionSettings,
  EvolutionStopConditions,
  EvolveCostFunction as CostFunction,
  EvolveOptions,
  FitnessSetup,
  NeatRuntime,
  PopulationFitnessFunction,
  PopulationWorkerEvaluationContext,
  SingleGenomeFitnessFunction,
  TrainingSample,
  WorkerTraversalContext,
} from '../network.types';

/**
 * Shared dataset compatibility error message.
 */
const DATASET_COMPATIBILITY_ERROR_MESSAGE =
  'Dataset is invalid or dimensions do not match network input/output size!';

/**
 * Shared evolve stopping-condition validation error.
 */
const STOPPING_CONDITION_REQUIRED_ERROR_MESSAGE =
  'At least one stopping condition (`iterations` or `error`) must be specified for evolution.';

/**
 * Default target error used when omitted.
 */
const DEFAULT_TARGET_ERROR = 0.05;

/**
 * Default complexity growth penalty.
 */
const DEFAULT_GROWTH = 0.0001;

/**
 * Default repeated evaluation amount.
 */
const DEFAULT_EVALUATION_AMOUNT = 1;

/**
 * Default logging frequency value.
 */
const DEFAULT_LOG_INTERVAL = 0;

/**
 * Default single-thread worker count.
 */
const DEFAULT_THREAD_COUNT = 1;

/**
 * Sentinel target error indicating that error-based stopping is disabled.
 */
const DISABLED_TARGET_ERROR = -1;

/**
 * Explicit zero-iteration value.
 */
const ZERO_ITERATIONS = 0;

/**
 * Population threshold considered "small" for mutation heuristics.
 */
const SMALL_POPULATION_THRESHOLD = 10;

/**
 * Mutation rate fallback used for very small populations.
 */
const SMALL_POPULATION_MUTATION_RATE = 0.5;

/**
 * Mutation amount fallback used for very small populations.
 */
const SMALL_POPULATION_MUTATION_AMOUNT = 1;

/**
 * Maximum consecutive invalid errors tolerated before loop abort.
 */
const MAX_CONSECUTIVE_INVALID_ERRORS = 5;

/**
 * Cache for complexity penalty computations keyed by genome (Network) reference.
 * We store counts used to derive a simple structural complexity measure so repeated
 * invocations during a generation avoid recomputing the same base value.
 */
const _complexityCache: WeakMap<
  Network,
  { nodes: number; conns: number; gates: number; value: number }
> = new WeakMap();

/**
 * Compute a structural complexity penalty scaled by a growth factor.
 *
 * Complexity heuristic:
 *   (hidden nodes) + (connections) + (gates)
 * hidden nodes = total nodes - input - output (to avoid penalizing fixed I/O interface size).
 *
 * Rationale: Encourages minimal / parsimonious networks by subtracting a term from fitness
 * proportional to network size, counteracting bloat. Growth hyper‑parameter tunes pressure.
 *
 * Caching strategy: We memoize the base complexity (pre‑growth scaling) per genome when its
 * structural counts (nodes / connections / gates) are unchanged. This is safe because only
 * structural mutations alter these counts, and those invalidate earlier entries naturally
 * (since mutated genomes are distinct object references in typical NEAT flows).
 *
 * @param genome - Candidate network whose complexity to measure.
 * @param growth - Positive scalar controlling strength of parsimony pressure.
 * @returns Complexity * growth (used directly to subtract from fitness score).
 */
function computeComplexityPenalty(genome: Network, growth: number): number {
  const structureCounts = getGenomeStructureCounts(genome);
  const cachedComplexity = getCachedComplexityBase(genome, structureCounts);
  const complexityBase =
    cachedComplexity ?? computeComplexityBase(genome, structureCounts);

  if (cachedComplexity === null)
    cacheComplexityBase(genome, structureCounts, complexityBase);

  return complexityBase * growth;
}

/**
 * Get structural counts used by complexity heuristic.
 *
 * @param genome - Candidate network whose structure is being measured.
 * @returns Structural counts used for complexity computation.
 */
function getGenomeStructureCounts(genome: Network): {
  nodeCount: number;
  connectionCount: number;
  gateCount: number;
} {
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
  structureCounts: {
    nodeCount: number;
    connectionCount: number;
    gateCount: number;
  },
): number | null {
  const cachedComplexity = _complexityCache.get(genome);
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
  structureCounts: {
    nodeCount: number;
    connectionCount: number;
    gateCount: number;
  },
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
  structureCounts: {
    nodeCount: number;
    connectionCount: number;
    gateCount: number;
  },
  complexityBase: number,
): void {
  _complexityCache.set(genome, {
    nodes: structureCounts.nodeCount,
    conns: structureCounts.connectionCount,
    gates: structureCounts.gateCount,
    value: complexityBase,
  });
}

/**
 * Build a single-threaded fitness evaluation function (classic NEAT style) evaluating a genome
 * over the provided dataset and returning a scalar score where higher is better.
 *
 * Fitness Definition:
 *   fitness = -averageError - complexityPenalty
 * We accumulate negative error (so lower error => higher fitness) over `amount` independent
 * evaluations (amount>1 can smooth stochastic evaluation noise) then subtract complexity penalty.
 *
 * Error handling: If evaluation throws (numerical instability, internal error) we return -Infinity
 * so such genomes are strongly disfavored.
 *
 * @param set - Dataset of training samples.
 * @param cost - Cost function reference (should expose error computation in genome.test).
 * @param amount - Number of repeated evaluations to average.
 * @param growth - Complexity penalty scalar.
 * @returns Function mapping a Network genome to a numeric fitness.
 */
function buildSingleThreadFitness(
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

  for (let iterationIndex = 0; iterationIndex < amount; iterationIndex++) {
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
 * Build a multi-threaded (worker-based) population fitness evaluator if worker infrastructure is available.
 *
 * Strategy:
 *  - Attempt to dynamically obtain a Worker constructor (node or browser variant).
 *  - If not possible, gracefully fall back to single-thread evaluation.
 *  - Spawn N workers (threads) each capable of evaluating genomes by calling worker.evaluate(genome).
 *  - Provide a fitness function that takes the whole population and returns a Promise that resolves
 *    when all queued genomes have been processed. Each genome's score is written in-place.
 *
 * Implementation details:
 *  - Queue: simple FIFO (array shift) suffices because ordering is not critical.
 *  - Robustness: Each worker evaluation is wrapped with error handling to prevent a single failure
 *    from stalling the batch; failed evaluations simply proceed to next genome.
 *  - Complexity penalty applied after raw result retrieval: genome.score = -result - penalty.
 *
 * Returned metadata sets options.fitnessPopulation=true so downstream NEAT logic treats the fitness
 * function as operating over the entire population at once (rather than per-genome).
 *
 * @param set - Dataset.
 * @param cost - Cost function.
 * @param amount - Repetition count (unused directly here; assumed handled inside worker.evaluate result metric if needed).
 * @param growth - Complexity penalty scalar.
 * @param threads - Desired worker count.
 * @param options - Evolution options object (mutated to add cleanup hooks & flags).
 * @returns Object with fitnessFunction (population evaluator) and resolved thread count.
 */
async function buildMultiThreadFitness(
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
    threads: 1,
  };
}

/**
 * Spawn worker instances up to requested thread count.
 *
 * @param workerConstructor - Worker constructor resolved for current runtime.
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

  for (let threadIndex = 0; threadIndex < threads; threadIndex++) {
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
    return new Promise<void>((resolve) => {
      const evaluationContext = createPopulationWorkerEvaluationContext(
        workers,
        population,
        growth,
        resolve,
      );
      if (hasNoWorkers(evaluationContext)) {
        resolveEvaluation(evaluationContext);
        return;
      }

      startWorkerTraversal(evaluationContext);
    });
  };

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
}

/**
 * Evaluate one genome with a worker and assign penalized score.
 *
 * @param worker - Worker instance.
 * @param genome - Genome under evaluation.
 * @param growth - Complexity penalty scalar.
 * @returns Promise resolving once score assignment completes.
 */
async function evaluateGenomeWithWorker(
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
function installWorkerTerminationHook(
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
 * Evolve (optimize) the current network's topology and weights using a NEAT-like evolutionary loop
 * until a stopping criterion (target error or max iterations) is met.
 *
 * High-level process:
 *  1. Validate dataset shape (input/output vector sizes must match network I/O counts).
 *  2. Normalize / default option values and construct an internal configuration summary.
 *  3. Build appropriate fitness evaluation function (single or multi-thread).
 *  4. Initialize a Neat population (optionally with speciation) seeded by this network.
 *  5. Iteratively call neat.evolve():
 *       - Retrieve fittest genome + its fitness.
 *       - Derive an error metric from fitness (inverse relationship considering complexity penalty).
 *       - Track best genome overall (elitism) and perform logging/scheduling callbacks.
 *       - Break if error criterion satisfied or iterations exceeded.
 *  6. Replace this network's internal structural arrays with the best discovered genome's (in-place upgrade).
 *  7. Cleanup any worker threads and report final statistics.
 *
 * Fitness / Error relationship:
 *   fitness = -error - complexityPenalty  =>  error = -(fitness - complexityPenalty)
 * We recompute error from the stored fitness plus penalty to ensure consistent reporting.
 *
 * Resilience strategies:
 *  - Guard against infinite / NaN errors; after MAX_INF consecutive invalid errors we abort.
 *  - Fallback for tiny populations: increase mutation aggressiveness to prevent premature convergence.
 *
 * @param this - Bound {@link Network} instance being evolved in-place.
 * @param set - Supervised dataset (array of {input, output}).
 * @param options - Evolution options (see README / docs). Key fields include:
 *    - iterations: maximum generations (if omitted must supply error target)
 *    - error: target error threshold (if omitted must supply iterations)
 *    - growth: complexity penalty scaling
 *    - amount: number of score evaluations (averaged) per genome
 *    - threads: desired worker count (>=2 enables multi-thread path if available)
 *    - popsize / populationSize: population size
 *    - schedule: { iterations: number, function: (ctx) => void } periodic callback
 *    - log: generation interval for console logging
 *    - clear: whether to call network.clear() after adopting best genome
 * @returns Summary object { error, iterations, time(ms) }.
 * @throws If dataset is empty or dimensionally incompatible, or if neither iterations nor error is specified.
 */
export async function evolveNetwork(
  this: Network,
  set: TrainingSample[],
  options: EvolveOptions = {},
): Promise<{ error: number; iterations: number; time: number }> {
  // Step 1: Validate dataset shape and initialize normalized options.
  assertEvolutionDatasetCompatibility(this, set);
  const normalizedOptions = getNormalizedOptions(options);
  const startTime = Date.now();

  // Step 2: Resolve scalar evolution settings and stopping rules.
  const settings = resolveEvolutionSettings(normalizedOptions);
  const stopConditions = resolveStopConditions(
    normalizedOptions,
    settings.targetError,
  );

  // Step 3: Keep summarized configuration for optional introspection.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const evolutionConfig = createEvolutionConfig(settings);

  // Step 4: Prepare fitness evaluation function (single-thread or worker-based).
  const fitnessSetup = await prepareFitnessFunction(
    set,
    settings,
    normalizedOptions,
  );

  // Step 5: Configure NEAT options and create NEAT instance.
  configureNeatOptions(this, normalizedOptions);
  const neat = await createNeatInstance(
    this,
    fitnessSetup.fitnessFunction,
    normalizedOptions,
  );
  warnIfNoBestGenomeMayOccur(neat, normalizedOptions);
  applySmallPopulationHeuristics(neat, normalizedOptions);

  // Step 6: Run evolution loop and adopt best discovered genome if available.
  const loopResult = await runEvolutionLoop(
    neat,
    settings,
    stopConditions.targetError,
  );
  adoptBestGenomeOrWarn(this, neat, loopResult.bestGenome, settings.clear);

  // Step 7: Cleanup worker resources and return summary.
  terminateWorkersSafely(normalizedOptions);
  return buildEvolutionSummary(loopResult.error, neat.generation, startTime);

  /**
   * Validate dataset existence and dimensional compatibility with network I/O.
   *
   * @param network - Network being evolved.
   * @param dataSet - Supervised dataset.
   * @returns Nothing.
   */
  function assertEvolutionDatasetCompatibility(
    network: Network,
    dataSet: TrainingSample[],
  ): void {
    if (!dataSet || dataSet.length === 0) {
      throw new Error(DATASET_COMPATIBILITY_ERROR_MESSAGE);
    }

    const firstSample = dataSet[0];
    const inputMatches = firstSample.input.length === network.input;
    const outputMatches = firstSample.output.length === network.output;
    if (!inputMatches || !outputMatches) {
      throw new Error(DATASET_COMPATIBILITY_ERROR_MESSAGE);
    }
  }

  /**
   * Ensure options object exists.
   *
   * @param evolveOptions - Incoming evolve options.
   * @returns Safe options object.
   */
  function getNormalizedOptions(evolveOptions: EvolveOptions): EvolveOptions {
    return evolveOptions || {};
  }

  /**
   * Resolve normalized scalar settings with defaults.
   *
   * @param evolveOptions - Evolve options object.
   * @returns Normalized scalar settings.
   */
  function resolveEvolutionSettings(
    evolveOptions: EvolveOptions,
  ): EvolutionSettings {
    return {
      targetError: evolveOptions.error ?? DEFAULT_TARGET_ERROR,
      growth: evolveOptions.growth ?? DEFAULT_GROWTH,
      cost: evolveOptions.cost ?? methods.Cost.mse,
      amount: evolveOptions.amount ?? DEFAULT_EVALUATION_AMOUNT,
      log: evolveOptions.log ?? DEFAULT_LOG_INTERVAL,
      schedule: evolveOptions.schedule,
      clear: evolveOptions.clear ?? false,
      threads:
        typeof evolveOptions.threads === 'undefined'
          ? DEFAULT_THREAD_COUNT
          : evolveOptions.threads,
    };
  }

  /**
   * Resolve stopping-condition semantics while preserving legacy behavior.
   *
   * @param evolveOptions - Evolve options object.
   * @param initialTargetError - Target error resolved from options.
   * @returns Final stop conditions.
   */
  function resolveStopConditions(
    evolveOptions: EvolveOptions,
    initialTargetError: number,
  ): EvolutionStopConditions {
    let resolvedTargetError = initialTargetError;

    const iterationsMissing = typeof evolveOptions.iterations === 'undefined';
    const errorMissing = typeof evolveOptions.error === 'undefined';
    if (iterationsMissing && errorMissing) {
      throw new Error(STOPPING_CONDITION_REQUIRED_ERROR_MESSAGE);
    }

    if (errorMissing) {
      resolvedTargetError = DISABLED_TARGET_ERROR;
    } else if (iterationsMissing) {
      evolveOptions.iterations = ZERO_ITERATIONS;
    }

    return { targetError: resolvedTargetError };
  }

  /**
   * Build optional structured evolution config summary.
   *
   * @param settingsToSummarize - Scalar evolution settings.
   * @returns Optional summary config.
   */
  function createEvolutionConfig(
    settingsToSummarize: EvolutionSettings,
  ): EvolutionConfig | undefined {
    if (!settingsToSummarize.schedule) return undefined;

    return {
      targetError: settingsToSummarize.targetError,
      growth: settingsToSummarize.growth,
      cost: settingsToSummarize.cost,
      amount: settingsToSummarize.amount,
      log: settingsToSummarize.log,
      schedule: settingsToSummarize.schedule,
      clear: settingsToSummarize.clear,
      threads: settingsToSummarize.threads,
    };
  }

  /**
   * Build fitness function according to threading configuration.
   *
   * @param dataSet - Supervised dataset.
   * @param resolvedSettings - Scalar evolution settings.
   * @param evolveOptions - Evolve options object.
   * @returns Fitness function and resolved thread count.
   */
  async function prepareFitnessFunction(
    dataSet: TrainingSample[],
    resolvedSettings: EvolutionSettings,
    evolveOptions: EvolveOptions,
  ): Promise<FitnessSetup> {
    if (resolvedSettings.threads === DEFAULT_THREAD_COUNT) {
      return {
        fitnessFunction: buildSingleThreadFitness(
          dataSet,
          resolvedSettings.cost as CostFunction,
          resolvedSettings.amount,
          resolvedSettings.growth,
        ),
        threads: DEFAULT_THREAD_COUNT,
      };
    }

    const multiThreadSetup = await buildMultiThreadFitness(
      dataSet,
      resolvedSettings.cost,
      resolvedSettings.amount,
      resolvedSettings.growth,
      resolvedSettings.threads,
      evolveOptions,
    );

    return {
      fitnessFunction: multiThreadSetup.fitnessFunction,
      threads: multiThreadSetup.threads,
    };
  }

  /**
   * Normalize options used by NEAT constructor.
   *
   * @param network - Network instance being evolved.
   * @param evolveOptions - Evolve options object.
   * @returns Nothing.
   */
  function configureNeatOptions(
    network: Network,
    evolveOptions: EvolveOptions,
  ): void {
    evolveOptions.network = network;
    if (evolveOptions.populationSize != null && evolveOptions.popsize == null)
      evolveOptions.popsize = evolveOptions.populationSize;
    if (typeof evolveOptions.speciation === 'undefined')
      evolveOptions.speciation = false;
  }

  /**
   * Lazy-load and create NEAT instance.
   *
   * @param network - Network instance being evolved.
   * @param fitnessFunction - Prepared fitness evaluator.
   * @param evolveOptions - Evolve options object.
   * @returns Constructed NEAT instance.
   */
  async function createNeatInstance(
    network: Network,
    fitnessFunction: EvolutionFitnessFunction,
    evolveOptions: EvolveOptions,
  ): Promise<NeatRuntime> {
    const { default: Neat } = await import('../../../neat');
    return new Neat(
      network.input,
      network.output,
      fitnessFunction,
      evolveOptions,
    ) as unknown as NeatRuntime;
  }

  /**
   * Emit warning when zero-iteration configuration may produce no best genome.
   *
   * @param neatInstance - Active NEAT instance.
   * @param evolveOptions - Evolve options object.
   * @returns Nothing.
   */
  function warnIfNoBestGenomeMayOccur(
    neatInstance: NeatRuntime,
    evolveOptions: EvolveOptions,
  ): void {
    if (
      typeof evolveOptions.iterations !== 'number' ||
      evolveOptions.iterations !== ZERO_ITERATIONS
    )
      return;

    if (!neatInstance._warnIfNoBestGenome) return;

    try {
      neatInstance._warnIfNoBestGenome();
    } catch {
      // Ignore warning errors
    }
  }

  /**
   * Increase mutation aggressiveness for tiny populations.
   *
   * @param neatInstance - Active NEAT instance.
   * @param evolveOptions - Evolve options object.
   * @returns Nothing.
   */
  function applySmallPopulationHeuristics(
    neatInstance: NeatRuntime,
    evolveOptions: EvolveOptions,
  ): void {
    if (
      !evolveOptions.popsize ||
      evolveOptions.popsize > SMALL_POPULATION_THRESHOLD
    )
      return;
    neatInstance.options.mutationRate =
      neatInstance.options.mutationRate ?? SMALL_POPULATION_MUTATION_RATE;
    neatInstance.options.mutationAmount =
      neatInstance.options.mutationAmount ?? SMALL_POPULATION_MUTATION_AMOUNT;
  }

  /**
   * Run core evolution loop until stop condition is met.
   *
   * @param neatInstance - Active NEAT instance.
   * @param resolvedSettings - Scalar evolution settings.
   * @param targetError - Effective target error (-1 means disabled).
   * @returns Loop result snapshot.
   */
  async function runEvolutionLoop(
    neatInstance: NeatRuntime,
    resolvedSettings: EvolutionSettings,
    targetError: number,
  ): Promise<{ error: number; bestGenome: Network | undefined }> {
    const loopState = createInitialLoopState();
    const iterationsSpecified =
      typeof normalizedOptions.iterations === 'number';

    while (
      shouldContinueEvolution(
        loopState.currentError,
        targetError,
        iterationsSpecified,
        neatInstance.generation,
        normalizedOptions.iterations,
      )
    ) {
      const evolvedGenome = await neatInstance.evolve();
      applyEvolutionStep(loopState, evolvedGenome, resolvedSettings.growth);

      if (shouldAbortForInvalidErrors(loopState)) {
        break;
      }

      runScheduleCallbackSafely(
        resolvedSettings.schedule,
        neatInstance.generation,
        loopState.bestFitness,
        loopState.currentError,
      );
    }

    return {
      error: loopState.currentError,
      bestGenome: loopState.bestGenome,
    };

    /**
     * Creates initial loop state snapshot.
     *
     * @returns Initial loop state.
     */
    function createInitialLoopState(): EvolutionLoopState {
      return {
        currentError: Infinity,
        bestFitness: -Infinity,
        bestGenome: undefined,
        consecutiveInvalidErrorCount: 0,
      };
    }

    /**
     * Applies one evolve() result to loop state.
     *
     * @param state - Mutable loop state.
     * @param evolvedGenome - Genome returned by NEAT evolve step.
     * @param growth - Complexity growth scalar.
     * @returns Nothing.
     */
    function applyEvolutionStep(
      state: EvolutionLoopState,
      evolvedGenome: Network,
      growth: number,
    ): void {
      const evolvedFitness = evolvedGenome.score ?? -Infinity;
      state.currentError = deriveErrorFromFitness(
        evolvedFitness,
        evolvedGenome,
        growth,
      );

      const bestSnapshot = updateBestGenomeIfImproved(
        state.bestFitness,
        state.bestGenome,
        evolvedFitness,
        evolvedGenome,
      );
      state.bestFitness = bestSnapshot.bestFitness;
      state.bestGenome = bestSnapshot.bestGenome;
      state.consecutiveInvalidErrorCount = updateInvalidErrorCounter(
        state.consecutiveInvalidErrorCount,
        state.currentError,
      );
    }

    /**
     * Determines whether loop must abort due to invalid-error streak.
     *
     * @param state - Mutable loop state.
     * @returns True when invalid-error threshold is reached.
     */
    function shouldAbortForInvalidErrors(state: EvolutionLoopState): boolean {
      return (
        state.consecutiveInvalidErrorCount >= MAX_CONSECUTIVE_INVALID_ERRORS
      );
    }
  }

  /**
   * Determine whether evolution loop should continue.
   *
   * @param currentError - Current derived error value.
   * @param targetError - Effective target error (-1 means disabled).
   * @param iterationsSpecified - Whether iterations limit is active.
   * @param currentGeneration - Current NEAT generation index.
   * @param maxIterations - Maximum iteration limit.
   * @returns True when loop should continue.
   */
  function shouldContinueEvolution(
    currentError: number,
    targetError: number,
    iterationsSpecified: boolean,
    currentGeneration: number,
    maxIterations: number | undefined,
  ): boolean {
    const targetConditionMet =
      targetError !== DISABLED_TARGET_ERROR && currentError <= targetError;
    if (targetConditionMet) return false;
    if (!iterationsSpecified) return true;
    return currentGeneration < (maxIterations ?? ZERO_ITERATIONS);
  }

  /**
   * Derive error from fitness by inverting score composition.
   *
   * @param fitness - Fitness value from fittest genome.
   * @param genome - Fittest genome.
   * @param growth - Complexity growth scalar.
   * @returns Derived error value.
   */
  function deriveErrorFromFitness(
    fitness: number,
    genome: Network,
    growth: number,
  ): number {
    return -(fitness - computeComplexityPenalty(genome, growth)) || Infinity;
  }

  /**
   * Update best fitness/genome snapshot when improved.
   *
   * @param currentBestFitness - Current best fitness.
   * @param currentBestGenome - Current best genome.
   * @param candidateFitness - Candidate fitness.
   * @param candidateGenome - Candidate genome.
   * @returns Updated best snapshot.
   */
  function updateBestGenomeIfImproved(
    currentBestFitness: number,
    currentBestGenome: Network | undefined,
    candidateFitness: number,
    candidateGenome: Network,
  ): { bestFitness: number; bestGenome: Network | undefined } {
    if (candidateFitness <= currentBestFitness) {
      return {
        bestFitness: currentBestFitness,
        bestGenome: currentBestGenome,
      };
    }

    return { bestFitness: candidateFitness, bestGenome: candidateGenome };
  }

  /**
   * Update invalid-error counter and decide whether to abort.
   *
   * @param currentCount - Current consecutive invalid-error count.
   * @param currentError - Current derived error value.
   * @returns Updated guard state.
   */
  function updateInvalidErrorCounter(
    currentCount: number,
    currentError: number,
  ): number {
    if (Number.isFinite(currentError) && !Number.isNaN(currentError)) {
      return 0;
    }

    return currentCount + 1;
  }

  /**
   * Run schedule callback if schedule trigger is reached.
   *
   * @param scheduleConfig - Optional schedule configuration.
   * @param generation - Current generation.
   * @param bestFitness - Current best fitness.
   * @param error - Current error.
   * @returns Nothing.
   */
  function runScheduleCallbackSafely(
    scheduleConfig: EvolveOptions['schedule'],
    generation: number,
    bestFitness: number,
    error: number,
  ): void {
    if (!scheduleConfig) return;
    if (generation % scheduleConfig.iterations !== 0) return;

    try {
      scheduleConfig.function({
        fitness: bestFitness,
        error,
        iteration: generation,
      });
    } catch {
      // Ignore schedule callback errors
    }
  }

  /**
   * Adopt best genome structure or emit warning when unavailable.
   *
   * @param network - Network instance being evolved.
   * @param neatInstance - Active NEAT instance.
   * @param bestGenome - Best genome snapshot.
   * @param clearState - Whether to clear network after adoption.
   * @returns Nothing.
   */
  function adoptBestGenomeOrWarn(
    network: Network,
    neatInstance: NeatRuntime,
    bestGenome: Network | undefined,
    clearState: boolean,
  ): void {
    if (bestGenome) {
      network.nodes = bestGenome.nodes;
      network.connections = bestGenome.connections;
      network.selfconns = bestGenome.selfconns;
      network.gates = bestGenome.gates;
      if (clearState) network.clear();
      return;
    }

    if (!neatInstance._warnIfNoBestGenome) return;
    try {
      neatInstance._warnIfNoBestGenome();
    } catch {
      // Ignore warning errors
    }
  }

  /**
   * Terminate worker resources registered in options.
   *
   * @param evolveOptions - Evolve options object.
   * @returns Nothing.
   */
  function terminateWorkersSafely(evolveOptions: EvolveOptions): void {
    try {
      evolveOptions._workerTerminators?.();
    } catch {
      // Ignore termination errors
    }
  }

  /**
   * Build final evolve return payload.
   *
   * @param error - Final loop error.
   * @param iterations - Final generation count.
   * @param loopStartTime - Loop start timestamp.
   * @returns Evolution summary object.
   */
  function buildEvolutionSummary(
    error: number,
    iterations: number,
    loopStartTime: number,
  ): { error: number; iterations: number; time: number } {
    return { error, iterations, time: Date.now() - loopStartTime };
  }
}
