import Network from '../network';
import * as methods from '../../methods/methods';
import { config } from '../../config';
import Multi from '../../multithreading/multi';
import type {
  TestWorkerInstance,
  TestWorkerConstructor,
} from '../../multithreading/types';

/**
 * A single supervised training example used to evaluate fitness.
 */
interface TrainingSample {
  input: number[];
  output: number[];
}

/**
 * Cost function type: takes target and output arrays, returns error scalar.
 */
type CostFunction = (target: number[], output: number[]) => number;

/**
 * Cost function OR a serializable reference with just the name (for worker threads).
 */
type CostFunctionOrRef = CostFunction | { name: string };

/**
 * Internal evolution configuration summary (for potential logging / debugging)
 * capturing normalized option values used by the local evolutionary loop.
 */
interface EvolutionConfig {
  targetError: number;
  growth: number;
  cost: CostFunctionOrRef;
  amount: number;
  log: number;
  schedule: {
    iterations: number;
    function: (stats: {
      fitness: number;
      error: number;
      iteration: number;
    }) => void;
  };
  clear: boolean;
  threads: number;
}

/**
 * Evolution options for the evolveNetwork method.
 * Includes core evolution parameters plus NEAT-specific options passed through.
 */
interface EvolveOptions extends Record<string, unknown> {
  error?: number;
  iterations?: number;
  growth?: number;
  cost?: CostFunctionOrRef;
  amount?: number;
  log?: number;
  schedule?: {
    iterations: number;
    function: (stats: {
      fitness: number;
      error: number;
      iteration: number;
    }) => void;
  };
  clear?: boolean;
  threads?: number;
  fitnessPopulation?: boolean;
  network?: Network;
  populationSize?: number;
  popsize?: number;
  speciation?: boolean;
  _workerTerminators?: () => void;
}

/** Fitness function evaluating one genome at a time. */
type SingleGenomeFitnessFunction = (genome: Network) => number;

/** Fitness function evaluating a full population asynchronously. */
type PopulationFitnessFunction = (population: Network[]) => Promise<void>;

/** Unified fitness function shape accepted by NEAT constructor. */
type EvolutionFitnessFunction =
  | SingleGenomeFitnessFunction
  | PopulationFitnessFunction;

/** Result of preparing a fitness function strategy. */
interface FitnessSetup {
  fitnessFunction: EvolutionFitnessFunction;
  threads: number;
}

/** Minimal runtime contract used from the NEAT instance in this module. */
interface NeatRuntime {
  generation: number;
  options: {
    mutationRate?: number;
    mutationAmount?: number;
  };
  evolve: () => Promise<Network>;
  _warnIfNoBestGenome?: () => void;
}

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
      if (!workers.length) {
        resolve();
        return;
      }

      const genomeQueue = population.slice();
      let activeWorkerCount = workers.length;

      function runNextGenome(worker: TestWorkerInstance): void {
        if (!genomeQueue.length) {
          activeWorkerCount -= 1;
          if (activeWorkerCount === 0) resolve();
          return;
        }

        const queuedGenome = genomeQueue.shift();
        if (!queuedGenome) {
          activeWorkerCount -= 1;
          if (activeWorkerCount === 0) resolve();
          return;
        }

        evaluateGenomeWithWorker(worker, queuedGenome, growth)
          .then(() => runNextGenome(worker))
          .catch(() => runNextGenome(worker));
      }

      workers.forEach((worker) => runNextGenome(worker));
    });
  };
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
      genome as unknown as import('../../multithreading/types').SerializableNetwork,
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
      throw new Error(
        'Dataset is invalid or dimensions do not match network input/output size!',
      );
    }

    const firstSample = dataSet[0];
    const inputMatches = firstSample.input.length === network.input;
    const outputMatches = firstSample.output.length === network.output;
    if (!inputMatches || !outputMatches) {
      throw new Error(
        'Dataset is invalid or dimensions do not match network input/output size!',
      );
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
  function resolveEvolutionSettings(evolveOptions: EvolveOptions): {
    targetError: number;
    growth: number;
    cost: CostFunctionOrRef;
    amount: number;
    log: number;
    schedule: EvolveOptions['schedule'];
    clear: boolean;
    threads: number;
  } {
    return {
      targetError: evolveOptions.error ?? 0.05,
      growth: evolveOptions.growth ?? 0.0001,
      cost: evolveOptions.cost || methods.Cost.mse,
      amount: evolveOptions.amount || 1,
      log: evolveOptions.log || 0,
      schedule: evolveOptions.schedule,
      clear: evolveOptions.clear || false,
      threads:
        typeof evolveOptions.threads === 'undefined'
          ? 1
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
  ): { targetError: number } {
    let resolvedTargetError = initialTargetError;

    const iterationsMissing = typeof evolveOptions.iterations === 'undefined';
    const errorMissing = typeof evolveOptions.error === 'undefined';
    if (iterationsMissing && errorMissing) {
      throw new Error(
        'At least one stopping condition (`iterations` or `error`) must be specified for evolution.',
      );
    }

    if (errorMissing) {
      resolvedTargetError = -1;
    } else if (iterationsMissing) {
      evolveOptions.iterations = 0;
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
    settingsToSummarize: ReturnType<typeof resolveEvolutionSettings>,
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
    resolvedSettings: ReturnType<typeof resolveEvolutionSettings>,
    evolveOptions: EvolveOptions,
  ): Promise<FitnessSetup> {
    if (resolvedSettings.threads === 1) {
      return {
        fitnessFunction: buildSingleThreadFitness(
          dataSet,
          resolvedSettings.cost as CostFunction,
          resolvedSettings.amount,
          resolvedSettings.growth,
        ),
        threads: 1,
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
    const { default: Neat } = await import('../../neat');
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
      evolveOptions.iterations !== 0
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
    if (!evolveOptions.popsize || evolveOptions.popsize > 10) return;
    neatInstance.options.mutationRate =
      neatInstance.options.mutationRate ?? 0.5;
    neatInstance.options.mutationAmount =
      neatInstance.options.mutationAmount ?? 1;
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
    resolvedSettings: ReturnType<typeof resolveEvolutionSettings>,
    targetError: number,
  ): Promise<{ error: number; bestGenome: Network | undefined }> {
    const maxInfiniteErrors = 5;
    let currentError = Infinity;
    let bestFitness = -Infinity;
    let bestGenome: Network | undefined;
    let consecutiveInfiniteErrorCount = 0;
    const iterationsSpecified =
      typeof normalizedOptions.iterations === 'number';

    while (
      shouldContinueEvolution(
        currentError,
        targetError,
        iterationsSpecified,
        neatInstance.generation,
        normalizedOptions.iterations,
      )
    ) {
      const fittestGenome = await neatInstance.evolve();
      const fitness = fittestGenome.score ?? -Infinity;
      currentError = deriveErrorFromFitness(
        fitness,
        fittestGenome,
        resolvedSettings.growth,
      );

      const bestSnapshot = updateBestGenomeIfImproved(
        bestFitness,
        bestGenome,
        fitness,
        fittestGenome,
      );
      bestFitness = bestSnapshot.bestFitness;
      bestGenome = bestSnapshot.bestGenome;

      const errorGuard = updateInfiniteErrorCounter(
        consecutiveInfiniteErrorCount,
        currentError,
      );
      consecutiveInfiniteErrorCount = errorGuard.consecutiveInfiniteErrorCount;
      if (
        errorGuard.shouldAbort ||
        consecutiveInfiniteErrorCount >= maxInfiniteErrors
      )
        break;

      runScheduleCallbackSafely(
        resolvedSettings.schedule,
        neatInstance.generation,
        bestFitness,
        currentError,
      );
    }

    return { error: currentError, bestGenome };
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
      targetError !== -1 && currentError <= targetError;
    if (targetConditionMet) return false;
    if (!iterationsSpecified) return true;
    return currentGeneration < (maxIterations ?? 0);
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
  function updateInfiniteErrorCounter(
    currentCount: number,
    currentError: number,
  ): { consecutiveInfiniteErrorCount: number; shouldAbort: boolean } {
    if (Number.isFinite(currentError) && !Number.isNaN(currentError)) {
      return { consecutiveInfiniteErrorCount: 0, shouldAbort: false };
    }

    return {
      consecutiveInfiniteErrorCount: currentCount + 1,
      shouldAbort: false,
    };
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
