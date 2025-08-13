import Network from '../network';
import * as methods from '../../methods/methods';
import { config } from '../../config';
import Multi from '../../multithreading/multi';

/**
 * A single supervised training example used to evaluate fitness.
 */
interface TrainingSample {
  input: number[];
  output: number[];
}

/**
 * Internal evolution configuration summary (for potential logging / debugging)
 * capturing normalized option values used by the local evolutionary loop.
 */
interface EvolutionConfig {
  targetError: number;
  growth: number;
  cost: any;
  amount: number;
  log: number;
  schedule: any;
  clear: boolean;
  threads: number;
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
  // Extract structural counts once.
  const n = genome.nodes.length;
  const c = genome.connections.length;
  const g = genome.gates.length;
  // Fast path: counts unchanged -> reuse cached base complexity value.
  const cached = _complexityCache.get(genome);
  if (cached && cached.nodes === n && cached.conns === c && cached.gates === g)
    return cached.value * growth;
  // Base complexity ignoring growth factor.
  const base = n - genome.input - genome.output + c + g;
  _complexityCache.set(genome, { nodes: n, conns: c, gates: g, value: base });
  return base * growth;
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
  cost: any,
  amount: number,
  growth: number
) {
  return (genome: Network) => {
    let score = 0; // Accumulate negative errors.
    for (let i = 0; i < amount; i++) {
      try {
        score -= genome.test(set, cost).error; // negative adds fitness.
      } catch (e: any) {
        if (config.warnings)
          console.warn(
            `Genome evaluation failed: ${
              (e && e.message) || e
            }. Penalizing with -Infinity fitness.`
          );
        return -Infinity;
      }
    }
    // Apply structural parsimony pressure.
    score -= computeComplexityPenalty(genome, growth);
    // Guard against NaN pollution.
    score = isNaN(score) ? -Infinity : score;
    // Average over repeats.
    return score / amount;
  };
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
  cost: any,
  amount: number,
  growth: number,
  threads: number,
  options: any
) {
  // Serialize dataset once for worker initialization (avoids deep cloning per evaluation call).
  const serializedSet = Multi.serializeDataSet(set);
  /** Collection of worker instances. */
  const workers: any[] = [];
  let WorkerCtor: any = null; // Will hold dynamic Worker class.
  try {
    const isNode =
      typeof process !== 'undefined' && !!(process.versions as any)?.node;
    if (isNode && Multi.workers?.getNodeTestWorker)
      WorkerCtor = await Multi.workers.getNodeTestWorker();
    else if (!isNode && Multi.workers?.getBrowserTestWorker)
      WorkerCtor = await Multi.workers.getBrowserTestWorker();
  } catch (e) {
    if (config.warnings)
      console.warn(
        'Failed to load worker class; falling back to single-thread path:',
        (e as any)?.message || e
      );
  }
  // Fallback path if no worker support.
  if (!WorkerCtor)
    return {
      fitnessFunction: buildSingleThreadFitness(set, cost, amount, growth),
      threads: 1,
    };
  // Spin up requested workers (best-effort; partial successes still useful).
  for (let i = 0; i < threads; i++) {
    try {
      workers.push(
        new WorkerCtor(serializedSet, {
          name: cost.name || cost.toString?.() || 'cost',
        })
      );
    } catch (e) {
      if (config.warnings) console.warn('Worker spawn failed', e);
    }
  }
  // Population-level fitness function: resolves when all genomes processed.
  const fitnessFunction = (population: Network[]) =>
    new Promise<void>((resolve) => {
      if (!workers.length) {
        resolve();
        return;
      }
      const queue = population.slice(); // Shallow copy so we can mutate.
      let active = workers.length; // Number of workers still draining tasks.
      const startNext = (worker: any) => {
        if (!queue.length) {
          if (--active === 0) resolve();
          return;
        }
        const genome = queue.shift();
        worker
          .evaluate(genome)
          .then((result: number) => {
            if (typeof genome !== 'undefined' && typeof result === 'number') {
              genome.score = -result - computeComplexityPenalty(genome, growth);
              genome.score = isNaN(result) ? -Infinity : genome.score;
            }
            startNext(worker); // Tail recursion style loop.
          })
          .catch(() => startNext(worker)); // On error: skip but keep draining.
      };
      workers.forEach((w) => startNext(w));
    });
  options.fitnessPopulation = true; // Signal population-level semantics.
  // Provide cleanup hook (used after evolution loop) to terminate workers.
  (options as any)._workerTerminators = () => {
    workers.forEach((w) => {
      try {
        w.terminate && w.terminate();
      } catch {}
    });
  };
  return { fitnessFunction, threads };
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
  options: any
): Promise<{ error: number; iterations: number; time: number }> {
  // 1. Dataset validation (shape + existence).
  if (
    !set ||
    set.length === 0 ||
    set[0].input.length !== this.input ||
    set[0].output.length !== this.output
  ) {
    throw new Error(
      'Dataset is invalid or dimensions do not match network input/output size!'
    );
  }
  // Defensive defaulting.
  options = options || {};
  let targetError: number = options.error ?? 0.05; // Default target error if provided unspecified.
  const growth: number = options.growth ?? 0.0001; // Complexity penalty scaling.
  const cost = options.cost || methods.Cost.mse; // Default cost function.
  const amount: number = options.amount || 1; // Repetition count for averaging.
  const log: number = options.log || 0; // Logging interval (0 disables).
  const schedule = options.schedule; // Optional user schedule callback spec.
  const clear: boolean = options.clear || false; // Whether to clear state after structural adoption.
  let threads: number =
    typeof options.threads === 'undefined' ? 1 : options.threads; // Worker count.
  const start = Date.now(); // Benchmark start time.
  const evoConfig: EvolutionConfig = {
    targetError,
    growth,
    cost,
    amount,
    log,
    schedule,
    clear,
    threads,
  }; // (Currently unused externally; placeholder for future structured logging.)

  // 2. Stopping condition checks / normalization.
  if (
    typeof options.iterations === 'undefined' &&
    typeof options.error === 'undefined'
  ) {
    throw new Error(
      'At least one stopping condition (`iterations` or `error`) must be specified for evolution.'
    );
  } else if (typeof options.error === 'undefined') targetError = -1;
  // Only iterations constrain.
  else if (typeof options.iterations === 'undefined') options.iterations = 0; // Only error constrains (0 sentinel lets loop run until satisfied).

  // 3. Build fitness function (single or multi-thread variant).
  let fitnessFunction: any;
  if (threads === 1)
    fitnessFunction = buildSingleThreadFitness(set, cost, amount, growth);
  else {
    const multi = await buildMultiThreadFitness(
      set,
      cost,
      amount,
      growth,
      threads,
      options
    );
    fitnessFunction = multi.fitnessFunction;
    threads = multi.threads;
  }

  // Provide network reference for NEAT initialization / reproduction methods.
  options.network = this;
  // Alias populationSize -> popsize for backward compat.
  if (options.populationSize != null && options.popsize == null)
    options.popsize = options.populationSize;
  // Speciation default off unless explicitly enabled (simpler baseline behavior).
  if (typeof options.speciation === 'undefined') options.speciation = false;

  // 4. Lazy import NEAT (avoid heavier modules if evolve isn't used).
  const { default: Neat } = await import('../../neat');
  const neat = new Neat(this.input, this.output, fitnessFunction, options);

  // Warn if immediate termination conditions could yield empty best genome tracking.
  if (typeof options.iterations === 'number' && options.iterations === 0) {
    if ((neat as any)._warnIfNoBestGenome) {
      try {
        (neat as any)._warnIfNoBestGenome();
      } catch {}
    }
  }
  // Micro-population heuristics: increase mutation intensity to promote exploration.
  if (options.popsize && options.popsize <= 10) {
    neat.options.mutationRate = neat.options.mutationRate ?? 0.5;
    neat.options.mutationAmount = neat.options.mutationAmount ?? 1;
  }

  // 5. Evolution loop state variables.
  let error = Infinity; // Best error observed this generation (derived from fitness).
  let bestFitness = -Infinity; // Track highest fitness seen.
  let bestGenome: Network | undefined; // Best genome snapshot.
  let infiniteErrorCount = 0; // Consecutive invalid error tallies.
  const MAX_INF = 5; // Abort threshold to prevent endless invalid loops.
  const iterationsSpecified = typeof options.iterations === 'number';

  // 5a. Main generation loop (terminates on error target or iteration cap).
  while (
    (targetError === -1 || error > targetError) &&
    (!iterationsSpecified || neat.generation < options.iterations)
  ) {
    // Perform one generation: breed + evaluate population, returning fittest genome.
    const fittest = await neat.evolve();
    const fitness = fittest.score ?? -Infinity;
    // Derive error metric from fitness (undo sign & complexity adjustment) with fallback Infinity.
    error = -(fitness - computeComplexityPenalty(fittest, growth)) || Infinity;
    // Update elite if improved.
    if (fitness > bestFitness) {
      bestFitness = fitness;
      bestGenome = fittest;
    }
    // Detect runaway invalid values.
    if (!isFinite(error) || isNaN(error)) {
      if (++infiniteErrorCount >= MAX_INF) break;
    } else infiniteErrorCount = 0;
    // User schedule callback hook.
    if (schedule && neat.generation % schedule.iterations === 0) {
      try {
        schedule.function({
          fitness: bestFitness,
          error,
          iteration: neat.generation,
        });
      } catch {}
    }
  }

  // 6. Adopt best genome's structure into this network instance (in-place upgrade) if available.
  if (typeof bestGenome !== 'undefined') {
    this.nodes = bestGenome.nodes;
    this.connections = bestGenome.connections;
    this.selfconns = bestGenome.selfconns;
    this.gates = bestGenome.gates;
    if (clear) this.clear();
  } else if ((neat as any)._warnIfNoBestGenome) {
    try {
      (neat as any)._warnIfNoBestGenome();
    } catch {}
  }

  // 7. Cleanup worker resources if any.
  try {
    (options as any)._workerTerminators &&
      (options as any)._workerTerminators();
  } catch {}

  return { error, iterations: neat.generation, time: Date.now() - start };
}
