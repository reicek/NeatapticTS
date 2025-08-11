import Network from './network';
import * as methods from '../methods/methods';
import { config } from '../config';
import Multi from '../multithreading/multi';

/**
 * File: network.evolve.ts
 * ---------------------------------------------
 * Contains the extracted evolutionary optimization loop for a single Network instance.
 * This keeps the monolithic Network class lean while providing a focused, well‑documented
 * implementation of NEAT-style topology & weight evolution using the higher-level Neat class.
 *
 * Design principles:
 *  - Pure adaptor: translates high-level evolution options into a configured Neat instance.
 *  - Zero hidden side effects besides mutating the calling network with the fittest genome at the end.
 *  - Defensive validation: dataset dimensions and stopping criteria are verified early.
 *  - Minimal branching inside the main loop; expensive concerns (evaluation / speciation) delegated to Neat.
 *  - Resilient to NaNs / Infinity (will tolerate a small number before aborting gracefully).
 */

/**
 * Evolutionary optimization loop (NEAT-style) extracted from `network.ts` for clarity and smaller bundle size.
 *
 * Core responsibilities:
 *  - Validates dataset shape (throws if invalid or dimension mismatch).
 *  - Builds a fitness function (single-thread direct evaluation; multi-thread via workers if threads>1).
 *  - Applies a structural complexity penalty (growth) discouraging uncontrolled bloat.
 *  - Supports stopping by target error and/or max generations (`iterations`). At least one must be provided.
 *  - Tracks the fittest genome; on completion replaces this network's topology with the best genome (optionally clearing state).
 *  - Emits a warning if no valid best genome is discovered (always, including when iterations=0) via `_warnIfNoBestGenome`.
 *  - Optional periodic logging (`log`) and user scheduling callbacks (`schedule`).
 *  - Heuristic adjustments for very small populations (<=10) to keep mutation pressure high (unless user specified values).
 *  - Guards against repeated infinite / NaN evaluation results and aborts after a small threshold.
 *
 * Option highlights (partial):
 *  - iterations {number}  Maximum generations. If 0, loop is skipped but warning still may emit (dry-run).
 *  - error {number}       Target error (converted to fitness by negation & penalty).
 *  - cost {Function}      Cost function (default: methods.Cost.mse).
 *  - amount {number}      Re-evaluation count per genome averaged into fitness.
 *  - growth {number}      Structural penalty coefficient (default 0.0001).
 *  - popsize {number}     Population size.
 *  - speciation {boolean} Enable species partitioning (default false here unless provided).
 *  - threads {number}     Worker thread count (>1 triggers parallel fitness path; tests generally use 1).
 *  - log {number}         Log frequency in generations (0 disables).
 *  - schedule {object}    { iterations:number, function:({fitness,error,iteration})=>void }
 *  - clear {boolean}      If true, calls clear() after adopting best genome for stateless inference.
 *
 * Edge cases & behaviors:
 *  - Must supply either `iterations` or `error` (throws otherwise).
 *  - Only error supplied -> open-ended generations until target reached.
 *  - Only iterations supplied -> target error disabled (internally -1 sentinel).
 *  - iterations === 0 -> performs validation & immediate warning path without evolution steps.
 *  - Warning message text aligned with README/tests ("Evolution completed without finding a valid best genome").
 *
 * Returns:
 *  { error, iterations, time } where
 *    - error: Best error (Infinity if none finite)
 *    - iterations: Actual generations executed (neat.generation)
 *    - time: Wall clock duration in ms
 *
 * NOTE: Derived error re-adds the structural penalty removed during fitness so reporting remains intuitive.
 */
export async function evolveNetwork(this: Network, set: { input: number[]; output: number[] }[], options: any): Promise<{ error: number; iterations: number; time: number }> {
  // ---- Validation & normalization ----
  if (!set || set.length === 0 || set[0].input.length !== this.input || set[0].output.length !== this.output) {
    throw new Error('Dataset is invalid or dimensions do not match network input/output size!');
  }
  options = options || {};
  /** Target error (negated into fitness); -1 signals disabled when only iterations specified */
  let targetError: number = options.error ?? 0.05;
  /** Structural complexity penalty coefficient discouraging uncontrolled growth */
  const growth: number = options.growth ?? 0.0001;
  /** Cost function used to compute raw error per evaluation */
  const cost = options.cost || methods.Cost.mse;
  /** Number of repeated evaluations (averaged) per genome for noise reduction */
  const amount: number = options.amount || 1;
  /** Logging cadence (generations); 0 disables */
  const log: number = options.log || 0;
  /** Optional schedule callback container */
  const schedule = options.schedule;
  /** Whether to clear stateful activations on adoption of the best genome */
  const clear: boolean = options.clear || false;
  /** Requested worker threads ( >1 => population evaluation mode ) */
  let threads = options.threads; if (typeof threads === 'undefined') threads = 1;
  /** Wall clock start time (ms) for elapsed time reporting */
  const start = Date.now();
  // Enforce at least one stopping condition
  if (typeof options.iterations === 'undefined' && typeof options.error === 'undefined') {
    throw new Error('At least one stopping condition (`iterations` or `error`) must be specified for evolution.');
  } else if (typeof options.error === 'undefined') {
    targetError = -1; // disabled: rely solely on iterations
  } else if (typeof options.iterations === 'undefined') {
    options.iterations = 0; // open‑ended until target error reached
  }

  // ---- Fitness function construction ----
  /**
   * Fitness evaluation implementation.
   *  - Single-thread: (genome)=>number (immediate value)
   *  - Multi-thread: (population)=>Promise<void> (Neat sets genome.score)
   */
  let fitnessFunction: any;
  if (threads === 1) {
    // Single-thread: direct synchronous evaluation per genome
    fitnessFunction = (genome: Network) => {
      let score = 0; // accumulate negative errors (so higher is better)
      for (let i = 0; i < amount; i++) {
        try {
            // test() returns { error }, so subtract error to convert to fitness
            score -= genome.test(set, cost).error;
        } catch (e: any) {
          if (config.warnings) console.warn(`Genome evaluation failed: ${(e && e.message) || e}. Penalizing with -Infinity fitness.`);
          return -Infinity; // immediate bailout for this genome
        }
      }
      // Complexity penalty: nodes (excluding IO) + connections + gates
      score -= (genome.nodes.length - genome.input - genome.output + genome.connections.length + genome.gates.length) * growth;
      score = isNaN(score) ? -Infinity : score;
      return score / amount; // average across evaluation repeats
    };
  } else {
    // Multi-thread path (workers). NOTE: workers array wiring omitted (in original design external infra attaches workers)
  /** Serialized dataset passed to workers (placeholder for external infra) */
  const converted = Multi.serializeDataSet(set); // eslint-disable-line @typescript-eslint/no-unused-vars
  /** Worker pool placeholder (external system would populate) */
  const workers: any[] = []; // eslint-disable-line @typescript-eslint/no-unused-vars
    fitnessFunction = (population: Network[]) => new Promise<void>((resolve) => {
      const queue = population.slice(); // shallow copy queue of genomes
      let done = 0; // number of workers finished (idle at time of exhaustion)
      const startWorker = (worker: any) => {
        if (!queue.length) { if (++done === threads) resolve(); return; }
        const genome = queue.shift();
        worker.evaluate(genome).then((result: number) => {
          if (typeof genome !== 'undefined' && typeof result === 'number') {
            // Convert raw error result into penalized fitness
            genome.score = -result - (genome.nodes.length - genome.input - genome.output + genome.connections.length + genome.gates.length) * growth;
            genome.score = isNaN(result) ? -Infinity : genome.score;
          }
          startWorker(worker); // recurse until queue empty
        });
      };
      workers.forEach(startWorker);
    });
    options.fitnessPopulation = true; // instruct Neat to call fitness(population)
  }

  // ---- Neat instance bootstrap ----
  options.network = this; // allow Neat to reference owning network if needed
  if (options.populationSize != null && options.popsize == null) options.popsize = options.populationSize; // alias support
  if (typeof options.speciation === 'undefined') options.speciation = false; // default speciation off here (caller can override)
  const { default: Neat } = await import('../neat');
  const neat = new Neat(this.input, this.output, fitnessFunction, options);
  if (typeof options.iterations === 'number' && options.iterations === 0) {
    // Immediate warning path: no evolutionary steps will execute
    if ((neat as any)._warnIfNoBestGenome) { try { (neat as any)._warnIfNoBestGenome(); } catch {} }
  }
  // Small populations: assert stronger mutation defaults unless user already provided
  if (options.popsize && options.popsize <= 10) {
    neat.options.mutationRate = neat.options.mutationRate ?? 0.5;
    neat.options.mutationAmount = neat.options.mutationAmount ?? 1;
  }

  // ---- Main loop ----
  /** Current best error this generation (recomputed from best fitness) */
  let error = Infinity;
  /** Running maximum fitness across generations */
  let bestFitness = -Infinity;
  /** Pointer to structurally best genome found so far */
  let bestGenome: Network | undefined;
  /** Guard counter tracking consecutive invalid (NaN/Infinity) errors */
  let infiniteErrorCount = 0;
  /** Abort threshold for consecutive invalid errors */
  const MAX_INF = 5;
  /** Flag denoting whether iterations stopping criterion is active */
  const iterationsSpecified = typeof options.iterations === 'number';
  while ((targetError === -1 || error > targetError) && (!iterationsSpecified || neat.generation < options.iterations)) {
    const fittest = await neat.evolve(); // one evolutionary generation
    const fitness = fittest.score ?? -Infinity;
    // Reconstruct raw error (undo penalty) so reported error aligns with cost metric users expect
    error = -(fitness - (fittest.nodes.length - fittest.input - fittest.output + fittest.connections.length + fittest.gates.length) * growth) || Infinity;
    if (fitness > bestFitness) { bestFitness = fitness; bestGenome = fittest; }
    if (!isFinite(error) || isNaN(error)) {
      if (++infiniteErrorCount >= MAX_INF) break; // abort after repeated invalid values
    } else {
      infiniteErrorCount = 0;
    }
    if (log > 0 && neat.generation % log === 0) {
      console.log(`Generation: ${neat.generation}, Best Fitness: ${bestFitness.toFixed(6)}, Best Error: ${error.toFixed(6)}`);
    }
    if (schedule && neat.generation % schedule.iterations === 0) {
      schedule.function({ fitness: bestFitness, error, iteration: neat.generation });
    }
  }

  // ---- Adoption of best genome ----
  if (typeof bestGenome !== 'undefined') {
    // Replace structural arrays (nodes / connections / etc.) so current network becomes the champion
    this.nodes = bestGenome.nodes;
    this.connections = bestGenome.connections;
    this.selfconns = bestGenome.selfconns;
    this.gates = bestGenome.gates;
    if (clear) this.clear(); // optional state reset
  } else if ((neat as any)._warnIfNoBestGenome) {
    try { (neat as any)._warnIfNoBestGenome(); } catch {}
  }

  // ---- Return summary ----
  return { error, iterations: neat.generation, time: Date.now() - start };
}
