# architecture/network/evolve

## architecture/network/evolve/network.evolve.utils.ts

### buildMultiThreadFitness

`(set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").CostFunctionOrRef, amount: number, growth: number, threads: number, options: Record<string, unknown>) => Promise<import("C:/NeatapticTS/src/architecture/network/network.types").FitnessSetup>`

Build a multi-threaded (worker-based) population fitness evaluator if worker infrastructure is available.

Strategy:
 - Attempt to dynamically obtain a Worker constructor (node or browser variant).
 - If not possible, gracefully fall back to single-thread evaluation.
 - Spawn N workers (threads) each capable of evaluating genomes by calling worker.evaluate(genome).
 - Provide a fitness function that takes the whole population and returns a Promise that resolves
   when all queued genomes have been processed. Each genome's score is written in-place.

Implementation details:
 - Queue: simple FIFO (array shift) suffices because ordering is not critical.
 - Robustness: Each worker evaluation is wrapped with error handling to prevent a single failure
   from stalling the batch; failed evaluations simply proceed to next genome.
 - Complexity penalty applied after raw result retrieval: genome.score = -result - penalty.

Returned metadata sets options.fitnessPopulation=true so downstream NEAT logic treats the fitness
function as operating over the entire population at once (rather than per-genome).

Parameters:
- `set` - - Dataset.
- `cost` - - Cost function.
- `amount` - - Repetition count (unused directly here; assumed handled inside worker.evaluate result metric if needed).
- `growth` - - Complexity penalty scalar.
- `threads` - - Desired worker count.
- `options` - - Evolution options object (mutated to add cleanup hooks & flags).

Returns: Object with fitnessFunction (population evaluator) and resolved thread count.

### buildPopulationWorkerFitnessFunction

`(workers: import("C:/NeatapticTS/src/multithreading/types").TestWorkerInstance[], growth: number) => import("C:/NeatapticTS/src/architecture/network/network.types").PopulationFitnessFunction`

Build population-level fitness function powered by worker queue.

Parameters:
- `workers` - - Spawned worker instances.
- `growth` - - Complexity penalty scalar.

Returns: Population-level fitness function.

### buildSingleThreadFitness

`(set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveCostFunction, amount: number, growth: number) => import("C:/NeatapticTS/src/architecture/network/network.types").SingleGenomeFitnessFunction`

Build a single-threaded fitness evaluation function (classic NEAT style) evaluating a genome
over the provided dataset and returning a scalar score where higher is better.

Fitness Definition:
  fitness = -averageError - complexityPenalty
We accumulate negative error (so lower error => higher fitness) over `amount` independent
evaluations (amount>1 can smooth stochastic evaluation noise) then subtract complexity penalty.

Error handling: If evaluation throws (numerical instability, internal error) we return -Infinity
so such genomes are strongly disfavored.

Parameters:
- `set` - - Dataset of training samples.
- `cost` - - Cost function reference (should expose error computation in genome.test).
- `amount` - - Number of repeated evaluations to average.
- `growth` - - Complexity penalty scalar.

Returns: Function mapping a Network genome to a numeric fitness.

### cacheComplexityBase

`(genome: import("C:/NeatapticTS/src/architecture/network").default, structureCounts: { nodeCount: number; connectionCount: number; gateCount: number; }, complexityBase: number) => void`

Store complexity base cache entry for future reuse.

Parameters:
- `genome` - - Candidate network used as cache key.
- `structureCounts` - - Current structural counts.
- `complexityBase` - - Computed base complexity value.

Returns: Nothing.

### computeComplexityBase

`(genome: import("C:/NeatapticTS/src/architecture/network").default, structureCounts: { nodeCount: number; connectionCount: number; gateCount: number; }) => number`

Compute non-scaled complexity base from structural counts.

Parameters:
- `genome` - - Candidate network whose complexity to compute.
- `structureCounts` - - Current structural counts.

Returns: Base complexity value before growth scaling.

### computeComplexityPenalty

`(genome: import("C:/NeatapticTS/src/architecture/network").default, growth: number) => number`

Compute a structural complexity penalty scaled by a growth factor.

Complexity heuristic:
  (hidden nodes) + (connections) + (gates)
hidden nodes = total nodes - input - output (to avoid penalizing fixed I/O interface size).

Rationale: Encourages minimal / parsimonious networks by subtracting a term from fitness
proportional to network size, counteracting bloat. Growth hyper‑parameter tunes pressure.

Caching strategy: We memoize the base complexity (pre‑growth scaling) per genome when its
structural counts (nodes / connections / gates) are unchanged. This is safe because only
structural mutations alter these counts, and those invalidate earlier entries naturally
(since mutated genomes are distinct object references in typical NEAT flows).

Parameters:
- `genome` - - Candidate network whose complexity to measure.
- `growth` - - Positive scalar controlling strength of parsimony pressure.

Returns: Complexity * growth (used directly to subtract from fitness score).

### createSingleThreadFallbackFitness

`(set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").CostFunctionOrRef, amount: number, growth: number) => import("C:/NeatapticTS/src/architecture/network/network.types").FitnessSetup`

Build single-thread fallback fitness setup.

Parameters:
- `set` - - Dataset.
- `cost` - - Cost function.
- `amount` - - Repetition count.
- `growth` - - Complexity penalty scalar.

Returns: Single-thread fitness setup.

### evaluateGenomeAmountTimes

`(genome: import("C:/NeatapticTS/src/architecture/network").default, set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveCostFunction, amount: number) => number`

Evaluate one genome repeatedly and accumulate negative error.

Parameters:
- `genome` - - Genome under evaluation.
- `set` - - Dataset used for evaluation.
- `cost` - - Cost function reference.
- `amount` - - Number of repeated evaluations.

Returns: Accumulated negative error or -Infinity on failure.

### evaluateGenomeErrorSafely

`(genome: import("C:/NeatapticTS/src/architecture/network").default, set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveCostFunction) => number | null`

Evaluate one genome and return error, with warning-protected failure handling.

Parameters:
- `genome` - - Genome under evaluation.
- `set` - - Dataset used for evaluation.
- `cost` - - Cost function reference.

Returns: Error value, or null when evaluation fails.

### evaluateGenomeWithWorker

`(worker: import("C:/NeatapticTS/src/multithreading/types").TestWorkerInstance, genome: import("C:/NeatapticTS/src/architecture/network").default, growth: number) => Promise<void>`

Evaluate one genome with a worker and assign penalized score.

Parameters:
- `worker` - - Worker instance.
- `genome` - - Genome under evaluation.
- `growth` - - Complexity penalty scalar.

Returns: Promise resolving once score assignment completes.

### evolveNetwork

`(set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], options: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => Promise<{ error: number; iterations: number; time: number; }>`

Evolve (optimize) the current network's topology and weights using a NEAT-like evolutionary loop
until a stopping criterion (target error or max iterations) is met.

High-level process:
 1. Validate dataset shape (input/output vector sizes must match network I/O counts).
 2. Normalize / default option values and construct an internal configuration summary.
 3. Build appropriate fitness evaluation function (single or multi-thread).
 4. Initialize a Neat population (optionally with speciation) seeded by this network.
 5. Iteratively call neat.evolve():
      - Retrieve fittest genome + its fitness.
      - Derive an error metric from fitness (inverse relationship considering complexity penalty).
      - Track best genome overall (elitism) and perform logging/scheduling callbacks.
      - Break if error criterion satisfied or iterations exceeded.
 6. Replace this network's internal structural arrays with the best discovered genome's (in-place upgrade).
 7. Cleanup any worker threads and report final statistics.

Fitness / Error relationship:
  fitness = -error - complexityPenalty  =>  error = -(fitness - complexityPenalty)
We recompute error from the stored fitness plus penalty to ensure consistent reporting.

Resilience strategies:
 - Guard against infinite / NaN errors; after MAX_INF consecutive invalid errors we abort.
 - Fallback for tiny populations: increase mutation aggressiveness to prevent premature convergence.

Parameters:
- `this` - - Bound  {@link Network} instance being evolved in-place.
 *
- `set` - - Supervised dataset (array of {input, output}).
- `options` - - Evolution options (see README / docs). Key fields include:
- iterations: maximum generations (if omitted must supply error target)
- error: target error threshold (if omitted must supply iterations)
- growth: complexity penalty scaling
- amount: number of score evaluations (averaged) per genome
- threads: desired worker count (>=2 enables multi-thread path if available)
- popsize / populationSize: population size
- schedule: { iterations: number, function: (ctx) => void } periodic callback
- log: generation interval for console logging
- clear: whether to call network.clear() after adopting best genome

Returns: Summary object { error, iterations, time(ms) }.

### getCachedComplexityBase

`(genome: import("C:/NeatapticTS/src/architecture/network").default, structureCounts: { nodeCount: number; connectionCount: number; gateCount: number; }) => number | null`

Retrieve cached complexity base if cached structure counts still match.

Parameters:
- `genome` - - Candidate network whose cached complexity is queried.
- `structureCounts` - - Current structural counts.

Returns: Cached complexity base or null when cache miss occurs.

### getGenomeStructureCounts

`(genome: import("C:/NeatapticTS/src/architecture/network").default) => { nodeCount: number; connectionCount: number; gateCount: number; }`

Get structural counts used by complexity heuristic.

Parameters:
- `genome` - - Candidate network whose structure is being measured.

Returns: Structural counts used for complexity computation.

### installWorkerTerminationHook

`(options: Record<string, unknown>, workers: import("C:/NeatapticTS/src/multithreading/types").TestWorkerInstance[]) => void`

Register worker termination hook onto options object.

Parameters:
- `options` - - Evolve options object.
- `workers` - - Spawned worker instances.

Returns: Nothing.

### resolveCostName

`(cost: import("C:/NeatapticTS/src/architecture/network/network.types").CostFunctionOrRef) => string`

Resolve serializable cost name for worker payload.

Parameters:
- `cost` - - Cost function or cost reference.

Returns: Cost name string.

### resolveTestWorkerConstructor

`() => Promise<import("C:/NeatapticTS/src/multithreading/types").TestWorkerConstructor | null>`

Resolve worker constructor for current runtime environment.

Returns: Worker constructor or null when unavailable.

### spawnTestWorkers

`(workerConstructor: import("C:/NeatapticTS/src/multithreading/types").TestWorkerConstructor, serializedSet: number[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").CostFunctionOrRef, threads: number) => import("C:/NeatapticTS/src/multithreading/types").TestWorkerInstance[]`

Spawn worker instances up to requested thread count.

Parameters:
- `workerConstructor` - - Worker constructor resolved for current runtime.
- `serializedSet` - - Serialized dataset shared with workers.
- `cost` - - Cost function or cost reference.
- `threads` - - Requested worker count.

Returns: Spawned worker instances.

### warnGenomeEvaluationFailure

`(error: unknown) => void`

Emit warning when genome evaluation fails.

Parameters:
- `error` - - Unknown evaluation error.

Returns: Nothing.
