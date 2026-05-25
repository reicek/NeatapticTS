# architecture/network/evolve

Network-level neuroevolution loop for improving one runnable graph in place.

This chapter sits between the lightweight `Network` facade and the broader
`neat/` subsystem. Call `network.evolve()` when the question is "improve
this graph against a supervised dataset" without first wiring a separate
experiment harness. The method treats the current network as a seed genome,
configures a short-lived NEAT runtime around it, evaluates descendants, then
copies the best discovered structure back into the original instance.

The folder is split by the same stages the public call executes. `setup`
validates dataset shape and resolves defaults. `fitness` turns prediction
error into a comparable score. `loop` owns generation-to-generation stopping
logic. `finalize` adopts the winning genome and tears down worker resources.
Keeping those shelves separate makes the README read like an evolution run
rather than an alphabetical pile of helpers.

```mermaid
flowchart LR
  Seed[Seed Network] --> Normalize[Normalize options and dataset]
  Normalize --> Fitness[Build fitness evaluator]
  Fitness --> Neat[Create temporary NEAT runtime]
  Neat --> Loop[Run evolve loop]
  Loop --> Adopt[Adopt best genome into original network]
```

Use this boundary when you want a bounded local search over network
structure, not a long-lived population controller. `error` answers "stop
when good enough", `iterations` answers "stop after this many generations",
and `growth` answers "how much should extra structure cost while searching".

For compact background reading on the wider search family behind this folder,
see Wikipedia contributors,
[Evolutionary algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm).
The implementation here is intentionally narrower: it keeps the public call
site small while reusing the repo's NEAT runtime under the hood.

Example: stop when the network gets below an error target or hits a
generation cap.

```ts
const summary = await network.evolve(trainingSet, {
  error: 0.02,
  iterations: 500,
  growth: 0.0005,
});
```

Example: cap the search more tightly and fan evaluation across workers when
dataset scoring dominates.

```ts
const summary = await network.evolve(trainingSet, {
  iterations: 120,
  threads: 2,
  log: 20,
});
```

## architecture/network/evolve/network.evolve.utils.ts

### evolveNetwork

```ts
evolveNetwork(
  set: TrainingSample[],
  options: EvolveOptions,
): Promise<{ error: number; iterations: number; time: number; }>
```

Evolves a network with a NEAT-style search loop until an error target or generation limit is reached.

Overview:
- This method treats the current network as a *seed genome* and explores better variants.
- Candidate genomes are scored by prediction error plus a structural complexity penalty.
- The best discovered genome is copied back into the current instance (in-place upgrade).

Typical usage guidance:
- Use `error` when you care about reaching a quality threshold.
- Use `iterations` when you need deterministic runtime bounds.
- Use both when you want "stop when good enough, otherwise cap time" behavior.
- Increase `threads` only when worker support exists and dataset evaluation is expensive.

Parameters:
- `this` - Bound Network instance that receives the best evolved structure.
- `set` - Supervised samples; sample input/output dimensions must match network I/O.
- `options` - Evolution hyperparameters and stop conditions.

Returns: Final summary containing best error estimate, generations processed, and elapsed milliseconds.

Example:

```ts
const summary = await network.evolve(trainingSet, {
  error: 0.02,
  iterations: 500,
  growth: 0.0005,
  threads: 2,
});
console.log(summary.error, summary.iterations, summary.time);
```

## architecture/network/evolve/network.evolve.setup.utils.ts

### applySmallPopulationHeuristics

```ts
applySmallPopulationHeuristics(
  neatInstance: NeatRuntime,
  evolveOptions: EvolveOptions,
): void
```

Apply conservative mutation-rate and mutation-amount defaults for tiny populations to reduce early stagnation risk while avoiding aggressive overrides when callers already configured explicit mutation controls.

Parameters:
- `neatInstance` - Active NEAT instance.
- `evolveOptions` - Evolve options object.

Returns: Nothing.

### assertEvolutionDatasetCompatibility

```ts
assertEvolutionDatasetCompatibility(
  network: default,
  dataSet: TrainingSample[],
): void
```

Validate that training samples exist and that each sample shape matches the network input and output arity before any evolution work starts.

Parameters:
- `network` - Network being evolved.
- `dataSet` - Supervised dataset.

Returns: Nothing.

### configureNeatOptions

```ts
configureNeatOptions(
  network: default,
  evolveOptions: EvolveOptions,
): void
```

Normalize options consumed by the NEAT constructor, including compatibility mapping between populationSize and popsize, plus default speciation behavior needed for historical evolve-call consistency.

Parameters:
- `network` - Network instance being evolved.
- `evolveOptions` - Evolve options object.

Returns: Nothing.

### createEvolutionConfig

```ts
createEvolutionConfig(
  settingsToSummarize: EvolutionSettings,
): EvolutionConfig | undefined
```

Build a structured evolution-config snapshot for diagnostics, callback payloads, and log hooks whenever scheduling metadata exists, while keeping omitted schedule paths represented as an explicit undefined result.

Parameters:
- `settingsToSummarize` - Scalar evolution settings.

Returns: Optional summary config.

### createNeatInstance

```ts
createNeatInstance(
  network: default,
  fitnessFunction: EvolutionFitnessFunction,
  evolveOptions: EvolveOptions,
): Promise<NeatRuntime>
```

Lazy-load the NEAT runtime class and create an instance bound to the prepared fitness evaluator and normalized options so evolution setup can remain lightweight until construction is truly needed.

Parameters:
- `network` - Network instance being evolved.
- `fitnessFunction` - Prepared fitness evaluator.
- `evolveOptions` - Evolve options object.

Returns: Constructed NEAT instance.

### getNormalizedOptions

```ts
getNormalizedOptions(
  evolveOptions: EvolveOptions,
): EvolveOptions
```

Return the normalized evolve-options object that every downstream setup helper reads so defaults, compatibility shims, and diagnostics all start from one stable and predictable structure.

Parameters:
- `evolveOptions` - Incoming evolve options.

Returns: Safe options object.

### prepareFitnessFunction

```ts
prepareFitnessFunction(
  dataSet: TrainingSample[],
  resolvedSettings: EvolutionSettings,
  evolveOptions: EvolveOptions,
): Promise<FitnessSetup>
```

Build the effective fitness evaluator by selecting either single-thread or worker-backed execution from resolved threading settings, then return both the callable function and the resolved runtime thread count.

Parameters:
- `dataSet` - Supervised dataset.
- `resolvedSettings` - Scalar evolution settings.
- `evolveOptions` - Evolve options object.

Returns: Fitness function and resolved thread count.

### resolveEvolutionSettings

```ts
resolveEvolutionSettings(
  evolveOptions: EvolveOptions,
): EvolutionSettings
```

Resolve scalar evolution settings with explicit and stable defaults so later setup phases can run deterministically without repeating option-default logic in several different orchestration branches.

Parameters:
- `evolveOptions` - Evolve options object.

Returns: Normalized scalar settings.

### resolveStopConditions

```ts
resolveStopConditions(
  evolveOptions: EvolveOptions,
  initialTargetError: number,
): EvolutionStopConditions
```

Resolve stopping conditions while preserving legacy iteration-and-error semantics so existing callers keep the same termination behavior, including compatibility with historical zero-iteration and disabled-error workflows.

Parameters:
- `evolveOptions` - Evolve options object.
- `initialTargetError` - Target error resolved from options.

Returns: Final stop conditions.

### warnIfNoBestGenomeMayOccur

```ts
warnIfNoBestGenomeMayOccur(
  neatInstance: NeatRuntime,
  evolveOptions: EvolveOptions,
): void
```

Emit a best-genome advisory warning for zero-iteration runs where legacy flows can complete without setting a champion, preventing silent confusion when a caller expects a populated best network.

Parameters:
- `neatInstance` - Active NEAT instance.
- `evolveOptions` - Evolve options object.

Returns: Nothing.

## architecture/network/evolve/network.evolve.fitness.utils.ts

### buildMultiThreadFitness

```ts
buildMultiThreadFitness(
  set: TrainingSample[],
  cost: CostFunctionOrRef,
  amount: number,
  growth: number,
  threads: number,
  options: Record<string, unknown>,
): Promise<FitnessSetup>
```

Build worker-based population fitness setup that serializes the dataset once and assigns a population-scoring function.
When worker construction fails, this helper falls back to the single-thread evaluator automatically.

Parameters:
- `set` - Dataset.
- `cost` - Cost function or reference.
- `amount` - Repetition count.
- `growth` - Complexity penalty scalar.
- `threads` - Desired worker count.
- `options` - Evolution options object.

Returns: Population fitness setup.

### buildPopulationWorkerFitnessFunction

```ts
buildPopulationWorkerFitnessFunction(
  workers: TestWorkerInstance[],
  growth: number,
): PopulationFitnessFunction
```

Build population-level fitness function powered by worker queue.

Parameters:
- `workers` - Spawned worker instances.
- `growth` - Complexity penalty scalar.

Returns: Population-level fitness function.

### buildSingleThreadFitness

```ts
buildSingleThreadFitness(
  set: TrainingSample[],
  cost: EvolveCostFunction,
  amount: number,
  growth: number,
): SingleGenomeFitnessFunction
```

Build a single-threaded genome fitness evaluator that repeats scoring, applies complexity penalty, and normalizes by evaluation count.
This path is the deterministic fallback when worker-based evaluation is unavailable.

Parameters:
- `set` - Dataset of training samples.
- `cost` - Cost function reference.
- `amount` - Number of repeated evaluations.
- `growth` - Complexity penalty scalar.

Returns: Single-genome fitness function.

### cacheComplexityBase

```ts
cacheComplexityBase(
  genome: default,
  structureCounts: GenomeStructureCounts,
  complexityBase: number,
): void
```

Store complexity base cache entry for future reuse.

Parameters:
- `genome` - Candidate network used as cache key.
- `structureCounts` - Current structural counts.
- `complexityBase` - Computed base complexity value.

Returns: Nothing.

### claimNextGenome

```ts
claimNextGenome(
  context: PopulationWorkerEvaluationContext,
): default | undefined
```

Claims the next genome index from shared queue state.

Parameters:
- `context` - Population evaluation context.

Returns: Next genome, or undefined when queue is exhausted.

### computeComplexityBase

```ts
computeComplexityBase(
  genome: default,
  structureCounts: GenomeStructureCounts,
): number
```

Compute non-scaled complexity base from structural counts.

Parameters:
- `genome` - Candidate network whose complexity to compute.
- `structureCounts` - Current structural counts.

Returns: Base complexity value before growth scaling.

### computeComplexityPenalty

```ts
computeComplexityPenalty(
  genome: default,
  growth: number,
): number
```

Compute structural complexity penalty scaled by growth so larger genomes receive deterministic parsimony pressure during fitness scoring.
Cached structure counts avoid repeated recomputation for unchanged genome objects.

Parameters:
- `genome` - Candidate network whose complexity to measure.
- `growth` - Positive scalar controlling parsimony pressure.

Returns: Complexity penalty.

### createPopulationWorkerEvaluationContext

```ts
createPopulationWorkerEvaluationContext(
  sourceWorkers: TestWorkerInstance[],
  sourcePopulation: default[],
  sourceGrowth: number,
  sourceResolve: () => void,
): PopulationWorkerEvaluationContext
```

Creates the shared evaluation context for one population run.

Parameters:
- `sourceWorkers` - Worker pool.
- `sourcePopulation` - Population to evaluate.
- `sourceGrowth` - Complexity penalty scalar.
- `sourceResolve` - Promise resolver.

Returns: Population evaluation context.

### createSingleThreadFallbackFitness

```ts
createSingleThreadFallbackFitness(
  set: TrainingSample[],
  cost: CostFunctionOrRef,
  amount: number,
  growth: number,
): FitnessSetup
```

Build single-thread fallback fitness setup.

Parameters:
- `set` - Dataset.
- `cost` - Cost function.
- `amount` - Repetition count.
- `growth` - Complexity penalty scalar.

Returns: Single-thread fitness setup.

### createWorkerTraversalContext

```ts
createWorkerTraversalContext(
  context: PopulationWorkerEvaluationContext,
  worker: TestWorkerInstance,
): WorkerTraversalContext
```

Creates traversal context for one worker.

Parameters:
- `context` - Population evaluation context.
- `worker` - Worker instance.

Returns: Worker traversal context.

### evaluateGenomeAmountTimes

```ts
evaluateGenomeAmountTimes(
  genome: default,
  set: TrainingSample[],
  cost: EvolveCostFunction,
  amount: number,
): number
```

Evaluate one genome repeatedly and accumulate negative error.

Parameters:
- `genome` - Genome under evaluation.
- `set` - Dataset used for evaluation.
- `cost` - Cost function reference.
- `amount` - Number of repeated evaluations.

Returns: Accumulated negative error or -Infinity on failure.

### evaluateGenomeErrorSafely

```ts
evaluateGenomeErrorSafely(
  genome: default,
  set: TrainingSample[],
  cost: EvolveCostFunction,
): number | null
```

Evaluate one genome and return error, with warning-protected failure handling.

Parameters:
- `genome` - Genome under evaluation.
- `set` - Dataset used for evaluation.
- `cost` - Cost function reference.

Returns: Error value, or null when evaluation fails.

### evaluateGenomeWithWorker

```ts
evaluateGenomeWithWorker(
  worker: TestWorkerInstance,
  genome: default,
  growth: number,
): Promise<void>
```

Evaluate one genome with a worker and assign a penalized score that includes structural complexity pressure.
Non-numeric worker results are ignored so calling loops can continue safely.

Parameters:
- `worker` - Worker instance.
- `genome` - Genome under evaluation.
- `growth` - Complexity penalty scalar.

Returns: Promise resolving when score assignment completes.

### finalizeWorker

```ts
finalizeWorker(
  context: PopulationWorkerEvaluationContext,
): void
```

Marks one worker as completed and resolves when all workers finish.

Parameters:
- `context` - Population evaluation context.

Returns: Nothing.

### getCachedComplexityBase

```ts
getCachedComplexityBase(
  genome: default,
  structureCounts: GenomeStructureCounts,
): number | null
```

Retrieve cached complexity base if cached structure counts still match.

Parameters:
- `genome` - Candidate network whose cached complexity is queried.
- `structureCounts` - Current structural counts.

Returns: Cached complexity base or null when cache miss occurs.

### getGenomeStructureCounts

```ts
getGenomeStructureCounts(
  genome: default,
): GenomeStructureCounts
```

Get structural counts used by complexity heuristic.

Parameters:
- `genome` - Candidate network whose structure is being measured.

Returns: Structural counts used for complexity computation.

### hasNoWorkers

```ts
hasNoWorkers(
  context: PopulationWorkerEvaluationContext,
): boolean
```

Checks whether there are workers available to process genomes.

Parameters:
- `context` - Population evaluation context.

Returns: True when worker pool is empty.

### installWorkerTerminationHook

```ts
installWorkerTerminationHook(
  options: Record<string, unknown>,
  workers: TestWorkerInstance[],
): void
```

Register a worker termination hook onto the evolve options object so spawned workers can be cleaned up deterministically.
Termination errors are swallowed to avoid masking primary evolution outcomes.

Parameters:
- `options` - Evolve options object.
- `workers` - Spawned worker instances.

Returns: Nothing.

### resolveCostName

```ts
resolveCostName(
  cost: CostFunctionOrRef,
): string
```

Resolve serializable cost name for worker payload.

Parameters:
- `cost` - Cost function or cost reference.

Returns: Cost name string.

### resolveEvaluation

```ts
resolveEvaluation(
  context: PopulationWorkerEvaluationContext,
): void
```

Resolves the population evaluation promise.

Parameters:
- `context` - Population evaluation context.

Returns: Nothing.

### resolveTestWorkerConstructor

```ts
resolveTestWorkerConstructor(): Promise<TestWorkerConstructor | null>
```

Resolve worker constructor for current runtime environment.

Returns: Worker constructor or null when unavailable.

### runWorkerTraversalStep

```ts
runWorkerTraversalStep(
  traversalContext: WorkerTraversalContext,
): void
```

Runs one asynchronous traversal step for a worker.

Parameters:
- `traversalContext` - Worker traversal context.

Returns: Nothing.

### spawnTestWorkers

```ts
spawnTestWorkers(
  workerConstructor: TestWorkerConstructor,
  serializedSet: number[],
  cost: CostFunctionOrRef,
  threads: number,
): TestWorkerInstance[]
```

Spawn worker instances up to requested thread count.

Parameters:
- `workerConstructor` - Worker constructor resolved for runtime.
- `serializedSet` - Serialized dataset shared with workers.
- `cost` - Cost function or cost reference.
- `threads` - Requested worker count.

Returns: Spawned worker instances.

### startWorkerTraversal

```ts
startWorkerTraversal(
  context: PopulationWorkerEvaluationContext,
): void
```

Starts traversal loops for all workers.

Parameters:
- `context` - Population evaluation context.

Returns: Nothing.

### warnGenomeEvaluationFailure

```ts
warnGenomeEvaluationFailure(
  error: unknown,
): void
```

Emit warning when genome evaluation fails.

Parameters:
- `error` - Unknown evaluation error.

Returns: Nothing.

## architecture/network/evolve/network.evolve.loop.utils.ts

### applyEvolutionStep

```ts
applyEvolutionStep(
  state: EvolutionLoopState,
  evolvedGenome: default,
  growth: number,
): void
```

Applies one evolve() result to loop state.

Parameters:
- `state` - Mutable loop state.
- `evolvedGenome` - Genome returned by NEAT evolve step.
- `growth` - Complexity growth scalar.

Returns: Nothing.

### createInitialLoopState

```ts
createInitialLoopState(): EvolutionLoopState
```

Creates initial loop state snapshot.

Returns: Initial loop state.

### deriveErrorFromFitness

```ts
deriveErrorFromFitness(
  fitness: number,
  genome: default,
  growth: number,
): number
```

Derive error from fitness by inverting score composition.

Parameters:
- `fitness` - Fitness value from fittest genome.
- `genome` - Fittest genome.
- `growth` - Complexity growth scalar.

Returns: Derived error value.

### runEvolutionLoop

```ts
runEvolutionLoop(
  neatInstance: NeatRuntime,
  resolvedSettings: EvolutionSettings,
  targetError: number,
  iterations: number | undefined,
): Promise<{ error: number; bestGenome: default | undefined; }>
```

Run the core evolution loop until error target or iteration limit is reached.

Parameters:
- `neatInstance` - Active NEAT instance.
- `resolvedSettings` - Scalar evolution settings.
- `targetError` - Effective target error (-1 means disabled).
- `iterations` - Optional max iteration count.

Returns: Loop result snapshot.

### runScheduleCallbackSafely

```ts
runScheduleCallbackSafely(
  scheduleConfig: { iterations: number; function: (stats: { fitness: number; error: number; iteration: number; }) => void; } | undefined,
  generation: number,
  bestFitness: number,
  error: number,
): void
```

Run schedule callback if schedule trigger is reached.

Parameters:
- `scheduleConfig` - Optional schedule configuration.
- `generation` - Current generation.
- `bestFitness` - Current best fitness.
- `error` - Current error.

Returns: Nothing.

### shouldAbortForInvalidErrors

```ts
shouldAbortForInvalidErrors(
  state: EvolutionLoopState,
): boolean
```

Determines whether loop must abort due to invalid-error streak.

Parameters:
- `state` - Mutable loop state.

Returns: True when invalid-error threshold is reached.

### shouldContinueEvolution

```ts
shouldContinueEvolution(
  currentError: number,
  targetError: number,
  iterationsSpecified: boolean,
  currentGeneration: number,
  maxIterations: number | undefined,
): boolean
```

Determine whether evolution loop should continue.

Parameters:
- `currentError` - Current derived error value.
- `targetError` - Effective target error (-1 means disabled).
- `iterationsSpecified` - Whether iterations limit is active.
- `currentGeneration` - Current NEAT generation index.
- `maxIterations` - Maximum iteration limit.

Returns: True when loop should continue.

### updateBestGenomeIfImproved

```ts
updateBestGenomeIfImproved(
  currentBestFitness: number,
  currentBestGenome: default | undefined,
  candidateFitness: number,
  candidateGenome: default,
): { bestFitness: number; bestGenome: default | undefined; }
```

Update best fitness/genome snapshot when improved.

Parameters:
- `currentBestFitness` - Current best fitness.
- `currentBestGenome` - Current best genome.
- `candidateFitness` - Candidate fitness.
- `candidateGenome` - Candidate genome.

Returns: Updated best snapshot.

### updateInvalidErrorCounter

```ts
updateInvalidErrorCounter(
  currentCount: number,
  currentError: number,
): number
```

Update invalid-error counter.

Parameters:
- `currentCount` - Current consecutive invalid-error count.
- `currentError` - Current derived error value.

Returns: Updated guard state.

## architecture/network/evolve/network.evolve.finalize.utils.ts

### adoptBestGenomeOrWarn

```ts
adoptBestGenomeOrWarn(
  network: default,
  neatInstance: NeatRuntime,
  bestGenome: default | undefined,
  clearState: boolean,
): void
```

Finalization helper that adopts the best evolved genome into the caller's
network when one exists.

If no best genome is available, the optional NEAT warning hook is invoked so
callers can surface diagnostic context without throwing from finalize flow.

### buildEvolutionSummary

```ts
buildEvolutionSummary(
  error: number,
  iterations: number,
  loopStartTime: number,
): EvolutionSummary
```

Build the final evolution summary payload returned by the evolve loop.

Parameters:
- `error` - Final loop error.
- `iterations` - Final generation count.
- `loopStartTime` - Loop start timestamp.

Returns: Evolution summary object.

### terminateWorkersSafely

```ts
terminateWorkersSafely(
  evolveOptions: EvolveOptions,
): void
```

Best-effort shutdown for worker terminators attached to evolve options.

This keeps finalize paths resilient when background evaluators were used and
avoids leaking worker resources if teardown throws.

Parameters:
- `evolveOptions` - Evolve options object.

Returns: Nothing.

## architecture/network/evolve/network.evolve.utils.types.ts

Shared dataset compatibility error message.

### DATASET_COMPATIBILITY_ERROR_MESSAGE

Shared dataset compatibility error message.

### DEFAULT_EVALUATION_AMOUNT

Default repeated evaluation amount.

### DEFAULT_GROWTH

Default complexity growth penalty.

### DEFAULT_LOG_INTERVAL

Default logging frequency value.

### DEFAULT_TARGET_ERROR

Default target error used when omitted.

### DEFAULT_THREAD_COUNT

Default single-thread worker count.

### DISABLED_TARGET_ERROR

Sentinel target error indicating that error-based stopping is disabled.

### EvolutionSummary

Shared evolution summary payload.

### GenomeStructureCounts

Structural counts used by complexity heuristics.

### MAX_CONSECUTIVE_INVALID_ERRORS

Maximum consecutive invalid errors tolerated before loop abort.

### SMALL_POPULATION_MUTATION_AMOUNT

Mutation amount fallback used for very small populations.

### SMALL_POPULATION_MUTATION_RATE

Mutation rate fallback used for very small populations.

### SMALL_POPULATION_THRESHOLD

Population threshold considered "small" for mutation heuristics.

### STOPPING_CONDITION_REQUIRED_ERROR_MESSAGE

Shared evolve stopping-condition validation error.

### ZERO_ITERATIONS

Explicit zero-iteration value.

## architecture/network/evolve/network.evolve.errors.ts

Raised when the evolve dataset is missing or does not match network IO.

Example:

```ts
throw new NetworkEvolveDatasetCompatibilityError(
  'Dataset should have at least one sample and matching input/output sizes.',
);
```

### NetworkEvolveDatasetCompatibilityError

Contract for NetworkEvolveDatasetCompatibilityError.

### NetworkEvolveStoppingConditionRequiredError

Contract for NetworkEvolveStoppingConditionRequiredError.
