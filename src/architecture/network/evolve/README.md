# architecture/network/evolve

## architecture/network/evolve/network.evolve.utils.types.ts

### network.evolve.utils.types

Shared dataset compatibility error message.

### DATASET_COMPATIBILITY_ERROR_MESSAGE

### DEFAULT_EVALUATION_AMOUNT

### DEFAULT_GROWTH

### DEFAULT_LOG_INTERVAL

### DEFAULT_TARGET_ERROR

### DEFAULT_THREAD_COUNT

### DISABLED_TARGET_ERROR

### EvolutionSummary

Shared evolution summary payload.

### GenomeStructureCounts

Structural counts used by complexity heuristics.

### MAX_CONSECUTIVE_INVALID_ERRORS

### SMALL_POPULATION_MUTATION_AMOUNT

### SMALL_POPULATION_MUTATION_RATE

### SMALL_POPULATION_THRESHOLD

### STOPPING_CONDITION_REQUIRED_ERROR_MESSAGE

### ZERO_ITERATIONS

## architecture/network/evolve/network.evolve.utils.ts

### evolveNetwork

`(set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], options: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => Promise<{ error: number; iterations: number; time: number; }>`

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
- `this` - - Bound Network instance that receives the best evolved structure.
- `set` - - Supervised samples; sample input/output dimensions must match network I/O.
- `options` - - Evolution hyperparameters and stop conditions.

Returns: Final summary containing best error estimate, generations processed, and elapsed milliseconds.

## architecture/network/evolve/network.evolve.loop.utils.ts

### applyEvolutionStep

`(state: import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionLoopState, evolvedGenome: import("C:/NeatapticTS/src/architecture/network").default, growth: number) => void`

Applies one evolve() result to loop state.

Parameters:
- `state` - - Mutable loop state.
- `evolvedGenome` - - Genome returned by NEAT evolve step.
- `growth` - - Complexity growth scalar.

Returns: Nothing.

### createInitialLoopState

`() => import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionLoopState`

Creates initial loop state snapshot.

Returns: Initial loop state.

### deriveErrorFromFitness

`(fitness: number, genome: import("C:/NeatapticTS/src/architecture/network").default, growth: number) => number`

Derive error from fitness by inverting score composition.

Parameters:
- `fitness` - - Fitness value from fittest genome.
- `genome` - - Fittest genome.
- `growth` - - Complexity growth scalar.

Returns: Derived error value.

### runEvolutionLoop

`(neatInstance: import("C:/NeatapticTS/src/architecture/network/network.types").NeatRuntime, resolvedSettings: import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionSettings, targetError: number, iterations: number | undefined) => Promise<{ error: number; bestGenome: import("C:/NeatapticTS/src/architecture/network").default | undefined; }>`

Run core evolution loop until stop condition is met.

Parameters:
- `neatInstance` - - Active NEAT instance.
- `resolvedSettings` - - Scalar evolution settings.
- `targetError` - - Effective target error (-1 means disabled).
- `iterations` - - Optional max iteration count.

Returns: Loop result snapshot.

### runScheduleCallbackSafely

`(scheduleConfig: { iterations: number; function: (stats: { fitness: number; error: number; iteration: number; }) => void; } | undefined, generation: number, bestFitness: number, error: number) => void`

Run schedule callback if schedule trigger is reached.

Parameters:
- `scheduleConfig` - - Optional schedule configuration.
- `generation` - - Current generation.
- `bestFitness` - - Current best fitness.
- `error` - - Current error.

Returns: Nothing.

### shouldAbortForInvalidErrors

`(state: import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionLoopState) => boolean`

Determines whether loop must abort due to invalid-error streak.

Parameters:
- `state` - - Mutable loop state.

Returns: True when invalid-error threshold is reached.

### shouldContinueEvolution

`(currentError: number, targetError: number, iterationsSpecified: boolean, currentGeneration: number, maxIterations: number | undefined) => boolean`

Determine whether evolution loop should continue.

Parameters:
- `currentError` - - Current derived error value.
- `targetError` - - Effective target error (-1 means disabled).
- `iterationsSpecified` - - Whether iterations limit is active.
- `currentGeneration` - - Current NEAT generation index.
- `maxIterations` - - Maximum iteration limit.

Returns: True when loop should continue.

### updateBestGenomeIfImproved

`(currentBestFitness: number, currentBestGenome: import("C:/NeatapticTS/src/architecture/network").default | undefined, candidateFitness: number, candidateGenome: import("C:/NeatapticTS/src/architecture/network").default) => { bestFitness: number; bestGenome: import("C:/NeatapticTS/src/architecture/network").default | undefined; }`

Update best fitness/genome snapshot when improved.

Parameters:
- `currentBestFitness` - - Current best fitness.
- `currentBestGenome` - - Current best genome.
- `candidateFitness` - - Candidate fitness.
- `candidateGenome` - - Candidate genome.

Returns: Updated best snapshot.

### updateInvalidErrorCounter

`(currentCount: number, currentError: number) => number`

Update invalid-error counter.

Parameters:
- `currentCount` - - Current consecutive invalid-error count.
- `currentError` - - Current derived error value.

Returns: Updated guard state.

## architecture/network/evolve/network.evolve.setup.utils.ts

### applySmallPopulationHeuristics

`(neatInstance: import("C:/NeatapticTS/src/architecture/network/network.types").NeatRuntime, evolveOptions: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => void`

Increase mutation aggressiveness for tiny populations.

Parameters:
- `neatInstance` - - Active NEAT instance.
- `evolveOptions` - - Evolve options object.

Returns: Nothing.

### assertEvolutionDatasetCompatibility

`(network: import("C:/NeatapticTS/src/architecture/network").default, dataSet: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[]) => void`

Validate dataset existence and dimensional compatibility with network I/O.

Parameters:
- `network` - - Network being evolved.
- `dataSet` - - Supervised dataset.

Returns: Nothing.

### configureNeatOptions

`(network: import("C:/NeatapticTS/src/architecture/network").default, evolveOptions: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => void`

Normalize options used by NEAT constructor.

Parameters:
- `network` - - Network instance being evolved.
- `evolveOptions` - - Evolve options object.

Returns: Nothing.

### createEvolutionConfig

`(settingsToSummarize: import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionSettings) => import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionConfig | undefined`

Build optional structured evolution config summary.

Parameters:
- `settingsToSummarize` - - Scalar evolution settings.

Returns: Optional summary config.

### createNeatInstance

`(network: import("C:/NeatapticTS/src/architecture/network").default, fitnessFunction: import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionFitnessFunction, evolveOptions: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => Promise<import("C:/NeatapticTS/src/architecture/network/network.types").NeatRuntime>`

Lazy-load and create NEAT instance.

Parameters:
- `network` - - Network instance being evolved.
- `fitnessFunction` - - Prepared fitness evaluator.
- `evolveOptions` - - Evolve options object.

Returns: Constructed NEAT instance.

### getNormalizedOptions

`(evolveOptions: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions`

Ensure options object exists.

Parameters:
- `evolveOptions` - - Incoming evolve options.

Returns: Safe options object.

### prepareFitnessFunction

`(dataSet: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], resolvedSettings: import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionSettings, evolveOptions: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => Promise<import("C:/NeatapticTS/src/architecture/network/network.types").FitnessSetup>`

Build fitness function according to threading configuration.

Parameters:
- `dataSet` - - Supervised dataset.
- `resolvedSettings` - - Scalar evolution settings.
- `evolveOptions` - - Evolve options object.

Returns: Fitness function and resolved thread count.

### resolveEvolutionSettings

`(evolveOptions: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionSettings`

Resolve normalized scalar settings with defaults.

Parameters:
- `evolveOptions` - - Evolve options object.

Returns: Normalized scalar settings.

### resolveStopConditions

`(evolveOptions: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions, initialTargetError: number) => import("C:/NeatapticTS/src/architecture/network/network.types").EvolutionStopConditions`

Resolve stopping-condition semantics while preserving legacy behavior.

Parameters:
- `evolveOptions` - - Evolve options object.
- `initialTargetError` - - Target error resolved from options.

Returns: Final stop conditions.

### warnIfNoBestGenomeMayOccur

`(neatInstance: import("C:/NeatapticTS/src/architecture/network/network.types").NeatRuntime, evolveOptions: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => void`

Emit warning when zero-iteration configuration may produce no best genome.

Parameters:
- `neatInstance` - - Active NEAT instance.
- `evolveOptions` - - Evolve options object.

Returns: Nothing.

## architecture/network/evolve/network.evolve.fitness.utils.ts

### buildMultiThreadFitness

`(set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").CostFunctionOrRef, amount: number, growth: number, threads: number, options: Record<string, unknown>) => Promise<import("C:/NeatapticTS/src/architecture/network/network.types").FitnessSetup>`

Build worker-based population fitness setup.

Parameters:
- `set` - - Dataset.
- `cost` - - Cost function or reference.
- `amount` - - Repetition count.
- `growth` - - Complexity penalty scalar.
- `threads` - - Desired worker count.
- `options` - - Evolution options object.

Returns: Population fitness setup.

### buildPopulationWorkerFitnessFunction

`(workers: import("C:/NeatapticTS/src/multithreading/types").TestWorkerInstance[], growth: number) => import("C:/NeatapticTS/src/architecture/network/network.types").PopulationFitnessFunction`

Build population-level fitness function powered by worker queue.

Parameters:
- `workers` - - Spawned worker instances.
- `growth` - - Complexity penalty scalar.

Returns: Population-level fitness function.

### buildSingleThreadFitness

`(set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveCostFunction, amount: number, growth: number) => import("C:/NeatapticTS/src/architecture/network/network.types").SingleGenomeFitnessFunction`

Build a single-threaded genome fitness evaluator.

Parameters:
- `set` - - Dataset of training samples.
- `cost` - - Cost function reference.
- `amount` - - Number of repeated evaluations.
- `growth` - - Complexity penalty scalar.

Returns: Single-genome fitness function.

### cacheComplexityBase

`(genome: import("C:/NeatapticTS/src/architecture/network").default, structureCounts: import("C:/NeatapticTS/src/architecture/network/evolve/network.evolve.utils.types").GenomeStructureCounts, complexityBase: number) => void`

Store complexity base cache entry for future reuse.

Parameters:
- `genome` - - Candidate network used as cache key.
- `structureCounts` - - Current structural counts.
- `complexityBase` - - Computed base complexity value.

Returns: Nothing.

### claimNextGenome

`(context: import("C:/NeatapticTS/src/architecture/network/network.types").PopulationWorkerEvaluationContext) => import("C:/NeatapticTS/src/architecture/network").default | undefined`

Claims the next genome index from shared queue state.

Parameters:
- `context` - - Population evaluation context.

Returns: Next genome, or undefined when queue is exhausted.

### computeComplexityBase

`(genome: import("C:/NeatapticTS/src/architecture/network").default, structureCounts: import("C:/NeatapticTS/src/architecture/network/evolve/network.evolve.utils.types").GenomeStructureCounts) => number`

Compute non-scaled complexity base from structural counts.

Parameters:
- `genome` - - Candidate network whose complexity to compute.
- `structureCounts` - - Current structural counts.

Returns: Base complexity value before growth scaling.

### computeComplexityPenalty

`(genome: import("C:/NeatapticTS/src/architecture/network").default, growth: number) => number`

Compute structural complexity penalty scaled by growth.

Parameters:
- `genome` - - Candidate network whose complexity to measure.
- `growth` - - Positive scalar controlling parsimony pressure.

Returns: Complexity penalty.

### createPopulationWorkerEvaluationContext

`(sourceWorkers: import("C:/NeatapticTS/src/multithreading/types").TestWorkerInstance[], sourcePopulation: import("C:/NeatapticTS/src/architecture/network").default[], sourceGrowth: number, sourceResolve: () => void) => import("C:/NeatapticTS/src/architecture/network/network.types").PopulationWorkerEvaluationContext`

Creates the shared evaluation context for one population run.

Parameters:
- `sourceWorkers` - - Worker pool.
- `sourcePopulation` - - Population to evaluate.
- `sourceGrowth` - - Complexity penalty scalar.
- `sourceResolve` - - Promise resolver.

Returns: Population evaluation context.

### createSingleThreadFallbackFitness

`(set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").CostFunctionOrRef, amount: number, growth: number) => import("C:/NeatapticTS/src/architecture/network/network.types").FitnessSetup`

Build single-thread fallback fitness setup.

Parameters:
- `set` - - Dataset.
- `cost` - - Cost function.
- `amount` - - Repetition count.
- `growth` - - Complexity penalty scalar.

Returns: Single-thread fitness setup.

### createWorkerTraversalContext

`(context: import("C:/NeatapticTS/src/architecture/network/network.types").PopulationWorkerEvaluationContext, worker: import("C:/NeatapticTS/src/multithreading/types").TestWorkerInstance) => import("C:/NeatapticTS/src/architecture/network/network.types").WorkerTraversalContext`

Creates traversal context for one worker.

Parameters:
- `context` - - Population evaluation context.
- `worker` - - Worker instance.

Returns: Worker traversal context.

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

Returns: Promise resolving when score assignment completes.

### finalizeWorker

`(context: import("C:/NeatapticTS/src/architecture/network/network.types").PopulationWorkerEvaluationContext) => void`

Marks one worker as completed and resolves when all workers finish.

Parameters:
- `context` - - Population evaluation context.

Returns: Nothing.

### getCachedComplexityBase

`(genome: import("C:/NeatapticTS/src/architecture/network").default, structureCounts: import("C:/NeatapticTS/src/architecture/network/evolve/network.evolve.utils.types").GenomeStructureCounts) => number | null`

Retrieve cached complexity base if cached structure counts still match.

Parameters:
- `genome` - - Candidate network whose cached complexity is queried.
- `structureCounts` - - Current structural counts.

Returns: Cached complexity base or null when cache miss occurs.

### getGenomeStructureCounts

`(genome: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network/evolve/network.evolve.utils.types").GenomeStructureCounts`

Get structural counts used by complexity heuristic.

Parameters:
- `genome` - - Candidate network whose structure is being measured.

Returns: Structural counts used for complexity computation.

### hasNoWorkers

`(context: import("C:/NeatapticTS/src/architecture/network/network.types").PopulationWorkerEvaluationContext) => boolean`

Checks whether there are workers available to process genomes.

Parameters:
- `context` - - Population evaluation context.

Returns: True when worker pool is empty.

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

### resolveEvaluation

`(context: import("C:/NeatapticTS/src/architecture/network/network.types").PopulationWorkerEvaluationContext) => void`

Resolves the population evaluation promise.

Parameters:
- `context` - - Population evaluation context.

Returns: Nothing.

### resolveTestWorkerConstructor

`() => Promise<import("C:/NeatapticTS/src/multithreading/types").TestWorkerConstructor | null>`

Resolve worker constructor for current runtime environment.

Returns: Worker constructor or null when unavailable.

### runWorkerTraversalStep

`(traversalContext: import("C:/NeatapticTS/src/architecture/network/network.types").WorkerTraversalContext) => void`

Runs one asynchronous traversal step for a worker.

Parameters:
- `traversalContext` - - Worker traversal context.

Returns: Nothing.

### spawnTestWorkers

`(workerConstructor: import("C:/NeatapticTS/src/multithreading/types").TestWorkerConstructor, serializedSet: number[], cost: import("C:/NeatapticTS/src/architecture/network/network.types").CostFunctionOrRef, threads: number) => import("C:/NeatapticTS/src/multithreading/types").TestWorkerInstance[]`

Spawn worker instances up to requested thread count.

Parameters:
- `workerConstructor` - - Worker constructor resolved for runtime.
- `serializedSet` - - Serialized dataset shared with workers.
- `cost` - - Cost function or cost reference.
- `threads` - - Requested worker count.

Returns: Spawned worker instances.

### startWorkerTraversal

`(context: import("C:/NeatapticTS/src/architecture/network/network.types").PopulationWorkerEvaluationContext) => void`

Starts traversal loops for all workers.

Parameters:
- `context` - - Population evaluation context.

Returns: Nothing.

### warnGenomeEvaluationFailure

`(error: unknown) => void`

Emit warning when genome evaluation fails.

Parameters:
- `error` - - Unknown evaluation error.

Returns: Nothing.

## architecture/network/evolve/network.evolve.finalize.utils.ts

### adoptBestGenomeOrWarn

`(network: import("C:/NeatapticTS/src/architecture/network").default, neatInstance: import("C:/NeatapticTS/src/architecture/network/network.types").NeatRuntime, bestGenome: import("C:/NeatapticTS/src/architecture/network").default | undefined, clearState: boolean) => void`

Adopt best genome structure or emit warning when unavailable.

Parameters:
- `network` - - Network instance being evolved.
- `neatInstance` - - Active NEAT instance.
- `bestGenome` - - Best genome snapshot.
- `clearState` - - Whether to clear network after adoption.

Returns: Nothing.

### buildEvolutionSummary

`(error: number, iterations: number, loopStartTime: number) => import("C:/NeatapticTS/src/architecture/network/evolve/network.evolve.utils.types").EvolutionSummary`

Build final evolve return payload.

Parameters:
- `error` - - Final loop error.
- `iterations` - - Final generation count.
- `loopStartTime` - - Loop start timestamp.

Returns: Evolution summary object.

### terminateWorkersSafely

`(evolveOptions: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => void`

Terminate worker resources registered in options.

Parameters:
- `evolveOptions` - - Evolve options object.

Returns: Nothing.
