# trainer

## trainer/trainer.types.ts

### FlappyGenerationEvaluationPlan

Generation-level rollout plans for staged evaluation.

### FlappyGenerationReport

Compact generation report used for training logs.

### FlappyTrainerNeatController

Local typed view for population-level fitness mode used by this trainer.

### FlappyTrainerNetwork

Network shape expected by the Flappy trainer.

### FlappyTrainerRuntimeState

Trainer runtime state shared by orchestration helpers.

### FlappyTrainerSetup

Immutable trainer setup values.

### ScoredGenomeEntry

Score carrier used for deterministic ordering helpers.

## trainer/trainer.ts

### handleTrainerMainError

`(error: unknown) => void`

Handles fatal `main` rejection path.

### runTrainer

`() => Promise<void>`

Flappy Bird neuroevolution demo.

This script runs a small NEAT population where each genome controls a bird.
The network sees a temporal observation (38 floats) and outputs two competing
action scores (`no flap` vs `flap`).

Run (from repo root):
`npx ts-node test/examples/flappy_bird/trainFlappyBird.ts`

## trainer/trainer.errors.ts

### trainer.errors

Prefix used when rendering unexpected trainer failures to stderr.

### FLAPPY_TRAINER_UNEXPECTED_ERROR_PREFIX

### formatTrainerErrorMessage

`(error: unknown) => string`

Formats unknown trainer failures into a stable human-readable message.

Parameters:

- `error` - - Unknown rejection reason from trainer execution.

Returns: Formatted error string for CLI logging.

## trainer/trainer.constants.ts

### FLAPPY_TRAINER_DEFAULT_ELITISM_COUNT

### FLAPPY_TRAINER_DEFAULT_POPULATION_SIZE

### FLAPPY_TRAINER_DEFAULT_RNG_SEED

### FLAPPY_TRAINER_DUMMY_FLAP_OUTPUT

### FLAPPY_TRAINER_DUMMY_NETWORK_ID

### FLAPPY_TRAINER_DUMMY_NO_FLAP_OUTPUT

### FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE

### FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT

### FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT

### FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT

### FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER

### FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION

### FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES

### FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES

### FLAPPY_TRAINER_FULL_ROLLOUT_PIPE_PROGRESS_TARGET

### FLAPPY_TRAINER_LOG_PARTS_DELIMITER

### FLAPPY_TRAINER_MUTATION_AMOUNT_END

### FLAPPY_TRAINER_MUTATION_AMOUNT_START

### FLAPPY_TRAINER_MUTATION_ANNEAL_GENERATIONS

### FLAPPY_TRAINER_MUTATION_RATE_END

### FLAPPY_TRAINER_MUTATION_RATE_START

### FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_AMOUNT

### FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_RATE

### FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT

### FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE

### FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES

### FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES

### FLAPPY_TRAINER_QUICK_ROLLOUT_MAX_FRAMES

### FLAPPY_TRAINER_QUICK_ROLLOUT_PIPE_PROGRESS_TARGET

### FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT

### FLAPPY_TRAINER_SCORE_MEDIAN_PERCENTILE

### FLAPPY_TRAINER_SCORE_P90_PERCENTILE

### FLAPPY_TRAINER_STOPPED_MESSAGE

## trainer/trainer.loop.service.ts

### applyMutationSchedule

`(neatController: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNeatController, mutationSchedule: import("test/examples/flappy_bird/trainer/trainer.evaluation-plan.utils").FlappyMutationSchedule) => void`

Applies mutation schedule values to the NEAT controller options.

Parameters:

- `neatController` - - Trainer NEAT controller.
- `mutationSchedule` - - Mutation schedule for current generation.

### LogGenerationSummaryCallback

`(generationLabel: number, mutationSchedule: import("test/examples/flappy_bird/trainer/trainer.evaluation-plan.utils").FlappyMutationSchedule, report: import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationReport | undefined, fittestGenome: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, fallbackEpisode: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyEpisodeResult) => void`

Callback signature for one-line generation logging.

### runTrainerEvolutionLoop

`(neatController: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNeatController, trainerRuntimeState: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerRuntimeState, logGenerationSummary: import("test/examples/flappy_bird/trainer/trainer.loop.service").LogGenerationSummaryCallback) => Promise<void>`

Runs the outer evolution loop until runtime stop is requested.

Parameters:

- `neatController` - - Trainer NEAT controller.
- `trainerRuntimeState` - - Mutable trainer runtime state.
- `logGenerationSummary` - - Callback that emits compact generation logs.

## trainer/trainer.setup.service.ts

### createNeatController

`(trainerSetup: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerSetup) => import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNeatController`

Builds the NEAT controller with baseline options.

Parameters:

- `trainerSetup` - - Immutable trainer setup values.

Returns: Typed NEAT controller used by the trainer loop.

### createTrainerRuntimeState

`() => import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerRuntimeState`

Creates mutable runtime state container.

Returns: Fresh runtime state used by loop orchestration.

### createTrainerSetup

`() => import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerSetup`

Creates immutable setup values for the trainer.

Returns: Default trainer setup values used for NEAT configuration.

### resolveNoopFitness

`() => number`

Trivial baseline fitness used before attaching population evaluator.

Returns: Constant zero fitness.

## trainer/trainer.report.service.ts

### buildGenerationReport

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], aggregateByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, generationEvaluationPlan: import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationEvaluationPlan) => import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationReport`

Builds a compact report for the current generation.

Parameters:

- `population` - - Current population.
- `aggregateByGenome` - - Aggregate evaluation results keyed by genome.
- `generationEvaluationPlan` - - Per-generation staged evaluation plan.

Returns: Aggregated generation report.

### logGenerationSummary

`(generationLabel: number, mutationSchedule: import("test/examples/flappy_bird/trainer/trainer.evaluation-plan.utils").FlappyMutationSchedule, report: import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationReport | undefined, fittestGenome: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, fallbackEpisode: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyEpisodeResult) => void`

Emits one compact generation log line.

Parameters:

- `generationLabel` - - Current generation label.
- `mutationSchedule` - - Active mutation schedule.
- `report` - - Optional aggregated generation report.
- `fittestGenome` - - Fittest genome returned by the NEAT controller.
- `fallbackEpisode` - - Fallback representative rollout episode.

Returns: Nothing.

## trainer/trainer.fitness.service.ts

### attachPopulationFitnessEvaluator

`(neatController: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNeatController, trainerRuntimeState: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerRuntimeState, elitismCount: number, dependencies: import("test/examples/flappy_bird/trainer/trainer.fitness.service").TrainerFitnessServiceDependencies) => void`

Attaches population-level staged evaluator to the NEAT controller.

Parameters:

- `neatController` - - Trainer NEAT controller.
- `trainerRuntimeState` - - Mutable trainer runtime state.
- `elitismCount` - - Number of elite genomes preserved each generation.
- `dependencies` - - Pure/impure helper callbacks used by the evaluator.

### createPopulationFitnessEvaluator

`(neatController: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNeatController, trainerRuntimeState: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerRuntimeState, elitismCount: number, dependencies: import("test/examples/flappy_bird/trainer/trainer.fitness.service").TrainerFitnessServiceDependencies) => (population: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[]) => Promise<void>`

Creates the asynchronous population fitness evaluator.

Parameters:

- `neatController` - - Trainer NEAT controller.
- `trainerRuntimeState` - - Mutable trainer runtime state.
- `elitismCount` - - Number of elite genomes preserved each generation.
- `dependencies` - - Pure/impure helper callbacks used by the evaluator.

Returns: Evaluator callback assigned to `neatController.fitness`.

### TrainerFitnessServiceDependencies

Callback dependencies required by the trainer fitness orchestration service.

## trainer/trainer.signals.service.ts

### handleTrainerStopSignal

`(trainerRuntimeState: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerRuntimeState) => void`

Handles one stop signal update.

Parameters:

- `trainerRuntimeState` - - Mutable trainer runtime state.

### registerTrainerStopSignals

`(trainerRuntimeState: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerRuntimeState) => void`

Registers graceful stop signal handlers.

Parameters:

- `trainerRuntimeState` - - Mutable trainer runtime state.

## trainer/trainer.evaluation.service.ts

### trainer.evaluation.service

Trainer evaluation compatibility facade.

The staged population-evaluation implementation now lives in the dedicated
`trainer/evaluation/` submodule so orchestration, scoring helpers, internal
contracts, and sub-services can evolve behind a focused boundary.

### commitPopulationScores

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], provisionalScoresByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>) => void`

Commits provisional scores to genome score fields.

Parameters:

- `population` - - Current population.
- `provisionalScoresByGenome` - - Final provisional score map.

Returns: Nothing.

### evaluatePopulationFullStage

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], generationEvaluationPlan: import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationEvaluationPlan, aggregateByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, provisionalScoresByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>, elitismCount: number) => void`

Executes the full evaluation stage over the top provisional candidates.

Parameters:

- `population` - - Current population.
- `generationEvaluationPlan` - - Per-generation staged evaluation plan.
- `aggregateByGenome` - - Mutable aggregate cache keyed by genome.
- `provisionalScoresByGenome` - - Mutable provisional score map.
- `elitismCount` - - Configured elitism count.

Returns: Nothing.

### evaluatePopulationQuickStage

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], generationEvaluationPlan: import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationEvaluationPlan, aggregateByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, provisionalScoresByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>) => void`

Executes the quick evaluation stage over the full population.

Parameters:

- `population` - - Current population.
- `generationEvaluationPlan` - - Per-generation staged evaluation plan.
- `aggregateByGenome` - - Mutable aggregate cache keyed by genome.
- `provisionalScoresByGenome` - - Mutable provisional score map.

Returns: Nothing.

### evaluatePopulationReevaluationStage

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], generationEvaluationPlan: import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationEvaluationPlan, aggregateByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, provisionalScoresByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>, elitismCount: number) => void`

Executes the large-seed reevaluation stage over top candidates.

Parameters:

- `population` - - Current population.
- `generationEvaluationPlan` - - Per-generation staged evaluation plan.
- `aggregateByGenome` - - Mutable aggregate cache keyed by genome.
- `provisionalScoresByGenome` - - Mutable provisional score map.
- `elitismCount` - - Configured elitism count.

Returns: Nothing.

## trainer/trainer.report.service.services.ts

### collectFiniteGenomeScores

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[]) => number[]`

Collects only finite scores from the current population.

Unevaluated or invalid scores are intentionally skipped so percentile and
standard deviation calculations operate on stable numeric inputs only.

Parameters:

- `population` - - Current population.

Returns: Finite scores in population order.

### resolveBestGenerationDetails

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], bestGenome: import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork | undefined, aggregateByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, fallbackSeeds: readonly number[], fallbackRolloutOptions: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions) => import("test/examples/flappy_bird/trainer/trainer.report.service.services").ResolvedBestGenerationDetails`

Resolves cached or fallback best-of-generation details for reporting.

The report layer needs both aggregate seed statistics and one representative
episode. This helper centralizes the fallback rules so the service facade can
remain a thin orchestration layer.

Parameters:

- `population` - - Current population.
- `bestGenome` - - Genome selected as generation best.
- `aggregateByGenome` - - Cached aggregate evaluations keyed by genome.
- `fallbackSeeds` - - Seeds used when the aggregate must be recomputed.
- `fallbackRolloutOptions` - - Rollout options for fallback evaluation.

Returns: Aggregate metrics and a representative best-genome episode.

### ResolvedBestGenerationDetails

Aggregate and representative rollout resolved for the best genome.

Keeping these values together lets the report facade stay focused on
orchestration while this helper module owns cache fallback behavior.

## trainer/trainer.reporting.utils.ts

### buildGenerationLogParts

`(generationLabel: number, bestFitness: number, bestPipesPassed: number, bestFramesSurvived: number, report: import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationReport | undefined, mutationSchedule: import("test/examples/flappy_bird/trainer/trainer.evaluation-plan.utils").FlappyMutationSchedule) => string[]`

Builds one-line generation log tokens.

Parameters:

- `generationLabel` - - Generation label shown in logs.
- `bestFitness` - - Best resolved fitness value for this generation.
- `bestPipesPassed` - - Best resolved pipes passed value.
- `bestFramesSurvived` - - Best resolved frames survived value.
- `report` - - Optional aggregated generation report.
- `mutationSchedule` - - Active mutation schedule for this generation.

Returns: Ordered log tokens for compact console output.

## trainer/trainer.selection.utils.ts

### resolveBestGenomeByScore

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[]) => import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork | undefined`

Resolves the best genome by current score.

Parameters:

- `population` - - Current trainer population.

Returns: Highest-scoring genome or `undefined` when population is empty.

### selectTopGenomesByScore

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], provisionalScoresByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>, targetCount: number) => import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[]`

Returns top genomes ordered by current provisional score.

Parameters:

- `population` - - Current trainer population.
- `provisionalScoresByGenome` - - Optional map of staged provisional scores.
- `targetCount` - - Maximum number of genomes to return.

Returns: Highest-scoring genomes in descending score order.

## trainer/trainer.evaluation-plan.utils.ts

### buildSharedSeedBatch

`(generationIndex: number, stageSalt: number, seedCount: number) => number[]`

Build deterministic shared seeds for one generation stage.

Parameters:

- `generationIndex` - - Zero-based generation index.
- `stageSalt` - - Constant stage-specific salt.
- `seedCount` - - Number of seeds to produce.

Returns: Deterministic shared seed list.

### createFullRolloutOptions

`(difficultyScale: number) => import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions`

Builds full-stage rollout options.

Parameters:

- `difficultyScale` - - Difficulty scale for this generation.

Returns: Full stage rollout options.

### createQuickRolloutOptions

`(difficultyScale: number) => import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions`

Builds quick-screen rollout options.

Parameters:

- `difficultyScale` - - Difficulty scale for this generation.

Returns: Quick stage rollout options.

### createReevaluationRolloutOptions

`(difficultyScale: number) => import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions`

Builds high-confidence reevaluation rollout options.

Parameters:

- `difficultyScale` - - Difficulty scale for this generation.

Returns: Reevaluation stage rollout options.

### FlappyMutationSchedule

Mutation schedule used by generation planning and outer loop logging.

### mixSeed

`(generationIndex: number, stageSalt: number) => number`

Mixes generation and stage salts into a deterministic uint32 RNG seed.

Parameters:

- `generationIndex` - - Current generation index.
- `stageSalt` - - Stage-specific salt.

Returns: Mixed uint32 seed.

### resolveCurriculumDifficultyScale

`(generationIndex: number) => number`

Resolve curriculum difficulty scale for the current generation.

Parameters:

- `generationIndex` - - Zero-based generation index.

Returns: Difficulty scale in [0, 1].

### resolveGenerationEvaluationPlan

`(generationIndex: number) => import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationEvaluationPlan`

Resolves all per-generation evaluation controls.

Parameters:

- `generationIndex` - - Zero-based generation index.

Returns: Full staged evaluation plan for the generation.

### resolveMutationSchedule

`(generationIndex: number) => import("test/examples/flappy_bird/trainer/trainer.evaluation-plan.utils").FlappyMutationSchedule`

Resolve a smooth mutation annealing schedule.

Parameters:

- `generationIndex` - - Zero-based generation index.

Returns: Mutation rate and mutation amount for this generation.
