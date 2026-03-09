# trainer/evaluation

## trainer/evaluation/trainer.evaluation.service.types.ts

### PopulationAggregateScoringContext

Aggregate scoring context shared while computing frame-primary scores.

### PopulationStageEvaluationRequest

Candidate-stage request used by the staged population evaluator.

Keeping this internal contract narrow lets the orchestration service choose
a candidate budget without coupling the execution helpers to generation-plan
details.

## trainer/evaluation/trainer.evaluation.service.ts

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

### resolveFullPassCandidateCount

`(populationSize: number, elitismCount: number) => number`

Resolves how many genomes should advance to the full-pass stage.

Parameters:
- `populationSize` - - Population size.
- `elitismCount` - - Configured elitism count.

Returns: Full-pass candidate count.

## trainer/evaluation/trainer.evaluation.service.services.ts

### evaluatePopulationSelectedCandidateStage

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], populationStageEvaluationRequest: import("test/examples/flappy_bird/trainer/evaluation/trainer.evaluation.service.types").PopulationStageEvaluationRequest, aggregateByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, provisionalScoresByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>) => void`

Evaluates a selected candidate subset for a population stage.

Parameters:
- `population` - - Current population.
- `populationStageEvaluationRequest` - - Candidate-stage evaluation request.
- `aggregateByGenome` - - Mutable aggregate cache keyed by genome.
- `provisionalScoresByGenome` - - Mutable provisional score map.

Returns: Nothing.

### evaluateSpecificGenomesAcrossSeeds

`(genomes: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], sharedSeeds: readonly number[], rolloutOptions: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions, aggregateByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>) => void`

Evaluates a specific genome subset across shared seeds.

Parameters:
- `genomes` - - Genomes selected for evaluation.
- `sharedSeeds` - - Shared deterministic seeds.
- `rolloutOptions` - - Rollout options for this stage.
- `aggregateByGenome` - - Mutable aggregate cache keyed by genome.

Returns: Nothing.

## trainer/evaluation/trainer.evaluation.service.constants.ts

### trainer.evaluation.service.constants

Fallback score assigned to genomes that have not yet been evaluated.

### FLAPPY_TRAINER_MIN_PIPE_PROGRESS

### FLAPPY_TRAINER_NEGATIVE_INFINITY_SCORE

## trainer/evaluation/trainer.evaluation.service.utils.ts

### assignFramePrimaryScores

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], aggregateByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, provisionalScoresByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>) => void`

Assigns refreshed frame-primary scores to the current population.

Parameters:
- `population` - - Current population.
- `aggregateByGenome` - - Aggregate cache keyed by genome.
- `provisionalScoresByGenome` - - Mutable provisional score map.

Returns: Nothing.

### collectAggregateValues

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], aggregateByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>) => import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation[]`

Collects all currently available aggregate values.

Parameters:
- `population` - - Current population.
- `aggregateByGenome` - - Aggregate cache keyed by genome.

Returns: Collected aggregate values.

### resolveMaximumMeanPipesPassed

`(aggregateValues: readonly import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation[]) => number`

Resolves the leading mean pipe-progress value across available aggregates.

Parameters:
- `aggregateValues` - - Aggregate values currently available.

Returns: Highest mean pipe-progress value.

### resolvePopulationAggregateScoringContext

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], aggregateByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>) => import("test/examples/flappy_bird/trainer/evaluation/trainer.evaluation.service.types").PopulationAggregateScoringContext`

Resolves the aggregate scoring context used by frame-primary scoring.

Parameters:
- `population` - - Current population.
- `aggregateByGenome` - - Aggregate cache keyed by genome.

Returns: Aggregate scoring context.

### scoreAggregateFramePrimary

`(aggregate: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation, maximumMeanPipesPassed: number) => number`

Scores one aggregate using the frame-primary heuristic.

Parameters:
- `aggregate` - - Aggregate evaluation result.
- `maximumMeanPipesPassed` - - Best mean pipe progress in the population.

Returns: Provisional score.
