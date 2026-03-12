# trainer/evaluation

## trainer/evaluation/trainer.evaluation.service.types.ts

### PopulationAggregateScoringContext

Aggregate scoring context shared while computing frame-primary scores.

The context precomputes population-wide reference values so per-genome scoring
can stay simple and deterministic.

### PopulationStageEvaluationRequest

Candidate-stage request used by the staged population evaluator.

Keeping this internal contract narrow lets the orchestration service choose
a candidate budget without coupling the execution helpers to generation-plan
details.

This shape is intentionally stage-agnostic: quick, full, and reevaluation can
all use the same execution helper by changing only candidate count, seed set,
and rollout budget.

## trainer/evaluation/trainer.evaluation.service.ts

### commitPopulationScores

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], provisionalScoresByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>) => void`

Commits provisional scores to genome score fields.

Provisional scores are kept in a map during staging so each phase can refresh
them without mutating the genomes too early. This helper performs the final
write-back once staged evaluation is complete.

Parameters:
- `population` - - Current population.
- `provisionalScoresByGenome` - - Final provisional score map.

Returns: Nothing.

### evaluatePopulationFullStage

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], generationEvaluationPlan: import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationEvaluationPlan, aggregateByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, provisionalScoresByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>, elitismCount: number) => void`

Executes the full evaluation stage over the top provisional candidates.

This is the middle-cost stage in the ranking ladder: not every genome
survives into it, but the survivors receive a more trustworthy estimate than
the quick screen alone can provide.

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

Educational note:
The quick stage is a cheap screening pass. Every genome is tested on the same
small shared seed batch so the trainer can discard obviously weak candidates
before spending more rollout budget on them.

Parameters:
- `population` - - Current population.
- `generationEvaluationPlan` - - Per-generation staged evaluation plan.
- `aggregateByGenome` - - Mutable aggregate cache keyed by genome.
- `provisionalScoresByGenome` - - Mutable provisional score map.

Returns: Nothing.

### evaluatePopulationReevaluationStage

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], generationEvaluationPlan: import("test/examples/flappy_bird/trainer/trainer.types").FlappyGenerationEvaluationPlan, aggregateByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, provisionalScoresByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>, elitismCount: number) => void`

Executes the large-seed reevaluation stage over top candidates.

Educational note:
Reevaluation is the trainer's anti-luck pass. The best provisional genomes
are tested again on a larger shared seed batch so leaderboard positions are
less sensitive to a fortunate early sample.

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

The trainer uses the larger of two budgets so the full stage stays large
enough to preserve competitive diversity while still shrinking meaningfully
relative to the full population.

Parameters:
- `populationSize` - - Population size.
- `elitismCount` - - Configured elitism count.

Returns: Full-pass candidate count.

## trainer/evaluation/trainer.evaluation.service.services.ts

### evaluatePopulationSelectedCandidateStage

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], populationStageEvaluationRequest: import("test/examples/flappy_bird/trainer/evaluation/trainer.evaluation.service.types").PopulationStageEvaluationRequest, aggregateByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, provisionalScoresByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>) => void`

Evaluates a selected candidate subset for a population stage.

Educational note:
This helper is the workhorse behind the full and reevaluation stages. It
turns a stage request into three steps: pick candidates, evaluate them across
shared seeds, then refresh the provisional ranking for the whole population.

Parameters:
- `population` - - Current population.
- `populationStageEvaluationRequest` - - Candidate-stage evaluation request.
- `aggregateByGenome` - - Mutable aggregate cache keyed by genome.
- `provisionalScoresByGenome` - - Mutable provisional score map.

Returns: Nothing.

### evaluateSpecificGenomesAcrossSeeds

`(genomes: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], sharedSeeds: readonly number[], rolloutOptions: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappyRolloutOptions, aggregateByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>) => void`

Evaluates a specific genome subset across shared seeds.

Shared seeds are the fairness mechanism in this trainer. Every selected genome
sees the same randomized episode batch for the stage, so comparisons are much
less noisy than per-genome private seed sampling.

For background reading, the Wikipedia article on "control variates" is a good
intuition pump for why holding part of the randomness fixed can reduce
variance when comparing alternatives.

Parameters:
- `genomes` - - Genomes selected for evaluation.
- `sharedSeeds` - - Shared deterministic seeds.
- `rolloutOptions` - - Rollout options for this stage.
- `aggregateByGenome` - - Mutable aggregate cache keyed by genome.

Returns: Nothing.

## trainer/evaluation/trainer.evaluation.service.constants.ts

### trainer.evaluation.service.constants

Fallback score assigned to genomes that have not yet been evaluated.

Using negative infinity guarantees unevaluated genomes lose any ranking tie
against genomes that already have real aggregate results.

### FLAPPY_TRAINER_MIN_PIPE_PROGRESS

### FLAPPY_TRAINER_NEGATIVE_INFINITY_SCORE

## trainer/evaluation/trainer.evaluation.service.utils.ts

### assignFramePrimaryScores

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], aggregateByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>, provisionalScoresByGenome: Map<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, number>) => void`

Assigns refreshed frame-primary scores to the current population.

Educational note:
The trainer does not rank genomes purely by one raw metric. It combines pipe
progress, survival, and stability into a provisional score so early-stage
selection remains robust when several genomes are close in quality.

Parameters:
- `population` - - Current population.
- `aggregateByGenome` - - Aggregate cache keyed by genome.
- `provisionalScoresByGenome` - - Mutable provisional score map.

Returns: Nothing.

### collectAggregateValues

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], aggregateByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>) => import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation[]`

Collects all currently available aggregate values.

Only genomes with completed aggregate results are included. That lets the
scoring helpers distinguish between genuinely weak genomes and genomes that
simply have not yet reached a later stage.

Parameters:
- `population` - - Current population.
- `aggregateByGenome` - - Aggregate cache keyed by genome.

Returns: Collected aggregate values.

### resolveMaximumMeanPipesPassed

`(aggregateValues: readonly import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation[]) => number`

Resolves the leading mean pipe-progress value across available aggregates.

Mean pipe progress acts as the leading indicator for the frame-primary score:
if a genome is far behind the current pipe leader, it falls back to a simpler
progress-first score.

Parameters:
- `aggregateValues` - - Aggregate values currently available.

Returns: Highest mean pipe-progress value.

### resolvePopulationAggregateScoringContext

`(population: readonly import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork[], aggregateByGenome: ReadonlyMap<import("test/examples/flappy_bird/trainer/trainer.types").FlappyTrainerNetwork, import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation>) => import("test/examples/flappy_bird/trainer/evaluation/trainer.evaluation.service.types").PopulationAggregateScoringContext`

Resolves the aggregate scoring context used by frame-primary scoring.

This precomputation step keeps the per-genome scoring loop lean and avoids
recomputing population-wide maxima for every genome.

Parameters:
- `population` - - Current population.
- `aggregateByGenome` - - Aggregate cache keyed by genome.

Returns: Aggregate scoring context.

### scoreAggregateFramePrimary

`(aggregate: import("test/examples/flappy_bird/evaluation/evaluation.types").FlappySeedBatchEvaluation, maximumMeanPipesPassed: number) => number`

Scores one aggregate using the frame-primary heuristic.

Educational note:
The heuristic intentionally mixes progress and stability. A genome that passes
many pipes but has wildly inconsistent fitness across seeds is treated more
cautiously than a similarly strong but steadier genome.

Parameters:
- `aggregate` - - Aggregate evaluation result.
- `maximumMeanPipesPassed` - - Best mean pipe progress in the population.

Returns: Provisional score.
