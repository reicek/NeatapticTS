import { pathToFileURL } from 'node:url';
import {
  evaluateFlappyFitnessAcrossSeeds,
  rolloutEpisode,
  type FlappyRolloutOptions,
  type FlappySeedBatchEvaluation,
} from '../flappyEvaluation';
import type {
  FlappyGenerationEvaluationPlan,
  FlappyGenerationReport,
  FlappyTrainerNetwork,
  FlappyTrainerRuntimeState,
  FlappyTrainerSetup,
  ScoredGenomeEntry,
} from './trainer.types';
import {
  FLAPPY_TRAINER_DEFAULT_RNG_SEED,
  FLAPPY_TRAINER_DUMMY_FLAP_OUTPUT,
  FLAPPY_TRAINER_DUMMY_NETWORK_ID,
  FLAPPY_TRAINER_DUMMY_NO_FLAP_OUTPUT,
  FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE,
  FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT,
  FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT,
  FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT,
  FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER,
  FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION,
  FLAPPY_TRAINER_LOG_PARTS_DELIMITER,
  FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT,
  FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE,
  FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT,
  FLAPPY_TRAINER_SCORE_MEDIAN_PERCENTILE,
  FLAPPY_TRAINER_SCORE_P90_PERCENTILE,
  FLAPPY_TRAINER_STOPPED_MESSAGE,
} from './trainer.constants';
import { formatTrainerErrorMessage } from './trainer.errors';
import { resolveGenerationEvaluationPlan } from './trainer.evaluation-plan.utils';
import {
  createNeatController,
  createTrainerRuntimeState,
  createTrainerSetup,
} from './trainer.setup.service';
import { attachPopulationFitnessEvaluator } from './trainer.fitness.service';
import { buildGenerationLogParts } from './trainer.reporting.utils';
import { runTrainerEvolutionLoop } from './trainer.loop.service';
import { registerTrainerStopSignals } from './trainer.signals.service';
import {
  computeMean,
  computePercentile,
  computePopulationStandardDeviation,
} from './trainer.statistics.utils';

/**
 * Flappy Bird neuroevolution demo.
 *
 * This script runs a small NEAT population where each genome controls a bird.
 * The network sees a temporal observation (38 floats) and outputs two competing
 * action scores (`no flap` vs `flap`).
 *
 * Run (from repo root):
 * `npx ts-node test/examples/flappy_bird/trainFlappyBird.ts`
 */
export async function runTrainer(): Promise<void> {
  const trainerSetup = createTrainerSetup();
  const trainerRuntimeState = createTrainerRuntimeState();
  const neatController = createNeatController(trainerSetup);

  attachPopulationFitnessEvaluator(
    neatController,
    trainerRuntimeState,
    trainerSetup.elitismCount,
    {
      resolveGenerationEvaluationPlan,
      evaluatePopulationQuickStage,
      evaluatePopulationFullStage,
      evaluatePopulationReevaluationStage,
      commitPopulationScores,
      buildGenerationReport,
    },
  );
  neatController.restoreRNGState(FLAPPY_TRAINER_DEFAULT_RNG_SEED);
  registerTrainerStopSignals(trainerRuntimeState);

  await runTrainerEvolutionLoop(
    neatController,
    trainerRuntimeState,
    logGenerationSummary,
  );

  // eslint-disable-next-line no-console
  console.log(FLAPPY_TRAINER_STOPPED_MESSAGE);
}

/**
 * Executes quick evaluation stage over the full population.
 */
function evaluatePopulationQuickStage(
  population: readonly FlappyTrainerNetwork[],
  generationEvaluationPlan: FlappyGenerationEvaluationPlan,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
): void {
  evaluateSpecificGenomesAcrossSeeds(
    population,
    generationEvaluationPlan.quickSeeds,
    generationEvaluationPlan.quickRolloutOptions,
    aggregateByGenome,
  );

  assignFramePrimaryScores(
    population,
    aggregateByGenome,
    provisionalScoresByGenome,
  );
}

/**
 * Executes full evaluation stage over top provisional candidates.
 */
function evaluatePopulationFullStage(
  population: readonly FlappyTrainerNetwork[],
  generationEvaluationPlan: FlappyGenerationEvaluationPlan,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
  elitismCount: number,
): void {
  const fullPassCandidateCount = resolveFullPassCandidateCount(
    population.length,
    elitismCount,
  );
  const fullPassCandidates = selectTopGenomesByScore(
    population,
    provisionalScoresByGenome,
    fullPassCandidateCount,
  );

  evaluateSpecificGenomesAcrossSeeds(
    fullPassCandidates,
    generationEvaluationPlan.fullSeeds,
    generationEvaluationPlan.fullRolloutOptions,
    aggregateByGenome,
  );

  assignFramePrimaryScores(
    population,
    aggregateByGenome,
    provisionalScoresByGenome,
  );
}

/**
 * Executes large-seed reevaluation stage over top candidates.
 */
function evaluatePopulationReevaluationStage(
  population: readonly FlappyTrainerNetwork[],
  generationEvaluationPlan: FlappyGenerationEvaluationPlan,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
  elitismCount: number,
): void {
  const reevaluationCount = Math.max(
    elitismCount,
    FLAPPY_TRAINER_REEVALUATION_MIN_CANDIDATE_COUNT,
  );
  const reevaluationCandidates = selectTopGenomesByScore(
    population,
    provisionalScoresByGenome,
    reevaluationCount,
  );

  evaluateSpecificGenomesAcrossSeeds(
    reevaluationCandidates,
    generationEvaluationPlan.reevaluationSeeds,
    generationEvaluationPlan.reevaluationRolloutOptions,
    aggregateByGenome,
  );

  assignFramePrimaryScores(
    population,
    aggregateByGenome,
    provisionalScoresByGenome,
  );
}

/**
 * Evaluates a specific genome set across a shared seed batch.
 */
function evaluateSpecificGenomesAcrossSeeds(
  genomes: readonly FlappyTrainerNetwork[],
  sharedSeeds: readonly number[],
  rolloutOptions: FlappyRolloutOptions,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
): void {
  for (const genome of genomes) {
    const aggregate = evaluateFlappyFitnessAcrossSeeds(
      genome,
      sharedSeeds,
      rolloutOptions,
    );
    aggregateByGenome.set(genome, aggregate);
  }
}

/**
 * Commits provisional scores to genome `.score` fields.
 */
function commitPopulationScores(
  population: readonly FlappyTrainerNetwork[],
  provisionalScoresByGenome: ReadonlyMap<FlappyTrainerNetwork, number>,
): void {
  for (const genome of population) {
    genome.score =
      provisionalScoresByGenome.get(genome) ?? Number.NEGATIVE_INFINITY;
  }
}

/**
 * Builds compact report for current generation.
 */
function buildGenerationReport(
  population: readonly FlappyTrainerNetwork[],
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
  generationEvaluationPlan: FlappyGenerationEvaluationPlan,
): FlappyGenerationReport {
  const finalScores = collectFiniteGenomeScores(population);
  const scoreMean = computeMean(finalScores);
  const scoreStdDev = computePopulationStandardDeviation(
    finalScores,
    scoreMean,
  );
  const bestGenome = resolveBestGenomeByScore(population);
  const bestAggregate = resolveBestAggregate(
    population,
    bestGenome,
    aggregateByGenome,
    generationEvaluationPlan.reevaluationSeeds,
    generationEvaluationPlan.reevaluationRolloutOptions,
  );
  const bestEpisode = resolveBestEpisode(
    population,
    bestGenome,
    generationEvaluationPlan.reevaluationSeeds,
    generationEvaluationPlan.reevaluationRolloutOptions,
  );

  return {
    generationIndex: generationEvaluationPlan.generationIndex,
    difficultyScale: generationEvaluationPlan.difficultyScale,
    mutationRate: generationEvaluationPlan.mutationRate,
    mutationAmount: generationEvaluationPlan.mutationAmount,
    quickSeedCount: generationEvaluationPlan.quickSeeds.length,
    fullSeedCount: generationEvaluationPlan.fullSeeds.length,
    reevaluationSeedCount: generationEvaluationPlan.reevaluationSeeds.length,
    evaluatedPopulationSize: population.length,
    scoreMean,
    scoreMedian: computePercentile(
      finalScores,
      FLAPPY_TRAINER_SCORE_MEDIAN_PERCENTILE,
    ),
    scoreP90: computePercentile(
      finalScores,
      FLAPPY_TRAINER_SCORE_P90_PERCENTILE,
    ),
    scoreStdDev,
    bestRobustFitness: bestGenome?.score ?? Number.NaN,
    bestMeanFitness: bestAggregate.meanFitness,
    bestPipesPassed: bestEpisode.pipesPassed,
    bestFramesSurvived: bestEpisode.framesSurvived,
  };
}

/**
 * Collects finite genome scores from population.
 */
function collectFiniteGenomeScores(
  population: readonly FlappyTrainerNetwork[],
): number[] {
  const finiteScores: number[] = [];

  for (const genome of population) {
    const genomeScore = genome.score ?? Number.NEGATIVE_INFINITY;
    if (Number.isFinite(genomeScore)) {
      finiteScores.push(genomeScore);
    }
  }

  return finiteScores;
}

/**
 * Resolves the best genome by current score.
 */
function resolveBestGenomeByScore(
  population: readonly FlappyTrainerNetwork[],
): FlappyTrainerNetwork | undefined {
  const scoredGenomes = buildScoredGenomeEntries(population, undefined);
  scoredGenomes.sort(compareScoredGenomeEntriesDescending);
  return scoredGenomes[0]?.genome;
}

/**
 * Resolves aggregate for the selected best genome.
 */
function resolveBestAggregate(
  population: readonly FlappyTrainerNetwork[],
  bestGenome: FlappyTrainerNetwork | undefined,
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
  fallbackSeeds: readonly number[],
  fallbackRolloutOptions: FlappyRolloutOptions,
): FlappySeedBatchEvaluation {
  const fallbackGenome = resolveFallbackGenome(population, bestGenome);
  if (!fallbackGenome) {
    return buildEmptySeedBatchEvaluation();
  }

  const cachedAggregate = aggregateByGenome.get(fallbackGenome);
  if (cachedAggregate) {
    return cachedAggregate;
  }

  return evaluateFlappyFitnessAcrossSeeds(
    fallbackGenome,
    fallbackSeeds,
    fallbackRolloutOptions,
  );
}

/**
 * Resolves one representative episode for the selected best genome.
 */
function resolveBestEpisode(
  population: readonly FlappyTrainerNetwork[],
  bestGenome: FlappyTrainerNetwork | undefined,
  fallbackSeeds: readonly number[],
  fallbackRolloutOptions: FlappyRolloutOptions,
): ReturnType<typeof rolloutEpisode> {
  const fallbackGenome = resolveFallbackGenome(population, bestGenome);
  if (!fallbackGenome) {
    return rolloutEpisode(resolveDummyNetwork(), fallbackRolloutOptions);
  }

  const episodeSeed = fallbackSeeds[0];
  return rolloutEpisode(fallbackGenome, {
    ...fallbackRolloutOptions,
    seed: episodeSeed,
  });
}

/**
 * Picks the best genome when available, else falls back to the first genome.
 */
function resolveFallbackGenome(
  population: readonly FlappyTrainerNetwork[],
  bestGenome: FlappyTrainerNetwork | undefined,
): FlappyTrainerNetwork | undefined {
  return bestGenome ?? population[0];
}

/**
 * Builds empty aggregate object for defensive fallback paths.
 */
function buildEmptySeedBatchEvaluation(): FlappySeedBatchEvaluation {
  return {
    seedCount: 0,
    meanFitness: 0,
    medianFitness: 0,
    p90Fitness: 0,
    fitnessStdDev: 0,
    robustFitness: 0,
    meanPipesPassed: 0,
    meanFramesSurvived: 0,
  };
}

/**
 * Builds a minimal dummy network used only by fallback episode path.
 */
function resolveDummyNetwork(): FlappyTrainerNetwork {
  return {
    activate: activateWithoutFlap,
    _id: FLAPPY_TRAINER_DUMMY_NETWORK_ID,
  };
}

/**
 * Dummy network activation used by fallback code paths.
 */
function activateWithoutFlap(): number[] {
  return [
    FLAPPY_TRAINER_DUMMY_NO_FLAP_OUTPUT,
    FLAPPY_TRAINER_DUMMY_FLAP_OUTPUT,
  ];
}

/**
 * Emits one compact generation log line.
 */
function logGenerationSummary(
  generationLabel: number,
  mutationSchedule: { mutationRate: number; mutationAmount: number },
  report: FlappyGenerationReport | undefined,
  fittestGenome: FlappyTrainerNetwork,
  fallbackEpisode: ReturnType<typeof rolloutEpisode>,
): void {
  const bestFitness =
    report?.bestRobustFitness ??
    (fittestGenome.score as number) ??
    fallbackEpisode.fitness;
  const bestPipesPassed =
    report?.bestPipesPassed ?? fallbackEpisode.pipesPassed;
  const bestFramesSurvived =
    report?.bestFramesSurvived ?? fallbackEpisode.framesSurvived;

  const logParts = buildGenerationLogParts(
    generationLabel,
    bestFitness,
    bestPipesPassed,
    bestFramesSurvived,
    report,
    mutationSchedule,
  );

  // eslint-disable-next-line no-console
  console.log(logParts.join(FLAPPY_TRAINER_LOG_PARTS_DELIMITER));
}

/**
 * Rebuild provisional scores using frame-primary ranking with a pipe filter.
 */
function assignFramePrimaryScores(
  population: readonly FlappyTrainerNetwork[],
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
): void {
  const aggregateValues = collectAggregateValues(population, aggregateByGenome);
  const maxMeanPipesPassed = resolveMaxMeanPipesPassed(aggregateValues);

  for (const genome of population) {
    const aggregate = aggregateByGenome.get(genome);
    if (!aggregate) {
      provisionalScoresByGenome.set(genome, Number.NEGATIVE_INFINITY);
      continue;
    }

    provisionalScoresByGenome.set(
      genome,
      scoreAggregateFramePrimary(aggregate, maxMeanPipesPassed),
    );
  }
}

/**
 * Collects present aggregate values in population order.
 */
function collectAggregateValues(
  population: readonly FlappyTrainerNetwork[],
  aggregateByGenome: ReadonlyMap<
    FlappyTrainerNetwork,
    FlappySeedBatchEvaluation
  >,
): FlappySeedBatchEvaluation[] {
  const aggregateValues: FlappySeedBatchEvaluation[] = [];

  for (const genome of population) {
    const aggregate = aggregateByGenome.get(genome);
    if (aggregate) {
      aggregateValues.push(aggregate);
    }
  }

  return aggregateValues;
}

/**
 * Resolves best mean pipe progress across aggregates.
 */
function resolveMaxMeanPipesPassed(
  aggregateValues: readonly FlappySeedBatchEvaluation[],
): number {
  return aggregateValues.reduce(function resolveBest(
    bestPipeProgress,
    aggregate,
  ): number {
    return Math.max(bestPipeProgress, aggregate.meanPipesPassed);
  }, 0);
}

/**
 * Score one aggregate with pipes as filter and frames as primary objective.
 */
function scoreAggregateFramePrimary(
  aggregate: FlappySeedBatchEvaluation,
  maxMeanPipesPassed: number,
): number {
  const frameStabilityPenalty =
    aggregate.fitnessStdDev * FLAPPY_TRAINER_FRAME_STABILITY_STDDEV_WEIGHT;
  const passesPipeFilter =
    aggregate.meanPipesPassed >=
    maxMeanPipesPassed - FLAPPY_TRAINER_PIPE_FILTER_TOLERANCE;

  if (passesPipeFilter) {
    return (
      FLAPPY_TRAINER_FRAME_PRIMARY_BASE_SCORE +
      aggregate.meanFramesSurvived *
        FLAPPY_TRAINER_FRAME_PRIMARY_SURVIVAL_WEIGHT +
      aggregate.meanPipesPassed * FLAPPY_TRAINER_FRAME_PRIMARY_PIPE_WEIGHT -
      frameStabilityPenalty
    );
  }

  return (
    aggregate.meanPipesPassed * FLAPPY_TRAINER_PIPE_FALLBACK_PIPE_WEIGHT +
    aggregate.meanFramesSurvived -
    frameStabilityPenalty
  );
}

/**
 * Resolves full-stage candidate count from population and elitism.
 */
function resolveFullPassCandidateCount(
  populationSize: number,
  elitismCount: number,
): number {
  return Math.max(
    elitismCount * FLAPPY_TRAINER_FULL_PASS_ELITISM_MULTIPLIER,
    Math.floor(populationSize * FLAPPY_TRAINER_FULL_PASS_POPULATION_FRACTION),
  );
}

/**
 * Returns top genomes ordered by current provisional score.
 */
function selectTopGenomesByScore(
  population: readonly FlappyTrainerNetwork[],
  provisionalScoresByGenome: ReadonlyMap<FlappyTrainerNetwork, number>,
  targetCount: number,
): FlappyTrainerNetwork[] {
  const scoredEntries = buildScoredGenomeEntries(
    population,
    provisionalScoresByGenome,
  );

  scoredEntries.sort(compareScoredGenomeEntriesDescending);

  const selectedGenomes: FlappyTrainerNetwork[] = [];
  const maxSelectionCount = Math.max(
    0,
    Math.min(targetCount, scoredEntries.length),
  );

  let selectedIndex = 0;
  while (selectedIndex < maxSelectionCount) {
    const scoredEntry = scoredEntries[selectedIndex];
    if (scoredEntry) {
      selectedGenomes.push(scoredEntry.genome);
    }
    selectedIndex++;
  }

  return selectedGenomes;
}

/**
 * Builds sortable scored entries from genomes.
 */
function buildScoredGenomeEntries(
  population: readonly FlappyTrainerNetwork[],
  provisionalScoresByGenome:
    | ReadonlyMap<FlappyTrainerNetwork, number>
    | undefined,
): ScoredGenomeEntry[] {
  const scoredEntries: ScoredGenomeEntry[] = [];

  for (const genome of population) {
    const score = resolveGenomeScore(genome, provisionalScoresByGenome);
    scoredEntries.push({ genome, score });
  }

  return scoredEntries;
}

/**
 * Resolves genome score either from map or direct genome field.
 */
function resolveGenomeScore(
  genome: FlappyTrainerNetwork,
  provisionalScoresByGenome:
    | ReadonlyMap<FlappyTrainerNetwork, number>
    | undefined,
): number {
  if (provisionalScoresByGenome) {
    return provisionalScoresByGenome.get(genome) ?? Number.NEGATIVE_INFINITY;
  }
  return genome.score ?? Number.NEGATIVE_INFINITY;
}

/**
 * Sort comparator for scored genome entries (descending by score).
 */
function compareScoredGenomeEntriesDescending(
  leftEntry: ScoredGenomeEntry,
  rightEntry: ScoredGenomeEntry,
): number {
  return rightEntry.score - leftEntry.score;
}

/**
 * Handles fatal `main` rejection path.
 */
export function handleTrainerMainError(error: unknown): void {
  // eslint-disable-next-line no-console
  console.error(formatTrainerErrorMessage(error));
  process.exitCode = 1;
}

if (isDirectTrainerExecution()) {
  runTrainer().catch(handleTrainerMainError);
}

function isDirectTrainerExecution(): boolean {
  const entryScriptPath = process.argv[1];
  if (!entryScriptPath) {
    return false;
  }
  return import.meta.url === pathToFileURL(entryScriptPath).href;
}
