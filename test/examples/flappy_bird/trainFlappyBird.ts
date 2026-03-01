import Neat from '../../../src/neat.ts';
import Architect from '../../../src/architecture/architect.ts';
import * as methods from '../../../src/methods/methods.ts';
import {
  evaluateFlappyFitnessAcrossSeeds,
  rolloutEpisode,
  type FlappyNetworkLike,
  type FlappyRolloutOptions,
  type FlappySeedBatchEvaluation,
} from './flappyEvaluation';
import { clampValue, interpolateValue } from './flappy.simulation.shared.utils';
import {
  FLAPPY_MAX_FRAMES_PER_EPISODE,
  FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from './constants.ts';
import { createXorshift32 } from './rng';

/** Network shape expected by the Flappy trainer. */
interface FlappyTrainerNetwork extends FlappyNetworkLike {
  score?: number;
}

/** Local typed view for population-level fitness mode used by this trainer. */
interface FlappyTrainerNeatController {
  generation: number;
  options: {
    mutationRate: number;
    mutationAmount: number;
    fitnessPopulation?: boolean;
  };
  fitness: (population: FlappyTrainerNetwork[]) => Promise<void>;
  evolve(): Promise<FlappyTrainerNetwork>;
  restoreRNGState(seed: number): void;
}

/** Compact generation report used for training logs. */
interface FlappyGenerationReport {
  generationIndex: number;
  difficultyScale: number;
  mutationRate: number;
  mutationAmount: number;
  quickSeedCount: number;
  fullSeedCount: number;
  reevaluationSeedCount: number;
  evaluatedPopulationSize: number;
  scoreMean: number;
  scoreMedian: number;
  scoreP90: number;
  scoreStdDev: number;
  bestRobustFitness: number;
  bestMeanFitness: number;
  bestPipesPassed: number;
  bestFramesSurvived: number;
}

/** Trainer runtime state shared by orchestration helpers. */
interface FlappyTrainerRuntimeState {
  shouldStop: boolean;
  latestGenerationReport?: FlappyGenerationReport;
}

/** Immutable trainer setup values. */
interface FlappyTrainerSetup {
  inputSize: number;
  outputSize: number;
  populationSize: number;
  elitismCount: number;
}

/** Generation-level rollout plans for staged evaluation. */
interface FlappyGenerationEvaluationPlan {
  generationIndex: number;
  mutationRate: number;
  mutationAmount: number;
  difficultyScale: number;
  quickSeeds: number[];
  fullSeeds: number[];
  reevaluationSeeds: number[];
  quickRolloutOptions: FlappyRolloutOptions;
  fullRolloutOptions: FlappyRolloutOptions;
  reevaluationRolloutOptions: FlappyRolloutOptions;
}

/** Score carrier used for deterministic ordering helpers. */
interface ScoredGenomeEntry {
  genome: FlappyTrainerNetwork;
  score: number;
}

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
async function main(): Promise<void> {
  const trainerSetup = createTrainerSetup();
  const trainerRuntimeState = createTrainerRuntimeState();
  const neatController = createNeatController(trainerSetup);

  attachPopulationFitnessEvaluator(
    neatController,
    trainerRuntimeState,
    trainerSetup.elitismCount,
  );
  neatController.restoreRNGState(0x1234abcd);
  registerStopSignals(trainerRuntimeState);

  await runEvolutionLoop(neatController, trainerRuntimeState);

  // eslint-disable-next-line no-console
  console.log('Flappy training stopped gracefully.');
}

/**
 * Creates immutable setup values for the trainer.
 */
function createTrainerSetup(): FlappyTrainerSetup {
  return {
    inputSize: FLAPPY_NETWORK_INPUT_SIZE,
    outputSize: FLAPPY_NETWORK_OUTPUT_SIZE,
    populationSize: 200,
    elitismCount: 20,
  };
}

/**
 * Creates mutable runtime state container.
 */
function createTrainerRuntimeState(): FlappyTrainerRuntimeState {
  return {
    shouldStop: false,
    latestGenerationReport: undefined,
  };
}

/**
 * Builds the NEAT controller with baseline options.
 */
function createNeatController(
  trainerSetup: FlappyTrainerSetup,
): FlappyTrainerNeatController {
  const neatInstance = new Neat(
    trainerSetup.inputSize,
    trainerSetup.outputSize,
    resolveNoopFitness,
    {
      popsize: trainerSetup.populationSize,
      elitism: trainerSetup.elitismCount,
      mutationRate: 0.75,
      mutationAmount: 2,
      mutation: methods.mutation.FFW,
      network: Architect.perceptron(
        trainerSetup.inputSize,
        ...FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
        trainerSetup.outputSize,
      ),
      fitnessPopulation: true,
      speciation: true,
      multiObjective: { enabled: false },
      novelty: { enabled: false },
    },
  );

  return neatInstance as never as FlappyTrainerNeatController;
}

/**
 * Trivial baseline fitness used before attaching population evaluator.
 */
function resolveNoopFitness(): number {
  return 0;
}

/**
 * Attaches population-level staged evaluator to the NEAT controller.
 */
function attachPopulationFitnessEvaluator(
  neatController: FlappyTrainerNeatController,
  trainerRuntimeState: FlappyTrainerRuntimeState,
  elitismCount: number,
): void {
  neatController.fitness = createPopulationFitnessEvaluator(
    neatController,
    trainerRuntimeState,
    elitismCount,
  );
}

/**
 * Creates the asynchronous population fitness evaluator.
 */
function createPopulationFitnessEvaluator(
  neatController: FlappyTrainerNeatController,
  trainerRuntimeState: FlappyTrainerRuntimeState,
  elitismCount: number,
): (population: FlappyTrainerNetwork[]) => Promise<void> {
  async function evaluatePopulationFitness(
    population: FlappyTrainerNetwork[],
  ): Promise<void> {
    const generationEvaluationPlan = resolveGenerationEvaluationPlan(
      neatController.generation,
    );
    const provisionalScoresByGenome = new Map<FlappyTrainerNetwork, number>();
    const aggregateByGenome = new Map<
      FlappyTrainerNetwork,
      FlappySeedBatchEvaluation
    >();

    evaluatePopulationQuickStage(
      population,
      generationEvaluationPlan,
      aggregateByGenome,
      provisionalScoresByGenome,
    );

    evaluatePopulationFullStage(
      population,
      generationEvaluationPlan,
      aggregateByGenome,
      provisionalScoresByGenome,
      elitismCount,
    );

    evaluatePopulationReevaluationStage(
      population,
      generationEvaluationPlan,
      aggregateByGenome,
      provisionalScoresByGenome,
      elitismCount,
    );

    commitPopulationScores(population, provisionalScoresByGenome);

    trainerRuntimeState.latestGenerationReport = buildGenerationReport(
      population,
      aggregateByGenome,
      generationEvaluationPlan,
    );
  }

  return evaluatePopulationFitness;
}

/**
 * Resolves all per-generation evaluation controls.
 */
function resolveGenerationEvaluationPlan(
  generationIndex: number,
): FlappyGenerationEvaluationPlan {
  const mutationSchedule = resolveMutationSchedule(generationIndex);
  const difficultyScale = resolveCurriculumDifficultyScale(generationIndex);

  return {
    generationIndex,
    mutationRate: mutationSchedule.mutationRate,
    mutationAmount: mutationSchedule.mutationAmount,
    difficultyScale,
    quickSeeds: buildSharedSeedBatch(generationIndex, 0x41a7, 3),
    fullSeeds: buildSharedSeedBatch(generationIndex, 0x7d2b, 8),
    reevaluationSeeds: buildSharedSeedBatch(generationIndex, 0xb8f3, 32),
    quickRolloutOptions: createQuickRolloutOptions(difficultyScale),
    fullRolloutOptions: createFullRolloutOptions(difficultyScale),
    reevaluationRolloutOptions:
      createReevaluationRolloutOptions(difficultyScale),
  };
}

/**
 * Builds quick-screen rollout options.
 */
function createQuickRolloutOptions(
  difficultyScale: number,
): FlappyRolloutOptions {
  return {
    difficultyScale,
    maxFrames: 1_500,
    enableEarlyTermination: true,
    earlyTerminationGraceFrames: 120,
    earlyTerminationConsecutiveFrames: 18,
    normalizeFitness: true,
    pipeProgressTarget: 12,
  };
}

/**
 * Builds full-stage rollout options.
 */
function createFullRolloutOptions(
  difficultyScale: number,
): FlappyRolloutOptions {
  return {
    difficultyScale,
    maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    enableEarlyTermination: true,
    earlyTerminationGraceFrames: 220,
    earlyTerminationConsecutiveFrames: 28,
    normalizeFitness: true,
    pipeProgressTarget: 20,
  };
}

/**
 * Builds high-confidence reevaluation rollout options.
 */
function createReevaluationRolloutOptions(
  difficultyScale: number,
): FlappyRolloutOptions {
  return {
    difficultyScale,
    maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    enableEarlyTermination: false,
    normalizeFitness: true,
    pipeProgressTarget: 20,
  };
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
  const reevaluationCount = Math.max(elitismCount, 6);
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
    scoreMedian: computePercentile(finalScores, 0.5),
    scoreP90: computePercentile(finalScores, 0.9),
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
    _id: 0,
  };
}

/**
 * Dummy network activation used by fallback code paths.
 */
function activateWithoutFlap(): number[] {
  return [1, 0];
}

/**
 * Registers graceful stop signal handlers.
 */
function registerStopSignals(
  trainerRuntimeState: FlappyTrainerRuntimeState,
): void {
  process.on('SIGINT', function onSigInt(): void {
    handleStopSignal(trainerRuntimeState);
  });

  process.on('SIGTERM', function onSigTerm(): void {
    handleStopSignal(trainerRuntimeState);
  });
}

/**
 * Handles one stop signal update.
 */
function handleStopSignal(
  trainerRuntimeState: FlappyTrainerRuntimeState,
): void {
  trainerRuntimeState.shouldStop = true;
}

/**
 * Runs the outer evolution loop until stopped.
 */
async function runEvolutionLoop(
  neatController: FlappyTrainerNeatController,
  trainerRuntimeState: FlappyTrainerRuntimeState,
): Promise<void> {
  while (!trainerRuntimeState.shouldStop) {
    const mutationSchedule = resolveMutationSchedule(neatController.generation);
    applyMutationSchedule(neatController, mutationSchedule);

    const fittestGenome = await neatController.evolve();
    const fallbackEpisode = rolloutEpisode(fittestGenome, {
      normalizeFitness: true,
      maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    });

    logGenerationSummary(
      neatController.generation,
      mutationSchedule,
      trainerRuntimeState.latestGenerationReport,
      fittestGenome,
      fallbackEpisode,
    );
  }
}

/**
 * Applies mutation schedule to the NEAT controller options.
 */
function applyMutationSchedule(
  neatController: FlappyTrainerNeatController,
  mutationSchedule: { mutationRate: number; mutationAmount: number },
): void {
  neatController.options.mutationRate = mutationSchedule.mutationRate;
  neatController.options.mutationAmount = mutationSchedule.mutationAmount;
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
  console.log(logParts.join(' '));
}

/**
 * Builds one-line generation log tokens.
 */
function buildGenerationLogParts(
  generationLabel: number,
  bestFitness: number,
  bestPipesPassed: number,
  bestFramesSurvived: number,
  report: FlappyGenerationReport | undefined,
  mutationSchedule: { mutationRate: number; mutationAmount: number },
): string[] {
  return [
    `gen=${generationLabel}`,
    `best=${bestFitness.toFixed(2)}`,
    `mean=${(report?.scoreMean ?? Number.NaN).toFixed(2)}`,
    `median=${(report?.scoreMedian ?? Number.NaN).toFixed(2)}`,
    `p90=${(report?.scoreP90 ?? Number.NaN).toFixed(2)}`,
    `std=${(report?.scoreStdDev ?? Number.NaN).toFixed(2)}`,
    `pipes=${bestPipesPassed}`,
    `frames=${bestFramesSurvived}`,
    `difficulty=${(report?.difficultyScale ?? 1).toFixed(2)}`,
    `mutRate=${(report?.mutationRate ?? mutationSchedule.mutationRate).toFixed(3)}`,
    `mutAmount=${(report?.mutationAmount ?? mutationSchedule.mutationAmount).toFixed(2)}`,
    `seeds=${report?.quickSeedCount ?? 0}/${report?.fullSeedCount ?? 0}/${report?.reevaluationSeedCount ?? 0}`,
  ];
}

/**
 * Resolve a smooth mutation annealing schedule.
 *
 * @param generationIndex - Zero-based generation index.
 * @returns Mutation rate and mutation amount for this generation.
 */
function resolveMutationSchedule(generationIndex: number): {
  mutationRate: number;
  mutationAmount: number;
} {
  const annealProgress = clampValue(generationIndex / 120, 0, 1);
  return {
    mutationRate: interpolateValue(0.7, 0.25, annealProgress),
    mutationAmount: interpolateValue(2, 1, annealProgress),
  };
}

/**
 * Resolve curriculum difficulty scale for the current generation.
 *
 * @param generationIndex - Zero-based generation index.
 * @returns Difficulty scale in [0, 1].
 */
function resolveCurriculumDifficultyScale(generationIndex: number): number {
  if (generationIndex < 25) return 0;
  if (generationIndex >= 95) return 1;
  return (generationIndex - 25) / 70;
}

/**
 * Build deterministic shared seeds for one generation stage.
 *
 * @param generationIndex - Zero-based generation index.
 * @param stageSalt - Constant stage-specific salt.
 * @param seedCount - Number of seeds to produce.
 * @returns Deterministic shared seed list.
 */
function buildSharedSeedBatch(
  generationIndex: number,
  stageSalt: number,
  seedCount: number,
): number[] {
  const mixedSeed = mixSeed(generationIndex, stageSalt);
  const deterministicRandom = createXorshift32(mixedSeed);
  const sharedSeeds: number[] = [];

  let seedIndex = 0;
  while (seedIndex < seedCount) {
    sharedSeeds.push(deterministicRandom.nextInt(1, 0x7fffffff));
    seedIndex++;
  }

  return sharedSeeds;
}

/**
 * @param generationIndex - Current generation index.
 * @param stageSalt - Stage-specific salt.
 * @returns Mixed uint32 seed.
 */
function mixSeed(generationIndex: number, stageSalt: number): number {
  let seed = (generationIndex >>> 0) ^ (stageSalt >>> 0) ^ 0x9e3779b9;
  seed ^= seed >>> 16;
  seed = Math.imul(seed, 0x85ebca6b);
  seed ^= seed >>> 13;
  seed = Math.imul(seed, 0xc2b2ae35);
  seed ^= seed >>> 16;
  return seed >>> 0;
}

/**
 * @param values - Numeric samples.
 * @returns Arithmetic mean.
 */
function computeMean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return (
    values.reduce(function accumulate(sumValue, currentValue): number {
      return sumValue + currentValue;
    }, 0) / values.length
  );
}

/**
 * @param values - Numeric samples.
 * @param meanValue - Precomputed mean.
 * @returns Population standard deviation.
 */
function computePopulationStandardDeviation(
  values: readonly number[],
  meanValue: number,
): number {
  if (values.length === 0) return 0;

  const variance =
    values.reduce(function accumulateVariance(sumValue, currentValue): number {
      const deltaFromMean = currentValue - meanValue;
      return sumValue + deltaFromMean * deltaFromMean;
    }, 0) / values.length;

  return Math.sqrt(Math.max(0, variance));
}

/**
 * @param values - Numeric samples.
 * @param percentile - Percentile in [0, 1].
 * @returns Interpolated percentile value.
 */
function computePercentile(
  values: readonly number[],
  percentile: number,
): number {
  if (values.length === 0) return Number.NaN;

  const sortedValues = [...values];
  sortedValues.sort(compareNumbersAscending);

  const clampedPercentile = clampValue(percentile, 0, 1);
  const percentileIndex = clampedPercentile * (sortedValues.length - 1);
  const lowerIndex = Math.floor(percentileIndex);
  const upperIndex = Math.ceil(percentileIndex);
  const interpolationWeight = percentileIndex - lowerIndex;

  const lowerValue = sortedValues[lowerIndex] ?? sortedValues[0] ?? Number.NaN;
  const upperValue = sortedValues[upperIndex] ?? lowerValue;
  return lowerValue + (upperValue - lowerValue) * interpolationWeight;
}

/**
 * Numeric ascending comparator.
 */
function compareNumbersAscending(
  leftValue: number,
  rightValue: number,
): number {
  return leftValue - rightValue;
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
  const frameStabilityPenalty = aggregate.fitnessStdDev * 0.5;
  const passesPipeFilter =
    aggregate.meanPipesPassed >= maxMeanPipesPassed - 0.05;

  if (passesPipeFilter) {
    return (
      1_000_000 +
      aggregate.meanFramesSurvived * 100 +
      aggregate.meanPipesPassed * 10 -
      frameStabilityPenalty
    );
  }

  return (
    aggregate.meanPipesPassed * 10_000 +
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
  return Math.max(elitismCount * 3, Math.floor(populationSize * 0.3));
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
function handleMainError(error: unknown): void {
  // eslint-disable-next-line no-console
  console.error(error);
  process.exitCode = 1;
}

// Allow execution via `node`/`ts-node` without importing.
main().catch(handleMainError);
