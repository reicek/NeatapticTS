import {
  evaluateFlappyFitnessAcrossSeeds,
  type FlappyRolloutOptions,
  type FlappySeedBatchEvaluation,
} from '../../flappyEvaluation';
import { evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker } from '../../evaluation/evaluation.fitness.utils';
import { createNeatParallelPopulationEvaluator } from '../../../../src/neataptic';
import type { SharedInferenceWorker } from '../../../../src/neataptic';
import { selectTopGenomesByScore } from '../trainer.selection.utils';
import type { FlappyTrainerNetwork } from '../trainer.types';
import { assignFramePrimaryScores } from './trainer.evaluation.service.utils';
import type {
  PopulationStageEvaluationDependencies,
  PopulationStageEvaluationRequest,
} from './trainer.evaluation.service.types';

/**
 * Evaluates a selected candidate subset for a population stage.
 *
 * Educational note:
 * This helper is the workhorse behind the full and reevaluation stages. It
 * turns a stage request into three steps: pick candidates, evaluate them across
 * shared seeds, then refresh the provisional ranking for the whole population.
 *
 * @param population - Current population.
 * @param populationStageEvaluationRequest - Candidate-stage evaluation request.
 * @param aggregateByGenome - Mutable aggregate cache keyed by genome.
 * @param provisionalScoresByGenome - Mutable provisional score map.
 * @returns Nothing.
 */
export async function evaluatePopulationSelectedCandidateStage(
  population: readonly FlappyTrainerNetwork[],
  populationStageEvaluationRequest: PopulationStageEvaluationRequest,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
  provisionalScoresByGenome: Map<FlappyTrainerNetwork, number>,
  populationStageEvaluationDependencies: PopulationStageEvaluationDependencies = {},
): Promise<void> {
  // Step 1: Select the top-scoring candidates for the requested stage.
  const selectedCandidates = selectTopGenomesByScore(
    population,
    provisionalScoresByGenome,
    populationStageEvaluationRequest.candidateCount,
  );

  // Step 2: Evaluate the selected candidates across the stage's shared seeds.
  await evaluateSpecificGenomesAcrossSeeds(
    selectedCandidates,
    populationStageEvaluationRequest.sharedSeeds,
    populationStageEvaluationRequest.rolloutOptions,
    aggregateByGenome,
    populationStageEvaluationDependencies,
  );

  // Step 3: Refresh frame-primary scores for the full population.
  assignFramePrimaryScores(
    population,
    aggregateByGenome,
    provisionalScoresByGenome,
  );
}

/**
 * Evaluates a specific genome subset across shared seeds.
 *
 * Shared seeds are the fairness mechanism in this trainer. Every selected genome
 * sees the same randomized episode batch for the stage, so comparisons are much
 * less noisy than per-genome private seed sampling.
 *
 * For background reading, the Wikipedia article on "control variates" is a good
 * intuition pump for why holding part of the randomness fixed can reduce
 * variance when comparing alternatives.
 *
 * Educational note:
 * When a worker pool is available, this helper now routes through the public
 * `createNeatParallelPopulationEvaluator(...)` factory rather than owning a
 * demo-local batch loop. Flappy still prepares the rollout payload shelf and
 * decides when the staged trainer should opt into workers, but ordered scoring,
 * fallback behavior, and population-level assignment now flow through the same
 * reusable library helper that other NEAT callers can adopt.
 *
 * @param genomes - Genomes selected for evaluation.
 * @param sharedSeeds - Shared deterministic seeds.
 * @param rolloutOptions - Rollout options for this stage.
 * @param aggregateByGenome - Mutable aggregate cache keyed by genome.
 * @returns Nothing.
 */
export async function evaluateSpecificGenomesAcrossSeeds(
  genomes: readonly FlappyTrainerNetwork[],
  sharedSeeds: readonly number[],
  rolloutOptions: FlappyRolloutOptions,
  aggregateByGenome: Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>,
  populationStageEvaluationDependencies: PopulationStageEvaluationDependencies = {},
): Promise<void> {
  const workerPool = populationStageEvaluationDependencies.workerPool;

  // Step 1: Resolve the optional ordered worker payload shelf for this batch.
  const orderedPayloads = workerPool
    ? await workerPool.resolveOrderedPayloads(genomes as never)
    : undefined;

  // Step 2: Route through the boolean-first NEAT helper, which still keeps the worker overrides explicit.
  const evaluatePopulation = createNeatParallelPopulationEvaluator({
    parallel: Boolean(workerPool),
    evaluateGenome: async (genome: FlappyTrainerNetwork) =>
      evaluateFlappyFitnessAcrossSeeds(genome, sharedSeeds, rolloutOptions),
    evaluateWithWorker: workerPool
      ? async (
          sharedWorker: SharedInferenceWorker,
          genome: FlappyTrainerNetwork,
        ) => {
          return orderedPayloads
            ? evaluateWorkerAggregate(
                sharedWorker,
                genome,
                sharedSeeds,
                rolloutOptions,
              )
            : evaluateFlappyFitnessAcrossSeeds(
                genome,
                sharedSeeds,
                rolloutOptions,
              );
        }
      : undefined,
    resolvePayload: orderedPayloads
      ? (_genome: FlappyTrainerNetwork, genomeIndex: number) =>
          resolveRequiredBatchPayload(orderedPayloads, genomeIndex)
      : undefined,
    workerPool: workerPool?.parallelWorkerPool,
    assignResult: (
      genome: FlappyTrainerNetwork,
      aggregateEvaluation: FlappySeedBatchEvaluation,
    ) => {
      aggregateByGenome.set(genome, aggregateEvaluation);
    },
  });

  // Step 3: Copy ordered helper results back into the aggregate cache.
  await evaluatePopulation([...genomes]);
}

async function evaluateWorkerAggregate(
  sharedWorker: SharedInferenceWorker,
  genome: FlappyTrainerNetwork,
  sharedSeeds: readonly number[],
  rolloutOptions: FlappyRolloutOptions,
): Promise<FlappySeedBatchEvaluation> {
  return evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker(
    sharedWorker,
    sharedSeeds,
    {
      networkId: typeof genome._id === 'number' ? genome._id : undefined,
      rolloutOptions,
    },
  );
}

function resolveRequiredBatchPayload<TPayload>(
  orderedPayloads: readonly TPayload[],
  genomeIndex: number,
): TPayload {
  const payload = orderedPayloads[genomeIndex];

  if (!payload) {
    throw new Error(
      'evaluateSpecificGenomesAcrossSeeds did not resolve every queued genome payload.',
    );
  }

  return payload;
}
