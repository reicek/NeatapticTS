import type { Neat } from '../../../src/neataptic';
import type Network from '../../../src/architecture/network';
import { createXorshift32 } from '../rng';
import {
  evaluateFlappyFitnessAcrossSeeds,
  type FlappySeedBatchEvaluation,
} from '../flappyEvaluation';
import { resolveObservationVector } from '../browser-entry/browser-entry.observation.utils';
import type {
  WorkerHeuristicObservationFeatures,
  WorkerPopulationPipe,
} from './flappy-evolution-worker.types';
import {
  FLAPPY_BIRD_X_PX,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_WORLD_HEIGHT_PX,
} from '../constants/constants';
import {
  FLAPPY_WORKER_GEN0_PRETRAIN_BATCH_SIZE,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_END,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_START,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_OPTIMIZATION_STEPS,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_SEED_COUNT,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_END,
  FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_START,
  FLAPPY_WORKER_GEN0_PRETRAIN_BIAS_NOISE_STDDEV,
  FLAPPY_WORKER_GEN0_PRETRAIN_ITERATIONS,
  FLAPPY_WORKER_GEN0_PRETRAIN_RATE,
  FLAPPY_WORKER_GEN0_PRETRAIN_SAMPLE_COUNT,
  FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX,
  FLAPPY_WORKER_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV,
} from './flappy-evolution-worker.constants';
import {
  createSharedObservationMemoryState,
  resolveAdaptiveDifficultyProfile,
} from '../flappy.simulation.shared.utils';

/**
 * State carried between generation requests for one worker runtime.
 *
 * The warm-start service is intentionally one-shot. These fields let the worker
 * remember whether generation 0 has already been bootstrapped and which initial
 * RNG seed should be reused for deterministic synthetic sample generation.
 */
export interface WorkerWarmStartState {
  workerInitSeed: number;
  generationZeroWarmStartApplied: boolean;
}

/**
 * Dependency bag for generation-0 warm-start orchestration.
 *
 * The production path uses the real heuristic dataset builder and rollout-guided
 * template refinement. Tests can override these seams to keep assertions small
 * and deterministic.
 */
export interface WorkerWarmStartDependencies {
  buildHeuristicPretrainSet: (
    rng: ReturnType<typeof createXorshift32>,
    sampleCount: number,
  ) => Array<{ input: number[]; output: number[] }>;
  optimizeWarmStartTemplateNetwork: (
    templateNetwork: Network,
    workerInitSeed: number,
  ) => Network;
}

const DEFAULT_WORKER_WARM_START_DEPENDENCIES: WorkerWarmStartDependencies = {
  buildHeuristicPretrainSet,
  optimizeWarmStartTemplateNetwork: optimizeWarmStartTemplateNetwork,
};

/**
 * Applies a one-time generation-0 warm-start to improve initial demo quality.
 *
 * Educational note:
 * The worker entry should stay protocol-first. This service owns the short
 * supervised bootstrap pass that nudges generation 0 away from pure noise while
 * preserving the later NEAT-driven search loop.
 *
 * Conceptually this is a lightweight behavior-cloning pass. If you want more
 * background, the Wikipedia article on "imitation learning" is a helpful bridge
 * between the heuristic teacher used here and the later evolutionary search.
 *
 * @param neatController - Initialized NEAT runtime.
 * @param warmStartState - Mutable warm-start lifecycle state.
 * @returns Promise resolved when warm-start evaluation finishes.
 * @example
 * ```ts
 * await warmStartWorkerGenerationZeroIfNeeded(neatRuntime, {
 *   workerInitSeed: 123,
 *   generationZeroWarmStartApplied: false,
 * });
 * ```
 */
export async function warmStartWorkerGenerationZeroIfNeeded(
  neatController: Neat,
  warmStartState: WorkerWarmStartState,
  dependencies: WorkerWarmStartDependencies = DEFAULT_WORKER_WARM_START_DEPENDENCIES,
): Promise<void> {
  // Step 1: Exit fast when warm-start is already processed.
  if (warmStartState.generationZeroWarmStartApplied) return;

  // Step 2: Warm-start only generation 0 to avoid skewing later evolution.
  if (neatController.generation !== 0) {
    warmStartState.generationZeroWarmStartApplied = true;
    return;
  }

  // Step 3: Validate population availability.
  const population = neatController.population;
  if (!Array.isArray(population) || population.length === 0) {
    warmStartState.generationZeroWarmStartApplied = true;
    return;
  }

  // Step 4: Build a small synthetic dataset labeled by a simple heuristic.
  const warmStartRng = createXorshift32(
    warmStartState.workerInitSeed ^ 0x9e37_79b9,
  );
  const trainingSet = dependencies.buildHeuristicPretrainSet(
    warmStartRng,
    FLAPPY_WORKER_GEN0_PRETRAIN_SAMPLE_COUNT,
  );

  // Step 5: Train a single template network, then seed the whole population from it.
  const templateNetwork = population[0]?.clone();
  if (!templateNetwork) {
    warmStartState.generationZeroWarmStartApplied = true;
    return;
  }

  try {
    templateNetwork.train(trainingSet, {
      iterations: FLAPPY_WORKER_GEN0_PRETRAIN_ITERATIONS,
      rate: FLAPPY_WORKER_GEN0_PRETRAIN_RATE,
      batchSize: FLAPPY_WORKER_GEN0_PRETRAIN_BATCH_SIZE,
      optimizer: 'adam',
      mixedPrecision: false,
    });
  } catch {
    // If training fails for any reason, fall back to pure noise seeding.
  }

  // Step 6: Refine the trained template with a short rollout-guided hill-climb.
  const optimizedTemplateNetwork =
    dependencies.optimizeWarmStartTemplateNetwork(
      templateNetwork,
      warmStartState.workerInitSeed,
    );

  // Step 7: Copy trained weights/biases into each genome with small noise for diversity.
  for (const genome of population) {
    applyTemplateWeightsWithNoise(
      genome,
      optimizedTemplateNetwork,
      warmStartRng,
      {
        weightStdDev: FLAPPY_WORKER_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV,
        biasStdDev: FLAPPY_WORKER_GEN0_PRETRAIN_BIAS_NOISE_STDDEV,
      },
    );
    (genome as unknown as { score?: number }).score = undefined;
  }

  warmStartState.generationZeroWarmStartApplied = true;
}

/**
 * Builds synthetic supervised samples for generation-0 behavior cloning.
 *
 * Educational note:
 * These samples are not recorded gameplay traces. They are synthetic states
 * generated from the same observation pipeline used during real playback so the
 * teacher labels and the evolved policy inputs stay in the same feature space.
 *
 * @param rng - Deterministic random source.
 * @param sampleCount - Requested number of synthetic samples.
 * @returns Supervised dataset of input/output pairs.
 */
function buildHeuristicPretrainSet(
  rng: ReturnType<typeof createXorshift32>,
  sampleCount: number,
): Array<{ input: number[]; output: number[] }> {
  // Step 1: Resolve bounded sample count and a baseline difficulty profile.
  const clampedSampleCount = Math.max(8, Math.trunc(sampleCount));
  const difficultyProfile = resolveAdaptiveDifficultyProfile(0, 1);
  const trainingSet: Array<{ input: number[]; output: number[] }> = [];

  // Step 2: Sample state tuples, encode observations, and attach heuristic labels.
  for (let sampleIndex = 0; sampleIndex < clampedSampleCount; sampleIndex++) {
    const pipeGapPx = difficultyProfile.pipeGapPx;
    const gapHalfPx = pipeGapPx * 0.5;

    const birdYPx = rng.nextFloat01() * FLAPPY_WORLD_HEIGHT_PX;
    const velocityYPxPerFrame =
      (rng.nextFloat01() * 2 - 1) * FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME;

    const pipeXMinPx = FLAPPY_BIRD_X_PX + 40;
    const pipeXMaxPx =
      FLAPPY_BIRD_X_PX +
      FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX * 0.92;
    const pipeXSamplePx =
      pipeXMinPx + rng.nextFloat01() * Math.max(1, pipeXMaxPx - pipeXMinPx);

    const minGapCenterYPx = gapHalfPx;
    const maxGapCenterYPx = Math.max(
      minGapCenterYPx,
      FLAPPY_WORLD_HEIGHT_PX - gapHalfPx,
    );
    const gapCenterYPx =
      minGapCenterYPx + rng.nextFloat01() * (maxGapCenterYPx - minGapCenterYPx);

    const pipes: WorkerPopulationPipe[] = [
      {
        id: 1,
        xPx: pipeXSamplePx,
        gapCenterYPx,
        gapSizePx: pipeGapPx,
      },
    ];

    const observationMemoryState = createSharedObservationMemoryState();
    const observation = resolveObservationVector(
      birdYPx,
      velocityYPxPerFrame,
      pipes,
      FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX,
      FLAPPY_WORLD_HEIGHT_PX,
      difficultyProfile,
      difficultyProfile.pipeSpawnIntervalFrames,
      observationMemoryState,
    );

    const shouldFlap = resolveHeuristicTeacherFlapDecision(
      observation.observationFeatures,
    );
    trainingSet.push({
      input: observation.observationVector,
      output: shouldFlap ? [0, 1] : [1, 0],
    });
  }

  // Step 3: Return immutable training pairs consumed by Network.train.
  return trainingSet;
}

/**
 * Refines the generation-0 template against real Flappy rollouts.
 *
 * The warm-start teacher gets the template out of pure-random territory, but it
 * still only imitates a simple flap heuristic. This refinement pass keeps the
 * topology fixed and searches the parameter surface directly against actual
 * rollout fitness so the first visible generation starts closer to competent
 * control.
 *
 * @param templateNetwork - Heuristic-pretrained template network.
 * @param workerInitSeed - Deterministic worker seed.
 * @returns Best rollout-refined template found within the bounded budget.
 */
function optimizeWarmStartTemplateNetwork(
  templateNetwork: Network,
  workerInitSeed: number,
): Network {
  // Step 1: Build deterministic shared rollout seeds for candidate comparison.
  const rolloutSeedRng = createXorshift32(workerInitSeed ^ 0xa341_316c);
  const sharedRolloutSeeds = buildWarmStartRolloutSeedBatch(
    rolloutSeedRng,
    FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_SEED_COUNT,
  );

  // Step 2: Start from the teacher-fitted template as the current best policy.
  const optimizationRng = createXorshift32(workerInitSeed ^ 0xc801_3ea4);
  let bestTemplateNetwork = templateNetwork;
  let bestTemplateEvaluation = evaluateWarmStartTemplateAcrossRollouts(
    bestTemplateNetwork,
    sharedRolloutSeeds,
  );

  // Step 3: Run a bounded topology-fixed hill-climb on actual rollout fitness.
  for (
    let optimizationStepIndex = 0;
    optimizationStepIndex <
    FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_OPTIMIZATION_STEPS;
    optimizationStepIndex++
  ) {
    const candidateTemplateNetwork = bestTemplateNetwork.clone();
    const annealRatio = resolveWarmStartAnnealRatio(
      optimizationStepIndex,
      FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_OPTIMIZATION_STEPS,
    );

    perturbNetworkParametersInPlace(candidateTemplateNetwork, optimizationRng, {
      weightStdDev: interpolateValue(
        FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_START,
        FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_END,
        annealRatio,
      ),
      biasStdDev: interpolateValue(
        FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_START,
        FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_END,
        annealRatio,
      ),
    });

    const candidateTemplateEvaluation = evaluateWarmStartTemplateAcrossRollouts(
      candidateTemplateNetwork,
      sharedRolloutSeeds,
    );
    if (
      !isWarmStartEvaluationBetter(
        candidateTemplateEvaluation,
        bestTemplateEvaluation,
      )
    ) {
      continue;
    }

    bestTemplateNetwork = candidateTemplateNetwork;
    bestTemplateEvaluation = candidateTemplateEvaluation;
  }

  return bestTemplateNetwork;
}

/**
 * Builds the deterministic shared rollout seed batch used during warm-start refinement.
 *
 * @param rng - Deterministic random source.
 * @param seedCount - Requested seed count.
 * @returns Shared rollout seed batch.
 */
function buildWarmStartRolloutSeedBatch(
  rng: ReturnType<typeof createXorshift32>,
  seedCount: number,
): number[] {
  // Step 1: Clamp the request and sample uint32-compatible rollout seeds.
  const clampedSeedCount = Math.max(1, Math.trunc(seedCount));
  return Array.from({ length: clampedSeedCount }, () =>
    rng.nextInt(0, 0x1_0000_0000),
  );
}

/**
 * Evaluates one warm-start template across the shared rollout seed batch.
 *
 * @param templateNetwork - Candidate template to score.
 * @param sharedRolloutSeeds - Shared rollout seeds used for stable comparison.
 * @returns Aggregate shared-seed evaluation.
 */
function evaluateWarmStartTemplateAcrossRollouts(
  templateNetwork: Network,
  sharedRolloutSeeds: readonly number[],
): FlappySeedBatchEvaluation {
  // Step 1: Reset network state when the runtime exposes a clear hook.
  const maybeClearableNetwork = templateNetwork as Network & {
    clear?: () => void;
  };
  maybeClearableNetwork.clear?.();

  // Step 2: Score the fixed topology on real Flappy rollouts.
  return evaluateFlappyFitnessAcrossSeeds(templateNetwork, sharedRolloutSeeds, {
    enableEarlyTermination: true,
    maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
  });
}

/**
 * Resolves whether the candidate batch evaluation beats the current best one.
 *
 * Robust fitness is the primary signal. Mean pipe progress and mean frame
 * survival act as deterministic tie-breakers so upgrades remain stable when the
 * robust score is identical.
 *
 * @param candidateEvaluation - Newly scored candidate aggregate.
 * @param bestEvaluation - Current best aggregate.
 * @returns True when the candidate should replace the incumbent template.
 */
function isWarmStartEvaluationBetter(
  candidateEvaluation: FlappySeedBatchEvaluation,
  bestEvaluation: FlappySeedBatchEvaluation,
): boolean {
  // Step 1: Prefer higher robust shared-seed fitness.
  if (candidateEvaluation.robustFitness !== bestEvaluation.robustFitness) {
    return candidateEvaluation.robustFitness > bestEvaluation.robustFitness;
  }

  // Step 2: Break ties with more practical gameplay progress.
  if (candidateEvaluation.meanPipesPassed !== bestEvaluation.meanPipesPassed) {
    return candidateEvaluation.meanPipesPassed > bestEvaluation.meanPipesPassed;
  }

  return (
    candidateEvaluation.meanFramesSurvived > bestEvaluation.meanFramesSurvived
  );
}

/**
 * Resolves the annealing ratio for rollout-guided warm-start refinement.
 *
 * @param optimizationStepIndex - Zero-based optimization step index.
 * @param totalOptimizationSteps - Total number of optimization steps.
 * @returns Clamped ratio in the inclusive range [0, 1].
 */
function resolveWarmStartAnnealRatio(
  optimizationStepIndex: number,
  totalOptimizationSteps: number,
): number {
  // Step 1: Collapse one-step schedules to the broad-search endpoint.
  if (totalOptimizationSteps <= 1) {
    return 0;
  }

  // Step 2: Convert the step index into a stable cooling ratio.
  return Math.min(
    1,
    Math.max(0, optimizationStepIndex / (totalOptimizationSteps - 1)),
  );
}

/**
 * Applies additive Gaussian noise to an existing network in-place.
 *
 * Unlike the later population seeding copy step, this helper perturbs the
 * candidate template directly so the rollout optimizer can evaluate one local
 * parameter move at a time while keeping the topology unchanged.
 *
 * @param network - Candidate template to perturb.
 * @param rng - Deterministic random source.
 * @param noise - Standard deviations for weight and bias perturbations.
 * @returns Nothing.
 */
function perturbNetworkParametersInPlace(
  network: Network,
  rng: ReturnType<typeof createXorshift32>,
  noise: { weightStdDev: number; biasStdDev: number },
): void {
  // Step 1: Clamp noise scales to valid non-negative values.
  const weightStdDev = Math.max(0, noise.weightStdDev);
  const biasStdDev = Math.max(0, noise.biasStdDev);

  // Step 2: Perturb node biases in-place.
  for (const networkNode of network.nodes) {
    networkNode.bias += sampleGaussian(rng) * biasStdDev;
  }

  // Step 3: Perturb connection weights in-place.
  for (const networkConnection of network.connections) {
    networkConnection.weight += sampleGaussian(rng) * weightStdDev;
  }
}

/**
 * Linearly interpolates between two scalar values.
 *
 * @param startValue - Value at ratio `0`.
 * @param endValue - Value at ratio `1`.
 * @param ratio - Interpolation ratio.
 * @returns Interpolated value.
 */
function interpolateValue(
  startValue: number,
  endValue: number,
  ratio: number,
): number {
  // Step 1: Clamp the ratio before blending.
  const clampedRatio = Math.min(1, Math.max(0, ratio));
  return startValue + (endValue - startValue) * clampedRatio;
}

/**
 * Heuristic teacher policy used to label synthetic pretraining samples.
 *
 * The rule intentionally stays simple and interpretable: flap when the bird is
 * meaningfully below the next gap center, not already rising fast, and either
 * close to the gap entry or in an urgent approach state.
 *
 * @param features - Structured observation features for one synthetic state.
 * @returns True when the teacher says to flap.
 */
function resolveHeuristicTeacherFlapDecision(
  features: WorkerHeuristicObservationFeatures,
): boolean {
  // Step 1: Resolve the few interpretable teacher conditions.
  const isBelowNextGapCenter = features.normalizedDeltaToNextGap > 0.035;
  const isNotAlreadyRisingFast = features.normalizedVelocity > -0.25;
  const isNearGapEntry = features.normalizedFramesToGapEntry < 0.8;
  const isUrgent = features.normalizedEntryUrgency > 0.18;

  // Step 2: Fold the teacher decision from those conditions.
  return (
    isBelowNextGapCenter &&
    isNotAlreadyRisingFast &&
    (isNearGapEntry || isUrgent)
  );
}

/**
 * Copies template parameters into a genome and injects small Gaussian noise.
 *
 * Educational note:
 * The template network gives generation 0 a shared prior, while the noise terms
 * restore diversity so the population is still worth evolving.
 *
 * @param genome - Target genome to mutate in-place.
 * @param template - Trained template source network.
 * @param rng - Deterministic random source for noise sampling.
 * @param noise - Standard deviations for weight and bias perturbations.
 * @returns Nothing.
 */
function applyTemplateWeightsWithNoise(
  genome: Network,
  template: Network,
  rng: ReturnType<typeof createXorshift32>,
  noise: { weightStdDev: number; biasStdDev: number },
): void {
  // Step 1: Clamp noise scales to valid non-negative values.
  const weightStdDev = Math.max(0, noise.weightStdDev);
  const biasStdDev = Math.max(0, noise.biasStdDev);

  // Step 2: Copy node biases with additive Gaussian noise.
  const genomeNodes = genome.nodes;
  const templateNodes = template.nodes;
  const nodeCopyCount = Math.min(genomeNodes.length, templateNodes.length);
  for (let nodeIndex = 0; nodeIndex < nodeCopyCount; nodeIndex++) {
    const templateBias = templateNodes[nodeIndex]?.bias ?? 0;
    genomeNodes[nodeIndex].bias =
      templateBias + sampleGaussian(rng) * biasStdDev;
  }

  // Step 3: Copy connection weights with additive Gaussian noise.
  const genomeConnections = genome.connections;
  const templateConnections = template.connections;
  const connectionCopyCount = Math.min(
    genomeConnections.length,
    templateConnections.length,
  );
  for (
    let connectionIndex = 0;
    connectionIndex < connectionCopyCount;
    connectionIndex++
  ) {
    const templateWeight = templateConnections[connectionIndex]?.weight ?? 0;
    genomeConnections[connectionIndex].weight =
      templateWeight + sampleGaussian(rng) * weightStdDev;
  }
}

/**
 * Samples one standard-normal value using the Box-Muller transform.
 *
 * If you are unfamiliar with the transform, the Wikipedia article on
 * "Box-Muller transform" is a useful short background read. The worker uses it
 * here because it is deterministic, dependency-light, and good enough for small
 * noise injection during warm-start diversification.
 *
 * @param rng - Deterministic random source.
 * @returns One approximately standard-normal random value.
 */
function sampleGaussian(rng: ReturnType<typeof createXorshift32>): number {
  // Step 1: Avoid log(0) by flooring uniforms away from zero.
  const epsilon = 1e-12;
  const uniformA = Math.max(epsilon, rng.nextFloat01());
  const uniformB = Math.max(epsilon, rng.nextFloat01());

  // Step 2: Convert two uniforms into one normal sample.
  const magnitude = Math.sqrt(-2 * Math.log(uniformA));
  const angle = 2 * Math.PI * uniformB;
  return magnitude * Math.cos(angle);
}
