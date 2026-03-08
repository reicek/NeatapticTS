/**
 * trainingWarmStart.ts
 *
 * Lamarckian warm-start and population pretraining subsystem.
 *
 * Responsibilities:
 * - Build supervised training datasets for compass-guided navigation
 * - Apply Lamarckian backpropagation training to populations
 * - Adjust output biases after training to maintain exploration
 * - Orchestrate conditional warm-start pretraining with buffer management
 *
 * All functions are pure/side-effect-free except where explicitly documented.
 * Mutations are limited to NEAT population networks and engine scratch buffers.
 *
 * @module trainingWarmStart
 */

import type { Neat, Network } from '../../../../src/neataptic';
import { methods } from '../../../../src/neataptic';
import type { EngineState, RngCacheParameters } from './engineState.types';
import { initialiseTelemetryScratch } from './engineState';
import { drawFastRandom, readHighResolutionTime } from './rngAndTiming';
import { sampleArray } from './sampling';

/** Empty array constant for defensive fallbacks. */
const EMPTY_VEC: readonly never[] = [];

/**
 * Build the supervised training set used for Lamarckian warm-start training.
 *
 * Generates a small curated dataset of canonical navigation scenarios combining:
 * - Single-path corridors with varying progress signals
 * - Two-way junctions with directional bias
 * - Four-way intersections with full openness variety
 * - Regression cases (backtracking or stalled progress)
 * - Mild data augmentation via random jitter on openness and progress values
 *
 * Each training case maps a 6-dimensional input to a soft one-hot output:
 * - Input: [compassScalar, openN, openE, openS, openW, progressDelta]
 * - Output: [pN, pE, pS, pW] with probabilities (high for target, low for others)
 *
 * @param state - Shared engine state (used for RNG when applying jitter augmentation).
 * @param constants - Training hyperparameters and signal constants.
 * @returns Array of training cases `{ input: number[], output: number[] }`.
 *
 * @example
 * const trainingSet = buildLamarckianTrainingSet(engineState, {
 *   TRAIN_OUT_PROB_HIGH: 0.9,
 *   TRAIN_OUT_PROB_LOW: 0.033,
 *   PROGRESS_MEDIUM: 0.5,
 *   // ... other constants
 * });
 */
export const buildLamarckianTrainingSet = (
  state: EngineState,
  constants: {
    TRAIN_OUT_PROB_HIGH: number;
    TRAIN_OUT_PROB_LOW: number;
    PROGRESS_MEDIUM: number;
    PROGRESS_STRONG: number;
    PROGRESS_JUNCTION: number;
    PROGRESS_FOURWAY: number;
    PROGRESS_REGRESS: number;
    PROGRESS_MIN_SIGNAL: number;
    PROGRESS_MILD_REGRESS: number;
    DEFAULT_JITTER_PROB: number;
    AUGMENT_JITTER_BASE: number;
    AUGMENT_JITTER_RANGE: number;
    AUGMENT_PROGRESS_JITTER_PROB: number;
    AUGMENT_PROGRESS_DELTA_RANGE: number;
    AUGMENT_PROGRESS_DELTA_HALF: number;
    RNG_PARAMETERS: RngCacheParameters;
  },
): { input: number[]; output: number[] }[] => {
  // Step 1: Prepare the result container (small, bounded dataset).
  const trainingSet: { input: number[]; output: number[] }[] = [];

  // Step 2: Precompute and reuse the four canonical soft one-hot output vectors.
  const high = constants.TRAIN_OUT_PROB_HIGH;
  const low = constants.TRAIN_OUT_PROB_LOW;
  const OUTPUTS: number[][] = [
    [high, low, low, low],
    [low, high, low, low],
    [low, low, high, low],
    [low, low, low, high],
  ];

  // Helper to construct an input vector (small and explicit for readability).
  const makeInput = (
    compassScalar: number,
    openN: number,
    openE: number,
    openS: number,
    openW: number,
    progressDelta: number,
  ) => [compassScalar, openN, openE, openS, openW, progressDelta];

  // Local helper to append a case (keeps call sites terse).
  const pushCase = (inputVector: number[], directionIndex: number) =>
    trainingSet.push({ input: inputVector, output: OUTPUTS[directionIndex] });

  // Step 3: Populate the dataset with canonical scenarios.

  // Single open path with steady progress
  pushCase(makeInput(0, 1, 0, 0, 0, constants.PROGRESS_MEDIUM), 0);
  pushCase(makeInput(0.25, 0, 1, 0, 0, constants.PROGRESS_MEDIUM), 1);
  pushCase(makeInput(0.5, 0, 0, 1, 0, constants.PROGRESS_MEDIUM), 2);
  pushCase(makeInput(0.75, 0, 0, 0, 1, constants.PROGRESS_MEDIUM), 3);

  // Strong progress cases
  pushCase(makeInput(0, 1, 0, 0, 0, constants.PROGRESS_STRONG), 0);
  pushCase(makeInput(0.25, 0, 1, 0, 0, constants.PROGRESS_STRONG), 1);

  // Two-way junctions (ambiguous openings => bias toward one direction)
  pushCase(makeInput(0, 1, 0.6, 0, 0, constants.PROGRESS_JUNCTION), 0);
  pushCase(makeInput(0, 1, 0, 0.6, 0, constants.PROGRESS_JUNCTION), 0);
  pushCase(makeInput(0.25, 0.6, 1, 0, 0, constants.PROGRESS_JUNCTION), 1);
  pushCase(makeInput(0.25, 0, 1, 0.6, 0, constants.PROGRESS_JUNCTION), 1);
  pushCase(makeInput(0.5, 0, 0.6, 1, 0, constants.PROGRESS_JUNCTION), 2);
  pushCase(makeInput(0.5, 0, 0, 1, 0.6, constants.PROGRESS_JUNCTION), 2);
  pushCase(makeInput(0.75, 0, 0, 0.6, 1, constants.PROGRESS_JUNCTION), 3);
  pushCase(makeInput(0.75, 0.6, 0, 0, 1, constants.PROGRESS_JUNCTION), 3);

  // Four-way junctions (full variety)
  pushCase(makeInput(0, 1, 0.8, 0.5, 0.4, constants.PROGRESS_FOURWAY), 0);
  pushCase(makeInput(0.25, 0.7, 1, 0.6, 0.5, constants.PROGRESS_FOURWAY), 1);
  pushCase(makeInput(0.5, 0.6, 0.55, 1, 0.65, constants.PROGRESS_FOURWAY), 2);
  pushCase(makeInput(0.75, 0.5, 0.45, 0.7, 1, constants.PROGRESS_FOURWAY), 3);

  // Regressing cases
  pushCase(makeInput(0, 1, 0.3, 0, 0, constants.PROGRESS_REGRESS), 0);
  pushCase(makeInput(0.25, 0.5, 1, 0.4, 0, constants.PROGRESS_REGRESS), 1);
  pushCase(makeInput(0.5, 0, 0.3, 1, 0.2, constants.PROGRESS_REGRESS), 2);
  pushCase(makeInput(0.75, 0, 0.5, 0.4, 1, constants.PROGRESS_REGRESS), 3);
  pushCase(
    makeInput(
      0,
      0,
      0,
      constants.PROGRESS_MIN_SIGNAL,
      0,
      constants.PROGRESS_MILD_REGRESS,
    ),
    2,
  );

  // Step 4: Mild augmentation (jitter openness & progress).
  const randomParameters = constants.RNG_PARAMETERS;
  for (
    let datasetIndex = 0;
    datasetIndex < trainingSet.length;
    datasetIndex++
  ) {
    const caseEntry = trainingSet[datasetIndex];

    // Jitter openness values (directional inputs 1-4)
    for (let directionIndex = 1; directionIndex <= 4; directionIndex++) {
      if (
        caseEntry.input[directionIndex] === 1 &&
        drawFastRandom(state, randomParameters) < constants.DEFAULT_JITTER_PROB
      ) {
        caseEntry.input[directionIndex] =
          constants.AUGMENT_JITTER_BASE +
          drawFastRandom(state, randomParameters) *
            constants.AUGMENT_JITTER_RANGE;
      }
    }

    // Jitter progress delta (input[5])
    if (
      drawFastRandom(state, randomParameters) <
      constants.AUGMENT_PROGRESS_JITTER_PROB
    ) {
      caseEntry.input[5] = Math.min(
        1,
        Math.max(
          0,
          caseEntry.input[5] +
            (drawFastRandom(state, randomParameters) *
              constants.AUGMENT_PROGRESS_DELTA_RANGE -
              constants.AUGMENT_PROGRESS_DELTA_HALF),
        ),
      );
    }
  }

  return trainingSet;
};

/**
 * Adjust output node biases after training to maintain exploration diversity.
 *
 * This heuristic prevents trained networks from collapsing into deterministic behavior
 * by centering output biases (subtracting mean) and optionally scaling when variance is low.
 *
 * Steps:
 * 1. Collect output node biases into scratch buffer
 * 2. Compute mean and standard deviation via Welford's one-pass algorithm
 * 3. Subtract mean from each bias
 * 4. If std is very small, apply multiplicative boost to increase variance
 * 5. Clamp adjusted biases to safe operational range [-5, 5] and write back
 *
 * @param network - Network object containing `nodes` array. Missing nodes treated as empty.
 * @param state - Shared engine state (provides scratch buffers and node index pool).
 * @param constants - Hyperparameters for adjustment (std threshold, multiplier).
 * @param scratchNodeIdx - Pooled index buffer for output nodes (written by getNodeIndicesByType).
 * @param getNodeIndicesByType - Helper function to collect node indices by type.
 *
 * @example
 * adjustOutputBiasesAfterTraining(
 *   trainedNetwork,
 *   engineState,
 *   { DEFAULT_STD_SMALL: 0.05, DEFAULT_STD_ADJUST_MULT: 1.5 },
 *   scratchIndexBuffer,
 *   getNodeIndicesByType
 * );
 */
export const adjustOutputBiasesAfterTraining = (
  network: Network,
  state: EngineState,
  constants: {
    DEFAULT_STD_SMALL: number;
    DEFAULT_STD_ADJUST_MULT: number;
  },
  scratchNodeIdx: Int32Array,
  // Type assertion: helper function working with dynamic node arrays
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  getNodeIndicesByType: (nodes: any[], nodeType: string) => number,
): void => {
  try {
    // Step 1: Early exit when no network or no nodes exist.
    if (!network) return;

    const nodesRef = network.nodes ?? EMPTY_VEC;
    const outputNodeCount = getNodeIndicesByType(nodesRef, 'output');
    if (outputNodeCount <= 0) return;

    // Step 2: Ensure pooled scratch buffer capacity via shared helper (geometric growth).
    const biasScratch = initialiseTelemetryScratch(
      { biasCount: outputNodeCount },
      state,
    ).biasScratch;

    // Step 3: Welford one-pass accumulate into local variables while writing raw biases into scratch.
    let mean = 0;
    let M2 = 0;
    for (let outputIndex = 0; outputIndex < outputNodeCount; outputIndex++) {
      const nodeIndex = scratchNodeIdx[outputIndex];
      const currentBias = nodesRef[nodeIndex]?.bias ?? 0;
      biasScratch[outputIndex] = currentBias;

      const sampleIndex = outputIndex + 1;
      const delta = currentBias - mean;
      mean += delta / sampleIndex;
      M2 += delta * (currentBias - mean);
    }

    // Step 4: Population standard deviation (avoid division by zero).
    const populationStd =
      outputNodeCount > 0 ? Math.sqrt(M2 / outputNodeCount) : 0;
    const smallStdThreshold = constants.DEFAULT_STD_SMALL;
    const smallStdMultiplier = constants.DEFAULT_STD_ADJUST_MULT;

    // Step 5: Subtract mean, optionally scale small-std results, clamp and write back.
    for (let outputIndex = 0; outputIndex < outputNodeCount; outputIndex++) {
      const nodeIndex = scratchNodeIdx[outputIndex];
      let adjustedBias = biasScratch[outputIndex] - mean;
      if (populationStd < smallStdThreshold) {
        adjustedBias *= smallStdMultiplier;
      }
      // Clamp to safe operational range
      nodesRef[nodeIndex].bias = Math.max(-5, Math.min(5, adjustedBias));
    }
  } catch {
    // Best-effort: swallow errors to avoid breaking the engine loop.
  }
};

/**
 * Pretrain the population using a small supervised dataset and apply warm-start heuristics.
 *
 * Behaviour & contract:
 * - Runs a short supervised training pass (backprop) on each network in `neat.population`
 * - Applies lightweight warm-start heuristics after training: compass wiring and output bias centering
 * - Errors are isolated per-network: a failing network does not abort the overall pretrain step
 * - This helper is allocation-light and does not create sizable temporary buffers
 *
 * Steps:
 * 1. Validate inputs and obtain `population` (fast-exit on empty populations)
 * 2. For each network: guard missing `train` method, compute conservative iteration budget, then call `train`
 * 3. Apply warm-start heuristics (compass wiring + bias centering). Swallow any per-network exceptions.
 *
 * @param neat - NEAT instance exposing a `population` array of networks.
 * @param lamarckianTrainingSet - Array of `{input:number[], output:number[]}` training cases.
 * @param constants - Training hyperparameters (iteration limits, learning rates, etc.).
 * @param applyCompassWarmStart - Helper function for compass wiring adjustment.
 * @param centerOutputBiases - Helper function for output bias centering.
 *
 * @example
 * pretrainPopulationWarmStart(
 *   neatInstance,
 *   trainingDataset,
 *   { PRETRAIN_MAX_ITER: 10, PRETRAIN_BASE_ITER: 3, ... },
 *   applyCompassWarmStart,
 *   centerOutputBiases
 * );
 */
export const pretrainPopulationWarmStart = (
  neat: Neat,
  // Type assertion: training dataset structure
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  lamarckianTrainingSet: any[],
  constants: {
    PRETRAIN_MAX_ITER: number;
    PRETRAIN_BASE_ITER: number;
    DEFAULT_TRAIN_ERROR: number;
    DEFAULT_PRETRAIN_RATE: number;
    DEFAULT_PRETRAIN_MOMENTUM: number;
    DEFAULT_TRAIN_BATCH_SMALL: number;
  },
  // Type assertion: helper functions working with networks
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  applyCompassWarmStart: (network: any) => void,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  centerOutputBiases: (network: any) => void,
): void => {
  // Step 1: Defensive validation & fast exit.
  if (!neat) return;
  const population = neat.population ?? EMPTY_VEC;
  if (!Array.isArray(population) || population.length === 0) return;

  // Step 2: Iterate population and apply supervised training per network (best-effort).
  for (let networkIndex = 0; networkIndex < population.length; networkIndex++) {
    // Type assertion: network from population array
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const network: any = population[networkIndex];
    try {
      if (!network || typeof network.train !== 'function') continue; // skip non-trainable entries

      // Compute conservative per-network iteration budget (bounded by PRETRAIN_MAX_ITER).
      const iterations = Math.min(
        constants.PRETRAIN_MAX_ITER,
        constants.PRETRAIN_BASE_ITER +
          Math.floor((lamarckianTrainingSet?.length || 0) / 2),
      );

      // Delegate to the network's own training routine; options are intentionally conservative.
      network.train(lamarckianTrainingSet, {
        iterations,
        error: constants.DEFAULT_TRAIN_ERROR,
        rate: constants.DEFAULT_PRETRAIN_RATE,
        momentum: constants.DEFAULT_PRETRAIN_MOMENTUM,
        batchSize: constants.DEFAULT_TRAIN_BATCH_SMALL,
        allowRecurrent: true,
        cost: methods.Cost.softmaxCrossEntropy,
      });

      // Step 3: Apply warm-start heuristics after training (best-effort; swallow individual failures).
      try {
        applyCompassWarmStart(network);
      } catch {
        // ignore compass warm-start failures
      }

      try {
        centerOutputBiases(network);
      } catch {
        // ignore bias centering failures
      }
    } catch {
      // Per-network training failure is non-fatal; continue with others.
    }
  }
};

/**
 * Apply Lamarckian backpropagation training to the entire population with optional profiling.
 *
 * Runs a bounded supervised training pass on each network in the population, optionally
 * downsampling the training set for efficiency. Collects gradient norm statistics and
 * applies bias adjustment heuristics to maintain exploration after training.
 *
 * Steps:
 * 1. Validate inputs & early exits
 * 2. Start profiling timer if requested
 * 3. Optionally down-sample the training set (with replacement) to reduce cost for large sets
 * 4. Iterate networks performing a bounded training pass with softmax cross-entropy cost
 * 5. Apply post-training bias adjustment to each network
 * 6. Collect optional training stats (gradient norms)
 * 7. Emit aggregate gradient telemetry if samples were collected
 * 8. Return elapsed time when profiling; otherwise return 0
 *
 * @param neat - NEAT instance exposing `population` array.
 * @param trainingSet - Array of `{input:number[], output:number[]}` training cases.
 * @param iterations - Number of training iterations to run per-network (must be > 0).
 * @param sampleSize - Optional sample size to down-sample `trainingSet` (with replacement).
 * @param safeWrite - Logging helper used for telemetry lines (string writer).
 * @param profileEnabled - When true, function returns elapsed ms spent training; otherwise returns 0.
 * @param completedGenerations - Generation index used when emitting telemetry lines.
 * @param state - Shared engine state (provides RNG for sampling and profiling timer).
 * @param constants - Training hyperparameters (learning rates, batch sizes, etc.).
 * @param adjustOutputBiases - Helper function for post-training bias adjustment.
 * @returns Elapsed milliseconds spent in training when profiling is enabled; otherwise 0.
 *
 * @example
 * const elapsed = applyLamarckianTraining(
 *   neatInstance,
 *   trainingExamples,
 *   2,
 *   8,
 *   console.log,
 *   true,
 *   currentGen,
 *   engineState,
 *   trainingConstants,
 *   adjustOutputBiasesAfterTraining
 * );
 */
export const applyLamarckianTraining = (
  neat: Neat,
  // Type assertion: training dataset structure
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  trainingSet: any[],
  iterations: number,
  sampleSize: number | undefined,
  safeWrite: (msg: string) => void,
  profileEnabled: boolean,
  completedGenerations: number,
  state: EngineState,
  constants: {
    DEFAULT_TRAIN_ERROR: number;
    DEFAULT_TRAIN_RATE: number;
    DEFAULT_TRAIN_MOMENTUM: number;
    DEFAULT_TRAIN_BATCH_SMALL: number;
  },
  // Type assertion: helper function working with networks
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  adjustOutputBiases: (network: any) => void,
): number => {
  // Step 1: Validate inputs & early exits.
  if (
    !neat ||
    !Array.isArray(neat.population) ||
    neat.population.length === 0
  ) {
    return 0;
  }
  if (!Array.isArray(trainingSet) || trainingSet.length === 0) return 0;
  if (!Number.isFinite(iterations) || iterations <= 0) return 0;

  // Step 2: Start profiling timer if requested.
  const profileStart = profileEnabled ? readHighResolutionTime() : 0;

  // Step 3: Optionally down-sample the training set (with replacement).
  const trainingSetRef =
    sampleSize && sampleSize > 0 && sampleSize < trainingSet.length
      ? sampleArray(state, trainingSet, sampleSize)
      : trainingSet;

  // Step 4: Iterate networks performing a bounded training pass.
  let gradientNormSum = 0;
  let gradientNormSamples = 0;
  // Type assertion: population is Network array from Neat
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const populationRef = neat.population as any[];

  for (const network of populationRef) {
    if (!network) continue; // defensive guard for sparse arrays
    try {
      // 4.1: Run a conservative training invocation to avoid destroying diversity.
      network.train(trainingSetRef, {
        iterations,
        error: constants.DEFAULT_TRAIN_ERROR,
        rate: constants.DEFAULT_TRAIN_RATE,
        momentum: constants.DEFAULT_TRAIN_MOMENTUM,
        batchSize: constants.DEFAULT_TRAIN_BATCH_SMALL,
        allowRecurrent: true,
        cost: methods.Cost.softmaxCrossEntropy,
      });

      // 4.2: Heuristic bias adjustment after training to maintain exploration.
      adjustOutputBiases(network);

      // 4.3: Collect optional training stats (use optional chaining to avoid errors).
      try {
        // Type assertion: accessing optional training stats method
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const stats = (network as any).getTrainingStats?.();
        const gradNorm = stats?.gradNorm;
        if (Number.isFinite(gradNorm)) {
          gradientNormSum += gradNorm;
          gradientNormSamples++;
        }
      } catch {
        // Non-fatal: skip stat collection errors silently.
      }
    } catch {
      // Per-network training failure is non-fatal; continue with others.
    }
  }

  // Step 5: Emit aggregate gradient telemetry if samples were collected.
  if (gradientNormSamples > 0) {
    const meanGrad = gradientNormSum / gradientNormSamples;
    safeWrite(
      `[GRAD] gen=${completedGenerations} meanGradNorm=${meanGrad.toFixed(
        4,
      )} samples=${gradientNormSamples}\n`,
    );
  }

  // Step 6: Return elapsed time when profiling; otherwise return 0.
  return profileEnabled ? readHighResolutionTime() - profileStart : 0;
};

/**
 * Conditionally warm-start / pretrain the population using a provided training set.
 *
 * Behaviour & contract:
 * - If a non-empty `trainingSet` is provided this method will attempt a best-effort
 *   invocation of `pretrainPopulationWarmStart(neat, trainingSet)`
 * - To reduce first-use allocation spikes the helper will also attempt to ensure
 *   engine-level pooled buffers exist and have a reasonable capacity before
 *   pretraining begins (plain Array for sampling and Float64Array for numeric work)
 * - All operations are non-throwing; any internal error is swallowed to keep the
 *   evolution loop resilient
 *
 * Steps:
 * 1. Fast-guard invalid inputs – nothing to do when no data or driver
 * 2. Best-effort ensure pooled buffers exist and have capacity
 * 3. Invoke the pretrain helper (best-effort). This may mutate the NEAT driver
 *
 * @param neat - NEAT driver instance which will be pre-trained (may be `null`/`undefined`).
 * @param trainingSet - Array of supervised training cases used for warm-start; ignored when empty.
 * @param state - Shared engine state (provides scratch buffer pools).
 * @param pretrainPopulation - Helper function for population pretraining.
 *
 * @example
 * warmStartPopulationIfNeeded(
 *   neatDriver,
 *   trainingCasesArray,
 *   engineState,
 *   pretrainPopulationWarmStart
 * );
 */
export const warmStartPopulationIfNeeded = (
  neat: Neat,
  // Type assertion: training dataset structure
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  trainingSet: any[],
  state: EngineState,
  // Type assertion: helper function for population pretraining
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  pretrainPopulation: (neat: any, trainingSet: any[]) => void,
): void => {
  try {
    // Step 1: Fast-guard invalid inputs – nothing to do when no data or driver.
    const hasTrainingCases =
      Array.isArray(trainingSet) && trainingSet.length > 0;
    if (!hasTrainingCases || !neat) return; // best-effort: quietly skip

    // Step 2: Best-effort ensure pooled buffers exist and have capacity.
    // Determine a sensible sizing target (prefer driver configured popSize when available).
    const configuredPopulationSize = Number.isFinite(neat?.options?.popSize)
      ? Math.max(0, Math.floor(neat.options.popSize))
      : Math.max(8, trainingSet.length);

    // Helper: next power-of-two >= n
    const nextPowerOfTwo = (n: number) =>
      1 << Math.ceil(Math.log2(Math.max(1, n)));
    const targetCapacity = nextPowerOfTwo(
      Math.max(8, configuredPopulationSize),
    );

    // Ensure a plain-array pooled sample buffer exists and has the target capacity.
    try {
      const scratchBundle = state.scratch;
      let pooledSampleBuffer = scratchBundle.samplePool;
      if (!Array.isArray(pooledSampleBuffer)) {
        pooledSampleBuffer = scratchBundle.samplePool = [];
      }
      if (pooledSampleBuffer.length < targetCapacity) {
        pooledSampleBuffer.length = targetCapacity;
      }
    } catch {
      // ignore pooling failures – non-critical
    }

    // Ensure a numeric scratch Float64Array for temporary numeric reductions exists.
    try {
      const scratchBundle = state.scratch;
      let pooledNumericScratch = scratchBundle.exps;
      if (
        !(pooledNumericScratch instanceof Float64Array) ||
        pooledNumericScratch.length < 16
      ) {
        const numericSize = Math.max(
          16,
          nextPowerOfTwo(Math.min(256, configuredPopulationSize)),
        );
        pooledNumericScratch = new Float64Array(numericSize);
        scratchBundle.exps = pooledNumericScratch;
      }
    } catch {
      // ignore allocation failures – continue without pooled numeric scratch
    }

    // Step 3: Invoke the pretrain helper (best-effort). This may mutate the NEAT driver.
    try {
      pretrainPopulation(neat, trainingSet);
    } catch {
      // swallow – pretraining is optional and must not disrupt the run
    }
  } catch {
    // global swallow: keep best-effort semantics
  }
};
