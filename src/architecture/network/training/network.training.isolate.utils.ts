import type Network from '../../network/network';
import {
  fromParameterVector,
  toParameterVector,
} from '../serialize/network.serialize.utils';
import type { ParameterVector } from '../serialize/network.serialize.utils';
import { trainImpl } from './network.training.utils';
import type { TrainingSample } from './network.training.utils.types';

/**
 * Explicit settings for one isolated fine-tune pass.
 *
 * `steps` maps to the training loop iteration count and `learningRate` maps to
 * the training rate. `seed` is optional because same-runtime ordered
 * determinism only becomes a strong claim when the caller supplies both a
 * stable dataset order and an explicit deterministic seed.
 *
 * Without an explicit `seed`, the helper still guarantees isolation: the
 * original network and vector are never mutated, but repeated calls may
 * produce different trained vectors when the underlying training loop contains
 * stochastic behaviour such as dropout.
 */
export interface FineTuneOptions {
  /** Ordered training iterations to run on the working copy. */
  steps: number;
  /** Learning rate passed through to the training loop. */
  learningRate: number;
  /** Optional deterministic seed installed on the working copy only. */
  seed?: number;
}

/**
 * Detached result from one isolated fine-tune pass.
 *
 * The helper returns a trained parameter vector rather than a mutated network
 * so shared candidate state stays outside the training-owned boundary. Metrics
 * are numeric summaries from the existing training loop, not persisted
 * optimizer or runtime state.
 *
 * The returned `trainedVector` can be compared against the original vector,
 * forwarded to a worker for scoring, persisted as a checkpoint delta, or
 * discarded when only the fitness score matters.
 */
export interface FineTuneResult {
  /** Detached parameter-vector snapshot exported from the trained working copy. */
  trainedVector: ParameterVector;
  /** Numeric training summary aligned with the underlying training loop. */
  metrics?: Record<string, number>;
}

/**
 * Fine-tune one parameter vector against an ordered dataset without mutating shared state.
 *
 * The helper clones `baseNetwork`, applies `vector` to that working copy,
 * optionally installs an explicit deterministic seed via `Network.setSeed(...)`,
 * runs the existing training loop in the caller-provided dataset order, and
 * returns a new `ParameterVector` exported from the trained working copy. The
 * supplied `baseNetwork` and `vector` are read-only inputs to this helper.
 *
 * Determinism is intentionally scoped. On the same runtime, repeated calls can
 * return the same trained vector when topology, dataset order, training
 * settings, and explicit `seed` all match. This helper does not claim
 * cross-runtime exact replay, and it does not return transient optimizer,
 * activation, or recurrent runtime state.
 *
 * ```ts
 * // Export the current parameter vector, fine-tune a working copy, and
 * // inspect fitness metrics without modifying the shared candidate network.
 * const vector = toParameterVector(candidate);
 * const { trainedVector, metrics } = fineTuneVector(candidate, vector, dataset, {
 *   steps: 50,
 *   learningRate: 0.01,
 *   seed: 42,
 * });
 * console.log('training error:', metrics?.error);
 * // `candidate` and `vector` are unchanged after this call.
 * ```
 *
 * @param baseNetwork - Topology source cloned for the isolated working copy.
 * @param vector - Ordered parameter payload applied to the working copy only.
 * @param dataset - Ordered training samples consumed without shuffling.
 * @param options - Explicit training settings and optional deterministic seed.
 * @returns Detached trained vector plus numeric training metrics.
 */
export function fineTuneVector(
  baseNetwork: Network,
  vector: ParameterVector,
  dataset: TrainingSample[],
  options: FineTuneOptions,
): FineTuneResult {
  // Step 1: Clone the topology source so shared candidate state stays untouched.
  const workingCopy = baseNetwork.clone();

  // Step 2: Apply the supplied vector to the working copy only.
  fromParameterVector(workingCopy, vector);

  // Step 3: Install deterministic RNG ownership only when the caller provides a seed.
  if (options.seed !== undefined) {
    workingCopy.setSeed(options.seed);
  }

  // Step 4: Run the ordered training pass with explicit caller-provided settings.
  const trainingSummary = trainImpl(
    workingCopy,
    dataset,
    createFineTuneTrainingOptions(options),
  );

  // Step 5: Export the detached trained vector and numeric training metrics.
  return {
    trainedVector: toParameterVector(workingCopy),
    metrics: createFineTuneMetrics(trainingSummary),
  };
}

function createFineTuneTrainingOptions(options: FineTuneOptions): {
  iterations: number;
  rate: number;
} {
  return {
    iterations: options.steps,
    rate: options.learningRate,
  };
}

function createFineTuneMetrics(trainingSummary: {
  error: number;
  iterations: number;
  time: number;
}): Record<string, number> {
  return {
    error: trainingSummary.error,
    iterations: trainingSummary.iterations,
    time: trainingSummary.time,
  };
}
