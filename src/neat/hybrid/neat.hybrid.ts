/**
 * Hybrid evaluation joins deterministic parameter vectors, isolated gradient
 * updates, and policy-aware scoring into one explicit workflow.
 *
 * This boundary is where you decide whether learning is only a temporary aid
 * for scoring or a change that should propagate back into the population. In
 * this module, "Lamarckian" means trained weights are written back to the live
 * candidate, either by calling `fromParameterVector(...)` yourself or by using
 * `persistTrainedWeights: true` with `evaluateCandidate(...)`. Fitness-only
 * scoring is closer to the Baldwin effect: learning changes the phenotype that
 * gets scored, but the genotype that remains in the population is unchanged.
 * See Wikipedia contributors,
 * [Lamarckism](https://en.wikipedia.org/wiki/Lamarckism), and Wikipedia
 * contributors,
 * [Baldwin effect](https://en.wikipedia.org/wiki/Baldwin_effect), for the
 * historical vocabulary behind that distinction.
 *
 * The determinism promise is intentionally laddered instead of absolute:
 *
 * 1. Same-runtime ordered deterministic when topology, dataset order, training
 *    settings, and an explicit `seed` all match.
 * 2. Best-effort reproducible when dataset order and settings match but no
 *    `seed` is supplied, so stochastic training features may still drift.
 * 3. Cross-runtime exact replay is out of scope for this lane; matching
 *    worker, browser, and Node results still depends on broader replay
 *    contracts.
 *
 * Recommended progression:
 *
 * 1. Start with `fineTune: 'never'` to measure the pure evolutionary
 *    baseline.
 * 2. Move to `fineTune: 'always'` with `persistTrainedWeights: false` when you
 *    want exploratory scoring without mutating the population.
 * 3. Opt into `persistTrainedWeights: true` only when Lamarckian
 *    carry-forward is an explicit design choice.
 *
 * `fineTune: 'conditional'` remains blocked until the evaluation boundary
 * exposes a deterministic ranking or tie-break rule. Without that ordering
 * contract, "train only the best candidates" is ambiguous.
 *
 * @example
 * ```ts
 * import {
 *   Network,
 *   evaluateCandidate,
 *   fineTuneVector,
 *   fromParameterVector,
 *   toParameterVector,
 * } from 'neataptic';
 *
 * const candidate = new Network(2, 1, { seed: 42 });
 * const dataset = [
 *   { input: [0, 0], output: [0] },
 *   { input: [0, 1], output: [1] },
 *   { input: [1, 0], output: [1] },
 *   { input: [1, 1], output: [0] },
 * ];
 * const fineTuneOptions = { steps: 50, learningRate: 0.05, seed: 123 };
 *
 * // Inspect a detached trained vector first.
 * const baselineVector = toParameterVector(candidate);
 * const { trainedVector, metrics } = fineTuneVector(
 *   candidate,
 *   baselineVector,
 *   dataset,
 *   fineTuneOptions,
 * );
 * console.log('training error:', metrics?.error);
 *
 * // Materialize the trained variant on a clone so the live candidate stays untouched.
 * const inspectedCandidate = candidate.clone();
 * fromParameterVector(inspectedCandidate, trainedVector);
 *
 * // Score the trained variant without persisting it back to the population.
 * const fitnessOnly = await evaluateCandidate(candidate, dataset, {
 *   policy: { fineTune: 'always', persistTrainedWeights: false },
 *   fineTuneOptions,
 *   scoreNetwork: (network) => -network.test(dataset, { cost: 'mse' }).error,
 * });
 * console.log('fitness-only score:', fitnessOnly.fitness);
 *
 * // Opt into Lamarckian persistence only when you intentionally want it.
 * if (fitnessOnly.fitness > -0.05) {
 *   fromParameterVector(candidate, trainedVector);
 * }
 * ```
 */
import type Network from '../../architecture/network';
import {
  fromParameterVector,
  toParameterVector,
  type ParameterVector,
} from '../../architecture/network/serialize/network.serialize.utils';
import { fineTuneVector } from '../../architecture/network/training/network.training.isolate.utils';
import type { TrainingSample } from '../../architecture/network/training/network.training.utils.types';
import type {
  EvaluateCandidateOptions,
  HybridEvaluationResult,
} from './neat.hybrid.types';

/**
 * Evaluate one candidate under an explicit hybrid fine-tune policy.
 *
 * The helper keeps policy decisions visible: `fineTune` controls whether
 * training runs, `scoreNetwork` decides how the selected network state is
 * scored, and `persistTrainedWeights` controls whether the trained vector is
 * applied back to the original candidate after a successful score.
 * `persistTrainedWeights: false` is the safe default because the trained
 * variant is otherwise discarded after scoring.
 *
 * `conditional` remains blocked until the NEAT evaluation surface exposes a
 * deterministic ranking or tie-break contract. This helper inherits the
 * same-runtime ordered determinism limits of `fineTuneVector(...)`; it does
 * not claim cross-runtime exact replay.
 *
 * ```ts
 * import { Network, evaluateCandidate } from 'neataptic';
 *
 * const network = new Network(2, 1);
 * const dataset = [
 *   { input: [0, 0], output: [0] },
 *   { input: [1, 1], output: [0] },
 *   { input: [1, 0], output: [1] },
 *   { input: [0, 1], output: [1] },
 * ];
 *
 * // Fitness-only: fine-tune a detached clone, score it, discard trained weights.
 * const fitnessOnly = await evaluateCandidate(network, dataset, {
 *   policy: { fineTune: 'always', persistTrainedWeights: false },
 *   fineTuneOptions: { steps: 50, learningRate: 0.01, seed: 42 },
 *   scoreNetwork: (candidate) => -candidate.test(dataset, { cost: 'mse' }).error,
 * });
 * console.log('fitness:', fitnessOnly.fitness); // original candidate unchanged
 *
 * // Lamarckian: apply trained weights back to the candidate on explicit opt-in.
 * const lamarckian = await evaluateCandidate(network, dataset, {
 *   policy: { fineTune: 'always', persistTrainedWeights: true },
 *   fineTuneOptions: { steps: 50, learningRate: 0.01, seed: 42 },
 *   scoreNetwork: (candidate) => -candidate.test(dataset, { cost: 'mse' }).error,
 * });
 * console.log('fitness after persistence:', lamarckian.fitness);
 * ```
 *
 * @param network - Live candidate selected by the caller's fitness delegate.
 * @param dataset - Ordered training samples passed to `fineTuneVector(...)` when training runs.
 * @param options - Explicit policy, training settings, and scoring callback.
 * @returns Fitness plus the detached trained network when fine-tuning runs.
 * @throws {Error} When fineTune policy is "conditional" (blocked until deterministic ranking surface exists).
 */
export async function evaluateCandidate(
  network: Network,
  dataset: TrainingSample[],
  options: EvaluateCandidateOptions,
): Promise<HybridEvaluationResult> {
  const fineTuneMode = options.policy.fineTune;

  // Step 1: Route the no-fine-tune path without touching vector or training helpers.
  if (fineTuneMode === 'never') {
    return {
      fitness: await options.scoreNetwork(network),
    };
  }

  // Step 2: Reject the unresolved conditional path until a deterministic ranking seam exists.
  if (fineTuneMode === 'conditional') {
    throw new Error(
      'HybridEvaluationPolicy fineTune="conditional" is blocked until a deterministic ranking surface exists.',
    );
  }

  // Step 3: Fine-tune a detached vector and materialize the trained network used for scoring.
  const trainedVector = createTrainedVector();
  const trainedNetwork = createTrainedNetwork(trainedVector);

  // Step 4: Score the detached trained variant chosen by the policy.
  const fitness = await options.scoreNetwork(trainedNetwork);

  // Step 5: Persist trained weights only on explicit Lamarckian opt-in.
  if (options.policy.persistTrainedWeights === true) {
    fromParameterVector(network, trainedVector);
  }

  return {
    fitness,
    trainedNetwork,
  };

  function createTrainedVector(): ParameterVector {
    const baselineVector = toParameterVector(network);
    const fineTuneResult = fineTuneVector(
      network,
      baselineVector,
      dataset,
      readRequiredFineTuneOptions(),
    );

    return fineTuneResult.trainedVector;
  }

  function createTrainedNetwork(trainedVector: ParameterVector): Network {
    const trainedNetwork = network.clone();
    fromParameterVector(trainedNetwork, trainedVector);
    return trainedNetwork;
  }

  function readRequiredFineTuneOptions(): NonNullable<
    EvaluateCandidateOptions['fineTuneOptions']
  > {
    if (options.fineTuneOptions == null) {
      throw new Error(
        'evaluateCandidate requires fineTuneOptions when HybridEvaluationPolicy.fineTune is not "never".',
      );
    }

    return options.fineTuneOptions;
  }
}
