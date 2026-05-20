import type Network from '../../architecture/network';
import type { FineTuneOptions } from '../../architecture/network/training/network.training.isolate.utils';

/**
 * Explicit fine-tune trigger for one hybrid evaluation pass.
 *
 * - `never` evaluates the live candidate as-is without calling any training helper.
 * - `always` fine-tunes a detached clone before scoring; whether trained weights
 *   persist back is controlled separately by `HybridEvaluationPolicy.persistTrainedWeights`.
 * - `conditional` requires a deterministic ranking or tie-break contract at the
 *   evaluation boundary before it can be unblocked. Passing `conditional` to
 *   `evaluateCandidate` throws at runtime until that surface exists.
 */
export type HybridFineTuneMode = 'never' | 'always' | 'conditional';

/**
 * Explicit hybrid policy for one candidate evaluation.
 *
 * Recommended progression keeps the policy easy to reason about: start with
 * `fineTune: 'never'` to measure the evolutionary baseline, move to
 * `fineTune: 'always'` with `persistTrainedWeights: false` for fitness-only
 * Baldwin-style scoring, and opt into `persistTrainedWeights: true` only when
 * deliberate Lamarckian carry-forward is part of the experiment. `conditional`
 * remains blocked until the caller can supply a deterministic ranking or
 * tie-break contract.
 *
 * Persistence stays separate from the fine-tune trigger so fitness-only
 * scoring remains the safe default. Callers should keep
 * `persistTrainedWeights` false unless they explicitly want Lamarckian
 * persistence after a successful score.
 */
export interface HybridEvaluationPolicy {
  /** Fine-tune trigger for this evaluation pass. */
  fineTune: HybridFineTuneMode;
  /** Explicit Lamarckian opt-in. `false` is the safe default. */
  persistTrainedWeights: boolean;
}

/**
 * Scoring callback used after the helper resolves the network state to score.
 *
 * The callback may be synchronous or async, but it should treat the supplied
 * network as the exact candidate state selected by the hybrid policy.
 */
export type HybridScoreNetwork = (
  candidate: Network,
) => number | Promise<number>;

/**
 * Inputs for one standalone hybrid candidate evaluation.
 *
 * `fineTuneOptions` is required whenever `policy.fineTune !== 'never'`
 * because this helper delegates training to `fineTuneVector(...)` rather than
 * guessing learning settings.
 */
export interface EvaluateCandidateOptions {
  /** Explicit fine-tune and persistence policy for this pass. */
  policy: HybridEvaluationPolicy;
  /** Training settings forwarded only when fine-tuning runs. */
  fineTuneOptions?: FineTuneOptions;
  /** Caller-owned scoring function for the chosen network state. */
  scoreNetwork: HybridScoreNetwork;
}

/**
 * Result from one hybrid candidate evaluation pass.
 *
 * `trainedNetwork` is present only when fine-tuning runs. Fitness-only callers
 * can inspect the detached trained variant without mutating the canonical
 * candidate, while Lamarckian callers receive the same trained snapshot that
 * was scored before explicit persistence is applied.
 *
 * Typical downstream uses include: forwarding `fitness` to the NEAT population
 * score, comparing `trainedNetwork` weights against the original candidate to
 * measure fine-tune delta, checkpointing the trained snapshot, or discarding
 * the result entirely when only the fitness score matters.
 */
export interface HybridEvaluationResult {
  /** Fitness score returned by the caller-owned scoring callback. */
  fitness: number;
  /** Detached trained network used for scoring when fine-tuning runs. */
  trainedNetwork?: Network;
}
