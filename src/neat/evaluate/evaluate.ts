/**
 * Root orchestration for NEAT population evaluation.
 *
 * ## What Evaluation Means in NEAT
 *
 * Evaluation is the grounding step that connects the evolutionary search to the
 * real task. Without it, NEAT has genomes and topologies but no evidence about
 * which ones are worth keeping. With it, the controller can sort, speciate,
 * apply adaptive pressure, and breed the next generation from a well-evidenced
 * view of the current population.
 *
 * In NEAT, the fitness function is deliberately kept external to the library.
 * The caller supplies a scoring delegate — anything from a physics simulation
 * to an analytic reward function — and the evaluation layer calls it once per
 * genome (or hands the whole population to a parallel worker batch). That
 * separation keeps the library domain-agnostic while letting each caller define
 * what "good behavior" means for their specific task.
 *
 * ## Beyond Raw Fitness
 *
 * Pure fitness maximization can be a poor proxy for the goal of finding
 * behaviorally diverse solutions. This evaluation boundary supports two
 * additional evidence sources that can supplement or replace raw fitness:
 *
 * - **Novelty search** — scores genomes by how different their behavioral
 *   *descriptor* is from previously encountered behaviors, rather than by task
 *   performance. Useful when the fitness landscape is deceptive or has many
 *   dead-end attractors. See Lehman and Stanley,
 *   [Abandoning Objectives: Evolution Through the Search for Novelty Alone](http://eplex.cs.ucf.edu/papers/lehman_ecj11.pdf),
 *   for the original motivation.
 * - **Multi-objective ranking** — instead of one scalar fitness, multiple
 *   objectives are tracked simultaneously and genomes are ranked by Pareto
 *   dominance. The evaluation layer injects an entropy-diversity objective when
 *   multi-objective mode is active.
 *
 * ## The Six-Stage Evaluation Pass
 *
 * The full evaluation pass runs these stages in order:
 *
 * 1. Run the configured fitness pathway (per-genome or whole-population).
 * 2. Blend novelty evidence and maintain the novelty archive when enabled.
 * 3. Ensure diversity-stat storage exists for downstream tuning.
 * 4. Tune entropy-sharing and compatibility parameters from fresh evidence.
 * 5. Refresh lightweight speciation state.
 * 6. Register the entropy objective when multi-objective evaluation is active.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:1.5px;
 *   classDef accent fill:#0f1f33,stroke:#00e5ff,color:#d8f6ff,stroke-width:2px;
 *
 *   Fitness["1 · Run fitness delegate\nper-genome or whole-population"]:::accent
 *   Novelty["2 · Blend novelty\nupdate archive if enabled"]:::base
 *   Stats["3 · Ensure diversity stats container"]:::base
 *   Sharing["4 · Tune entropy sharing and\ncompatibility coefficients"]:::base
 *   Speciation["5 · Refresh lightweight speciation state"]:::base
 *   Objectives["6 · Inject entropy objective\nif multi-objective mode active"]:::base
 *
 *   Fitness --> Novelty --> Stats --> Sharing --> Speciation --> Objectives
 * ```
 *
 * Reading order:
 * - start with {@link evaluate} for the controller-facing flow,
 * - jump into `fitness/` for the per-genome vs whole-population scoring split,
 * - jump into `novelty/` for descriptor and archive semantics,
 * - jump into `entropy-sharing/`, `entropy-compat/`, and `auto-distance/` for tuning math,
 * - jump into `objectives/` for the automatic entropy-objective path.
 *
 * The exported constants fall into four tuning families:
 * - novelty defaults for descriptor-based exploration,
 * - entropy-sharing defaults for diversity-distribution control,
 * - compatibility defaults for speciation pressure,
 * - distance-coefficient defaults for automatic structural-distance balancing.
 */
import {
  ensureDiversityStatsContainer,
  runEntropySharingTuning,
} from './entropy-sharing/evaluate.entropy-sharing';
import { runAutoDistanceCoefficientTuning } from './auto-distance/evaluate.auto-distance';
import { runEntropyCompatibilityTuning } from './entropy-compat/evaluate.entropy-compat';
import { runFitnessEvaluation } from './fitness/evaluate.fitness';
import { runNoveltyBlendAndArchive } from './novelty/evaluate.novelty';
import { runAutoEntropyObjectiveInjection } from './objectives/evaluate.objectives';
import { runLightweightSpeciation } from './speciation/evaluate.speciation';
import type { NeatControllerForEval } from './shared/evaluate.types';

export {
  NOVELTY_DEFAULT_NEIGHBORS,
  NOVELTY_DEFAULT_BLEND,
  NOVELTY_ARCHIVE_CAP,
  ENTROPY_VAR_TARGET_DEFAULT,
  ENTROPY_VAR_ADJUST_DEFAULT,
  ENTROPY_VAR_MIN_SIGMA_DEFAULT,
  ENTROPY_VAR_MAX_SIGMA_DEFAULT,
  ENTROPY_VAR_LOW_BAND,
  ENTROPY_VAR_HIGH_BAND,
  ENTROPY_TARGET_DEFAULT,
  ENTROPY_DEADBAND_DEFAULT,
  ENTROPY_ADJUST_DEFAULT,
  COMPAT_THRESHOLD_DEFAULT,
  COMPAT_MIN_THRESHOLD_DEFAULT,
  COMPAT_MAX_THRESHOLD_DEFAULT,
  AUTO_COEFF_ADJUST_DEFAULT,
  AUTO_COEFF_MIN_DEFAULT,
  AUTO_COEFF_MAX_DEFAULT,
  DISTANCE_COEFF_DEFAULT,
  VARIANCE_DECREASE_THRESHOLD,
  VARIANCE_INCREASE_THRESHOLD,
} from './shared/evaluate.constants';
export type { NeatControllerForEval } from './shared/evaluate.types';

/**
 * Evaluate the population or population-wide fitness delegate.
 *
 * This is the controller-facing scoring pass that prepares one generation for every downstream
 * decision the NEAT runtime will make next. Unlike `evolve()`, this method does not replace the
 * population or build offspring. Its job is narrower and just as important: establish current
 * scores, enrich them with optional novelty evidence, and refresh the adaptive statistics that the
 * next orchestration step will rely on.
 *
 * The flow has two major halves:
 *
 * 1. produce evidence about the current population,
 * 2. update lightweight controller state that depends on that evidence.
 *
 * In practice that means:
 * 1. resolve whether fitness runs once per genome or once for the entire population,
 * 2. optionally clear per-genome runtime state before scoring,
 * 3. optionally blend novelty search into the resulting scores,
 * 4. make sure diversity-stat storage exists before tuning begins,
 * 5. run the small adaptive tuning passes that respond to the freshly observed population,
 * 6. optionally register the entropy objective for later multi-objective ranking.
 *
 * A useful mental model is that `evaluate()` prepares the evidence layer, while `evolve()` later
 * consumes that evidence to rank, speciate, and rebuild the population. If evaluation is stale or
 * skipped, the rest of the lifecycle has less trustworthy data to work from.
 *
 * Important side effects:
 * - updates genome scores in place,
 * - may update novelty values and append to the novelty archive,
 * - ensures `_diversityStats` exists before tuning helpers write into it,
 * - may adjust sharing sigma, compatibility thresholds, and distance coefficients,
 * - may refresh lightweight speciation state and register the entropy objective.
 *
 * Read the neighboring chapters like this:
 * - `fitness/` explains how the scoring delegate is actually invoked,
 * - `novelty/` explains the descriptor-distance path and archive writes,
 * - `speciation/` explains the lightweight maintenance that can happen after scores land,
 * - `objectives/` explains why entropy objective registration lives in evaluation instead of evolve.
 *
 * @returns Promise that resolves after evaluation and adaptive follow-up steps.
 *
 * @example
 * ```ts
 * await evaluate.call(controller);
 *
 * // the population is now freshly scored and ready for selection or evolve()
 * const bestScore = Math.max(...controller.population.map((genome) => genome.score ?? 0));
 * console.log('best score after evaluation:', bestScore);
 * ```
 */
export async function evaluate(this: NeatControllerForEval): Promise<void> {
  const evaluationOptions = this.options || {};

  // Step 1: Score the population through the configured fitness delegate.
  await runFitnessEvaluation(this, evaluationOptions);

  // Step 2: Blend novelty and archive descriptors when novelty search is active.
  runNoveltyBlendAndArchive(this, evaluationOptions);

  // Step 3: Ensure downstream tuning has a diversity container to write into.
  ensureDiversityStatsContainer(this);

  // Step 4: Apply the lightweight adaptive post-evaluation maintenance steps.
  runEntropySharingTuning(this, evaluationOptions);
  runEntropyCompatibilityTuning(this, evaluationOptions);
  runLightweightSpeciation(this, evaluationOptions);
  runAutoDistanceCoefficientTuning(this, evaluationOptions);
  runAutoEntropyObjectiveInjection(this, evaluationOptions);
}

export default { evaluate };
