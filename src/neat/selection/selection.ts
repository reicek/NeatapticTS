import type { NeatLike } from '../shared/neat.shared.types';
import {
  calculateTotalScore,
  DEFAULT_SCORE,
  ensurePopulationEvaluated,
  ensurePopulationSortedDescending,
  FIRST_INDEX,
  selectParentByStrategy,
} from './core/selection.core';
import type {
  GenomeWithScore,
  NeatLikeWithSelection,
} from './core/selection.types';

export {
  DEFAULT_POWER,
  DEFAULT_TOURNAMENT_SIZE,
  DEFAULT_TOURNAMENT_PROBABILITY,
  DEFAULT_SCORE,
  FIRST_INDEX,
  SECOND_INDEX,
  LAST_INDEX_OFFSET,
  LOOP_INDEX_INCREMENT,
  LAST_ELEMENT_INDEX,
  INITIAL_TOTAL_FITNESS,
  INITIAL_MOST_NEGATIVE_SCORE,
  INITIAL_CUMULATIVE_FITNESS,
} from './core/selection.core';

/**
 * Controller-facing selection helpers for the NEAT lifecycle.
 *
 * Selection is the point where a scored population becomes something the rest
 * of the controller can reason about. The helpers in this root chapter answer
 * the questions that show up most often during evolution work: which genome is
 * currently best, what is the current average score, should the population be
 * re-ordered before inspection, and which parent should breed next under the
 * active selection strategy.
 *
 * Keep the mental model in four steps:
 *
 * 1. summary helpers such as {@link getFittest} and {@link getAverage} make
 *    sure evaluation has happened before they report on the population,
 * 2. ordering helpers such as {@link sort} keep descending-score reads
 *    deterministic,
 * 3. {@link getParent} delegates the actual parent choice to `core/`, where
 *    POWER, FITNESS_PROPORTIONATE, and TOURNAMENT strategy mechanics live,
 * 4. `facade/` mirrors the stable `Neat` class entrypoints so callers can use
 *    these same behaviors without importing the lower-level module directly.
 *
 * That split matters because selection is where a NEAT controller turns raw
 * evaluation into search pressure. Once genomes have scores, the controller has
 * to answer two different practical questions without mixing them together:
 *
 * - inspection questions such as "who is winning right now?" and "what does
 *   the generation average look like?"
 * - breeding questions such as "which genome should parent the next child?"
 *
 * The public helpers stay readable by keeping those questions adjacent but not
 * collapsed into one overloaded routine.
 *
 * The re-exported constants in this file are the small tuning and traversal
 * anchors that make those behaviors predictable: fallback scores for
 * unevaluated genomes, default parameters for the built-in parent-selection
 * strategies, and explicit index sentinels for threshold scans and tournament
 * walks.
 *
 * It helps to read those constants as three compact families instead of as one
 * long shelf of numbers:
 *
 * - selection-pressure defaults such as {@link DEFAULT_POWER} and
 *   {@link DEFAULT_TOURNAMENT_SIZE} explain how strongly the built-in
 *   strategies lean toward front-running genomes,
 * - score semantics such as {@link DEFAULT_SCORE} explain how selection stays
 *   deterministic before or between evaluation passes,
 * - traversal sentinels such as {@link FIRST_INDEX} and
 *   {@link INITIAL_CUMULATIVE_FITNESS} keep the lower-level scans explicit and
 *   self-consistent.
 *
 * Read this root chapter when you want the controller story first. Drop into
 * `core/` when you need to understand the exact selection math, overflow rules,
 * or threshold scans. Read `facade/` when you are tracing how the public
 * `Neat` class exposes the same inspection helpers.
 *
 * ```mermaid
 * flowchart TD
 *   Population[Population with scores or pending evaluation]
 *   Summaries[Summary helpers<br/>getFittest / getAverage]
 *   Ordering[Ordering helper<br/>sort]
 *   ParentChoice[Parent helper<br/>getParent]
 *   Core[core/<br/>strategy mechanics]
 *   Facade[facade/<br/>stable Neat wrappers]
 *
 *   Population --> Summaries
 *   Population --> Ordering
 *   Population --> ParentChoice
 *   Summaries --> Ordering
 *   ParentChoice --> Core
 *   Ordering --> Facade
 *   Summaries --> Facade
 * ```
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Selection[Selection chapter]:::accent --> Inspection[Inspection reads]:::base
 *   Selection --> Breeding[Breeding read]:::base
 *   Selection --> Constants[Shared constant families]:::base
 *   Inspection --> Champion[getFittest / getAverage / sort]:::base
 *   Breeding --> Parent[getParent]:::base
 *   Constants --> Pressure[Strategy defaults]:::base
 *   Constants --> Fallbacks[Score fallback semantics]:::base
 *   Constants --> Traversal[Index and accumulator sentinels]:::base
 * ```
 *
 * Example:
 *
 * ```ts
 * neat.sort();
 * const champion = neat.getFittest();
 * const meanScore = neat.getAverage();
 * const parent = neat.getParent();
 * ```
 */

/**
 * Sort the internal population in place by descending fitness.
 *
 * Use this when later controller steps should read the population in explicit
 * best-first order. The helper applies the same fallback-score semantics used
 * elsewhere in selection, so genomes without a score are treated as if they had
 * {@link DEFAULT_SCORE} until evaluation supplies a real value.
 *
 * This helper is intentionally narrow: it only reorders the current population.
 * It does not evaluate genomes, mutate them, or change the active parent
 * selection strategy.
 *
 * @this NeatLike NEAT instance whose population should be sorted.
 * @returns Nothing. The population array is reordered in place.
 *
 * @example
 * ```ts
 * neat.sort();
 * const bestScore = neat.population[0]?.score;
 * ```
 */
export function sort(this: NeatLike): void {
  const internal = this as unknown as NeatLikeWithSelection;
  internal.population.sort(
    (left, right) =>
      (right.score ?? DEFAULT_SCORE) - (left.score ?? DEFAULT_SCORE),
  );
}

/**
 * Select a parent genome according to the configured selection strategy.
 *
 * This is the controller-facing gateway into the three built-in strategies:
 *
 * - `POWER` biases selection toward the front of the sorted population,
 * - `FITNESS_PROPORTIONATE` performs roulette-style sampling and shifts
 *   negative scores into a usable threshold space,
 * - `TOURNAMENT` samples a temporary bracket and walks it with the configured
 *   win probability.
 *
 * If the selection mode is unrecognized, the helper falls back to the first
 * genome in the current population so the controller still has a deterministic
 * parent candidate instead of failing deep inside crossover logic.
 *
 * @this NeatLike NEAT instance containing population, selection options, and RNG access.
 * @returns Genome chosen according to the active selection strategy.
 *
 * @example
 * ```ts
 * const parent = neat.getParent();
 * const strategyName = neat.options.selection?.name;
 * ```
 */
export function getParent(this: NeatLike): GenomeWithScore {
  const internal = this as unknown as NeatLikeWithSelection;

  // Step 1: Resolve parent selection via the core strategy helpers.
  return selectParentByStrategy(internal);
}

/**
 * Return the fittest genome in the population.
 *
 * This is the safest "show me the current champion" helper for controller code,
 * telemetry probes, and tests. If the population has not been evaluated yet,
 * the existing evaluation path is triggered first. If scores exist but the
 * population is out of descending order, the helper restores that order before
 * returning the leading genome.
 *
 * That behavior keeps call sites simple: callers do not need to remember
 * whether evaluation or sorting has already happened earlier in the generation.
 *
 * @this NeatLike NEAT instance containing population and evaluation support.
 * @returns Genome with the highest current score.
 *
 * @example
 * ```ts
 * const champion = neat.getFittest();
 * console.log(champion.score);
 * ```
 */
export function getFittest(this: NeatLike): GenomeWithScore {
  const internal = this as unknown as NeatLikeWithSelection;
  const population = internal.population;

  // Step 1: Ensure the population has scores.
  ensurePopulationEvaluated(internal);

  // Step 2: Ensure the population is ordered by descending score.
  ensurePopulationSortedDescending(internal);

  // Step 3: Return the best genome.
  return population[FIRST_INDEX];
}

/**
 * Compute the average fitness across the population.
 *
 * Use this when you want a coarse health signal for the whole generation rather
 * than the single best genome. The helper ensures evaluation has happened,
 * folds the total score across the full population, and returns the arithmetic
 * mean that telemetry, progress logging, and quick sanity checks usually need.
 *
 * Unlike parent selection, this helper does not care about order. It reports on
 * the population as a group, which makes it a convenient companion to
 * {@link getFittest} when you want both "best genome" and "overall generation"
 * signals side by side.
 *
 * @this NeatLike NEAT instance containing population and evaluation support.
 * @returns Mean fitness across the current population.
 *
 * @example
 * ```ts
 * const meanScore = neat.getAverage();
 * console.log(`Average score: ${meanScore}`);
 * ```
 */
export function getAverage(this: NeatLike): number {
  const internal = this as unknown as NeatLikeWithSelection;
  const population = internal.population;

  // Step 1: Ensure the population has scores.
  ensurePopulationEvaluated(internal);

  // Step 2: Fold total fitness into the arithmetic mean.
  const totalScore = calculateTotalScore(population);
  return totalScore / population.length;
}
