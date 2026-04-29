/**
 * Assign genomes into species and maintain the controller's species state.
 *
 * ## Why Speciation Exists
 *
 * In a purely competitive population, a genome with a temporarily strong
 * topology would rapidly clone itself and crowd out every structurally different
 * solution. New topologies — which often need several generations to tune their
 * weights — would be eliminated before they had a chance to prove useful. NEAT
 * solves this with *speciation*: genomes are grouped by structural similarity,
 * and fitness is shared within each group rather than globally. A new topology
 * that scores below the population average still earns enough shared fitness to
 * survive long enough to refine itself.
 *
 * This idea parallels niche preservation in natural evolution, where
 * ecological niches let different strategies coexist even when one currently
 * dominates on a shared metric. See Wikipedia contributors,
 * [Fitness sharing](https://en.wikipedia.org/wiki/Fitness_sharing), and
 * Stanley and Miikkulainen,
 * [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/?stanley:ec02),
 * for the original motivation and formal definition.
 *
 * ## Compatibility Distance
 *
 * Two genomes belong to the same species when their *compatibility distance* δ
 * falls below the current threshold:
 *
 * ```
 * δ = (c₁ · E) / N  +  (c₂ · D) / N  +  c₃ · W̄
 * ```
 *
 * E = excess gene count, D = disjoint gene count, N = larger genome size,
 * W̄ = mean weight difference of matching genes, c₁/c₂/c₃ = tunable
 * coefficients. High δ means the genomes represent structurally different
 * evolutionary lineages.
 *
 * ## Lifecycle
 *
 * This boundary repeatedly answers three practical questions per generation:
 *
 * 1. Which genomes still belong together under the current compatibility rule?
 * 2. Should the compatibility threshold move to keep the species count healthy?
 * 3. Which species should be protected, penalized, or pruned for the next round?
 *
 * The root file stays orchestration-first. The detailed mechanics live in four
 * helper chapters:
 *
 * - `assignment/` — snapshot old memberships, clear, reassign, refresh representatives,
 * - `threshold/` — adaptively tune δ to keep the species count near a target,
 * - `history/` — record teaching-friendly snapshots and apply age-based protection,
 * - `sharing/` — normalize scores within species and track stagnation.
 *
 * The two-phase ownership matters: assignment and threshold tuning establish
 * canonical species identity (which determines crossover alignment and replay
 * semantics). The history and sharing stages are controller-policy overlays
 * applied *after* grouping; they rewrite fitness views and bookkeeping without
 * changing identity or innovation-number alignment.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:1.5px;
 *   classDef accent fill:#0f1f33,stroke:#00e5ff,color:#d8f6ff,stroke-width:2px;
 *
 *   Population["Population entering speciation"]:::accent
 *   Snapshot["Snapshot previous memberships"]:::base
 *   Assignment["assignment/<br/>reset · match · create · refresh"]:::base
 *   Threshold["threshold/<br/>adapt compatibility threshold"]:::base
 *   History["history/<br/>protect young species · record snapshot"]:::base
 *   Sharing["sharing/<br/>share fitness · track stagnation"]:::base
 *   SpeciesState["Updated species registry"]:::accent
 *
 *   Population --> Snapshot
 *   Snapshot --> Assignment
 *   Assignment --> Threshold
 *   Threshold --> History
 *   History --> SpeciesState
 *   SpeciesState --> Sharing
 * ```
 *
 * Read this root chapter when you want the speciation lifecycle as a whole.
 * Drop into the helper folders for the exact assignment heuristics, threshold
 * controller, or history bookkeeping.
 *
 * Example:
 *
 * ```ts
 * neat._speciate();
 * neat._applyFitnessSharing();
 * neat._updateSpeciesStagnation();
 * ```
 */
import type {
  NeatLike,
  GenomeDetailed,
  SpeciesLike,
  SpeciationOptions,
  SpeciationHarnessContext,
} from '../shared/neat.shared.types';
import {
  DEFAULT_MAX_COMPATIBILITY_THRESHOLD,
  DEFAULT_MIN_COMPATIBILITY_THRESHOLD,
  DEFAULT_SCORE_FALLBACK,
  DEFAULT_SHARING_SIGMA,
  DEFAULT_STAGNATION_WINDOW,
  type CompatAdjust,
} from './shared/speciation.shared';
import {
  assignPopulationToSpecies,
  refreshSpeciesRepresentatives,
  resetSpeciesMembers,
  snapshotPreviousMembers,
} from './assignment/speciation.assignment.utils';
import {
  applyAgeProtection,
  recordHistory,
  trimHistory,
} from './history/speciation.history.utils';
import {
  applyFitnessSharing,
  updateSpeciesStagnation,
} from './sharing/speciation.sharing.utils';
import { adjustCompatibilityThreshold } from './threshold/speciation.threshold.utils';

/**
 * Assign genomes into species based on compatibility distance.
 *
 * This is the main controller-facing speciation pass. It preserves the previous
 * species snapshot for telemetry and history, rebuilds memberships against the
 * current representatives, adjusts the compatibility threshold, refreshes the
 * live representatives, applies optional age-based protection, records a new
 * history row, and trims the history buffer.
 *
 * In other words, this helper does not merely "cluster genomes." It keeps the
 * long-lived species registry coherent across generations so later phases such
 * as selection, pruning, telemetry, and archive inspection can reason about a
 * stable notion of species identity.
 *
 * @param this - Speciation harness context.
 * @returns Nothing.
 *
 * @example
 * ```ts
 * neat._speciate();
 * console.log(neat._species.length);
 * ```
 */
export function _speciate<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(this: SpeciationHarnessContext<TOptions>) {
  const options = this.options;
  const compatAdjust: CompatAdjust = options.compatAdjust ?? {};
  const minThreshold =
    compatAdjust.minThreshold ??
    options.minThreshold ??
    DEFAULT_MIN_COMPATIBILITY_THRESHOLD;
  const maxThreshold =
    compatAdjust.maxThreshold ??
    options.maxThreshold ??
    DEFAULT_MAX_COMPATIBILITY_THRESHOLD;

  // Step 1: Snapshot current memberships for telemetry.
  snapshotPreviousMembers(this);
  // Step 2: Clear members and reassign population.
  resetSpeciesMembers(this);
  assignPopulationToSpecies(this, options);
  // Step 3: Update adaptive compatibility threshold.
  adjustCompatibilityThreshold(
    this,
    options,
    compatAdjust,
    minThreshold,
    maxThreshold,
  );
  // Step 4: Prune and refresh representatives.
  refreshSpeciesRepresentatives(this);
  // Step 5: Apply age-based protection penalties.
  applyAgeProtection(this, options);
  // Step 6: Record history snapshot.
  recordHistory(this, options);
  // Step 7: Trim history buffer.
  trimHistory(this);
}

/**
 * Apply fitness sharing to penalize similarity within species.
 *
 * Use this after species assignment when a dense cluster should no longer keep
 * all of its raw score advantage. The helper resolves the configured sharing
 * radius and delegates to `sharing/`, where each genome's current score field is
 * rewritten as a controller-side shared-fitness overlay according to how
 * crowded its neighborhood is. The underlying evaluated score evidence remains
 * upstream of this pass.
 *
 * This is intentionally separate from {@link _speciate}. Some runs want species
 * bookkeeping without immediately renormalizing scores, while others use
 * sharing as a deliberate second pass after assignment has stabilized.
 *
 * @param this - Neat instance context with species array and compatibility distance function.
 *
 * @example
 * ```ts
 * neat._speciate();
 * neat._applyFitnessSharing();
 * ```
 */
export function _applyFitnessSharing(
  this: NeatLike & {
    _species: SpeciesLike[];
    _compatibilityDistance: (a: GenomeDetailed, b: GenomeDetailed) => number;
  },
) {
  interface OptionsWithSharing {
    sharingSigma?: number;
  }
  const sharingSigma =
    (this.options as OptionsWithSharing).sharingSigma ?? DEFAULT_SHARING_SIGMA;

  // Step 1: Apply the configured sharing strategy.
  applyFitnessSharing(this, sharingSigma);
}

/**
 * Sort species members by descending score.
 *
 * This small helper keeps species-local ranking deterministic before best-score
 * reads, stagnation updates, or representative decisions depend on member
 * order. Missing scores fall back to the shared speciation score sentinel so
 * unevaluated members do not break the ordering step.
 *
 * Even though the implementation is tiny, the helper exists as a named boundary
 * because multiple speciation flows need the same ordering rule.
 *
 * @param species - Species to sort.
 *
 * @example
 * ```ts
 * neat._sortSpeciesMembers(neat._species[0]);
 * ```
 */
export function _sortSpeciesMembers(species: SpeciesLike) {
  (species.members as GenomeDetailed[]).sort(
    (a, b) =>
      (b.score || DEFAULT_SCORE_FALLBACK) - (a.score || DEFAULT_SCORE_FALLBACK),
  );
}

/**
 * Update stagnation counters for all species.
 *
 * This is the maintenance pass that decides whether a species is still making
 * progress. It resolves the configured stagnation window, sorts each species so
 * "best member" comparisons are stable, and then delegates to `sharing/` to
 * mark stagnant species and prune them when necessary.
 *
 * The resulting `bestScore` and `lastImproved` values belong to controller-side
 * species bookkeeping, not to the canonical genome contract.
 *
 * Read this together with {@link _applyFitnessSharing} if you want the
 * post-assignment story: one helper reduces the dominance of crowded species,
 * and the other decides whether a species has stopped earning its place in the
 * run.
 *
 * @param this - Neat instance context with species array and generation counter.
 *
 * @example
 * ```ts
 * neat._speciate();
 * neat._updateSpeciesStagnation();
 * ```
 */
export function _updateSpeciesStagnation(
  this: NeatLike & { _species: SpeciesLike[]; generation: number },
) {
  interface OptionsWithStagnation {
    stagnationGenerations?: number;
  }
  const stagnationWindow =
    (this.options as OptionsWithStagnation).stagnationGenerations ??
    DEFAULT_STAGNATION_WINDOW;

  // Step 1: Update per-species stagnation metrics and prune survivors.
  updateSpeciesStagnation(this, stagnationWindow, _sortSpeciesMembers);
}
