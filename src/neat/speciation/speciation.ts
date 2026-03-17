/**
 * Assign genomes into species and maintain the controller's species state.
 *
 * Speciation is the controller phase that keeps a NEAT population diverse
 * enough to keep exploring. Instead of letting one temporarily strong lineage
 * absorb the whole run, this boundary repeatedly answers three practical
 * questions:
 *
 * 1. which genomes still belong together under the current compatibility rule,
 * 2. whether the compatibility threshold should move to keep the species count
 *    healthy,
 * 3. which species should remain protected, penalized, or recorded for the next
 *    generation.
 *
 * The root file stays orchestration-first so callers can read the full
 * controller story in one place. The detailed mechanics then branch into the
 * helper chapters that own one responsibility each:
 *
 * - `assignment/` snapshots old memberships, clears species, reassigns genomes,
 *   and refreshes representatives,
 * - `threshold/` keeps the compatibility threshold near the intended species
 *   count,
 * - `history/` records teaching- and telemetry-friendly snapshots and applies
 *   age-based protection,
 * - `sharing/` normalizes within-species scores and tracks stagnation.
 *
 * Read this root chapter when you want the speciation lifecycle first. Drop
 * into the helper folders when you want the exact assignment heuristics,
 * threshold controller, or history bookkeeping.
 *
 * ```mermaid
 * flowchart TD
 *   Population[Population entering speciation]
 *   Snapshot[Snapshot previous memberships]
 *   Assignment[assignment/<br/>reset, match, create, refresh]
 *   Threshold[threshold/<br/>adapt compatibility threshold]
 *   History[history/<br/>protect and record]
 *   Sharing[sharing/<br/>share fitness and track stagnation]
 *   SpeciesState[Updated species registry]
 *
 *   Population --> Snapshot
 *   Snapshot --> Assignment
 *   Assignment --> Threshold
 *   Threshold --> History
 *   History --> SpeciesState
 *   SpeciesState --> Sharing
 * ```
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
 * radius and delegates to `sharing/`, where each genome's score contribution is
 * softened according to how crowded its neighborhood is.
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
