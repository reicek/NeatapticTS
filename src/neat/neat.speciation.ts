/**
 * Assign genomes into species based on compatibility distance and maintain species structures.
 * This function creates new species for unassigned genomes, prunes empty species, updates
 * dynamic compatibility threshold controllers, performs optional auto coefficient tuning, and
 * records per‑species history statistics used by telemetry and adaptive controllers.
 *
 * Implementation notes:
 */
import type {
  NeatLike,
  GenomeDetailed,
  SpeciesLike,
  SpeciationOptions,
  SpeciationHarnessContext,
} from './neat.types';
import {
  applyAgeProtection,
  assignPopulationToSpecies,
  adjustCompatibilityThreshold,
  applyFitnessSharing,
  DEFAULT_MAX_COMPATIBILITY_THRESHOLD,
  DEFAULT_MIN_COMPATIBILITY_THRESHOLD,
  DEFAULT_SCORE_FALLBACK,
  DEFAULT_SHARING_SIGMA,
  DEFAULT_STAGNATION_WINDOW,
  refreshSpeciesRepresentatives,
  recordHistory,
  resetSpeciesMembers,
  snapshotPreviousMembers,
  trimHistory,
  updateSpeciesStagnation,
  type CompatAdjust,
} from './neat.speciation.utils';

/**
 * Assign genomes into species based on compatibility distance.
 *
 * @param this - Speciation harness context.
 * @returns Nothing.
 */
export function _speciate<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(this: SpeciationHarnessContext<TOptions>) {
  const options = this.options;
  const speciationContext = this;
  const compatAdjust: CompatAdjust = options.compatAdjust ?? {};
  const minThreshold =
    compatAdjust.minThreshold ??
    options.minThreshold ??
    DEFAULT_MIN_COMPATIBILITY_THRESHOLD;
  const maxThreshold =
    compatAdjust.maxThreshold ??
    options.maxThreshold ??
    DEFAULT_MAX_COMPATIBILITY_THRESHOLD;

  // 1) Snapshot current memberships for telemetry.
  snapshotPreviousMembers(speciationContext);
  // 2) Clear members and reassign population.
  resetSpeciesMembers(speciationContext);
  assignPopulationToSpecies(speciationContext, options);
  // 3) Update adaptive compatibility threshold.
  adjustCompatibilityThreshold(
    speciationContext,
    options,
    compatAdjust,
    minThreshold,
    maxThreshold,
  );
  // 4) Prune and refresh representatives.
  refreshSpeciesRepresentatives(speciationContext);
  // 5) Apply age-based protection penalties.
  applyAgeProtection(speciationContext, options);
  // 6) Record history snapshot.
  recordHistory(speciationContext, options);
  // 7) Trim history buffer.
  trimHistory(speciationContext);
}

/**
 * Apply fitness sharing to penalize similarity within species.
 *
 * @param this - Neat instance context with species array and compatibility distance function.
 */
export function _applyFitnessSharing(
  this: NeatLike & {
    _species: SpeciesLike[];
    _compatibilityDistance: (a: GenomeDetailed, b: GenomeDetailed) => number;
  },
) {
  const speciationContext = this;
  interface OptionsWithSharing {
    sharingSigma?: number;
  }
  const sharingSigma =
    (this.options as OptionsWithSharing).sharingSigma ?? DEFAULT_SHARING_SIGMA;

  // 1) Apply the configured sharing strategy.
  applyFitnessSharing(speciationContext, sharingSigma);
}

/**
 * Sort species members by descending score.
 *
 * @param this - Neat instance context.
 * @param sp - Species to sort.
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
 * @param this - Neat instance context with species array and generation counter.
 */
export function _updateSpeciesStagnation(
  this: NeatLike & { _species: SpeciesLike[]; generation: number },
) {
  const speciationContext = this;
  interface OptionsWithStagnation {
    stagnationGenerations?: number;
  }
  const stagnationWindow =
    (this.options as OptionsWithStagnation).stagnationGenerations ??
    DEFAULT_STAGNATION_WINDOW;

  // 1) Update per-species stagnation metrics and prune survivors.
  updateSpeciesStagnation(
    speciationContext,
    stagnationWindow,
    _sortSpeciesMembers,
  );
}
