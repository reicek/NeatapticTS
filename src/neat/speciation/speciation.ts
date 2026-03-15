/**
 * Assign genomes into species based on compatibility distance and maintain species structures.
 *
 * This root entrypoint stays orchestration-first so the generated README reads
 * like the speciation lifecycle: snapshot the previous state, reassign the
 * population, tune the threshold, refresh representatives, apply optional
 * protection, and capture history. The detailed mechanics now live in focused
 * teaching folders for assignment, threshold control, history, and sharing.
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
 * @param this - Speciation harness context.
 * @returns Nothing.
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
 * @param this - Neat instance context with species array and compatibility distance function.
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
 * @param species - Species to sort.
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
  interface OptionsWithStagnation {
    stagnationGenerations?: number;
  }
  const stagnationWindow =
    (this.options as OptionsWithStagnation).stagnationGenerations ??
    DEFAULT_STAGNATION_WINDOW;

  // Step 1: Update per-species stagnation metrics and prune survivors.
  updateSpeciesStagnation(this, stagnationWindow, _sortSpeciesMembers);
}
