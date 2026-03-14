/**
 * Shared vocabulary and defaults for NEAT speciation.
 *
 * This chapter defines the common runtime contracts used across assignment,
 * threshold control, history capture, and fitness sharing so each speciation
 * category can stay focused on one responsibility.
 */
import type {
  GenomeDetailed,
  SpeciesLike,
  SpeciationOptions,
} from '../../neat.types';

/**
 * Resolved compatibility-threshold adjustment settings.
 *
 * This is the non-nullable form of {@link SpeciationOptions.compatAdjust} used
 * by the speciation PID controller.
 *
 * @remarks
 * The PID controller uses `kp` (proportional gain) and `ki` (integral gain) to
 * update `options.compatibilityThreshold` based on how far the current species
 * count deviates from `options.targetSpecies`.
 */
export type CompatAdjust = NonNullable<SpeciationOptions['compatAdjust']>;

/**
 * Minimal context required to apply fitness sharing.
 *
 * Fitness sharing normalizes per-genome fitness within each species to reduce
 * selection pressure toward dense clusters of very similar genomes.
 */
export type FitnessSharingContext = {
  /**
   * Current species list. Each species must expose its `members` array.
   */
  _species: SpeciesLike[];

  /**
   * Compatibility distance between two genomes.
   *
   * @param a - First genome.
   * @param b - Second genome.
   * @returns Non-negative distance; smaller means more similar.
   */
  _compatibilityDistance: (a: GenomeDetailed, b: GenomeDetailed) => number;
};

/**
 * Minimal context required to update species stagnation.
 *
 * Stagnation pruning removes species that have not improved their best score
 * within a configured number of generations.
 */
export type StagnationContext = {
  /**
   * Current species list to update and/or prune.
   */
  _species: SpeciesLike[];

  /**
   * Current generation index used to compute time since last improvement.
   */
  generation: number;
};

/**
 * Accumulator for innovation-id statistics across a set of connections.
 *
 * Used for extended history telemetry (mean innovation, innovation range, and
 * enabled/disabled ratios).
 */
export type InnovationAccumulator = {
  /**
   * Sum of all collected innovation ids.
   */
  innovationSum: number;

  /**
   * Count of collected innovation ids.
   */
  innovationCount: number;

  /**
   * Maximum observed innovation id.
   */
  maxInnovation: number;

  /**
   * Minimum observed innovation id.
   */
  minInnovation: number;

  /**
   * Number of enabled connections observed.
   */
  enabledCount: number;

  /**
   * Number of disabled connections observed.
   */
  disabledCount: number;
};

/** Default minimum compatibility threshold. */
export const DEFAULT_MIN_COMPATIBILITY_THRESHOLD = 1;
/** Default maximum compatibility threshold. */
export const DEFAULT_MAX_COMPATIBILITY_THRESHOLD = 10;
/** Default compatibility threshold when unspecified. */
export const DEFAULT_COMPATIBILITY_THRESHOLD = 3;
/** Default target number of species for PID controller. */
export const DEFAULT_TARGET_SPECIES = 5;
/** Default proportional gain for compatibility PID. */
export const DEFAULT_COMPATIBILITY_PROPORTIONAL_GAIN = 0.5;
/** Default integral gain for compatibility PID. */
export const DEFAULT_COMPATIBILITY_INTEGRAL_GAIN = 10;
/** Default integral accumulator value. */
export const DEFAULT_COMPAT_INTEGRAL = 0;
/** Default grace period for young species. */
export const DEFAULT_SPECIES_AGE_GRACE = 3;
/** Multiplier used to convert grace generations to age threshold. */
export const SPECIES_AGE_GRACE_MULTIPLIER = 10;
/** Default penalty applied to old species. */
export const DEFAULT_SPECIES_OLD_PENALTY = 0.5;
/** Penalty cutoff where no reduction should occur. */
export const PENALTY_NO_EFFECT_THRESHOLD = 1;
/** Max number of history entries to keep. */
export const HISTORY_BUFFER_MAX_ENTRIES = 200;
/** Default sigma for fitness sharing. */
export const DEFAULT_SHARING_SIGMA = 0;
/** Fallback divisor when sharing sum is zero. */
export const SHARING_SUM_FLOOR = 1;
/** Maximum sharing contribution per peer. */
export const SHARING_MAX_CONTRIBUTION = 1;
/** Fallback divisor when member count is zero. */
export const DEFAULT_MEMBER_COUNT_FALLBACK = 1;
/** Distance used when comparing a member with itself. */
export const SHARING_SELF_DISTANCE = 0;
/** Default stagnation window in generations. */
export const DEFAULT_STAGNATION_WINDOW = 15;
/** Default last improved generation when missing. */
export const DEFAULT_LAST_IMPROVED_GENERATION = 0;
/** Fallback numeric score when missing. */
export const DEFAULT_SCORE_FALLBACK = 0;
/** Shared negative infinity constant for score initialization. */
export const NEGATIVE_INFINITY = Number.NEGATIVE_INFINITY;