/**
 * Shared vocabulary and defaults for NEAT speciation.
 *
 * This chapter is the shared language layer beneath the speciation subtree.
 * The adjacent chapters split responsibilities on purpose: assignment decides
 * membership, threshold tuning adjusts the future compatibility boundary,
 * sharing and stagnation reshape pressure after assignment, and history records
 * what happened. This file supplies the common words and defaults they all need
 * in order to cooperate without re-defining the same contracts in each folder.
 *
 * Read the exports in three families:
 *
 * 1. narrow runtime contexts such as {@link FitnessSharingContext} and
 *    {@link StagnationContext},
 * 2. compact summary types such as {@link CompatAdjust} and
 *    {@link InnovationAccumulator},
 * 3. default constants that define the baseline threshold, sharing, age,
 *    history, and score semantics across the subtree.
 */
import type {
  GenomeDetailed,
  SpeciesLike,
  SpeciationOptions,
} from '../../shared/neat.shared.types';

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
 *
 * Keeping this as a shared type prevents the threshold chapter from widening
 * into a full options surface when it really only needs the resolved threshold
 * adjustment knobs.
 */
export type CompatAdjust = NonNullable<SpeciationOptions['compatAdjust']>;

/**
 * Minimal context required to apply fitness sharing.
 *
 * Fitness sharing normalizes per-genome fitness within each species to reduce
 * selection pressure toward dense clusters of very similar genomes.
 *
 * The contract stays deliberately small: one current species registry and one
 * compatibility-distance reader. Sharing does not need assignment state,
 * history buffers, or threshold integrals, so those concerns stay outside this
 * context.
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
 *
 * This context is intentionally narrower than the full speciation harness
 * because stagnation only needs a live registry and a generation counter to
 * decide whether a species is still earning its place.
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
 *
 * Read this as the folded evidence bag for one species-history snapshot. The
 * history chapter gathers raw connection-level signals here first and only then
 * converts them into reader-friendly summary numbers.
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

/** Lower bound that keeps adaptive threshold control from collapsing to zero. */
export const DEFAULT_MIN_COMPATIBILITY_THRESHOLD = 1;
/** Upper bound that keeps adaptive threshold control from merging too aggressively. */
export const DEFAULT_MAX_COMPATIBILITY_THRESHOLD = 10;
/** Baseline compatibility boundary used before adaptive tuning moves it. */
export const DEFAULT_COMPATIBILITY_THRESHOLD = 3;
/** Default species-count target that the threshold controller tries to maintain. */
export const DEFAULT_TARGET_SPECIES = 5;
/** Default proportional gain for immediate threshold response to species-count error. */
export const DEFAULT_COMPATIBILITY_PROPORTIONAL_GAIN = 0.5;
/** Default integral gain for accumulated threshold response across generations. */
export const DEFAULT_COMPATIBILITY_INTEGRAL_GAIN = 10;
/** Neutral starting value for the compatibility-threshold integral accumulator. */
export const DEFAULT_COMPAT_INTEGRAL = 0;
/** Default grace window before older-species penalties are allowed to apply. */
export const DEFAULT_SPECIES_AGE_GRACE = 3;
/** Multiplier that turns the coarse grace setting into the runtime age threshold. */
export const SPECIES_AGE_GRACE_MULTIPLIER = 10;
/** Default score multiplier applied when an old species is no longer protected. */
export const DEFAULT_SPECIES_OLD_PENALTY = 0.5;
/** Penalty cutoff where no reduction should occur. */
export const PENALTY_NO_EFFECT_THRESHOLD = 1;
/** Maximum number of recent species-history rows retained in memory. */
export const HISTORY_BUFFER_MAX_ENTRIES = 200;
/** Default sharing radius; zero selects the simpler uniform sharing fallback. */
export const DEFAULT_SHARING_SIGMA = 0;
/** Fallback divisor when sharing sum is zero. */
export const SHARING_SUM_FLOOR = 1;
/** Maximum sharing contribution per peer. */
export const SHARING_MAX_CONTRIBUTION = 1;
/** Fallback divisor when member count is zero. */
export const DEFAULT_MEMBER_COUNT_FALLBACK = 1;
/** Distance used when comparing a member with itself. */
export const SHARING_SELF_DISTANCE = 0;
/** Default number of generations a species may stagnate before pruning is allowed. */
export const DEFAULT_STAGNATION_WINDOW = 15;
/** Default last improved generation when missing. */
export const DEFAULT_LAST_IMPROVED_GENERATION = 0;
/** Shared numeric fallback used when a speciation summary needs a missing score. */
export const DEFAULT_SCORE_FALLBACK = 0;
/** Shared negative infinity constant for score initialization. */
export const NEGATIVE_INFINITY = Number.NEGATIVE_INFINITY;
