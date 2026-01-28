/**
 * Utility helpers for NEAT speciation orchestration.
 */
import type {
  GenomeDetailed,
  SpeciesLike,
  ConnectionLike,
  SpeciationOptions,
  SpeciationHarnessContext,
} from './neat.types';

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

/**
 * Snapshot current species memberships for telemetry.
 *
 * @param speciationContext - Speciation harness context.
 * @returns Nothing.
 */
export function snapshotPreviousMembers<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(speciationContext: SpeciationHarnessContext<TOptions>): void {
  // Step 1: Ensure the previous map exists and is cleared.
  speciationContext._prevSpeciesMembers =
    speciationContext._prevSpeciesMembers ?? new Map();
  speciationContext._prevSpeciesMembers.clear();
  // Step 2: Capture member IDs per species.
  for (const species of speciationContext._species) {
    const previousMembers = new Set<number>();
    for (const member of species.members as GenomeDetailed[]) {
      previousMembers.add(member._id);
    }
    speciationContext._prevSpeciesMembers.set(species.id, previousMembers);
  }
}

/**
 * Clear member lists for all species.
 *
 * @param speciationContext - Speciation harness context.
 * @returns Nothing.
 */
export function resetSpeciesMembers<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(speciationContext: SpeciationHarnessContext<TOptions>): void {
  // Step 1: Empty each species membership list.
  speciationContext._species.forEach((species: SpeciesLike) => {
    species.members = [];
  });
}

/**
 * Assign each genome in the population to a compatible species.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @returns Nothing.
 */
export function assignPopulationToSpecies<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void {
  // Step 1: Assign each genome to an existing species when compatible.
  for (const genome of speciationContext.population) {
    const matchedSpecies = findCompatibleSpecies(
      speciationContext,
      options,
      genome,
    );
    if (matchedSpecies) {
      matchedSpecies.members.push(genome);
      continue;
    }
    // Step 2: Create a new species when no match is found.
    createSpeciesForGenome(speciationContext, genome);
  }
}

/**
 * Update the adaptive compatibility threshold and clamp to bounds.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @param compatAdjust - Compatibility adjustment settings.
 * @param minCompatibilityThreshold - Lower clamp bound.
 * @param maxCompatibilityThreshold - Upper clamp bound.
 * @returns Nothing.
 */
export function adjustCompatibilityThreshold<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  compatAdjust: CompatAdjust,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): void {
  // Step 1: Ensure the integral term is initialized.
  if (typeof speciationContext._compatIntegral !== 'number')
    speciationContext._compatIntegral = DEFAULT_COMPAT_INTEGRAL;
  // Step 2: Run PID update if the threshold is numeric.
  if (typeof options.compatibilityThreshold === 'number') {
    const updatedThreshold = computePidThreshold(
      speciationContext,
      options,
      compatAdjust,
      options.compatibilityThreshold,
      minCompatibilityThreshold,
      maxCompatibilityThreshold,
    );
    options.compatibilityThreshold = updatedThreshold;
  }
  // Step 3: Always clamp to configured min/max.
  clampCompatibilityThreshold(
    options,
    minCompatibilityThreshold,
    maxCompatibilityThreshold,
  );
}

/**
 * Refresh representatives and remove empty species.
 *
 * @param speciationContext - Speciation harness context.
 * @returns Nothing.
 */
export function refreshSpeciesRepresentatives<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(speciationContext: SpeciationHarnessContext<TOptions>): void {
  // Step 1: Remove empty species.
  speciationContext._species = speciationContext._species.filter(
    (species: SpeciesLike) => species.members.length > 0,
  );
  // Step 2: Assign representatives from member lists.
  speciationContext._species.forEach((species: SpeciesLike) => {
    species.representative = species.members[0] as GenomeDetailed;
  });
}

/**
 * Apply age protection penalties to old species.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @returns Nothing.
 */
export function applyAgeProtection<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void {
  // Step 1: Resolve configured protection policy.
  const ageProtection = options.speciesAgeProtection ?? {
    grace: DEFAULT_SPECIES_AGE_GRACE,
    oldPenalty: DEFAULT_SPECIES_OLD_PENALTY,
  };
  // Step 2: Apply penalties to old species when configured.
  for (const species of speciationContext._species) {
    const createdGeneration =
      speciationContext._speciesCreated.get(species.id) ??
      speciationContext.generation;
    const speciesAge = speciationContext.generation - createdGeneration;
    const graceMultiplier =
      (ageProtection.grace ?? DEFAULT_SPECIES_AGE_GRACE) *
      SPECIES_AGE_GRACE_MULTIPLIER;
    if (speciesAge < graceMultiplier) continue;
    const penalty = ageProtection.oldPenalty ?? DEFAULT_SPECIES_OLD_PENALTY;
    if (penalty >= PENALTY_NO_EFFECT_THRESHOLD) continue;
    (species.members as GenomeDetailed[]).forEach((member) => {
      if (typeof member.score === 'number') member.score *= penalty;
    });
  }
}

/**
 * Record the current species history snapshot.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @returns Nothing.
 */
export function recordHistory<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void {
  // Step 1: Select history format based on configuration.
  if (options.speciesAllocation?.extendedHistory) {
    const speciesStats = speciationContext._species.map(
      (species: SpeciesLike) =>
        buildExtendedHistoryStats(speciationContext, species),
    );
    speciationContext._speciesHistory.push({
      generation: speciationContext.generation,
      stats: speciesStats,
    });
    return;
  }
  speciationContext._speciesHistory.push({
    generation: speciationContext.generation,
    stats: speciationContext._species.map((species: SpeciesLike) => ({
      id: species.id,
      size: species.members.length,
      best: species.bestScore,
    })),
  });
}

/**
 * Trim species history to the maximum buffer size.
 *
 * @param speciationContext - Speciation harness context.
 * @returns Nothing.
 */
export function trimHistory<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(speciationContext: SpeciationHarnessContext<TOptions>): void {
  // Step 1: Drop the oldest history entry when the buffer is too large.
  if (speciationContext._speciesHistory.length > HISTORY_BUFFER_MAX_ENTRIES)
    speciationContext._speciesHistory.shift();
}

/**
 * Apply fitness sharing to penalize similarity within species.
 *
 * @param speciationContext - Neat instance context with species and distance function.
 * @param sharingSigma - Sharing radius used for distance weighting.
 * @returns Nothing.
 */
export function applyFitnessSharing(
  speciationContext: FitnessSharingContext,
  sharingSigma: number,
): void {
  const useSigmaSharing = sharingSigma > 0;

  // Step 1: Route to the sigma-aware or uniform strategy.
  if (useSigmaSharing) {
    applySigmaSharing(speciationContext, sharingSigma);
    return;
  }
  applyUniformSharing(speciationContext);

  /**
   * @param context - Fitness sharing context.
   * @param sigmaValue - Sharing radius used for distance weighting.
   * @returns Nothing.
   */
  function applySigmaSharing(
    context: FitnessSharingContext,
    sigmaValue: number,
  ): void {
    // Step 1: Apply sigma-based sharing to each species.
    for (const species of context._species) {
      const members = species.members as GenomeDetailed[];
      applySigmaSharingToMembers(context, members, sigmaValue);
    }
  }

  /**
   * @param context - Fitness sharing context.
   * @returns Nothing.
   */
  function applyUniformSharing(context: FitnessSharingContext): void {
    // Step 1: Apply uniform sharing across each species.
    for (const species of context._species) {
      const members = species.members as GenomeDetailed[];
      applyUniformSharingToMembers(members);
    }
  }

  /**
   * @param context - Fitness sharing context.
   * @param members - Members to share fitness within.
   * @param sigmaValue - Sharing radius used for distance weighting.
   * @returns Nothing.
   */
  function applySigmaSharingToMembers(
    context: FitnessSharingContext,
    members: GenomeDetailed[],
    sigmaValue: number,
  ): void {
    // Step 1: Normalize each member score by its sharing sum.
    for (let memberIndex = 0; memberIndex < members.length; memberIndex++) {
      const member = members[memberIndex];
      if (typeof member.score !== 'number') continue;
      const sharingSum = computeSharingSum(
        context,
        members,
        memberIndex,
        member,
        sigmaValue,
      );
      const safeSharingSum = sharingSum > 0 ? sharingSum : SHARING_SUM_FLOOR;
      member.score = member.score / safeSharingSum;
    }
  }

  /**
   * @param members - Members to share fitness within.
   * @returns Nothing.
   */
  function applyUniformSharingToMembers(members: GenomeDetailed[]): void {
    // Step 1: Normalize scores by member count.
    const memberCount = members.length || DEFAULT_MEMBER_COUNT_FALLBACK;
    for (const member of members) {
      if (typeof member.score === 'number') {
        member.score = member.score / memberCount;
      }
    }
  }

  /**
   * @param context - Fitness sharing context.
   * @param members - All members in the species.
   * @param memberIndex - Index of the current member.
   * @param member - Current member being normalized.
   * @param sigmaValue - Sharing radius used for distance weighting.
   * @returns Sharing sum for the member.
   */
  function computeSharingSum(
    context: FitnessSharingContext,
    members: GenomeDetailed[],
    memberIndex: number,
    member: GenomeDetailed,
    sigmaValue: number,
  ): number {
    // Step 1: Accumulate distance-based sharing contributions.
    let sharingSum = 0;
    for (let peerIndex = 0; peerIndex < members.length; peerIndex++) {
      const peerMember = members[peerIndex];
      const distance =
        memberIndex === peerIndex
          ? SHARING_SELF_DISTANCE
          : context._compatibilityDistance(member, peerMember);
      if (distance < sigmaValue) {
        const ratio = distance / sigmaValue;
        sharingSum += SHARING_MAX_CONTRIBUTION - ratio * ratio;
      }
    }
    return sharingSum;
  }
}

/**
 * Update stagnation counters and prune stagnant species.
 *
 * @param speciationContext - Neat instance context with species array and generation counter.
 * @param stagnationWindow - Allowed stagnation window.
 * @param sortSpeciesMembers - Sort function for species members.
 * @returns Nothing.
 */
export function updateSpeciesStagnation(
  speciationContext: StagnationContext,
  stagnationWindow: number,
  sortSpeciesMembers: (species: SpeciesLike) => void,
): void {
  // 1) Update per-species stagnation metrics.
  updateSpeciesStagnationCounters();
  // 2) Remove species that exceeded the stagnation window.
  pruneStagnantSpecies();

  /** @returns Nothing. */
  function updateSpeciesStagnationCounters(): void {
    // Step 1: Refresh best scores and improvement timestamps.
    for (const species of speciationContext._species) {
      sortSpeciesMembers(species);
      const topMember = (species.members as GenomeDetailed[])[0];
      const currentBest = species.bestScore ?? NEGATIVE_INFINITY;
      const candidateBest = topMember?.score ?? NEGATIVE_INFINITY;
      if (candidateBest > currentBest) {
        species.bestScore = candidateBest;
        species.lastImproved = speciationContext.generation;
      }
    }
  }

  /** @returns Nothing. */
  function pruneStagnantSpecies(): void {
    // Step 1: Filter to species within the stagnation window.
    const survivors = speciationContext._species.filter((species) =>
      isWithinStagnationWindow(species, stagnationWindow),
    );
    if (survivors.length) speciationContext._species = survivors;
  }

  /**
   * @param species - Species to evaluate.
   * @param windowSize - Allowed stagnation window.
   * @returns True when species is still within the window.
   */
  function isWithinStagnationWindow(
    species: SpeciesLike,
    windowSize: number,
  ): boolean {
    // Step 1: Compute elapsed generations since last improvement.
    const lastImproved =
      species.lastImproved ?? DEFAULT_LAST_IMPROVED_GENERATION;
    return speciationContext.generation - lastImproved <= windowSize;
  }
}

/**
 * Find a compatible species representative for the given genome.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @param genome - Genome to match.
 * @returns Matching species or undefined.
 */
function findCompatibleSpecies<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  genome: GenomeDetailed,
): SpeciesLike | undefined {
  // Step 1: Use the current threshold to match against representatives.
  const compatibilityThreshold =
    options.compatibilityThreshold ?? DEFAULT_COMPATIBILITY_THRESHOLD;
  for (const species of speciationContext._species) {
    const compatibilityDistance = speciationContext._compatibilityDistance(
      genome,
      species.representative as GenomeDetailed,
    );
    if (compatibilityDistance < compatibilityThreshold) return species;
  }
  return undefined;
}

/**
 * Create a new species for the provided genome.
 *
 * @param speciationContext - Speciation harness context.
 * @param genome - Genome that starts a new species.
 * @returns Nothing.
 */
function createSpeciesForGenome<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  genome: GenomeDetailed,
): void {
  // Step 1: Allocate a fresh species id.
  const newSpeciesId = speciationContext._nextSpeciesId++;
  // Step 2: Seed the new species with this genome.
  speciationContext._species.push({
    id: newSpeciesId,
    members: [genome],
    representative: genome,
    lastImproved: speciationContext.generation,
    bestScore: genome.score ?? NEGATIVE_INFINITY,
  });
  speciationContext._speciesCreated.set(
    newSpeciesId,
    speciationContext.generation,
  );
}

/**
 * Compute a PID-based threshold update and clamp when needed.
 *
 * @param speciationContext - Speciation harness context.
 * @param options - Speciation options.
 * @param compatAdjust - Compatibility adjustment settings.
 * @param currentThreshold - Current compatibility threshold.
 * @param minCompatibilityThreshold - Lower clamp bound.
 * @param maxCompatibilityThreshold - Upper clamp bound.
 * @returns Updated threshold.
 */
function computePidThreshold<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  compatAdjust: CompatAdjust,
  currentThreshold: number,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): number {
  // Step 1: Resolve target/observed species counts for the PID error.
  const targetSpeciesCount = options.targetSpecies ?? DEFAULT_TARGET_SPECIES;
  const observedSpeciesCount = speciationContext._species.length;
  // Step 2: Compute the signed error (positive means too few species).
  const speciesError = targetSpeciesCount - observedSpeciesCount;
  // Step 3: Resolve PID gains from configuration with defaults.
  const proportionalGain =
    compatAdjust.kp ?? DEFAULT_COMPATIBILITY_PROPORTIONAL_GAIN;
  const integralGain = compatAdjust.ki ?? DEFAULT_COMPATIBILITY_INTEGRAL_GAIN;
  // Step 4: Update the integral accumulator and compute the PID delta.
  const updatedIntegral = updateCompatibilityIntegral(
    speciationContext,
    speciesError,
  );
  const thresholdDelta = computePidDelta(
    speciesError,
    proportionalGain,
    integralGain,
    updatedIntegral,
  );
  // Step 5: Apply the delta to the current threshold.
  const rawThreshold = currentThreshold - thresholdDelta;
  // Step 6: Clamp to bounds and reset integral if clamped.
  return clampPidThreshold(
    speciationContext,
    rawThreshold,
    minCompatibilityThreshold,
    maxCompatibilityThreshold,
  );

  /**
   * @param context - Speciation harness context.
   * @param errorValue - Difference between target and observed species.
   * @returns Updated integral accumulator value.
   */
  function updateCompatibilityIntegral(
    context: SpeciationHarnessContext<TOptions>,
    errorValue: number,
  ): number {
    // Step 1: Read the current integral accumulator.
    const previousIntegral = context._compatIntegral ?? DEFAULT_COMPAT_INTEGRAL;
    // Step 2: Accumulate the error into the integral term.
    const nextIntegral = previousIntegral + errorValue;
    // Step 3: Persist the updated accumulator back to context.
    context._compatIntegral = nextIntegral;
    return nextIntegral;
  }

  /**
   * @param errorValue - Difference between target and observed species.
   * @param proportional - Proportional gain.
   * @param integral - Integral gain.
   * @param integralValue - Current integral accumulator value.
   * @returns Threshold delta to apply.
   */
  function computePidDelta(
    errorValue: number,
    proportional: number,
    integral: number,
    integralValue: number,
  ): number {
    // Step 1: Compute the proportional contribution.
    const proportionalContribution = proportional * errorValue;
    // Step 2: Compute the integral contribution.
    const integralContribution = integral * integralValue;
    // Step 3: Combine contributions into the delta.
    return proportionalContribution + integralContribution;
  }

  /**
   * @param context - Speciation harness context.
   * @param candidateThreshold - Threshold before clamping.
   * @param minThreshold - Lower clamp bound.
   * @param maxThreshold - Upper clamp bound.
   * @returns Clamped threshold.
   */
  function clampPidThreshold(
    context: SpeciationHarnessContext<TOptions>,
    candidateThreshold: number,
    minThreshold: number,
    maxThreshold: number,
  ): number {
    // Step 1: Clamp low and reset integral when below the minimum.
    if (candidateThreshold < minThreshold) {
      context._compatIntegral = DEFAULT_COMPAT_INTEGRAL;
      return minThreshold;
    }
    // Step 2: Clamp high and reset integral when above the maximum.
    if (candidateThreshold > maxThreshold) {
      context._compatIntegral = DEFAULT_COMPAT_INTEGRAL;
      return maxThreshold;
    }
    // Step 3: Return the unclamped threshold.
    return candidateThreshold;
  }
}

/**
 * Clamp the compatibility threshold to configured bounds.
 *
 * @param options - Speciation options.
 * @param minCompatibilityThreshold - Lower clamp bound.
 * @param maxCompatibilityThreshold - Upper clamp bound.
 * @returns Nothing.
 */
function clampCompatibilityThreshold(
  options: SpeciationOptions,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): void {
  // Step 1: Clamp when the threshold is present.
  if (typeof options.compatibilityThreshold !== 'number') return;
  if (options.compatibilityThreshold < minCompatibilityThreshold)
    options.compatibilityThreshold = minCompatibilityThreshold;
  if (options.compatibilityThreshold > maxCompatibilityThreshold)
    options.compatibilityThreshold = maxCompatibilityThreshold;
}

/**
 * Build extended history stats for a species.
 *
 * @param speciationContext - Speciation harness context.
 * @param species - Species to snapshot.
 * @returns Extended history entry.
 */
function buildExtendedHistoryStats<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  species: SpeciesLike,
): Record<string, unknown> {
  // Step 1: Snapshot members for structural and innovation summaries.
  const members = species.members as GenomeDetailed[];
  // Step 2: Compute structural sizes and their averages.
  const structuralSizes = computeMemberStructuralSizes(
    speciationContext,
    members,
  );
  const meanNodes = averageNumbers(structuralSizes.map((entry) => entry.nodes));
  const meanConnections = averageNumbers(
    structuralSizes.map((entry) => entry.connections),
  );
  // Step 3: Aggregate innovation statistics.
  const innovationStats = summarizeInnovations(speciationContext, members);
  // Step 4: Fold into the history stats payload.
  return {
    id: species.id,
    size: species.members.length,
    best: species.bestScore,
    meanNodes,
    meanConns: meanConnections,
    meanInnovation: innovationStats.meanInnovation,
    innovationRange: innovationStats.innovationRange,
    enabledRatio: innovationStats.enabledRatio,
  } as Record<string, unknown>;

  /**
   * @param context - Speciation harness context.
   * @param memberList - Members to summarize.
   * @returns Structural size stats per member.
   */
  function computeMemberStructuralSizes(
    context: SpeciationHarnessContext<TOptions>,
    memberList: GenomeDetailed[],
  ): Array<{
    nodes: number;
    connections: number;
    score: number;
    entropy: number;
  }> {
    // Step 1: Map members into structural summary rows.
    return memberList.map((member) => ({
      nodes: member.nodes.length,
      connections: member.connections.length,
      score: member.score ?? DEFAULT_SCORE_FALLBACK,
      entropy: context._structuralEntropy(member),
    }));
  }
}

/**
 * Average a list of numbers, returning zero when empty.
 *
 * @param values - Numeric values to average.
 * @returns Mean of the values or zero.
 */
function averageNumbers(values: number[]): number {
  // Step 1: Compute mean with a safe empty fallback.
  if (!values.length) return DEFAULT_SCORE_FALLBACK;
  const sum = values.reduce((total, value) => total + value, 0);
  return sum / values.length;
}

/**
 * Summarize innovation statistics for a set of members.
 *
 * @param speciationContext - Speciation harness context.
 * @param members - Members to summarize.
 * @returns Innovation summary statistics.
 */
function summarizeInnovations<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(
  speciationContext: SpeciationHarnessContext<TOptions>,
  members: GenomeDetailed[],
): {
  meanInnovation: number;
  innovationRange: number;
  enabledRatio: number;
} {
  // Step 1: Accumulate innovation data across all member connections.
  const innovationAccumulation = accumulateInnovationStats(
    speciationContext,
    members,
  );
  // Step 2: Compute mean/range and enabled ratio from the accumulation.
  const meanInnovation = computeInnovationMean(innovationAccumulation);
  const innovationRange = computeInnovationRange(innovationAccumulation);
  const enabledRatio = computeEnabledRatio(innovationAccumulation);
  return { meanInnovation, innovationRange, enabledRatio };

  /**
   * @param context - Speciation harness context.
   * @param memberList - Members to summarize.
   * @returns Accumulated innovation stats.
   */
  function accumulateInnovationStats(
    context: SpeciationHarnessContext<TOptions>,
    memberList: GenomeDetailed[],
  ): InnovationAccumulator {
    // Step 1: Seed accumulator defaults.
    const accumulator: InnovationAccumulator = {
      innovationSum: 0,
      innovationCount: 0,
      maxInnovation: NEGATIVE_INFINITY,
      minInnovation: Infinity,
      enabledCount: 0,
      disabledCount: 0,
    };
    // Step 2: Walk every connection and update the counters.
    for (const member of memberList) {
      for (const connection of member.connections as ConnectionLike[]) {
        applyConnectionInnovation(context, accumulator, connection);
      }
    }
    return accumulator;
  }

  /**
   * @param context - Speciation harness context.
   * @param accumulator - Mutable innovation accumulator.
   * @param connection - Connection to process.
   * @returns Nothing.
   */
  function applyConnectionInnovation(
    context: SpeciationHarnessContext<TOptions>,
    accumulator: InnovationAccumulator,
    connection: ConnectionLike,
  ): void {
    // Step 1: Resolve the innovation id for this connection.
    const innovation =
      connection.innovation ?? context._fallbackInnov(connection);
    // Step 2: Update numeric aggregates.
    accumulator.innovationSum += innovation;
    accumulator.innovationCount += 1;
    if (innovation > accumulator.maxInnovation)
      accumulator.maxInnovation = innovation;
    if (innovation < accumulator.minInnovation)
      accumulator.minInnovation = innovation;
    // Step 3: Track enabled/disabled totals.
    if (connection.enabled === false) accumulator.disabledCount += 1;
    else accumulator.enabledCount += 1;
  }

  /**
   * @param accumulator - Innovation accumulator.
   * @returns Mean innovation or fallback.
   */
  function computeInnovationMean(accumulator: InnovationAccumulator): number {
    // Step 1: Return the mean when there are observations.
    if (!accumulator.innovationCount) return DEFAULT_SCORE_FALLBACK;
    return accumulator.innovationSum / accumulator.innovationCount;
  }

  /**
   * @param accumulator - Innovation accumulator.
   * @returns Range of innovation ids or fallback.
   */
  function computeInnovationRange(accumulator: InnovationAccumulator): number {
    // Step 1: Validate min/max and compute the range.
    if (
      !Number.isFinite(accumulator.maxInnovation) ||
      !Number.isFinite(accumulator.minInnovation) ||
      accumulator.maxInnovation <= accumulator.minInnovation
    ) {
      return DEFAULT_SCORE_FALLBACK;
    }
    return accumulator.maxInnovation - accumulator.minInnovation;
  }

  /**
   * @param accumulator - Innovation accumulator.
   * @returns Ratio of enabled connections or fallback.
   */
  function computeEnabledRatio(accumulator: InnovationAccumulator): number {
    // Step 1: Compute ratio when any connections were observed.
    const enabledTotal = accumulator.enabledCount + accumulator.disabledCount;
    if (!enabledTotal) return DEFAULT_SCORE_FALLBACK;
    return accumulator.enabledCount / enabledTotal;
  }
}
