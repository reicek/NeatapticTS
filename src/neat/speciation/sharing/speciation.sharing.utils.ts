import type {
  GenomeDetailed,
  SpeciesLike,
} from '../../shared/neat.shared.types';
import {
  DEFAULT_LAST_IMPROVED_GENERATION,
  DEFAULT_MEMBER_COUNT_FALLBACK,
  NEGATIVE_INFINITY,
  SHARING_MAX_CONTRIBUTION,
  SHARING_SELF_DISTANCE,
  SHARING_SUM_FLOOR,
} from '../shared/speciation.shared';
import type {
  FitnessSharingContext,
  StagnationContext,
} from '../shared/speciation.shared';

/**
 * Fitness-sharing and stagnation mechanics for speciation.
 *
 * These helpers run after species are assigned. One normalizes scores within a
 * species so dense clusters do not dominate selection, and the other tracks
 * whether a species has stopped improving.
 */

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
  // Step 1: Update per-species stagnation metrics.
  updateSpeciesStagnationCounters();
  // Step 2: Remove species that exceeded the stagnation window.
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
