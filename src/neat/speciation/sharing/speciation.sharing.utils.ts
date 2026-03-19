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
 * This chapter owns the two post-assignment adjustments that make a species
 * registry useful over time rather than merely descriptive for one generation.
 * After assignment has grouped genomes and threshold tuning has updated the
 * future boundary, these helpers answer two follow-up questions:
 *
 * 1. should crowded species keep all of their raw score advantage,
 * 2. has a species stopped improving long enough to be removed from the run.
 *
 * Read the file in two halves. {@link applyFitnessSharing} reshapes scores so a
 * dense cluster of very similar genomes does not overwhelm selection simply by
 * volume. {@link updateSpeciesStagnation} then tracks whether each species is
 * still producing better members and prunes lineages that have gone stale.
 *
 * The boundary stays intentionally narrow. These helpers do not decide species
 * membership, adapt compatibility thresholds, or write history rows. They take
 * the current species registry as given and adjust how that registry affects
 * later selection pressure and long-run survival.
 *
 * ```mermaid
 * flowchart TD
 *   Assigned[Assigned species registry]
 *   Sharing[Normalize scores within each species]
 *   Ranked[Species members re-read with shared scores]
 *   Stagnation[Update best-score progress and prune stale species]
 *   Output[Species registry ready for later controller phases]
 *
 *   Assigned --> Sharing
 *   Sharing --> Ranked
 *   Ranked --> Stagnation
 *   Stagnation --> Output
 * ```
 */

/**
 * Apply fitness sharing to penalize similarity within species.
 *
 * Fitness sharing lowers the effective score of genomes that sit inside a dense
 * neighborhood of similar peers. That keeps one crowded species from dominating
 * later parent selection purely because many near-duplicates all retained their
 * full raw score.
 *
 * The helper supports two modes:
 * - sigma-aware sharing, which weights neighbors by compatibility distance when
 *   the sharing radius is positive,
 * - uniform sharing, which falls back to dividing each member's score by the
 *   species size when no positive radius is configured.
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
 * This is the long-run maintenance half of post-assignment speciation. It first
 * sorts each species so the "best current member" read is deterministic, then
 * refreshes best-score and last-improved bookkeeping, and finally removes
 * species whose improvement gap has exceeded the allowed stagnation window.
 *
 * Read this as the answer to "is this species still earning its place in the
 * population?" Species that keep improving remain eligible for future rounds;
 * species that stop improving long enough are pruned from the live registry.
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
