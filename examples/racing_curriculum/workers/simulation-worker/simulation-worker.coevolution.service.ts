import { createTeamFitnessEvaluator } from '../../../../src/neat/nge-collective/neat.nge-collective';

/**
 * Team A/B coevolution container for the racing curriculum benchmark.
 *
 * Each call to `createCoevolutionContainer` allocates two independent
 * population handles — one for Team A and one for Team B.  Team-level fitness
 * is routed through the reusable NGE core evaluator, using the racing policy
 * "best (lowest) finishing position among that team's cars."
 *
 * TODO: NGE_TODO — When ModulatorBroadcaster and EpisodicSlot become available
 * (upstream Phase G), each team container should support neuromodulation
 * broadcast across team members.
 *
 * TODO: NGE_TODO — Polyandric reproduction (`modeIsEvolvable`, upstream Phase E)
 * should extend the population container once that primitive exists.
 */

/** Narrow config used to allocate the racing coevolution container. */
export type CoevolutionConfig = {
  readonly populationSize: number;
  readonly rngSeed: number;
  readonly tier: number;
};

/** Opaque per-team population handle wrapping a future `Neat` instance. */
export type TeamPopulationContainer = {
  /** Stable unique identity token — distinct between Team A and Team B. */
  readonly populationId: string;
};

/** Paired Team A/B coevolution container with team-fitness resolver. */
export type CoevolutionContainer = {
  readonly teamA: TeamPopulationContainer;
  readonly teamB: TeamPopulationContainer;
  /**
   * Resolves team fitness as the best (lowest) finishing position among the
   * team's cars.  Position 1 = first place (best).
   *
   * @param teamId - 0 for Team A, 1 for Team B.
   * @param carFinishPositions - Finish positions for that team's cars only.
   * @returns Lowest finish position, or `Infinity` when the list is empty.
   */
  resolveTeamFitness(
    teamId: 0 | 1,
    carFinishPositions: readonly number[],
  ): number;
};

/** Monotonic counter used to generate distinct population IDs per container. */
let containerSerialNumber = 0;
/** Racing stores finish positions in `rawScore`; support score is unused here. */
type RacingTeamMemberResult = {
  readonly memberId: string;
  readonly rawScore: number;
  readonly supportScore: number;
};
/** Stable racing team identifiers used to route local results through the core seam. */
type RacingTeamId = 'team-a' | 'team-b';
/** Core-owned evaluator reused by the racing benchmark with a racing-local policy. */
const evaluateRacingTeamFitness = createTeamFitnessEvaluator<
  RacingTeamId,
  RacingTeamMemberResult
>(selectBestFinishingPosition);

/**
 * Creates a paired Team A/B coevolution container with independent population
 * handles and a best-position team-fitness resolver.
 *
 * @param _config - Container configuration (population size, seed, tier).
 * @returns Paired coevolution container with distinct team handles.
 *
 * @example
 * ```ts
 * const container = createCoevolutionContainer({ populationSize: 50, rngSeed: 1, tier: 1 });
 * // container.teamA.populationId !== container.teamB.populationId
 * const fitness = container.resolveTeamFitness(0, [3, 7]); // → 3
 * ```
 */
export function createCoevolutionContainer(
  config: CoevolutionConfig,
): CoevolutionContainer {
  // Step 1: Allocate unique serial numbers for both team handles.
  const serialA = ++containerSerialNumber;
  const serialB = ++containerSerialNumber;

  const teamA: TeamPopulationContainer = {
    populationId: `team-a-seed${config.rngSeed}-${serialA}`,
  };
  const teamB: TeamPopulationContainer = {
    populationId: `team-b-seed${config.rngSeed}-${serialB}`,
  };

  return { teamA, teamB, resolveTeamFitness };

  /**
   * Resolves team fitness through the reusable NGE core evaluator.
   *
   * @param teamId - Team index routed to a stable core team identifier.
   * @param carFinishPositions - Positions for that team's cars.
   * @returns Minimum position, or `Infinity` when no cars finished.
   */
  function resolveTeamFitness(
    teamId: 0 | 1,
    carFinishPositions: readonly number[],
  ): number {
    // Step 1: Convert benchmark-local finish positions into the reusable core group shape.
    const racingTeamGroup = createRacingTeamResultGroup(
      teamId,
      carFinishPositions,
    );

    // Step 2: Reuse the NGE core evaluator seam instead of local aggregation logic.
    return (
      evaluateRacingTeamFitness([racingTeamGroup]).at(0)?.teamFitness ??
      Number.POSITIVE_INFINITY
    );
  }
}

/**
 * Convert one racing team's finish positions into the generic team-group seam.
 *
 * @param teamId - Racing-local team index.
 * @param carFinishPositions - Finish positions for this team's cars.
 * @returns Generic team-result group ready for the core evaluator.
 */
function createRacingTeamResultGroup(
  teamId: 0 | 1,
  carFinishPositions: readonly number[],
): {
  readonly teamId: RacingTeamId;
  readonly memberResults: readonly RacingTeamMemberResult[];
} {
  // Step 1: Normalize the racing-local team index to a stable team identifier.
  const normalizedTeamId = teamId === 0 ? 'team-a' : 'team-b';

  return {
    teamId: normalizedTeamId,
    memberResults: carFinishPositions.map((finishPosition, memberIndex) => ({
      memberId: `${normalizedTeamId}-member-${memberIndex}`,
      rawScore: finishPosition,
      supportScore: 0,
    })),
  };
}

/**
 * Racing policy: the best team car defines the team's fitness.
 *
 * Lower finishing positions are better, so this policy selects the minimum
 * recorded `rawScore`. Empty groups preserve the existing `Infinity` fallback.
 *
 * @param group - Generic team group routed through the core evaluator seam.
 * @returns Best finishing position, or `Infinity` when the group is empty.
 */
function selectBestFinishingPosition(group: {
  readonly memberResults: readonly RacingTeamMemberResult[];
}): number {
  if (group.memberResults.length === 0) {
    return Number.POSITIVE_INFINITY;
  }

  // Step 1: Fold to the best (lowest) position — not sum, not average.
  return group.memberResults.reduce(
    (bestFinishPosition, memberResult) =>
      Math.min(bestFinishPosition, memberResult.rawScore),
    Number.POSITIVE_INFINITY,
  );
}
