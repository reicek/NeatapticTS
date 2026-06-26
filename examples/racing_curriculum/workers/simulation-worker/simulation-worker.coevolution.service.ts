import { createTeamFitnessEvaluator } from '../../../../src/neat/nge-collective/neat.nge-collective';
import { Network, methods } from '../../../../src/browser-entry.ts';

/**
 * Team A/B coevolution container for the racing curriculum benchmark.
 *
 * Each call to `createCoevolutionContainer` allocates two independent
 * population handles — one for Team A and one for Team B.  Team-level fitness
 * is routed through the reusable NGE core evaluator, using the racing policy
 * "best (lowest) finishing position among that team's cars."
 *
 * The two-population shape follows the competitive coevolution pattern: each
 * team is evaluated against frozen snapshots of the other, so neither side
 * optimizes against a stationary target.  This helps keep the search from
 * collapsing into a one-sided arms race.  See
 * [Coevolution (Wikipedia)](https://en.wikipedia.org/wiki/Coevolution)
 * for background, and
 * [Neuroevolution of augmenting topologies (Wikipedia)](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
 * for the NEAT algorithm that underpins the population containers.
 *
 * Extension points:
 * - Extend this container to broadcast neuromodulation across team members
 *   when `ModulatorBroadcaster` and `EpisodicSlot` primitives are available.
 * - Extend this container with polyandric reproduction (`modeIsEvolvable`)
 *   once that primitive is available.
 */

/** Narrow config used to allocate the racing coevolution container. */
export type CoevolutionConfig = {
  readonly populationSize: number;
  readonly rngSeed: number;
  readonly tier: number;
};

/**
 * Opaque per-team population handle wrapping a future `Neat` instance.
 *
 * The handle exposes a stable identity and a mutable generation counter so the
 * racing benchmark can track each team's progress independently.
 */
export type TeamPopulationContainer = {
  /** Stable unique identity token — distinct between Team A and Team B. */
  readonly populationId: string;
  /** Current NEAT generation for this team's isolated population. */
  generation: number;
};

/**
 * Handle for one car's independent genome.
 *
 * Each car genome is a fully independent network with its own evolution state.
 * Mutating one car's genome must not affect any other car's genome.  Each car
 * gets a distinct `Network` instance seeded with a car-specific seed so
 * activation outputs differ from the first generation onward.
 */
export type CarGenome = {
  /** Car index within the race pack (0-based). */
  readonly carIndex: number;
  /** Team id: 0 for Team A (blue), 1 for Team B (red). */
  readonly teamId: 0 | 1;
  /** Population id of the team this car belongs to. */
  readonly populationId: string;
  /** Runs inference and returns the controller output vector. */
  activate(inputs: number[]): number[];
  /** Mutates this genome in place; must not affect other cars' genomes. */
  mutate(): void;
  /** Serializes the genome's connection weights to a typed-array payload. */
  serialize(): Float32Array;
};

/**
 * Paired Team A/B coevolution container with a racing-specific team-fitness
 * resolver.
 *
 * The container exposes both team handles and the policy that converts finishing
 * positions into a scalar fitness value for each side.
 */
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
  /**
   * Advances one team's isolated population by a single generation.
   *
   * @param teamId - `'team-a'` for Team A or `'team-b'` for Team B.
   */
  advanceTeamGeneration(teamId: 'team-a' | 'team-b'): void;
  /**
   * Returns the distinct genome for the requested car index.
   * Each car gets its own independent genome — no sharing across cars.
   *
   * @param carIndex - 0-based car index within the race pack.
   * @returns The independent genome handle for that car.
   */
  getCarGenome(carIndex: number): CarGenome;
  /**
   * Returns all car genomes as an array (one per car).
   * Each entry must be a distinct genome object.
   *
   * @returns Array of per-car genomes.
   */
  getCarGenomes(): readonly CarGenome[];
};

/** Monotonic counter used to generate distinct population IDs per container. */
let containerSerialNumber = 0;

/** Number of cars in the race pack — one per team. */
const CAR_COUNT = 2 as const;

/** Input dimension for the per-car controller network. */
const CONTROLLER_INPUT_SIZE = 4 as const;

/** Output dimension for the per-car controller network. */
const CONTROLLER_OUTPUT_SIZE = 2 as const;

/**
 * Creates a single car's independent genome backed by a real `Network` instance.
 *
 * Each car gets a unique seed derived from the base rngSeed plus the car index,
 * so activation outputs differ from generation 1.
 *
 * @param carIndex - 0-based car index within the race pack.
 * @param baseSeed - Base RNG seed from the coevolution config.
 * @param teamId - 0 for Team A (blue), 1 for Team B (red).
 * @param populationId - The population id of the team this car belongs to.
 */
function createCarGenome(
  carIndex: number,
  baseSeed: number,
  teamId: 0 | 1,
  populationId: string,
): CarGenome {
  const network = new Network(CONTROLLER_INPUT_SIZE, CONTROLLER_OUTPUT_SIZE, {
    seed: baseSeed + carIndex,
  });

  return {
    carIndex,
    teamId,
    populationId,
    activate(inputs: number[]): number[] {
      return network.activate(inputs);
    },
    mutate(): void {
      network.mutate(methods.mutation.MOD_WEIGHT);
    },
    serialize(): Float32Array {
      return Float32Array.from(network.connections.map((c) => c.weight));
    },
  };
}

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
    generation: 0,
  };
  const teamB: TeamPopulationContainer = {
    populationId: `team-b-seed${config.rngSeed}-${serialB}`,
    generation: 0,
  };

  // Step 2: Create independent per-car genomes — one per car, each with a
  // distinct seed so activation outputs differ from generation 1 onward.
  // Car 0 belongs to Team A (blue), car 1 belongs to Team B (red).
  const carGenomes: CarGenome[] = Array.from({ length: CAR_COUNT }, (_, carIndex) => {
    const teamId = carIndex === 0 ? 0 : 1;
    const populationId = teamId === 0 ? teamA.populationId : teamB.populationId;
    return createCarGenome(carIndex, config.rngSeed, teamId, populationId);
  });

  return { teamA, teamB, resolveTeamFitness, advanceTeamGeneration, getCarGenome, getCarGenomes };

  /**
   * Returns the distinct genome for the requested car index.
   *
   * @param carIndex - 0-based car index within the race pack.
   * @returns The independent genome handle for that car.
   */
  function getCarGenome(carIndex: number): CarGenome {
    return carGenomes[carIndex];
  }

  /**
   * Returns all car genomes as an array (one per car).
   *
   * @returns Array of per-car genomes.
   */
  function getCarGenomes(): readonly CarGenome[] {
    return carGenomes;
  }

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

  /**
   * Advances the requested team's isolated generation counter by one.
   *
   * @param teamId - Stable team identifier to advance.
   */
  function advanceTeamGeneration(teamId: 'team-a' | 'team-b'): void {
    const target = teamId === 'team-a' ? teamA : teamB;
    target.generation += 1;
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
