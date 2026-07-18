import { createTeamFitnessEvaluator } from '../../../../src/neat/nge-collective/neat.nge-collective';
import { Network, methods } from '../../../../src/browser-entry.ts';
import { NGE_DNA } from '../../../../src/neat/nge-dna/neat.nge-dna';
import { activateNgeNetworkFromEnvelope } from '../../../../src/neat/nge-dna/neat.nge-dna.operator';
import type {
  NgeDnaCanonicalEnvelope,
  NgeRulePlacement,
} from '../../../../src/neat/nge-dna/neat.nge-dna.types';
import type { NeatGenomeSubstrateCoordinate } from '../../../../src/neat/genome/genome.types';
import {
  TIER6_TOTAL_INPUT_SIZE,
  TOTAL_TIER4_INPUT_SIZE,
} from '../../controller/observation.assembler';

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
  /** Deterministic seed used to materialize this car's runtime network. */
  readonly seed: number;
  /** Controller network input dimension (tier-dependent). */
  readonly inputSize: number;
  /** Controller network output dimension (tier-dependent). */
  readonly outputSize: number;
  /** Canonical NGE DNA envelope carried by this car for polyandric reproduction. */
  readonly envelope: NgeDnaCanonicalEnvelope;
  /** Runs inference and returns the controller output vector. */
  activate(inputs: number[]): number[];
  /** Mutates this genome in place; must not affect other cars' genomes. */
  mutate(): void;
  /** Serializes the genome's connection weights to a typed-array payload. */
  serialize(): Float32Array;
  /** Returns the underlying Network instance for worker-side adaptation. */
  getNetwork(): Network;
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
  /**
   * Replace one car genome in the container with a new genome.
   *
   * Used at generation boundaries when polyandric reproduction produces
   * offspring envelopes that must be materialized into the team's car slots.
   *
   * @param carIndex - 0-based car index within the race pack.
   * @param genome - New genome to assign to that slot.
   */
  replaceCarGenome(carIndex: number, genome: CarGenome): void;
};

/** Monotonic counter used to generate distinct population IDs per container. */
let containerSerialNumber = 0;

/** Team layout for Tier 1–2: one car per team. */
const TIER_ONE_TWO_TEAM_LAYOUT = [0, 1] as const;
/** Team layout for Tier 3–4: two cars per team (2v2). */
const TIER_THREE_TEAM_LAYOUT = [0, 0, 1, 1] as const;
/** Team layout for Tier 5: three cars per team (3v3). */
const TIER_FIVE_TEAM_LAYOUT = [0, 0, 0, 1, 1, 1] as const;
/** Tier 1–2 car count. */
const TIER_ONE_TWO_CAR_COUNT = 2;
/** Tier 3–4 car count. */
const TIER_THREE_CAR_COUNT = 4;
/** Tier 5 car count (3v3 = 6 cars). */
const TIER_FIVE_CAR_COUNT = 6;
/** Tier 1–2 controller input dimension. */
const TIER_ONE_TWO_CONTROLLER_INPUT_SIZE = 4;
/** Tier 3+ controller input dimension (91-channel observation). */
const TIER_THREE_CONTROLLER_INPUT_SIZE = 91;
/** Tier 4+ controller input dimension (103-channel observation with tire health and pit/strategy). */
const TIER_FOUR_CONTROLLER_INPUT_SIZE = TOTAL_TIER4_INPUT_SIZE;
/** Tier 6 controller input dimension (124-channel observation with opponent perception). */
const TIER_SIX_CONTROLLER_INPUT_SIZE = TIER6_TOTAL_INPUT_SIZE;
/** Tier 1–2 controller output dimension. */
const TIER_ONE_TWO_CONTROLLER_OUTPUT_SIZE = 2;
/** Tier 3+ controller output dimension (2 control + 7 radio-write). */
const TIER_THREE_CONTROLLER_OUTPUT_SIZE = 9;

/** Archetype id for controller input modules placed at substrate z=0. */
const CONTROLLER_INPUT_ARCHETYPE_ID = 'controller-input' as const;
/** Archetype id for controller output modules placed at substrate z=1. */
const CONTROLLER_OUTPUT_ARCHETYPE_ID = 'controller-output' as const;
/** Computation motif used for all controller input/output modules. */
const CONTROLLER_COMPUTATION_TYPE = 'DenseFeedForward' as const;

/**
 * Builds the smallest materializable NGE DNA envelope for a racing controller.
 *
 * The envelope contains one input archetype and one output archetype, each with
 * a `replicate` rule pass that places `inputSize` modules at z=0 and
 * `outputSize` modules at z=1. A single CPPN program wires the `dist` input to the
 * `weight` output, which creates a sparse set of directed edges during phenotype
 * materialization. Per-car distinctness is then introduced by seeding each
 * materialized network and applying a deterministic weight-perturbation pass.
 *
 * @param inputSize - Controller network input dimension.
 * @param outputSize - Controller network output dimension.
 * @returns Canonical NGE DNA envelope ready for materialization.
 */
function createMinimalControllerEnvelope(
  inputSize: number,
  outputSize: number,
): NgeDnaCanonicalEnvelope {
  const inputPlacements: NgeRulePlacement[] = Array.from(
    { length: inputSize },
    (_, index): NgeRulePlacement => ({
      coordinate: [
        index / Math.max(1, inputSize),
        0,
        0,
      ] as NeatGenomeSubstrateCoordinate,
      computationType: CONTROLLER_COMPUTATION_TYPE,
    }),
  );
  const outputPlacements: NgeRulePlacement[] = Array.from(
    { length: outputSize },
    (_, index): NgeRulePlacement => ({
      coordinate: [
        index / Math.max(1, outputSize),
        0,
        1,
      ] as NeatGenomeSubstrateCoordinate,
      computationType: CONTROLLER_COMPUTATION_TYPE,
    }),
  );

  return new NGE_DNA({
    moduleArchetypes: [
      {
        archetypeId: CONTROLLER_INPUT_ARCHETYPE_ID,
        computationType: CONTROLLER_COMPUTATION_TYPE,
      },
      {
        archetypeId: CONTROLLER_OUTPUT_ARCHETYPE_ID,
        computationType: CONTROLLER_COMPUTATION_TYPE,
      },
    ],
    rulePasses: [
      {
        kind: 'replicate',
        archetypeId: CONTROLLER_INPUT_ARCHETYPE_ID,
        priority: 1,
        placements: inputPlacements,
      },
      {
        kind: 'replicate',
        archetypeId: CONTROLLER_OUTPUT_ARCHETYPE_ID,
        priority: 2,
        placements: outputPlacements,
      },
    ],
    cppnPrograms: [
      {
        programId: 'dist-to-controls',
        edges: [
          {
            sourceNodeId: 'dist',
            targetNodeId: 'weight',
            weight: 1,
          },
        ],
      },
    ],
    reproductionPolicy: {
      mode: 'polyandric',
      polyandricDroneCount: 2,
      polyandricDroneContributionFraction: 0.25,
      queenBias: 0.85,
      assignedRegionStrategy: 'non-overlapping',
      modeIsEvolvable: true,
      seedPolicy: 'queen-weighted',
    },
  }).toCanonical();
}

/**
 * Configuration options for `createCarGenome`.
 */
export type CarGenomeOptions = {
  /** Car index within the race pack (0-based). */
  readonly carIndex: number;
  /** Deterministic seed used to materialize the runtime network. */
  readonly seed: number;
  /** Team id: 0 for Team A (blue), 1 for Team B (red). */
  readonly teamId: 0 | 1;
  /** Population id of the team this car belongs to. */
  readonly populationId: string;
  /** Controller network input dimension (tier-dependent). */
  readonly inputSize: number;
  /** Controller network output dimension (tier-dependent). */
  readonly outputSize: number;
  /** Optional pre-built canonical envelope (used for polyandric offspring). */
  readonly envelope?: NgeDnaCanonicalEnvelope;
  /** Optional pre-materialized network (used for polyandric offspring). */
  readonly network?: Network;
};

/**
 * Creates a single car's independent genome backed by a real `Network` instance.
 *
 * When no envelope or network is supplied, the genome is built from a minimal
 * NGE DNA envelope and materialized deterministically with the provided seed.
 * This keeps every car NGE-enabled so that polyandric reproduction can read
 * its `envelope` at the generation boundary.
 *
 * @param options - Car genome configuration.
 * @returns A car genome handle with a materialized runtime network.
 */
export function createCarGenome(options: CarGenomeOptions): CarGenome {
  const {
    carIndex,
    seed,
    teamId,
    populationId,
    inputSize,
    outputSize,
    envelope: providedEnvelope,
    network: providedNetwork,
  } = options;

  const envelope =
    providedEnvelope ?? createMinimalControllerEnvelope(inputSize, outputSize);
  const network =
    providedNetwork ?? activateNgeNetworkFromEnvelope(envelope, seed);

  if (!providedNetwork) {
    network.setSeed(seed);
    const mutationCount = Math.max(1, network.connections.length);
    for (let index = 0; index < mutationCount; index++) {
      network.mutate(methods.mutation.MOD_WEIGHT);
    }
  }

  return {
    carIndex,
    teamId,
    populationId,
    seed,
    inputSize,
    outputSize,
    envelope,
    activate(inputs: number[]): number[] {
      return network.activate(inputs);
    },
    mutate(): void {
      network.mutate(methods.mutation.MOD_WEIGHT);
    },
    serialize(): Float32Array {
      return Float32Array.from(network.connections.map((c) => c.weight));
    },
    getNetwork(): Network {
      return network;
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
 * The `tier` field in the config controls two dimension switches:
 * - **Car count:** Tier 1–2 allocates 2 cars (one per team); Tier 3+ allocates
 *   4 cars (two per team) using the `[0, 0, 1, 1]` team layout; Tier 6 uses a
 *   6-car `[0, 0, 0, 1, 1, 1]` layout.
 * - **Controller input dimension:** Tier 1 produces 70-input networks; Tier 2
 *   produces 77-input networks (70 base + 7 self-radio); Tier 3 produces
 *   91-input networks (70 base + 21 teammate-radio); Tier 4/5 produces 103-input
 *   networks (91 Tier 3 + 4 tire-health + 8 pit/strategy channels); Tier 6
 *   produces 124-input networks (103 Tier 4/5 + 21 opponent-perception channels).
 *
 * The output dimension is 2 for Tier 1–2 (throttle + steer) and 9 for Tier 3+
 * (2 control + 7 radio-write). Each car gets a distinct seed derived from the
 * base `rngSeed` plus the car index so activation outputs differ from
 * generation 1 onward.
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
 *
 * @example
 * ```ts
 * // Tier 4: 2v2 with 103-input / 9-output controller networks.
 * const container = createCoevolutionContainer({ populationSize: 50, rngSeed: 42, tier: 4 });
 * const genomes = container.getCarGenomes();
 * // genomes.length === 4 (two blue, two red)
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
  // Tier 5 uses a 6-car [0, 0, 0, 1, 1, 1] layout; Tier 3–4 uses 4 cars
  // [0, 0, 1, 1]; Tier 1–2 uses 2 cars [0, 1].
  const isTier6 = config.tier >= 6;
  const isTier5 = config.tier >= 5;
  const isTier4 = config.tier >= 4;
  const isTier3 = config.tier >= 3;
  const carCount = isTier5
    ? TIER_FIVE_CAR_COUNT
    : isTier3
      ? TIER_THREE_CAR_COUNT
      : TIER_ONE_TWO_CAR_COUNT;
  const inputSize = isTier6
    ? TIER_SIX_CONTROLLER_INPUT_SIZE
    : isTier4
      ? TIER_FOUR_CONTROLLER_INPUT_SIZE
      : isTier3
        ? TIER_THREE_CONTROLLER_INPUT_SIZE
        : TIER_ONE_TWO_CONTROLLER_INPUT_SIZE;
  const outputSize = isTier3
    ? TIER_THREE_CONTROLLER_OUTPUT_SIZE
    : TIER_ONE_TWO_CONTROLLER_OUTPUT_SIZE;
  const teamLayout = isTier5
    ? TIER_FIVE_TEAM_LAYOUT
    : isTier3
      ? TIER_THREE_TEAM_LAYOUT
      : TIER_ONE_TWO_TEAM_LAYOUT;
  const carGenomes: CarGenome[] = Array.from(
    { length: carCount },
    (_, carIndex) => {
      const teamId = teamLayout[carIndex] as 0 | 1;
      const populationId =
        teamId === 0 ? teamA.populationId : teamB.populationId;
      return createCarGenome({
        carIndex,
        seed: config.rngSeed + carIndex,
        teamId,
        populationId,
        inputSize,
        outputSize,
      });
    },
  );

  return {
    teamA,
    teamB,
    resolveTeamFitness,
    advanceTeamGeneration,
    getCarGenome,
    getCarGenomes,
    replaceCarGenome,
  };

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
   * Replace one car genome in the container with a new genome.
   *
   * @param carIndex - 0-based car index within the race pack.
   * @param genome - New genome to assign to that slot.
   */
  function replaceCarGenome(carIndex: number, genome: CarGenome): void {
    carGenomes[carIndex] = genome;
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

/**
 * Result of queen selection for one team.
 *
 * The queen is the best-finishing car on the team — the one whose individual
 * finishing position is lowest (best).  This selection drives polyandric
 * reproduction: the queen's genome becomes the template, and the other team
 * cars become drones.
 *
 * @property teamId - 0 for Team A, 1 for Team B.
 * @property queenCarIndex - Car index of the best-finishing car on this team.
 * @property queenFinishPosition - Individual finishing position of the queen.
 * @property droneCarIndices - Car indices of the remaining team cars (drones).
 */
export type QueenSelectionResult = {
  readonly teamId: 0 | 1;
  readonly queenCarIndex: number;
  readonly queenFinishPosition: number;
  readonly droneCarIndices: readonly number[];
};

/**
 * Selects the queen (best-finishing car) for each team after a race.
 *
 * The queen is the car with the lowest (best) individual finishing position
 * on its team. The remaining team cars become drones. This selection is
 * consumed by the FSM generation-boundary transition in
 * `simulation-worker.evolution.protocol.service.ts`, which calls
 * `reproducePolyandric` once per team to produce the next generation's
 * genomes.
 *
 * The queen-selection policy uses best-finishing-position (the lowest
 * individual finish wins the queen role), while population-level team fitness
 * remains shared-equal (average of all members) for team coordination. This
 * split policy lets queen selection reward the winning car's DNA without
 * destabilizing the cooperative fitness signal that drives team coordination.
 *
 * @param carFinishPositions - Finish positions for all cars, indexed by carIndex.
 * @param teamLayout - Team assignment per car (0 for Team A, 1 for Team B).
 * @returns One QueenSelectionResult per team (Team A first, Team B second).
 *
 * @example
 * ```ts
 * // 6-car Tier 5 pack: Team A cars 0,1,2 finish at positions 3,1,5
 * // Team B cars 3,4,5 finish at positions 2,4,6
 * const results = selectQueenPerTeam([3, 1, 5, 2, 4, 6], [0, 0, 0, 1, 1, 1]);
 * // results[0].queenCarIndex === 1 (Team A best finish = position 1)
 * // results[1].queenCarIndex === 3 (Team B best finish = position 2)
 * ```
 */
export function selectQueenPerTeam(
  carFinishPositions: readonly number[],
  teamLayout: readonly (0 | 1)[],
): readonly QueenSelectionResult[] {
  const teamCarIndices: number[][] = [[], []];

  // Step 1: Group car indices by team.
  for (let carIndex = 0; carIndex < carFinishPositions.length; carIndex++) {
    const teamId = teamLayout[carIndex] ?? 0;
    teamCarIndices[teamId].push(carIndex);
  }

  // Step 2: For each team, find the best-finishing car (queen) and the rest (drones).
  return teamCarIndices.map((teamCars, teamId) => {
    let bestCarIndex = teamCars[0] ?? 0;
    let bestPosition =
      carFinishPositions[bestCarIndex] ?? Number.POSITIVE_INFINITY;

    for (const carIndex of teamCars) {
      const position = carFinishPositions[carIndex] ?? Number.POSITIVE_INFINITY;
      if (position < bestPosition) {
        bestPosition = position;
        bestCarIndex = carIndex;
      }
    }

    const droneCarIndices = teamCars.filter(
      (carIndex) => carIndex !== bestCarIndex,
    );

    return {
      teamId: teamId as 0 | 1,
      queenCarIndex: bestCarIndex,
      queenFinishPosition: bestPosition,
      droneCarIndices,
    };
  });
}
