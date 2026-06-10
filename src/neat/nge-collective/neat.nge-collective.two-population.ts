import Neat, { type NeatOptions } from '../../neat';
import type {
  CollectiveEvaluationContext,
  OpponentSnapshotPool,
} from './neat.nge-collective.types';
import { createSharedField } from './neat.nge-collective.shared-field';
import { createCollectiveEvaluationContext } from './neat.nge-collective.evaluation';
import {
  createOpponentSnapshotPool,
  addOpponentSnapshot,
} from './neat.nge-collective.metrics';

/** Number of cars evaluated in the Phase 3 two-vs-two scaffold. */
const TWO_POPULATION_AGENT_COUNT = 4;
/** Number of cars per team in the Phase 3 scaffold. */
const TEAM_AGENT_COUNT = 2;
/** Number of teammate-radio channels available per car. */
const TWO_POPULATION_RADIO_CHANNEL_COUNT = 7;
/** Default opponent snapshot capacity retained per team. */
const DEFAULT_OPPONENT_SNAPSHOT_CAPACITY = 10;
/** Total shared-field size for the 2v2 radio scaffold. */
const TWO_POPULATION_FIELD_SIZE =
  TWO_POPULATION_AGENT_COUNT * TWO_POPULATION_RADIO_CHANNEL_COUNT;

/**
 * Runtime state owned by one team inside the two-population harness.
 *
 * A team-scoped state is intentionally narrow: one `Neat` controller owns that
 * side's population, innovation tracker, and species bookkeeping, while the
 * rolling `opponentSnapshotPool` preserves frozen rival champions from the
 * other side. That split lets each team evolve independently without losing the
 * tournament-style pressure needed for coevolution.
 *
 * @example
 * ```ts
 * const harness = createTwoPopulationHarness({}, {});
 * const teamAState = harness.teamA;
 *
 * teamAState.controller.generation;
 * teamAState.opponentSnapshotPool.snapshots;
 * ```
 */
export interface TeamScopedState {
  /** Team-local `Neat` controller that owns one side's population lifecycle. */
  readonly controller: Neat;
  /** Rolling frozen-opponent archive used to pressure future generations. */
  opponentSnapshotPool: OpponentSnapshotPool;
}

/**
 * Shared scaffold for the smallest honest 2v2 race-pack harness.
 *
 * The harness keeps two isolated `TeamScopedState` objects plus one shared
 * `CollectiveEvaluationContext`. That context binds four agent slots to a
 * 28-float stigmergy field laid out as 4 rows × 7 channels, so the evaluation
 * seam can reason about the whole race pack while each team still owns its own
 * evolutionary controller.
 */
export interface TwoPopulationHarnessState {
  /** Team A runtime state. */
  readonly teamA: TeamScopedState;
  /** Team B runtime state. */
  readonly teamB: TeamScopedState;
  /** Shared four-slot context bound to the 28-float 2v2 radio field. */
  readonly sharedEvaluationContext: CollectiveEvaluationContext;
  /** Width of one row in the shared radio field; Phase 3 fixes this at 7 channels per car. */
  readonly radioChannelCount: number;
  /** Total scalar cell count in the shared row-major field; Phase 3 fixes this at 4 × 7 = 28. */
  readonly fieldSize: number;
}

/**
 * Creates the smallest honest Phase 3 two-population harness.
 *
 * The returned scaffold contains two isolated team controllers plus one shared
 * row-major radio field sized for four cars and seven channels per car
 * (`4 × 7 = 28` floats). The field layout matches the race-pack contract used
 * by the example worker seam: Team A occupies rows `0` and `1`, and Team B
 * occupies rows `2` and `3`.
 *
 * Aliasing is rejected when `injected.teamA` and `injected.teamB` point to the
 * same `Neat` instance because a two-population harness only makes sense when
 * each side owns independent mutation, speciation, and generation state. If
 * both teams shared one controller, one side could silently cross-mutate the
 * other.
 *
 * Use `injected` for owner-local tests or deterministic fixtures that need to
 * provide prebuilt controllers instead of allocating fresh ones.
 *
 * @param configA - Team A controller configuration.
 * @param configB - Team B controller configuration.
 * @param injected - Optional prebuilt team controllers for owner-local testing.
 * @returns Fully initialized two-population harness state.
 * @throws {Error} When both injected teams alias the same `Neat` instance.
 * @example
 * ```ts
 * const harness = createTwoPopulationHarness({}, {});
 *
 * harness.sharedEvaluationContext.agentCount; // 4
 * harness.radioChannelCount; // 7
 * harness.fieldSize; // 28
 * ```
 */
export function createTwoPopulationHarness(
  configA: NeatOptions,
  configB: NeatOptions,
  injected?: {
    readonly teamA?: Neat;
    readonly teamB?: Neat;
  },
): TwoPopulationHarnessState {
  // Step 1: Reject aliased controller injection so team state cannot cross-mutate.
  if (injected?.teamA !== undefined && injected.teamA === injected.teamB) {
    throw new Error(
      'Team A and Team B must be distinct Neat controller instances (alias detected)',
    );
  }

  // Step 2: Resolve the team-local controllers and shared four-car evaluation context.
  const teamAController =
    injected?.teamA ?? new Neat(undefined, undefined, undefined, configA);
  const teamBController =
    injected?.teamB ?? new Neat(undefined, undefined, undefined, configB);
  const radioField = createSharedField(
    TWO_POPULATION_RADIO_CHANNEL_COUNT,
    TWO_POPULATION_AGENT_COUNT,
  );

  return {
    teamA: {
      controller: teamAController,
      opponentSnapshotPool: createOpponentSnapshotPool(
        DEFAULT_OPPONENT_SNAPSHOT_CAPACITY,
      ),
    },
    teamB: {
      controller: teamBController,
      opponentSnapshotPool: createOpponentSnapshotPool(
        DEFAULT_OPPONENT_SNAPSHOT_CAPACITY,
      ),
    },
    sharedEvaluationContext: createCollectiveEvaluationContext(
      TWO_POPULATION_AGENT_COUNT,
      radioField,
    ),
    radioChannelCount: TWO_POPULATION_RADIO_CHANNEL_COUNT,
    fieldSize: TWO_POPULATION_FIELD_SIZE,
  };
}

/**
 * Produces the Phase 3 evaluation scaffold for one 2v2 race tick.
 *
 * This helper intentionally stays small: it partitions the shared four-car
 * context into Team A and Team B slices so the surrounding racing example can
 * exercise the two-population seam without pretending that full game-theory
 * evaluation already exists. Richer payoff shaping, opponent modeling, and
 * role-specialized coevolution stay Phase 4+ concerns.
 *
 * @param harness - Active two-population harness.
 * @param raceState - Current race state propagated to the placeholder results.
 * @returns Distinct Team A and Team B result slices aligned to the shared four-slot pack.
 * @example
 * ```ts
 * const harness = createTwoPopulationHarness({}, {});
 * const result = runTwoTeamEvaluationTick(harness, { tick: 12 });
 *
 * result.teamA.length; // 2
 * result.teamB.length; // 2
 * ```
 */
export function runTwoTeamEvaluationTick(
  harness: TwoPopulationHarnessState,
  raceState: unknown,
): {
  readonly teamA: readonly unknown[];
  readonly teamB: readonly unknown[];
} {
  // Step 1: Partition the four-car scaffold into the two Team A and two Team B slots.
  const teamBoundaryIndex =
    harness.sharedEvaluationContext.agentCount / TEAM_AGENT_COUNT;
  const teamAResults = Array.from(
    { length: teamBoundaryIndex },
    (_unusedValue, teamOffset) => ({
      team: 'A',
      agentIndex: teamOffset,
      raceState,
    }),
  );
  const teamBResults = Array.from(
    { length: teamBoundaryIndex },
    (_unusedValue, teamOffset) => ({
      team: 'B',
      agentIndex: teamOffset + teamBoundaryIndex,
      raceState,
    }),
  );

  return {
    teamA: teamAResults,
    teamB: teamBResults,
  };
}

/**
 * Advances both team controllers only after the shared generation barrier completes.
 *
 * The barrier is **shared**: a generation increment for either team — and any
 * mutation of either team's `opponentSnapshotPool` — is suppressed until **both**
 * teams have produced results for the completed shared race. An empty result
 * slice on either side means "the shared evaluation has not finished," not
 * "advance with zero fitness." This invariant keeps the rolling rival archive
 * aligned to the same generation tick on both sides, which is the smallest
 * honest contract needed for coevolution-style role specialization.
 *
 * Once the barrier is complete (both slices non-empty), each team's controller
 * advances by one generation, and each team's `opponentSnapshotPool` receives
 * one frozen deep-cloned snapshot of the opposing side's results. Snapshots are
 * produced through `addOpponentSnapshot`, which preserves the bounded FIFO
 * invariant and the deep-clone immutability contract.
 *
 * Transport-neutral by design: this function does not depend on packed
 * `race-step` frames, transfer lists, or worker topology. Those details are
 * deferred to Phase 4 transport normalization.
 *
 * @param harness - Active two-population harness.
 * @param resultsA - Team A evaluation results for the completed race.
 * @param resultsB - Team B evaluation results for the completed race.
 *
 * @example
 * ```ts
 * const harness = createTwoPopulationHarness({}, {});
 *
 * // Partial advance: Team A alone has results — barrier is NOT complete,
 * // so neither generation nor either snapshot pool is mutated.
 * advanceTwoPopulations(harness, [{ genomeId: 'team-a-0', fitness: 12 }], []);
 * harness.teamA.controller.generation; // 0
 * harness.teamB.opponentSnapshotPool.snapshots.length; // 0
 *
 * // Barrier complete: both sides advanced and cross-registered.
 * advanceTwoPopulations(
 *   harness,
 *   [{ genomeId: 'team-a-0', fitness: 12 }],
 *   [{ genomeId: 'team-b-0', fitness: 11 }],
 * );
 * harness.teamA.controller.generation; // 1
 * harness.teamB.controller.generation; // 1
 * harness.teamA.opponentSnapshotPool.snapshots.length; // 1
 * harness.teamB.opponentSnapshotPool.snapshots.length; // 1
 * ```
 */
export function advanceTwoPopulations(
  harness: TwoPopulationHarnessState,
  resultsA: readonly unknown[],
  resultsB: readonly unknown[],
): void {
  // Step 1: Shared barrier — refuse to mutate any team state until both
  //         sides produced results for the completed shared evaluation.
  if (resultsA.length === 0 || resultsB.length === 0) {
    return;
  }

  // Step 2: Barrier complete — advance both controllers by one generation.
  harness.teamA.controller.generation += 1;
  harness.teamB.controller.generation += 1;

  // Step 3: Cross-register the opposing-side results as frozen snapshots
  //         in each team's rolling archive. The snapshot payload is
  //         deep-cloned inside `addOpponentSnapshot`, so post-registration
  //         mutations of the result arrays do not corrupt the archive.
  harness.teamA.opponentSnapshotPool = addOpponentSnapshot(
    harness.teamA.opponentSnapshotPool,
    'team-b-champion',
    { results: resultsB },
    harness.teamB.controller.generation,
  );

  harness.teamB.opponentSnapshotPool = addOpponentSnapshot(
    harness.teamB.opponentSnapshotPool,
    'team-a-champion',
    { results: resultsA },
    harness.teamA.controller.generation,
  );
}
