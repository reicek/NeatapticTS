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
  if (
    injected?.teamA !== undefined &&
    injected.teamA === injected.teamB
  ) {
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
  const teamBoundaryIndex = harness.sharedEvaluationContext.agentCount / TEAM_AGENT_COUNT;
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
 * Advances both team controllers after a completed shared race.
 *
 * The advance contract is intentionally two-phase. First, each team only
 * advances when it actually produced results for the finished race; an empty
 * result slice means "no completed evaluation," not "advance with zero
 * fitness." Second, every non-empty side cross-registers its results into the
 * opposing `opponentSnapshotPool` via `addOpponentSnapshot`, preserving a
 * rolling history of frozen rival champions.
 *
 * That rolling archive matters because future generations should not face only
 * the opponent's latest mutable state. They instead encounter recent frozen
 * rivals, which is the first honest selection pressure needed for role
 * specialization to emerge.
 *
 * @param harness - Active two-population harness.
 * @param resultsA - Team A evaluation results for the completed race.
 * @param resultsB - Team B evaluation results for the completed race.
 * @example
 * ```ts
 * const harness = createTwoPopulationHarness({}, {});
 *
 * advanceTwoPopulations(
 *   harness,
 *   [{ genomeId: 'team-a-0', fitness: 12 }],
 *   [{ genomeId: 'team-b-0', fitness: 11 }],
 * );
 *
 * harness.teamA.controller.generation; // 1
 * harness.teamB.opponentSnapshotPool.snapshots.length; // 1
 * ```
 */
export function advanceTwoPopulations(
  harness: TwoPopulationHarnessState,
  resultsA: readonly unknown[],
  resultsB: readonly unknown[],
): void {
  // Step 1: Advance each team only when it produced results for the completed race.
  if (resultsA.length > 0) {
    harness.teamA.controller.generation += 1;
  }

  if (resultsB.length > 0) {
    harness.teamB.controller.generation += 1;
  }

  // Step 2: Cross-register the opponent champions into the opposing snapshot pools.
  if (resultsB.length > 0) {
    harness.teamA.opponentSnapshotPool = addOpponentSnapshot(
      harness.teamA.opponentSnapshotPool,
      'team-b-champion',
      { results: resultsB },
      harness.teamB.controller.generation,
    );
  }

  if (resultsA.length > 0) {
    harness.teamB.opponentSnapshotPool = addOpponentSnapshot(
      harness.teamB.opponentSnapshotPool,
      'team-a-champion',
      { results: resultsA },
      harness.teamA.controller.generation,
    );
  }
}
