/**
 * Red-phase contracts for the FSM polyandric reproduction integration.
 *
 * Step 04 will wire `reproducePolyandric` and `activateNgeNetworkFromEnvelope`
 * into `transitionToGenerationReady` inside
 * `simulation-worker.evolution.protocol.service.ts`.  These tests fail honestly
 * today because the production code neither imports nor calls those operators.
 *
 * All tests use deterministic fixtures and a mocked `RaceEpisodeRunner` whose
 * race is already finished, so no real physics or randomness is exercised.
 * Single-expect rule is enforced throughout.
 */

import type { NgeDnaCanonicalEnvelope } from '../../../../src/neat/nge-dna/neat.nge-dna.types';
import { Network } from '../../../../src/browser-entry.ts';
import { reproducePolyandric } from '../../../../src/neat/nge-evolution/neat.nge-evolution';
import { activateNgeNetworkFromEnvelope } from '../../../../src/neat/nge-dna/neat.nge-dna.operator';
import type { RacingRenderFrame } from './simulation-worker.types';
import type { RaceEpisodeRunner } from './simulation-worker.race-pack.service';
import * as coevolutionModule from './simulation-worker.coevolution.service';
import {
  createInitialProtocolState,
  routeRacingWorkerProtocolMessage,
} from './simulation-worker.evolution.protocol.service';

// ---------------------------------------------------------------------------
// Mutable runner factory used by the race-pack mock.
//
// Jest hoists `jest.mock` above `let` declarations, so the factory captures a
// live binding: the mock implementation simply delegates here, and individual
// tests reassign this variable to inject custom runners.  This avoids the
// brittle `mockReturnValueOnce` queue, which was being reset by
// `jest.clearAllMocks()` before the FSM consumed it.
// ---------------------------------------------------------------------------
let activeRaceRunnerFactory: (agentCount: number, networks?: readonly unknown[]) => RaceEpisodeRunner = (
  agentCount,
) => createDeterministicDoneRunner(agentCount);

// ---------------------------------------------------------------------------
// Module mocks — keep real implementations where possible, spy on boundaries
// ---------------------------------------------------------------------------

jest.mock('./simulation-worker.coevolution.service', () => {
  const actual = jest.requireActual(
    './simulation-worker.coevolution.service',
  ) as typeof import('./simulation-worker.coevolution.service');
  return {
    ...actual,
    selectQueenPerTeam: jest.fn(
      (carFinishPositions: readonly number[], teamLayout: readonly (0 | 1)[]) =>
        actual.selectQueenPerTeam(carFinishPositions, teamLayout),
    ),
  };
});

jest.mock('./simulation-worker.race-pack.service', () => {
  const actual = jest.requireActual(
    './simulation-worker.race-pack.service',
  ) as typeof import('./simulation-worker.race-pack.service');
  return {
    ...actual,
    createRaceEpisodeRunner: jest.fn(
      (_seed: number, _snapshot: unknown, networks: readonly unknown[]) =>
        activeRaceRunnerFactory(networks.length, networks),
    ),
  };
});

jest.mock('../../../../src/neat/nge-evolution/neat.nge-evolution', () => ({
  reproducePolyandric: jest.fn(() => ({}) as NgeDnaCanonicalEnvelope),
}));

jest.mock('../../../../src/neat/nge-dna/neat.nge-dna.operator', () => ({
  activateNgeNetworkFromEnvelope: jest.fn(() => new Network(2, 2, { seed: 1 })),
}));

const mockedSelectQueenPerTeam = jest.mocked(
  coevolutionModule.selectQueenPerTeam,
);
const mockedReproducePolyandric = jest.mocked(reproducePolyandric);
const mockedActivateNgeNetworkFromEnvelope = jest.mocked(
  activateNgeNetworkFromEnvelope,
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Resolves the fixed team layout for the supported racing tiers. */
function resolveTeamLayout(agentCount: number): number[] {
  switch (agentCount) {
    case 2:
      return [0, 1];
    case 4:
      return [0, 0, 1, 1];
    case 6:
      return [0, 0, 0, 1, 1, 1];
    default:
      return Array.from({ length: agentCount }, (_, index) => index % 2);
  }
}

/**
 * Returns lap-time ticks that produce deterministic finish positions.
 *
 * - 6 cars (Tier 5): positions [2, 5, 1, 3, 4, 6]
 * - 4 cars (Tier 3-4): positions [2, 1, 3, 4]
 * - 2 cars (Tier 1-2): positions [2, 1]
 */
function resolveLapTimeTicks(agentCount: number): number[] {
  switch (agentCount) {
    case 6:
      return [100, 300, 50, 200, 250, 400];
    case 4:
      return [200, 100, 300, 400];
    case 2:
      return [200, 100];
    default:
      return Array.from(
        { length: agentCount },
        (_, index) => (index + 1) * 100,
      );
  }
}

/**
 * Drives the protocol FSM from idle through a finished race so that
 * `transitionToGenerationReady` is executed.
 */
function runFsmToGenerationReady(
  tier: number,
  populationSize = 10,
  rngSeed = 42,
) {
  let state = createInitialProtocolState();
  state = routeRacingWorkerProtocolMessage(
    { type: 'init', populationSize, rngSeed, tier },
    state,
  ).nextState;
  state = routeRacingWorkerProtocolMessage(
    { type: 'request-generation' },
    state,
  ).nextState;
  state = routeRacingWorkerProtocolMessage(
    { type: 'start-race', tierConfig: {}, opponentSnapshotId: 'snap-0' },
    state,
  ).nextState;
  return routeRacingWorkerProtocolMessage(
    { type: 'request-race-step', requestId: 'r-1', stepsToAdvance: 1 },
    state,
  );
}
/**
 * Builds a minimal mock `RaceEpisodeRunner` whose race is already finished.
 *
 * The returned runner exposes the lap-completion fields required by
 * `tryExtractFinishPositions` in the protocol service, plus the team layout
 * and pit-lap fields consumed by `extractPitLapDistribution`.
 */
function createDeterministicDoneRunner(agentCount: number): RaceEpisodeRunner {
  const frame: RacingRenderFrame & { progress01: Float32Array } = {
    schemaVersion: 'racing-packed-v1',
    tick: 1,
    seed: 42,
    trackId: 42,
    agentCount,
    featureFlags: 0,
    carX: new Float32Array(agentCount),
    carY: new Float32Array(agentCount),
    carHeading: new Float32Array(agentCount),
    carActive: new Uint8Array(agentCount).fill(1),
    carTeam: new Uint8Array(resolveTeamLayout(agentCount)),
    carMode: new Uint8Array(agentCount),
    tireState: new Float32Array(agentCount * 4).fill(1),
    radioField: new Float32Array(0),
    lap: new Uint16Array(agentCount).fill(1),
    place: new Uint8Array(agentCount),
    raceTimeMs: 0,
    done: true,
    progress01: new Float32Array(agentCount).fill(1),
  };

  return {
    frame,
    tick: jest.fn(),
    computeFitness: () => 0,
    resolveTeamFitness: () => 0,
    createRaceStepMessage: () => ({
      type: 'race-step' as const,
      frame,
      transferList: [],
    }),
    lapCompleted: new Uint8Array(agentCount).fill(1),
    lapTimeTicks: new Uint32Array(resolveLapTimeTicks(agentCount)),
    pitLapPerCar: new Uint16Array(agentCount),
    endedOffTrack: false,
    endedOffTrackPerCar: new Uint8Array(agentCount),
    guidingLines: Array.from({ length: agentCount }, () => []),
    serializeVisualizationNetwork: jest.fn(() => new Float32Array([0.1, 0.2])),
  } as unknown as RaceEpisodeRunner;
}

/**
 * Builds a runner with no lap-completion data and no `computeFitness` so the
 * protocol must fall back to car-index-derived fitness scores.
 */
function createNoLapDataRunner(agentCount: number): RaceEpisodeRunner {
  const base = createDeterministicDoneRunner(agentCount);
  return {
    ...base,
    computeFitness: undefined,
    lapCompleted: undefined,
    lapTimeTicks: undefined,
    frame: { ...base.frame, progress01: undefined },
  } as unknown as RaceEpisodeRunner;
}

/**
 * Builds a runner with a mix of completed and uncompleted cars so the ranking
 * comparator exercises all three sorting arms.
 */
function createMixedCompletionRunner(agentCount: number): RaceEpisodeRunner {
  const base = createDeterministicDoneRunner(agentCount);
  if (agentCount < 4) {
    throw new Error('createMixedCompletionRunner requires at least 4 cars');
  }
  const lapCompleted = new Uint8Array(agentCount);
  const lapTimeTicks = new Uint32Array(agentCount);
  const progress01 = new Float32Array(agentCount);
  // Car 0: completed fastest
  lapCompleted[0] = 1;
  lapTimeTicks[0] = 100;
  progress01[0] = 0.9;
  // Car 1: completed more slowly
  lapCompleted[1] = 1;
  lapTimeTicks[1] = 200;
  progress01[1] = 0.8;
  // Car 2: not completed, lowest progress
  lapCompleted[2] = 0;
  lapTimeTicks[2] = 0;
  progress01[2] = 0.5;
  // Car 3: not completed, highest progress
  lapCompleted[3] = 0;
  lapTimeTicks[3] = 0;
  progress01[3] = 0.95;
  return {
    ...base,
    lapCompleted,
    lapTimeTicks,
    frame: { ...base.frame, progress01 },
  } as unknown as RaceEpisodeRunner;
}

/**
 * Builds a mock `RaceEpisodeRunner` whose race is not finished after one tick.
 *
 * The `tick` callback optionally invokes the provided controller networks so
 * the per-car `activate` wrapper created in `createRaceRunnerForState` is
 * exercised.
 */
function createActiveRunner(
  agentCount: number,
  controllerNetworks?: readonly { activate(inputs: number[]): number[] }[],
): RaceEpisodeRunner {
  const base = createDeterministicDoneRunner(agentCount);
  return {
    ...base,
    frame: { ...base.frame, done: false },
    tick: () => {
      controllerNetworks?.forEach((network) => network.activate([0, 0]));
    },
  } as unknown as RaceEpisodeRunner;
}

/**
 * Builds a two-car runner where one car completed a lap and the other did not.
 *
 * This exercises the mixed-completion sorting arms in both fitness extraction
 * and finish-position rank extraction while avoiding the computeFitness fast
 * path.
 */
function createTwoCarMixedNoFitnessRunner(): RaceEpisodeRunner {
  const agentCount = 2;
  const base = createDeterministicDoneRunner(agentCount);
  const lapCompleted = new Uint8Array(agentCount);
  const lapTimeTicks = new Uint32Array(agentCount);
  const progress01 = new Float32Array(agentCount);
  // Car 0 did not finish; car 1 finished in 100 ticks.
  lapCompleted[0] = 0;
  lapTimeTicks[0] = 0;
  progress01[0] = 0.5;
  lapCompleted[1] = 1;
  lapTimeTicks[1] = 100;
  progress01[1] = 1;
  return {
    ...base,
    computeFitness: undefined,
    lapCompleted,
    lapTimeTicks,
    frame: { ...base.frame, progress01, done: true },
  } as unknown as RaceEpisodeRunner;
}

/** Expected polyandric reproduction policy passed to `reproducePolyandric`. */
const EXPECTED_POLYANDRIC_POLICY = {
  mode: 'polyandric',
  polyandricDroneCount: 2,
  polyandricDroneContributionFraction: 0.25,
  queenBias: 0.85,
  assignedRegionStrategy: 'non-overlapping',
  modeIsEvolvable: true,
  seedPolicy: 'queen-weighted',
  parthenogenesisMutationRate: 0,
};

// ---------------------------------------------------------------------------
// Red tests
// ---------------------------------------------------------------------------

describe('FSM polyandric reproduction integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    activeRaceRunnerFactory = (agentCount) =>
      createDeterministicDoneRunner(agentCount);
  });

  describe('queen selection', () => {
    it('calls selectQueenPerTeam with derived finish positions and team layout for Tier 5', () => {
      runFsmToGenerationReady(5, 10, 42);

      expect(mockedSelectQueenPerTeam).toHaveBeenCalledWith(
        [2, 5, 1, 3, 4, 6],
        [0, 0, 0, 1, 1, 1],
      );
    });
  });

  describe('finish position extraction', () => {
    it('falls back to carFitnessScores when finish position ranks cannot be extracted', () => {
      activeRaceRunnerFactory = (agentCount) =>
        createNoLapDataRunner(agentCount);
      runFsmToGenerationReady(3, 10, 42);

      expect(mockedSelectQueenPerTeam).toHaveBeenCalledWith(
        [1, 2, 3, 4],
        [0, 0, 1, 1],
      );
    });

    it('orders mixed lap-completion results by completed laps then progress', () => {
      activeRaceRunnerFactory = (agentCount) =>
        createMixedCompletionRunner(agentCount);
      runFsmToGenerationReady(3, 10, 42);

      expect(mockedSelectQueenPerTeam).toHaveBeenCalledWith(
        [1, 2, 4, 3],
        [0, 0, 1, 1],
      );
    });
    it('ranks a two-car race with one finisher ahead of one non-finisher', () => {
      activeRaceRunnerFactory = () => createTwoCarMixedNoFitnessRunner();
      runFsmToGenerationReady(1, 10, 42);

      expect(mockedSelectQueenPerTeam).toHaveBeenCalledWith([2, 1], [0, 1]);
    });
  });

  describe('polyandric reproduction calls', () => {
    it('calls reproducePolyandric once per team', () => {
      runFsmToGenerationReady(5, 10, 42);

      expect(mockedReproducePolyandric).toHaveBeenCalledTimes(2);
    });

    it('passes the expected polyandric policy values', () => {
      runFsmToGenerationReady(5, 10, 42);
      const firstInput = mockedReproducePolyandric.mock.calls[0]?.[0];

      expect(firstInput?.policy).toEqual(EXPECTED_POLYANDRIC_POLICY);
    });

    it('passes a queen envelope and two same-team drone envelopes for Tier 5', () => {
      runFsmToGenerationReady(5, 10, 42);
      const firstInput = mockedReproducePolyandric.mock.calls[0]?.[0];

      expect(firstInput?.drones).toHaveLength(2);
    });

    it('caps drones to zero for Tier 1–2 (one car per team)', () => {
      runFsmToGenerationReady(1, 10, 42);
      const firstInput = mockedReproducePolyandric.mock.calls[0]?.[0];

      expect(firstInput?.drones).toHaveLength(0);
    });

    it('caps drones to one for Tier 3–4 (two cars per team)', () => {
      runFsmToGenerationReady(3, 10, 42);
      const firstInput = mockedReproducePolyandric.mock.calls[0]?.[0];

      expect(firstInput?.drones).toHaveLength(1);
    });
  });

  describe('offspring network materialization', () => {
    it('materializes one offspring Network per car via activateNgeNetworkFromEnvelope', () => {
      runFsmToGenerationReady(5, 10, 42);

      expect(mockedActivateNgeNetworkFromEnvelope).toHaveBeenCalledTimes(12);
    });

    it('uses deterministic per-car seed formula for offspring materialization', () => {
      runFsmToGenerationReady(5, 10, 42);

      expect(mockedActivateNgeNetworkFromEnvelope).toHaveBeenCalledWith(
        expect.any(Object),
        10042,
      );
    });
  });

  describe('offspring envelope extraction', () => {
    it('unwraps an { offspring } envelope before materializing the network', () => {
      const offspringEnvelope = {
        marker: 'offspring',
      } as unknown as NgeDnaCanonicalEnvelope;
      mockedReproducePolyandric.mockReturnValueOnce({
        offspring: offspringEnvelope,
      } as unknown as ReturnType<typeof reproducePolyandric>);
      activeRaceRunnerFactory = (agentCount) =>
        createDeterministicDoneRunner(agentCount);
      runFsmToGenerationReady(3, 10, 42);

      expect(
        mockedActivateNgeNetworkFromEnvelope.mock.calls.filter(
          ([envelope, seed]) =>
            envelope === offspringEnvelope &&
            (seed === 10042 || seed === 10043),
        ),
      ).toHaveLength(2);
    });
  });

  describe('determinism', () => {
    it('produces identical reproduction calls when the same FSM state is replayed', () => {
      runFsmToGenerationReady(5, 10, 42);
      const firstRunCalls = [...mockedReproducePolyandric.mock.calls];

      jest.clearAllMocks();
      runFsmToGenerationReady(5, 10, 42);
      const secondRunCalls = [...mockedReproducePolyandric.mock.calls];

      expect(firstRunCalls).toEqual(secondRunCalls);
    });
  });

  describe('protocol edge coverage', () => {
    it('leaves raceRunner undefined when start-race is routed without a coevolution container', () => {
      const state = {
        ...createInitialProtocolState(),
        phase: "generation-ready" as const,
      };
      const result = routeRacingWorkerProtocolMessage(
        {
          type: "start-race",
          tierConfig: {},
          opponentSnapshotId: "snap-0",
        },
        state,
      );

      expect(result.nextState).toMatchObject({
        phase: "racing",
        raceRunner: undefined,
      });
    });

    it('returns an error response when request-race-step is routed without a raceRunner', () => {
      const state = {
        ...createInitialProtocolState(),
        phase: "racing" as const,
      };
      const result = routeRacingWorkerProtocolMessage(
        {
          type: "request-race-step",
          requestId: "r-no-runner",
          stepsToAdvance: 1,
        },
        state,
      );

      expect(result.response).toEqual({
        type: "error",
        message: "No race episode runner available.",
      });
    });

    it('returns an unfinished race-step response and exercises per-car controller activation', () => {
      activeRaceRunnerFactory = (_agentCount, networks) =>
        createActiveRunner(
          networks?.length ?? _agentCount,
          networks as unknown as readonly { activate(inputs: number[]): number[] }[],
        );
      let state = createInitialProtocolState();
      state = routeRacingWorkerProtocolMessage(
        { type: "init", populationSize: 10, rngSeed: 42, tier: 1 },
        state,
      ).nextState;
      state = routeRacingWorkerProtocolMessage(
        { type: "request-generation" },
        state,
      ).nextState;
      state = routeRacingWorkerProtocolMessage(
        {
          type: "start-race",
          tierConfig: {},
          opponentSnapshotId: "snap-0",
        },
        state,
      ).nextState;
      const result = routeRacingWorkerProtocolMessage(
        {
          type: "request-race-step",
          requestId: "r-unfinished",
          stepsToAdvance: 1,
        },
        state,
      );

      expect(result.response).toMatchObject({
        type: "race-step",
        requestId: "r-unfinished",
        done: false,
      });
    });
  });
});
