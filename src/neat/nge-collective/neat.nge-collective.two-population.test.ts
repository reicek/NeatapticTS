import Neat, { type NeatOptions } from '../../neat';
import type {
  CollectiveEvaluationContext,
  OpponentSnapshotPool,
} from './neat.nge-collective.types';

interface TeamScopedState {
  readonly controller: Neat;
  readonly opponentSnapshotPool: OpponentSnapshotPool;
}

interface TwoPopulationHarnessState {
  readonly teamA: TeamScopedState;
  readonly teamB: TeamScopedState;
  readonly sharedEvaluationContext: CollectiveEvaluationContext;
  readonly radioChannelCount: number;
  readonly fieldSize: number;
}

interface TwoPopulationHarnessModule {
  createTwoPopulationHarness(
    configA: NeatOptions,
    configB: NeatOptions,
    injectedControllers?: {
      readonly teamA?: Neat;
      readonly teamB?: Neat;
    },
  ): TwoPopulationHarnessState;
  runTwoTeamEvaluationTick(
    harness: TwoPopulationHarnessState,
    raceState: unknown,
  ): {
    readonly teamA: readonly unknown[];
    readonly teamB: readonly unknown[];
  };
  advanceTwoPopulations(
    harness: TwoPopulationHarnessState,
    resultsA: readonly unknown[],
    resultsB: readonly unknown[],
  ): void;
}

/**
 * Red-phase contracts for the missing Phase 3 two-population harness seam.
 *
 * The production boundary does not exist yet, so these tests intentionally stay
 * red until `neat.nge-collective.two-population.ts` lands.
 */
describe('two-population harness', () => {
  describe('createTwoPopulationHarness', () => {
    it('creates a harness with two distinct TeamScopedState instances', async () => {
      const configA = createHarnessConfig();
      const configB = createHarnessConfig();

      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness }) => {
            const harness = createTwoPopulationHarness(configA, configB);
            return {
              distinctTeamStates: harness.teamA !== harness.teamB,
              distinctControllers:
                harness.teamA.controller !== harness.teamB.controller,
              distinctSnapshotPools:
                harness.teamA.opponentSnapshotPool !==
                harness.teamB.opponentSnapshotPool,
            };
          },
        ),
      ).resolves.toEqual({
        distinctTeamStates: true,
        distinctControllers: true,
        distinctSnapshotPools: true,
      });
    });

    it('rejects aliased team controllers (same Neat instance for both teams)', async () => {
      const sharedController = new Neat();

      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness }) =>
            createTwoPopulationHarness(
              createHarnessConfig(),
              createHarnessConfig(),
              {
                teamA: sharedController,
                teamB: sharedController,
              },
            ),
        ),
      ).rejects.toThrow(/alias|shared|distinct/i);
    });

    it('sets shared evaluation context agentCount to 4 for 2v2', async () => {
      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness }) => {
            const harness = createTwoPopulationHarness(
              createHarnessConfig(),
              createHarnessConfig(),
            );
            return harness.sharedEvaluationContext.agentCount;
          },
        ),
      ).resolves.toBe(4);
    });
  });

  describe('runTwoTeamEvaluationTick', () => {
    it('returns results partitioned into teamA and teamB slices', async () => {
      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness, runTwoTeamEvaluationTick }) => {
            const harness = createTwoPopulationHarness(
              createHarnessConfig(),
              createHarnessConfig(),
            );
            const evaluationResult = runTwoTeamEvaluationTick(
              harness,
              createRaceStateFixture(),
            );
            return {
              teamACount: evaluationResult.teamA.length,
              teamBCount: evaluationResult.teamB.length,
            };
          },
        ),
      ).resolves.toEqual({
        teamACount: 2,
        teamBCount: 2,
      });
    });

    it('keeps Team A and Team B result slices non-overlapping', async () => {
      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness, runTwoTeamEvaluationTick }) => {
            const harness = createTwoPopulationHarness(
              createHarnessConfig(),
              createHarnessConfig(),
            );
            const evaluationResult = runTwoTeamEvaluationTick(
              harness,
              createRaceStateFixture(),
            );
            return {
              distinctArrays: evaluationResult.teamA !== evaluationResult.teamB,
              noSharedEntries: evaluationResult.teamA.every(
                (teamAResult) => !evaluationResult.teamB.includes(teamAResult),
              ),
            };
          },
        ),
      ).resolves.toEqual({
        distinctArrays: true,
        noSharedEntries: true,
      });
    });
  });

  describe('advanceTwoPopulations', () => {
    it('keeps both generations at 0 when Team A has results but Team B does not (shared barrier)', async () => {
      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness, advanceTwoPopulations }) => {
            const harness = createTwoPopulationHarness(
              createHarnessConfig(),
              createHarnessConfig(),
            );

            advanceTwoPopulations(harness, createTeamResults('team-a'), []);

            return {
              teamAGeneration: harness.teamA.controller.generation,
              teamBGeneration: harness.teamB.controller.generation,
            };
          },
        ),
      ).resolves.toEqual({
        teamAGeneration: 0,
        teamBGeneration: 0,
      });
    });

    it('keeps both team generations at 0 until the shared barrier completes', async () => {
      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness, advanceTwoPopulations }) => {
            const harness = createTwoPopulationHarness(
              createHarnessConfig(),
              createHarnessConfig(),
            );
            const initialTeamAGeneration = harness.teamA.controller.generation;
            const initialTeamBGeneration = harness.teamB.controller.generation;

            advanceTwoPopulations(harness, createTeamResults('team-a'), []);

            return {
              teamAGeneration: harness.teamA.controller.generation,
              teamBGeneration: harness.teamB.controller.generation,
              initialTeamAGeneration,
              initialTeamBGeneration,
            };
          },
        ),
      ).resolves.toEqual({
        teamAGeneration: 0,
        teamBGeneration: 0,
        initialTeamAGeneration: 0,
        initialTeamBGeneration: 0,
      });
    });

    it('keeps both generations at 0 when Team B has results but Team A does not (shared barrier)', async () => {
      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness, advanceTwoPopulations }) => {
            const harness = createTwoPopulationHarness(
              createHarnessConfig(),
              createHarnessConfig(),
            );

            advanceTwoPopulations(harness, [], createTeamResults('team-b'));

            return {
              teamAGeneration: harness.teamA.controller.generation,
              teamBGeneration: harness.teamB.controller.generation,
            };
          },
        ),
      ).resolves.toEqual({
        teamAGeneration: 0,
        teamBGeneration: 0,
      });
    });

    it('keeps both opponent snapshot pools unchanged until the shared barrier completes', async () => {
      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness, advanceTwoPopulations }) => {
            const harness = createTwoPopulationHarness(
              createHarnessConfig(),
              createHarnessConfig(),
            );
            const initialTeamASnapshotPool = harness.teamA.opponentSnapshotPool;
            const initialTeamBSnapshotPool = harness.teamB.opponentSnapshotPool;

            advanceTwoPopulations(harness, createTeamResults('team-a'), []);

            return {
              teamASnapshotPoolUnchanged:
                harness.teamA.opponentSnapshotPool === initialTeamASnapshotPool,
              teamBSnapshotPoolUnchanged:
                harness.teamB.opponentSnapshotPool === initialTeamBSnapshotPool,
              teamASnapshotCount:
                harness.teamA.opponentSnapshotPool.snapshots.length,
              teamBSnapshotCount:
                harness.teamB.opponentSnapshotPool.snapshots.length,
            };
          },
        ),
      ).resolves.toEqual({
        teamASnapshotPoolUnchanged: true,
        teamBSnapshotPoolUnchanged: true,
        teamASnapshotCount: 0,
        teamBSnapshotCount: 0,
      });
    });

    it('registers opponent snapshots from Team B into Team A snapshot pool after Team A advance', async () => {
      await expect(
        loadTwoPopulationHarnessModule().then(
          ({ createTwoPopulationHarness, advanceTwoPopulations }) => {
            const harness = createTwoPopulationHarness(
              createHarnessConfig(),
              createHarnessConfig(),
            );

            advanceTwoPopulations(
              harness,
              createTeamResults('team-a'),
              createTeamResults('team-b'),
            );

            return harness.teamA.opponentSnapshotPool.snapshots.length;
          },
        ),
      ).resolves.toBe(1);
    });
  });
});

function createHarnessConfig(): NeatOptions {
  return {};
}

function createRaceStateFixture(): {
  readonly tick: number;
  readonly carOrder: readonly string[];
} {
  return {
    tick: 12,
    carOrder: ['A0', 'A1', 'B0', 'B1'],
  };
}

function createTeamResults(teamLabel: 'team-a' | 'team-b'): readonly {
  readonly genomeId: string;
  readonly fitness: number;
}[] {
  return [
    {
      genomeId: `${teamLabel}-genome-0`,
      fitness: 1,
    },
  ];
}

async function loadTwoPopulationHarnessModule(): Promise<TwoPopulationHarnessModule> {
  const modulePath = './neat.nge-collective.two-population';
  return (await import(modulePath)) as TwoPopulationHarnessModule;
}
