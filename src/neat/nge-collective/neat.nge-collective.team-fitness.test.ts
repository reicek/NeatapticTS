type TeamMemberResult = {
  readonly memberId: string;
  readonly rawScore: number;
  readonly supportScore: number;
};

type TeamResultGroup<TTeamId extends string> = {
  readonly teamId: TTeamId;
  readonly memberResults: readonly TeamMemberResult[];
};

type TeamFitnessResult<TTeamId extends string> = {
  readonly teamId: TTeamId;
  readonly memberResults: readonly TeamMemberResult[];
  readonly teamFitness: number;
};

type TeamFitnessPolicy<TTeamId extends string> = (
  group: TeamResultGroup<TTeamId>,
) => number;

interface TeamFitnessModule {
  createTeamFitnessEvaluator<TTeamId extends string>(
    policy: TeamFitnessPolicy<TTeamId>,
  ): (
    groups: readonly TeamResultGroup<TTeamId>[],
  ) => readonly TeamFitnessResult<TTeamId>[];
}

interface NgeCollectiveBarrelModule {
  createTeamFitnessEvaluator?: TeamFitnessModule['createTeamFitnessEvaluator'];
}

describe('team-level fitness core evaluator', () => {
  describe('group aggregation seam', () => {
    it('aggregates one reusable team-fitness result per team group', async () => {
      const evaluatorInput = createGenericTeamGroups();

      await expect(
        loadTeamFitnessModule().then(({ createTeamFitnessEvaluator }) => {
          const evaluator = createTeamFitnessEvaluator(selectWeightedSignalScore);

          return evaluator(evaluatorInput).map(({ teamFitness, teamId }) => ({
            teamFitness,
            teamId,
          }));
        }),
      ).resolves.toEqual([
        { teamFitness: 10, teamId: 'team-alpha' },
        { teamFitness: 11, teamId: 'team-beta' },
      ]);
    });
  });

  describe('benchmark-facing reuse seam', () => {
    it('exports createTeamFitnessEvaluator from the nge-collective barrel', async () => {
      await expect(
        loadNgeCollectiveBarrelModule().then(
          (ngeCollectiveModule) =>
            typeof ngeCollectiveModule.createTeamFitnessEvaluator,
        ),
      ).resolves.toBe('function');
    });
  });

  describe('policy-bound behavior', () => {
    it('uses the injected aggregation policy instead of hard-coded benchmark scoring', async () => {
      const evaluatorInput = createGenericTeamGroups();

      await expect(
        loadTeamFitnessModule().then(({ createTeamFitnessEvaluator }) => {
          const evaluator = createTeamFitnessEvaluator(selectSupportWeightedRawScore);

          return evaluator(evaluatorInput).at(0)?.teamFitness;
        }),
      ).resolves.toBe(10);
    });
  });
});

function createGenericTeamGroups(): readonly TeamResultGroup<'team-alpha' | 'team-beta'>[] {
  return [
    {
      teamId: 'team-alpha',
      memberResults: [
        { memberId: 'alpha-0', rawScore: 4, supportScore: 1 },
        { memberId: 'alpha-1', rawScore: 3, supportScore: 2 },
      ],
    },
    {
      teamId: 'team-beta',
      memberResults: [
        { memberId: 'beta-0', rawScore: 2, supportScore: 3 },
        { memberId: 'beta-1', rawScore: 5, supportScore: 1 },
      ],
    },
  ];
}

function selectWeightedSignalScore<TTeamId extends string>(
  group: TeamResultGroup<TTeamId>,
): number {
  return group.memberResults.reduce(
    (totalScore, memberResult) =>
      totalScore + memberResult.rawScore + memberResult.supportScore,
    0,
  );
}

function selectSupportWeightedRawScore<TTeamId extends string>(
  group: TeamResultGroup<TTeamId>,
): number {
  return group.memberResults.reduce(
    (totalScore, memberResult) =>
      totalScore + memberResult.rawScore * memberResult.supportScore,
    0,
  );
}

async function loadTeamFitnessModule(): Promise<TeamFitnessModule> {
  const modulePath = './neat.nge-collective.team-fitness';
  return (await import(modulePath)) as TeamFitnessModule;
}

async function loadNgeCollectiveBarrelModule(): Promise<NgeCollectiveBarrelModule> {
  const modulePath = './neat.nge-collective';
  return (await import(modulePath)) as NgeCollectiveBarrelModule;
}
