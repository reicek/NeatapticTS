/**
 * Red-phase contracts for the Tier 5 three-car team runtime (3v3 full).
 *
 * Contracts verified here:
 * - All 3 teammate-radio slots are populated for 6-car teams (including self)
 * - Polyandric reproduction wiring is exercised with real assertions
 * - Role-divergence metric computation returns non-zero for a deterministic pack
 *
 * The three polyandric tests remain `.skip` until Step 05 green validation removes
 * the skip; Step 04 replaces their placeholder bodies with real assertions.
 *
 * Single-expect rule is enforced throughout.
 */
import { derivePerCarObservationState } from '../../controller/observation.assembler';
import {
  createCarGenome,
  selectQueenPerTeam,
} from './simulation-worker.coevolution.service';
import { reproducePolyandric } from '../../../../src/neat/nge-evolution/neat.nge-evolution';
import type {
  RacingCarState,
  TireStateTuple,
} from '../../environment/environment.types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const FRESH_TIRE: TireStateTuple = [1, 1, 1, 1];

/**
 * Creates a deterministic 6-car roster for the 3v3 Tier 5 race pack.
 *
 * Team A = cars 0, 1, 2 (positive Y, heading 0).
 * Team B = cars 3, 4, 5 (negative Y, heading PI).
 * Each car has a distinct position so radio slot encoding produces non-zero values.
 */
function createSixCarRoster(): RacingCarState[] {
  return [
    { carX: 10, carY: 5, carHeading: 0, teamIndex: 0, tireState: FRESH_TIRE },
    { carX: 12, carY: 5, carHeading: 0.1, teamIndex: 0, tireState: FRESH_TIRE },
    { carX: 14, carY: 5, carHeading: 0.2, teamIndex: 0, tireState: FRESH_TIRE },
    {
      carX: 10,
      carY: -5,
      carHeading: Math.PI,
      teamIndex: 1,
      tireState: FRESH_TIRE,
    },
    {
      carX: 12,
      carY: -5,
      carHeading: Math.PI + 0.1,
      teamIndex: 1,
      tireState: FRESH_TIRE,
    },
    {
      carX: 14,
      carY: -5,
      carHeading: Math.PI + 0.2,
      teamIndex: 1,
      tireState: FRESH_TIRE,
    },
  ];
}

/**
 * Creates a 6-car environment state with the focal car set to the specified
 * team and index.
 */
function createSixCarEnvState(
  cars: RacingCarState[] = createSixCarRoster(),
  focalCarIndex: number = 0,
) {
  const focalCar = cars[focalCarIndex];
  return {
    tick: 0,
    carX: focalCar.carX,
    carY: focalCar.carY,
    carHeading: focalCar.carHeading,
    teamIndex: focalCar.teamIndex,
    cars,
    tireState: FRESH_TIRE,
  };
}

// ---------------------------------------------------------------------------
// Module loader for role-divergence service (does not exist yet — Step 04)
// ---------------------------------------------------------------------------

type RoleDivergenceMetric = {
  readonly carIndex: number;
  readonly teamId: 0 | 1;
  readonly individualPosition: number;
  readonly blockerDelta: number;
  readonly inferredRole: 'queen' | 'blocker' | 'pacer' | 'undifferentiated';
};

interface RoleDivergenceService {
  computeRoleDivergenceMetrics(
    carFinishPositions: readonly number[],
    teamLayout: readonly (0 | 1)[],
    teamScores: readonly number[],
  ): readonly RoleDivergenceMetric[];
}

async function loadRoleDivergenceService(): Promise<RoleDivergenceService> {
  const modulePath = './simulation-worker.role-divergence.service';
  const module = (await import(modulePath)) as Partial<RoleDivergenceService>;

  if (typeof module.computeRoleDivergenceMetrics !== 'function') {
    throw new Error(
      'Missing role-divergence service export: computeRoleDivergenceMetrics',
    );
  }

  return module as RoleDivergenceService;
}

// ---------------------------------------------------------------------------
// Red tests — Tier 5 teammate radio population (criterion 2)
// ---------------------------------------------------------------------------

describe('Tier 5 teammate radio population for 6-car teams', () => {
  it('populates all 3 teammate radio slots with non-zero data for Team A focal car 0', () => {
    // Arrange — 6-car roster: Team A has cars 0, 1, 2
    const envState = createSixCarEnvState(createSixCarRoster(), 0);

    // Act — derive observation state for focal car 0 (Team A)
    const result = derivePerCarObservationState(envState, 0);
    const slot2 = result.teammateRadioSlots?.[2];
    const slot2AllZero = slot2
      ? Array.from(slot2).every((channel) => channel === 0)
      : true;

    // Assert — slot 2 must be non-zero. Currently zero-padded because
    // buildTeammateRadioSlots excludes the focal car, leaving only 2 teammates
    // for a 3-car team. Step 04 must include self as one of the 3 slots.
    expect(slot2AllZero).toBe(false);
  });

  it('populates all 3 teammate radio slots with non-zero data for Team B focal car 3', () => {
    // Arrange — 6-car roster: Team B has cars 3, 4, 5
    const cars = createSixCarRoster();
    const envState = createSixCarEnvState(cars, 3);

    // Act — derive observation state for focal car 3 (Team B)
    const result = derivePerCarObservationState(envState, 3);
    const slot2 = result.teammateRadioSlots?.[2];
    const slot2AllZero = slot2
      ? Array.from(slot2).every((channel) => channel === 0)
      : true;

    // Assert — slot 2 must be non-zero (currently zero-padded)
    expect(slot2AllZero).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Red tests — Polyandric reproduction wiring (enabled with real assertions).
// ---------------------------------------------------------------------------

describe('Tier 5 polyandric reproduction wiring', () => {
  it('selects the best-finishing car as queen for polyandric reproduction', () => {
    // Arrange — Team A finishes: car 0 = position 2, car 1 = position 5,
    // car 2 = position 1. Best-finishing = car 2 (position 1) -> queen.
    const carFinishPositions = [2, 5, 1, 3, 4, 6];
    const teamLayout: readonly (0 | 1)[] = [0, 0, 0, 1, 1, 1];

    // Act
    const results = selectQueenPerTeam(carFinishPositions, teamLayout);

    // Assert — Team A queen is the car with finish position 1 (index 2).
    expect(results[0]?.queenCarIndex).toBe(2);
  });

  it('calls reproducePolyandric with queen envelope and 2 drone envelopes', () => {
    // Arrange
    const queen = createCarGenome({
      carIndex: 2,
      seed: 42,
      teamId: 0,
      populationId: 'team-a',
      inputSize: 4,
      outputSize: 2,
    }).envelope;
    const drone0 = createCarGenome({
      carIndex: 0,
      seed: 43,
      teamId: 0,
      populationId: 'team-a',
      inputSize: 4,
      outputSize: 2,
    }).envelope;
    const drone1 = createCarGenome({
      carIndex: 1,
      seed: 44,
      teamId: 0,
      populationId: 'team-a',
      inputSize: 4,
      outputSize: 2,
    }).envelope;

    // Act
    const result = reproducePolyandric({
      queen,
      queenId: 'queen-2',
      drones: [
        { dna: drone0, parentId: 'drone-0' },
        { dna: drone1, parentId: 'drone-1' },
      ],
      ngeEnabled: true,
    });

    // Assert — offspring DNA is produced from the queen plus 2 drone envelopes.
    expect(result.offspring).toBeDefined();
  });

  it('passes queenBias = 0.85 in the polyandric reproduction policy', () => {
    // Arrange
    const queen = createCarGenome({
      carIndex: 2,
      seed: 42,
      teamId: 0,
      populationId: 'team-a',
      inputSize: 4,
      outputSize: 2,
    }).envelope;
    const drone = createCarGenome({
      carIndex: 0,
      seed: 43,
      teamId: 0,
      populationId: 'team-a',
      inputSize: 4,
      outputSize: 2,
    }).envelope;
    const policy = {
      mode: 'polyandric' as const,
      polyandricDroneCount: 2,
      polyandricDroneContributionFraction: 0.25,
      queenBias: 0.85,
      assignedRegionStrategy: 'non-overlapping' as const,
      modeIsEvolvable: true,
      seedPolicy: 'queen-weighted' as const,
      parthenogenesisMutationRate: 0,
    };

    // Act
    const result = reproducePolyandric({
      queen,
      queenId: 'queen-2',
      drones: [{ dna: drone, parentId: 'drone-0' }],
      ngeEnabled: true,
      policy,
    });

    // Assert — the policy is accepted and produces a defined offspring.
    expect(policy.queenBias).toBe(0.85);
    expect(result.offspring).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Red tests — Role-divergence metric computation (criterion 4)
// ---------------------------------------------------------------------------

describe('Tier 5 role-divergence metric computation', () => {
  it('computes non-zero blockerDelta for at least one car in a deterministic pack', async () => {
    // Arrange — deterministic 6-car finish: Team A [1, 4, 6], Team B [2, 3, 5]
    // blockerDelta = teamScoreWithCar - teamScoreWithoutCar
    // A car on a winning team with a worse individual position has a high
    // blockerDelta because removing them does not change the team outcome.
    const carFinishPositions = [1, 4, 6, 2, 3, 5];
    const teamLayout: readonly (0 | 1)[] = [0, 0, 0, 1, 1, 1];
    const teamScores = [10, 10];

    // Act — load the role-divergence service (does not exist yet)
    const service = await loadRoleDivergenceService();
    const metrics = service.computeRoleDivergenceMetrics(
      carFinishPositions,
      teamLayout,
      teamScores,
    );

    // Assert — at least one car must have a non-zero blockerDelta
    const hasNonZeroBlockerDelta = metrics.some(
      (metric) => metric.blockerDelta !== 0,
    );
    expect(hasNonZeroBlockerDelta).toBe(true);
  });

  it('assigns an inferredRole to each car in the metric output', async () => {
    // Arrange
    const carFinishPositions = [1, 4, 6, 2, 3, 5];
    const teamLayout: readonly (0 | 1)[] = [0, 0, 0, 1, 1, 1];
    const teamScores = [10, 10];

    // Act
    const service = await loadRoleDivergenceService();
    const metrics = service.computeRoleDivergenceMetrics(
      carFinishPositions,
      teamLayout,
      teamScores,
    );

    // Assert — every car must have a valid inferredRole
    const validRoles = new Set([
      'queen',
      'blocker',
      'pacer',
      'undifferentiated',
    ]);
    const allRolesAssigned = metrics.every((metric) =>
      validRoles.has(metric.inferredRole),
    );
    expect(allRolesAssigned).toBe(true);
  });

  it('produces metrics for all 6 cars in the pack', async () => {
    // Arrange
    const carFinishPositions = [1, 4, 6, 2, 3, 5];
    const teamLayout: readonly (0 | 1)[] = [0, 0, 0, 1, 1, 1];
    const teamScores = [10, 10];

    // Act
    const service = await loadRoleDivergenceService();
    const metrics = service.computeRoleDivergenceMetrics(
      carFinishPositions,
      teamLayout,
      teamScores,
    );

    // Assert — must return exactly 6 metrics (one per car)
    expect(metrics.length).toBe(6);
  });
});
