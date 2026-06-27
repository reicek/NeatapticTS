/**
 * Red-phase contracts for the Tier 5 three-car team runtime (3v3 full).
 *
 * Contracts verified here:
 * - All 3 teammate-radio slots are populated for 6-car teams (including self)
 * - Polyandric reproduction wiring (BLOCKED by P1/P2 — written as skip contracts)
 * - Role-divergence metric computation returns non-zero for a deterministic pack
 *
 * All non-skipped tests stay red until Step 04 implements the service boundaries.
 * Single-expect rule is enforced throughout.
 *
 * TODO: NGE_TODO — Polyandric reproduction tests are BLOCKED by:
 *   P1: NGE_DNA adoption gap (racing uses Network, not NgeDnaCanonicalEnvelope)
 *   P2: NgePolyandricInput/NgePolyandricDroneInput not exported from reproduction.ts
 *   These tests are written as skip contracts documenting the expected behavior.
 */
import { derivePerCarObservationState } from '../../controller/observation.assembler';
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
// Red tests — Polyandric reproduction wiring (criterion 3 — BLOCKED by P1/P2)
// ---------------------------------------------------------------------------

describe('Tier 5 polyandric reproduction wiring (BLOCKED — P1/P2)', () => {
  // BLOCKER P1: NGE_DNA adoption gap — racing uses Network, not NgeDnaCanonicalEnvelope.
  //   The racing coevolution container creates genomes backed by Network instances,
  //   but reproducePolyandric requires NgeDnaCanonicalEnvelope inputs. Until the
  //   nge-core-algorithm bridge is built (P1), these tests cannot compile against
  //   the real API.
  //
  // BLOCKER P2: NgePolyandricInput / NgePolyandricDroneInput types are not exported
  //   from src/neat/nge-evolution/neat.nge-evolution.reproduction.ts. Without the
  //   exported types, test code cannot construct valid input objects.
  //
  // These tests are written as skip contracts that document the expected behavior
  // once P1 and P2 are resolved. Step 04 implementation must:
  // 1. Build the NGE_DNA bridge (Network -> NgeDnaCanonicalEnvelope)
  // 2. Export the required input types from reproduction.ts
  // 3. Wire queen selection (best-finishing car) -> reproducePolyandric call
  // 4. Pass drones = other 2 team cars with queenBias = 0.85

  it.skip('selects the best-finishing car as queen for polyandric reproduction', () => {
    // Arrange — Team A finishes: car 0 = position 2, car 1 = position 5,
    // car 2 = position 1. Best-finishing = car 2 (position 1) -> queen.
    // Drones = car 0 and car 1.
    // Once P1/P2 resolved: construct NgePolyandricInput with queen = car 2
    // envelope and assert queen selection logic identifies car 2.
    expect(true).toBe(true); // Placeholder — would assert queen selection
  });

  it.skip('calls reproducePolyandric with queen envelope and 2 drone envelopes', () => {
    // Arrange — queen = best-finishing car, drones = other 2 team cars.
    // Once P1/P2 resolved: verify reproducePolyandric is called with:
    //   queen: NgeDnaCanonicalEnvelope (best-finishing car)
    //   drones: [NgeDnaCanonicalEnvelope, NgeDnaCanonicalEnvelope] (other 2)
    //   polyandricDroneCount: 2
    expect(true).toBe(true); // Placeholder — would assert polyandric call
  });

  it.skip('passes queenBias = 0.85 in the polyandric reproduction policy', () => {
    // Arrange — polyandric policy: mode=polyandric, polyandricDroneCount=2,
    //   polyandricDroneContributionFraction=0.25, queenBias=0.85,
    //   assignedRegionStrategy=non-overlapping, modeIsEvolvable=true,
    //   seedPolicy=queen-weighted
    // Once P1/P2 resolved: verify the policy object passed to reproducePolyandric
    // has queenBias = 0.85.
    expect(true).toBe(true); // Placeholder — would assert queenBias
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
