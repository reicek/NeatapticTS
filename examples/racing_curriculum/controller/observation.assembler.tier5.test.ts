import type { EnvironmentState } from '../environment/environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';

type TireStateTuple = readonly [number, number, number, number];
type Tier5ObservationState = EnvironmentState & {
  forwardSpeedWorld?: number;
  lateralSpeedWorld?: number;
  speedWorld?: number;
  yawRateRadiansPerSecond?: number;
  slipAngleRadians?: number;
  progress01?: number;
  lapProgress01?: number;
  boundaryDistanceLeftWorld?: number;
  boundaryDistanceRightWorld?: number;
  hazardDistanceWorld?: number;
  waypointDistanceWorld?: number;
  optimalLineLateralOffsetWorld?: number;
  optimalLineHeadingErrorRadians?: number;
  targetSpeedWorld?: number;
  memoryTrace?: readonly number[];
  teammateRadioSlots?: readonly (Float32Array | readonly number[])[];
  tireState: TireStateTuple;
};

interface Tier5ObservationAssemblerModule {
  assembleTier5Observation(
    envState: Tier5ObservationState,
    trackSpec: TrackSpec,
  ): Float32Array | readonly number[];
}

interface Tier5ObservationOptionsModule {
  createTier5ObservationOptions(): {
    readonly tier: 5;
  };
}

describe('observation assembler Tier 5 six-car seam', () => {
  describe('ObservationTier type', () => {
    it('accepts 5 as a valid ObservationTier value', async () => {
      // Arrange
      const tier5ObservationOptionsModulePromise =
        loadTier5ObservationOptionsModule();

      // Act + Assert
      await expect(
        tier5ObservationOptionsModulePromise.then(
          ({ createTier5ObservationOptions }) =>
            acceptTier5ObservationTier(createTier5ObservationOptions().tier),
        ),
      ).resolves.toBe(5);
    });
  });

  describe('assembleTier5Observation', () => {
    it('exists for the Tier 5 3v3 observation seam', async () => {
      // Arrange
      const tier5ObservationAssemblerModulePromise =
        loadTier5ObservationAssemblerModule();

      // Act + Assert
      await expect(
        tier5ObservationAssemblerModulePromise.then(
          ({ assembleTier5Observation }) =>
            typeof assembleTier5Observation === 'function',
        ),
      ).resolves.toBe(true);
    });

    it('returns exactly 103 channels', async () => {
      // Arrange
      const tier5ObservationState = createTier5ObservationState();
      const trackSpec = createTrackSpec();
      const tier5ObservationAssemblerModulePromise =
        loadTier5ObservationAssemblerModule();

      // Act + Assert
      await expect(
        tier5ObservationAssemblerModulePromise.then(
          ({ assembleTier5Observation }) =>
            Array.from(
              assembleTier5Observation(tier5ObservationState, trackSpec),
            ).length,
        ),
      ).resolves.toBe(103);
    });

    it('writes three fully populated 7-float team-radio rows into channels 70 through 90', async () => {
      // Arrange
      const teamRadioRows = [
        Float32Array.from([0.11, -0.12, 0.13, -0.14, 0.15, -0.16, 0.17]),
        Float32Array.from([0.21, -0.22, 0.23, -0.24, 0.25, -0.26, 0.27]),
        Float32Array.from([0.31, -0.32, 0.33, -0.34, 0.35, -0.36, 0.37]),
      ] as const;
      const tier5ObservationState = createTier5ObservationState({
        teammateRadioSlots: teamRadioRows,
      });
      const trackSpec = createTrackSpec();
      const tier5ObservationAssemblerModulePromise =
        loadTier5ObservationAssemblerModule();

      // Act + Assert
      await expect(
        tier5ObservationAssemblerModulePromise.then(
          ({ assembleTier5Observation }) =>
            Array.from(
              assembleTier5Observation(tier5ObservationState, trackSpec),
            ).slice(70, 91),
        ),
      ).resolves.toEqual(
        teamRadioRows.flatMap((radioRow) => Array.from(radioRow)),
      );
    });

    it('writes own-car tire values into channels 91 through 94', async () => {
      // Arrange
      const tireState: TireStateTuple = [0.91, 0.82, 0.73, 0.64];
      const tier5ObservationState = createTier5ObservationState({ tireState });
      const trackSpec = createTrackSpec();
      const tier5ObservationAssemblerModulePromise =
        loadTier5ObservationAssemblerModule();

      // Act + Assert
      await expect(
        tier5ObservationAssemblerModulePromise.then(
          ({ assembleTier5Observation }) =>
            Array.from(
              assembleTier5Observation(tier5ObservationState, trackSpec),
            ).slice(91, 95),
        ),
      ).resolves.toEqual(Array.from(Float32Array.from(tireState)));
    });
  });

  describe('createTier5ObservationOptions', () => {
    it('creates a reusable { tier: 5 } options object', async () => {
      // Arrange
      const tier5ObservationOptionsModulePromise =
        loadTier5ObservationOptionsModule();

      // Act + Assert
      await expect(
        tier5ObservationOptionsModulePromise.then(
          ({ createTier5ObservationOptions }) =>
            createTier5ObservationOptions(),
        ),
      ).resolves.toEqual({ tier: 5 });
    });
  });
});

function createTier5ObservationState(
  overrides: Partial<Tier5ObservationState> = {},
): Tier5ObservationState {
  const defaultState: Tier5ObservationState = {
    tick: 34,
    carX: 18,
    carY: -24,
    carHeading: Math.PI / 4,
    forwardSpeedWorld: 18,
    lateralSpeedWorld: -4,
    speedWorld: 18.4390889146,
    yawRateRadiansPerSecond: -0.25,
    slipAngleRadians: 0.12,
    progress01: 0.35,
    lapProgress01: 0.6,
    boundaryDistanceLeftWorld: 14,
    boundaryDistanceRightWorld: 9,
    hazardDistanceWorld: 32,
    waypointDistanceWorld: 21,
    optimalLineLateralOffsetWorld: -3.5,
    optimalLineHeadingErrorRadians: 0.2,
    targetSpeedWorld: 24,
    memoryTrace: [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5],
    teammateRadioSlots: [
      Float32Array.from([0.11, -0.12, 0.13, -0.14, 0.15, -0.16, 0.17]),
      Float32Array.from([0.21, -0.22, 0.23, -0.24, 0.25, -0.26, 0.27]),
      Float32Array.from([0.31, -0.32, 0.33, -0.34, 0.35, -0.36, 0.37]),
    ],
    tireState: [0.95, 0.9, 0.85, 0.8],
  };

  return {
    ...defaultState,
    ...overrides,
    teammateRadioSlots:
      overrides.teammateRadioSlots ?? defaultState.teammateRadioSlots,
    tireState: overrides.tireState ?? defaultState.tireState,
  };
}

function createTrackSpec(): TrackSpec {
  return generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'medium' });
}

function acceptTier5ObservationTier(
  tier: 1 | 2 | 3 | 4 | 5,
): 1 | 2 | 3 | 4 | 5 {
  return tier;
}

async function loadTier5ObservationAssemblerModule(): Promise<Tier5ObservationAssemblerModule> {
  const modulePath = './observation.assembler';
  const module = (await import(
    modulePath
  )) as Partial<Tier5ObservationAssemblerModule>;

  if (typeof module.assembleTier5Observation !== 'function') {
    throw new Error(
      'Missing Tier 5 observation export: assembleTier5Observation',
    );
  }

  return module as Tier5ObservationAssemblerModule;
}

async function loadTier5ObservationOptionsModule(): Promise<Tier5ObservationOptionsModule> {
  const modulePath = './observation.assembler';
  const module = (await import(
    modulePath
  )) as Partial<Tier5ObservationOptionsModule>;

  if (typeof module.createTier5ObservationOptions !== 'function') {
    throw new Error(
      'Missing Tier 5 observation export: createTier5ObservationOptions',
    );
  }

  return module as Tier5ObservationOptionsModule;
}
