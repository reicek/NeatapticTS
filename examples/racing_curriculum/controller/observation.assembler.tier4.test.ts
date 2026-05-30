import type { EnvironmentState } from '../environment/environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';

type TireStateTuple = readonly [number, number, number, number];
type Tier4ObservationState = EnvironmentState & {
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

interface Tier4ObservationAssemblerModule {
  assembleTier3Observation(
    envState: Tier4ObservationState,
    trackSpec: TrackSpec,
  ): Float32Array | readonly number[];
  assembleTier4Observation(
    envState: Tier4ObservationState,
    trackSpec: TrackSpec,
  ): Float32Array | readonly number[];
  createTier4ObservationOptions(): {
    readonly tier: 4;
  };
}

describe('observation assembler Tier 4 tire tail', () => {
  describe('assembleTier4Observation', () => {
    it('returns exactly 95 channels', async () => {
      // Arrange
      const tier4ObservationState = createTier4ObservationState();
      const trackSpec = createTrackSpec();

      // Act + Assert
      await expect(
        loadTier4ObservationAssemblerModule().then(
          ({ assembleTier4Observation }) =>
            Array.from(
              assembleTier4Observation(tier4ObservationState, trackSpec),
            ).length,
        ),
      ).resolves.toBe(95);
    });

    it('preserves channels 0 through 90 exactly from Tier 3', async () => {
      // Arrange
      const tier4ObservationState = createTier4ObservationState();
      const trackSpec = createTrackSpec();

      // Act + Assert
      await expect(
        loadTier4ObservationAssemblerModule().then(
          ({ assembleTier3Observation, assembleTier4Observation }) => {
            const tier3Vector = Array.from(
              assembleTier3Observation(tier4ObservationState, trackSpec),
            );
            const tier4Vector = Array.from(
              assembleTier4Observation(tier4ObservationState, trackSpec),
            );
            return (
              tier3Vector.length === 91 &&
              tier4Vector
                .slice(0, 91)
                .every(
                  (tier4Value, channelIndex) =>
                    tier4Value === tier3Vector[channelIndex],
                )
            );
          },
        ),
      ).resolves.toBe(true);
    });

    it('writes own-car tire state into channels 91 through 94', async () => {
      // Arrange
      const tireState: TireStateTuple = [0.91, 0.82, 0.73, 0.64];
      const tier4ObservationState = createTier4ObservationState({ tireState });
      const trackSpec = createTrackSpec();

      // Act + Assert
      await expect(
        loadTier4ObservationAssemblerModule().then(
          ({ assembleTier4Observation }) =>
            Array.from(
              assembleTier4Observation(tier4ObservationState, trackSpec),
            ).slice(91, 95),
        ),
      ).resolves.toEqual(Array.from(Float32Array.from(tireState)));
    });
  });

  describe('Tier 4 observation options', () => {
    it('creates a reusable { tier: 4 } options object', async () => {
      // Arrange + Act + Assert
      await expect(
        loadTier4ObservationAssemblerModule().then(
          ({ createTier4ObservationOptions }) =>
            createTier4ObservationOptions(),
        ),
      ).resolves.toEqual({ tier: 4 });
    });

    it('accepts 4 as a valid ObservationTier value', async () => {
      // Arrange + Act + Assert
      await expect(
        loadTier4ObservationAssemblerModule().then(
          ({ createTier4ObservationOptions }) =>
            acceptObservationTier(createTier4ObservationOptions().tier),
        ),
      ).resolves.toBe(4);
    });
  });
});

function createTier4ObservationState(
  overrides: Partial<Tier4ObservationState> = {},
): Tier4ObservationState {
  const defaultState: Tier4ObservationState = {
    tick: 21,
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
      Float32Array.from([0.21, -0.18, 0.44, -0.39, 0.62, 0.17, -0.27]),
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

function acceptObservationTier(tier: 1 | 2 | 3 | 4): 1 | 2 | 3 | 4 {
  return tier;
}

async function loadTier4ObservationAssemblerModule(): Promise<Tier4ObservationAssemblerModule> {
  const modulePath = './observation.assembler';
  const module = (await import(
    modulePath
  )) as Partial<Tier4ObservationAssemblerModule>;

  if (typeof module.assembleTier4Observation !== 'function') {
    throw new Error(
      'Missing Tier 4 observation export: assembleTier4Observation',
    );
  }

  if (typeof module.createTier4ObservationOptions !== 'function') {
    throw new Error(
      'Missing Tier 4 observation export: createTier4ObservationOptions',
    );
  }

  return module as Tier4ObservationAssemblerModule;
}
