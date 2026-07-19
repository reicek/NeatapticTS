import type { EnvironmentState } from '../environment/environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';

type TireStateTuple = readonly [number, number, number, number];

type PitStrategyObservationState = EnvironmentState & {
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
  pitDistanceToEntrance01?: number;
  pitOccupancyStatus?: number;
  lapsSincePit?: number;
  teammatePitStatus?: number;
  tireDegradationRate?: number;
  estimatedLapsBeforeFailure?: number;
  reservedPitContext1?: number;
  reservedPitContext2?: number;
};

interface PitStrategyObservationAssemblerModule {
  assembleTier3Observation(
    envState: PitStrategyObservationState,
    trackSpec: TrackSpec,
  ): Float32Array | readonly number[];
  assembleTier4Observation(
    envState: PitStrategyObservationState,
    trackSpec: TrackSpec,
  ): Float32Array | readonly number[];
  assembleTier5Observation(
    envState: PitStrategyObservationState,
    trackSpec: TrackSpec,
  ): Float32Array | readonly number[];
}

const pitStrategyValues = [
  0.55, 0.66, 0.77, 0.88, 0.99, 0.11, 0.22, 0.33,
] as const;

describe('observation assembler pit/strategy tail', () => {
  describe('assembleTier4Observation', () => {
    it('returns a 103-channel vector', async () => {
      // Arrange
      const state = createPitStrategyObservationState();
      const trackSpec = createTrackSpec();

      // Act + Assert
      await expect(
        loadPitStrategyObservationAssemblerModule().then(
          ({ assembleTier4Observation }) =>
            Array.from(assembleTier4Observation(state, trackSpec)).length,
        ),
      ).resolves.toBe(103);
    });

    it('writes the eight pit/strategy values into channels 95 through 102', async () => {
      // Arrange
      const state = createPitStrategyObservationState();
      const trackSpec = createTrackSpec();

      // Act + Assert
      await expect(
        loadPitStrategyObservationAssemblerModule().then(
          ({ assembleTier4Observation }) =>
            Array.from(assembleTier4Observation(state, trackSpec)).slice(
              95,
              103,
            ),
        ),
      ).resolves.toEqual(Array.from(Float32Array.from(pitStrategyValues)));
    });

    it('keeps the pre-pit channel layout unchanged at offsets 0 through 94', async () => {
      // Arrange
      const state = createPitStrategyObservationState();
      const trackSpec = createTrackSpec();

      // Act + Assert
      await expect(
        loadPitStrategyObservationAssemblerModule().then(
          ({ assembleTier3Observation, assembleTier4Observation }) => {
            const tier3Vector = Array.from(
              assembleTier3Observation(state, trackSpec),
            );
            const expectedPrefix = [
              ...tier3Vector,
              ...Float32Array.from(state.tireState),
            ];
            const tier4Vector = Array.from(
              assembleTier4Observation(state, trackSpec),
            );
            return (
              tier4Vector.length === 103 &&
              expectedPrefix.every(
                (expectedValue, index) => tier4Vector[index] === expectedValue,
              )
            );
          },
        ),
      ).resolves.toBe(true);
    });
  });

  describe('assembleTier5Observation', () => {
    it('returns a 103-channel vector', async () => {
      // Arrange
      const state = createPitStrategyObservationState();
      const trackSpec = createTrackSpec();

      // Act + Assert
      await expect(
        loadPitStrategyObservationAssemblerModule().then(
          ({ assembleTier5Observation }) =>
            Array.from(assembleTier5Observation(state, trackSpec)).length,
        ),
      ).resolves.toBe(103);
    });

    it('writes the eight pit/strategy values into channels 95 through 102', async () => {
      // Arrange
      const state = createPitStrategyObservationState();
      const trackSpec = createTrackSpec();

      // Act + Assert
      await expect(
        loadPitStrategyObservationAssemblerModule().then(
          ({ assembleTier5Observation }) =>
            Array.from(assembleTier5Observation(state, trackSpec)).slice(
              95,
              103,
            ),
        ),
      ).resolves.toEqual(Array.from(Float32Array.from(pitStrategyValues)));
    });

    it('keeps the pre-pit channel layout unchanged at offsets 0 through 94', async () => {
      // Arrange
      const state = createPitStrategyObservationState();
      const trackSpec = createTrackSpec();

      // Act + Assert
      await expect(
        loadPitStrategyObservationAssemblerModule().then(
          ({ assembleTier3Observation, assembleTier5Observation }) => {
            const tier3Vector = Array.from(
              assembleTier3Observation(state, trackSpec),
            );
            const expectedPrefix = [
              ...tier3Vector,
              ...Float32Array.from(state.tireState),
            ];
            const tier5Vector = Array.from(
              assembleTier5Observation(state, trackSpec),
            );
            return (
              tier5Vector.length === 103 &&
              expectedPrefix.every(
                (expectedValue, index) => tier5Vector[index] === expectedValue,
              )
            );
          },
        ),
      ).resolves.toBe(true);
    });
  });
});

function createPitStrategyObservationState(
  overrides: Partial<PitStrategyObservationState> = {},
): PitStrategyObservationState {
  const defaultState: PitStrategyObservationState = {
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
      Float32Array.from([0.31, -0.28, 0.43, -0.37, 0.52, 0.27, -0.17]),
      Float32Array.from([0.41, -0.38, 0.33, -0.27, 0.42, 0.37, -0.07]),
    ],
    tireState: [0.95, 0.9, 0.85, 0.8],
    pitDistanceToEntrance01: pitStrategyValues[0],
    pitOccupancyStatus: pitStrategyValues[1],
    lapsSincePit: pitStrategyValues[2],
    teammatePitStatus: pitStrategyValues[3],
    tireDegradationRate: pitStrategyValues[4],
    estimatedLapsBeforeFailure: pitStrategyValues[5],
    reservedPitContext1: pitStrategyValues[6],
    reservedPitContext2: pitStrategyValues[7],
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

async function loadPitStrategyObservationAssemblerModule(): Promise<PitStrategyObservationAssemblerModule> {
  const modulePath = './observation.assembler';
  const module = (await import(
    modulePath
  )) as Partial<PitStrategyObservationAssemblerModule>;

  if (typeof module.assembleTier4Observation !== 'function') {
    throw new Error(
      'Missing pit/strategy observation export: assembleTier4Observation',
    );
  }

  if (typeof module.assembleTier5Observation !== 'function') {
    throw new Error(
      'Missing pit/strategy observation export: assembleTier5Observation',
    );
  }

  if (typeof module.assembleTier3Observation !== 'function') {
    throw new Error(
      'Missing pit/strategy observation export: assembleTier3Observation',
    );
  }

  return module as PitStrategyObservationAssemblerModule;
}
