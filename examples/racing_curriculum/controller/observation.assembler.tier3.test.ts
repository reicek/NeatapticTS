import type { EnvironmentState } from '../environment/environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';

type Tier3ObservationState = EnvironmentState & {
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
};

interface Tier3ObservationAssemblerModule {
  assembleTier3Observation(
    envState: Tier3ObservationState,
    trackSpec: TrackSpec,
  ): Float32Array | readonly number[];
  createTier3ObservationOptions(): {
    readonly tier: 3;
  };
}

/**
 * Red-phase contracts for the Tier 3 21-channel teammate-radio observation seam.
 *
 * The current assembler is still Tier 1–2 only, so these tests must stay red
 * until the 91-channel Tier 3 extension lands.
 */
describe('Tier 3 observation assembler (21-channel radio)', () => {
  describe('assembleTier3Observation', () => {
    it('returns a 91-channel observation vector', async () => {
      await expect(
        loadTier3ObservationAssemblerModule().then(({ assembleTier3Observation }) =>
          Array.from(
            assembleTier3Observation(
              createTier3ObservationState(),
              createTrackSpec(),
            ),
          ).length,
        ),
      ).resolves.toBe(91);
    });

    it('populates teammate slot 0 with live teammate radio data', async () => {
      const liveTeammateRadio = Float32Array.from([
        0.21, -0.18, 0.44, -0.39, 0.62, 0.17, -0.27,
      ]);

      await expect(
        loadTier3ObservationAssemblerModule().then(({ assembleTier3Observation }) => {
          const observationVector = Array.from(
            assembleTier3Observation(
              createTier3ObservationState({
                teammateRadioSlots: [liveTeammateRadio],
              }),
              createTrackSpec(),
            ),
          );
          const teammateSlotZero = observationVector.slice(70, 77);
          return teammateSlotZero.every((value) => value !== 0);
        }),
      ).resolves.toBe(true);
    });

    it('zero-pads teammate slots 1 and 2 in 2v2 layout', async () => {
      await expect(
        loadTier3ObservationAssemblerModule().then(({ assembleTier3Observation }) => {
          const observationVector = Array.from(
            assembleTier3Observation(
              createTier3ObservationState({
                teammateRadioSlots: [
                  Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                ],
              }),
              createTrackSpec(),
            ),
          );
          return observationVector.slice(77, 91);
        }),
      ).resolves.toEqual(new Array<number>(14).fill(0));
    });

    it('allocates exactly 7 channels per teammate slot', async () => {
      await expect(
        loadTier3ObservationAssemblerModule().then(({ assembleTier3Observation }) => {
          const observationVector = Array.from(
            assembleTier3Observation(
              createTier3ObservationState({
                teammateRadioSlots: [
                  Float32Array.from([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4]),
                ],
              }),
              createTrackSpec(),
            ),
          );
          return [
            observationVector.slice(70, 77).length,
            observationVector.slice(77, 84).length,
            observationVector.slice(84, 91).length,
          ];
        }),
      ).resolves.toEqual([7, 7, 7]);
    });
  });

  describe('ObservationTier type', () => {
    it('accepts tier 3 as a valid ObservationTier value', async () => {
      await expect(
        loadTier3ObservationAssemblerModule().then(
          ({ createTier3ObservationOptions }) =>
            createTier3ObservationOptions().tier,
        ),
      ).resolves.toBe(3);
    });
  });
});

function createTier3ObservationState(
  overrides: Partial<Tier3ObservationState> = {},
): Tier3ObservationState {
  const defaultState: Tier3ObservationState = {
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
    teammateRadioSlots: [],
  };

  return {
    ...defaultState,
    ...overrides,
    teammateRadioSlots:
      overrides.teammateRadioSlots ?? defaultState.teammateRadioSlots,
  };
}

function createTrackSpec(): TrackSpec {
  return generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'medium' });
}

async function loadTier3ObservationAssemblerModule(): Promise<Tier3ObservationAssemblerModule> {
  const modulePath = './observation.assembler';
  return (await import(modulePath)) as Tier3ObservationAssemblerModule;
}
