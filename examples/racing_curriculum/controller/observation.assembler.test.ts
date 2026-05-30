import type { EnvironmentState } from '../environment/environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';
import {
  BOUNDARY_DISTANCE_WORLD_SCALE,
  DISTANCE_WORLD_SCALE,
  createCurvedTrackSpec,
  resolveOffsetObservationProbe,
  selectObservationFocalSample,
  wrapAngleToMinusPiPi,
} from './pathtracking.test.fixtures';

/**
 * Red-phase contracts for the owner-local Tier 1–2 observation-vector seam.
 *
 * The assembler implementation does not exist yet; runtime string imports keep
 * the suite narrowly red until Step 04 introduces the owner-local module.
 */
describe('observation.assembler', () => {
  describe('assembleNormalizedObservationVector', () => {
    it('returns a finite Tier 1 vector with 70 normalized channels', async () => {
      const trackSpec = createTrackSpec();
      const envState = createExpandedEnvironmentState();

      await expect(
        loadObservationAssemblerModule().then(
          ({ assembleNormalizedObservationVector }) => {
            const observationVector = Array.from(
              assembleNormalizedObservationVector(envState, trackSpec, {
                tier: 1,
              }),
            );

            return {
              length: observationVector.length,
              allFinite: observationVector.every((value) =>
                Number.isFinite(value),
              ),
              allNormalized: observationVector.every(
                (value) => value >= -1 && value <= 1,
              ),
            };
          },
        ),
      ).resolves.toEqual({
        length: 70,
        allFinite: true,
        allNormalized: true,
      });
    });

    it('appends the seven self-radio channels at the tail of the Tier 2 vector without renormalizing them', async () => {
      const radioField = new Float32Array([
        0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77,
      ]);
      const trackSpec = createTrackSpec();
      const envState = createExpandedEnvironmentState({ radioField });

      await expect(
        loadObservationAssemblerModule().then(
          ({ assembleNormalizedObservationVector }) => {
            const observationVector = Array.from(
              assembleNormalizedObservationVector(envState, trackSpec, {
                tier: 2,
              }),
            );
            return {
              length: observationVector.length,
              radioTail: observationVector.slice(-radioField.length),
            };
          },
        ),
      ).resolves.toEqual({
        length: 77,
        radioTail: Array.from(radioField),
      });
    });

    describe('rounded spline seam contracts', () => {
      it('aligns the first look-ahead slice to the rounded spline sample rather than the raw chord midpoint', async () => {
        const trackSpec = createCurvedTrackSpec();
        const { splineSamples, focalSample, focalFrame } =
          selectObservationFocalSample(trackSpec);
        const observationProbe = resolveOffsetObservationProbe(
          splineSamples,
          focalSample.globalIndex,
        );
        const envState = createExpandedEnvironmentState({
          carX: observationProbe.carX,
          carY: observationProbe.carY,
          carHeading: focalFrame.tangentHeadingRadians,
          boundaryDistanceLeftWorld: undefined,
          boundaryDistanceRightWorld: undefined,
        });

        await expect(
          loadObservationAssemblerModule().then(
            ({ assembleNormalizedObservationVector }) => {
              const observationVector = Array.from(
                assembleNormalizedObservationVector(envState, trackSpec, {
                  tier: 1,
                }),
              );
              const lookAheadHeadingRadians = Math.atan2(
                observationVector[24] ?? 0,
                observationVector[25] ?? 1,
              );
              const lookAheadDistanceWorld =
                (observationVector[27] ?? 0) * DISTANCE_WORLD_SCALE;

              return {
                usesSplineHeading:
                  Math.abs(
                    wrapAngleToMinusPiPi(
                      lookAheadHeadingRadians -
                        focalFrame.tangentHeadingRadians,
                    ),
                  ) < 0.05,
                usesSplineDistance:
                  Math.abs(
                    lookAheadDistanceWorld - observationProbe.offsetWorld,
                  ) < 0.5,
              };
            },
          ),
        ).resolves.toEqual({
          usesSplineHeading: true,
          usesSplineDistance: true,
        });
      });

      it("derives left and right boundary distances from the car's lateral offset inside the rounded lane", async () => {
        const trackSpec = createCurvedTrackSpec();
        const { splineSamples, focalSample, focalFrame } =
          selectObservationFocalSample(trackSpec);
        const observationProbe = resolveOffsetObservationProbe(
          splineSamples,
          focalSample.globalIndex,
        );
        const envState = createExpandedEnvironmentState({
          carX: observationProbe.carX,
          carY: observationProbe.carY,
          carHeading: focalFrame.tangentHeadingRadians,
          boundaryDistanceLeftWorld: undefined,
          boundaryDistanceRightWorld: undefined,
        });
        const expectedLeftBoundaryDistanceWorld =
          focalSample.width / 2 - observationProbe.offsetWorld;
        const expectedRightBoundaryDistanceWorld =
          focalSample.width / 2 + observationProbe.offsetWorld;
        const expectedBoundaryBalance =
          (expectedRightBoundaryDistanceWorld -
            expectedLeftBoundaryDistanceWorld) /
          (expectedRightBoundaryDistanceWorld +
            expectedLeftBoundaryDistanceWorld);

        await expect(
          loadObservationAssemblerModule().then(
            ({ assembleNormalizedObservationVector }) => {
              const observationVector = Array.from(
                assembleNormalizedObservationVector(envState, trackSpec, {
                  tier: 1,
                }),
              );
              const boundaryDistanceLeftWorld =
                (observationVector[11] ?? 0) * BOUNDARY_DISTANCE_WORLD_SCALE;
              const boundaryDistanceRightWorld =
                (observationVector[12] ?? 0) * BOUNDARY_DISTANCE_WORLD_SCALE;
              const boundaryBalance = observationVector[13] ?? 0;

              return {
                variesWithLateralOffset:
                  Math.abs(
                    boundaryDistanceRightWorld - boundaryDistanceLeftWorld,
                  ) > 1,
                matchesExpectedLeftBoundary:
                  Math.abs(
                    boundaryDistanceLeftWorld -
                      expectedLeftBoundaryDistanceWorld,
                  ) < 0.75,
                matchesExpectedRightBoundary:
                  Math.abs(
                    boundaryDistanceRightWorld -
                      expectedRightBoundaryDistanceWorld,
                  ) < 0.75,
                matchesExpectedBoundaryBalance:
                  Math.abs(boundaryBalance - expectedBoundaryBalance) < 0.05,
              };
            },
          ),
        ).resolves.toEqual({
          variesWithLateralOffset: true,
          matchesExpectedLeftBoundary: true,
          matchesExpectedRightBoundary: true,
          matchesExpectedBoundaryBalance: true,
        });
      });
    });
  });
});

type ExpandedEnvironmentState = EnvironmentState & {
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
  radioField?: Float32Array;
};

interface ObservationAssemblerModule {
  assembleNormalizedObservationVector(
    envState: ExpandedEnvironmentState,
    trackSpec: TrackSpec,
    options: { tier: 1 | 2 },
  ): Float32Array | readonly number[];
}

function createExpandedEnvironmentState(
  overrides: Partial<ExpandedEnvironmentState> = {},
): ExpandedEnvironmentState {
  const defaultState: ExpandedEnvironmentState = {
    tick: 12,
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
    radioField: new Float32Array(0),
  };

  return {
    ...defaultState,
    ...overrides,
    radioField: overrides.radioField ?? defaultState.radioField,
  };
}

function createTrackSpec(): TrackSpec {
  return generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'medium' });
}

async function loadObservationAssemblerModule(): Promise<ObservationAssemblerModule> {
  const modulePath = './observation.assembler';
  return (await import(modulePath)) as ObservationAssemblerModule;
}
