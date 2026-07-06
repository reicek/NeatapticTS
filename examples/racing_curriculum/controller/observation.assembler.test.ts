import type {
  CarState,
  EnvironmentState,
} from '../environment/environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';
import {
  buildTrackSplineSamples,
  resolveSplineSampleFrame,
} from '../track/track.spline.utils';
import {
  BOUNDARY_DISTANCE_WORLD_SCALE,
  DISTANCE_WORLD_SCALE,
  createCurvedTrackSpec,
  resolveOffsetObservationProbe,
  selectObservationFocalSample,
  wrapAngleToMinusPiPi,
} from './pathtracking.test.fixtures';
import { createNgeController } from './nge.controller';

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

    it('feeds a 77-dimensional observation vector to the Tier 2 controller network', async () => {
      const trackSpec = createTrackSpec();
      const envState = createExpandedEnvironmentState();
      const activationInputs: number[][] = [];
      const network = {
        activate(inputVector: readonly number[] | Float32Array) {
          activationInputs.push(Array.from(inputVector));
          return [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8, -0.9];
        },
      };

      createNgeController(network, { tier: 2 }).computeControl(
        envState,
        trackSpec,
      );

      expect(activationInputs[0]?.length).toBe(77);
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

  describe('derivePerCarObservationState', () => {
    it('exports derivePerCarObservationState from the observation assembler module', async () => {
      const mod = await loadObservationAssemblerModule();
      expect(typeof mod.derivePerCarObservationState).toBe('function');
    });

    it('reads the requested car pose from envState.cars[carIndex]', async () => {
      const trackSpec = createStraightTrackSpec();
      const envState = createTwoCarEnvironmentState(trackSpec);
      const mod = await loadObservationAssemblerModule();
      const derived = mod.derivePerCarObservationState(envState, 1);

      expect(derived.carX).toBe(envState.cars![1].carX);
    });

    it('derives teamIndex from the selected car instead of envState.teamIndex', async () => {
      const trackSpec = createStraightTrackSpec();
      const envState = createTwoCarEnvironmentState(trackSpec);
      envState.teamIndex = 0;

      const mod = await loadObservationAssemblerModule();
      const derived = mod.derivePerCarObservationState(envState, 1);

      expect(derived.teamIndex).toBe(1);
    });

    it('produces different Tier 1 observation vectors for blue and red cars on the same tick', async () => {
      const trackSpec = createStraightTrackSpec();
      const envState = createTwoCarEnvironmentState(trackSpec);
      const mod = await loadObservationAssemblerModule();
      const blueState = mod.derivePerCarObservationState(envState, 0);
      const redState = mod.derivePerCarObservationState(envState, 1);
      const blueVector = Array.from(
        mod.assembleNormalizedObservationVector(blueState, trackSpec, {
          tier: 1,
        }),
      );
      const redVector = Array.from(
        mod.assembleNormalizedObservationVector(redState, trackSpec, {
          tier: 1,
        }),
      );

      expect(redVector).not.toEqual(blueVector);
    });
  });

  describe('Tier 3 teammate-state observation channels', () => {
    it('populates teammateRadioSlots[0] with non-zero values when a same-team teammate exists', async () => {
      const trackSpec = createStraightTrackSpec();
      const envState = createFourCarEnvironmentState(trackSpec);
      const mod = await loadObservationAssemblerModule();
      const derived = mod.derivePerCarObservationState(envState, 0);

      // Car 0's teammate is car 1 (both Team A, layout [0, 0, 1, 1])
      const slot0 = derived.teammateRadioSlots?.[0];
      const hasNonZero =
        slot0 !== undefined && Array.from(slot0).some((v) => v !== 0);

      expect(hasNonZero).toBe(true);
    });

    it('leaves teammateRadioSlots[1] and [2] zero-padded for a 2v2 layout', async () => {
      const trackSpec = createStraightTrackSpec();
      const envState = createFourCarEnvironmentState(trackSpec);
      const mod = await loadObservationAssemblerModule();
      const derived = mod.derivePerCarObservationState(envState, 0);

      // 2v2: slot 0 has teammate, slots 1 and 2 are zero-padded
      const slot1 = derived.teammateRadioSlots?.[1];
      const slot2 = derived.teammateRadioSlots?.[2];
      const slotsAreZeroPadded =
        (slot1 === undefined || Array.from(slot1).every((v) => v === 0)) &&
        (slot2 === undefined || Array.from(slot2).every((v) => v === 0));

      expect(slotsAreZeroPadded).toBe(true);
    });

    it('populates teammateRadioSlots with the teammate car position and heading for car 2 teammate car 3', async () => {
      const trackSpec = createStraightTrackSpec();
      const envState = createFourCarEnvironmentState(trackSpec);
      const mod = await loadObservationAssemblerModule();
      const derived = mod.derivePerCarObservationState(envState, 2);

      // Car 2's teammate is car 3 (both Team B, layout [0, 0, 1, 1])
      const slot0 = derived.teammateRadioSlots?.[0];
      const hasNonZero =
        slot0 !== undefined && Array.from(slot0).some((v) => v !== 0);

      expect(hasNonZero).toBe(true);
    });
  });
});

describe('assembleNormalizedObservationVector inner-lane optimal line', () => {
  it('reports near-zero optimal-line offset for a car on the inner-lane centerline', async () => {
    const trackSpec = createStraightTrackSpec();
    const focalSampleIndex = 2;
    const focalSample = trackSpec.splineSamples[focalSampleIndex]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      focalSample.globalIndex,
    );
    const laneCount = 2;
    const laneWidthWorld = focalSample.width / laneCount;
    const innerOffsetWorld = focalSample.width / 2 - laneWidthWorld / 2;
    const envState = createExpandedEnvironmentState({
      carX: focalSample.x + frame.normalX * innerOffsetWorld,
      carY: focalSample.y + frame.normalY * innerOffsetWorld,
      carHeading: frame.tangentHeadingRadians,
      boundaryDistanceLeftWorld: undefined,
      boundaryDistanceRightWorld: undefined,
      optimalLineLateralOffsetWorld: undefined,
      optimalLineHeadingErrorRadians: undefined,
    });

    await expect(
      loadObservationAssemblerModule().then(
        ({ assembleNormalizedObservationVector }) => {
          const observationVector = Array.from(
            assembleNormalizedObservationVector(envState, trackSpec, {
              tier: 1,
            }),
          );

          return {
            channel16: observationVector[16] ?? 0,
            channel17: observationVector[17] ?? 0,
          };
        },
      ),
    ).resolves.toEqual({
      channel16: expect.closeTo(0, 2),
      channel17: expect.closeTo(0, 2),
    });
  });
});

describe('assembleNormalizedObservationVector team-aware optimal line', () => {
  it('reports near-zero optimal-line offset for team 0 on the inner-lane centerline', async () => {
    const trackSpec = createStraightTrackSpec();
    const focalSampleIndex = 2;
    const focalSample = trackSpec.splineSamples[focalSampleIndex]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      focalSample.globalIndex,
    );
    const laneCount = 2;
    const laneWidthWorld = focalSample.width / laneCount;
    const innerOffsetWorld = focalSample.width / 2 - laneWidthWorld / 2;
    const envState = createExpandedEnvironmentState({
      teamIndex: 0,
      carX: focalSample.x + frame.normalX * innerOffsetWorld,
      carY: focalSample.y + frame.normalY * innerOffsetWorld,
      carHeading: frame.tangentHeadingRadians,
      boundaryDistanceLeftWorld: undefined,
      boundaryDistanceRightWorld: undefined,
      optimalLineLateralOffsetWorld: undefined,
      optimalLineHeadingErrorRadians: undefined,
    });

    await expect(
      loadObservationAssemblerModule().then(
        ({ assembleNormalizedObservationVector }) => {
          const observationVector = Array.from(
            assembleNormalizedObservationVector(envState, trackSpec, {
              tier: 1,
            }),
          );

          return {
            channel16: observationVector[16] ?? 0,
          };
        },
      ),
    ).resolves.toEqual({
      channel16: expect.closeTo(0, 2),
    });
  });

  it('reports near-zero optimal-line offset for team 1 on the outer-lane centerline', async () => {
    const trackSpec = createStraightTrackSpec();
    const focalSampleIndex = 2;
    const focalSample = trackSpec.splineSamples[focalSampleIndex]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      focalSample.globalIndex,
    );
    const laneCount = 2;
    const laneWidthWorld = focalSample.width / laneCount;
    const innerOffsetWorld = focalSample.width / 2 - laneWidthWorld / 2;
    const envState = createExpandedEnvironmentState({
      teamIndex: 1,
      carX: focalSample.x - frame.normalX * innerOffsetWorld,
      carY: focalSample.y - frame.normalY * innerOffsetWorld,
      carHeading: frame.tangentHeadingRadians,
      boundaryDistanceLeftWorld: undefined,
      boundaryDistanceRightWorld: undefined,
      optimalLineLateralOffsetWorld: undefined,
      optimalLineHeadingErrorRadians: undefined,
    });

    await expect(
      loadObservationAssemblerModule().then(
        ({ assembleNormalizedObservationVector }) => {
          const observationVector = Array.from(
            assembleNormalizedObservationVector(envState, trackSpec, {
              tier: 1,
            }),
          );

          return {
            channel16: observationVector[16] ?? 0,
          };
        },
      ),
    ).resolves.toEqual({
      channel16: expect.closeTo(0, 2),
    });
  });
});

function createStraightTrackSpec(): TrackSpec {
  const segments = [
    { startX: 0, startY: 0, endX: 100, endY: 0, width: 24 },
    { startX: 100, startY: 0, endX: 100, endY: 100, width: 24 },
    { startX: 100, startY: 100, endX: 0, endY: 100, width: 24 },
    { startX: 0, startY: 100, endX: 0, endY: 0, width: 24 },
  ];

  return {
    seed: 1,
    layoutVersion: 1,
    sizeBucket: 'straight-red',
    segments,
    splineSamples: [...buildTrackSplineSamples(segments)],
  };
}

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
  teammateRadioSlots?: readonly (Float32Array | readonly number[])[];
};

interface ObservationAssemblerModule {
  assembleNormalizedObservationVector(
    envState: ExpandedEnvironmentState,
    trackSpec: TrackSpec,
    options: { tier: 1 | 2 },
  ): Float32Array | readonly number[];
  derivePerCarObservationState(
    envState: ExpandedEnvironmentState,
    carIndex: number,
  ): ExpandedEnvironmentState;
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

function createTwoCarEnvironmentState(
  trackSpec: TrackSpec,
): ExpandedEnvironmentState {
  const focalSampleIndex = 2;
  const focalSample = trackSpec.splineSamples[focalSampleIndex]!;
  const frame = resolveSplineSampleFrame(
    trackSpec.splineSamples,
    focalSample.globalIndex,
  );
  const innerOffsetWorld = focalSample.width / 4;
  const team0Car: CarState = {
    carX: focalSample.x + frame.normalX * innerOffsetWorld,
    carY: focalSample.y + frame.normalY * innerOffsetWorld,
    carHeading: frame.tangentHeadingRadians,
    teamIndex: 0,
    tireState: [1, 1, 1, 1],
  };
  const team1Car: CarState = {
    carX: focalSample.x - frame.normalX * innerOffsetWorld,
    carY: focalSample.y - frame.normalY * innerOffsetWorld,
    carHeading: frame.tangentHeadingRadians,
    teamIndex: 1,
    tireState: [1, 1, 1, 1],
  };

  return createExpandedEnvironmentState({
    carX: team0Car.carX,
    carY: team0Car.carY,
    carHeading: team0Car.carHeading,
    cars: [team0Car, team1Car] as const,
  });
}

function createFourCarEnvironmentState(
  trackSpec: TrackSpec,
): ExpandedEnvironmentState {
  const focalSampleIndex = 2;
  const focalSample = trackSpec.splineSamples[focalSampleIndex]!;
  const frame = resolveSplineSampleFrame(
    trackSpec.splineSamples,
    focalSample.globalIndex,
  );
  const innerOffsetWorld = focalSample.width / 4;
  const tangentUnitX = Math.cos(frame.tangentHeadingRadians);
  const tangentUnitY = Math.sin(frame.tangentHeadingRadians);
  const carSpacing = 5;

  const car0: CarState = {
    carX: focalSample.x + frame.normalX * innerOffsetWorld,
    carY: focalSample.y + frame.normalY * innerOffsetWorld,
    carHeading: frame.tangentHeadingRadians,
    teamIndex: 0,
    tireState: [1, 1, 1, 1],
  };
  const car1: CarState = {
    carX:
      focalSample.x +
      frame.normalX * innerOffsetWorld +
      tangentUnitX * -carSpacing,
    carY:
      focalSample.y +
      frame.normalY * innerOffsetWorld +
      tangentUnitY * -carSpacing,
    carHeading: frame.tangentHeadingRadians,
    teamIndex: 0,
    tireState: [1, 1, 1, 1],
  };
  const car2: CarState = {
    carX: focalSample.x - frame.normalX * innerOffsetWorld,
    carY: focalSample.y - frame.normalY * innerOffsetWorld,
    carHeading: frame.tangentHeadingRadians,
    teamIndex: 1,
    tireState: [1, 1, 1, 1],
  };
  const car3: CarState = {
    carX:
      focalSample.x -
      frame.normalX * innerOffsetWorld +
      tangentUnitX * -carSpacing,
    carY:
      focalSample.y -
      frame.normalY * innerOffsetWorld +
      tangentUnitY * -carSpacing,
    carHeading: frame.tangentHeadingRadians,
    teamIndex: 1,
    tireState: [1, 1, 1, 1],
  };

  return createExpandedEnvironmentState({
    carX: car0.carX,
    carY: car0.carY,
    carHeading: car0.carHeading,
    cars: [car0, car1, car2, car3] as const,
  });
}

async function loadObservationAssemblerModule(): Promise<ObservationAssemblerModule> {
  const modulePath = './observation.assembler';
  return (await import(modulePath)) as ObservationAssemblerModule;
}
