import {
  createInitialState,
  stepEnvironment,
  stepEnvironmentBatch,
} from './environment.step.service';
import type { EnvironmentState, RacingCarState } from './environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';
import {
  resolveInnerLaneCenterlinePoint,
  resolveSplineSampleFrame,
} from '../track/track.spline.utils';

function computeSignedLateralOffset(
  x: number,
  y: number,
  trackSpec: TrackSpec,
): number {
  let nearestSample = trackSpec.splineSamples[0]!;
  let nearestDistance = Infinity;
  for (const sample of trackSpec.splineSamples) {
    const dx = x - sample.x;
    const dy = y - sample.y;
    const distance = dx * dx + dy * dy;
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestSample = sample;
    }
  }
  const frame = resolveSplineSampleFrame(
    trackSpec.splineSamples,
    nearestSample.globalIndex,
  );
  const dx = x - nearestSample.x;
  const dy = y - nearestSample.y;
  return dx * frame.normalX + dy * frame.normalY;
}

describe('environment step service sibling seam', () => {
  describe('stepEnvironmentBatch', () => {
    it('advances the tick count by the requested batch length', () => {
      const nextState = stepEnvironmentBatch(
        createInitialState(),
        { throttle: 1, steer: 0 },
        3,
      );

      expect(nextState.tick).toBe(3);
    });
  });
});

describe('Tier 1/Tier 2 track boundary walls', () => {
  it('keeps a car from crossing the inner track boundary when driving inward', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const halfWidth = sample.width / 2;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const heading = Math.atan2(frame.normalY, frame.normalX);
    const carX = sample.x + frame.normalX * halfWidth;
    const carY = sample.y + frame.normalY * halfWidth;
    const car: RacingCarState = {
      carX,
      carY,
      carHeading: heading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX,
      carY,
      carHeading: heading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [car],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 1, steer: 0 });
    const nextCar = nextState.cars![0]!;
    const offset = computeSignedLateralOffset(
      nextCar.carX,
      nextCar.carY,
      trackSpec,
    );

    expect(offset).toBeLessThanOrEqual(halfWidth + 1e-6);
  });

  it('keeps a car from crossing the outer track boundary when driving outward', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const halfWidth = sample.width / 2;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const heading = Math.atan2(-frame.normalY, -frame.normalX);
    const carX = sample.x - frame.normalX * halfWidth;
    const carY = sample.y - frame.normalY * halfWidth;
    const car: RacingCarState = {
      carX,
      carY,
      carHeading: heading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX,
      carY,
      carHeading: heading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [car],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 1, steer: 0 });
    const nextCar = nextState.cars![0]!;
    const offset = computeSignedLateralOffset(
      nextCar.carX,
      nextCar.carY,
      trackSpec,
    );

    expect(offset).toBeGreaterThanOrEqual(-halfWidth - 1e-6);
  });
});

describe('Tier 1 physics hardening — penalties and car separation', () => {
  it('applies a negative reward when a car is clamped back from outside the track', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const halfWidth = sample.width / 2;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const carX = sample.x + frame.normalX * (halfWidth + 5);
    const carY = sample.y + frame.normalY * (halfWidth + 5);
    const car: RacingCarState = {
      carX,
      carY,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX,
      carY,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [car],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
    const nextCar = nextState.cars![0]!;
    const reward = (nextCar as { reward?: number }).reward;

    expect(reward).toBeLessThan(0);
  });

  it('applies a wrong-direction penalty when the car moves opposite to the track tangent', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const startPoint = resolveInnerLaneCenterlinePoint(sample, frame);
    const wrongHeading = frame.tangentHeadingRadians + Math.PI;
    const car: RacingCarState = {
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: wrongHeading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: wrongHeading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [car],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 1, steer: 0 });
    const nextCar = nextState.cars![0]!;
    const reward = (nextCar as { reward?: number }).reward;

    expect(reward).toBeLessThan(0);
  });

  it('pushes overlapping cars apart so their centers are no longer coincident', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const startPoint = resolveInnerLaneCenterlinePoint(sample, frame);
    const carA: RacingCarState = {
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const carB: RacingCarState = {
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 1,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [carA, carB],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
    const separation = Math.hypot(
      nextState.cars![0]!.carX - nextState.cars![1]!.carX,
      nextState.cars![0]!.carY - nextState.cars![1]!.carY,
    );

    expect(separation).toBeGreaterThan(1e-6);
  });
});
