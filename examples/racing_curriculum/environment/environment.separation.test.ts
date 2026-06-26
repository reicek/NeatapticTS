import {
  createInitialState,
  stepEnvironment,
} from './environment.step.service';
import type { EnvironmentState, RacingCarState } from './environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';
import {
  resolveInnerLaneCenterlinePoint,
  resolveSplineSampleFrame,
} from '../track/track.spline.utils';

/**
 * Car bounding-box half-dimensions in world units.
 *
 * These mirror the renderer constants `CAR_HALF_LENGTH_WORLD` (3.8) and
 * `CAR_HALF_WIDTH_WORLD` (2.2) so the environment separation contract uses the
 * same physical car size the visual layer renders.
 */
const CAR_HALF_LENGTH = 3.8;
const CAR_HALF_WIDTH = 2.2;

/**
 * Axis-aligned bounding box for one car centred at (carX, carY).
 *
 * Cars are treated as axis-aligned rectangles for separation purposes. The
 * half-extents are `CAR_HALF_WIDTH` along X and `CAR_HALF_LENGTH` along Y,
 * matching the renderer's world-space car footprint.
 */
type CarAabb = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
};

function carAabb(car: RacingCarState): CarAabb {
  return {
    minX: car.carX - CAR_HALF_WIDTH,
    maxX: car.carX + CAR_HALF_WIDTH,
    minY: car.carY - CAR_HALF_LENGTH,
    maxY: car.carY + CAR_HALF_LENGTH,
  };
}

/** Returns true when two axis-aligned bounding boxes overlap (share interior area). */
function aabbOverlap(a: CarAabb, b: CarAabb): boolean {
  return a.minX < b.maxX && a.maxX > b.minX && a.minY < b.maxY && a.maxY > b.minY;
}

/** Minimum required center-to-center distance so bounding boxes do not overlap. */
const MIN_CENTER_SEPARATION = CAR_HALF_WIDTH * 2 + 1;

/**
 * Builds a minimal two-car state with both cars placed at the given offset from
 * the inner-lane start point, sharing the same heading.
 */
function buildTwoCarState(
  trackSpec: TrackSpec,
  offsetX: number,
  offsetY: number,
): EnvironmentState {
  const sample = trackSpec.splineSamples[0]!;
  const frame = resolveSplineSampleFrame(
    trackSpec.splineSamples,
    sample.globalIndex,
  );
  const startPoint = resolveInnerLaneCenterlinePoint(sample, frame);
  const heading = frame.tangentHeadingRadians;
  const carA: RacingCarState = {
    carX: startPoint.x,
    carY: startPoint.y,
    carHeading: heading,
    teamIndex: 0,
    tireState: [1, 1, 1, 1],
  };
  const carB: RacingCarState = {
    carX: startPoint.x + offsetX,
    carY: startPoint.y + offsetY,
    carHeading: heading,
    teamIndex: 1,
    tireState: [1, 1, 1, 1],
  };

  return {
    tick: 0,
    carX: carA.carX,
    carY: carA.carY,
    carHeading: heading,
    teamIndex: 0,
    tireState: [1, 1, 1, 1],
    cars: [carA, carB],
    trackSpec,
  };
}

/**
 * Builds a multi-car state with `count` cars stacked at the same position.
 *
 * All cars share the start point and heading so every pair overlaps.
 */
function buildStackedCarState(
  trackSpec: TrackSpec,
  count: number,
): EnvironmentState {
  const sample = trackSpec.splineSamples[0]!;
  const frame = resolveSplineSampleFrame(
    trackSpec.splineSamples,
    sample.globalIndex,
  );
  const startPoint = resolveInnerLaneCenterlinePoint(sample, frame);
  const heading = frame.tangentHeadingRadians;
  const cars: RacingCarState[] = [];
  for (let index = 0; index < count; index++) {
    cars.push({
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: heading,
      teamIndex: (index % 2) as 0 | 1,
      tireState: [1, 1, 1, 1],
    });
  }

  return {
    tick: 0,
    carX: cars[0]!.carX,
    carY: cars[0]!.carY,
    carHeading: heading,
    teamIndex: 0,
    tireState: [1, 1, 1, 1],
    cars,
    trackSpec,
  };
}

describe('Strengthened car separation — bounding box non-overlap', () => {
  const trackSpec = generateTrack({
    seed: 42,
    layoutVersion: 1,
    sizeBucket: 'medium',
  });

  describe('static overlap prevention', () => {
    it('pushes two coincident cars apart so their bounding boxes do not overlap', () => {
      const state = buildTwoCarState(trackSpec, 0, 0);

      const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
      const carA = nextState.cars![0]!;
      const carB = nextState.cars![1]!;
      const overlap = aabbOverlap(carAabb(carA), carAabb(carB));

      expect(overlap).toBe(false);
    });

    it('pushes two partially overlapping cars apart so their bounding boxes do not overlap', () => {
      const state = buildTwoCarState(trackSpec, CAR_HALF_WIDTH, 0);

      const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
      const carA = nextState.cars![0]!;
      const carB = nextState.cars![1]!;
      const overlap = aabbOverlap(carAabb(carA), carAabb(carB));

      expect(overlap).toBe(false);
    });

    it('pushes two cars overlapping along the Y axis apart so their bounding boxes do not overlap', () => {
      const state = buildTwoCarState(trackSpec, 0, CAR_HALF_LENGTH);

      const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
      const carA = nextState.cars![0]!;
      const carB = nextState.cars![1]!;
      const overlap = aabbOverlap(carAabb(carA), carAabb(carB));

      expect(overlap).toBe(false);
    });
  });

  describe('minimum separation distance', () => {
    it('maintains at least the bounding-box-derived minimum center distance between separated cars', () => {
      const state = buildTwoCarState(trackSpec, 0, 0);

      const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
      const carA = nextState.cars![0]!;
      const carB = nextState.cars![1]!;
      const separation = Math.hypot(
        carB.carX - carA.carX,
        carB.carY - carA.carY,
      );

      expect(separation).toBeGreaterThanOrEqual(MIN_CENTER_SEPARATION);
    });

    it('maintains minimum center distance even when cars start partially overlapping', () => {
      const state = buildTwoCarState(trackSpec, CAR_HALF_WIDTH * 0.5, 0);

      const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
      const carA = nextState.cars![0]!;
      const carB = nextState.cars![1]!;
      const separation = Math.hypot(
        carB.carX - carA.carX,
        carB.carY - carA.carY,
      );

      expect(separation).toBeGreaterThanOrEqual(MIN_CENTER_SEPARATION);
    });
  });

  describe('separation applies to all cars', () => {
    it('separates every pair in a four-car stack so no bounding boxes overlap', () => {
      const state = buildStackedCarState(trackSpec, 4);

      const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
      const cars = nextState.cars!;
      let anyOverlap = false;
      for (let i = 0; i < cars.length && !anyOverlap; i++) {
        for (let j = i + 1; j < cars.length; j++) {
          if (aabbOverlap(carAabb(cars[i]!), carAabb(cars[j]!))) {
            anyOverlap = true;
          }
        }
      }

      expect(anyOverlap).toBe(false);
    });

    it('separates every pair in a six-car stack so no bounding boxes overlap', () => {
      const state = buildStackedCarState(trackSpec, 6);

      const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
      const cars = nextState.cars!;
      let anyOverlap = false;
      for (let i = 0; i < cars.length && !anyOverlap; i++) {
        for (let j = i + 1; j < cars.length; j++) {
          if (aabbOverlap(carAabb(cars[i]!), carAabb(cars[j]!))) {
            anyOverlap = true;
          }
        }
      }

      expect(anyOverlap).toBe(false);
    });

    it('maintains minimum center separation across all pairs in a four-car stack', () => {
      const state = buildStackedCarState(trackSpec, 4);

      const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
      const cars = nextState.cars!;
      let minSeparation = Infinity;
      for (let i = 0; i < cars.length; i++) {
        for (let j = i + 1; j < cars.length; j++) {
          const dist = Math.hypot(
            cars[j]!.carX - cars[i]!.carX,
            cars[j]!.carY - cars[i]!.carY,
          );
          if (dist < minSeparation) {
            minSeparation = dist;
          }
        }
      }

      expect(minSeparation).toBeGreaterThanOrEqual(MIN_CENTER_SEPARATION);
    });
  });

  describe('separation during movement', () => {
    it('separates cars that move into overlapping positions when throttle is applied', () => {
      const sample = trackSpec.splineSamples[0]!;
      const frame = resolveSplineSampleFrame(
        trackSpec.splineSamples,
        sample.globalIndex,
      );
      const startPoint = resolveInnerLaneCenterlinePoint(sample, frame);
      const heading = frame.tangentHeadingRadians;
      const carA: RacingCarState = {
        carX: startPoint.x,
        carY: startPoint.y,
        carHeading: heading,
        teamIndex: 0,
        tireState: [1, 1, 1, 1],
      };
      const carB: RacingCarState = {
        carX: startPoint.x + CAR_HALF_WIDTH * 0.3,
        carY: startPoint.y,
        carHeading: heading,
        teamIndex: 1,
        tireState: [1, 1, 1, 1],
      };
      const state: EnvironmentState = {
        tick: 0,
        carX: carA.carX,
        carY: carA.carY,
        carHeading: heading,
        teamIndex: 0,
        tireState: [1, 1, 1, 1],
        cars: [carA, carB],
        trackSpec,
      };

      const nextState = stepEnvironment(state, [
        { throttle: 1, steer: 0 },
        { throttle: 1, steer: 0 },
      ]);
      const nextCarA = nextState.cars![0]!;
      const nextCarB = nextState.cars![1]!;
      const overlap = aabbOverlap(carAabb(nextCarA), carAabb(nextCarB));

      expect(overlap).toBe(false);
    });

    it('maintains minimum center separation when both cars accelerate from a near-overlap start', () => {
      const state = buildTwoCarState(trackSpec, CAR_HALF_WIDTH * 0.4, 0);

      const nextState = stepEnvironment(state, [
        { throttle: 1, steer: 0 },
        { throttle: 1, steer: 0 },
      ]);
      const carA = nextState.cars![0]!;
      const carB = nextState.cars![1]!;
      const separation = Math.hypot(
        carB.carX - carA.carX,
        carB.carY - carA.carY,
      );

      expect(separation).toBeGreaterThanOrEqual(MIN_CENTER_SEPARATION);
    });

    it('prevents bounding box overlap after multiple steps from a coincident start with throttle', () => {
      const state = buildTwoCarState(trackSpec, 0, 0);

      let nextState = state;
      for (let step = 0; step < 5; step++) {
        nextState = stepEnvironment(nextState, [
          { throttle: 1, steer: 0 },
          { throttle: 1, steer: 0 },
        ]);
      }
      const carA = nextState.cars![0]!;
      const carB = nextState.cars![1]!;
      const overlap = aabbOverlap(carAabb(carA), carAabb(carB));

      expect(overlap).toBe(false);
    });
  });

  describe('separation does not break single-car mode', () => {
    it('still advances a single car without error when no other cars are present', () => {
      const initial = createInitialState();
      const singleCarState: EnvironmentState = {
        ...initial,
        cars: [initial.cars![0]!],
      };

      const nextState = stepEnvironment(singleCarState, { throttle: 1, steer: 0 });
      const car = nextState.cars![0]!;

      expect(car.carX).not.toBe(singleCarState.cars![0]!.carX);
    });
  });
});