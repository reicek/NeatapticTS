/**
 * Contract tests for the Neat Racing physics step service (slice P1-S7-B;
 * AC-023, spec §5.1-5.3, LD-15, TP-01/02/04).
 *
 * Red phase: this suite must fail while './physics.step.service' does not exist
 * ("Cannot find module './physics.step.service'" under ts-jest diagnostics);
 * a green suite during the red phase would be a contract defect. The plan
 * red_test note was reconciled against the spec source of truth: determinism
 * across runs and slip/drift envelopes.
 *
 * Determinism rung: same-runtime byte-identical via deterministic serialization;
 * the step service is pure and contains no Math.random state (§5.3).
 *
 * Green contract for the implementer: './physics.step.service' exports the
 * named constants NEATRACING_PHYSICS_FIXED_TIMESTEP_MS,
 * NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
 * NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S, NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2,
 * NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2,
 * NEATRACING_PHYSICS_NATURAL_DECEL_UNITS_PER_S2,
 * NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2,
 * NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S,
 * NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER, NEATRACING_PHYSICS_STEERING_RATE,
 * the pure step function `stepPhysics(state, input, surfaceResolver?)`, and the
 * factory `createInitialPhysicsCarState(seed?)`. The surface resolver returns a
 * `PhysicsSurfaceType` such as `'road' | 'curb' | 'grass'`; when omitted the
 * service treats the surface as road.
 */

import { describe, expect, it } from '@jest/globals';
import {
  createInitialPhysicsCarState,
  NEATRACING_PHYSICS_FIXED_TIMESTEP_MS,
  NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2,
  NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2,
  NEATRACING_PHYSICS_NATURAL_DECEL_UNITS_PER_S2,
  NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2,
  NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S,
  NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER,
  NEATRACING_PHYSICS_STEERING_RATE,
  stepPhysics,
} from './physics.step.service';
import type { PhysicsCarState, PhysicsStepInput } from './physics.step.service';

/** Minimal surface labels used by the fixture resolver. */
type FixtureSurface = 'road' | 'grass';

/** Steered, full-throttle input used for the slip/drift/determinism fixtures. */
const FIXTURE_INPUT: PhysicsStepInput = { throttle: 1, brake: 0, steer: 1 };

/**
 * Runs the step service for a fixed number of 60 Hz ticks from a seeded
 * initial state and a constant surface.
 */
const runSteps = (
  steps: number,
  seed = 42,
  input = FIXTURE_INPUT,
  surface: FixtureSurface = 'road',
): PhysicsCarState => {
  let state = createInitialPhysicsCarState(seed);
  const surfaceResolver = (): FixtureSurface => surface;
  for (let step = 0; step < steps; step++) {
    state = stepPhysics(state, input, surfaceResolver);
  }
  return state;
};

/** Deterministic JSON serialization for byte-identity comparisons. */
const serializeState = (state: PhysicsCarState): string =>
  JSON.stringify(state);

describe('physics step service constants (spec §5.2, AC-023)', () => {
  it('exposes a deterministic 60 Hz fixed timestep', () => {
    expect(NEATRACING_PHYSICS_FIXED_TIMESTEP_MS).toBeCloseTo(1000 / 60, 10);
    expect(NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS).toBeCloseTo(1 / 60, 12);
  });

  it('targets the ~89 units/s arcade top-speed feel (TP-01)', () => {
    expect(NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S).toBe(89);
  });

  it('derives accel, braking, and natural decel from maxSpeed (§5.2)', () => {
    expect(NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2).toBe(
      NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 5,
    );
    expect(NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2).toBe(
      -NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
    );
    expect(NEATRACING_PHYSICS_NATURAL_DECEL_UNITS_PER_S2).toBe(
      -NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 5,
    );
  });

  it('derives off-road decel and speed floor from maxSpeed (§5.2)', () => {
    expect(NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2).toBe(
      -NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 2,
    );
    expect(NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S).toBe(
      NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 4,
    );
  });

  it('exposes the centrifugal multiplier and steering rate (TP-02, TP-04)', () => {
    expect(NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER).toBe(0.3);
    expect(NEATRACING_PHYSICS_STEERING_RATE).toBe(2);
  });
});

describe('physics step service determinism (spec §5.3)', () => {
  it('produces identical car state for the same seed and inputs across runs', () => {
    const runA = runSteps(60, 42, FIXTURE_INPUT, 'road');
    const runB = runSteps(60, 42, FIXTURE_INPUT, 'road');

    expect(serializeState(runA)).toBe(serializeState(runB));
  });
});

describe('physics step service slip and drift envelopes (LD-15)', () => {
  it('enters a bounded non-zero slip angle under sustained steering', () => {
    const state = runSteps(60, 42, FIXTURE_INPUT, 'road');

    expect(Math.abs(state.slipAngle)).toBeGreaterThan(0);
    expect(Math.abs(state.slipAngle)).toBeLessThan(Math.PI / 2);
    expect(Number.isFinite(state.slipAngle)).toBe(true);
  });

  it('enters a bounded non-zero yaw rate under sustained steering and throttle', () => {
    const state = runSteps(60, 42, FIXTURE_INPUT, 'road');

    expect(Math.abs(state.yawRate)).toBeGreaterThan(0);
    expect(Math.abs(state.yawRate)).toBeLessThan(1);
    expect(Number.isFinite(state.yawRate)).toBe(true);
  });

  it('enforces the off-road speed floor on grass (§5.2)', () => {
    const startSpeed = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
    let state = createInitialPhysicsCarState(42);
    state = { ...state, speed: startSpeed };

    const surfaceResolver = (): FixtureSurface => 'grass';
    for (let step = 0; step < 120; step++) {
      state = stepPhysics(
        state,
        { throttle: 0, brake: 0, steer: 0 },
        surfaceResolver,
      );
    }

    expect(state.speed).toBeGreaterThanOrEqual(
      NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S - 1e-6,
    );
    expect(state.speed).toBeLessThan(startSpeed);
  });
});

describe('physics step service longitudinal motion on road (AC-023)', () => {
  it('accelerates forward from a standing start under full throttle', () => {
    const ticks = 1;
    const input: PhysicsStepInput = { throttle: 1, brake: 0, steer: 0 };
    let state = createInitialPhysicsCarState(42);
    const spawnSpeed = state.speed;

    for (let step = 0; step < ticks; step++) {
      state = stepPhysics(state, input);
    }

    expect(state.speed).toBeGreaterThan(spawnSpeed);
  });

  it('moves forward in x under full throttle with no steering', () => {
    const ticks = 10;
    const input: PhysicsStepInput = { throttle: 1, brake: 0, steer: 0 };
    let state = createInitialPhysicsCarState(42);
    const spawnX = state.x;

    for (let step = 0; step < ticks; step++) {
      state = stepPhysics(state, input);
    }

    expect(state.x).toBeGreaterThan(spawnX);
  });
});
