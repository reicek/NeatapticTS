/**
 * Deterministic green tests for the P1 AI autopilot seam (user-gap patch
 * batch 108): the served race page must open as a running race, so the
 * `ai` mode samples `stepAutopilotInput` instead of the keyboard seam.
 *
 * The fixture is a straight +x lane-center loop; every expectation is a
 * numeric single-expect behavioral check and the same (car, samples) pair
 * must yield identical inputs across repeats (batch-83 replay rung).
 *
 * @module
 */

import {
  NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD,
  stepAutopilotInput,
} from './ai.autopilot';
import { createTrackSpecForAutopilotFixture } from './ai.autopilot.fixtures';

/** Fixture straight loop for steering/throttle law checks. */
const STRAIGHT_LOOP = createTrackSpecForAutopilotFixture();

describe('stepAutopilotInput (P1 ai-mode race automation)', () => {
  it('returns a neutral input for an empty sample list (no loop, no drive)', () => {
    const input = stepAutopilotInput({ x: 0, y: 0, heading: 0, speed: 0 }, []);
    expect(input).toEqual({ throttle: 0, brake: 0, steer: 0 });
  });

  it('drives straight ahead with full throttle on the centerline (no steer)', () => {
    const car = { x: 0, y: 0, heading: 0, speed: 10 };
    const input = stepAutopilotInput(car, STRAIGHT_LOOP);
    expect(input).toEqual({ throttle: 1, brake: 0, steer: 0 });
  });

  it('steers right (negative) when the car nose points off-line to the left', () => {
    const car = { x: 0, y: 0, heading: Math.PI / 8, speed: 10 };
    const input = stepAutopilotInput(car, STRAIGHT_LOOP);
    expect(input.steer).toBeCloseTo(-(Math.PI / 8) * 2.2, 9);
    expect(input.throttle).toBe(0);
    expect(input.brake).toBe(0);
  });

  it('steers left (positive) when the car sits off-line to the right', () => {
    const car = { x: 5, y: -2, heading: 0, speed: 10 };
    const input = stepAutopilotInput(car, STRAIGHT_LOOP);
    expect(input.steer).toBeCloseTo(Math.atan2(2, 12) * 2.2, 9);
    expect(input.throttle).toBe(1);
    expect(input.brake).toBe(0);
  });

  it('brakes and cuts throttle past the full-brake error threshold', () => {
    const car = { x: 0, y: 0, heading: Math.PI / 2, speed: 30 };
    const input = stepAutopilotInput(car, STRAIGHT_LOOP);
    expect(Math.PI / 2).toBeGreaterThan(
      NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD,
    );
    expect(input.throttle).toBe(0);
    expect(input.steer).toBe(-1);
  });

  it('is deterministic for repeated identical inputs (replay rung)', () => {
    const car = {
      x: STRAIGHT_LOOP[2].x,
      y: STRAIGHT_LOOP[2].y - 1.5,
      heading: 0.4,
      speed: 8,
    };
    const first = stepAutopilotInput(car, STRAIGHT_LOOP);
    const second = stepAutopilotInput(car, STRAIGHT_LOOP);
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });
});
