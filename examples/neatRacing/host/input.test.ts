/**
 * Contract tests for the NeatRacing keyboard-to-analog input mapper
 * (slice P1-physics-drive; AC-024, spec section 5.1 decision D5).
 *
 * The mapper is pure and transport-neutral: it reads a one-frame
 * pressed-key map and emits the `PhysicsStepInput` axes sample that the
 * physics step consumes. The same sample object is replayed into every
 * catch-up tick of the frame it was sampled in.
 */

import { describe, expect, it } from '@jest/globals';
import { sampleKeyboardInput, type NeatRacingKeyboardPresses } from './input';

/** Full-throttle, no-steer key map used for the throttle fixture. */
const UP_ONLY: NeatRacingKeyboardPresses = {
  up: true,
  down: false,
  left: false,
  right: false,
};

/** Full-brake, no-steer key map used for the brake fixture. */
const DOWN_ONLY: NeatRacingKeyboardPresses = {
  up: false,
  down: true,
  left: false,
  right: false,
};

/** Counterclockwise steer-only key map. */
const LEFT_ONLY: NeatRacingKeyboardPresses = {
  up: false,
  down: false,
  left: true,
  right: false,
};

/** Clockwise steer-only key map. */
const RIGHT_ONLY: NeatRacingKeyboardPresses = {
  up: false,
  down: false,
  left: false,
  right: true,
};

/** Opposing steer keys held together. */
const LEFT_AND_RIGHT: NeatRacingKeyboardPresses = {
  up: false,
  down: false,
  left: true,
  right: true,
};

/** No keys held. */
const NEUTRAL: NeatRacingKeyboardPresses = {
  up: false,
  down: false,
  left: false,
  right: false,
};

describe('sampleKeyboardInput (D5 pure key-to-axis mapper)', () => {
  it('maps the up key to full positive throttle', () => {
    expect(sampleKeyboardInput(UP_ONLY).throttle).toBe(1);
  });

  it('maps the down key to full positive brake', () => {
    expect(sampleKeyboardInput(DOWN_ONLY).brake).toBe(1);
  });

  it('maps the left key to positive steer (counterclockwise)', () => {
    expect(sampleKeyboardInput(LEFT_ONLY).steer).toBe(1);
  });

  it('maps the right key to negative steer (clockwise)', () => {
    expect(sampleKeyboardInput(RIGHT_ONLY).steer).toBe(-1);
  });

  it('resolves opposing left and right steer keys to zero steer', () => {
    expect(sampleKeyboardInput(LEFT_AND_RIGHT).steer).toBe(0);
  });

  it('returns zero axes when no keys are held', () => {
    const sample = sampleKeyboardInput(NEUTRAL);

    expect(sample.throttle).toBe(0);
    expect(sample.brake).toBe(0);
    expect(sample.steer).toBe(0);
  });

  it('produces byte-identical samples for identical pressed-key maps', () => {
    const sampleA = sampleKeyboardInput(UP_ONLY);
    const sampleB = sampleKeyboardInput(UP_ONLY);

    expect(JSON.stringify(sampleA)).toBe(JSON.stringify(sampleB));
  });
});
