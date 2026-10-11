/**
 * Keyboard input map for the neatRacing example (plan slice P1-S7-C;
 * AC-024 partial, spec section 5.1 decision D5).
 *
 * D5 sampling rule: keyboard state is sampled ONCE per rAF frame by the
 * host into a transport-neutral pressed-key map
 * ({@link NeatRacingKeyboardPresses}); this module maps that sample onto
 * the analog axes the physics step consumes, and the host replays the
 * SAME sample object unchanged into EVERY catch-up tick of that frame.
 * Key-to-press binding is deliberately the host's DOM concern (key
 * codes, focus, preventDefault), so this mapper holds no browser state
 * and reads no DOM: it is a pure boolean-to-axis map.
 *
 * The output type is the `PhysicsStepInput` axes shape itself, so a
 * sampled keyboard sample replays into `stepPhysics` verbatim with no
 * adapter step that could re-order or re-quantize the axes.
 *
 * Determinism: `sampleKeyboardInput` is pure — identical pressed-key
 * maps always produce byte-identical samples (`JSON.stringify`-equal)
 * with no runtime-random state.
 */

import type { PhysicsStepInput } from '../physics/physics.step.service';

/**
 * The once-per-rAF-frame pressed-key map the host feeds to
 * {@link sampleKeyboardInput} (D5). Transport neutral: the map carries
 * plain boolean pressed state; how `up`/`down`/`left`/`right` bind to
 * physical key codes is the host's DOM concern, not the sim's.
 */
export interface NeatRacingKeyboardPresses {
  /** Forward key held down. */
  readonly up: boolean;
  /** Brake key held down. */
  readonly down: boolean;
  /** Counterclockwise (left) steer key held down. */
  readonly left: boolean;
  /** Clockwise (right) steer key held down. */
  readonly right: boolean;
}

/**
 * One-keyboard-frame analog axes sample (D5): exactly the
 * `PhysicsStepInput` shape, so the host can apply the same sample
 * object unchanged to every catch-up tick of the frame it was sampled
 * in, and fresh identical samples per tick always land byte-identical.
 */
export type NeatRacingKeyboardSample = PhysicsStepInput;

/** Axis ratio of a pressed key (keys are unit-quantized on/off inputs). */
const PRESSED_AXIS_RATIO = 1;

/** Axis ratio of an unpressed key. */
const NEUTRAL_AXIS_RATIO = 0;

/**
 * Maps one once-per-rAF-frame pressed-key map onto the analog axes
 * sample (`spec section 5.1` D5).
 *
 * The up key maps to full positive throttle, the down key to full
 * positive brake, the left key to steer +1 (counterclockwise per the
 * steer axis contract), and the right key to steer -1. A neutral map
 * yields three zero axes. Pressing left and right together resolves to
 * a zero steer axis (the symmetric analog difference of the two
 * directions); pressing up and down together keeps both axes at full
 * ratio — brake precedence is enforced by the physics tick itself (the
 * `PhysicsStepInput` contract: "a positive brake wins over throttle"),
 * not by remapping keys here.
 *
 * @param presses - Host-collected pressed-key map sampled once for this
 *   rAF frame.
 * @returns The axes sample applied, unchanged, to every catch-up tick
 *   of that frame before the next per-frame sample is taken (D5).
 *
 * @example
 * ```ts
 * const sample = sampleKeyboardInput({
 *   up: true, down: false, left: true, right: false,
 * });
 * console.log(sample.throttle, sample.steer); // 1, 1 — forward and left
 * ```
 */
export function sampleKeyboardInput(
  presses: NeatRacingKeyboardPresses,
): NeatRacingKeyboardSample {
  return {
    throttle: presses.up ? PRESSED_AXIS_RATIO : NEUTRAL_AXIS_RATIO,
    brake: presses.down ? PRESSED_AXIS_RATIO : NEUTRAL_AXIS_RATIO,
    steer:
      (presses.left ? PRESSED_AXIS_RATIO : NEUTRAL_AXIS_RATIO) -
      (presses.right ? PRESSED_AXIS_RATIO : NEUTRAL_AXIS_RATIO),
  };
}
