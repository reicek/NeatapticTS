/**
 * DOM input binding functions for the Neatenstein host game layer.
 *
 * These functions wire pointer-lock mouse look, keyboard arrow-key look, touch
 * drag-to-look, mouse fire, and host-to-worker input forwarding.
 *
 * They are intentionally separate from {@link ../input.ts}: this module owns
 * low-level DOM bindings, while the input router owns authoritative input state
 * and snapshot consumption.
 *
 * @module
 */

import { NEATENSTEIN_INPUT_MESSAGE_TYPE } from '../../constants';
import {
  DOM_EVENT_CLICK,
  DOM_EVENT_KEYDOWN,
  DOM_EVENT_MOUSEDOWN,
  DOM_EVENT_MOUSEMOVE,
  DOM_EVENT_TOUCHCANCEL,
  DOM_EVENT_TOUCHEND,
  DOM_EVENT_TOUCHMOVE,
  DOM_EVENT_TOUCHSTART,
} from '../dom-events.constants';
import { ERROR_NOT_SUPPORTED } from '../hud.constants';
import type {
  BindingDetach,
  FireCallback,
  InputSnapshot,
  LightToggleCallback,
  LookCallback,
  LookDelta,
  TouchActiveCallback,
} from '../types';
import {
  NEATENSTEIN_KEY_MAP_LOOK,
  NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT,
  NEATENSTEIN_LIGHT_TOGGLE_KEY,
  NEATENSTEIN_MOUSE_SENSITIVITY,
  NEATENSTEIN_POINTER_LOCK_OPTIONS,
  NEATENSTEIN_PRIMARY_MOUSE_BUTTON,
  NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX,
} from './constants';

// Re-export consolidated types so existing imports from this module remain valid.
export type {
  BindingDetach,
  FireCallback,
  LightToggleCallback,
  LookCallback,
  LookDelta,
  TouchActiveCallback,
} from '../types';

/**
 * Find a touch in a `TouchList` by identifier.
 *
 * @param touches - Browser touch list.
 * @param identifier - Identifier to match.
 * @returns Matching touch, or `undefined` if not present.
 */
export function findTouch(
  touches: TouchList,
  identifier: number,
): Touch | undefined {
  for (let index = 0; index < touches.length; index += 1) {
    const touch = touches.item(index);

    if (touch && touch.identifier === identifier) {
      return touch;
    }
  }

  return undefined;
}

/**
 * Return whether an error likely means `PointerLockOptions` is unsupported.
 *
 * Browsers that do not support the options dictionary can throw either a
 * `TypeError` or a DOM-style `NotSupportedError`.
 *
 * @param error - Caught pointer-lock error.
 * @returns Whether retrying without options is appropriate.
 */
function isUnsupportedPointerLockOptionsError(error: unknown): boolean {
  return (
    error instanceof TypeError ||
    (error instanceof Error && error.name === ERROR_NOT_SUPPORTED)
  );
}

/**
 * Request pointer lock, retrying without options when needed.
 *
 * Some browsers support pointer lock but not the `unadjustedMovement` options
 * dictionary. This helper first attempts the configured options and then falls
 * back to a plain request when the options are unsupported.
 *
 * @param target - Element requesting pointer lock.
 */
async function requestPointerLockWithFallback(
  target: HTMLElement,
): Promise<void> {
  if (
    typeof document === 'undefined' ||
    typeof target.requestPointerLock !== 'function'
  ) {
    return;
  }

  try {
    await target.requestPointerLock(NEATENSTEIN_POINTER_LOCK_OPTIONS);
    return;
  } catch (error) {
    if (!isUnsupportedPointerLockOptionsError(error)) {
      return;
    }
  }

  try {
    await target.requestPointerLock();
  } catch {
    // Pointer-lock requests can fail if not triggered by a user gesture, denied
    // by browser policy, or blocked by the embedding context. The caller can
    // still provide keyboard/touch fallback controls.
  }
}

/**
 * Request pointer lock on a canvas with raw mouse movement when available.
 *
 * @param canvas - Element that should own pointer lock.
 * @returns Idempotent detach function that removes the click listener and exits
 *   pointer lock if this canvas currently owns it.
 *
 * @example
 * ```ts
 * const unbindPointerLock = bindPointerLock(canvas);
 * unbindPointerLock();
 * ```
 */
export function bindPointerLock(canvas: HTMLElement): BindingDetach {
  let detached = false;

  const requestLock = (): void => {
    void requestPointerLockWithFallback(canvas);
  };

  canvas.addEventListener(DOM_EVENT_CLICK, requestLock);

  return () => {
    if (detached) {
      return;
    }

    detached = true;
    canvas.removeEventListener(DOM_EVENT_CLICK, requestLock);

    if (
      typeof document !== 'undefined' &&
      document.pointerLockElement === canvas &&
      typeof document.exitPointerLock === 'function'
    ) {
      document.exitPointerLock();
    }
  };
}

/**
 * Bind pointer-lock mouse movement to a look callback.
 *
 * Only emits while `document.pointerLockElement === canvas`, so callers receive
 * raw aim deltas only when the player is actively controlling the canvas.
 *
 * @param canvas - Element that owns pointer lock.
 * @param callback - Receives yaw/pitch deltas in radians.
 * @returns Idempotent detach function.
 *
 * @example
 * ```ts
 * const unbind = bindMouseLook(canvas, (delta) => {
 *   camera.yaw += delta.yawDelta;
 * });
 * ```
 */
export function bindMouseLook(
  canvas: HTMLElement,
  callback: LookCallback,
): BindingDetach {
  let detached = false;

  const handleMouseMove = (event: MouseEvent): void => {
    if (detached || document.pointerLockElement !== canvas) {
      return;
    }

    callback({
      yawDelta: event.movementX * NEATENSTEIN_MOUSE_SENSITIVITY,
      pitchDelta: event.movementY * NEATENSTEIN_MOUSE_SENSITIVITY,
    });
  };

  document.addEventListener(DOM_EVENT_MOUSEMOVE, handleMouseMove);

  return () => {
    if (detached) {
      return;
    }

    detached = true;

    document.removeEventListener(DOM_EVENT_MOUSEMOVE, handleMouseMove);
  };
}

/**
 * Bind the primary mouse button to a fire callback.
 *
 * Fire activates on `mousedown` for the configured primary button. The binding
 * does not listen for `mouseup`; the input router decides whether fire is
 * edge-triggered, held, or latched.
 *
 * @param target - Element that receives the `mousedown` event.
 * @param callback - Invoked once for each primary-button press.
 * @returns Idempotent detach function.
 *
 * @example
 * ```ts
 * const unbind = bindMouseFire(canvas, () => {
 *   pendingFire = true;
 * });
 * ```
 */
export function bindMouseFire(
  target: HTMLElement,
  callback: FireCallback,
): BindingDetach {
  let detached = false;

  const handleMouseDown = (event: MouseEvent): void => {
    if (detached || event.button !== NEATENSTEIN_PRIMARY_MOUSE_BUTTON) {
      return;
    }

    event.preventDefault();
    callback();
  };

  target.addEventListener(DOM_EVENT_MOUSEDOWN, handleMouseDown);

  return () => {
    if (detached) {
      return;
    }

    detached = true;
    target.removeEventListener(DOM_EVENT_MOUSEDOWN, handleMouseDown);
  };
}

/**
 * Bind keyboard arrow keys as fallback look controls.
 *
 * Each arrow-key `keydown` emits a fixed look delta. Repeated keydown events
 * from the browser naturally produce continuous keyboard look while the key is
 * held.
 *
 * @param target - Event target that receives `keydown` events.
 * @param callback - Receives yaw/pitch deltas in radians.
 * @returns Idempotent detach function.
 *
 * @example
 * ```ts
 * const unbind = bindKeyboardLook(window, (delta) => {
 *   camera.yaw += delta.yawDelta;
 * });
 * ```
 */
export function bindKeyboardLook(
  target: HTMLElement | Window,
  callback: LookCallback,
): BindingDetach {
  let detached = false;
  const step = NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT;

  const keyboardLookDeltaByCode: Record<string, LookDelta> = {
    [NEATENSTEIN_KEY_MAP_LOOK.left]: { yawDelta: -step, pitchDelta: 0 },
    [NEATENSTEIN_KEY_MAP_LOOK.right]: { yawDelta: step, pitchDelta: 0 },
    [NEATENSTEIN_KEY_MAP_LOOK.up]: { yawDelta: 0, pitchDelta: -step },
    [NEATENSTEIN_KEY_MAP_LOOK.down]: { yawDelta: 0, pitchDelta: step },
  };

  const handleKeyDown = (event: KeyboardEvent): void => {
    const delta = keyboardLookDeltaByCode[event.code];

    if (delta !== undefined) {
      event.preventDefault();
      callback(delta);
    }
  };

  target.addEventListener(DOM_EVENT_KEYDOWN, handleKeyDown as EventListener);

  return () => {
    if (detached) {
      return;
    }

    detached = true;
    target.removeEventListener(DOM_EVENT_KEYDOWN, handleKeyDown as EventListener);
  };
}

/**
 * Bind the configured light-toggle key to a callback.
 *
 * The binding only triggers on the initial `keydown` edge for the exact
 * configured key code. Repeated keydown events from the browser (sent while the
 * key is held) are ignored so the light does not flicker. The binding prevents
 * the default browser action and invokes the supplied callback once per press.
 *
 * @param target - Event target that receives `keydown` events.
 * @param callback - Invoked once for each light-toggle press.
 * @returns Idempotent detach function.
 *
 * @example
 * ```ts
 * const unbind = bindKeyboardLightToggle(window, () => {
 *   state.lightEnabled = !state.lightEnabled;
 * });
 * ```
 */
export function bindKeyboardLightToggle(
  target: HTMLElement | Window,
  callback: LightToggleCallback,
): BindingDetach {
  let detached = false;

  const handleKeyDown = (event: KeyboardEvent): void => {
    if (
      detached ||
      event.code !== NEATENSTEIN_LIGHT_TOGGLE_KEY ||
      event.repeat
    ) {
      return;
    }

    event.preventDefault();
    callback();
  };

  target.addEventListener(DOM_EVENT_KEYDOWN, handleKeyDown as EventListener);

  return () => {
    if (detached) {
      return;
    }

    detached = true;
    target.removeEventListener(DOM_EVENT_KEYDOWN, handleKeyDown as EventListener);
  };
}

/**
 * Bind touch drag-to-look on a mobile fallback surface.
 *
 * A drag rotates yaw/pitch. The binding stays in a deadzone until cumulative
 * movement exceeds the configured threshold; once engaged, it emits incremental
 * deltas from the last emitted position. On the first emitted delta, that last
 * emitted position is the original touch start, so the threshold-crossing
 * motion is included rather than discarded.
 *
 * @param target - Touch surface.
 * @param callback - Receives yaw/pitch deltas in radians.
 * @param onActive - Optional callback invoked when tracking starts/ends.
 * @returns Idempotent detach function.
 *
 * @example
 * ```ts
 * const unbind = bindTouchLook(overlay, (delta) => {
 *   camera.yaw += delta.yawDelta;
 * });
 * ```
 */
export function bindTouchLook(
  target: HTMLElement,
  callback: LookCallback,
  onActive?: TouchActiveCallback,
): BindingDetach {
  let detached = false;
  let activeTouchId: number | null = null;
  let startX = 0;
  let startY = 0;
  let lastEmittedX = 0;
  let lastEmittedY = 0;
  let engaged = false;

  const threshold = NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX;

  /**
   * Clear current touch tracking and notify active-state listeners.
   */
  const endActiveTouch = (): void => {
    activeTouchId = null;
    engaged = false;
    onActive?.(false);
  };

  const handleTouchStart = (event: TouchEvent): void => {
    if (detached || activeTouchId !== null) {
      return;
    }

    const touch = event.changedTouches.item(0);

    if (!touch) {
      return;
    }

    event.preventDefault();

    activeTouchId = touch.identifier;
    startX = touch.clientX;
    startY = touch.clientY;
    lastEmittedX = touch.clientX;
    lastEmittedY = touch.clientY;
    engaged = false;

    onActive?.(true);
  };

  const handleTouchMove = (event: TouchEvent): void => {
    if (detached || activeTouchId === null) {
      return;
    }

    const touch = findTouch(event.changedTouches, activeTouchId);

    if (!touch) {
      return;
    }

    event.preventDefault();

    if (!engaged) {
      const cumulativeX = touch.clientX - startX;
      const cumulativeY = touch.clientY - startY;

      if (
        Math.abs(cumulativeX) <= threshold &&
        Math.abs(cumulativeY) <= threshold
      ) {
        return;
      }

      engaged = true;
    }

    const yawDelta =
      (touch.clientX - lastEmittedX) * NEATENSTEIN_MOUSE_SENSITIVITY;
    const pitchDelta =
      (touch.clientY - lastEmittedY) * NEATENSTEIN_MOUSE_SENSITIVITY;

    callback({ yawDelta, pitchDelta });

    lastEmittedX = touch.clientX;
    lastEmittedY = touch.clientY;
  };

  const handleTouchEnd = (event: TouchEvent): void => {
    if (detached || activeTouchId === null) {
      return;
    }

    const touch = findTouch(event.changedTouches, activeTouchId);

    if (touch) {
      event.preventDefault();
      endActiveTouch();
    }
  };

  const handleTouchCancel = (event: TouchEvent): void => {
    if (detached || activeTouchId === null) {
      return;
    }

    const touch = findTouch(event.changedTouches, activeTouchId);

    if (touch) {
      event.preventDefault();
      endActiveTouch();
    }
  };

  const touchListenerOptions: AddEventListenerOptions = { passive: false };

  target.addEventListener(DOM_EVENT_TOUCHSTART, handleTouchStart, touchListenerOptions);
  target.addEventListener(DOM_EVENT_TOUCHMOVE, handleTouchMove, touchListenerOptions);
  target.addEventListener(DOM_EVENT_TOUCHEND, handleTouchEnd, touchListenerOptions);
  target.addEventListener(
    DOM_EVENT_TOUCHCANCEL,
    handleTouchCancel,
    touchListenerOptions,
  );

  return () => {
    if (detached) {
      return;
    }

    detached = true;

    // Detach removes listeners and silently clears internal tracking state.
    // The active callback reflects only real touch lifecycle events; it must
    // not emit an artificial onActive(false) during teardown.
    activeTouchId = null;
    engaged = false;

    target.removeEventListener(
      DOM_EVENT_TOUCHSTART,
      handleTouchStart,
      touchListenerOptions,
    );
    target.removeEventListener(
      DOM_EVENT_TOUCHMOVE,
      handleTouchMove,
      touchListenerOptions,
    );
    target.removeEventListener(
      DOM_EVENT_TOUCHEND,
      handleTouchEnd,
      touchListenerOptions,
    );
    target.removeEventListener(
      DOM_EVENT_TOUCHCANCEL,
      handleTouchCancel,
      touchListenerOptions,
    );
  };
}

/**
 * Forward the worker-consumed subset of an input snapshot to the display worker.
 *
 * The worker consumes movement intent, look deltas, fire, dash, and light
 * toggle flags. This helper posts exactly those fields instead of the complete
 * snapshot.
 *
 * @param worker - Dedicated display worker.
 * @param snapshot - Current host input snapshot.
 *
 * @example
 * ```ts
 * const snapshot = router.getSnapshot();
 * forwardWorkerInput(displayWorker, snapshot);
 * ```
 */
export function forwardWorkerInput(
  worker: Worker,
  snapshot: InputSnapshot,
): void {
  worker.postMessage({
    type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
    input: {
      movement: snapshot.movement,
      look: {
        yawDelta: snapshot.look.yawDelta,
        pitchDelta: snapshot.look.pitchDelta,
      },
      fire: snapshot.fire,
      dash: snapshot.dash,
      lightToggle: snapshot.lightToggle,
    },
  });
}
