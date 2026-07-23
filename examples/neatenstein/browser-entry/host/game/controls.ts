/**
 * DOM input binding functions for the Neatenstein host game layer.
 *
 * These functions wire pointer-lock mouse look, keyboard arrow-key look,
 * touch drag-to-look, and host-to-worker input forwarding. They are kept
 * separate from {@link ../input.ts} so each binding can be attached and detached
 * independently while the central router owns the authoritative input state.
 *
 * @module
 */

import { NEATENSTEIN_INPUT_MESSAGE_TYPE } from '../../constants';
import {
  NEATENSTEIN_KEY_MAP_LOOK,
  NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT,
  NEATENSTEIN_MOUSE_SENSITIVITY,
  NEATENSTEIN_POINTER_LOCK_OPTIONS,
  NEATENSTEIN_PRIMARY_MOUSE_BUTTON,
  NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX,
} from './constants';
import { type InputSnapshot } from '../input';

/**
 * Look delta produced by mouse, keyboard and touch bindings.
 *
 * Values are in radians and are signed so callers can add them directly to
 * the camera yaw/pitch.
 */
export interface LookDelta {
  /** Horizontal rotation (yaw) delta. */
  yawDelta: number;
  /** Vertical rotation (pitch) delta. */
  pitchDelta: number;
}

/**
 * Callback invoked whenever a look binding emits a new delta.
 */
export type LookCallback = (delta: LookDelta) => void;

/**
 * Callback invoked whenever the primary fire button (left mouse) is pressed.
 *
 * The router consumes each press exactly once when the input snapshot is read,
 * so the callback only signals a new fire event rather than a continuous hold
 * state.
 */
export type FireCallback = () => void;

/**
 * Callback invoked when a touch binding starts or ends tracking an active touch.
 */
export type TouchActiveCallback = (active: boolean) => void;

/**
 * Detaches a previously installed input binding.
 */
export type BindingDetach = () => void;

/**
 * Find a touch in a `TouchList` by its identifier.
 *
 * @param touches - Browser touch list.
 * @param identifier - Identifier to match.
 * @returns The matching touch, or `undefined` if not present.
 */
export function findTouch(
  touches: TouchList,
  identifier: number,
): Touch | undefined {
  for (let index = 0; index < touches.length; index++) {
    const touch = touches.item(index);
    if (touch && touch.identifier === identifier) {
      return touch;
    }
  }
  return undefined;
}

/**
 * Request pointer lock on a canvas with raw (unadjusted) mouse movement.
 *
 * Browsers that support `unadjustedMovement: true` disable OS-level mouse
 * acceleration, which is essential for consistent FPS aiming. Older browsers
 * fall back to a plain pointer-lock request.
 *
 * @param canvas - The canvas element to lock the pointer to.
 * @returns A detach function that removes the click listener and exits pointer
 *   lock if this canvas currently owns it.
 *
 * @example
 * ```ts
 * const unbindPointerLock = bindPointerLock(canvas);
 * // later
 * unbindPointerLock();
 * ```
 */
export function bindPointerLock(canvas: HTMLElement): BindingDetach {
  const requestLock = async (): Promise<void> => {
    if (typeof document === 'undefined' || !('requestPointerLock' in canvas)) {
      return;
    }
    try {
      await canvas.requestPointerLock(NEATENSTEIN_POINTER_LOCK_OPTIONS);
    } catch (error) {
      // Some browsers do not support the PointerLockOptions dictionary. Retry
      // without options so older clients still get pointer lock.
      const isUnsupportedOptionsError =
        error instanceof TypeError ||
        (error instanceof Error && error.name === 'NotSupportedError');
      if (!isUnsupportedOptionsError) {
        return;
      }
      try {
        await canvas.requestPointerLock();
      } catch {
        // Pointer-lock requests may fail if not triggered by a user gesture.
        // The caller can surface a fallback UI.
      }
    }
  };

  canvas.addEventListener('click', requestLock);

  return () => {
    canvas.removeEventListener('click', requestLock);
    if (
      typeof document !== 'undefined' &&
      document.pointerLockElement === canvas
    ) {
      if (typeof document.exitPointerLock === 'function') {
        document.exitPointerLock();
      }
    }
  };
}

/**
 * Bind pointer-lock mouse movement to a look callback.
 *
 * Only fires while `document.pointerLockElement === canvas`, so the callback
 * receives raw aim deltas only when the player is actively mousing inside the
 * locked canvas.
 *
 * @param canvas - The canvas that owns pointer lock.
 * @param callback - Receives yaw/pitch deltas in radians.
 * @returns A detach function that removes the `mousemove` listener.
 *
 * @example
 * ```ts
 * const unbind = bindMouseLook(canvas, (delta) => {
 *   camera.yaw += delta.yawDelta;
 *   camera.pitch += delta.pitchDelta;
 * });
 * ```
 */
export function bindMouseLook(
  canvas: HTMLElement,
  callback: LookCallback,
): BindingDetach {
  const handleMouseMove = (event: MouseEvent): void => {
    if (
      typeof document === 'undefined' ||
      document.pointerLockElement !== canvas
    ) {
      return;
    }
    callback({
      yawDelta: event.movementX * NEATENSTEIN_MOUSE_SENSITIVITY,
      pitchDelta: event.movementY * NEATENSTEIN_MOUSE_SENSITIVITY,
    });
  };

  if (typeof document !== 'undefined') {
    document.addEventListener('mousemove', handleMouseMove);
  }

  return () => {
    if (typeof document !== 'undefined') {
      document.removeEventListener('mousemove', handleMouseMove);
    }
  };
}

/**
 * Bind the left mouse button to a fire callback.
 *
 * Fire activates on `mousedown` for the primary (left) button. The router
 * consumes each press exactly once when the input snapshot is read, so this
 * binding does not clear the flag on `mouseup`. This ensures a quick click that
 * falls between simulation ticks still registers one fire event. The binding is
 * intentionally separate from pointer-lock request handling; both can be
 * attached to the same canvas without interfering with each other.
 *
 * @param target - Element that receives the `mousedown` event (usually the
 *   canvas).
 * @param callback - Invoked once for each left-button press.
 * @returns A detach function that removes the mouse button listeners.
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
  const handleMouseDown = (event: MouseEvent): void => {
    if (event.button === NEATENSTEIN_PRIMARY_MOUSE_BUTTON) {
      event.preventDefault();
      callback();
    }
  };

  target.addEventListener('mousedown', handleMouseDown);

  return () => {
    target.removeEventListener('mousedown', handleMouseDown);
  };
}

/**
 * Bind keyboard arrow keys as a fallback look control.
 *
 * Each left/right arrow press emits a fixed yaw delta. This gives players a
 * workable fallback when pointer lock is unavailable (e.g., touch-only or
 * accessibility contexts).
 *
 * @param target - Element that receives `keydown` events.
 * @param callback - Receives yaw/pitch deltas in radians.
 * @returns A detach function that removes the `keydown` listener.
 *
 * @example
 * ```ts
 * const unbind = bindKeyboardLook(canvas, (delta) => {
 *   camera.yaw += delta.yawDelta;
 * });
 * ```
 */
export function bindKeyboardLook(
  target: HTMLElement,
  callback: LookCallback,
): BindingDetach {
  const step = NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT;

  const KEYBOARD_LOOK_DELTA_BY_CODE: Record<string, LookDelta> = {
    [NEATENSTEIN_KEY_MAP_LOOK.left]: { yawDelta: -step, pitchDelta: 0 },
    [NEATENSTEIN_KEY_MAP_LOOK.right]: { yawDelta: step, pitchDelta: 0 },
    [NEATENSTEIN_KEY_MAP_LOOK.up]: { yawDelta: 0, pitchDelta: -step },
    [NEATENSTEIN_KEY_MAP_LOOK.down]: { yawDelta: 0, pitchDelta: step },
  };

  const handleKeyDown = (event: KeyboardEvent): void => {
    const delta = KEYBOARD_LOOK_DELTA_BY_CODE[event.code];
    if (delta !== undefined) {
      event.preventDefault();
      callback(delta);
    }
  };

  target.addEventListener('keydown', handleKeyDown);

  return () => {
    target.removeEventListener('keydown', handleKeyDown);
  };
}

/**
 * Bind touch drag-to-look on a mobile fallback surface.
 *
 * A horizontal drag rotates yaw; vertical drag rotates pitch. The first delta is
 * only emitted once the cumulative drag exceeds the configured threshold so
 * accidental taps do not jerk the camera. After the threshold is crossed, every
 * subsequent move emits the incremental delta from the last emitted position for
 * smooth continuous look control.
 *
 * @param target - Touch surface (usually the canvas or a touch overlay).
 * @param callback - Receives yaw/pitch deltas in radians.
 * @param onActive - Optional callback invoked with `true` when the first active
 *   touch starts and `false` when that touch ends or is cancelled.
 * @returns A detach function that removes the touch listeners.
 *
 * @example
 * ```ts
 * const unbind = bindTouchLook(overlay, (delta) => {
 *   camera.yaw += delta.yawDelta;
 *   camera.pitch += delta.pitchDelta;
 * });
 * ```
 */
export function bindTouchLook(
  target: HTMLElement,
  callback: LookCallback,
  onActive?: TouchActiveCallback,
): BindingDetach {
  let activeTouchId: number | null = null;
  let startX = 0;
  let startY = 0;
  let lastEmittedX = 0;
  let lastEmittedY = 0;
  let engaged = false;

  const threshold = NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX;

  const endActiveTouch = (): void => {
    if (activeTouchId !== null) {
      onActive?.(false);
    }
    activeTouchId = null;
    engaged = false;
  };

  const handleTouchStart = (event: TouchEvent): void => {
    if (activeTouchId !== null) {
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
    if (activeTouchId === null) {
      return;
    }
    const touch = findTouch(event.changedTouches, activeTouchId);
    if (!touch) {
      return;
    }
    event.preventDefault();

    // Keep a small deadzone around the original touch origin so accidental
    // taps do not jerk the camera. Once the drag has left that deadzone, stay
    // engaged until the touch ends for smooth continuous look control.
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

    // Emit the delta relative to the last emitted position so every subsequent
    // drag event contributes smooth, continuous look input.
    const yawDelta =
      (touch.clientX - lastEmittedX) * NEATENSTEIN_MOUSE_SENSITIVITY;
    const pitchDelta =
      (touch.clientY - lastEmittedY) * NEATENSTEIN_MOUSE_SENSITIVITY;
    callback({ yawDelta, pitchDelta });
    lastEmittedX = touch.clientX;
    lastEmittedY = touch.clientY;
  };

  const handleTouchEnd = (event: TouchEvent): void => {
    if (activeTouchId === null) {
      return;
    }
    const touch = findTouch(event.changedTouches, activeTouchId);
    if (touch) {
      endActiveTouch();
    }
  };

  const handleTouchCancel = (event: TouchEvent): void => {
    if (activeTouchId === null) {
      return;
    }
    const touch = findTouch(event.changedTouches, activeTouchId);
    if (touch) {
      endActiveTouch();
    }
  };

  const touchListenerOptions: AddEventListenerOptions = { passive: false };

  target.addEventListener('touchstart', handleTouchStart, touchListenerOptions);
  target.addEventListener('touchmove', handleTouchMove, touchListenerOptions);
  target.addEventListener('touchend', handleTouchEnd);
  target.addEventListener('touchcancel', handleTouchCancel);

  return () => {
    target.removeEventListener(
      'touchstart',
      handleTouchStart,
      touchListenerOptions,
    );
    target.removeEventListener(
      'touchmove',
      handleTouchMove,
      touchListenerOptions,
    );
    target.removeEventListener('touchend', handleTouchEnd);
    target.removeEventListener('touchcancel', handleTouchCancel);
  };
}

/**
 * Forward the worker-consumed subset of an input snapshot to the display worker.
 *
 * The worker only consumes movement intent, look deltas, fire, and dash flags,
 * so this helper posts exactly those fields rather than the complete snapshot.
 *
 * @param worker - The dedicated display worker.
 * @param snapshot - The current host input snapshot.
 * @returns Nothing; the snapshot is posted to the worker asynchronously.
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
      yawDelta: snapshot.look.yawDelta,
      pitchDelta: snapshot.look.pitchDelta,
      fire: snapshot.fire,
      dash: snapshot.dash,
    },
  });
}
