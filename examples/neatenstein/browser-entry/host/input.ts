/**
 * Host-side input router for the Neatenstein neon raycasting demo.
 *
 * The router attaches to a DOM target, accumulates key, mouse, touch and
 * pointer-lock events, and produces a consumable {@link InputSnapshot} each
 * simulation tick. Controls.ts binds individual input modalities to callbacks,
 * while this module centralises the authoritative state that is forwarded to the
 * worker tier.
 *
 * @module
 */

import {
  NEATENSTEIN_DASH_KEY,
  NEATENSTEIN_FIRE_KEY,
  NEATENSTEIN_KEY_MAP_MOVEMENT,
} from './game/constants';
import {
  bindKeyboardLook,
  bindMouseFire,
  bindMouseLook,
  bindPointerLock,
  bindTouchLook,
  type BindingDetach,
  type LookDelta,
} from './game/controls';

/** Convenience type alias for a keyboard handler reference. */
type KeyboardEventHandler = (event: KeyboardEvent) => void;

/**
 * Snapshot of raw player input at a single point in time.
 *
 * The snapshot is intentionally clone-safe so it can be passed across the
 * worker boundary via `postMessage` without losing information.
 */
export interface InputSnapshot {
  /** Milliseconds since epoch when the snapshot was captured. */
  timestamp: number;

  /** Directional movement intent from the W/A/S/D keys. */
  movement: {
    /** True while the forward key is held. */
    forward: boolean;
    /** True while the backward key is held. */
    backward: boolean;
    /** True while the strafe-left key is held. */
    left: boolean;
    /** True while the strafe-right key is held. */
    right: boolean;
  };

  /**
   * Accumulated orientation change since the last snapshot, in radians.
   *
   * Mouse deltas are consumed when the snapshot is read so the same physical
   * movement is never applied twice.
   */
  look: {
    /** Horizontal rotation (yaw) delta. */
    yawDelta: number;
    /** Vertical rotation (pitch) delta. */
    pitchDelta: number;
  };

  /** Current touch drag-to-look state for mobile fallback. */
  touch: {
    /** True while at least one active touch is being tracked. */
    active: boolean;
    /** Horizontal yaw delta from the active touch drag. */
    yawDelta: number;
    /** Vertical pitch delta from the active touch drag. */
    pitchDelta: number;
  };

  /** Whether the pointer is currently locked to the attached target. */
  pointerLocked: boolean;

  /** True while the primary fire input (left mouse button or fire key) is held. */
  fire: boolean;

  /** True while the dash key is held. */
  dash: boolean;
}

/** Detaches all event listeners installed by the router. */
export type InputRouterDetach = () => void;

/**
 * Public surface of a host input router.
 */
export interface InputRouter {
  /**
   * Attach input listeners to a DOM target.
   *
   * @param target - The element that owns pointer lock and receives key/touch
   *   events. Mouse-move events are read from `document` so they continue to
   *   fire while the pointer is locked.
   * @throws {Error} if the router is already attached to another target.
   */
  attach(target: HTMLElement): InputRouterDetach;

  /** Remove all installed listeners and reset transient input state. */
  detach(): void;

  /**
   * Capture and reset the current input snapshot.
   *
   * Calling this consumes pending mouse and touch deltas; repeated calls
   * without new input will return zero deltas.
   *
   * @returns A clone-safe input snapshot.
   */
  getSnapshot(): InputSnapshot;
}

/**
 * Create a host-side input router.
 *
 * The returned router tracks keyboard movement/look keys, pointer-lock mouse
 * deltas, and touch drag-to-look state. It produces a fresh
 * {@link InputSnapshot} each tick that can be forwarded to the simulation and
 * the worker tier.
 *
 * @returns A new {@link InputRouter} instance.
 * @throws {Error} if `attach()` is called while the router is already attached.
 *
 * @example
 * ```ts
 * const router = createInputRouter();
 * const detach = router.attach(canvas);
 * const snapshot = router.getSnapshot();
 * detach();
 * ```
 */
export function createInputRouter(): InputRouter {
  let attachedTarget: HTMLElement | null = null;
  const keyStates: Record<string, boolean> = {};

  let mouseYawDelta = 0;
  let mousePitchDelta = 0;
  let keyboardYawDelta = 0;
  let keyboardPitchDelta = 0;
  let touchYawDelta = 0;
  let touchPitchDelta = 0;

  let keyDownHandler: KeyboardEventHandler | null = null;
  let keyUpHandler: KeyboardEventHandler | null = null;
  let pointerLockDetach: BindingDetach | null = null;
  let mouseFireDetach: BindingDetach | null = null;
  let mouseLookDetach: BindingDetach | null = null;
  let keyboardLookDetach: BindingDetach | null = null;
  let touchLookDetach: BindingDetach | null = null;

  /** True while the primary fire button (left mouse) is held. */
  let mouseFireActive = false;

  /**
   * Update movement/fire key states from a keyboard event.
   *
   * @param event - Keyboard event.
   * @param pressed - `true` for keydown, `false` for keyup.
   */
  function updateKeyState(event: KeyboardEvent, pressed: boolean): void {
    keyStates[event.code] = pressed;
  }

  /**
   * Accumulate keyboard look deltas.
   *
   * @param delta - Look delta emitted by the keyboard binding.
   */
  function onKeyboardLook(delta: LookDelta): void {
    keyboardYawDelta += delta.yawDelta;
    keyboardPitchDelta += delta.pitchDelta;
  }

  /**
   * Accumulate mouse look deltas.
   *
   * @param delta - Look delta emitted by the mouse binding.
   */
  function onMouseLook(delta: LookDelta): void {
    mouseYawDelta += delta.yawDelta;
    mousePitchDelta += delta.pitchDelta;
  }

  /**
   * Accumulate touch look deltas.
   *
   * @param delta - Look delta emitted by the touch binding.
   */
  function onTouchLook(delta: LookDelta): void {
    touchYawDelta += delta.yawDelta;
    touchPitchDelta += delta.pitchDelta;
  }

  /**
   * Reset all transient input state without touching event listeners.
   *
   * Used by both the public `detach()` method and the per-attach detach
   * closure so re-attaching starts from a clean state.
   */
  function resetInputState(): void {
    for (const code of Object.keys(keyStates)) {
      delete keyStates[code];
    }
    mouseYawDelta = 0;
    mousePitchDelta = 0;
    keyboardYawDelta = 0;
    keyboardPitchDelta = 0;
    touchYawDelta = 0;
    touchPitchDelta = 0;
    mouseFireActive = false;
  }

  function bindEventListeners(target: HTMLElement): void {
    keyDownHandler = (event) => updateKeyState(event, true);
    keyUpHandler = (event) => updateKeyState(event, false);
    target.addEventListener('keydown', keyDownHandler);
    target.addEventListener('keyup', keyUpHandler);
  }

  function bindLookListeners(target: HTMLElement): void {
    mouseLookDetach = bindMouseLook(target, onMouseLook);
    keyboardLookDetach = bindKeyboardLook(target, onKeyboardLook);
    touchLookDetach = bindTouchLook(target, onTouchLook);
  }

  function removeMovementListeners(): void {
    if (!attachedTarget) return;
    if (keyDownHandler) {
      attachedTarget.removeEventListener('keydown', keyDownHandler);
    }
    if (keyUpHandler) {
      attachedTarget.removeEventListener('keyup', keyUpHandler);
    }
    keyDownHandler = null;
    keyUpHandler = null;
  }

  function removeLookListeners(): void {
    mouseLookDetach?.();
    mouseLookDetach = null;
    keyboardLookDetach?.();
    keyboardLookDetach = null;
    touchLookDetach?.();
    touchLookDetach = null;
  }

  return {
    attach(target: HTMLElement): InputRouterDetach {
      if (attachedTarget !== null) {
        throw new Error(
          'InputRouter is already attached; detach before re-attaching.',
        );
      }

      attachedTarget = target;
      pointerLockDetach = bindPointerLock(target);
      mouseFireDetach = bindMouseFire(target, (active) => {
        mouseFireActive = active;
      });
      bindEventListeners(target);
      bindLookListeners(target);

      return () => {
        pointerLockDetach?.();
        pointerLockDetach = null;
        mouseFireDetach?.();
        mouseFireDetach = null;
        removeLookListeners();
        removeMovementListeners();
        attachedTarget = null;
        resetInputState();
      };
    },

    detach(): void {
      pointerLockDetach?.();
      pointerLockDetach = null;
      mouseFireDetach?.();
      mouseFireDetach = null;
      removeLookListeners();
      removeMovementListeners();
      attachedTarget = null;
      resetInputState();
    },

    getSnapshot(): InputSnapshot {
      const now = Date.now();
      const pointerLocked =
        typeof document !== 'undefined' &&
        attachedTarget !== null &&
        document.pointerLockElement === attachedTarget;

      const snapshot: InputSnapshot = {
        timestamp: now,
        movement: {
          forward: keyStates[NEATENSTEIN_KEY_MAP_MOVEMENT.forward] ?? false,
          backward: keyStates[NEATENSTEIN_KEY_MAP_MOVEMENT.backward] ?? false,
          left: keyStates[NEATENSTEIN_KEY_MAP_MOVEMENT.left] ?? false,
          right: keyStates[NEATENSTEIN_KEY_MAP_MOVEMENT.right] ?? false,
        },
        look: {
          yawDelta: mouseYawDelta + touchYawDelta + keyboardYawDelta,
          pitchDelta: mousePitchDelta + touchPitchDelta + keyboardPitchDelta,
        },
        touch: {
          active: false,
          yawDelta: touchYawDelta,
          pitchDelta: touchPitchDelta,
        },
        pointerLocked,
        fire: mouseFireActive || (keyStates[NEATENSTEIN_FIRE_KEY] ?? false),
        dash: keyStates[NEATENSTEIN_DASH_KEY] ?? false,
      };

      mouseYawDelta = 0;
      mousePitchDelta = 0;
      keyboardYawDelta = 0;
      keyboardPitchDelta = 0;
      touchYawDelta = 0;
      touchPitchDelta = 0;

      return snapshot;
    },
  };
}
