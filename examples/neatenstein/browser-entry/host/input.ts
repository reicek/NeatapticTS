/**
 * Host-side input router for the Neatenstein neon raycasting demo.
 *
 * The router attaches to a DOM target, accumulates keyboard, mouse, touch, and
 * pointer-lock input, and produces a consumable {@link InputSnapshot} each
 * simulation tick.
 *
 * Controls.ts owns low-level modality bindings. This module centralizes the
 * authoritative, clone-safe input state that is forwarded to the worker tier.
 *
 * @module
 */

import {
  NEATENSTEIN_DASH_KEY,
  NEATENSTEIN_FIRE_KEY,
  NEATENSTEIN_KEY_MAP_MOVEMENT,
} from './game/constants';
import {
  bindKeyboardLightToggle,
  bindKeyboardLook,
  bindMouseFire,
  bindMouseLook,
  bindPointerLock,
  bindTouchLook,
  type BindingDetach,
  type LookDelta,
  type TouchActiveCallback,
} from './game/controls';

/** Convenience type alias for a keyboard handler reference. */
type KeyboardEventHandler = (event: KeyboardEvent) => void;

/** Convenience type alias for no-argument DOM event handlers. */
type VoidDomEventHandler = () => void;

/**
 * Snapshot of raw player input at a single point in time.
 *
 * The snapshot is intentionally clone-safe so it can be passed across the
 * worker boundary via `postMessage` without losing information.
 */
export interface InputSnapshot {
  /** Milliseconds since epoch when the snapshot was captured. */
  timestamp: number;

  /** Directional movement intent from movement keys. */
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
   * Mouse, touch, and keyboard look deltas are consumed when the snapshot is
   * read so the same physical movement is never applied twice.
   */
  look: {
    /** Horizontal rotation delta. */
    yawDelta: number;
    /** Vertical rotation delta. */
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

  /**
   * Primary fire input for the current snapshot.
   *
   * Mouse clicks and keyboard fire presses are latched until consumed so quick
   * taps between render frames are not lost.
   */
  fire: boolean;

  /**
   * Dash input for the current snapshot.
   *
   * Dash presses are latched until consumed so quick taps between render frames
   * are not lost.
   */
  dash: boolean;

  /**
   * Dynamic light toggle input for the current snapshot.
   *
   * Toggle presses are latched until consumed so quick taps between render
   * frames are not lost.
   */
  lightToggle: boolean;
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
   * @param target - Element that owns pointer lock and receives mouse/touch
   *   events. Keyboard events are read from `window`.
   * @returns A detach function for this attachment.
   * @throws {Error} If the router is already attached.
   */
  attach(target: HTMLElement): InputRouterDetach;

  /** Remove all installed listeners and reset transient input state. */
  detach(): void;

  /**
   * Capture and reset the current input snapshot.
   *
   * Calling this consumes pending mouse, touch, keyboard-look, fire, and dash
   * deltas/actions.
   *
   * @returns A clone-safe input snapshot.
   */
  getSnapshot(): InputSnapshot;
}

/**
 * Create a host-side input router.
 *
 * @returns A new {@link InputRouter} instance.
 */
export function createInputRouter(): InputRouter {
  let attachedTarget: HTMLElement | null = null;
  let attachmentId = 0;

  const keyStates: Record<string, boolean> = {};

  let mouseYawDelta = 0;
  let mousePitchDelta = 0;
  let keyboardYawDelta = 0;
  let keyboardPitchDelta = 0;
  let touchYawDelta = 0;
  let touchPitchDelta = 0;

  let pendingFire = false;
  let pendingDash = false;
  let pendingLightToggle = false;
  let touchActive = false;

  let keyDownHandler: KeyboardEventHandler | null = null;
  let keyUpHandler: KeyboardEventHandler | null = null;
  let windowBlurHandler: VoidDomEventHandler | null = null;
  let visibilityChangeHandler: VoidDomEventHandler | null = null;

  let pointerLockDetach: BindingDetach | null = null;
  let mouseFireDetach: BindingDetach | null = null;
  let mouseLookDetach: BindingDetach | null = null;
  let keyboardLookDetach: BindingDetach | null = null;
  let keyboardLightToggleDetach: BindingDetach | null = null;
  let touchLookDetach: BindingDetach | null = null;

  /**
   * Return whether a key code is one of the router's action keys.
   *
   * @param code - KeyboardEvent.code value.
   * @returns Whether the code represents a one-shot action.
   */
  function isActionKey(code: string): boolean {
    return code === NEATENSTEIN_FIRE_KEY || code === NEATENSTEIN_DASH_KEY;
  }

  /**
   * Update movement/action key states from a keyboard event.
   *
   * Fire and dash keydown events are latched on first press so quick taps
   * between snapshots are preserved.
   *
   * @param event - Keyboard event.
   * @param pressed - `true` for keydown, `false` for keyup.
   */
  function updateKeyState(event: KeyboardEvent, pressed: boolean): void {
    const wasPressed = keyStates[event.code] === true;
    keyStates[event.code] = pressed;

    if (pressed && !wasPressed) {
      if (event.code === NEATENSTEIN_FIRE_KEY) {
        pendingFire = true;
      }

      if (event.code === NEATENSTEIN_DASH_KEY) {
        pendingDash = true;
      }
    }

    // Prevent common gameplay keys from scrolling or activating page controls.
    if (
      isActionKey(event.code) ||
      (
        Object.values(NEATENSTEIN_KEY_MAP_MOVEMENT) as readonly string[]
      ).includes(event.code)
    ) {
      event.preventDefault();
    }
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
   * Track whether touch look input is active.
   *
   * @param active - Whether a touch gesture is currently active.
   */
  const onTouchActive: TouchActiveCallback = (active): void => {
    touchActive = active;
  };

  /**
   * Reset tracked key state only.
   *
   * Useful when the window loses focus and keyup events may not be delivered.
   */
  function resetKeyState(): void {
    for (const code of Object.keys(keyStates)) {
      delete keyStates[code];
    }
  }

  /**
   * Reset transient input state without touching installed listeners.
   */
  function resetInputState(): void {
    resetKeyState();
    mouseYawDelta = 0;
    mousePitchDelta = 0;
    keyboardYawDelta = 0;
    keyboardPitchDelta = 0;
    touchYawDelta = 0;
    touchPitchDelta = 0;
    pendingFire = false;
    pendingDash = false;
    pendingLightToggle = false;
    touchActive = false;
  }

  /**
   * Bind keyboard and lifecycle listeners.
   */
  function bindEventListeners(): void {
    keyDownHandler = (event) => updateKeyState(event, true);
    keyUpHandler = (event) => updateKeyState(event, false);
    windowBlurHandler = () => resetKeyState();
    visibilityChangeHandler = () => {
      if (document.visibilityState !== 'visible') {
        resetKeyState();
      }
    };

    window.addEventListener('keydown', keyDownHandler);
    window.addEventListener('keyup', keyUpHandler);
    window.addEventListener('blur', windowBlurHandler);
    document.addEventListener('visibilitychange', visibilityChangeHandler);
  }

  /**
   * Bind pointer, mouse, keyboard-look, and touch-look listeners.
   *
   * @param target - Attached DOM target.
   */
  function bindLookListeners(target: HTMLElement): void {
    mouseLookDetach = bindMouseLook(target, onMouseLook);
    keyboardLookDetach = bindKeyboardLook(target, onKeyboardLook);
    touchLookDetach = bindTouchLook(target, onTouchLook, onTouchActive);
  }

  /**
   * Remove keyboard and lifecycle listeners.
   */
  function removeMovementListeners(): void {
    window.removeEventListener('keydown', keyDownHandler as EventListener);
    window.removeEventListener('keyup', keyUpHandler as EventListener);
    window.removeEventListener('blur', windowBlurHandler as EventListener);
    document.removeEventListener(
      'visibilitychange',
      visibilityChangeHandler as EventListener,
    );

    keyDownHandler = null;
    keyUpHandler = null;
    windowBlurHandler = null;
    visibilityChangeHandler = null;
  }

  /**
   * Remove look listeners.
   */
  function removeLookListeners(): void {
    mouseLookDetach?.();
    mouseLookDetach = null;

    keyboardLookDetach?.();
    keyboardLookDetach = null;

    touchLookDetach?.();
    touchLookDetach = null;
  }

  /**
   * Detach the current attachment if the token matches.
   *
   * @param token - Attachment token captured by the detach closure.
   */
  function detachByToken(token: number): void {
    if (attachedTarget === null || token !== attachmentId) {
      return;
    }

    pointerLockDetach?.();
    pointerLockDetach = null;

    mouseFireDetach?.();
    mouseFireDetach = null;

    keyboardLightToggleDetach?.();
    keyboardLightToggleDetach = null;

    removeLookListeners();
    removeMovementListeners();

    attachedTarget = null;
    resetInputState();
  }

  return {
    attach(target: HTMLElement): InputRouterDetach {
      if (attachedTarget !== null) {
        throw new Error(
          'InputRouter is already attached; detach before re-attaching.',
        );
      }

      attachedTarget = target;
      attachmentId += 1;
      const token = attachmentId;

      pointerLockDetach = bindPointerLock(target);

      mouseFireDetach = bindMouseFire(target, () => {
        pendingFire = true;
      });

      keyboardLightToggleDetach = bindKeyboardLightToggle(window, () => {
        pendingLightToggle = true;
      });

      bindEventListeners();
      bindLookListeners(target);

      return () => detachByToken(token);
    },

    detach(): void {
      detachByToken(attachmentId);
    },

    getSnapshot(): InputSnapshot {
      const pointerLocked =
        typeof document !== 'undefined' &&
        attachedTarget !== null &&
        document.pointerLockElement === attachedTarget;

      const snapshot: InputSnapshot = {
        timestamp: Date.now(),

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
          active: touchActive,
          yawDelta: touchYawDelta,
          pitchDelta: touchPitchDelta,
        },

        pointerLocked,

        // Include both latched edge-triggered actions and currently held keys.
        fire: pendingFire || (keyStates[NEATENSTEIN_FIRE_KEY] ?? false),
        dash: pendingDash || (keyStates[NEATENSTEIN_DASH_KEY] ?? false),
        lightToggle: pendingLightToggle,
      };

      // Consume deltas and one-shot actions after snapshot creation.
      pendingFire = false;
      pendingDash = false;
      pendingLightToggle = false;
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
