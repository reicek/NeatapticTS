/** @jest-environment jsdom */

import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_DASH_KEY,
  NEATENSTEIN_FIRE_KEY,
  NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT,
  NEATENSTEIN_KEY_MAP_LOOK,
  NEATENSTEIN_KEY_MAP_MOVEMENT,
  NEATENSTEIN_LIGHT_TOGGLE_KEY,
  NEATENSTEIN_MOUSE_SENSITIVITY,
  NEATENSTEIN_PRIMARY_MOUSE_BUTTON,
  NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX,
} from './game/constants.ts';
import { createInputRouter } from './input.ts';

/**
 * Build a fake {@link TouchList} backed by an array of touch objects.
 *
 * JSDOM does not implement the native `Touch` constructor, so tests use plain
 * objects that satisfy the properties consumed by the input router.
 *
 * @param touches - Touch-like objects.
 * @returns A `TouchList` shaped object with `length` and `item`.
 */
function createTouchList(
  touches: { identifier: number; clientX: number; clientY: number }[],
): TouchList {
  const list = touches as unknown as Touch[];
  return Object.assign(list, {
    item: (index: number) => (list[index] as Touch | undefined) ?? null,
  }) as unknown as TouchList;
}

/**
 * Dispatch a fake touch event whose `changedTouches` is a test-controlled list.
 *
 * @param target - Element that owns the touch listener.
 * @param type - Event type (`touchstart`, `touchmove`, `touchend`, etc.).
 * @param touches - Fake touch list to attach to the event.
 */
function dispatchFakeTouchEvent(
  target: HTMLElement,
  type: 'touchstart' | 'touchmove' | 'touchend' | 'touchcancel',
  touches: { identifier: number; clientX: number; clientY: number }[],
): void {
  const event = new Event(type, { bubbles: true });
  Object.defineProperty(event, 'changedTouches', {
    value: createTouchList(touches),
    enumerable: true,
  });
  target.dispatchEvent(event as unknown as TouchEvent);
}

/**
 * Create a `MouseEvent` with reliable `movementX`/`movementY`/`button` values.
 *
 * JSDOM ignores `movementX`/`movementY` in the MouseEvent init dictionary, so
 * the values must be patched after construction. Button defaults to the primary
 * (left) button for mousedown/mouseup tests.
 *
 * @param type - Event type (`mousemove`, `mousedown`, `mouseup`).
 * @param movementX - Horizontal pointer movement (mousemove only).
 * @param movementY - Vertical pointer movement (mousemove only).
 * @param button - Mouse button (default 0 for left button).
 * @returns A mouse event whose properties are writable by tests.
 */
function createMouseEvent(
  type: 'mousemove' | 'mousedown' | 'mouseup',
  movementX: number = 0,
  movementY: number = 0,
  button: number = NEATENSTEIN_PRIMARY_MOUSE_BUTTON,
): MouseEvent {
  const event = new MouseEvent(type, { bubbles: true });
  Object.defineProperty(event, 'movementX', { value: movementX });
  Object.defineProperty(event, 'movementY', { value: movementY });
  Object.defineProperty(event, 'button', { value: button });
  return event;
}

describe('Neatenstein host input', () => {
  it('creates a router with attach, detach and getSnapshot methods', () => {
    const router = createInputRouter();
    expect({
      hasAttach: typeof router.attach === 'function',
      hasDetach: typeof router.detach === 'function',
      hasGetSnapshot: typeof router.getSnapshot === 'function',
    }).toEqual({
      hasAttach: true,
      hasDetach: true,
      hasGetSnapshot: true,
    });
  });

  it('reflects movement key state in the snapshot', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        code: NEATENSTEIN_KEY_MAP_MOVEMENT.forward,
      }),
    );
    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_MOVEMENT.left }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.movement).toEqual({
      forward: true,
      backward: false,
      left: true,
      right: false,
    });
  });

  it('accumulates keyboard arrow look deltas in the snapshot', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_LOOK.left }),
    );
    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_LOOK.up }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    const step = NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT;
    expect(snapshot.look).toEqual({
      yawDelta: -step,
      pitchDelta: -step,
    });
  });

  it('accumulates pointer-lock mouse look deltas in the snapshot', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    Object.defineProperty(document, 'pointerLockElement', {
      value: target,
      configurable: true,
    });
    document.dispatchEvent(createMouseEvent('mousemove', 100, 50));

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.look).toEqual({
      yawDelta: 100 * NEATENSTEIN_MOUSE_SENSITIVITY,
      pitchDelta: 50 * NEATENSTEIN_MOUSE_SENSITIVITY,
    });
  });

  it('accumulates touch drag look deltas in the snapshot', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);
    const drag = NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX + 3;

    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchmove', [
      { identifier: 1, clientX: drag, clientY: 0 },
    ]);

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.look).toEqual({
      yawDelta: drag * NEATENSTEIN_MOUSE_SENSITIVITY,
      pitchDelta: 0,
    });
  });

  it('consumes look deltas after getSnapshot is called', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_LOOK.right }),
    );
    const first = router.getSnapshot();
    const second = router.getSnapshot();
    router.detach();

    expect({
      first: first.look.yawDelta,
      second: second.look.yawDelta,
    }).toEqual({
      first: NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT,
      second: 0,
    });
  });

  it('detaches listeners and resets transient state', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        code: NEATENSTEIN_KEY_MAP_MOVEMENT.forward,
      }),
    );
    router.detach();

    // After detach, a new snapshot should be clean even without an explicit
    // keyup event.
    expect(router.getSnapshot().movement).toEqual({
      forward: false,
      backward: false,
      left: false,
      right: false,
    });
  });

  it('reflects the dash key state in the snapshot', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_DASH_KEY }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.dash).toBe(true);
  });

  it('reflects the keyboard fire fallback in the snapshot', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_FIRE_KEY }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.fire).toBe(true);
  });

  it('consumes a pending mouse fire event in getSnapshot', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    target.dispatchEvent(
      createMouseEvent('mousedown', 0, 0, NEATENSTEIN_PRIMARY_MOUSE_BUTTON),
    );
    const first = router.getSnapshot();
    const second = router.getSnapshot();
    router.detach();

    expect({ first: first.fire, second: second.fire }).toEqual({
      first: true,
      second: false,
    });
  });

  it('throws when attaching while already attached', () => {
    const router = createInputRouter();
    const first = document.createElement('div');
    const second = document.createElement('div');
    router.attach(first);

    expect(() => router.attach(second)).toThrow(
      'InputRouter is already attached; detach before re-attaching.',
    );

    router.detach();
  });

  it('releases movement keys on keyup', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        code: NEATENSTEIN_KEY_MAP_MOVEMENT.forward,
      }),
    );
    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_MOVEMENT.left }),
    );
    window.dispatchEvent(
      new KeyboardEvent('keyup', {
        code: NEATENSTEIN_KEY_MAP_MOVEMENT.forward,
      }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.movement).toEqual({
      forward: false,
      backward: false,
      left: true,
      right: false,
    });
  });

  it('does not lose keyboard look delta when the look key is released', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_LOOK.left }),
    );
    target.dispatchEvent(
      new KeyboardEvent('keyup', { code: NEATENSTEIN_KEY_MAP_LOOK.left }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.look.yawDelta).toBe(
      -NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT,
    );
  });

  it('consumes fire and dash latches on the first getSnapshot after press', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_FIRE_KEY }),
    );
    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_DASH_KEY }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    expect({ fire: snapshot.fire, dash: snapshot.dash }).toEqual({
      fire: true,
      dash: true,
    });
  });

  it('reflects the keyboard light toggle in the snapshot', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_LIGHT_TOGGLE_KEY }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.lightToggle).toBe(true);
  });

  it('consumes the light toggle latch and allows re-toggling', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_LIGHT_TOGGLE_KEY }),
    );
    const first = router.getSnapshot();

    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_LIGHT_TOGGLE_KEY }),
    );
    const second = router.getSnapshot();
    router.detach();

    expect({ first: first.lightToggle, second: second.lightToggle }).toEqual({
      first: true,
      second: true,
    });
  });

  it('releases fire and dash keys on keyup', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_FIRE_KEY }),
    );
    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_DASH_KEY }),
    );

    // Latches are one-shot: consume them on the first frame after press.
    router.getSnapshot();

    window.dispatchEvent(
      new KeyboardEvent('keyup', { code: NEATENSTEIN_FIRE_KEY }),
    );
    window.dispatchEvent(
      new KeyboardEvent('keyup', { code: NEATENSTEIN_DASH_KEY }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    expect({ fire: snapshot.fire, dash: snapshot.dash }).toEqual({
      fire: false,
      dash: false,
    });
  });

  it('combines mouse, keyboard, and touch look deltas', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    Object.defineProperty(document, 'pointerLockElement', {
      value: target,
      configurable: true,
    });
    document.dispatchEvent(createMouseEvent('mousemove', 10, 5));
    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_LOOK.left }),
    );

    const drag = NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX + 2;
    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchmove', [
      { identifier: 1, clientX: drag, clientY: 0 },
    ]);

    const snapshot = router.getSnapshot();
    router.detach();

    const step = NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT;
    expect(snapshot.look).toEqual({
      yawDelta:
        10 * NEATENSTEIN_MOUSE_SENSITIVITY -
        step +
        drag * NEATENSTEIN_MOUSE_SENSITIVITY,
      pitchDelta: 5 * NEATENSTEIN_MOUSE_SENSITIVITY,
    });
  });

  it('reports pointerLocked as false when the target does not own lock', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    Object.defineProperty(document, 'pointerLockElement', {
      value: null,
      configurable: true,
    });

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.pointerLocked).toBe(false);
  });

  it('reports touch.active while a touch is being tracked', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    const drag = NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX + 1;
    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchmove', [
      { identifier: 1, clientX: drag, clientY: 0 },
    ]);

    const snapshot = router.getSnapshot();
    router.detach();

    expect({
      active: snapshot.touch.active,
      yawDelta: snapshot.touch.yawDelta,
    }).toEqual({
      active: true,
      yawDelta: drag * NEATENSTEIN_MOUSE_SENSITIVITY,
    });
  });

  it('resets touch.active when the tracked touch ends', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchend', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.touch.active).toBe(false);
  });

  it('includes a finite timestamp in the snapshot', () => {
    const router = createInputRouter();
    router.attach(document.createElement('div'));

    const before = Date.now();
    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.timestamp).toBeGreaterThanOrEqual(before);
  });

  it('the detach closure returned by attach removes all listeners and resets state', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    const detach = router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        code: NEATENSTEIN_KEY_MAP_MOVEMENT.forward,
      }),
    );
    detach();
    window.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_MOVEMENT.left }),
    );

    expect(router.getSnapshot().movement).toEqual({
      forward: false,
      backward: false,
      left: false,
      right: false,
    });
  });

  it('resets movement keys when the document becomes hidden', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        code: NEATENSTEIN_KEY_MAP_MOVEMENT.forward,
      }),
    );

    Object.defineProperty(document, 'visibilityState', {
      value: 'hidden',
      configurable: true,
    });
    document.dispatchEvent(new Event('visibilitychange'));

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.movement.forward).toBe(false);
  });

  it('ignores a detach closure whose token no longer matches the attachment', () => {
    const router = createInputRouter();
    const firstTarget = document.createElement('div');
    const firstDetach = router.attach(firstTarget);
    router.detach();

    const secondTarget = document.createElement('div');
    router.attach(secondTarget);

    // Calling the first detach closure should be a no-op because the attachment
    // id has changed.
    expect(() => firstDetach()).not.toThrow();
    router.detach();
  });

  it('does not prevent default for non-gameplay keys', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    const event = new KeyboardEvent('keydown', {
      code: 'KeyX',
      cancelable: true,
    });
    window.dispatchEvent(event);
    router.detach();

    expect(event.defaultPrevented).toBe(false);
  });

  it('leaves key state unchanged when visibility becomes visible', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        code: NEATENSTEIN_KEY_MAP_MOVEMENT.forward,
      }),
    );

    Object.defineProperty(document, 'visibilityState', {
      value: 'visible',
      configurable: true,
    });
    document.dispatchEvent(new Event('visibilitychange'));

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.movement.forward).toBe(true);
  });

  it('resets movement keys when the window loses focus', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        code: NEATENSTEIN_KEY_MAP_MOVEMENT.forward,
      }),
    );
    window.dispatchEvent(new Event('blur'));

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.movement.forward).toBe(false);
  });
});
