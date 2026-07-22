/** @jest-environment jsdom */

import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_DASH_KEY,
  NEATENSTEIN_FIRE_KEY,
  NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT,
  NEATENSTEIN_MOUSE_SENSITIVITY,
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
  button: number = 0,
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

    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyW' }));
    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyA' }));

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

    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowLeft' }));
    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowUp' }));

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

    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowRight' }));
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

    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyW' }));
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

    target.dispatchEvent(
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

    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_FIRE_KEY }),
    );

    const snapshot = router.getSnapshot();
    router.detach();

    expect(snapshot.fire).toBe(true);
  });

  it('reflects left mouse button state as fire in the snapshot', () => {
    const router = createInputRouter();
    const target = document.createElement('div');
    router.attach(target);

    target.dispatchEvent(createMouseEvent('mousedown', 0, 0, 0));
    const pressed = router.getSnapshot();
    document.dispatchEvent(createMouseEvent('mouseup', 0, 0, 0));
    const released = router.getSnapshot();
    router.detach();

    expect({ pressed: pressed.fire, released: released.fire }).toEqual({
      pressed: true,
      released: false,
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
});
