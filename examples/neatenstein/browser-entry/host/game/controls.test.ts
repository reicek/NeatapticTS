/** @jest-environment jsdom */

import { describe, expect, it, jest } from '@jest/globals';
import type { InputSnapshot } from '../input.ts';
import { NEATENSTEIN_INPUT_MESSAGE_TYPE } from '../../constants';
import {
  NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT,
  NEATENSTEIN_MOUSE_SENSITIVITY,
  NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX,
} from './constants.ts';
import {
  bindKeyboardLook,
  bindMouseFire,
  bindMouseLook,
  bindPointerLock,
  bindTouchLook,
  findTouch,
  forwardWorkerInput,
} from './controls.ts';

/**
 * Build a fake {@link TouchList} backed by an array of touch objects.
 *
 * JSDOM does not implement the native `Touch` constructor, so tests use plain
 * objects that satisfy the properties consumed by the controls layer.
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
  Object.defineProperty(event, 'touches', {
    value: createTouchList(touches),
    enumerable: true,
  });
  target.dispatchEvent(event as unknown as TouchEvent);
}

/**
 * Create a `MouseEvent` with reliable `movementX`/`movementY` values.
 *
 * JSDOM ignores `movementX`/`movementY` in the MouseEvent init dictionary, so
 * the values must be patched after construction.
 *
 * @param type - Event type (usually `mousemove`).
 * @param movementX - Horizontal pointer movement.
 * @param movementY - Vertical pointer movement.
 * @returns A mouse event whose movement properties are writable by tests.
 */
function createMouseEvent(
  type: 'mousemove',
  movementX: number,
  movementY: number,
): MouseEvent {
  const event = new MouseEvent(type, { bubbles: true });
  Object.defineProperty(event, 'movementX', { value: movementX });
  Object.defineProperty(event, 'movementY', { value: movementY });
  return event;
}

describe('Neatenstein game controls', () => {
  it('finds a touch by identifier', () => {
    const touches = createTouchList([
      { identifier: 1, clientX: 0, clientY: 0 },
      { identifier: 2, clientX: 10, clientY: 10 },
    ]);
    expect(findTouch(touches, 2)?.clientX).toBe(10);
  });

  it('requests pointer lock on click', () => {
    const canvas = document.createElement('canvas');
    const requestPointerLock = jest.fn(() => Promise.resolve());
    Object.defineProperty(canvas, 'requestPointerLock', {
      value: requestPointerLock,
      configurable: true,
    });

    const detach = bindPointerLock(canvas);
    canvas.click();
    detach();

    expect(requestPointerLock).toHaveBeenCalledTimes(1);
  });

  it('emits pointer-lock mouse deltas only while locked', () => {
    const canvas = document.createElement('canvas');
    const callback = jest.fn();
    const detach = bindMouseLook(canvas, callback);

    Object.defineProperty(document, 'pointerLockElement', {
      value: canvas,
      configurable: true,
    });
    document.dispatchEvent(createMouseEvent('mousemove', 100, 50));
    detach();

    expect(callback).toHaveBeenCalledWith({
      yawDelta: 100 * NEATENSTEIN_MOUSE_SENSITIVITY,
      pitchDelta: 50 * NEATENSTEIN_MOUSE_SENSITIVITY,
    });
  });

  it('ignores mouse movement when the pointer is not locked', () => {
    const canvas = document.createElement('canvas');
    const callback = jest.fn();
    const detach = bindMouseLook(canvas, callback);

    Object.defineProperty(document, 'pointerLockElement', {
      value: null,
      configurable: true,
    });
    document.dispatchEvent(createMouseEvent('mousemove', 100, 50));
    detach();

    expect(callback).not.toHaveBeenCalled();
  });

  it('emits yaw and pitch deltas for arrow-key look', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const detach = bindKeyboardLook(target, callback);

    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowLeft' }));
    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowRight' }));
    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowUp' }));
    target.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowDown' }));
    detach();

    const step = NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT;
    expect(callback.mock.calls).toEqual([
      [{ yawDelta: -step, pitchDelta: 0 }],
      [{ yawDelta: step, pitchDelta: 0 }],
      [{ yawDelta: 0, pitchDelta: -step }],
      [{ yawDelta: 0, pitchDelta: step }],
    ]);
  });

  it('emits a yaw delta once a horizontal touch drag crosses the threshold', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const detach = bindTouchLook(target, callback);
    const drag = NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX + 5;

    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchmove', [
      { identifier: 1, clientX: drag, clientY: 0 },
    ]);
    detach();

    expect(callback).toHaveBeenCalledWith({
      yawDelta: drag * NEATENSTEIN_MOUSE_SENSITIVITY,
      pitchDelta: 0,
    });
  });

  it('does not emit a touch delta below the drag threshold', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const detach = bindTouchLook(target, callback);

    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchmove', [
      { identifier: 1, clientX: 1, clientY: 0 },
    ]);
    detach();

    expect(callback).not.toHaveBeenCalled();
  });

  it('forwards a full input snapshot to the worker', () => {
    const worker = { postMessage: jest.fn() } as unknown as Worker;
    const snapshot = {
      movement: {
        forward: true,
        backward: false,
        left: false,
        right: true,
      },
      look: { yawDelta: 0.1, pitchDelta: 0.2 },
      fire: true,
      dash: false,
    } as unknown as InputSnapshot;

    forwardWorkerInput(worker, snapshot);

    expect(worker.postMessage).toHaveBeenCalledWith({
      type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
      input: {
        movement: {
          forward: true,
          backward: false,
          left: false,
          right: true,
        },
        yawDelta: 0.1,
        pitchDelta: 0.2,
        fire: true,
        dash: false,
      },
    });
  });

  it('emits true on left mouse down and false on mouse up for fire', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const detach = bindMouseFire(target, callback);

    target.dispatchEvent(
      new MouseEvent('mousedown', { button: 0, bubbles: true }),
    );
    target.dispatchEvent(
      new MouseEvent('mousedown', { button: 2, bubbles: true }),
    );
    document.dispatchEvent(
      new MouseEvent('mouseup', { button: 0, bubbles: true }),
    );
    detach();

    expect(callback.mock.calls).toEqual([[true], [false]]);
  });
});
