/** @jest-environment jsdom */

import { describe, expect, it, jest } from '@jest/globals';
import type { InputSnapshot } from '../input.ts';
import { NEATENSTEIN_INPUT_MESSAGE_TYPE } from '../../constants';
import {
  NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT,
  NEATENSTEIN_KEY_MAP_LOOK,
  NEATENSTEIN_MOUSE_SENSITIVITY,
  NEATENSTEIN_POINTER_LOCK_OPTIONS,
  NEATENSTEIN_PRIMARY_MOUSE_BUTTON,
  NEATENSTEIN_SECONDARY_MOUSE_BUTTON,
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

const TOUCH_ID_PRIMARY = 1;
const TOUCH_ID_OTHER = 2;
const TOUCH_ORIGIN_X = 0;
const TOUCH_ORIGIN_Y = 0;
const TOUCH_FAR_X = 100;

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

    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_LOOK.left }),
    );
    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_LOOK.right }),
    );
    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_LOOK.up }),
    );
    target.dispatchEvent(
      new KeyboardEvent('keydown', { code: NEATENSTEIN_KEY_MAP_LOOK.down }),
    );
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

  it('emits on left mouse down for fire', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const detach = bindMouseFire(target, callback);

    target.dispatchEvent(
      new MouseEvent('mousedown', {
        button: NEATENSTEIN_PRIMARY_MOUSE_BUTTON,
        bubbles: true,
      }),
    );
    target.dispatchEvent(
      new MouseEvent('mousedown', {
        button: NEATENSTEIN_SECONDARY_MOUSE_BUTTON,
        bubbles: true,
      }),
    );
    document.dispatchEvent(
      new MouseEvent('mouseup', {
        button: NEATENSTEIN_PRIMARY_MOUSE_BUTTON,
        bubbles: true,
      }),
    );
    detach();

    expect(callback).toHaveBeenCalledTimes(1);
  });

  it('ignores non-left mouse button for fire', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const detach = bindMouseFire(target, callback);

    target.dispatchEvent(
      new MouseEvent('mousedown', {
        button: NEATENSTEIN_SECONDARY_MOUSE_BUTTON,
        bubbles: true,
      }),
    );
    document.dispatchEvent(
      new MouseEvent('mouseup', {
        button: NEATENSTEIN_SECONDARY_MOUSE_BUTTON,
        bubbles: true,
      }),
    );
    detach();

    expect(callback).not.toHaveBeenCalled();
  });

  it('does not emit fire on mouse up', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const detach = bindMouseFire(target, callback);

    target.dispatchEvent(
      new MouseEvent('mousedown', {
        button: NEATENSTEIN_PRIMARY_MOUSE_BUTTON,
        bubbles: true,
      }),
    );
    document.dispatchEvent(
      new MouseEvent('mouseup', {
        button: NEATENSTEIN_PRIMARY_MOUSE_BUTTON,
        bubbles: true,
      }),
    );
    document.dispatchEvent(
      new MouseEvent('mouseup', {
        button: NEATENSTEIN_SECONDARY_MOUSE_BUTTON,
        bubbles: true,
      }),
    );
    detach();

    expect(callback).toHaveBeenCalledTimes(1);
  });

  it('requests pointer lock with unadjustedMovement options', () => {
    const canvas = document.createElement('canvas');
    const requestPointerLock = jest
      .fn<(options?: PointerLockOptions) => Promise<void>>()
      .mockResolvedValue(undefined);
    Object.defineProperty(canvas, 'requestPointerLock', {
      value: requestPointerLock,
      configurable: true,
    });

    const detach = bindPointerLock(canvas);
    canvas.click();
    detach();

    expect(requestPointerLock).toHaveBeenCalledWith(
      NEATENSTEIN_POINTER_LOCK_OPTIONS,
    );
  });

  it('retries pointer lock without options when options are unsupported', async () => {
    const canvas = document.createElement('canvas');
    const requestPointerLock = jest
      .fn<(options?: PointerLockOptions) => Promise<void>>()
      .mockRejectedValueOnce(new TypeError('Unsupported'))
      .mockResolvedValue(undefined);
    Object.defineProperty(canvas, 'requestPointerLock', {
      value: requestPointerLock,
      configurable: true,
    });

    const detach = bindPointerLock(canvas);
    canvas.click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    detach();

    expect(requestPointerLock.mock.calls).toEqual([
      [NEATENSTEIN_POINTER_LOCK_OPTIONS],
      [],
    ]);
  });

  it('returns undefined when findTouch cannot match the identifier', () => {
    const touches = createTouchList([
      { identifier: 1, clientX: 0, clientY: 0 },
      { identifier: 2, clientX: 10, clientY: 10 },
    ]);
    expect(findTouch(touches, 99)).toBeUndefined();
  });

  it('emits a pitch delta once a vertical touch drag crosses the threshold', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const detach = bindTouchLook(target, callback);
    const drag = NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX + 5;

    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchmove', [
      { identifier: 1, clientX: 0, clientY: drag },
    ]);
    detach();

    expect(callback).toHaveBeenCalledWith({
      yawDelta: 0,
      pitchDelta: drag * NEATENSTEIN_MOUSE_SENSITIVITY,
    });
  });

  it('reports touch active state through callback on start and end', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, callback, activeCallback);

    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchmove', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchend', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    detach();

    expect(activeCallback.mock.calls).toEqual([[true], [false]]);
  });

  it('releases active touch on touchcancel', () => {
    const target = document.createElement('div');
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, () => undefined, activeCallback);

    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchcancel', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    detach();

    expect(activeCallback.mock.calls).toEqual([[true], [false]]);
  });

  it('returns silently when the target lacks pointer lock support', () => {
    const unsupported = document.createElement('div');

    expect(() => {
      const detach = bindPointerLock(unsupported);
      unsupported.click();
      detach();
    }).not.toThrow();
  });

  it('gives up when both pointer lock request variants fail', async () => {
    const canvas = document.createElement('canvas');
    const requestPointerLock = jest
      .fn<(options?: PointerLockOptions) => Promise<void>>()
      .mockRejectedValueOnce(new TypeError('Unsupported'))
      .mockRejectedValueOnce(new Error('Still fails'));
    Object.defineProperty(canvas, 'requestPointerLock', {
      value: requestPointerLock,
      configurable: true,
    });

    const detach = bindPointerLock(canvas);
    canvas.click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    detach();

    expect(requestPointerLock.mock.calls).toEqual([
      [NEATENSTEIN_POINTER_LOCK_OPTIONS],
      [],
    ]);
  });

  it('does not prevent default for unmapped keyboard look keys', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const detach = bindKeyboardLook(target, callback);

    const event = new KeyboardEvent('keydown', { code: 'KeyA' });
    const preventDefault = jest.spyOn(event, 'preventDefault');
    target.dispatchEvent(event);
    detach();

    expect({
      preventDefaultCalls: preventDefault.mock.calls.length,
      callbackCalls: callback.mock.calls.length,
    }).toEqual({ preventDefaultCalls: 0, callbackCalls: 0 });
  });

  it('ignores a second touch while one is already active', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, callback, activeCallback);
    const drag = NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX + 5;

    dispatchFakeTouchEvent(target, 'touchstart', [
      {
        identifier: TOUCH_ID_PRIMARY,
        clientX: TOUCH_ORIGIN_X,
        clientY: TOUCH_ORIGIN_Y,
      },
    ]);
    dispatchFakeTouchEvent(target, 'touchstart', [
      {
        identifier: TOUCH_ID_OTHER,
        clientX: TOUCH_ORIGIN_X,
        clientY: TOUCH_ORIGIN_Y,
      },
    ]);
    dispatchFakeTouchEvent(target, 'touchmove', [
      { identifier: TOUCH_ID_OTHER, clientX: drag, clientY: TOUCH_ORIGIN_Y },
    ]);
    detach();

    expect({
      activeCalls: activeCallback.mock.calls.length,
      callbackCalls: callback.mock.calls.length,
    }).toEqual({ activeCalls: 1, callbackCalls: 0 });
  });

  it('removes touch listeners on detach so callbacks stop firing', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, callback, activeCallback);

    dispatchFakeTouchEvent(target, 'touchstart', [
      { identifier: 1, clientX: 0, clientY: 0 },
    ]);
    detach();
    callback.mockClear();
    activeCallback.mockClear();

    dispatchFakeTouchEvent(target, 'touchmove', [
      { identifier: 1, clientX: 100, clientY: 0 },
    ]);
    dispatchFakeTouchEvent(target, 'touchend', [
      { identifier: 1, clientX: 100, clientY: 0 },
    ]);

    expect({
      callbackCalls: callback.mock.calls.length,
      activeCallbackCalls: activeCallback.mock.calls.length,
    }).toEqual({ callbackCalls: 0, activeCallbackCalls: 0 });
  });

  it('does not retry pointer lock when the rejection is unrelated to options', async () => {
    const canvas = document.createElement('canvas');
    const requestPointerLock = jest
      .fn<(options?: PointerLockOptions) => Promise<void>>()
      .mockRejectedValueOnce(new Error('Unrelated failure'));
    Object.defineProperty(canvas, 'requestPointerLock', {
      value: requestPointerLock,
      configurable: true,
    });

    const detach = bindPointerLock(canvas);
    canvas.click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    detach();

    expect(requestPointerLock.mock.calls).toEqual([
      [NEATENSTEIN_POINTER_LOCK_OPTIONS],
    ]);
  });

  it('exits pointer lock on detach when the canvas owns it', () => {
    const canvas = document.createElement('canvas');
    const requestPointerLock = jest.fn(() => Promise.resolve());
    Object.defineProperty(canvas, 'requestPointerLock', {
      value: requestPointerLock,
      configurable: true,
    });
    const exitPointerLock = jest.fn();
    Object.defineProperty(document, 'exitPointerLock', {
      value: exitPointerLock,
      configurable: true,
    });
    Object.defineProperty(document, 'pointerLockElement', {
      value: canvas,
      configurable: true,
    });

    const detach = bindPointerLock(canvas);
    detach();

    expect(exitPointerLock).toHaveBeenCalledTimes(1);
  });

  it('ignores touch start when changedTouches is empty', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, callback, activeCallback);

    dispatchFakeTouchEvent(target, 'touchstart', []);
    detach();

    expect({
      callbackCalls: callback.mock.calls.length,
      activeCallbackCalls: activeCallback.mock.calls.length,
    }).toEqual({ callbackCalls: 0, activeCallbackCalls: 0 });
  });

  it('ignores touch move when no touch is active', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, callback, activeCallback);

    dispatchFakeTouchEvent(target, 'touchmove', [
      {
        identifier: TOUCH_ID_PRIMARY,
        clientX: TOUCH_FAR_X,
        clientY: TOUCH_ORIGIN_Y,
      },
    ]);
    detach();

    expect({
      callbackCalls: callback.mock.calls.length,
      activeCallbackCalls: activeCallback.mock.calls.length,
    }).toEqual({ callbackCalls: 0, activeCallbackCalls: 0 });
  });

  it('ignores touch end when no touch is active', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, callback, activeCallback);

    dispatchFakeTouchEvent(target, 'touchend', [
      {
        identifier: TOUCH_ID_PRIMARY,
        clientX: TOUCH_ORIGIN_X,
        clientY: TOUCH_ORIGIN_Y,
      },
    ]);
    detach();

    expect({
      callbackCalls: callback.mock.calls.length,
      activeCallbackCalls: activeCallback.mock.calls.length,
    }).toEqual({ callbackCalls: 0, activeCallbackCalls: 0 });
  });

  it('ignores touch cancel when no touch is active', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, callback, activeCallback);

    dispatchFakeTouchEvent(target, 'touchcancel', [
      {
        identifier: TOUCH_ID_PRIMARY,
        clientX: TOUCH_ORIGIN_X,
        clientY: TOUCH_ORIGIN_Y,
      },
    ]);
    detach();

    expect({
      callbackCalls: callback.mock.calls.length,
      activeCallbackCalls: activeCallback.mock.calls.length,
    }).toEqual({ callbackCalls: 0, activeCallbackCalls: 0 });
  });

  it('ignores touch move when the changed touch id does not match the active touch', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, callback, activeCallback);
    const drag = NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX + 5;

    dispatchFakeTouchEvent(target, 'touchstart', [
      {
        identifier: TOUCH_ID_PRIMARY,
        clientX: TOUCH_ORIGIN_X,
        clientY: TOUCH_ORIGIN_Y,
      },
    ]);
    dispatchFakeTouchEvent(target, 'touchmove', [
      {
        identifier: TOUCH_ID_OTHER,
        clientX: drag,
        clientY: TOUCH_ORIGIN_Y,
      },
    ]);
    detach();

    expect({
      callbackCalls: callback.mock.calls.length,
      activeCallbackCalls: activeCallback.mock.calls.length,
    }).toEqual({ callbackCalls: 0, activeCallbackCalls: 1 });
  });

  it('ignores touch end when the changed touch id does not match the active touch', () => {
    const target = document.createElement('div');
    const callback = jest.fn();
    const activeCallback = jest.fn();
    const detach = bindTouchLook(target, callback, activeCallback);

    dispatchFakeTouchEvent(target, 'touchstart', [
      {
        identifier: TOUCH_ID_PRIMARY,
        clientX: TOUCH_ORIGIN_X,
        clientY: TOUCH_ORIGIN_Y,
      },
    ]);
    dispatchFakeTouchEvent(target, 'touchend', [
      {
        identifier: TOUCH_ID_OTHER,
        clientX: TOUCH_ORIGIN_X,
        clientY: TOUCH_ORIGIN_Y,
      },
    ]);
    detach();

    expect({
      callbackCalls: callback.mock.calls.length,
      activeCallbackCalls: activeCallback.mock.calls.length,
    }).toEqual({ callbackCalls: 0, activeCallbackCalls: 1 });
  });
});
