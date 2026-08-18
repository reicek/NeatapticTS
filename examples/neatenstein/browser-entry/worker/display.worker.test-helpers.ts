/**
 * Shared test helpers for the Neatenstein display worker test suite.
 *
 * Provides a mock worker global, typed message senders, mock canvas/context
 * factories, and a shared `beforeEach` setup. Focused worker test files import
 * these fixtures instead of duplicating boilerplate.
 *
 * @module
 */

import { beforeEach, jest } from '@jest/globals';
import {
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
} from '../constants';
import type { ImpactSpot } from '../host/game/types';

// ---------------------------------------------------------------------------
// Dynamic module loader
// ---------------------------------------------------------------------------

/**
 * Dynamically import a Neatenstein worker module for testing.
 *
 * Resolves `./browser-entry/` paths relative to the neatenstein root and
 * all other paths relative to this helper module, matching the layout of
 * the worker test files that import it.
 *
 * @param path - Module path to import.
 * @returns The imported module as `unknown`.
 */
export const loadModule = async (path: string): Promise<unknown> => {
  // Paths starting with './browser-entry/' are relative to the
  // neatenstein root (two levels up from this module's directory in
  // browser-entry/worker/).  All other paths are relative to this
  // module's directory, matching the original behaviour used by the
  // worker test files that live alongside this helper.
  if (path.startsWith('./browser-entry/')) {
    const resolved = `../../${path.slice(2)}`;
    return import(resolved);
  }
  return import(path);
};

// ---------------------------------------------------------------------------
// Mock worker global
// ---------------------------------------------------------------------------

/**
 * Minimal worker global (`self`) used in Node-based worker tests.
 */
export interface MockWorkerGlobal {
  postMessage: jest.Mock;
  requestAnimationFrame: jest.Mock;
  onmessage: ((event: MessageEvent) => void) | null;
}

/**
 * Generic shape of a postMessage payload captured by `postMessage` mocks.
 */
export type WorkerPost = {
  type?: string;
  frame?: Record<string, unknown>;
  [key: string]: unknown;
};

/**
 * Install a mock `self` object on the global scope and return it.
 *
 * The returned object has mocked `postMessage` and `requestAnimationFrame`
 * methods and an `onmessage` handler slot that tests can drive manually.
 *
 * @returns The installed mock worker global.
 */
export function installMockWorkerGlobal(): MockWorkerGlobal {
  const self: MockWorkerGlobal = {
    postMessage: jest.fn(),
    requestAnimationFrame: jest.fn(() => 0),
    onmessage: null,
  };
  (globalThis as unknown as Record<string, unknown>).self = self;
  return self;
}

/**
 * Shared mock worker global used by the message sender helpers.
 *
 * Created once at module load; call {@link installMockWorkerGlobal} if a
 * test needs a fresh instance.
 */
export const workerSelf = installMockWorkerGlobal();

// ---------------------------------------------------------------------------
// Message senders
// ---------------------------------------------------------------------------

/**
 * Deliver an `init` message to the worker's `onmessage` handler.
 *
 * @param tier - Which rendering tier to request (`worker`, `cpu`, or `gpu`).
 * @param canvas - Optional mock canvas to attach to the init payload.
 */
export function sendInitMessage(
  tier: 'worker' | 'cpu' | 'gpu',
  canvas?: { getContext: jest.Mock; width: number; height: number },
) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: 'init',
        tier,
        version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
        mapSeed: 42,
        canvas,
      },
    } as unknown as MessageEvent);
  }
}

/**
 * Deliver a `simState` message with a minimal deterministic game state.
 *
 * @param cameraYaw - Camera yaw in radians; defaults to 0.25.
 * @param overrides - Optional camera position and human-mode overrides.
 */
export function sendSimStateMessage(
  cameraYaw = 0.25,
  overrides: {
    cameraX?: number;
    cameraY?: number;
    humanMode?: 'auto' | 'human';
  } = {},
) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: 'simState',
        state: {
          canvasWidth: 640,
          canvasHeight: 360,
          simTick: 1,
          cameraX: overrides.cameraX ?? 12.5,
          cameraY: overrides.cameraY ?? 12.5,
          cameraYaw,
          mapSeed: 42,
          humanMode: overrides.humanMode,
        },
      },
    } as unknown as MessageEvent);
  }
}

/**
 * Deliver an input message containing a yaw delta.
 *
 * @param yawDelta - Change in camera yaw.
 */
export function sendInputMessage(yawDelta: number) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input: { yawDelta },
      },
    } as unknown as MessageEvent);
  }
}

/**
 * Deliver an input message containing movement flags.
 *
 * @param movement - Boolean movement flags.
 */
export function sendMovementInputMessage(movement: {
  forward?: boolean;
  backward?: boolean;
  left?: boolean;
  right?: boolean;
}) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input: { movement },
      },
    } as unknown as MessageEvent);
  }
}

/**
 * Deliver an input message containing a fire action.
 *
 * @param fire - Whether the fire button is pressed.
 */
export function sendActionInputMessage(fire: boolean) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input: { fire },
      },
    } as unknown as MessageEvent);
  }
}

/**
 * Deliver an arbitrary raw message to the worker's `onmessage` handler.
 *
 * @param data - Message payload.
 */
export function sendRawMessage(data: unknown) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({ data } as unknown as MessageEvent);
  }
}

/**
 * Deliver a resize message to the worker's `onmessage` handler.
 *
 * @param width - New canvas width in pixels.
 * @param height - New canvas height in pixels.
 */
export function sendResizeMessage(width: number, height: number) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: { type: 'resize', width, height },
    } as unknown as MessageEvent);
  }
}

// ---------------------------------------------------------------------------
// Mock canvas / context
// ---------------------------------------------------------------------------

/**
 * Build a mock canvas object that returns a pre-built 2D context.
 *
 * @param context - The mock context to return from `getContext`.
 * @returns A canvas stand-in with fixed 640×360 dimensions.
 */
export function createMockCanvas(context: OffscreenCanvasRenderingContext2D): {
  getContext: jest.Mock;
  width: number;
  height: number;
} {
  return {
    getContext: jest.fn(() => context),
    width: 640,
    height: 360,
  };
}

/**
 * Build a mock `OffscreenCanvasRenderingContext2D` with instrumented setters.
 *
 * Tracks calls to common 2D properties and methods so tests can assert on
 * draw commands, fill/stroke styles, and image operations.
 *
 * @returns The mock context plus captured setters and image data helpers.
 */
export function createMockContext(): {
  context: OffscreenCanvasRenderingContext2D & {
    canvas: { width: number; height: number };
  };
  setters: Record<string, jest.Mock>;
  imageData: { data: Uint8ClampedArray; width: number; height: number };
  getImageData: jest.Mock;
  putImageData: jest.Mock;
} {
  const canvas = { width: 640, height: 360 };
  const target: Record<string, unknown> = { canvas };
  const setters: Record<string, jest.Mock> = {};

  for (const property of [
    'fillStyle',
    'globalAlpha',
    'globalCompositeOperation',
    'shadowColor',
    'shadowBlur',
    'strokeStyle',
    'lineWidth',
  ]) {
    const setter = jest.fn();
    setters[property] = setter;
    Object.defineProperty(target, property, {
      get: () => {
        const calls = setter.mock.calls;
        return calls.length > 0 ? calls[calls.length - 1][0] : undefined;
      },
      set: (value: unknown) => {
        setter(value);
      },
      configurable: true,
      enumerable: true,
    });
  }

  const gradient = { addColorStop: jest.fn() };

  for (const method of [
    'fillRect',
    'beginPath',
    'arc',
    'moveTo',
    'lineTo',
    'stroke',
    'fill',
    'save',
    'restore',
    'translate',
    'closePath',
    'ellipse',
  ]) {
    target[method] = jest.fn();
  }

  target.createRadialGradient = jest.fn(() => gradient);
  target.createLinearGradient = jest.fn(() => gradient);

  const imageData = {
    data: new Uint8ClampedArray(canvas.width * canvas.height * 4),
    width: canvas.width,
    height: canvas.height,
  };
  const getImageData = jest.fn(
    (_x: number, _y: number, w: number, h: number) => ({
      data: imageData.data,
      width: w,
      height: h,
    }),
  );
  const putImageData = jest.fn();
  target.getImageData = getImageData;
  target.putImageData = putImageData;

  // Prime the composite operation so functions that read and restore it
  // receive a deterministic string instead of `undefined`.
  (
    target as unknown as { globalCompositeOperation: string }
  ).globalCompositeOperation = 'source-over';

  return {
    context: target as unknown as OffscreenCanvasRenderingContext2D & {
      canvas: { width: number; height: number };
    },
    setters,
    imageData,
    getImageData: getImageData as unknown as jest.Mock,
    putImageData: putImageData as unknown as jest.Mock,
  };
}

// ---------------------------------------------------------------------------
// Post-message search utility
// ---------------------------------------------------------------------------

/**
 * Find the first `postMessage` call matching a message type.
 *
 * @param postMessage - A `jest.Mock` capturing postMessage calls.
 * @param type - Message type to match.
 * @returns The matching posted payload, or `undefined` if not found.
 */
export function findPostByType<T extends WorkerPost>(
  postMessage: jest.Mock,
  type: string,
): T | undefined {
  const calls = postMessage.mock.calls as unknown[][];
  const match = calls.find((call) => {
    const first = call[0];
    return (
      first !== null &&
      typeof first === 'object' &&
      (first as WorkerPost).type === type
    );
  });
  return match ? (match[0] as T) : undefined;
}

// ---------------------------------------------------------------------------
// Mock impact spot builder
// ---------------------------------------------------------------------------

/**
 * Build a deterministic wall-impact spot for direct render-function tests.
 *
 * @param overrides - Optional property overrides.
 * @returns A mock {@link ImpactSpot}.
 */
export function createMockImpact(overrides?: {
  createdAtMs?: number;
  boltTravelTimeMs?: number;
  lifetimeMs?: number;
  position?: { x: number; y: number };
}): ImpactSpot {
  return {
    wallHit: { mapX: 10, mapY: 10, side: 0, wallX: 0.5 },
    position: overrides?.position ?? { x: 11, y: 11 },
    createdAtMs: overrides?.createdAtMs ?? 0,
    lifetimeMs: overrides?.lifetimeMs ?? 1000,
    perpWallDist: 1,
    boltTravelTimeMs: overrides?.boltTravelTimeMs ?? 100,
  };
}

// ---------------------------------------------------------------------------
// Shared beforeEach setup
// ---------------------------------------------------------------------------

/**
 * Shared `beforeEach` for worker-based tests.
 *
 * Clears mock call history and installs a minimal `ImageData` stand-in
 * for Node.js test environments (which lack the Web `ImageData` constructor).
 */
export function workerBeforeEach(): void {
  beforeEach(() => {
    workerSelf.postMessage.mockClear();
    workerSelf.requestAnimationFrame.mockClear();

    // The worker tier constructs a real ImageData object as a canvas snapshot
    // for encoded sprite rendering. Node's test environment has no ImageData, so
    // provide a minimal compatible stand-in.
    Object.defineProperty(globalThis, 'ImageData', {
      value: class {
        /** Flat RGBA pixel data. */
        data: Uint8ClampedArray;
        /** Image width in pixels. */
        width: number;
        /** Image height in pixels. */
        height: number;
        constructor(data: Uint8ClampedArray, width: number, height: number) {
          this.data = data;
          this.width = width;
          this.height = height;
        }
      },
      configurable: true,
      writable: true,
    });
  });
}