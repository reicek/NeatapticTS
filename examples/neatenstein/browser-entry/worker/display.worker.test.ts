import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import {
  NEATENSTEIN_GUN_ACCENT_COLOR,
  NEATENSTEIN_IMPACT_SPOT_COLOR,
  NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR,
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
} from '../constants';
import { drawBolts, drawImpactSpots } from '../renderer/bolt-render';
import type { EnemyControllerState } from '../../scripts/enemy-controller';

const loadModule = (path: string): Promise<unknown> => import(path);

interface MockWorkerGlobal {
  postMessage: jest.Mock;
  requestAnimationFrame: jest.Mock;
  onmessage: ((event: MessageEvent) => void) | null;
}

type WorkerPost = {
  type?: string;
  frame?: Record<string, unknown>;
  [key: string]: unknown;
};

function installMockWorkerGlobal(): MockWorkerGlobal {
  const self: MockWorkerGlobal = {
    postMessage: jest.fn(),
    requestAnimationFrame: jest.fn(() => 0),
    onmessage: null,
  };
  (globalThis as unknown as Record<string, unknown>).self = self;
  return self;
}

const workerSelf = installMockWorkerGlobal();

function sendInitMessage(
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

function sendSimStateMessage(cameraYaw = 0.25) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: 'simState',
        state: {
          canvasWidth: 640,
          canvasHeight: 360,
          simTick: 1,
          cameraX: 12.5,
          cameraY: 12.5,
          cameraYaw,
          mapSeed: 42,
        },
      },
    } as unknown as MessageEvent);
  }
}

function sendInputMessage(yawDelta: number) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input: { yawDelta },
      },
    } as unknown as MessageEvent);
  }
}

function sendActionInputMessage(fire: boolean) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input: { fire },
      },
    } as unknown as MessageEvent);
  }
}

function sendRawMessage(data: unknown) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({ data } as unknown as MessageEvent);
  }
}

function createMockCanvas(context: OffscreenCanvasRenderingContext2D): {
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

function createMockContext(): {
  context: OffscreenCanvasRenderingContext2D & {
    canvas: { width: number; height: number };
  };
  setters: Record<string, jest.Mock>;
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
  };
}

function findPostByType<T extends WorkerPost>(
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

/**
 * Build a deterministic wall-impact spot for direct render-function tests.
 */
function createMockImpact(overrides?: {
  createdAtMs?: number;
  boltTravelTimeMs?: number;
  lifetimeMs?: number;
  position?: { x: number; y: number };
}): import('../host/game/types').ImpactSpot {
  return {
    wallHit: { mapX: 10, mapY: 10, side: 0, wallX: 0.5 },
    position: overrides?.position ?? { x: 11, y: 11 },
    createdAtMs: overrides?.createdAtMs ?? 0,
    lifetimeMs: overrides?.lifetimeMs ?? 1000,
    perpWallDist: 1,
    boltTravelTimeMs: overrides?.boltTravelTimeMs ?? 100,
  };
}

describe('Neatenstein display worker', () => {
  beforeEach(() => {
    workerSelf.postMessage.mockClear();
    workerSelf.requestAnimationFrame.mockClear();
  });

  it('acknowledges an init message with the received tier and version', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');
    const initializedCall = findPostByType<{
      type: string;
      tier: string;
      version: string;
    }>(workerSelf.postMessage, 'initialized');
    expect(initializedCall).toEqual({
      type: 'initialized',
      tier: 'cpu',
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    });
  });

  it('posts a frame-shaped payload after receiving a simState message', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();
    sendSimStateMessage();
    const frameCall = findPostByType<{
      frame: { format: string; version: string; requestId: number };
    }>(workerSelf.postMessage, 'frame');
    expect({
      hasFrameMessage: frameCall !== undefined,
      format: frameCall?.frame?.format,
      version: frameCall?.frame?.version,
      requestIdType: typeof frameCall?.frame?.requestId,
    }).toEqual({
      hasFrameMessage: true,
      format: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
      requestIdType: 'number',
    });
  });

  it('does not schedule requestAnimationFrame for the worker tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.requestAnimationFrame.mock.calls.length;
    sendInitMessage('worker');
    expect(workerSelf.requestAnimationFrame.mock.calls.length).toBe(before);
  });

  it('does not schedule requestAnimationFrame for the cpu tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.requestAnimationFrame.mock.calls.length;
    sendInitMessage('cpu');
    expect(workerSelf.requestAnimationFrame.mock.calls.length).toBe(before);
  });

  it('does not schedule requestAnimationFrame for the gpu tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.requestAnimationFrame.mock.calls.length;
    sendInitMessage('gpu');
    expect(workerSelf.requestAnimationFrame.mock.calls.length).toBe(before);
  });

  it('changes the rendered frame after receiving an input message', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');

    sendSimStateMessage(0);
    const firstMessage = findPostByType<{
      frame: { wallDistances: number[] };
    }>(workerSelf.postMessage, 'frame');

    workerSelf.postMessage.mockClear();
    sendInputMessage(Math.PI / 2);
    sendSimStateMessage(0);
    const secondMessage = findPostByType<{
      frame: { wallDistances: number[] };
    }>(workerSelf.postMessage, 'frame');

    const firstWallDistances = firstMessage?.frame?.wallDistances as
      number[] | undefined;
    const secondWallDistances = secondMessage?.frame?.wallDistances as
      number[] | undefined;
    expect(
      secondWallDistances?.some(
        (distance, index) => distance !== firstWallDistances?.[index],
      ),
    ).toBe(true);
  });

  it('packs the gun state into the cpu frame', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();

    sendActionInputMessage(true);
    sendSimStateMessage();

    const frameCall = findPostByType<{
      frame: { gun?: Record<string, unknown> };
    }>(workerSelf.postMessage, 'frame');

    expect(frameCall?.frame?.gun).toEqual(expect.any(Object));
  });

  it('packs active bolts into the cpu frame', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();

    sendActionInputMessage(true);
    sendSimStateMessage();

    const frameCall = findPostByType<{
      frame: { bolts?: unknown[] };
    }>(workerSelf.postMessage, 'frame');

    expect((frameCall?.frame?.bolts ?? []).length).toBeGreaterThan(0);
  });

  it('persists enemy controller state across simState ticks', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
    };

    sendInitMessage('cpu');
    const initial = workerModule.__testOnlyGetEnemyControllerState?.();
    expect(initial?.enemies).toHaveLength(0);

    workerSelf.postMessage.mockClear();
    sendSimStateMessage();
    const afterOne = workerModule.__testOnlyGetEnemyControllerState?.();
    expect(afterOne?.enemies).toHaveLength(1);

    workerSelf.postMessage.mockClear();
    sendSimStateMessage();
    const afterTwo = workerModule.__testOnlyGetEnemyControllerState?.();
    expect(afterTwo?.enemies).toHaveLength(2);
    expect(afterTwo?.enemies[0].position).not.toEqual(
      afterOne?.enemies[0].position,
    );
  });

  it('initializes a 2D worker canvas context', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    // The worker lazily creates the 2D context during the first frame build,
    // not at init time, so drive a frame before asserting.
    sendActionInputMessage(false);
    sendSimStateMessage();

    expect(canvas.getContext).toHaveBeenCalledWith('2d');
  });

  it('draws bolts as circles in the worker tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendActionInputMessage(true);
    sendSimStateMessage();

    expect(context.arc).toHaveBeenCalled();
  });

  it('draws bolts without a stroke energy trail', () => {
    const { context } = createMockContext();

    drawBolts(
      context,
      [
        {
          active: true,
          position: { x: 5, y: 0 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 36,
          origin: { x: 0, y: 0 },
          targetDistance: 5,
          createdAtMs: 0,
        },
      ],
      { x: 0, y: 0, yaw: 0 },
      640,
      360,
      150,
    );

    expect(context.stroke).not.toHaveBeenCalled();
  });

  it('draws bolts without a lineTo energy trail', () => {
    const { context } = createMockContext();

    drawBolts(
      context,
      [
        {
          active: true,
          position: { x: 5, y: 0 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 36,
          origin: { x: 0, y: 0 },
          targetDistance: 5,
          createdAtMs: 0,
        },
      ],
      { x: 0, y: 0, yaw: 0 },
      640,
      360,
      150,
    );

    expect(context.lineTo).not.toHaveBeenCalled();
  });

  it('does not set impact spot fill style before the bolt has arrived', () => {
    const { context, setters } = createMockContext();
    const zBuffer = new Float32Array(1);
    zBuffer[0] = Number.POSITIVE_INFINITY;

    const impact = createMockImpact({
      createdAtMs: 0,
      boltTravelTimeMs: 100,
      position: { x: 10, y: 10 },
    });

    drawImpactSpots(
      context,
      [impact],
      zBuffer,
      { x: 9, y: 9, yaw: Math.PI / 4 },
      640,
      360,
      50, // elapsed 50ms < 100ms travel time
    );

    expect(setters.fillStyle).not.toHaveBeenCalledWith(
      NEATENSTEIN_IMPACT_SPOT_COLOR,
    );
  });

  it('does not set impact spot glow color before the bolt has arrived', () => {
    const { context, setters } = createMockContext();
    const zBuffer = new Float32Array(1);
    zBuffer[0] = Number.POSITIVE_INFINITY;

    const impact = createMockImpact({
      createdAtMs: 0,
      boltTravelTimeMs: 100,
      position: { x: 10, y: 10 },
    });

    drawImpactSpots(
      context,
      [impact],
      zBuffer,
      { x: 9, y: 9, yaw: Math.PI / 4 },
      640,
      360,
      50, // elapsed 50ms < 100ms travel time
    );

    expect(setters.shadowColor).not.toHaveBeenCalledWith(
      NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR,
    );
  });

  it('sets impact spot fill style once the bolt has arrived', () => {
    const { context, setters } = createMockContext();
    const zBuffer = new Float32Array(1);
    zBuffer[0] = Number.POSITIVE_INFINITY;

    const impact = createMockImpact({
      createdAtMs: 0,
      boltTravelTimeMs: 100,
      position: { x: 10, y: 10 },
    });

    drawImpactSpots(
      context,
      [impact],
      zBuffer,
      { x: 9, y: 9, yaw: Math.PI / 4 },
      640,
      360,
      150, // elapsed 150ms >= 100ms travel time
    );

    expect(setters.fillStyle).toHaveBeenCalledWith(
      NEATENSTEIN_IMPACT_SPOT_COLOR,
    );
  });

  it('sets impact spot glow color once the bolt has arrived', () => {
    const { context, setters } = createMockContext();
    const zBuffer = new Float32Array(1);
    zBuffer[0] = Number.POSITIVE_INFINITY;

    const impact = createMockImpact({
      createdAtMs: 0,
      boltTravelTimeMs: 100,
      position: { x: 10, y: 10 },
    });

    drawImpactSpots(
      context,
      [impact],
      zBuffer,
      { x: 9, y: 9, yaw: Math.PI / 4 },
      640,
      360,
      150, // elapsed 150ms >= 100ms travel time
    );

    expect(setters.shadowColor).toHaveBeenCalledWith(
      NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR,
    );
  });

  it('draws an impact spot arc once the bolt has arrived', () => {
    const { context } = createMockContext();
    const zBuffer = new Float32Array(1);
    zBuffer[0] = Number.POSITIVE_INFINITY;

    const impact = createMockImpact({
      createdAtMs: 0,
      boltTravelTimeMs: 100,
      position: { x: 10, y: 10 },
    });

    drawImpactSpots(
      context,
      [impact],
      zBuffer,
      { x: 9, y: 9, yaw: Math.PI / 4 },
      640,
      360,
      150, // elapsed 150ms >= 100ms travel time
    );

    expect(context.arc).toHaveBeenCalled();
  });

  it('draws the gun overlay in the worker tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context, setters } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendActionInputMessage(false);
    sendSimStateMessage();

    expect(setters.fillStyle).toHaveBeenCalledWith(
      NEATENSTEIN_GUN_ACCENT_COLOR,
    );
  });

  it('does not post a frame when simState arrives before init', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendSimStateMessage();

    expect(findPostByType(workerSelf.postMessage, 'frame')).toBeUndefined();
  });

  it('does not acknowledge init with an unrecognised tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendRawMessage({ type: 'init', tier: 'bad' });

    expect(
      findPostByType(workerSelf.postMessage, 'initialized'),
    ).toBeUndefined();
  });

  it('initializes with the default seed when mapSeed is omitted', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendRawMessage({
      type: 'init',
      tier: 'cpu',
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    });
    workerSelf.postMessage.mockClear();

    sendSimStateMessage();

    expect(findPostByType(workerSelf.postMessage, 'frame')).toBeDefined();
  });

  it('uses the default version when the init message omits version', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendRawMessage({ type: 'init', tier: 'cpu', mapSeed: 42 });

    const initializedCall = findPostByType(
      workerSelf.postMessage,
      'initialized',
    );
    expect(initializedCall).toEqual(
      expect.objectContaining({
        type: 'initialized',
        tier: 'cpu',
        version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
      }),
    );
  });

  it('does not render a worker frame when no canvas is transferred', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('worker');

    expect(() => sendSimStateMessage()).not.toThrow();
    expect(findPostByType(workerSelf.postMessage, 'frame')).toBeUndefined();
  });

  it('resizes the worker canvas to match the host-provided render size', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = {
      getContext: jest.fn(() => context),
      width: 100,
      height: 100,
    };
    sendInitMessage('worker', canvas);

    sendSimStateMessage();

    expect(canvas.width).toBe(640);
    expect(canvas.height).toBe(360);
  });

  it('reuses the worker 2D context across frames', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendSimStateMessage();
    sendSimStateMessage();

    expect(canvas.getContext).toHaveBeenCalledTimes(1);
  });

  it('does not crash when the canvas returns no 2D context', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const canvas = {
      getContext: jest.fn(() => null),
      width: 640,
      height: 360,
    };
    sendInitMessage('worker', canvas);

    expect(() => sendSimStateMessage()).not.toThrow();
  });

  it('ignores non-object messages', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.postMessage.mock.calls.length;

    sendRawMessage('not an object');

    expect(workerSelf.postMessage.mock.calls.length).toBe(before);
  });

  it('ignores messages with an unknown type', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.postMessage.mock.calls.length;

    sendRawMessage({ type: 'unknown' });

    expect(workerSelf.postMessage.mock.calls.length).toBe(before);
  });

  it('posts a packed frame for the gpu tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('gpu');
    workerSelf.postMessage.mockClear();

    sendSimStateMessage();

    expect(findPostByType(workerSelf.postMessage, 'frame')).toBeDefined();
  });

  it('does not post a frame for simState with invalid dimensions', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');

    sendRawMessage({
      type: 'simState',
      state: {
        canvasWidth: Number.NaN,
        canvasHeight: 360,
        simTick: 1,
        cameraX: 12.5,
        cameraY: 12.5,
        cameraYaw: 0.25,
        mapSeed: 42,
      },
    });

    expect(findPostByType(workerSelf.postMessage, 'frame')).toBeUndefined();
  });

  describe('AC-402R: no dynamic light overlay in worker tier', () => {
    it('does not use screen blending for dynamic light', async () => {
      jest.resetModules();
      await loadModule('./display.worker.ts');
      const { context, setters } = createMockContext();
      const canvas = createMockCanvas(context);
      sendInitMessage('worker', canvas);
      workerSelf.postMessage.mockClear();

      sendActionInputMessage(false);
      sendSimStateMessage();

      expect(setters.globalCompositeOperation).not.toHaveBeenCalledWith(
        'screen',
      );
    });

    it('does not create a radial gradient for dynamic light', async () => {
      jest.resetModules();
      await loadModule('./display.worker.ts');
      const { context } = createMockContext();
      const canvas = createMockCanvas(context);
      sendInitMessage('worker', canvas);
      workerSelf.postMessage.mockClear();

      sendActionInputMessage(false);
      sendSimStateMessage();

      expect(context.createRadialGradient).not.toHaveBeenCalled();
    });
  });
});
