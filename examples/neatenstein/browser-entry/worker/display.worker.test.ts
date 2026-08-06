import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import {
  NEATENSTEIN_GUN_ACCENT_COLOR,
  NEATENSTEIN_IMPACT_SPOT_COLOR,
  NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR,
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
} from '../constants';
import { drawBolts, drawImpactSpots } from '../renderer/bolt-render';
import { NEATENSTEIN_RENDER_DISTANCE_CAP } from '../renderer/framebuffer';
import {
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
} from '../renderer/floor';
import type {
  ControlledEnemy,
  EnemyControllerState,
} from '../../scripts/enemy-controller';
import * as robotSpriteData from '../../robot-sprite-data.js';
import type { GameState } from '../host/game/types';

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

function sendSimStateMessage(
  cameraYaw = 0.25,
  overrides: { cameraX?: number; cameraY?: number } = {},
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

function sendMovementInputMessage(movement: {
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

function sendResizeMessage(width: number, height: number) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: { type: 'resize', width, height },
    } as unknown as MessageEvent);
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

  it('shifts the wall camera when WASD moves the authoritative player position', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');

    // Host-provided camera snapshot stays at (12.5, 12.5) for both frames.
    sendSimStateMessage(0);
    const baselineMessage = findPostByType<{
      frame: { wallDistances: number[] };
    }>(workerSelf.postMessage, 'frame');

    workerSelf.postMessage.mockClear();

    // Forward movement is applied during the next gameTick.
    sendMovementInputMessage({ forward: true });
    sendSimStateMessage(0);
    const movedMessage = findPostByType<{
      frame: { wallDistances: number[] };
    }>(workerSelf.postMessage, 'frame');

    const baselineWallDistances = baselineMessage?.frame?.wallDistances as
      number[] | undefined;
    const movedWallDistances = movedMessage?.frame?.wallDistances as
      number[] | undefined;

    expect(
      movedWallDistances?.some(
        (distance, index) => distance !== baselineWallDistances?.[index],
      ),
    ).toBe(true);
  });

  it('ignores a stale host camera snapshot that differs from the authoritative player position', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');

    sendSimStateMessage(0, { cameraX: 10, cameraY: 10 });
    const firstMessage = findPostByType<{
      frame: { wallDistances: number[] };
    }>(workerSelf.postMessage, 'frame');

    workerSelf.postMessage.mockClear();

    // Host snapshot jumps to a different camera position, but the player did
    // not move. The wall camera must stay on the authoritative player position.
    sendSimStateMessage(0, { cameraX: 20, cameraY: 20 });
    const secondMessage = findPostByType<{
      frame: { wallDistances: number[] };
    }>(workerSelf.postMessage, 'frame');

    expect(secondMessage?.frame?.wallDistances).toEqual(
      firstMessage?.frame?.wallDistances,
    );
  });

  it('updates the wall camera yaw from the authoritative player angle on mouse input', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');

    sendSimStateMessage(0, { cameraX: 1, cameraY: 1 });
    const firstMessage = findPostByType<{
      frame: { wallDistances: number[] };
    }>(workerSelf.postMessage, 'frame');

    workerSelf.postMessage.mockClear();

    // The host snapshot keeps the same cameraYaw, but the worker updates the
    // authoritative player angle from the mouse delta during gameTick.
    sendInputMessage(Math.PI / 4);
    sendSimStateMessage(0, { cameraX: 1, cameraY: 1 });
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

  it('clears all controlled enemies when gameTick returns null gameState', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
    };

    sendInitMessage('cpu');
    // Build up some enemies first
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }
    const beforeNull = workerModule.__testOnlyGetEnemyControllerState?.();
    expect(beforeNull?.enemies.length).toBeGreaterThan(0);

    // Mock gameTick to return null — triggers the !gameState guard at line 1098.
    // Also mock updateEnemyController to prevent the subsequent call (line 1114)
    // from crashing when it receives null gameState.
    const tickModule = await import('../host/game/tick');
    const enemyControllerModule =
      await import('../../scripts/enemy-controller');
    const gameTickSpy = jest
      .spyOn(tickModule, 'gameTick')
      .mockReturnValue(null as unknown as GameState);
    const controllerSpy = jest
      .spyOn(enemyControllerModule, 'updateEnemyController')
      .mockImplementation((state: EnemyControllerState) => state);

    workerSelf.postMessage.mockClear();
    sendSimStateMessage();

    const afterNull = workerModule.__testOnlyGetEnemyControllerState?.();
    // All enemies should be cleared (active=false, then filtered out)
    expect(afterNull?.enemies.length).toBe(0);
    expect(gameTickSpy).toHaveBeenCalled();

    gameTickSpy.mockRestore();
    controllerSpy.mockRestore();
  });

  it('removes dead enemies from controller state after tick', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
    };

    sendInitMessage('cpu');
    // Build up enemies
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }
    const before = workerModule.__testOnlyGetEnemyControllerState?.();
    expect(before?.enemies.length).toBeGreaterThan(0);

    // Mock gameTick to return a state where enemy[0] is dead (health=0,
    // active=false). We avoid calling the real gameTick to ensure the mock
    // is deterministic and does not depend on module binding internals.
    const tickModule = await import('../host/game/tick');
    const gameTickSpy = jest.spyOn(tickModule, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          enemies: state.enemies.map((enemy, i) =>
            i === 0 ? { ...enemy, health: 0, active: false } : enemy,
          ),
        }) as GameState,
    );

    sendSimStateMessage();

    expect(gameTickSpy).toHaveBeenCalled();
    const after = workerModule.__testOnlyGetEnemyControllerState?.();
    // The dead enemy is filtered out and then re-added by updateEnemyController
    // in the de-rez (death) state. Verify the dead-enemy filter ran by checking
    // that at least one controller enemy is now in the death animation.
    expect(after?.enemies.length).toBeGreaterThan(0);
    const hasDeadEnemy = after?.enemies.some(
      (enemy) => enemy.health <= 0 && enemy.animationState === 'death',
    );
    expect(hasDeadEnemy).toBe(true);

    gameTickSpy.mockRestore();
  });

  it('renders enemy sprites using a single canvas snapshot and flush per frame', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
    };

    const { context, getImageData, putImageData } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    // Advance enough ticks to spawn several active enemies. Edge-based
    // spawning places them at map edges (~60 cells from center), so we
    // inject test enemies near the player to verify the sprite pipeline.
    for (let i = 0; i < 8; i += 1) {
      sendSimStateMessage();
    }

    // Override enemy positions to be within camera FOV and within the 30-cell
    // render distance cap so sprites are rendered (not culled). The display
    // worker uses gameState.player.position (60.5, 60.5) for the camera, not
    // the sim state's cameraX/cameraY. The cleared central arena spans cells
    // 56-64, so positions near the player are guaranteed open.
    workerModule.__testOnlyInjectTestEnemies?.([
      { x: 63.5, y: 60.5 },
      { x: 62.5, y: 61.5 },
    ]);

    // Send one more tick to trigger rendering with the injected enemies.
    sendSimStateMessage();

    const controller = workerModule.__testOnlyGetEnemyControllerState?.();
    expect(controller?.enemies.length).toBeGreaterThanOrEqual(2);

    // The sprite pipeline snapshots the canvas once per frame and flushes
    // it back once when enemies are present.
    expect(getImageData).toHaveBeenCalled();
    expect(putImageData).toHaveBeenCalled();

    // The flushed snapshot contains non-zero sprite pixels from an encoded
    // robot frame.
    const lastCall =
      putImageData.mock.calls[putImageData.mock.calls.length - 1];
    const flushedData = (lastCall[0] as { data: Uint8ClampedArray }).data;
    const nonZeroPixels = flushedData.filter((value) => value !== 0).length;
    expect(nonZeroPixels).toBeGreaterThan(0);
  });

  it('renders a frame when the worker context lacks getImageData', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
    };

    const { context, putImageData } = createMockContext();
    delete (context as unknown as Record<string, unknown>).getImageData;
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    for (let i = 0; i < 8; i += 1) {
      sendSimStateMessage();
    }

    const controller = workerModule.__testOnlyGetEnemyControllerState?.();
    expect(controller?.enemies.length).toBeGreaterThanOrEqual(2);
    // Without a canvas read-back path the sprite pass is skipped, but the
    // worker must still render the frame and must not throw.
    expect(putImageData).toHaveBeenCalledTimes(0);
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

  it('resizes the worker canvas immediately when a resize message is received', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendResizeMessage(1024, 768);

    expect(canvas.width).toBe(1024);
    expect(canvas.height).toBe(768);
  });

  it('patches latestState dimensions when a resize message is received', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetLatestState?():
        import('../renderer/frame').NeatensteinRenderState | null;
    };
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    sendSimStateMessage();
    workerSelf.postMessage.mockClear();

    sendResizeMessage(1024, 768);

    const latestState = workerModule.__testOnlyGetLatestState?.();
    expect(latestState?.canvasWidth).toBe(1024);
    expect(latestState?.canvasHeight).toBe(768);
  });

  it('ignores a resize message with invalid dimensions', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendResizeMessage(Number.NaN, Number.NaN);

    expect(canvas.width).toBe(640);
    expect(canvas.height).toBe(360);
  });

  it('ignores a resize message with non-number dimensions', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendRawMessage({ type: 'resize', width: '1024', height: '768' });

    expect(canvas.width).toBe(640);
    expect(canvas.height).toBe(360);
  });

  it('applies a pending resize to the worker canvas when init arrives later', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);

    sendResizeMessage(1024, 768);
    sendInitMessage('worker', canvas);

    expect(canvas.width).toBe(1024);
    expect(canvas.height).toBe(768);
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

  it('calls commit() on the worker 2D context when available', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context, putImageData } = createMockContext();
    const commit = jest.fn();
    (context as unknown as { commit: typeof commit }).commit = commit;
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendSimStateMessage();

    expect(putImageData).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
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

  it('falls back to the simulated player position when camera coordinates are non-finite', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();

    if (typeof workerSelf.onmessage === 'function') {
      workerSelf.onmessage({
        data: {
          type: 'simState',
          state: {
            canvasWidth: 640,
            canvasHeight: 360,
            simTick: 1,
            cameraX: Number.NaN,
            cameraY: Number.NaN,
            cameraYaw: 0.25,
            mapSeed: 42,
          },
        },
      } as unknown as MessageEvent);
    }

    expect(findPostByType(workerSelf.postMessage, 'frame')).toBeDefined();
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

  describe('encoded robot sprite red contracts', () => {
    it('worker sprite pass uses encoded robot frames with a single canvas snapshot per frame', async () => {
      jest.resetModules();
      await loadModule('./display.worker.ts');

      const { context, getImageData, putImageData } = createMockContext();
      const canvas = createMockCanvas(context);
      sendInitMessage('worker', canvas);
      workerSelf.postMessage.mockClear();

      const realSprites = await import('../renderer/sprites');
      const renderSpy = jest.spyOn(realSprites, 'renderNeatensteinSprite');

      for (let i = 0; i < 8; i += 1) {
        sendSimStateMessage();
      }

      expect(getImageData).toHaveBeenCalledTimes(8);
      expect(putImageData).toHaveBeenCalledTimes(8);

      const typedRobotSpriteData = robotSpriteData as unknown as {
        ROBOT_SPRITE_FRAMES: Record<string, Record<string, number[][]>>;
      };
      const encodedFrames: number[][][] = [];
      for (const direction of Object.values(
        typedRobotSpriteData.ROBOT_SPRITE_FRAMES,
      )) {
        for (const frame of Object.values(direction)) {
          encodedFrames.push(frame);
        }
      }

      expect(renderSpy.mock.calls.length).toBeGreaterThan(0);
      const arraysEqual = (a: unknown, b: unknown): boolean =>
        JSON.stringify(a) === JSON.stringify(b);
      const receivedEncodedFrame = renderSpy.mock.calls.some((call) =>
        encodedFrames.some((frame) =>
          arraysEqual(frame, call[3] as number[][]),
        ),
      );
      expect(receivedEncodedFrame).toBe(true);
    });

    it('passes each enemy teamColor as the 6th argument to renderNeatensteinSprite', async () => {
      jest.resetModules();
      const workerModule = (await loadModule('./display.worker.ts')) as {
        __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
        __testOnlyResolveEnemyTeamColor?(
          index: number,
        ): readonly [number, number, number];
      };

      const { context, getImageData, putImageData } = createMockContext();
      const canvas = createMockCanvas(context);
      sendInitMessage('worker', canvas);
      workerSelf.postMessage.mockClear();

      const realSprites = await import('../renderer/sprites');
      const renderSpy = jest.spyOn(realSprites, 'renderNeatensteinSprite');

      for (let i = 0; i < 8; i += 1) {
        sendSimStateMessage();
      }

      expect(getImageData).toHaveBeenCalledTimes(8);
      expect(putImageData).toHaveBeenCalledTimes(8);
      expect(renderSpy.mock.calls.length).toBeGreaterThan(0);

      const controller = workerModule.__testOnlyGetEnemyControllerState?.();
      expect(controller?.enemies.length).toBeGreaterThanOrEqual(2);

      const resolveTeamColor = workerModule.__testOnlyResolveEnemyTeamColor;
      expect(resolveTeamColor).toBeDefined();

      const arraysEqual = (a: unknown, b: unknown): boolean =>
        JSON.stringify(a) === JSON.stringify(b);

      // Every render call must pass a teamColor triple as the 6th argument,
      // and it must match resolveEnemyTeamColor(enemy.index) for an active enemy.
      const expectedTeamColors = controller!.enemies
        .filter((enemy) => enemy.active)
        .map((enemy) => resolveTeamColor!(enemy.index));

      const everyCallHasTeamColor = renderSpy.mock.calls.every((call) => {
        const passed = call[5];
        return (
          Array.isArray(passed) &&
          passed.length === 3 &&
          expectedTeamColors.some((color) => arraysEqual(color, passed))
        );
      });
      expect(everyCallHasTeamColor).toBe(true);

      // The first render call's 6th argument is the teamColor of one of the
      // active enemies, locking in the wiring regression.
      const firstTeamColor = renderSpy.mock.calls[0][5];
      expect(
        expectedTeamColors.some((color) => arraysEqual(color, firstTeamColor)),
      ).toBe(true);

      renderSpy.mockRestore();
    });

    it('flushes encoded robot sprite pixels from the canvas snapshot', async () => {
      jest.resetModules();
      const workerModule = (await loadModule('./display.worker.ts')) as {
        __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
        __testOnlyResolveEnemyTeamColor?(
          index: number,
        ): readonly [number, number, number];
        __testOnlyInjectTestEnemies?(
          positions: { x: number; y: number }[],
        ): void;
      };

      const { context, getImageData, putImageData } = createMockContext();
      const canvas = createMockCanvas(context);
      sendInitMessage('worker', canvas);
      workerSelf.postMessage.mockClear();

      for (let i = 0; i < 8; i += 1) {
        sendSimStateMessage();
      }

      // Override enemy positions to be within camera FOV and within the
      // 40-cell render distance cap so sprites are rendered (not culled).
      // The display worker uses gameState.player.position (60.5, 60.5) for
      // the camera. With step-function fog, fogFactor=0 below 40 cells means
      // pixel colors match the original team colors exactly.
      workerModule.__testOnlyInjectTestEnemies?.([
        { x: 63.5, y: 60.5 },
        { x: 62.5, y: 61.5 },
      ]);
      sendSimStateMessage();

      expect(getImageData).toHaveBeenCalled();
      expect(putImageData).toHaveBeenCalled();

      const controller = workerModule.__testOnlyGetEnemyControllerState?.();
      expect(controller?.enemies.length).toBeGreaterThanOrEqual(2);
      const resolveTeamColor = workerModule.__testOnlyResolveEnemyTeamColor;
      expect(resolveTeamColor).toBeDefined();

      // With team coloring wired, palette indices 5/6/7 are swapped with each
      // active enemy's team color while preserving alpha. The encoded robot
      // frame therefore flushes team-tinted pixels (alpha 255 for the opaque
      // red base at index 5) rather than the raw red palette entry.
      const expectedTeamColors = controller!.enemies
        .filter((enemy) => enemy.active)
        .map((enemy) => resolveTeamColor!(enemy.index));

      let foundTeamColoredPixel = false;
      for (const call of putImageData.mock.calls) {
        const data = (call[0] as { data: Uint8ClampedArray }).data;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i + 3] !== 255) {
            continue;
          }
          if (
            expectedTeamColors.some(
              (color) =>
                data[i] === color[0] &&
                data[i + 1] === color[1] &&
                data[i + 2] === color[2],
            )
          ) {
            foundTeamColoredPixel = true;
            break;
          }
        }
        if (foundTeamColoredPixel) {
          break;
        }
      }

      expect(foundTeamColoredPixel).toBe(true);
    });

    it('does not draw the debug red square overlay', async () => {
      jest.resetModules();
      await loadModule('./display.worker.ts');

      const { context } = createMockContext();
      const canvas = createMockCanvas(context);
      sendInitMessage('worker', canvas);
      workerSelf.postMessage.mockClear();

      sendSimStateMessage();

      const redSquareCall = (context.fillRect as jest.Mock).mock.calls.find(
        (call: unknown[]) =>
          call[0] === 0 && call[1] === 0 && call[2] === 40 && call[3] === 40,
      );
      expect(redSquareCall).toBeUndefined();
    });

    it('clears the full canvas before drawing the frame', async () => {
      jest.resetModules();
      await loadModule('./display.worker.ts');

      const { context } = createMockContext();
      const canvas = createMockCanvas(context);
      sendInitMessage('worker', canvas);
      workerSelf.postMessage.mockClear();

      sendSimStateMessage();

      const fullCanvasClear = (context.fillRect as jest.Mock).mock.calls.find(
        (call: unknown[]) =>
          call[0] === 0 && call[1] === 0 && call[2] === 640 && call[3] === 360,
      );
      expect(fullCanvasClear).toBeDefined();
    });

    it('draws floor and ceiling perspective grids in the worker tier', async () => {
      jest.resetModules();
      await loadModule('./display.worker.ts');

      const { context } = createMockContext();
      const canvas = createMockCanvas(context);
      sendInitMessage('worker', canvas);
      workerSelf.postMessage.mockClear();

      sendSimStateMessage();

      expect(context.stroke).toHaveBeenCalled();
    });

    it('draws fogged wall stripes with fillRect in the worker tier', async () => {
      jest.resetModules();
      await loadModule('./display.worker.ts');

      const { context, setters } = createMockContext();
      const canvas = createMockCanvas(context);
      sendInitMessage('worker', canvas);
      workerSelf.postMessage.mockClear();

      sendSimStateMessage();

      const wallStripeCall = (context.fillRect as jest.Mock).mock.calls.find(
        (call: unknown[]) => {
          const [x, , w, h] = call as number[];
          return (
            typeof x === 'number' &&
            typeof w === 'number' &&
            w > 0 &&
            w < canvas.width &&
            typeof h === 'number' &&
            h > 0 &&
            h < canvas.height
          );
        },
      );
      expect(wallStripeCall).toBeDefined();
      expect(setters.fillStyle).toHaveBeenCalledWith(
        expect.stringMatching(/^rgb\(/),
      );
    });

    it('falls back to palette index 0 for negative enemy type indices', async () => {
      jest.resetModules();
      const workerModule = (await loadModule('./display.worker.ts')) as {
        __testOnlyResolveEnemyTeamColor?(
          index: number,
        ): readonly [number, number, number];
      };

      const resolveTeamColor = workerModule.__testOnlyResolveEnemyTeamColor;
      expect(resolveTeamColor).toBeDefined();
      expect(resolveTeamColor!(-1)).toEqual(resolveTeamColor!(0));
    });
  });
});

describe('sprite render pass', () => {
  function makeEnemy(index: number, x: number, y: number): ControlledEnemy {
    return {
      index,
      position: { x, y },
      health: 100,
      yawRad: 0,
      animationState: 'idle',
      ammo: 100,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      bfsStallTicks: 0,
      weights: undefined,
      variantId: 0,
      previousStepDistance: -1,
    };
  }

  it('renders active enemies far-to-near so distant sprites do not overwrite closer ones', async () => {
    jest.resetModules();

    await loadModule('./display.worker.ts');

    const realEnemyController = await import('../../scripts/enemy-controller');
    const realSprites = await import('../renderer/sprites');
    const realRaycast = await import('../renderer/raycast');
    const realGameState = await import('../host/game/state');

    const referenceState = realGameState.createGameState({ seed: 42 });
    const cameraX = referenceState.player.position.x;
    const cameraY = referenceState.player.position.y;
    const cameraYaw = referenceState.player.angleRad;
    const nearDist = 2.5;
    const farDist = 12.5;
    const nearX = cameraX + Math.cos(cameraYaw) * nearDist;
    const nearY = cameraY + Math.sin(cameraYaw) * nearDist;
    const farX = cameraX + Math.cos(cameraYaw) * farDist;
    const farY = cameraY + Math.sin(cameraYaw) * farDist;

    jest.spyOn(realEnemyController, 'updateEnemyController').mockReturnValue({
      enemies: [makeEnemy(0, nearX, nearY), makeEnemy(1, farX, farY)],
      hitscanEvents: [],
    });
    const renderSpy = jest.spyOn(realSprites, 'renderNeatensteinSprite');
    jest.spyOn(realRaycast, 'castRayDDAFromFlatMap').mockReturnValue({
      perpWallDist: Number.POSITIVE_INFINITY,
      side: 0,
      mapX: 0,
      mapY: 0,
    });

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    const projections = renderSpy.mock.calls.map(
      (call) =>
        call[2] as unknown as {
          perpDist: number;
          visibleColumns: number[];
        },
    );
    const renderOrder = projections
      .filter((projection) => projection.visibleColumns.length > 0)
      .map((projection) => projection.perpDist);

    expect(renderOrder.length).toBe(2);
    expect(renderOrder[0]).toBeGreaterThan(renderOrder[1]);
  });

  it('does not clip sprite columns when the wall z-buffer contains NaN', async () => {
    jest.resetModules();

    await loadModule('./display.worker.ts');

    const realEnemyController = await import('../../scripts/enemy-controller');
    const realSprites = await import('../renderer/sprites');
    const realRaycast = await import('../renderer/raycast');
    const realGameState = await import('../host/game/state');

    const referenceState = realGameState.createGameState({ seed: 42 });
    const cameraX = referenceState.player.position.x;
    const cameraY = referenceState.player.position.y;
    const cameraYaw = referenceState.player.angleRad;
    const enemyDist = 5;
    const enemyX = cameraX + Math.cos(cameraYaw) * enemyDist;
    const enemyY = cameraY + Math.sin(cameraYaw) * enemyDist;

    jest.spyOn(realEnemyController, 'updateEnemyController').mockReturnValue({
      enemies: [makeEnemy(0, enemyX, enemyY)],
      hitscanEvents: [],
    });
    const renderSpy = jest.spyOn(realSprites, 'renderNeatensteinSprite');
    jest.spyOn(realRaycast, 'castRayDDAFromFlatMap').mockReturnValue({
      perpWallDist: Number.NaN,
      side: 0,
      mapX: 0,
      mapY: 0,
    });

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    expect(renderSpy).toHaveBeenCalledTimes(1);
    const projection = renderSpy.mock.calls[0][2] as unknown as {
      visibleColumns: number[];
    };
    expect(projection.visibleColumns.length).toBeGreaterThan(0);
  });

  it('skips encoded enemy sprites that resolve to no frame', async () => {
    jest.resetModules();

    await loadModule('./display.worker.ts');

    const realEnemyController = await import('../../scripts/enemy-controller');
    const realSprites = await import('../renderer/sprites');
    const realRaycast = await import('../renderer/raycast');
    const realGameState = await import('../host/game/state');

    const referenceState = realGameState.createGameState({ seed: 42 });
    const cameraX = referenceState.player.position.x;
    const cameraY = referenceState.player.position.y;
    const cameraYaw = referenceState.player.angleRad;
    const enemyDist = 5;
    const enemyX = cameraX + Math.cos(cameraYaw) * enemyDist;
    const enemyY = cameraY + Math.sin(cameraYaw) * enemyDist;

    jest.spyOn(realEnemyController, 'updateEnemyController').mockReturnValue({
      enemies: [{ ...makeEnemy(0, enemyX, enemyY), yawRad: Number.NaN }],
      hitscanEvents: [],
    });
    const renderSpy = jest.spyOn(realSprites, 'renderNeatensteinSprite');
    jest.spyOn(realRaycast, 'castRayDDAFromFlatMap').mockReturnValue({
      perpWallDist: 0,
      side: 0,
      mapX: 0,
      mapY: 0,
    });

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    expect(renderSpy).not.toHaveBeenCalled();
  });

  it('draws fog wall stripes at the render distance cap when the ray exceeds 40 cells', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realRaycast = await import('../renderer/raycast');
    jest.spyOn(realRaycast, 'castRayDDAFromFlatMap').mockReturnValue({
      perpWallDist: Number.POSITIVE_INFINITY,
      side: 0,
      mapX: 0,
      mapY: 0,
    });

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);

    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    // When the ray exceeds the 40-cell cap, the fog step function draws fog
    // wall stripes (background color) at the render distance cap instead of
    // skipping wall drawing entirely. Each column gets a fillRect with
    // stripePixelWidth === 1 and an integer x start.
    const fogWallCalls = (context.fillRect as jest.Mock).mock.calls.filter(
      (call) =>
        typeof call[0] === 'number' &&
        Number.isInteger(call[0]) &&
        call[2] === 1,
    );
    expect(fogWallCalls.length).toBeGreaterThan(0);
  });

  it('sets packed-frame zBuffer to render-distance cap for capped columns', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realRaycast = await import('../renderer/raycast');
    jest.spyOn(realRaycast, 'castRayDDAFromFlatMap').mockReturnValue({
      perpWallDist: Number.POSITIVE_INFINITY,
      side: 0,
      mapX: 0,
      mapY: 0,
    });

    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    const frameCall = findPostByType<{
      frame: { zBuffer: Float32Array; columnCount: number };
    }>(workerSelf.postMessage, 'frame');

    const zBuffer = frameCall?.frame?.zBuffer;
    expect(zBuffer).toBeDefined();
    expect(zBuffer![zBuffer!.length / 2]).toBe(NEATENSTEIN_RENDER_DISTANCE_CAP);
  });

  it('skips rendering enemy sprites beyond the 40-cell render distance cap (AC-10.3c-002)', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realEnemyController = await import('../../scripts/enemy-controller');
    const realSprites = await import('../renderer/sprites');
    const realRaycast = await import('../renderer/raycast');
    const realGameState = await import('../host/game/state');

    const referenceState = realGameState.createGameState({ seed: 42 });
    const cameraX = referenceState.player.position.x;
    const cameraY = referenceState.player.position.y;
    const cameraYaw = referenceState.player.angleRad;
    // Place enemy well beyond the 40-cell render distance cap.
    const farDist = 50;
    const enemyX = cameraX + Math.cos(cameraYaw) * farDist;
    const enemyY = cameraY + Math.sin(cameraYaw) * farDist;

    jest.spyOn(realEnemyController, 'updateEnemyController').mockReturnValue({
      enemies: [makeEnemy(0, enemyX, enemyY)],
      hitscanEvents: [],
    });
    const renderSpy = jest.spyOn(realSprites, 'renderNeatensteinSprite');
    jest.spyOn(realRaycast, 'castRayDDAFromFlatMap').mockReturnValue({
      perpWallDist: 1,
      side: 0,
      mapX: 0,
      mapY: 0,
    });

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    // The worker loop must skip (continue) sprites beyond the render cap
    // before calling renderNeatensteinSprite, mirroring the no-frame skip.
    expect(renderSpy).not.toHaveBeenCalled();
  });

  it('does not draw floor/ceiling strokes beyond the render distance cap (AC-10.3d-002)', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    const moveToCalls = (context.moveTo as jest.Mock).mock.calls;
    const lineToCalls = (context.lineTo as jest.Mock).mock.calls;
    const allPoints = [...moveToCalls, ...lineToCalls].map(
      (call: unknown[]) => ({ x: call[0] as number, y: call[1] as number }),
    );

    // Worker canvas is 640×360.
    const horizonY = 360 * NEATENSTEIN_FLOOR_HORIZON_RATIO;
    const focalLength = 360 / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

    const beyondCap = allPoints.filter((p) => {
      const dy = Math.abs(p.y - horizonY);
      if (dy < 1e-9) {
        return true;
      }
      const forwardDist =
        (NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD * focalLength) / dy;
      return forwardDist > NEATENSTEIN_RENDER_DISTANCE_CAP;
    });

    expect(beyondCap).toEqual([]);
  });
});

describe('AC-10.2c: worker simState collision sync', () => {
  it('calls updateEnemyController before gameTick and syncs controlled state into gameState.enemies', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realTick = await import('../host/game/tick');
    const realEnemyController = await import('../../scripts/enemy-controller');

    const gameTickSpy = jest.spyOn(realTick, 'gameTick');

    const controlledPosition = { x: 42.5, y: 43.5 };
    const updateSpy = jest
      .spyOn(realEnemyController, 'updateEnemyController')
      .mockImplementation((controllerState, _state) => {
        const controlled: ControlledEnemy = {
          index: 0,
          position: controlledPosition,
          health: 77,
          active: true,
          yawRad: 0,
          animationState: 'idle',
          ammo: 0,
          fireCooldownMs: 0,
          deRezElapsedMs: 0,
          walkTick: 0,
          shootBlinkTicks: 0,
          flankStallTicks: 0,
          bfsStallTicks: 0,
          weights: undefined,
          variantId: 0,
          previousStepDistance: -1,
        };
        controllerState.enemies = [controlled];
        _state.enemies = [
          {
            position: controlled.position,
            health: controlled.health,
            active: controlled.active,
          },
        ];
        return controllerState;
      });

    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    expect(gameTickSpy).toHaveBeenCalledTimes(1);
    const tickedState = gameTickSpy.mock.calls[0][0] as GameState;
    expect(tickedState.enemies).toHaveLength(1);
    expect(tickedState.enemies[0].position).toEqual(controlledPosition);
    expect(tickedState.enemies[0].health).toBe(77);
    expect(tickedState.enemies[0].active).toBe(true);

    gameTickSpy.mockRestore();
    updateSpy.mockRestore();
  });

  it('returns the original enemy unchanged when controlled.enemies[index] is undefined (line 1072 fallback)', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realTick = await import('../host/game/tick');
    const realEnemyController = await import('../../scripts/enemy-controller');

    const gameTickSpy = jest.spyOn(realTick, 'gameTick');

    // The worker's initial gameState starts with zero enemies. We inject one
    // enemy into the gameState via the mock's _state parameter (mirroring the
    // existing sync test pattern) but return a controller with an EMPTY enemies
    // array. This forces controlled.enemies[0] to be undefined and exercises the
    // fallback at display.worker.ts:1071-1072.
    const originalEnemyPosition = { x: 10.5, y: 20.5 };
    const updateSpy = jest
      .spyOn(realEnemyController, 'updateEnemyController')
      .mockImplementation((controllerState, _state) => {
        // Inject an enemy into the gameState so the map callback iterates.
        _state.enemies = [
          {
            position: { ...originalEnemyPosition },
            health: 50,
            active: true,
          },
        ];
        // Return controller with NO controlled enemies — index 0 is undefined.
        controllerState.enemies = [];
        return controllerState;
      });

    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    expect(gameTickSpy).toHaveBeenCalledTimes(1);
    const tickedState = gameTickSpy.mock.calls[0][0] as GameState;

    // The fallback returned the original enemy unchanged — position preserved.
    expect(tickedState.enemies).toHaveLength(1);
    expect(tickedState.enemies[0].position).toEqual(originalEnemyPosition);

    gameTickSpy.mockRestore();
    updateSpy.mockRestore();
  });
});

describe('AC-10.3 coverage iteration 2: uncovered branches', () => {
  it('uses positive deltaMs from simState as timestepMs (line 1060 true branch)', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realTick = await import('../host/game/tick');
    // Spy without mocking — let the real gameTick run; we only need to
    // capture the timestepMs argument (4th positional, index 3).
    const gameTickSpy = jest.spyOn(realTick, 'gameTick');

    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();

    // Send simState with a positive deltaMs to exercise the true branch of
    // `deltaMs > 0 ? deltaMs : 16` at display.worker.ts:1060.  Using 32
    // (not 16) distinguishes the true branch from the fallback.
    sendRawMessage({
      type: 'simState',
      state: {
        canvasWidth: 640,
        canvasHeight: 360,
        simTick: 1,
        cameraX: 12.5,
        cameraY: 12.5,
        cameraYaw: 0.25,
        mapSeed: 42,
        deltaMs: 32,
      },
    });

    expect(gameTickSpy).toHaveBeenCalledTimes(1);
    // gameTick(gameState, tickInput, collisionMap, timestepMs)
    const timestepMs = gameTickSpy.mock.calls[0][3];
    expect(timestepMs).toBe(32);

    gameTickSpy.mockRestore();
  });

  it('handles enemy with null health via nullish coalescing fallback (line 1100)', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
    };

    sendInitMessage('cpu');
    // Build up enemies so the controller has entries to filter.
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }
    const before = workerModule.__testOnlyGetEnemyControllerState?.();
    const beforeCount = before?.enemies.length ?? 0;
    expect(beforeCount).toBeGreaterThan(0);

    const tickModule = await import('../host/game/tick');
    const enemyControllerModule =
      await import('../../scripts/enemy-controller');

    // Mock gameTick to return enemy[0] with health: null, exercising the
    // `(live.health ?? 0)` nullish fallback at display.worker.ts:1100.
    // Keep active: true so the `live.active === false` branch is NOT the
    // trigger — only the nullish-coalescing path marks the enemy dead.
    const gameTickSpy = jest.spyOn(tickModule, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          enemies: state.enemies.map((enemy, i) =>
            i === 0
              ? {
                  ...enemy,
                  health: null as unknown as number,
                  active: true,
                }
              : enemy,
          ),
        }) as GameState,
    );

    // Passthrough updateEnemyController to prevent the post-tick call
    // (line 1112) from re-adding the dead enemy.
    const controllerSpy = jest
      .spyOn(enemyControllerModule, 'updateEnemyController')
      .mockImplementation((state: EnemyControllerState) => state);

    sendSimStateMessage();

    expect(gameTickSpy).toHaveBeenCalled();
    const after = workerModule.__testOnlyGetEnemyControllerState?.();
    // The enemy with health: null was treated as dead via `(null ?? 0) <= 0`
    // and filtered out by the `.filter((c) => c.active)` at line 1105.
    expect(after?.enemies.length ?? 0).toBeLessThan(beforeCount);

    gameTickSpy.mockRestore();
    controllerSpy.mockRestore();
  });
});

describe('AC-10.4 fix-coverage-gaps: fog factor and gradient feather branches', () => {
  it('resolveWallFogFactor returns 1 when perpWallDist >= 40 (render distance cap)', async () => {
    jest.resetModules();
    const mod = (await loadModule('./display.worker.ts')) as {
      __testOnlyResolveWallFogFactor: (d: number) => number;
    };
    // At and above the render distance cap, fog factor is 1 (full fog).
    expect(
      mod.__testOnlyResolveWallFogFactor(NEATENSTEIN_RENDER_DISTANCE_CAP),
    ).toBe(1);
    expect(mod.__testOnlyResolveWallFogFactor(100)).toBe(1);
    // Below the cap, fog factor is 0 (no fog).
    expect(
      mod.__testOnlyResolveWallFogFactor(NEATENSTEIN_RENDER_DISTANCE_CAP - 0.1),
    ).toBe(0);
  });

  it('exercises top gradient feather when isCapped column has drawStart > 0', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realRaycast = await import('../renderer/raycast');
    jest.spyOn(realRaycast, 'castRayDDAFromFlatMap').mockReturnValue({
      perpWallDist: Number.POSITIVE_INFINITY,
      side: 0,
      mapX: 0,
      mapY: 0,
    });

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);

    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    // With canvasHeight=360 and lineHeight = wallFocalLength / 30
    //   ≈ (360 / 2 / tan(π/6)) / 30 ≈ 10.39,
    // drawStart = (360 - 10.39) / 2 ≈ 174.8 > 0, so the top gradient
    // feather branch (line 630) is taken. Verify createLinearGradient
    // was called with y-coordinates in the top half (< 180).
    const gradientCalls = (context.createLinearGradient as jest.Mock).mock
      .calls as number[][];
    const topFeatherCalls = gradientCalls.filter(
      (call) => call[1] < 180 && call[3] < 180,
    );
    expect(topFeatherCalls.length).toBeGreaterThan(0);
  });

  it('exercises bottom gradient feather when isCapped column has drawEnd < canvasHeight', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realRaycast = await import('../renderer/raycast');
    jest.spyOn(realRaycast, 'castRayDDAFromFlatMap').mockReturnValue({
      perpWallDist: Number.POSITIVE_INFINITY,
      side: 0,
      mapX: 0,
      mapY: 0,
    });

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);

    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendSimStateMessage(0);

    // With canvasHeight=360 and lineHeight ≈ 10.39,
    // drawEnd = (360 + 10.39) / 2 ≈ 185.2 < 360, so the bottom gradient
    // feather branch (line 656) is taken. Verify createLinearGradient
    // was called with y-coordinates in the bottom half (> 180).
    const gradientCalls = (context.createLinearGradient as jest.Mock).mock
      .calls as number[][];
    const bottomFeatherCalls = gradientCalls.filter(
      (call) => call[1] > 180 && call[3] > 180,
    );
    expect(bottomFeatherCalls.length).toBeGreaterThan(0);
  });
});
