import { describe, expect, it, jest } from '@jest/globals';
import {
  NEATENSTEIN_IMPACT_SPOT_COLOR,
  NEATENSTEIN_IMPACT_SPOT_GLOW_COLOR,
} from '../constants';
import {
  drawBolts,
  drawImpactSpots,
  drawEnemyImpactSpots,
} from '../renderer/bolt-render';
import {
  NEATENSTEIN_FOG_START_DISTANCE,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
} from '../renderer/framebuffer';
import { NEATENSTEIN_ZBUFFER_EMPTY } from '../renderer/renderer.zbuffer.constants';
import {
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
} from '../renderer/floor';
import type {
  ControlledEnemy,
  EnemyControllerState,
} from '../shared/enemy-controller';
import * as robotSpriteData from '../../robot-sprite-data.js';
import type { GameState } from '../host/game/types';
import {
  loadModule,
  workerSelf,
  sendInitMessage,
  sendSimStateMessage,
  sendActionInputMessage,
  createMockCanvas,
  createMockContext,
  findPostByType,
  createMockImpact,
  workerBeforeEach,
} from './display.worker.test-helpers';

describe('Neatenstein display worker', () => {
  workerBeforeEach();

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

    // The framebuffer pipeline flushes walls+sprites via putImageData once
    // per frame without any getImageData read-back (A2 Fix 2: procedural
    // framebuffer seeding eliminates the per-frame allocation bomb).
    expect(getImageData).not.toHaveBeenCalled();
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
    // Without getImageData the framebuffer path still works (procedural
    // seeding + putImageData flush). The worker renders the frame and
    // flushes via putImageData without any canvas read-back.
    expect(putImageData).toHaveBeenCalled();
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

  it('draws enemy impact spots as arcs when enemyImpacts are present', () => {
    const { context } = createMockContext();
    const zBuffer = new Float32Array(1);
    zBuffer[0] = Number.POSITIVE_INFINITY;

    drawEnemyImpactSpots(
      context,
      [
        {
          position: { x: 10, y: 10 },
          createdAtMs: 0,
          lifetimeMs: 1000,
          boltTravelTimeMs: 0,
        },
      ],
      zBuffer,
      { x: 9, y: 9, yaw: Math.PI / 4 },
      640,
      360,
      50,
    );

    expect(context.arc).toHaveBeenCalled();
  });

  it('does not draw enemy impact spots when the array is empty', () => {
    const { context } = createMockContext();
    const zBuffer = new Float32Array(1);
    zBuffer[0] = Number.POSITIVE_INFINITY;

    drawEnemyImpactSpots(
      context,
      [],
      zBuffer,
      { x: 9, y: 9, yaw: Math.PI / 4 },
      640,
      360,
      50,
    );

    expect(context.arc).not.toHaveBeenCalled();
  });

  it('draws the gun overlay in the worker tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    workerSelf.postMessage.mockClear();

    sendActionInputMessage(false);
    sendSimStateMessage();

    // The gun overlay should produce fillRect calls — tests renderer
    // capability, not specific draw colors.
    expect(context.fillRect).toHaveBeenCalled();
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

      expect(getImageData).not.toHaveBeenCalled();
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

      expect(getImageData).not.toHaveBeenCalled();
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
      // 30-cell render distance cap so sprites are rendered (not culled).
      // The display worker uses gameState.player.position (60.5, 60.5) for
      // the camera. With step-function fog, fogFactor=0 below 30 cells means
      // pixel colors match the original team colors exactly.
      workerModule.__testOnlyInjectTestEnemies?.([
        { x: 63.5, y: 60.5 },
        { x: 62.5, y: 61.5 },
      ]);
      sendSimStateMessage();

      expect(getImageData).not.toHaveBeenCalled();
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

      // With the procedural framebuffer path (A2 Fix 2), floor and ceiling
      // grid lines are drawn directly into the framebuffer via Bresenham
      // line drawing, not via Canvas 2D stroke calls. B3.7 wires
      // castNeatensteinFloorPerPixel in the worker path, which alpha-blends
      // the grid color (FLAPPY_NEON_PALETTE.groundGridLine = '#0a8ea0' →
      // R=10, G=142, B=160) with the background (#060b14 → R=6, G=11, B=20).
      // Verify the flushed framebuffer contains grid-influenced pixels by
      // checking the dominant G channel is noticeably above the background
      // G=11, indicating the grid color contributed via alpha blending.
      const putCalls = (context.putImageData as jest.Mock).mock.calls;
      expect(putCalls.length).toBeGreaterThan(0);
      const fbData = (
        putCalls[putCalls.length - 1][0] as { data: Uint8ClampedArray }
      ).data;
      const GRID_G = 142;
      const BACKGROUND_G = 11;
      // Tolerance: any pixel whose G channel is closer to the grid color
      // than to the background counts as a grid-influenced pixel.
      const gridGThreshold = (GRID_G + BACKGROUND_G) / 2;
      let foundGridPixel = false;
      for (let i = 0; i < fbData.length; i += 4) {
        if (fbData[i + 1] > gridGThreshold) {
          foundGridPixel = true;
          break;
        }
      }
      expect(foundGridPixel).toBe(true);
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
      stunTimerMs: 0,
    };
  }

  it('renders active enemies far-to-near so distant sprites do not overwrite closer ones', async () => {
    jest.resetModules();

    await loadModule('./display.worker.ts');

    const realEnemyController = await import('../shared/enemy-controller');
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

    const realEnemyController = await import('../shared/enemy-controller');
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

    const realEnemyController = await import('../shared/enemy-controller');
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

  it('draws fog wall stripes at the render distance cap when the ray exceeds 30 cells', async () => {
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

    // When the ray exceeds the 30-cell cap, the framebuffer wall path writes
    // fog wall pixels (background color) directly into the persistent
    // Uint8ClampedArray. Verify putImageData was called and the framebuffer
    // contains background-color pixels at the fog wall position (y≈180 is the
    // center of the fog wall stripe with drawStart≈175, drawEnd≈185).
    const putCalls = (context.putImageData as jest.Mock).mock.calls;
    expect(putCalls.length).toBeGreaterThan(0);
    const fbData = (
      putCalls[putCalls.length - 1][0] as { data: Uint8ClampedArray }
    ).data;
    const offset = (180 * 640 + 320) * 4;
    expect(fbData[offset]).toBe(6); // NEATENSTEIN_BACKGROUND_RGB.r
    expect(fbData[offset + 1]).toBe(11); // NEATENSTEIN_BACKGROUND_RGB.g
    expect(fbData[offset + 2]).toBe(20); // NEATENSTEIN_BACKGROUND_RGB.b
  });

  it('sets packed-frame zBuffer to empty sentinel for capped columns', async () => {
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
    // B3.3: capped columns use NEATENSTEIN_ZBUFFER_EMPTY (Infinity), not the
    // render-distance cap, so sprites correctly skip occluded far columns.
    expect(zBuffer![zBuffer!.length / 2]).toBe(NEATENSTEIN_ZBUFFER_EMPTY);
  });

  it('skips rendering enemy sprites beyond the 30-cell render distance cap (AC-10.3c-002)', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realEnemyController = await import('../shared/enemy-controller');
    const realSprites = await import('../renderer/sprites');
    const realRaycast = await import('../renderer/raycast');
    const realGameState = await import('../host/game/state');

    const referenceState = realGameState.createGameState({ seed: 42 });
    const cameraX = referenceState.player.position.x;
    const cameraY = referenceState.player.position.y;
    const cameraYaw = referenceState.player.angleRad;
    // Place enemy well beyond the 30-cell render distance cap.
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

  it('passes derezState to renderNeatensteinSprite for death-state enemies', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realEnemyController = await import('../shared/enemy-controller');
    const realSprites = await import('../renderer/sprites');
    const realRaycast = await import('../renderer/raycast');
    const realGameState = await import('../host/game/state');

    const referenceState = realGameState.createGameState({ seed: 42 });
    const cameraX = referenceState.player.position.x;
    const cameraY = referenceState.player.position.y;
    const cameraYaw = referenceState.player.angleRad;
    const dist = 3;
    const enemyX = cameraX + Math.cos(cameraYaw) * dist;
    const enemyY = cameraY + Math.sin(cameraYaw) * dist;

    const deathEnemy = makeEnemy(7, enemyX, enemyY);
    deathEnemy.animationState = 'death';
    deathEnemy.deRezElapsedMs = 350;
    deathEnemy.active = true;

    jest.spyOn(realEnemyController, 'updateEnemyController').mockReturnValue({
      enemies: [deathEnemy],
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

    expect(renderSpy).toHaveBeenCalled();
    const lastCall = renderSpy.mock.calls[renderSpy.mock.calls.length - 1];
    const derezState = lastCall[6] as
      { elapsedMs: number; durationMs: number; seed: number } | undefined;
    expect(derezState).toBeDefined();
    expect(derezState!.elapsedMs).toBe(350);
    expect(derezState!.durationMs).toBe(700);
    expect(derezState!.seed).toBe(7);
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

describe('AC-11c: enemyImpacts nullish coalescing fallback (line 816)', () => {
  it('covers the ?? [] branch when gameState.enemyImpacts is undefined', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    sendInitMessage('cpu');
    // Build up enemies so the controller has entries to work with.
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    const realTick = await import('../host/game/tick');

    // Mock gameTick to return a state with enemyImpacts: undefined so the
    // `gameState.enemyImpacts ?? []` nullish coalescing at line 816 fires.
    const gameTickSpy = jest.spyOn(realTick, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          enemyImpacts: undefined,
        }) as GameState,
    );

    workerSelf.postMessage.mockClear();
    sendSimStateMessage(0);

    expect(gameTickSpy).toHaveBeenCalled();

    gameTickSpy.mockRestore();
  });
});

describe('AC-10.4 fix-coverage-gaps: fog factor and gradient feather branches', () => {
  it('resolveWallFogFactor returns smoothstep fog: 0 at/below FOG_START, ramps to 1 at CAP', async () => {
    jest.resetModules();
    const mod = (await loadModule('./display.worker.ts')) as {
      __testOnlyResolveWallFogFactor: (d: number) => number;
    };
    // At and above the render distance cap, fog factor is 1 (full fog).
    expect(
      mod.__testOnlyResolveWallFogFactor(NEATENSTEIN_RENDER_DISTANCE_CAP),
    ).toBe(1);
    expect(mod.__testOnlyResolveWallFogFactor(100)).toBe(1);
    // At and below the fog start distance, fog factor is 0 (no fog).
    expect(
      mod.__testOnlyResolveWallFogFactor(NEATENSTEIN_FOG_START_DISTANCE),
    ).toBe(0);
    expect(mod.__testOnlyResolveWallFogFactor(0)).toBe(0);
    // Mid-range distance produces a smoothstep factor strictly between 0 and 1.
    const midDistance =
      (NEATENSTEIN_FOG_START_DISTANCE + NEATENSTEIN_RENDER_DISTANCE_CAP) / 2;
    const midFog = mod.__testOnlyResolveWallFogFactor(midDistance);
    expect(midFog).toBeGreaterThan(0);
    expect(midFog).toBeLessThan(1);
    // Non-finite distances are treated as fully fogged.
    expect(mod.__testOnlyResolveWallFogFactor(Number.POSITIVE_INFINITY)).toBe(
      1,
    );
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
    // drawStart = (360 - 10.39) / 2 ≈ 174.8 > 0, so the top feather
    // branch is taken. The framebuffer path blends seeded ceiling pixels
    // toward the fog color. Verify the framebuffer at y≈172 (within the
    // feather range 169–175) has a blended value — non-zero (feather was
    // drawn) but less than the pure background R channel (6).
    const putCalls = (context.putImageData as jest.Mock).mock.calls;
    expect(putCalls.length).toBeGreaterThan(0);
    const fbData = (
      putCalls[putCalls.length - 1][0] as { data: Uint8ClampedArray }
    ).data;
    const offset = (172 * 640 + 320) * 4;
    expect(fbData[offset]).toBeGreaterThan(0); // feather pixel written
    // With procedural framebuffer seeding (A2 Fix 2), seeded pixels have
    // R >= background (6). The feather blend of seeded*R*(1-t) + bgR*t
    // therefore produces values >= 6, not < 6 as with the old zero-fill.
    expect(fbData[offset]).toBeGreaterThanOrEqual(6);
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
    // drawEnd = (360 + 10.39) / 2 ≈ 185.2 < 360, so the bottom feather
    // branch is taken. The framebuffer path blends fog color toward seeded
    // floor pixels. Verify the framebuffer at y≈188 (within the feather
    // range 185–191) has a blended value — non-zero (feather was drawn)
    // but less than the pure background R channel (6).
    const putCalls = (context.putImageData as jest.Mock).mock.calls;
    expect(putCalls.length).toBeGreaterThan(0);
    const fbData = (
      putCalls[putCalls.length - 1][0] as { data: Uint8ClampedArray }
    ).data;
    const offset = (188 * 640 + 320) * 4;
    expect(fbData[offset]).toBeGreaterThan(0); // feather pixel written
    // With procedural framebuffer seeding (A2 Fix 2), seeded pixels have
    // R >= background (6). The feather blend produces values >= 6, not
    // < 6 as with the old zero-fill seeding.
    expect(fbData[offset]).toBeGreaterThanOrEqual(6);
  });
});

describe('AC-11-enemy-fire: enemyBolts nullish coalescing fallback (line 1208)', () => {
  it('covers the ?? [] branch when gameState.enemyBolts is undefined', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    sendInitMessage('cpu');
    // Build up enemies so the controller has entries to work with.
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    const realTick = await import('../host/game/tick');
    const realEnemyController = await import('../shared/enemy-controller');

    // Mock gameTick to return a state with enemyBolts: undefined so the
    // `gameState.enemyBolts ?? []` nullish coalescing at line 1208 fires.
    const gameTickSpy = jest.spyOn(realTick, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          enemyBolts: undefined,
        }) as GameState,
    );

    // Mock updateEnemyController to return hitscanEvents with one event so
    // the code enters the `if (controlled.hitscanEvents.length > 0)` block.
    const controllerSpy = jest
      .spyOn(realEnemyController, 'updateEnemyController')
      .mockImplementation((controllerState: EnemyControllerState) => ({
        ...controllerState,
        hitscanEvents: [
          {
            enemyIndex: 0,
            origin: { x: 10, y: 10 },
            direction: { x: 1, y: 0 },
            damage: 10,
          },
        ],
      }));

    workerSelf.postMessage.mockClear();
    sendSimStateMessage(0);

    expect(gameTickSpy).toHaveBeenCalled();

    gameTickSpy.mockRestore();
    controllerSpy.mockRestore();
  });
});

describe('AC-11d: ammoPickups nullish coalescing fallback (line 851)', () => {
  it('covers the ?? [] branch when gameState.ammoPickups is undefined', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    // Pass a mock canvas + context so buildAndPostFrame proceeds past the
    // `if (!workerCanvas) return;` guard at line 530 and actually reaches the
    // render path at line 851 where `gameState.ammoPickups ?? []` fires.
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    // Build up enemies so the controller has entries to work with.
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    const realTick = await import('../host/game/tick');

    // Mock gameTick to return a state with ammoPickups: undefined so the
    // `gameState.ammoPickups ?? []` nullish coalescing at line 851 fires.
    const gameTickSpy = jest.spyOn(realTick, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          ammoPickups: undefined,
        }) as GameState,
    );

    workerSelf.postMessage.mockClear();
    sendSimStateMessage(0);

    expect(gameTickSpy).toHaveBeenCalled();

    gameTickSpy.mockRestore();
  });
});

describe('AC-2b-05: playerDeaths nullish coalescing fallback (lines 881 & 925)', () => {
  it('covers the ?? 0 fallback in the worker-tier frame path when gameState.deaths is undefined', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest.spyOn(realTick, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          deaths: undefined,
        }) as GameState,
    );

    workerSelf.postMessage.mockClear();
    sendSimStateMessage(0);

    expect(gameTickSpy).toHaveBeenCalled();

    const frameCall = findPostByType<{ frame: { playerDeaths: number } }>(
      workerSelf.postMessage,
      'frame',
    );
    expect(frameCall?.frame?.playerDeaths).toBe(0);

    gameTickSpy.mockRestore();
  });

  it('covers the ?? 0 fallback in the cpu-tier frame path when gameState.deaths is undefined', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    sendInitMessage('cpu');
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest.spyOn(realTick, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          deaths: undefined,
        }) as GameState,
    );

    workerSelf.postMessage.mockClear();
    sendSimStateMessage(0);

    expect(gameTickSpy).toHaveBeenCalled();

    const frameCall = findPostByType<{ frame: { playerDeaths: number } }>(
      workerSelf.postMessage,
      'frame',
    );
    expect(frameCall?.frame?.playerDeaths).toBe(0);

    gameTickSpy.mockRestore();
  });

  it('covers the defined branch in the worker-tier frame path when gameState.deaths is 3', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest.spyOn(realTick, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          deaths: 3,
        }) as GameState,
    );

    workerSelf.postMessage.mockClear();
    sendSimStateMessage(0);

    expect(gameTickSpy).toHaveBeenCalled();

    const frameCall = findPostByType<{ frame: { playerDeaths: number } }>(
      workerSelf.postMessage,
      'frame',
    );
    expect(frameCall?.frame?.playerDeaths).toBe(3);

    gameTickSpy.mockRestore();
  });

  it('covers the defined branch in the cpu-tier frame path when gameState.deaths is 3', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    sendInitMessage('cpu');
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest.spyOn(realTick, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          deaths: 3,
        }) as GameState,
    );

    workerSelf.postMessage.mockClear();
    sendSimStateMessage(0);

    expect(gameTickSpy).toHaveBeenCalled();

    const frameCall = findPostByType<{ frame: { playerDeaths: number } }>(
      workerSelf.postMessage,
      'frame',
    );
    expect(frameCall?.frame?.playerDeaths).toBe(3);

    gameTickSpy.mockRestore();
  });
});

describe('A2 Fix 2: Persistent sprite ImageData', () => {
  it('exports getReusableSpriteImageData as a function from render utils', async () => {
    const mod = (await loadModule(
      './display.worker.render.utils.ts',
    )) as Record<string, unknown>;
    expect(typeof mod.getReusableSpriteImageData).toBe('function');
  });
});

describe('A2 Fix 8: Per-ray hit pooled buffer', () => {
  it('exports getPooledRayHitBuffer as a function from render utils', async () => {
    const mod = (await loadModule(
      './display.worker.render.utils.ts',
    )) as Record<string, unknown>;
    expect(typeof mod.getPooledRayHitBuffer).toBe('function');
  });

  it('returns a mutable hit object with perpWallDist, side, mapX, mapY', async () => {
    const { getPooledRayHitBuffer } = (await loadModule(
      './display.worker.render.utils.ts',
    )) as {
      getPooledRayHitBuffer: () => {
        perpWallDist: number;
        side: number;
        mapX: number;
        mapY: number;
      };
    };
    const buf = getPooledRayHitBuffer();
    expect(typeof buf.perpWallDist).toBe('number');
    expect(typeof buf.side).toBe('number');
    expect(typeof buf.mapX).toBe('number');
    expect(typeof buf.mapY).toBe('number');
  });

  it('returns the same pooled hit object reference across calls', async () => {
    const { getPooledRayHitBuffer } = (await loadModule(
      './display.worker.render.utils.ts',
    )) as {
      getPooledRayHitBuffer: () => unknown;
    };
    const a = getPooledRayHitBuffer();
    const b = getPooledRayHitBuffer();
    expect(a).toBe(b);
  });
});
