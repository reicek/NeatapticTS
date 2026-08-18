import { describe, expect, it, jest } from '@jest/globals';
import { NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION } from '../constants';
import type {
  ControlledEnemy,
  EnemyControllerState,
} from '../shared/enemy-controller';
import type { GameState } from '../host/game/types';
import {
  loadModule,
  workerSelf,
  sendInitMessage,
  sendSimStateMessage,
  sendInputMessage,
  sendMovementInputMessage,
  sendActionInputMessage,
  sendRawMessage,
  findPostByType,
  workerBeforeEach,
} from './display.worker.test-helpers';

describe('Neatenstein display worker', () => {
  workerBeforeEach();

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
      await import('../shared/enemy-controller');
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

  it('does not post a frame when simState arrives before init', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendSimStateMessage();

    expect(findPostByType(workerSelf.postMessage, 'frame')).toBeUndefined();
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
});

describe('AC-10.2c: worker simState collision sync', () => {
  it('calls updateEnemyController before gameTick and syncs controlled state into gameState.enemies', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realTick = await import('../host/game/tick');
    const realEnemyController = await import('../shared/enemy-controller');

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
          stunTimerMs: 0,
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
    const realEnemyController = await import('../shared/enemy-controller');

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
  it('clamps timestep to NEATENSTEIN_FIXED_TIMESTEP_MS (16ms) regardless of rAF deltaMs (AC-018)', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');

    const realTick = await import('../host/game/tick');
    // Spy without mocking — let the real gameTick run; we only need to
    // capture the timestepMs argument (4th positional, index 3).
    const gameTickSpy = jest.spyOn(realTick, 'gameTick');

    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();

    // Send simState with a large deltaMs (32ms). After the AC-018 clamp,
    // the worker must always pass NEATENSTEIN_FIXED_TIMESTEP_MS (16) to
    // gameTick regardless of the incoming rAF delta.
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
    expect(timestepMs).toBe(16);

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
      await import('../shared/enemy-controller');

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
    // Step 03 derez fix: the enemy with health: null is treated as dead via
    // `(null ?? 0) <= 0`, but it is NOT pruned immediately.  It stays in the
    // roster with active: true until deRezElapsedMs >= 700ms.  Since the
    // passthrough mock prevents deRezElapsedMs from advancing, the enemy
    // remains — exercising the `live?.health ?? 0` nullish branch at
    // line 1293 (the keep path, not the prune path).
    expect(after?.enemies.length ?? 0).toBe(beforeCount);
    // The nullish coalescing `live?.health ?? 0` should produce health: 0.
    expect(after?.enemies[0]?.health).toBe(0);

    gameTickSpy.mockRestore();
    controllerSpy.mockRestore();
  });

  it('covers the ?? 0 nullish branch at line 1290 when derez completes with null health', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
    };

    sendInitMessage('cpu');
    // Build up enemies so the controller has entries to work with.
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }
    const before = workerModule.__testOnlyGetEnemyControllerState?.();
    const beforeCount = before?.enemies.length ?? 0;
    expect(beforeCount).toBeGreaterThan(0);

    const tickModule = await import('../host/game/tick');
    const enemyControllerModule =
      await import('../shared/enemy-controller');

    // Mock gameTick to return enemy[0] with health: null, exercising the
    // `(live.health ?? 0)` nullish fallback at display.worker.ts:1290.
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

    // Mock updateEnemyController to set deRezElapsedMs >= 700 so the derez
    // completion path (line 1284-1291) is taken, exercising the
    // `health: live?.health ?? 0` nullish branch at line 1290.
    const controllerSpy = jest
      .spyOn(enemyControllerModule, 'updateEnemyController')
      .mockImplementation((state: EnemyControllerState) => ({
        ...state,
        enemies: state.enemies.map((enemy, i) =>
          i === 0 ? { ...enemy, deRezElapsedMs: 700 } : enemy,
        ),
      }));

    sendSimStateMessage();

    expect(gameTickSpy).toHaveBeenCalled();
    const after = workerModule.__testOnlyGetEnemyControllerState?.();
    // With deRezElapsedMs >= 700, the enemy is pruned (active: false,
    // filtered out) — `after?.enemies.length < beforeCount`.
    expect(after?.enemies.length ?? 0).toBeLessThan(beforeCount);

    gameTickSpy.mockRestore();
    controllerSpy.mockRestore();
  });
});

describe('AC-017: wave-clear detection (allEnemiesCleared flag)', () => {
  it('sets allEnemiesCleared when enemies transition from present to empty', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetAllEnemiesCleared?(): boolean;
      __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
    };

    sendInitMessage('cpu');

    // Inject enemies so the roster is non-empty.
    workerModule.__testOnlyInjectTestEnemies?.([
      { x: 13.5, y: 12.5 },
      { x: 11.5, y: 12.5 },
    ]);
    expect(
      workerModule.__testOnlyGetEnemyControllerState?.()?.enemies.length ?? 0,
    ).toBeGreaterThan(0);

    // Advance a few ticks to let enemies exist.
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    // Flag should not be set while enemies are still alive.
    expect(workerModule.__testOnlyGetAllEnemiesCleared?.()).toBe(false);

    // Now mock gameTick to return zero enemies, simulating wave clear.
    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest.spyOn(realTick, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          enemies: [],
        }) as GameState,
    );

    sendSimStateMessage();

    expect(gameTickSpy).toHaveBeenCalled();
    expect(workerModule.__testOnlyGetAllEnemiesCleared?.()).toBe(true);

    gameTickSpy.mockRestore();
  });

  it('resets allEnemiesCleared when enemies appear again (new wave)', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetAllEnemiesCleared?(): boolean;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
    };

    sendInitMessage('cpu');

    // Inject enemies, then clear them.
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 13.5, y: 12.5 }]);
    for (let i = 0; i < 2; i += 1) {
      sendSimStateMessage();
    }

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest.spyOn(realTick, 'gameTick').mockImplementation(
      (state: GameState) =>
        ({
          ...state,
          enemies: [],
        }) as GameState,
    );
    sendSimStateMessage();
    expect(workerModule.__testOnlyGetAllEnemiesCleared?.()).toBe(true);

    // Re-inject enemies (new wave spawn).
    gameTickSpy.mockRestore();
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 14.5, y: 12.5 }]);
    sendSimStateMessage();
    expect(workerModule.__testOnlyGetAllEnemiesCleared?.()).toBe(false);
  });
});

describe('wave-clear guard coverage', () => {
  it('covers the alive-enemy predicate branches for mixed enemy states', async () => {
    jest.resetModules();

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyGetGameState?(): GameState | null;
    };

    const makeControlledEnemy = (
      index: number,
      health: number,
      active: boolean,
    ) => ({
      index,
      position: { x: 13.5 + index, y: 12.5 },
      health,
      yawRad: 0,
      animationState: 'idle' as const,
      ammo: 10,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      goal: { x: 13.5 + index, y: 12.5 },
      isRecovering: false,
      lastFireMs: -Infinity,
      path: [],
      pathIndex: 0,
      pathRepathTimerMs: 0,
      recalcTimerMs: 0,
      state: 'idle' as const,
      hitscanEvents: [],
    });

    const realEnemyController = await import('../shared/enemy-controller');
    const controllerSpy = jest
      .spyOn(realEnemyController, 'updateEnemyController')
      .mockReturnValue({
        enemies: [
          // Both predicate branches true (alive).
          makeControlledEnemy(0, 100, true),
          // health > 0 but inactive.
          makeControlledEnemy(1, 100, false),
          // inactive and zero health.
          makeControlledEnemy(2, 0, false),
        ],
        hitscanEvents: [],
      } as unknown as ReturnType<
        typeof realEnemyController.updateEnemyController
      >);

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest
      .spyOn(realTick, 'gameTick')
      .mockImplementation((state: GameState) => state);

    sendInitMessage('cpu');
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 13.5, y: 12.5 }]);

    expect(() =>
      sendSimStateMessage(0.25, { humanMode: 'auto' }),
    ).not.toThrow();

    expect(controllerSpy).toHaveBeenCalled();
    const afterState = workerModule.__testOnlyGetGameState?.();
    expect(afterState?.enemies.length).toBeGreaterThan(0);

    controllerSpy.mockRestore();
    gameTickSpy.mockRestore();
  });
});
