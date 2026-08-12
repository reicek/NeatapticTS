import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION } from '../constants';
import type {
  ControlledEnemy,
  EnemyControllerState,
} from '../../scripts/enemy-controller';
import type { GameState } from '../host/game/types';

const loadModule = (path: string): Promise<unknown> => import(path);

interface MockWorkerGlobal {
  postMessage: jest.Mock;
  requestAnimationFrame: jest.Mock;
  onmessage: ((event: MessageEvent) => void) | null;
}

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

function sendInitMessage(tier: 'worker' | 'cpu' | 'gpu') {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: 'init',
        tier,
        version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
        mapSeed: 42,
      },
    } as unknown as MessageEvent);
  }
}

function sendSimStateMessage() {
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
          cameraYaw: 0.25,
          mapSeed: 42,
        },
      },
    } as unknown as MessageEvent);
  }
}

describe('Neatenstein derez pruning fix', () => {
  beforeEach(() => {
    jest.resetModules();
  });

  describe('AC-801-S02-001: enemy with health=0 and deRezElapsedMs < 700 is NOT pruned', () => {
    it('advances deRezElapsedMs over ticks instead of resetting to 0', async () => {
      const workerModule = (await loadModule('./display.worker.ts')) as {
        __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
      };

      sendInitMessage('cpu');
      // Build up enemies with 3 sim state ticks
      for (let i = 0; i < 3; i += 1) {
        sendSimStateMessage();
      }
      const before = workerModule.__testOnlyGetEnemyControllerState?.();
      expect(before?.enemies.length).toBeGreaterThan(0);

      // Mock gameTick to kill enemy[0] (set health=0 only, matching real
      // gameTick behavior which does not set active=false on death).
      const tickModule = await import('../host/game/tick');
      const gameTickSpy = jest.spyOn(tickModule, 'gameTick').mockImplementation(
        (state: GameState) =>
          ({
            ...state,
            enemies: state.enemies.map((enemy, index) =>
              index === 0 ? { ...enemy, health: 0 } : enemy,
            ),
          }) as GameState,
      );

      // Send 2 ticks with the dead enemy
      sendSimStateMessage();
      sendSimStateMessage();

      const after = workerModule.__testOnlyGetEnemyControllerState?.();
      gameTickSpy.mockRestore();

      // Find the dead enemy in the controller roster
      const deadEnemy = after?.enemies.find(
        (enemy: ControlledEnemy) => enemy.health <= 0,
      );
      expect(deadEnemy).toBeDefined();
      // With the fix, deRezElapsedMs should advance over ticks instead of
      // resetting to 0 every tick due to premature pruning + re-add.
      expect(deadEnemy!.deRezElapsedMs).toBeGreaterThan(0);
    });
  });

  describe('AC-801-S02-002: enemy with health=0 and deRezElapsedMs >= 700 IS pruned', () => {
    it('removes dead enemy from controller roster after de-rez animation completes', async () => {
      const workerModule = (await loadModule('./display.worker.ts')) as {
        __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
      };

      sendInitMessage('cpu');
      // Build up enemies with 3 sim state ticks
      for (let i = 0; i < 3; i += 1) {
        sendSimStateMessage();
      }
      const before = workerModule.__testOnlyGetEnemyControllerState?.();
      expect(before?.enemies.length).toBeGreaterThan(0);

      // Mock gameTick to kill enemy[0] (set health=0 only, matching real
      // gameTick behavior which does not set active=false on death).
      const tickModule = await import('../host/game/tick');
      const gameTickSpy = jest.spyOn(tickModule, 'gameTick').mockImplementation(
        (state: GameState) =>
          ({
            ...state,
            enemies: state.enemies.map((enemy, index) =>
              index === 0 ? { ...enemy, health: 0 } : enemy,
            ),
          }) as GameState,
      );

      // Send enough ticks for deRezElapsedMs to reach 700ms at 16ms/tick.
      // 700 / 16 = 43.75, so 45 ticks is enough for the de-rez animation
      // to complete if deRezElapsedMs advances properly.
      for (let i = 0; i < 45; i += 1) {
        sendSimStateMessage();
      }

      const after = workerModule.__testOnlyGetEnemyControllerState?.();
      gameTickSpy.mockRestore();

      // After de-rez completes (deRezElapsedMs >= 700), the dead enemy
      // should be permanently pruned from the controller roster.
      const deadEnemy = after?.enemies.find(
        (enemy: ControlledEnemy) => enemy.health <= 0,
      );
      expect(deadEnemy).toBeUndefined();
    });
  });
});
