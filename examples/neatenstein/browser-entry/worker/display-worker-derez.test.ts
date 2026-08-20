import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import type {
  ControlledEnemy,
  EnemyControllerState,
} from '../shared/enemy-controller';
import type { GameState } from '../host/game/types';

import {
  loadModule,
  sendInitMessage,
  sendSimStateMessage,
} from './display.worker.test-helpers';

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

// ---------------------------------------------------------------------------
// A2 Fix 4 — Zero-timestep pass conditional skip
//
// display.worker.sim.utils.ts:373-379 runs a second updateEnemyController
// call unconditionally.  The fix makes it conditional: skip when
// completedDeRezIndices.length === 0, and reuse the distance map from pass 1.
// ---------------------------------------------------------------------------

describe('A2 Fix 4: Zero-timestep pass conditional skip', () => {
  it('exports __testOnlyGetZeroTimestepPassSkipped as a function from sim utils', async () => {
    const mod = (await loadModule('./display.worker.sim.utils.ts')) as Record<
      string,
      unknown
    >;
    expect(typeof mod.__testOnlyGetZeroTimestepPassSkipped).toBe('function');
  });

  it('returns true after a sim step with no completed de-rez', async () => {
    // Load the worker module and trigger sim steps via the message flow
    // so the diagnostic reflects actual runtime behavior.
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');
    // Send one tick to initialize the game state and controller.
    sendSimStateMessage();

    // Mock gameTick to return the state unchanged (no enemy spawning,
    // no deaths) so that the enemy count stays stable between the first
    // updateEnemyController call and the skipPass check.
    const tickModule = await import('../host/game/tick');
    const gameTickSpy = jest
      .spyOn(tickModule, 'gameTick')
      .mockImplementation((gameState: GameState) => gameState);

    // Send another tick — with gameTick mocked to return unchanged state,
    // the controller count matches the game count, no de-rez completions,
    // no dead enemies → skipPass = true.
    sendSimStateMessage();
    gameTickSpy.mockRestore();

    const { __testOnlyGetZeroTimestepPassSkipped } = (await loadModule(
      './display.worker.sim.utils.ts',
    )) as {
      __testOnlyGetZeroTimestepPassSkipped: () => boolean;
    };
    // When no enemies complete de-rez and the controller is synced with
    // the game state, the zero-timestep pass should be skipped (skipPass
    // = true).
    expect(__testOnlyGetZeroTimestepPassSkipped()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// A2 Fix 6 — De-rez pruning single-pass with Set
//
// display.worker.sim.utils.ts:333-362 runs 3 chained .map().filter().map()
// every tick.  The fix replaces this with a single in-place loop using a Set
// for completed indices, and only runs when completedDeRezIndices.length > 0.
// ---------------------------------------------------------------------------

describe('A2 Fix 6: De-rez pruning single-pass with Set', () => {
  it('exports __testOnlyGetDeRezPruningUsedSinglePass as a function from sim utils', async () => {
    const mod = (await loadModule('./display.worker.sim.utils.ts')) as Record<
      string,
      unknown
    >;
    expect(typeof mod.__testOnlyGetDeRezPruningUsedSinglePass).toBe('function');
  });

  it('returns true after a sim step (pruning used single-pass Set approach)', async () => {
    // Load the worker module and trigger a sim step via the message flow
    // so the diagnostic reflects actual runtime behavior.
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');
    sendSimStateMessage();

    const { __testOnlyGetDeRezPruningUsedSinglePass } = (await loadModule(
      './display.worker.sim.utils.ts',
    )) as {
      __testOnlyGetDeRezPruningUsedSinglePass: () => boolean;
    };
    // The de-rez pruning should use a single in-place loop with a Set
    // for completed indices, not chained .map().filter().map().
    expect(__testOnlyGetDeRezPruningUsedSinglePass()).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// A2 Fix 8 — runSimStep mutable internal state (no deep clone)
//
// runSimStep currently uses spread/map chains to clone state.  The fix
// mutates internal state in place to eliminate per-tick allocation.
// ---------------------------------------------------------------------------

describe('A2 Fix 8: runSimStep mutable internal state', () => {
  it('exports __testOnlyGetSimStepCloneCount as a function from sim utils', async () => {
    const mod = (await loadModule('./display.worker.sim.utils.ts')) as Record<
      string,
      unknown
    >;
    expect(typeof mod.__testOnlyGetSimStepCloneCount).toBe('function');
  });
});
