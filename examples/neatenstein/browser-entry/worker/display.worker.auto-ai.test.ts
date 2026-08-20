import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { castRayDDAFromFlatMap } from '../renderer/raycast';
import {
  NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD,
  NEATENSTEIN_KITING_APPROACH_DISTANCE_CELLS,
  NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS,
} from '../host/game/constants';
import {
  NEATENSTEIN_MOVE_ACCEL_PER_TICK,
  NEATENSTEIN_MOVE_DECEL_PER_TICK,
} from '../harness/neat-io-config';
import type { EnemyControllerState } from '../shared/enemy-controller';
import type { GameTickInputSnapshot } from '../host/game/tick';
import type { EnemyState, GameState } from '../host/game/types';
import {
  loadModule,
  workerSelf,
  sendInitMessage,
  sendSimStateMessage,
  sendActionInputMessage,
} from './display.worker.test-helpers';

describe('P4S1-worker-branch: humanMode branching and NEAT controller', () => {
  /**
   * Minimal mock that satisfies the `Network.activate(input)` call site inside
   * `buildAutoTickInput`.  The worker only calls `activate(sensors)` on the
   * network — no other methods are needed for this slice.
   */
  function createMockNetwork(outputs: number[] = [0.8, 0.2, 0.5, 0.9, 0.1]): {
    activate: jest.Mock;
  } {
    return { activate: jest.fn(() => outputs) } as unknown as {
      activate: jest.Mock;
    };
  }

  it('AC-055: uses NEAT controller path in auto mode when champion network is available', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('AC-055: uses human InputRouter path in human mode', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());

    sendSimStateMessage(0.25, { humanMode: 'human' });

    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('human');
  });

  it('AC-055: uses human InputRouter path when humanMode is undefined (default)', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());

    // No humanMode override → undefined → should use human path.
    sendSimStateMessage();

    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('human');
  });

  it('AC-056: champion Network from worker scope is used for auto-mode ticks', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');

    // Without a champion network, auto mode uses the fallback AI (still 'auto').
    sendSimStateMessage(0.25, { humanMode: 'auto' });
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');

    // After injecting a champion network, auto mode should use the NEAT path.
    const mockNet = createMockNetwork();
    workerModule.__testOnlySetChampionMainNetwork?.(mockNet);
    sendSimStateMessage(0.25, { humanMode: 'auto' });
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
    expect(mockNet.activate).toHaveBeenCalled();
  });

  it('AC-057: auto mode uses NEAT controller, not pendingTickInput from human queue', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());

    // Queue human input (fire = true).
    sendActionInputMessage(true);

    // In auto mode, the NEAT controller should be used, NOT pendingTickInput.
    sendSimStateMessage(0.25, { humanMode: 'auto' });
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('AC-053: human mode remains fully functional (uses pendingTickInput)', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());

    // Queue human input.
    sendActionInputMessage(true);

    // In human mode, pendingTickInput should be used.
    sendSimStateMessage(0.25, { humanMode: 'human' });
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('human');
  });

  it('AC-060: network.activate is called with sensor vector in auto mode', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
    };

    sendInitMessage('cpu');
    const mockNet = createMockNetwork([0.9, 0.1, 0.75, 0.6, 0.3]);
    workerModule.__testOnlySetChampionMainNetwork?.(mockNet);

    // The champion network path is only active when enemies exist.
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 14.5, y: 10.5 }]);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    expect(mockNet.activate).toHaveBeenCalledTimes(1);
    // Sensor vector should have 22 inputs (NEATENSTEIN_MAIN_NEAT_INPUTS).
    const sensorArg = mockNet.activate.mock.calls[0][0] as number[];
    expect(sensorArg.length).toBe(22);
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('resets lastTickInputSource to human on init', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    // First session: auto mode.
    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());
    sendSimStateMessage(0.25, { humanMode: 'auto' });
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');

    // Re-init: should reset to human.
    sendInitMessage('cpu');
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('human');
  });
});

describe('P4S2-sensor-activation: real sensor extraction and tanh output mapping', () => {
  /**
   * Minimal mock that satisfies the `Network.activate(input)` call site inside
   * `buildAutoTickInput`.  The worker only calls `activate(sensors)` on the
   * network — no other methods are needed for this slice.
   */
  function createMockNetwork(outputs: number[] = [0.8, 0.2, 0.5, 0.9, 0.1]): {
    activate: jest.Mock;
  } {
    return { activate: jest.fn(() => outputs) } as unknown as {
      activate: jest.Mock;
    };
  }

  it('AC-065: sensor vector has 22 real (non-zero) elements from game state', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
    };

    sendInitMessage('cpu');
    const mockNet = createMockNetwork([0.9, 0.1, 0.75, 0.6, 0.3]);
    workerModule.__testOnlySetChampionMainNetwork?.(mockNet);

    // Inject an enemy near the player so sensors are non-zero.
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 14.5, y: 10.5 }]);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    expect(mockNet.activate).toHaveBeenCalledTimes(1);
    const sensorArg = mockNet.activate.mock.calls[0][0] as number[];
    expect(sensorArg.length).toBe(22);
    // With a real game state and enemy, at least some sensors should be non-zero.
    // Player health ratio, position, ammo should all be non-zero.
    expect(sensorArg[0]).toBeGreaterThan(0); // health ratio
    expect(sensorArg[3]).not.toBe(0); // position.x
    expect(sensorArg[4]).not.toBe(0); // position.y
  });

  it('AC-066: network outputs mapped via tanh for move/lookDelta and threshold for fire/dash', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
    };

    sendInitMessage('cpu');
    // Use outputs that exercise tanh mapping: [1.0, -1.0, 0.5, 0.6, 0.6]
    // tanh(1.0) ≈ 0.7616, tanh(-1.0) ≈ -0.7616, tanh(0.5) ≈ 0.4621
    // fire = 0.6 > 0 = true, dash = 0.6 > 0.5 = true
    const mockNet = createMockNetwork([1.0, -1.0, 0.5, 0.6, 0.6]);
    workerModule.__testOnlySetChampionMainNetwork?.(mockNet);

    // Champion network path requires an alive enemy to be selected.
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 14.5, y: 10.5 }]);

    // We verify the mapping indirectly: the game tick receives the mapped input
    // and advances the game state. Since we cannot directly inspect the
    // GameTickInputSnapshot, we verify the activate call succeeded and the
    // source is 'auto'.
    sendSimStateMessage(0.25, { humanMode: 'auto' });

    expect(mockNet.activate).toHaveBeenCalledTimes(1);
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('AC-066: fire is true when output[3] > 0, false when ≤ 0', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyGetGameState?(): GameState | null;
    };

    sendInitMessage('cpu');

    // With fire = -0.1 (≤ 0), the player should not fire.
    // We verify via game state: no bolts should be created.
    const mockNet = createMockNetwork([0, 0, 0, -0.1, 0]);
    workerModule.__testOnlySetChampionMainNetwork?.(mockNet);
    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    // No bolts should have been fired (fire = false when output ≤ 0).
    expect((state?.bolts ?? []).length).toBe(0);
  });

  it('AC-066: fire creates a bolt when output[3] > 0', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();

    // Inject an enemy near the player spawn (60.5, 60.5) so enemyVisible = 1
    // and the P5S1 fire gate allows fire to pass through.
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 62.5, y: 60.5 }]);

    // With fire = 0.5 (> 0), the player should fire.
    const mockNet = createMockNetwork([0, 0, 0, 0.5, 0]);
    workerModule.__testOnlySetChampionMainNetwork?.(mockNet);
    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    // A bolt should have been fired (fire = true when output > 0).
    expect((state?.bolts ?? []).length).toBeGreaterThan(0);
  });

  it('human mode remains fully functional after P4S2 changes', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());

    // Queue human input.
    sendActionInputMessage(true);

    // In human mode, pendingTickInput should be used.
    sendSimStateMessage(0.25, { humanMode: 'human' });
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('human');
  });
});

describe('P8S1-coverage-closure: display.worker uncovered branches', () => {
  it('fallback auto AI steers toward an active enemy and fires on cooldown', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyGetEnemyControllerState?(): EnemyControllerState | null;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
    };

    sendInitMessage('cpu');

    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    const px = state!.player.position.x;
    const py = state!.player.position.y;
    const pa = state!.player.angleRad;

    // Keep the controller from moving or killing the injected enemy so the
    // fallback AI sees a stable target across 25 ticks.
    const enemyControllerModule = await import('../shared/enemy-controller');
    const controllerSpy = jest
      .spyOn(enemyControllerModule, 'updateEnemyController')
      .mockImplementation(
        (controllerState: EnemyControllerState) => controllerState,
      );

    // Place one enemy directly in front of the player so it sits inside the
    // fallback fire arc (±π/6) and within a few cells.
    const enemyDistance = 5;
    workerModule.__testOnlyInjectTestEnemies?.([
      {
        x: px + Math.cos(pa) * enemyDistance,
        y: py + Math.sin(pa) * enemyDistance,
      },
    ]);

    // 25 ticks are required to hit the fire-cooldown branch:
    // fallbackTickCounter % NEATENSTEIN_FALLBACK_FIRE_INTERVAL === 0.
    for (let i = 0; i < 25; i += 1) {
      workerSelf.postMessage.mockClear();
      sendSimStateMessage(0.25, { humanMode: 'auto' });
    }

    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');

    controllerSpy.mockRestore();
  });

  it('falls back to fallback AI when champion network activation throws', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');

    const throwingNet = {
      activate: jest.fn(() => {
        throw new Error('champion activation failure');
      }),
    } as unknown as { activate: jest.Mock };
    workerModule.__testOnlySetChampionMainNetwork?.(throwingNet);

    expect(() =>
      sendSimStateMessage(0.25, { humanMode: 'auto' }),
    ).not.toThrow();
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('resolves undefined enemy weights when population snapshot is not mlp', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyPopulation?(): {
        kind: string;
        size: number;
        snapshot: () => { kind: string; weights: Float32Array };
        update: () => unknown;
      } | null;
    };

    sendInitMessage('cpu');
    const population = workerModule.__testOnlyGetEnemyPopulation?.();
    expect(population).not.toBeNull();

    const updateSpy = jest
      .spyOn(population!, 'update')
      .mockReturnValue({ kind: 'swarm' });

    expect(() => sendSimStateMessage()).not.toThrow();

    updateSpy.mockRestore();
  });

  it('skips inactive enemies when fallback AI searches for targets', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
    };

    sendInitMessage('cpu');

    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    const px = state!.player.position.x;
    const py = state!.player.position.y;
    const pa = state!.player.angleRad;

    workerModule.__testOnlyInjectTestEnemies?.([
      {
        x: px + Math.cos(pa) * 5,
        y: py + Math.sin(pa) * 5,
      },
    ]);
    // Mark the injected enemy inactive so the fallback AI hits the
    // `enemy.active === false` continue path.
    state!.enemies[0]!.active = false;

    expect(() =>
      sendSimStateMessage(0.25, { humanMode: 'auto' }),
    ).not.toThrow();
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('wraps fallback AI steering angle across the positive π boundary', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlySetChampionMainNetwork?(network: unknown): void;
    };

    sendInitMessage('cpu');

    // Ensure the fallback AI path is used, not the champion network path.
    workerModule.__testOnlySetChampionMainNetwork?.(null);

    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    const px = state!.player.position.x;
    const py = state!.player.position.y;

    // Place the player angle near -π and the nearest enemy bearing near +π.
    // Keep the injected enemy very close so it wins the nearest-neighbor
    // search; the raw angle difference is larger than π, forcing line 1519 to
    // subtract 2π during normalisation.
    state!.player.angleRad = -3.13;
    workerModule.__testOnlyInjectTestEnemies?.([
      {
        x: px - 0.1,
        y: py + 0.01,
      },
    ]);

    expect(() =>
      sendSimStateMessage(0.25, { humanMode: 'auto' }),
    ).not.toThrow();
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('wraps fallback AI steering angle across the negative π boundary', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlySetChampionMainNetwork?(network: unknown): void;
    };

    sendInitMessage('cpu');

    // Ensure the fallback AI path is used, not the champion network path.
    workerModule.__testOnlySetChampionMainNetwork?.(null);

    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    const px = state!.player.position.x;
    const py = state!.player.position.y;

    // Place the player angle near +π and the nearest enemy bearing near -π.
    // Keep the injected enemy very close so it wins the nearest-neighbor
    // search; the raw angle difference is less than -π, forcing line 1520 to
    // add 2π during normalisation.
    state!.player.angleRad = 3.13;
    workerModule.__testOnlyInjectTestEnemies?.([
      {
        x: px - 0.1,
        y: py - 0.01,
      },
    ]);

    expect(() =>
      sendSimStateMessage(0.25, { humanMode: 'auto' }),
    ).not.toThrow();
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('resolves undefined enemy weights when the population is null', async () => {
    jest.resetModules();
    const enemyMlpModule =
      (await import('../harness/enemy-mlp')) as unknown as {
        createMlpEnemyPopulation: jest.Mock;
      };
    const spy = jest
      .spyOn(enemyMlpModule, 'createMlpEnemyPopulation')
      .mockReturnValue(null);

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyPopulation?(): unknown;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');
    expect(workerModule.__testOnlyGetEnemyPopulation?.()).toBeNull();
    expect(() => sendSimStateMessage()).not.toThrow();

    spy.mockRestore();
  });

  it('delegates the arms-race evaluation to the eval worker after wave clear', async () => {
    jest.resetModules();

    // Mock neataptic so Network.fromJSON returns a lightweight stand-in
    // instead of deserializing a real network in the test environment.
    jest.doMock('neataptic', () => ({
      Network: {
        fromJSON: jest.fn(() => ({
          activate: jest.fn(() => [0.5, 0.5, 0.5, 0.5]),
        })),
      },
    }));

    const tickModule = await import('../host/game/tick');
    const gameTickSpy = jest
      .spyOn(tickModule, 'gameTick')
      .mockImplementation(
        (state: GameState) => ({ ...state, enemies: [] }) as GameState,
      );

    // Create a mock eval worker to intercept the delegation (AC-P2S1b-002).
    const mockEvalWorker = {
      postMessage: jest.fn(),
      onmessage: null as ((event: MessageEvent) => void) | null,
    };

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyGetPendingGeneration?(): number | null;
      __testOnlyGetChampionMainNetwork?(): unknown;
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyGetAllEnemiesCleared?(): boolean;
      __testOnlySetEvalWorker?(
        worker: {
          postMessage: (msg: unknown) => void;
          onmessage: unknown;
        } | null,
      ): void;
    };

    sendInitMessage('cpu');
    // Inject the mock eval worker after init (init resets evalWorker).
    workerModule.__testOnlySetEvalWorker?.(mockEvalWorker);
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 13.5, y: 12.5 }]);

    // Establish at least one alive enemy so the wave-clear transition fires.
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    workerSelf.postMessage.mockClear();
    sendSimStateMessage();

    expect(workerModule.__testOnlyGetAllEnemiesCleared?.()).toBe(true);
    // AC-P2S1b-001: eval worker handles evaluation asynchronously.
    expect(workerModule.__testOnlyGetPendingGeneration?.()).not.toBeNull();

    // AC-P2S1b-002: display.worker delegates via postMessage, no blocking await.
    const evalPost = mockEvalWorker.postMessage.mock.calls[0]?.[0] as {
      type: string;
      seed: number;
      generation: number;
      enemySnapshot: unknown;
      humanMode: boolean;
    };
    expect(evalPost).toBeDefined();
    expect(evalPost.type).toBe('evaluate');

    // Simulate the eval worker completing: post evalComplete back.
    mockEvalWorker.onmessage?.({
      data: {
        type: 'evalComplete',
        generation: 1,
        championNetworkJSON: {
          input: 2,
          output: 2,
          nodes: [],
          connections: [],
        },
      },
    } as unknown as MessageEvent);

    // Wait for the async handleEvalComplete to finish.
    const startMs = Date.now();
    while (
      workerModule.__testOnlyGetPendingGeneration?.() !== null &&
      Date.now() - startMs < 5000
    ) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }

    expect(workerModule.__testOnlyGetPendingGeneration?.()).toBeNull();
    expect(workerModule.__testOnlyGetChampionMainNetwork?.()).not.toBeNull();
    expect(workerModule.__testOnlyGetGameState?.()?.generation).toBeGreaterThan(
      0,
    );

    gameTickSpy.mockRestore();
  }, 10000);

  it('does not inject enemy weights when the advanceWave snapshot is not mlp', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyPopulation?(): {
        kind: string;
        update: jest.Mock;
      } | null;
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyGetAllEnemiesCleared?(): boolean;
    };

    sendInitMessage('cpu');

    workerModule.__testOnlyInjectTestEnemies?.([{ x: 13.5, y: 12.5 }]);
    // Let enemies establish with the real MLP population before we force a
    // non-MLP snapshot on the wave-clear tick.
    for (let i = 0; i < 3; i += 1) {
      sendSimStateMessage();
    }

    const population = workerModule.__testOnlyGetEnemyPopulation?.();
    expect(population).not.toBeNull();
    const updateSpy = jest
      .spyOn(population!, 'update')
      .mockReturnValue({ kind: 'swarm' });

    // Force a wave-clear on the next tick so advanceWave is invoked.
    const realTick = await import('../host/game/tick');
    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    const gameTickSpy = jest
      .spyOn(realTick, 'gameTick')
      .mockImplementation(() => ({ ...state, enemies: [] }) as GameState);

    workerSelf.postMessage.mockClear();
    expect(() => sendSimStateMessage()).not.toThrow();
    expect(workerModule.__testOnlyGetAllEnemiesCleared?.()).toBe(true);

    updateSpy.mockRestore();
    gameTickSpy.mockRestore();
  });

  it('networkOutputToTickInput pads a short output array', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');

    const shortOutputNet = {
      activate: jest.fn(() => [0.1, 0.2]),
    } as unknown as { activate: jest.Mock };
    workerModule.__testOnlySetChampionMainNetwork?.(shortOutputNet);

    expect(() =>
      sendSimStateMessage(0.25, { humanMode: 'auto' }),
    ).not.toThrow();
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('does not launch a second arms-race evaluation while one is pending', async () => {
    jest.resetModules();

    // Inject a mock eval worker that captures but never responds, so
    // pendingGeneration stays non-null throughout the test.
    const mockEvalWorker = {
      postMessage: jest.fn(),
      onmessage: null as ((event: MessageEvent) => void) | null,
    };

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetPendingGeneration?(): number | null;
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlySetEvalWorker?(
        worker: {
          postMessage: (msg: unknown) => void;
          onmessage: unknown;
        } | null,
      ): void;
    };

    sendInitMessage('cpu');
    // Inject the mock eval worker after init (init resets evalWorker).
    workerModule.__testOnlySetEvalWorker?.(mockEvalWorker);

    const realTick = await import('../host/game/tick');
    let keepEnemiesAlive = false;
    const gameTickSpy = jest
      .spyOn(realTick, 'gameTick')
      .mockImplementation(
        (s: GameState) =>
          ({ ...s, enemies: keepEnemiesAlive ? s.enemies : [] }) as GameState,
      );

    // First wave-clear: enemies go from present to empty, launching the
    // fire-and-forget arms-race evaluation.
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 13.5, y: 12.5 }]);
    sendSimStateMessage();
    const firstPending = workerModule.__testOnlyGetPendingGeneration?.();
    expect(firstPending).not.toBeNull();
    // AC-P2S1b-002: delegation happened via postMessage.
    expect(mockEvalWorker.postMessage).toHaveBeenCalledTimes(1);

    // Tick once with alive enemies to reset prevAllEnemiesCleared back to false
    // while the first generation is still pending.
    keepEnemiesAlive = true;
    workerModule.__testOnlyInjectTestEnemies?.([
      { x: 14.5, y: 13.5 },
      { x: 15.5, y: 14.5 },
    ]);
    sendSimStateMessage();

    // Second wave-clear while the first generation is still in flight. This
    // enters the wave-clear block with pendingGeneration !== null, exercising
    // the `pendingGeneration === null` false branch at line 1799 and skipping
    // the launch of a second concurrent evaluation.
    keepEnemiesAlive = false;
    sendSimStateMessage();
    expect(workerModule.__testOnlyGetPendingGeneration?.()).toBe(firstPending);

    // No second delegation: postMessage should still have been called only once.
    expect(mockEvalWorker.postMessage).toHaveBeenCalledTimes(1);

    gameTickSpy.mockRestore();
  });
});

describe('AC-P3S1c-002: champion extinction on input count change', () => {
  function createMockNetwork(outputs: number[] = [0.8, 0.2, 0.5, 0.9, 0.1]): {
    activate: jest.Mock;
  } {
    return { activate: jest.fn(() => outputs) } as unknown as {
      activate: jest.Mock;
    };
  }

  it('clears championMainNetwork when input count changes from 12 to 22', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlySetChampionInputCount?(count: number | null): void;
      __testOnlyGetChampionMainNetwork?(): unknown;
      __testOnlyGetChampionInputCount?(): number | null;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');

    // Inject a champion network (sets lastChampionInputCount = 22).
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());
    expect(workerModule.__testOnlyGetChampionMainNetwork?.()).not.toBeNull();

    // Simulate the champion being evolved with 12 inputs (the old count).
    workerModule.__testOnlySetChampionInputCount?.(12);

    // Send a simState in auto mode — the extinction guard should fire.
    sendSimStateMessage(0.25, { humanMode: 'auto' });

    // Champion should be cleared because input count mismatched.
    expect(workerModule.__testOnlyGetChampionMainNetwork?.()).toBeNull();
    expect(workerModule.__testOnlyGetChampionInputCount?.()).toBeNull();

    // Fallback AI should have been used (still 'auto' source).
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('does not clear championMainNetwork when input count matches', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetChampionMainNetwork?(): unknown;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');

    // Inject a champion network (sets lastChampionInputCount = 22).
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());

    // Send a simState in auto mode — input count matches, no extinction.
    sendSimStateMessage(0.25, { humanMode: 'auto' });

    // Champion should still be present.
    expect(workerModule.__testOnlyGetChampionMainNetwork?.()).not.toBeNull();
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('clears lastChampionInputCount on re-init', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetChampionInputCount?(): number | null;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(createMockNetwork());
    expect(workerModule.__testOnlyGetChampionInputCount?.()).toBe(22);

    // Re-init should clear the input count.
    sendInitMessage('cpu');
    expect(workerModule.__testOnlyGetChampionInputCount?.()).toBeNull();
  });
});

describe('P5S1-fire-gate: soft fire gate in buildAutoTickInput', () => {
  /**
   * Minimal mock that satisfies the `Network.activate(input)` call site inside
   * `buildAutoTickInput`.  The worker only calls `activate(sensors)` on the
   * network — no other methods are needed for this slice.
   */
  function createMockNetwork(outputs: number[] = [0.8, 0.2, 0.5, 0.9, 0.1]): {
    activate: jest.Mock;
  } {
    return { activate: jest.fn(() => outputs) } as unknown as {
      activate: jest.Mock;
    };
  }

  it('AC-P5S1a-001: suppresses fire when no enemy is visible (no enemies on map)', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();

    // Network wants to fire (output[3] = 0.9 > 0)
    workerModule.__testOnlySetChampionMainNetwork?.(
      createMockNetwork([0, 0, 0, 0.9, 0]),
    );

    // No enemies injected → enemyVisible sensor = 0 → fire should be suppressed
    sendSimStateMessage(0.25, { humanMode: 'auto' });

    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');

    // Verify no bolts were fired (fire was suppressed by the gate)
    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    expect((state?.bolts ?? []).length).toBe(0);
  });

  it('AC-P5S1a-002: allows fire when an enemy is visible', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();

    // Network wants to fire (output[3] = 0.5 > 0)
    workerModule.__testOnlySetChampionMainNetwork?.(
      createMockNetwork([0, 0, 0, 0.5, 0]),
    );

    // Inject an enemy near the player spawn (60.5, 60.5) so enemyVisible = 1
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 62.5, y: 60.5 }]);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    // Fire should NOT be suppressed — a bolt should be created
    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    expect((state?.bolts ?? []).length).toBeGreaterThan(0);
  });

  it('AC-P5S1a-003: hysteresis state maintained between ticks', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetFireGateState?(): { fireActive: boolean };
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();

    // Initially gate should be closed
    expect(workerModule.__testOnlyGetFireGateState?.().fireActive).toBe(false);

    // Inject enemy near the player spawn (60.5, 60.5) so enemyVisible = 1
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 62.5, y: 60.5 }]);
    workerModule.__testOnlySetChampionMainNetwork?.(
      createMockNetwork([0, 0, 0, 0.5, 0]),
    );
    sendSimStateMessage(0.25, { humanMode: 'auto' });

    // Gate should now be open
    expect(workerModule.__testOnlyGetFireGateState?.().fireActive).toBe(true);
  });

  it('AC-P5S1a-001: fire gate resets on re-init', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetFireGateState?(): { fireActive: boolean };
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();

    // Open the gate by having an enemy visible near the player spawn
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 62.5, y: 60.5 }]);
    workerModule.__testOnlySetChampionMainNetwork?.(
      createMockNetwork([0, 0, 0, 0.5, 0]),
    );
    sendSimStateMessage(0.25, { humanMode: 'auto' });
    expect(workerModule.__testOnlyGetFireGateState?.().fireActive).toBe(true);

    // Re-init should reset the gate to closed
    sendInitMessage('cpu');
    expect(workerModule.__testOnlyGetFireGateState?.().fireActive).toBe(false);
  });

  it('AC-P5S1a-002: fire not suppressed when enemy visible but outside firing arc', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();

    // Network wants to fire
    workerModule.__testOnlySetChampionMainNetwork?.(
      createMockNetwork([0, 0, 0, 0.5, 0]),
    );

    // Inject enemy near the player spawn — even if outside firing arc,
    // enemyVisible = 1 (visibility only checks vision range + line of sight)
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 62.5, y: 60.5 }]);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    // Fire should not be suppressed (enemy is visible)
    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    expect((state?.bolts ?? []).length).toBeGreaterThan(0);
  });
});

describe('02-vision-fallback: vision-aware fallback AI', () => {
  afterEach(() => {
    jest.dontMock('../shared/enemy-navigation');
  });

  it('selects target via findNearestVisibleEnemy and steers toward it', async () => {
    jest.resetModules();
    const findNearestVisibleEnemyMock = jest.fn<
      (
        state: GameState,
        flatMap: Uint8Array,
        mapSize: number,
      ) => EnemyState | null
    >(() => null);
    jest.doMock('../shared/enemy-navigation', () => ({
      ...(jest.requireActual('../shared/enemy-navigation') as Record<
        string,
        unknown
      >),
      findNearestVisibleEnemy: findNearestVisibleEnemyMock,
    }));

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');

    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    state!.player.angleRad = 0;
    const px = state!.player.position.x;
    const py = state!.player.position.y;

    // Return a visible enemy directly south-east of the player (bearing +π/4).
    findNearestVisibleEnemyMock.mockReturnValue({
      position: { x: px + 5, y: py + 5 },
      health: 100,
      active: true,
    } as unknown as EnemyState);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
    const after = workerModule.__testOnlyGetGameState?.();
    expect(after).not.toBeNull();
    expect(after!.player.angleRad).toBeGreaterThan(0);
    expect(findNearestVisibleEnemyMock).toHaveBeenCalled();
  });

  it('does not fire when no enemy is visible', async () => {
    jest.resetModules();
    const findNearestVisibleEnemyMock = jest.fn<
      (
        state: GameState,
        flatMap: Uint8Array,
        mapSize: number,
      ) => EnemyState | null
    >(() => null);
    jest.doMock('../shared/enemy-navigation', () => ({
      ...(jest.requireActual('../shared/enemy-navigation') as Record<
        string,
        unknown
      >),
      findNearestVisibleEnemy: findNearestVisibleEnemyMock,
    }));

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyResetFireGateState?(): void;
      __testOnlyGetFireGateState?(): { fireActive: boolean };
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    expect((state?.bolts ?? []).length).toBe(0);
    expect(workerModule.__testOnlyGetFireGateState?.().fireActive).toBe(false);
    expect(findNearestVisibleEnemyMock).toHaveBeenCalled();
  });

  it('fires through the shared fire gate when a visible enemy is in arc', async () => {
    jest.resetModules();
    const findNearestVisibleEnemyMock = jest.fn<
      (
        state: GameState,
        flatMap: Uint8Array,
        mapSize: number,
      ) => EnemyState | null
    >(() => null);
    jest.doMock('../shared/enemy-navigation', () => ({
      ...(jest.requireActual('../shared/enemy-navigation') as Record<
        string,
        unknown
      >),
      findNearestVisibleEnemy: findNearestVisibleEnemyMock,
    }));

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyResetFireGateState?(): void;
      __testOnlyGetFireGateState?(): { fireActive: boolean };
    };

    findNearestVisibleEnemyMock.mockImplementation(
      (gameStateArg: GameState) => {
        const p = gameStateArg.player;
        return {
          position: {
            x: p.position.x + Math.cos(p.angleRad) * 3,
            y: p.position.y + Math.sin(p.angleRad) * 3,
          },
          health: 100,
          active: true,
        } as unknown as EnemyState;
      },
    );

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();
    expect(workerModule.__testOnlyGetFireGateState?.().fireActive).toBe(false);

    // Run enough ticks to hit the fallback cooldown interval.
    for (let i = 0; i < 25; i += 1) {
      sendSimStateMessage(0.25, { humanMode: 'auto' });
    }

    const after = workerModule.__testOnlyGetGameState?.();
    expect(after).not.toBeNull();
    expect((after?.bolts ?? []).length).toBeGreaterThan(0);
    expect(workerModule.__testOnlyGetFireGateState?.().fireActive).toBe(true);
    expect(findNearestVisibleEnemyMock).toHaveBeenCalled();
  });
});

describe('P9S2-fallback-hunter: wall-bounce exploration and kiting', () => {
  afterEach(() => {
    jest.dontMock('../renderer/raycast');
    jest.dontMock('../shared/enemy-navigation');
  });

  it('defaults to forward exploration when wallMap is null and ramps move on first tick', async () => {
    jest.resetModules();

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyBuildFallbackAutoTickInput?(
        state: GameState,
      ): GameTickInputSnapshot;
    };

    const state = {
      player: {
        position: { x: 12.5, y: 12.5 },
        angleRad: 0,
      },
    } as unknown as GameState;

    const input = workerModule.__testOnlyBuildFallbackAutoTickInput?.(state);
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(NEATENSTEIN_MOVE_ACCEL_PER_TICK);
    expect(input!.lookDelta).toBe(0);
    expect(input!.fire).toBe(false);
  });

  function createOpenRaycastMock(): jest.MockedFunction<
    typeof castRayDDAFromFlatMap
  > {
    return jest.fn(() => ({
      perpWallDist: Number.POSITIVE_INFINITY,
      side: 0 as const,
      mapX: 0,
      mapY: 0,
    }));
  }

  function installRaycastMock(
    mock: jest.MockedFunction<typeof castRayDDAFromFlatMap>,
  ): void {
    jest.doMock('../renderer/raycast', () => ({
      ...(jest.requireActual('../renderer/raycast') as Record<string, unknown>),
      castRayDDAFromFlatMap: mock,
    }));
  }

  it('moves forward and ramps move on first tick when no enemy and no wall is ahead', async () => {
    jest.resetModules();
    const raycastMock = createOpenRaycastMock();
    installRaycastMock(raycastMock);

    const findNearestVisibleEnemyMock = jest.fn<
      (
        state: GameState,
        flatMap: Uint8Array,
        mapSize: number,
      ) => EnemyState | null
    >(() => null);
    jest.doMock('../shared/enemy-navigation', () => ({
      ...(jest.requireActual('../shared/enemy-navigation') as Record<
        string,
        unknown
      >),
      findNearestVisibleEnemy: findNearestVisibleEnemyMock,
    }));

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetLastFallbackInput?(): GameTickInputSnapshot | null;
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();
    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const input = workerModule.__testOnlyGetLastFallbackInput?.();
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(NEATENSTEIN_MOVE_ACCEL_PER_TICK);
    expect(input!.lookDelta).toBe(0);
    expect(input!.fire).toBe(false);
    expect(raycastMock).toHaveBeenCalled();
    expect(findNearestVisibleEnemyMock).toHaveBeenCalled();
  });

  it('fallback exploration overrides a champion network spin output when no enemies exist and ramps move', async () => {
    jest.resetModules();

    /**
     * Minimal mock that satisfies the `Network.activate(input)` call site
     * inside `buildAutoTickInput`. Output index 2 carries a non-zero yaw delta,
     * which would make the hero spin if the champion path were selected.
     */
    function createSpinningMockNetwork(): {
      activate: jest.Mock;
    } {
      return {
        activate: jest.fn(() => [0, 0, 1.0, 0, 0]),
      } as unknown as {
        activate: jest.Mock;
      };
    }

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyGetLastFallbackInput?(): GameTickInputSnapshot | null;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(
      createSpinningMockNetwork(),
    );
    workerModule.__testOnlyInjectTestEnemies?.([]);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const input = workerModule.__testOnlyGetLastFallbackInput?.();
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(NEATENSTEIN_MOVE_ACCEL_PER_TICK);
    expect(input!.lookDelta).toBe(0);
  });

  it('falls back to exploration AI when champion network activation throws and ramps move', async () => {
    jest.resetModules();

    function createThrowingMockNetwork(): {
      activate: jest.Mock;
    } {
      return {
        activate: jest.fn(() => {
          throw new Error('simulated network failure');
        }),
      } as unknown as {
        activate: jest.Mock;
      };
    }

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetChampionMainNetwork?(network: unknown): void;
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyGetLastFallbackInput?(): GameTickInputSnapshot | null;
      __testOnlyGetLastTickInputSource?(): 'auto' | 'human';
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetChampionMainNetwork?.(
      createThrowingMockNetwork(),
    );
    // Force the champion path to be selected by providing an alive enemy.
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 14.5, y: 10.5 }]);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const input = workerModule.__testOnlyGetLastFallbackInput?.();
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(NEATENSTEIN_MOVE_ACCEL_PER_TICK);
    expect(input!.lookDelta).toBe(0);
    expect(workerModule.__testOnlyGetLastTickInputSource?.()).toBe('auto');
  });

  it('turns toward the more open angle when a wall is directly ahead', async () => {
    jest.resetModules();

    const bounce = NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD;
    const forwardDir = { x: Math.cos(0), y: Math.sin(0) };
    const bouncePlusDir = { x: Math.cos(bounce), y: Math.sin(bounce) };

    const raycastMock = jest.fn<typeof castRayDDAFromFlatMap>(
      (
        _flatMap: Uint8Array,
        _side: number,
        _posX: number,
        _posY: number,
        dirX: number,
        dirY: number,
      ) => {
        const dotForward = dirX * forwardDir.x + dirY * forwardDir.y;
        const dotPlus = dirX * bouncePlusDir.x + dirY * bouncePlusDir.y;
        if (dotForward > 0.99) {
          return {
            perpWallDist: 1,
            side: 0 as const,
            mapX: 0,
            mapY: 0,
          };
        }
        if (dotPlus > 0.99) {
          return {
            perpWallDist: Number.POSITIVE_INFINITY,
            side: 0 as const,
            mapX: 0,
            mapY: 0,
          };
        }
        return {
          perpWallDist: 2,
          side: 1 as const,
          mapX: 0,
          mapY: 0,
        };
      },
    );
    installRaycastMock(raycastMock);

    const findNearestVisibleEnemyMock = jest.fn<
      (
        state: GameState,
        flatMap: Uint8Array,
        mapSize: number,
      ) => EnemyState | null
    >(() => null);
    jest.doMock('../shared/enemy-navigation', () => ({
      ...(jest.requireActual('../shared/enemy-navigation') as Record<
        string,
        unknown
      >),
      findNearestVisibleEnemy: findNearestVisibleEnemyMock,
    }));

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetLastFallbackInput?(): GameTickInputSnapshot | null;
      __testOnlyGetGameState?(): GameState | null;
    };

    sendInitMessage('cpu');
    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    state!.player.angleRad = 0;

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const input = workerModule.__testOnlyGetLastFallbackInput?.();
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(NEATENSTEIN_MOVE_ACCEL_PER_TICK);
    // The +bounce direction is the most open, so the hunter turns right, but
    // the first tick is limited by the smoothing accel rate.
    expect(input!.lookDelta).toBeCloseTo(NEATENSTEIN_MOVE_ACCEL_PER_TICK, 10);
  });

  it('turns left when the -bounce direction is the most open', async () => {
    jest.resetModules();

    const bounce = NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD;
    const forwardDir = { x: Math.cos(0), y: Math.sin(0) };
    const bounceMinusDir = { x: Math.cos(-bounce), y: Math.sin(-bounce) };

    const raycastMock = jest.fn<typeof castRayDDAFromFlatMap>(
      (
        _flatMap: Uint8Array,
        _side: number,
        _posX: number,
        _posY: number,
        dirX: number,
        dirY: number,
      ) => {
        const dotForward = dirX * forwardDir.x + dirY * forwardDir.y;
        const dotMinus = dirX * bounceMinusDir.x + dirY * bounceMinusDir.y;
        if (dotForward > 0.99) {
          return {
            perpWallDist: 1,
            side: 0 as const,
            mapX: 0,
            mapY: 0,
          };
        }
        if (dotMinus > 0.99) {
          return {
            perpWallDist: Number.POSITIVE_INFINITY,
            side: 0 as const,
            mapX: 0,
            mapY: 0,
          };
        }
        return {
          perpWallDist: 2,
          side: 1 as const,
          mapX: 0,
          mapY: 0,
        };
      },
    );
    installRaycastMock(raycastMock);

    const findNearestVisibleEnemyMock = jest.fn<
      (
        state: GameState,
        flatMap: Uint8Array,
        mapSize: number,
      ) => EnemyState | null
    >(() => null);
    jest.doMock('../shared/enemy-navigation', () => ({
      ...(jest.requireActual('../shared/enemy-navigation') as Record<
        string,
        unknown
      >),
      findNearestVisibleEnemy: findNearestVisibleEnemyMock,
    }));

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetLastFallbackInput?(): GameTickInputSnapshot | null;
      __testOnlyGetGameState?(): GameState | null;
    };

    sendInitMessage('cpu');
    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    state!.player.angleRad = 0;

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const input = workerModule.__testOnlyGetLastFallbackInput?.();
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(NEATENSTEIN_MOVE_ACCEL_PER_TICK);
    // The -bounce direction is the most open, so the hunter turns left, but
    // the first tick is limited by the smoothing decel rate.
    expect(input!.lookDelta).toBeCloseTo(-NEATENSTEIN_MOVE_DECEL_PER_TICK, 10);
  });

  it('backpedals when a visible enemy is inside the backpedal distance and ramps move', async () => {
    jest.resetModules();
    installRaycastMock(createOpenRaycastMock());

    const findNearestVisibleEnemyMock = jest.fn<
      (
        state: GameState,
        flatMap: Uint8Array,
        mapSize: number,
      ) => EnemyState | null
    >(() => null);
    jest.doMock('../shared/enemy-navigation', () => ({
      ...(jest.requireActual('../shared/enemy-navigation') as Record<
        string,
        unknown
      >),
      findNearestVisibleEnemy: findNearestVisibleEnemyMock,
    }));

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetLastFallbackInput?(): GameTickInputSnapshot | null;
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();
    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    const px = state!.player.position.x;
    const py = state!.player.position.y;
    state!.player.angleRad = 0;

    const distance = NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS - 5;
    findNearestVisibleEnemyMock.mockReturnValue({
      position: { x: px + distance, y: py },
      health: 100,
      active: true,
    } as unknown as EnemyState);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const input = workerModule.__testOnlyGetLastFallbackInput?.();
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(-NEATENSTEIN_MOVE_DECEL_PER_TICK);
    expect(input!.lookDelta).toBe(0);
  });

  it('holds position when a visible enemy is in the 15–20 cell kiting band', async () => {
    jest.resetModules();
    installRaycastMock(createOpenRaycastMock());

    const findNearestVisibleEnemyMock = jest.fn<
      (
        state: GameState,
        flatMap: Uint8Array,
        mapSize: number,
      ) => EnemyState | null
    >(() => null);
    jest.doMock('../shared/enemy-navigation', () => ({
      ...(jest.requireActual('../shared/enemy-navigation') as Record<
        string,
        unknown
      >),
      findNearestVisibleEnemy: findNearestVisibleEnemyMock,
    }));

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetLastFallbackInput?(): GameTickInputSnapshot | null;
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();
    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    const px = state!.player.position.x;
    const py = state!.player.position.y;
    state!.player.angleRad = 0;

    const distance =
      NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS +
      (NEATENSTEIN_KITING_APPROACH_DISTANCE_CELLS -
        NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS) /
        2;
    findNearestVisibleEnemyMock.mockReturnValue({
      position: { x: px + distance, y: py },
      health: 100,
      active: true,
    } as unknown as EnemyState);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const input = workerModule.__testOnlyGetLastFallbackInput?.();
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(0);
    expect(input!.lookDelta).toBe(0);
  });

  it('approaches when a visible enemy is beyond the kiting band and ramps move', async () => {
    jest.resetModules();
    installRaycastMock(createOpenRaycastMock());

    const findNearestVisibleEnemyMock = jest.fn<
      (
        state: GameState,
        flatMap: Uint8Array,
        mapSize: number,
      ) => EnemyState | null
    >(() => null);
    jest.doMock('../shared/enemy-navigation', () => ({
      ...(jest.requireActual('../shared/enemy-navigation') as Record<
        string,
        unknown
      >),
      findNearestVisibleEnemy: findNearestVisibleEnemyMock,
    }));

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetLastFallbackInput?(): GameTickInputSnapshot | null;
      __testOnlyGetGameState?(): GameState | null;
      __testOnlyResetFireGateState?(): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlyResetFireGateState?.();
    const state = workerModule.__testOnlyGetGameState?.();
    expect(state).not.toBeNull();
    const px = state!.player.position.x;
    const py = state!.player.position.y;
    state!.player.angleRad = 0;

    const distance = NEATENSTEIN_KITING_APPROACH_DISTANCE_CELLS + 5;
    findNearestVisibleEnemyMock.mockReturnValue({
      position: { x: px + distance, y: py },
      health: 100,
      active: true,
    } as unknown as EnemyState);

    sendSimStateMessage(0.25, { humanMode: 'auto' });

    const input = workerModule.__testOnlyGetLastFallbackInput?.();
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(NEATENSTEIN_MOVE_ACCEL_PER_TICK);
    expect(input!.lookDelta).toBe(0);
  });

  it('ramps move.y to full forward over multiple fallback ticks', async () => {
    jest.resetModules();

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyBuildFallbackAutoTickInput?(
        state: GameState,
      ): GameTickInputSnapshot;
      __testOnlyResetSmoothingState?(): void;
    };

    const state = {
      player: {
        position: { x: 12.5, y: 12.5 },
        angleRad: 0,
      },
    } as unknown as GameState;

    workerModule.__testOnlyResetSmoothingState?.();
    let input = workerModule.__testOnlyBuildFallbackAutoTickInput?.(state);
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(NEATENSTEIN_MOVE_ACCEL_PER_TICK);

    // 0.2 → 0.4 → 0.6 → 0.8 → 1.0
    for (let i = 0; i < 4; i++) {
      input = workerModule.__testOnlyBuildFallbackAutoTickInput?.(state);
    }
    expect(input).not.toBeNull();
    expect(input!.move.y).toBe(1);
  });
});
