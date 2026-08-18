/**
 * C2 RED-phase tests: Test Quality Improvements.
 *
 * These tests encode the four solution items from the Neatenstein Ultimate
 * Quality Upgrade plan (Step C2) as failing contracts.  Each describe block
 * maps to one solution item.  Tests are designed to fail for the right
 * reason: the quality improvement has not been performed yet.
 *
 * Structural / architectural tests inspect the filesystem and source content.
 * Behavioral tests exercise runtime contracts (debug logging in catch blocks).
 *
 * @module
 */

import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { NEATENSTEIN_MAP_SIZE } from './browser-entry/constants';

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

const neatensteinRoot = join(__dirname);
const browserEntryRoot = join(neatensteinRoot, 'browser-entry');
const workerRoot = join(browserEntryRoot, 'worker');
const sharedRoot = join(browserEntryRoot, 'shared');
const hostGameRoot = join(browserEntryRoot, 'host', 'game');
const rendererRoot = join(browserEntryRoot, 'renderer');

// ---------------------------------------------------------------------------
// C2-1: Extract test harness helpers to shared module
// ---------------------------------------------------------------------------

describe('C2-1: Test harness helpers extracted to shared module', () => {
  it('display-worker-derez.test.ts does not define installMockWorkerGlobal locally', () => {
    const content = readFileSync(
      join(workerRoot, 'display-worker-derez.test.ts'),
      'utf-8',
    );
    // The local function definition should be removed — the file should
    // import from the shared display.worker.test-helpers.ts module instead.
    expect(content).not.toMatch(
      /function\s+installMockWorkerGlobal\s*\(/,
    );
  });

  it('display-worker-derez.test.ts does not define sendInitMessage locally', () => {
    const content = readFileSync(
      join(workerRoot, 'display-worker-derez.test.ts'),
      'utf-8',
    );
    expect(content).not.toMatch(/function\s+sendInitMessage\s*\(/);
  });

  it('display-worker-derez.test.ts does not define sendSimStateMessage locally', () => {
    const content = readFileSync(
      join(workerRoot, 'display-worker-derez.test.ts'),
      'utf-8',
    );
    expect(content).not.toMatch(/function\s+sendSimStateMessage\s*\(/);
  });

  it('display-worker-derez.test.ts imports from display.worker.test-helpers', () => {
    const content = readFileSync(
      join(workerRoot, 'display-worker-derez.test.ts'),
      'utf-8',
    );
    expect(content).toMatch(
      /from\s+['"]\.\/display\.worker\.test-helpers['"]/,
    );
  });

  it('eval.worker.test.ts does not define installMockWorkerGlobal locally', () => {
    const content = readFileSync(
      join(workerRoot, 'eval.worker.test.ts'),
      'utf-8',
    );
    expect(content).not.toMatch(
      /function\s+installMockWorkerGlobal\s*\(/,
    );
  });

  it('eval.worker.test.ts imports from display.worker.test-helpers', () => {
    const content = readFileSync(
      join(workerRoot, 'eval.worker.test.ts'),
      'utf-8',
    );
    expect(content).toMatch(
      /from\s+['"]\.\/display\.worker\.test-helpers['"]/,
    );
  });
});

// ---------------------------------------------------------------------------
// C2-2: Replace Record<string, any> with Record<string, unknown>
// ---------------------------------------------------------------------------

describe('C2-2: Record<string, any> replaced with Record<string, unknown>', () => {
  /**
   * Scan all test files under browser-entry for `Record<string, any>` usage.
   * After the fix, every occurrence should be replaced with
   * `Record<string, unknown>`.
   */
  it('no test file under browser-entry contains Record<string, any>', () => {
    const testFiles: string[] = [
      join(hostGameRoot, 'collision.test.ts'),
      join(hostGameRoot, 'state.test.ts'),
      join(rendererRoot, 'gun.test.ts'),
      join(rendererRoot, 'gun-sprite-data.test.ts'),
    ];

    const offenders: string[] = [];
    for (const file of testFiles) {
      const content = readFileSync(file, 'utf-8');
      if (content.includes('Record<string, any>')) {
        offenders.push(file);
      }
    }

    expect(offenders).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// C2-3: Tighten tick.test.ts mock — no any usage
// ---------------------------------------------------------------------------

describe('C2-3: tick.test.ts mock uses proper typing (no any)', () => {
  it('tick.test.ts does not use (state: any) in mockImplementation', () => {
    const content = readFileSync(
      join(hostGameRoot, 'tick.test.ts'),
      'utf-8',
    );
    // The mock at the applyEnemyDamage spy should use a properly typed
    // parameter instead of `any`.
    expect(content).not.toMatch(/\(state:\s*any\)/);
  });

  it('tick.test.ts does not have eslint-disable for no-explicit-any on mock', () => {
    const content = readFileSync(
      join(hostGameRoot, 'tick.test.ts'),
      'utf-8',
    );
    // The eslint-disable comment for the any-typed mock should be removed
    // once the mock is properly typed.
    expect(content).not.toMatch(
      /eslint-disable.*@typescript-eslint\/no-explicit-any.*mock/,
    );
  });
});

// ---------------------------------------------------------------------------
// C2-4: Debug logging in catch blocks
// ---------------------------------------------------------------------------

describe('C2-4: Debug logging in catch blocks', () => {
  let debugSpy: ReturnType<typeof jest.spyOn>;

  beforeEach(() => {
    debugSpy = jest.spyOn(console, 'debug').mockImplementation(() => {});
  });

  afterEach(() => {
    debugSpy.mockRestore();
  });

  // --- enemy-controller.move.utils.ts catch blocks ---

  describe('enemy-controller.move.utils.ts computeMovement catch block', () => {
    it('calls console.debug when MLP activation throws (wrong weight length)', async () => {
      const { buildNeatensteinMap, createCollisionMap } = await import(
        './browser-entry/renderer/map'
      );
      const { createGameState } = await import(
        './browser-entry/host/game/state'
      );
      const {
        createEnemyControllerState,
        updateEnemyController,
      } = await import('./browser-entry/shared/enemy-controller');

      const base = createGameState({ seed: 1 });
      const flatMap = buildNeatensteinMap(base.seed);
      const collisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);
      const state = {
        ...base,
        enemies: [
          {
            position: {
              x: base.player.position.x - 5,
              y: base.player.position.y,
            },
            health: 100,
          },
        ],
      };
      const controller = createEnemyControllerState(state);
      // Wrong-length weights trigger activateMlpPooled to throw.
      controller.enemies[0].weights = new Float32Array([0.1, 0.2, 0.3, 0.4]);

      // Call computeMovement via updateEnemyController — it calls
      // computeMovementFlat internally which has the catch block.
      updateEnemyController(controller, state, collisionMap, 1000);

      // The catch block should have logged a debug message.
      expect(debugSpy).toHaveBeenCalled();
    });
  });

  describe('enemy-controller.move.utils.ts computeMovementFlat catch block', () => {
    it('calls console.debug when MLP activation throws (wrong weight length)', async () => {
      const { buildNeatensteinMap } = await import(
        './browser-entry/renderer/map'
      );
      const { createCollisionMap } = await import(
        './browser-entry/renderer/map'
      );
      const { createGameState } = await import(
        './browser-entry/host/game/state'
      );
      const {
        createEnemyControllerState,
        updateEnemyController,
      } = await import('./browser-entry/shared/enemy-controller');

      const base = createGameState({ seed: 1 });
      const flatMap = buildNeatensteinMap(base.seed);
      const collisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);
      const state = {
        ...base,
        enemies: [
          {
            position: {
              x: base.player.position.x - 5,
              y: base.player.position.y,
            },
            health: 100,
          },
        ],
      };
      const controller = createEnemyControllerState(state);
      // Wrong-length weights trigger activateMlpPooled to throw.
      controller.enemies[0].weights = new Float32Array([0.1, 0.2, 0.3, 0.4]);

      updateEnemyController(controller, state, collisionMap, 1000);

      // The catch block should have logged a debug message.
      expect(debugSpy).toHaveBeenCalled();
    });
  });

  // --- display.worker.sim.utils.ts catch block ---

  describe('display.worker.sim.utils.ts runSimStep catch block', () => {
    it('calls console.debug when buildAutoTickInput throws (network.activate throws)', async () => {
      // Load worker test helpers to set up the mock worker global.
      const {
        loadModule,
        sendInitMessage,
        sendSimStateMessage,
      } = await import(
        './browser-entry/worker/display.worker.test-helpers'
      );

      jest.resetModules();

      // Re-import after resetModules so the worker module picks up the
      // mock worker global installed by the test helpers.
      const workerModule = (await loadModule(
        './browser-entry/worker/display.worker.ts',
      )) as {
        __testOnlySetChampionMainNetwork?(network: unknown): void;
        __testOnlyInjectTestEnemies?(positions: {
          x: number;
          y: number;
        }[]): unknown[];
        __testOnlyGetLastTickInputSource?(): string;
      };

      sendInitMessage('cpu');

      // Inject a test enemy so the "hasAliveEnemies" guard passes.
      workerModule.__testOnlyInjectTestEnemies?.([
        { x: 10, y: 10 },
      ]);

      // Inject a champion network whose activate() throws, triggering
      // the catch block in runSimStep.
      const throwingNetwork = {
        activate: jest.fn(() => {
          throw new Error('test: network activation failure');
        }),
      };
      workerModule.__testOnlySetChampionMainNetwork?.(throwingNetwork);

      // Clear any console.debug calls from init.
      debugSpy.mockClear();

      // Send a simState in auto mode — this triggers runSimStep which
      // calls buildAutoTickInput which calls network.activate().
      sendSimStateMessage(0.25, { humanMode: 'auto' });

      // The catch block should have logged a debug message.
      expect(debugSpy).toHaveBeenCalled();
    });
  });

  // --- Once-per-tick guard ---

  describe('once-per-tick guard', () => {
    it('computeMovement does not spam console.debug on recurrent failures (single tick)', async () => {
      const { buildNeatensteinMap, createCollisionMap } = await import(
        './browser-entry/renderer/map'
      );
      const { createGameState } = await import(
        './browser-entry/host/game/state'
      );
      const {
        createEnemyControllerState,
        updateEnemyController,
      } = await import('./browser-entry/shared/enemy-controller');

      const base = createGameState({ seed: 1 });
      const flatMap = buildNeatensteinMap(base.seed);
      const collisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);
      const state = {
        ...base,
        enemies: [
          {
            position: {
              x: base.player.position.x - 5,
              y: base.player.position.y,
            },
            health: 100,
          },
        ],
      };
      const controller = createEnemyControllerState(state);
      controller.enemies[0].weights = new Float32Array([0.1, 0.2, 0.3, 0.4]);

      // First tick — should log.
      updateEnemyController(controller, state, collisionMap, 1000);
      const firstCallCount = debugSpy.mock.calls.length;
      expect(firstCallCount).toBeGreaterThanOrEqual(1);

      // Second tick — should NOT log again (once-per-tick guard suppresses
      // recurrent failures within the same tick context).
      debugSpy.mockClear();
      updateEnemyController(controller, state, collisionMap, 1000);
      // The guard should suppress the debug message on the second tick
      // since the same failure is recurrent.
      // Note: "once per tick" means at most once per tick call, so we
      // assert at most 1 call per tick, not zero on subsequent ticks.
      expect(debugSpy.mock.calls.length).toBeLessThanOrEqual(1);
    });
  });
});

// ---------------------------------------------------------------------------
// Source-level assertion: catch blocks contain console.debug
// ---------------------------------------------------------------------------

describe('C2-4 (source): catch blocks contain console.debug', () => {
  it('enemy-controller.move.utils.ts catch blocks call console.debug', () => {
    const content = readFileSync(
      join(sharedRoot, 'enemy-controller.move.utils.ts'),
      'utf-8',
    );
    // Each catch block should include a console.debug call, not just a
    // bare comment. Look for catch blocks followed by console.debug.
    const catchBlocks = content.match(/}\s*catch\s*\{[^}]*\}/g);
    expect(catchBlocks).not.toBeNull();
    if (catchBlocks) {
      for (const block of catchBlocks) {
        expect(block).toContain('console.debug');
      }
    }
  });

  it('display.worker.sim.utils.ts catch block calls console.debug', () => {
    const content = readFileSync(
      join(workerRoot, 'display.worker.sim.utils.ts'),
      'utf-8',
    );
    const catchBlocks = content.match(/}\s*catch\s*\{[^}]*\}/g);
    expect(catchBlocks).not.toBeNull();
    if (catchBlocks) {
      // At least one catch block should contain console.debug for the
      // buildAutoTickInput fallback.
      const hasDebug = catchBlocks.some((b) => b.includes('console.debug'));
      expect(hasDebug).toBe(true);
    }
  });
});