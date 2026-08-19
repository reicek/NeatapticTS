import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import type { GameState } from '../host/game/types';
import {
  loadModule,
  sendInitMessage,
  sendSimStateMessage,
} from './display.worker.test-helpers';

describe('eval worker helpers coverage', () => {
  let originalWorkerCtor: typeof Worker | undefined;
  const g = globalThis as unknown as Record<string, unknown>;
  const s = self as unknown as Record<string, unknown>;

  beforeEach(() => {
    originalWorkerCtor = g.Worker as typeof Worker | undefined;
  });

  afterEach(() => {
    if (originalWorkerCtor === undefined) {
      delete g.Worker;
    } else {
      g.Worker = originalWorkerCtor;
    }
    delete s.location;
  });

  it('derives the eval worker URL from self.location and instantiates Worker', async () => {
    jest.resetModules();

    s.location = {
      href: 'http://example/assets/neatenstein.worker.js',
    };

    const created: {
      url: string;
      onmessage: unknown;
      postMessage: jest.Mock;
    }[] = [];
    g.Worker = class MockWorker {
      url: string;

      onmessage: unknown = null;

      postMessage = jest.fn();

      constructor(url: string) {
        this.url = url;
        created.push(this);
      }
    } as unknown as typeof Worker;

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyGetEvalWorker?(): unknown;
      __testOnlySetEvalWorker?(
        worker: {
          postMessage: (msg: unknown) => void;
          onmessage: unknown;
        } | null,
      ): void;
    };

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest
      .spyOn(realTick, 'gameTick')
      .mockImplementation(
        (state: GameState) => ({ ...state, enemies: [] }) as GameState,
      );

    sendInitMessage('cpu');
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 13.5, y: 12.5 }]);
    sendSimStateMessage(0.25, { humanMode: 'auto' });

    expect(created.length).toBe(1);
    expect(created[0].url).toBe(
      'http://example/assets/neatenstein.eval-worker.js',
    );
    expect(workerModule.__testOnlyGetEvalWorker?.()).toBe(created[0]);

    workerModule.__testOnlySetEvalWorker?.(null);
    gameTickSpy.mockRestore();
  });

  it('returns null when self.location access throws', async () => {
    jest.resetModules();

    Object.defineProperty(self, 'location', {
      configurable: true,
      get: () => {
        throw new Error('bad location');
      },
    });

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyGetEvalWorker?(): unknown;
      __testOnlySetEvalWorker?(
        worker: {
          postMessage: (msg: unknown) => void;
          onmessage: unknown;
        } | null,
      ): void;
    };

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest
      .spyOn(realTick, 'gameTick')
      .mockImplementation(
        (state: GameState) => ({ ...state, enemies: [] }) as GameState,
      );

    sendInitMessage('cpu');
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 13.5, y: 12.5 }]);
    expect(() =>
      sendSimStateMessage(0.25, { humanMode: 'auto' }),
    ).not.toThrow();
    expect(workerModule.__testOnlyGetEvalWorker?.()).toBeNull();

    workerModule.__testOnlySetEvalWorker?.(null);
    gameTickSpy.mockRestore();
  });

  it('falls back to null when the Worker constructor throws', async () => {
    jest.resetModules();

    s.location = {
      href: 'http://example/assets/neatenstein.worker.js',
    };

    g.Worker = class BadWorker {
      constructor() {
        throw new Error('no worker support');
      }
    } as unknown as typeof Worker;

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyGetEvalWorker?(): unknown;
      __testOnlySetEvalWorker?(
        worker: {
          postMessage: (msg: unknown) => void;
          onmessage: unknown;
        } | null,
      ): void;
    };

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest
      .spyOn(realTick, 'gameTick')
      .mockImplementation(
        (state: GameState) => ({ ...state, enemies: [] }) as GameState,
      );

    sendInitMessage('cpu');
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 13.5, y: 12.5 }]);
    expect(() =>
      sendSimStateMessage(0.25, { humanMode: 'auto' }),
    ).not.toThrow();
    expect(workerModule.__testOnlyGetEvalWorker?.()).toBeNull();

    workerModule.__testOnlySetEvalWorker?.(null);
    gameTickSpy.mockRestore();
  });

  it('returns null when the Worker constructor is unavailable', async () => {
    jest.resetModules();

    s.location = {
      href: 'http://example/assets/neatenstein.worker.js',
    };

    delete g.Worker;

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyInjectTestEnemies?(positions: { x: number; y: number }[]): void;
      __testOnlyGetEvalWorker?(): unknown;
      __testOnlySetEvalWorker?(
        worker: {
          postMessage: (msg: unknown) => void;
          onmessage: unknown;
        } | null,
      ): void;
    };

    const realTick = await import('../host/game/tick');
    const gameTickSpy = jest
      .spyOn(realTick, 'gameTick')
      .mockImplementation(
        (state: GameState) => ({ ...state, enemies: [] }) as GameState,
      );

    sendInitMessage('cpu');
    workerModule.__testOnlyInjectTestEnemies?.([{ x: 13.5, y: 12.5 }]);
    expect(() =>
      sendSimStateMessage(0.25, { humanMode: 'auto' }),
    ).not.toThrow();
    expect(workerModule.__testOnlyGetEvalWorker?.()).toBeNull();

    workerModule.__testOnlySetEvalWorker?.(null);
    gameTickSpy.mockRestore();
  });

  it('ignores evalComplete messages with invalid payloads', async () => {
    jest.resetModules();

    const mockEvalWorker = {
      postMessage: jest.fn(),
      onmessage: null as ((event: MessageEvent) => void) | null,
    };

    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlySetEvalWorker?(
        worker: {
          postMessage: (msg: unknown) => void;
          onmessage: unknown;
        } | null,
      ): void;
    };

    sendInitMessage('cpu');
    workerModule.__testOnlySetEvalWorker?.(mockEvalWorker);

    expect(mockEvalWorker.onmessage).not.toBeNull();
    const handler = mockEvalWorker.onmessage as (event: MessageEvent) => void;

    expect(() => handler({ data: null } as MessageEvent)).not.toThrow();
    expect(() => handler({ data: 123 } as MessageEvent)).not.toThrow();
    expect(() =>
      handler({ data: { type: 'notEvalComplete' } } as MessageEvent),
    ).not.toThrow();
  });
});
