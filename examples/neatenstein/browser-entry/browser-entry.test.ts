/**
 * @jest-environment jsdom
 */

import {
  describe,
  expect,
  it,
  jest,
  beforeEach,
  afterEach,
} from '@jest/globals';

import type { NeatensteinStart, NeatensteinStop } from './browser-entry';

interface MockWorkerInstance {
  url: string | URL;
  options?: WorkerOptions;
  terminated: boolean;
  postMessageCalls: unknown[][];
  onmessage?: ((event: MessageEvent) => void) | null;
  onerror?: ((event: ErrorEvent) => void) | null;
}

const workers: MockWorkerInstance[] = [];

function installMockWorker(): void {
  class MockWorker implements MockWorkerInstance {
    url: string | URL;
    options?: WorkerOptions;
    terminated = false;
    postMessageCalls: unknown[][] = [];
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: ErrorEvent) => void) | null = null;

    constructor(url: string | URL, options?: WorkerOptions) {
      this.url = url;
      this.options = options;
      workers.push(this);
    }

    postMessage(...args: unknown[]): void {
      this.postMessageCalls.push(args);
    }

    terminate(): void {
      this.terminated = true;
    }
  }

  globalThis.Worker = MockWorker as unknown as typeof Worker;
}

function removeOffscreenCanvasSupport(): void {
  if ('OffscreenCanvas' in globalThis) {
    delete (globalThis as unknown as Record<string, unknown>).OffscreenCanvas;
  }
  (
    HTMLCanvasElement.prototype as unknown as Record<string, unknown>
  ).transferControlToOffscreen = undefined;
}

function setHostScript(src: string): void {
  const script = document.createElement('script');
  script.src = src;
  Object.defineProperty(document, 'currentScript', {
    value: script,
    configurable: true,
    writable: true,
  });
}

function resetDocument(): void {
  document.body.innerHTML = `
    <p id="status" role="status" aria-live="polite"></p>
    <div id="neatenstein-output">
      <canvas id="neatenstein-canvas" style="width:640px;height:360px;"></canvas>
    </div>
  `;
}

const loadModule = async (path: string): Promise<void> => {
  await (
    jest as unknown as Record<string, (...args: unknown[]) => unknown>
  ).isolateModulesAsync(async () => {
    await import(path);
  });
};

function getGlobalStart(): NeatensteinStart {
  const globalRecord = globalThis as unknown as Record<string, unknown>;
  const start = globalRecord.neatensteinStart;
  if (typeof start !== 'function') {
    throw new Error('neatensteinStart was not registered on globalThis');
  }
  return start as NeatensteinStart;
}

describe('Neatenstein browser entry', () => {
  beforeEach(() => {
    workers.length = 0;
    installMockWorker();
    removeOffscreenCanvasSupport();
    resetDocument();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('matches the canvas backing store to its CSS pixel size', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 640,
      height: 360,
    });
  });

  it('throws an actionable error when the canvas element is missing', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    await loadModule('./browser-entry.ts');
    const start = getGlobalStart();
    expect(() => start('neatenstein-output', 'missing-canvas')).toThrow(
      'Canvas element #missing-canvas not found',
    );
  });

  it('resolves the worker URL relative to the host script', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    expect({ workerCount: workers.length, workerUrl: workers[0]?.url }).toEqual(
      {
        workerCount: 1,
        workerUrl:
          'http://localhost:8080/docs/assets/neatenstein.worker.esm.js',
      },
    );
  });

  it('starts the render loop by scheduling an animation frame', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };
    try {
      await loadModule('./browser-entry.ts');
      getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
    }
    expect(rafCallbacks.length).toBeGreaterThan(0);
  });

  it('falls back to the CPU tier and draws a status message when OffscreenCanvas is unavailable', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    const fillText = jest.fn();
    jest.spyOn(canvas, 'getContext').mockReturnValue({
      fillRect: jest.fn(),
      fillText,
    } as unknown as CanvasRenderingContext2D);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    expect({
      workerCount: workers.length,
      fillTextCalled: fillText.mock.calls.length > 0,
      lastFillText: fillText.mock.calls.at(-1)?.[0],
    }).toEqual({
      workerCount: 1,
      fillTextCalled: true,
      lastFillText:
        'OffscreenCanvas not available; using CPU fallback renderer.',
    });
  });

  it('teardown cancels the render loop, detaches the input router, and terminates the worker', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');

    const rafCallbacks: FrameRequestCallback[] = [];
    const canceledRafIds: number[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    const originalCancel = globalThis.cancelAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };
    globalThis.cancelAnimationFrame = (id: number) => {
      canceledRafIds.push(id);
    };

    const detach = jest.fn();
    const inputSnapshot = {
      timestamp: 0,
      movement: {
        forward: false,
        backward: false,
        left: false,
        right: false,
      },
      look: { yawDelta: 0, pitchDelta: 0 },
      touch: { active: false, yawDelta: 0, pitchDelta: 0 },
      pointerLocked: false,
      fire: false,
    };
    const router = {
      attach: jest.fn(() => detach),
      detach,
      getSnapshot: jest.fn(() => inputSnapshot),
    };

    let stop: NeatensteinStop | undefined;
    await jest.isolateModulesAsync(async () => {
      jest.doMock('./host/input', () => ({
        createInputRouter: jest.fn(() => router),
      }));
      await import('./browser-entry.ts');
      const start = (globalThis as unknown as Record<string, unknown>)
        .neatensteinStart as NeatensteinStart;
      stop = start('neatenstein-output', 'neatenstein-canvas');
    });

    try {
      expect(stop).toBeDefined();
      stop!();

      expect(workers).toHaveLength(1);
      expect(workers[0]?.terminated).toBe(true);
      expect(canceledRafIds).toContain(rafCallbacks.length);
      expect(detach).toHaveBeenCalledTimes(1);
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
      globalThis.cancelAnimationFrame = originalCancel;
    }
  });
});
