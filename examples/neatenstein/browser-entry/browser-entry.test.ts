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

import { type NeatensteinStart, type NeatensteinStop } from './browser-entry';
import { NEATENSTEIN_INPUT_MESSAGE_TYPE } from './constants';
import * as constantsNamespace from './constants';

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

function installOffscreenCanvasSupport(): void {
  const OffscreenCanvasStub = class OffscreenCanvasStub {
    width = 0;
    height = 0;
  } as unknown as typeof OffscreenCanvas;
  (globalThis as unknown as Record<string, unknown>).OffscreenCanvas =
    OffscreenCanvasStub;
  (
    HTMLCanvasElement.prototype as unknown as Record<string, unknown>
  ).transferControlToOffscreen = function () {
    return {} as OffscreenCanvas;
  };
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
      <canvas id="neatenstein-canvas"></canvas>
    </div>
  `;
}

function setCanvasSize(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): void {
  Object.defineProperty(canvas, 'clientWidth', {
    configurable: true,
    value: width,
    writable: true,
  });
  Object.defineProperty(canvas, 'clientHeight', {
    configurable: true,
    value: height,
    writable: true,
  });
}

function resetCanvasSize(): void {
  const canvas = document.getElementById(
    'neatenstein-canvas',
  ) as HTMLCanvasElement;
  setCanvasSize(canvas, 854, 480);
}

function setViewportSize(width: number, height: number): void {
  Object.defineProperty(window, 'innerWidth', {
    configurable: true,
    value: width,
    writable: true,
  });
  Object.defineProperty(window, 'innerHeight', {
    configurable: true,
    value: height,
    writable: true,
  });
}

function resetViewportSize(): void {
  setViewportSize(1024, 768);
}

interface MockResizeObserverInstance {
  callback: ResizeObserverCallback;
  targets: Element[];
  disconnected: boolean;
  trigger: () => void;
}

const resizeObserverInstances: MockResizeObserverInstance[] = [];

class MockResizeObserver implements MockResizeObserverInstance {
  callback: ResizeObserverCallback;
  targets: Element[] = [];
  disconnected = false;

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
    resizeObserverInstances.push(this);
  }

  observe(target: Element): void {
    this.targets.push(target);
  }

  disconnect(): void {
    this.disconnected = true;
    this.targets = [];
  }

  trigger(): void {
    if (!this.disconnected) {
      this.callback([], this as unknown as ResizeObserver);
    }
  }
}

function installMockResizeObserver(): void {
  resizeObserverInstances.length = 0;
  globalThis.ResizeObserver =
    MockResizeObserver as unknown as typeof ResizeObserver;
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
    installMockResizeObserver();
    removeOffscreenCanvasSupport();
    resetDocument();
    resetViewportSize();
    resetCanvasSize();
  });

  afterEach(() => {
    jest.restoreAllMocks();
    const globalRecord = globalThis as unknown as Record<string, unknown>;
    const stop = globalRecord.neatensteinStop;
    if (typeof stop === 'function') {
      (stop as () => void)();
    }
  });

  it('matches the canvas backing store to the CSS box aspect ratio with a fixed 480px height', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 854,
      height: 480,
    });
  });

  it('throws when the bundle cannot resolve a host script', async () => {
    Object.defineProperty(document, 'currentScript', {
      value: null,
      configurable: true,
      writable: true,
    });
    await loadModule('./browser-entry.ts');
    const start = getGlobalStart();
    expect(() => start('neatenstein-output', 'neatenstein-canvas')).toThrow(
      'Neatenstein bundle must be loaded through a <script> tag so the worker URL can be resolved relative to the host bundle.',
    );
  });

  it('falls back to the default canvas size when client and viewport dimensions are unavailable', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 0, 0);
    setViewportSize(0, 0);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 640,
      height: 480,
    });
  });

  it('falls back to viewport dimensions when the canvas client dimensions are zero', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 0, 0);
    setViewportSize(1024, 768);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 640,
      height: 480,
    });
  });

  it('selects the worker tier when OffscreenCanvas transfer is available', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    const getContext = jest.spyOn(canvas, 'getContext');
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    const worker = workers[0];
    const initCall = worker?.postMessageCalls[0];
    const initMessage = initCall?.[0] as Record<string, unknown> | undefined;
    expect(initMessage?.tier).toBe('worker');
    expect(getContext).not.toHaveBeenCalled();
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
        workerUrl: 'http://localhost:8080/docs/assets/neatenstein.worker.js',
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

  it('forwards the CSS-derived canvas dimensions in the first render loop tick', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };
    try {
      await loadModule('./browser-entry.ts');
      getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
      const worker = workers[0];
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }
      if (rafCallbacks.length > 0) {
        rafCallbacks[0](0);
      }
      const simStateCall = worker?.postMessageCalls.find((call) => {
        const first = call[0];
        return (
          first !== null &&
          typeof first === 'object' &&
          (first as Record<string, unknown>).type === 'simState'
        );
      });
      expect(simStateCall).toBeDefined();
      const postedState = (
        simStateCall![0] as {
          state: { canvasWidth: number; canvasHeight: number };
        }
      ).state;
      expect({
        width: postedState.canvasWidth,
        height: postedState.canvasHeight,
      }).toEqual({
        width: 854,
        height: 480,
      });
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
    }
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

  it('derives the backing-store width from a non-16:9 CSS box aspect ratio', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 1024, 768);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 640,
      height: 480,
    });
  });

  it('rounds the backing-store width to the nearest integer', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 1920, 1080);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 853,
      height: 480,
    });
  });

  it('allows teardown before the first animation frame is scheduled', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');

    const originalRaf = globalThis.requestAnimationFrame;
    const originalCancel = globalThis.cancelAnimationFrame;
    globalThis.requestAnimationFrame = () => 0;
    globalThis.cancelAnimationFrame = jest.fn();

    let stop: NeatensteinStop | undefined;
    await jest.isolateModulesAsync(async () => {
      jest.doMock('./host/input', () => ({
        createInputRouter: jest.fn(() => ({
          attach: jest.fn(() => jest.fn()),
          detach: jest.fn(),
          getSnapshot: jest.fn(() => ({
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
          })),
        })),
      }));
      await import('./browser-entry.ts');
      const start = (globalThis as unknown as Record<string, unknown>)
        .neatensteinStart as NeatensteinStart;
      stop = start('neatenstein-output', 'neatenstein-canvas');
    });

    try {
      expect(stop).toBeDefined();
      expect(() => stop!()).not.toThrow();
      expect(globalThis.cancelAnimationFrame).toHaveBeenCalledWith(0);
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
      globalThis.cancelAnimationFrame = originalCancel;
    }
  });

  it('updates the canvas backing store when the window is resized', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');

    setCanvasSize(canvas, 1024, 768);
    window.dispatchEvent(new Event('resize'));

    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 640,
      height: 480,
    });
  });

  it('installs a ResizeObserver on the canvas', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    expect(resizeObserverInstances).toHaveLength(1);
    expect(resizeObserverInstances[0]?.targets).toContain(canvas);
  });

  it('updates the canvas backing store when the ResizeObserver fires', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');

    setCanvasSize(canvas, 1024, 768);
    resizeObserverInstances[0]?.trigger();

    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 640,
      height: 480,
    });
  });

  it('disconnects the ResizeObserver during teardown', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    let stop: NeatensteinStop | undefined;
    await jest.isolateModulesAsync(async () => {
      await import('./browser-entry.ts');
      const start = (globalThis as unknown as Record<string, unknown>)
        .neatensteinStart as NeatensteinStart;
      stop = start('neatenstein-output', 'neatenstein-canvas');
    });

    expect(resizeObserverInstances).toHaveLength(1);
    stop?.();
    expect(resizeObserverInstances[0]?.disconnected).toBe(true);
  });

  it('falls back to window resize events when ResizeObserver is unavailable', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    delete (globalThis as unknown as Record<string, unknown>).ResizeObserver;
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');

    expect(resizeObserverInstances).toHaveLength(0);
    setCanvasSize(canvas, 1024, 768);
    window.dispatchEvent(new Event('resize'));

    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 640,
      height: 480,
    });
  });

  it('stops updating the canvas backing store after teardown', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    let stop: NeatensteinStop | undefined;
    await jest.isolateModulesAsync(async () => {
      await import('./browser-entry.ts');
      const start = (globalThis as unknown as Record<string, unknown>)
        .neatensteinStart as NeatensteinStart;
      stop = start('neatenstein-output', 'neatenstein-canvas');
    });

    setCanvasSize(canvas, 1024, 768);
    window.dispatchEvent(new Event('resize'));
    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 640,
      height: 480,
    });

    stop?.();
    setCanvasSize(canvas, 1280, 720);
    window.dispatchEvent(new Event('resize'));
    expect({ width: canvas.width, height: canvas.height }).toEqual({
      width: 640,
      height: 480,
    });
  });

  it('routes worker-tier resizes to the bridge instead of mutating the transferred canvas', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    await loadModule('./browser-entry.ts');
    getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
    const worker = workers[0];

    const initialWidth = canvas.width;
    const initialHeight = canvas.height;

    setCanvasSize(canvas, 1024, 768);
    window.dispatchEvent(new Event('resize'));

    expect(canvas.width).toBe(initialWidth);
    expect(canvas.height).toBe(initialHeight);

    const resizeCall = worker?.postMessageCalls.find((call) => {
      const first = call[0];
      return (
        first !== null &&
        typeof first === 'object' &&
        (first as Record<string, unknown>).type === 'resize'
      );
    });
    expect(resizeCall).toBeDefined();
    const resizeMessage = resizeCall![0] as {
      type: string;
      width: number;
      height: number;
    };
    expect({
      width: resizeMessage.width,
      height: resizeMessage.height,
    }).toEqual({
      width: 640,
      height: 480,
    });
  });

  it('uses CSS-derived dimensions for the first simState in the worker tier', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };
    try {
      await loadModule('./browser-entry.ts');
      getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
      const worker = workers[0];

      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }

      setCanvasSize(canvas, 1024, 768);
      window.dispatchEvent(new Event('resize'));

      if (rafCallbacks.length > 0) {
        rafCallbacks[0](0);
      }

      const simStateCall = worker?.postMessageCalls.find((call) => {
        const first = call[0];
        return (
          first !== null &&
          typeof first === 'object' &&
          (first as Record<string, unknown>).type === 'simState'
        );
      });
      expect(simStateCall).toBeDefined();
      const postedState = (
        simStateCall![0] as {
          state: { canvasWidth: number; canvasHeight: number };
        }
      ).state;
      expect({
        width: postedState.canvasWidth,
        height: postedState.canvasHeight,
      }).toEqual({
        width: 640,
        height: 480,
      });
      expect(canvas.width).not.toBe(postedState.canvasWidth);
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
    }
  });

  it('posts simState on every animation frame (per-frame delta-time posting)', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };
    try {
      await loadModule('./browser-entry.ts');
      getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
      const worker = workers[0];

      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }

      // Drive 7 animation frames. Send a frame ack after each tick so the
      // bridge backpressure allows the next simState to post immediately.
      const timestamps = [0, 16, 33, 49, 66, 83, 99];
      for (let i = 0; i < timestamps.length; i += 1) {
        rafCallbacks[i](timestamps[i]);
        if (typeof worker.onmessage === 'function') {
          worker.onmessage({
            data: { type: 'frame', frame: { requestId: i } },
          } as unknown as MessageEvent);
        }
      }

      const simStateCalls = worker?.postMessageCalls.filter((call) => {
        const first = call[0];
        return (
          first !== null &&
          typeof first === 'object' &&
          (first as Record<string, unknown>).type === 'simState'
        );
      });
      const inputCalls = worker?.postMessageCalls.filter((call) => {
        const first = call[0];
        return (
          first !== null &&
          typeof first === 'object' &&
          (first as Record<string, unknown>).type ===
            NEATENSTEIN_INPUT_MESSAGE_TYPE
        );
      });

      // Per-frame posting: one simState per animation frame (with backpressure
      // ack between frames).
      expect(simStateCalls?.length).toBe(timestamps.length);
      // Input forwarding must remain unthrottled for responsive look.
      expect(inputCalls?.length).toBe(timestamps.length);
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
    }
  });

  it('includes a delta-time derived from rAF timestamps in the posted simState', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };
    try {
      await loadModule('./browser-entry.ts');
      getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
      const worker = workers[0];

      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }

      // Drive two frames 50ms apart; the second posted simState must carry
      // deltaMs === 50 (derived from consecutive rAF timestamps). Send a
      // frame ack after the first tick so backpressure allows the second
      // simState to post.
      rafCallbacks[0](0);
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame: { requestId: 0 } },
        } as unknown as MessageEvent);
      }
      rafCallbacks[1](50);

      const simStateCalls = worker?.postMessageCalls.filter((call) => {
        const first = call[0];
        return (
          first !== null &&
          typeof first === 'object' &&
          (first as Record<string, unknown>).type === 'simState'
        );
      });
      expect(simStateCalls?.length).toBeGreaterThanOrEqual(2);

      const secondState = simStateCalls?.[1]?.[0] as
        Record<string, unknown> | undefined;
      const state = secondState?.state as Record<string, unknown> | undefined;
      expect(state?.deltaMs).toBe(50);
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
    }
  });

  it('advances simTick and cameraYaw across consecutive worker-paced ticks', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };

    const yawDelta = 0.1;
    const inputSnapshot = {
      timestamp: 0,
      movement: {
        forward: false,
        backward: false,
        left: false,
        right: false,
      },
      look: { yawDelta, pitchDelta: 0 },
      touch: { active: false, yawDelta: 0, pitchDelta: 0 },
      pointerLocked: false,
      fire: false,
    };
    const router = {
      attach: jest.fn(() => jest.fn()),
      detach: jest.fn(),
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
      const worker = workers[0];
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }

      // First tick → simState posted, worker busy.
      rafCallbacks[0](0);

      // Frame ack → onFrameReady → next rAF scheduled.
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame: { requestId: 0 } },
        } as unknown as MessageEvent);
      }

      // Second tick → simState posted, worker busy.
      rafCallbacks[1](16);

      // Frame ack → onFrameReady → next rAF scheduled.
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame: { requestId: 1 } },
        } as unknown as MessageEvent);
      }

      // Third tick → simState posted, worker busy.
      rafCallbacks[2](33);

      const simStateCalls = worker?.postMessageCalls.filter((call) => {
        const first = call[0];
        return (
          first !== null &&
          typeof first === 'object' &&
          (first as Record<string, unknown>).type === 'simState'
        );
      });
      expect(simStateCalls?.length).toBe(3);

      // Each successive state must have a higher simTick and cameraYaw,
      // proving the tick advanced across consecutive worker-paced frames.
      const firstState = (
        simStateCalls?.[0]?.[0] as {
          state: { simTick: number; cameraYaw: number };
        }
      ).state;
      const secondState = (
        simStateCalls?.[1]?.[0] as {
          state: { simTick: number; cameraYaw: number };
        }
      ).state;
      const thirdState = (
        simStateCalls?.[2]?.[0] as {
          state: { simTick: number; cameraYaw: number };
        }
      ).state;
      expect(secondState.simTick).toBeGreaterThan(firstState.simTick);
      expect(thirdState.simTick).toBeGreaterThan(secondState.simTick);
      expect(secondState.cameraYaw).toBeGreaterThan(firstState.cameraYaw);
      expect(thirdState.cameraYaw).toBeGreaterThan(secondState.cameraYaw);
    } finally {
      stop?.();
      globalThis.requestAnimationFrame = originalRaf;
    }
  });

  it('tick does not schedule a new rAF — the loop is worker-paced', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };
    try {
      await loadModule('./browser-entry.ts');
      getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
      const worker = workers[0];

      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }

      // The initial rAF schedules one callback.
      const rafCountAfterStart = rafCallbacks.length;
      expect(rafCountAfterStart).toBe(1);

      // Run the tick. In the worker-paced loop, tick must NOT schedule
      // a new rAF — that is now the responsibility of onFrameReady.
      rafCallbacks[0](0);

      expect(rafCallbacks.length).toBe(rafCountAfterStart);
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
    }
  });

  it('onFrameReady schedules the next rAF after a frame ack with no pending state', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };
    try {
      await loadModule('./browser-entry.ts');
      getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
      const worker = workers[0];

      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }

      // Run the initial tick (posts simState, worker busy).
      rafCallbacks[0](0);
      const rafCountAfterTick = rafCallbacks.length;

      // Send a frame ack → onFrameReady → next rAF scheduled.
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame: { requestId: 0 } },
        } as unknown as MessageEvent);
      }

      expect(rafCallbacks.length).toBe(rafCountAfterTick + 1);
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
    }
  });

  it('full render cycle works end-to-end: tick → simState → ack → onFrameReady → rAF → tick', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };
    try {
      await loadModule('./browser-entry.ts');
      getGlobalStart()('neatenstein-output', 'neatenstein-canvas');
      const worker = workers[0];

      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }

      // Initial rAF → first tick → simState posted.
      rafCallbacks[0](0);
      const simStateCallsAfterFirstTick = worker?.postMessageCalls.filter(
        (call) => {
          const first = call[0];
          return (
            first !== null &&
            typeof first === 'object' &&
            (first as Record<string, unknown>).type === 'simState'
          );
        },
      );
      expect(simStateCallsAfterFirstTick?.length).toBe(1);

      // Frame ack → onFrameReady → next rAF scheduled.
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame: { requestId: 0 } },
        } as unknown as MessageEvent);
      }
      expect(rafCallbacks.length).toBe(2);

      // Second tick → second simState posted.
      rafCallbacks[1](16);
      const simStateCallsAfterSecondTick = worker?.postMessageCalls.filter(
        (call) => {
          const first = call[0];
          return (
            first !== null &&
            typeof first === 'object' &&
            (first as Record<string, unknown>).type === 'simState'
          );
        },
      );
      expect(simStateCallsAfterSecondTick?.length).toBe(2);
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
    }
  });

  it('death feedback shows weaker direction when hive density decreases between frames', async () => {
    setHostScript('http://localhost:8080/docs/assets/neatenstein.bundle.js');
    installOffscreenCanvasSupport();
    const canvas = document.getElementById(
      'neatenstein-canvas',
    ) as HTMLCanvasElement;
    setCanvasSize(canvas, 854, 480);
    const rafCallbacks: FrameRequestCallback[] = [];
    const originalRaf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
      rafCallbacks.push(callback);
      return rafCallbacks.length;
    };

    // Mock createGameState to return a state with 8 enemies so hiveDensity = 1.
    // We keep a reference to the enemies array so we can mutate it between ticks.
    const mockEnemies: Array<{
      position: { x: number; y: number };
      health: number;
    }> = [];
    for (let i = 0; i < 8; i += 1) {
      mockEnemies.push({ position: { x: 0, y: 0 }, health: 1 });
    }

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
      attach: jest.fn(() => jest.fn()),
      detach: jest.fn(),
      getSnapshot: jest.fn(() => inputSnapshot),
    };

    let stop: NeatensteinStop | undefined;
    await jest.isolateModulesAsync(async () => {
      jest.doMock('./host/input', () => ({
        createInputRouter: jest.fn(() => router),
      }));
      jest.doMock('./host/game/state', () => ({
        createGameState: jest.fn(() => ({
          seed: 1,
          simTimeMs: 0,
          episodeTimeMs: 0,
          player: {
            position: { x: 0, y: 0 },
            previousPosition: { x: 0, y: 0 },
            angleRad: 0,
            health: 100,
            maxHealth: 100,
            ammo: 100,
            maxAmmo: 100,
            dashTimeRemainingMs: 0,
            dashCooldownMs: 0,
            contactIFrameMs: 0,
          },
          enemies: mockEnemies,
          impacts: [],
          gun: { recoilOffset: 0 },
          bolts: [],
          enemyBolts: [],
          kills: 0,
          spawnCount: 0,
          generation: 1,
        })),
      }));
      await import('./browser-entry.ts');
      const start = (globalThis as unknown as Record<string, unknown>)
        .neatensteinStart as NeatensteinStart;
      stop = start('neatenstein-output', 'neatenstein-canvas');
    });

    try {
      const worker = workers[0];
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }

      // First tick: hiveDensity = 8/8 = 1, prevHiveDensity = 0 → delta = 1
      // → direction 'stronger'. This sets prevHiveDensity to 1.
      rafCallbacks[0](0);
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame: { requestId: 0 } },
        } as unknown as MessageEvent);
      }

      // Clear enemies so hiveDensity drops to 0 on the next tick.
      mockEnemies.length = 0;

      // Second tick: hiveDensity = 0/8 = 0, prevHiveDensity = 1 → delta = -1
      // → direction 'weaker'. The death feedback label should show 'WEAKER'.
      rafCallbacks[1](16);

      const deathFeedbackEl = document.querySelector(
        '[data-role="death-feedback"]',
      );
      expect(deathFeedbackEl).not.toBeNull();
      const label = deathFeedbackEl?.querySelector('div');
      expect(label?.textContent).toBe('DEATH FEEDBACK: WEAKER');
    } finally {
      stop?.();
      globalThis.requestAnimationFrame = originalRaf;
    }
  });

  it('no longer exports the removed fixed-timestep and throttle constants', async () => {
    // Slice 10.3-raf-clock removes NEATENSTEIN_FIXED_TIMESTEP_MS from
    // constants and NEATENSTEIN_HOST_POST_INTERVAL_MS from browser-entry.
    // constants is checked via the static namespace import (already loaded
    // at file evaluation time). browser-entry is checked via an isolated
    // re-import so the runtime module shape is verified without conflicting
    // with the module cache entries created by loadModule in prior tests.
    expect(constantsNamespace).not.toHaveProperty(
      'NEATENSTEIN_FIXED_TIMESTEP_MS',
    );

    let browserEntryModule: Record<string, unknown> | undefined;
    await (
      jest as unknown as Record<string, (...args: unknown[]) => unknown>
    ).isolateModulesAsync(async () => {
      browserEntryModule = (await import('./browser-entry.ts')) as Record<
        string,
        unknown
      >;
    });
    expect(browserEntryModule).not.toHaveProperty(
      'NEATENSTEIN_HOST_POST_INTERVAL_MS',
    );
  });
});
