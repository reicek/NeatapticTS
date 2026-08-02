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

import {
  NEATENSTEIN_HOST_POST_INTERVAL_MS,
  type NeatensteinStart,
  type NeatensteinStop,
} from './browser-entry';
import { NEATENSTEIN_INPUT_MESSAGE_TYPE } from './constants';

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

  it('throttles simState posts to roughly 30 fps while still forwarding input every frame', async () => {
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

      const timestamps = [0, 16, 33, 49, 66, 83, 99];
      for (let i = 0; i < timestamps.length; i += 1) {
        rafCallbacks[i](timestamps[i]);
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

      // Model the throttle to verify the observed post count matches the
      // configured interval rather than the raw frame count.
      let expectedPostCount = 0;
      let lastPostTimestamp = -NEATENSTEIN_HOST_POST_INTERVAL_MS;
      for (const timestamp of timestamps) {
        if (
          timestamp - lastPostTimestamp >=
          NEATENSTEIN_HOST_POST_INTERVAL_MS
        ) {
          expectedPostCount += 1;
          lastPostTimestamp = timestamp;
        }
      }

      expect(simStateCalls?.length).toBe(expectedPostCount);
      // Input forwarding must remain unthrottled for responsive look.
      expect(inputCalls?.length).toBe(timestamps.length);
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
    }
  });
});
