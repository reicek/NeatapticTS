/** @jest-environment jsdom */

import type {
  BrowserEntryStartFunction,
  RuntimeWindow,
} from './browser-entry.types';
import {
  createBrowserEntryEvolutionHostAdapter,
  installBrowserEntryGlobals,
} from './browser-entry.globals.services';
import { BROWSER_ENTRY_CONSTANTS as C } from './browser-entry.constants';
import { animateSolvedMazeInBrowserHost } from './browser-entry.solved-maze-animation.services';

jest.mock('./browser-entry.solved-maze-animation.services', () => ({
  animateSolvedMazeInBrowserHost: jest.fn(() => Promise.resolve()),
}));

const mockedAnimateSolvedMazeInBrowserHost = jest.mocked(
  animateSolvedMazeInBrowserHost,
);

describe('createBrowserEntryEvolutionHostAdapter', () => {
  beforeEach(() => {
    mockedAnimateSolvedMazeInBrowserHost.mockClear();
  });

  it('reports pause state from the browser global flag', () => {
    const globalWindow = window as unknown as RuntimeWindow;
    globalWindow.asciiMazePaused = true;

    const adapter = createBrowserEntryEvolutionHostAdapter();

    expect(adapter.isPauseRequested?.()).toBe(true);
  });

  it('animates solved mazes before dispatching the solved event and respects host pause requests', async () => {
    document.body.innerHTML = '<div id="live-output"></div>';
    const liveElement = document.getElementById('live-output');
    if (!liveElement) {
      throw new Error('Expected a live output host element');
    }

    let releaseAnimation: (() => void) | undefined;
    mockedAnimateSolvedMazeInBrowserHost.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          releaseAnimation = resolve;
        }),
    );

    const receivedEvents: Array<CustomEvent<{ progress: number }>> = [];
    window.addEventListener('asciiMazeSolved', (event) => {
      receivedEvents.push(event as CustomEvent<{ progress: number }>);
    });

    const adapter = createBrowserEntryEvolutionHostAdapter({ liveElement });
    const stopPromise = adapter.handleStop?.({
      reason: 'solved',
      maze: ['S.E'],
      completedGenerations: 7,
      progress: 100,
      result: {
        path: [
          [0, 0],
          [1, 0],
          [2, 0],
        ],
        progress: 100,
        success: true,
      } as never,
      requestHostPause: true,
    });

    await Promise.resolve();
    releaseAnimation?.();
    await stopPromise;

    expect({
      animatedPathLength:
        mockedAnimateSolvedMazeInBrowserHost.mock.calls[0]?.[0].path?.length ??
        0,
      eventCount: receivedEvents.length,
      paused: (window as unknown as RuntimeWindow).asciiMazePaused,
      progress: receivedEvents[0]?.detail.progress,
    }).toEqual({
      animatedPathLength: 3,
      eventCount: 1,
      paused: true,
      progress: 100,
    });
  });

  it('ignores non-solved stop reasons without animating or dispatching a solved event', async () => {
    const adapter = createBrowserEntryEvolutionHostAdapter();
    let solvedEventCount = 0;

    window.addEventListener('asciiMazeSolved', () => {
      solvedEventCount += 1;
    });

    await adapter.handleStop?.({
      reason: 'maxGenerations',
      maze: ['S.E'],
      completedGenerations: 7,
      progress: 40,
    });

    expect({
      animationCallCount:
        mockedAnimateSolvedMazeInBrowserHost.mock.calls.length,
      solvedEventCount,
    }).toEqual({
      animationCallCount: 0,
      solvedEventCount: 0,
    });
  });

  it('swallows browser dispatch failures so the host adapter still resolves', async () => {
    const dispatchEventSpy = jest
      .spyOn(window, 'dispatchEvent')
      .mockImplementation(() => {
        throw new Error('dispatch failed');
      });

    const adapter = createBrowserEntryEvolutionHostAdapter();

    await expect(
      Promise.resolve(
        adapter.handleStop?.({
          reason: 'solved',
          maze: ['S.E'],
          completedGenerations: 2,
          progress: 100,
          result: {
            path: [[0, 0]],
            progress: 100,
            success: true,
          } as never,
        }),
      ),
    ).resolves.toBeUndefined();

    dispatchEventSpy.mockRestore();
  });

  it('returns a safe no-op adapter when the browser window is unavailable', async () => {
    const adapter = createBrowserEntryEvolutionHostAdapter({
      runtimeWindow: null,
    });
    await Promise.resolve(
      adapter.handleStop?.({
        reason: 'solved',
        maze: ['S.E'],
        completedGenerations: 1,
        progress: 100,
        result: {
          path: [[0, 0]],
          progress: 100,
          success: true,
        } as never,
      }),
    );

    expect({
      animationCallCount:
        mockedAnimateSolvedMazeInBrowserHost.mock.calls.length,
      pauseRequested: adapter.isPauseRequested?.(),
    }).toEqual({
      animationCallCount: 0,
      pauseRequested: false,
    });
  });
});

describe('installBrowserEntryGlobals', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    document.body.innerHTML = '<div id="ascii-maze-output"></div>';
    const globalWindow = window as unknown as RuntimeWindow;
    delete globalWindow.asciiMaze;
    delete globalWindow.asciiMazeStart;
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('installs the stable globals bridge and auto-starts once when the default host exists', () => {
    const startCalls: unknown[] = [];
    const start = jest.fn(async (...args: unknown[]) => {
      startCalls.push(args);
      return { done: Promise.resolve() } as never;
    }) as unknown as BrowserEntryStartFunction;

    installBrowserEntryGlobals(start);
    const globalWindow = window as unknown as RuntimeWindow;
    globalWindow.asciiMazeStart?.('custom-container');
    jest.advanceTimersByTime(C.AUTO_START_DELAY_MS);

    expect({
      autoStarted: globalWindow.asciiMaze?._autoStarted,
      deprecatedStartCallCount: startCalls.length,
      globalsStartMatches: globalWindow.asciiMaze?.start === start,
    }).toEqual({
      autoStarted: true,
      deprecatedStartCallCount: 2,
      globalsStartMatches: true,
    });
  });

  it('preserves an existing deprecated start bridge and skips duplicate auto-start wiring', () => {
    const start = jest.fn(
      async () => ({ done: Promise.resolve() }) as never,
    ) as unknown as BrowserEntryStartFunction;
    const globalWindow = window as unknown as RuntimeWindow;
    const existingDeprecatedStart = jest.fn();
    globalWindow.asciiMaze = { _autoStarted: true };
    globalWindow.asciiMazeStart = existingDeprecatedStart;

    installBrowserEntryGlobals(start);
    jest.advanceTimersByTime(C.AUTO_START_DELAY_MS);

    expect({
      deprecatedStartPreserved:
        globalWindow.asciiMazeStart === existingDeprecatedStart,
      startCallCount: jest.mocked(start).mock.calls.length,
    }).toEqual({
      deprecatedStartPreserved: true,
      startCallCount: 0,
    });
  });

  it('returns immediately when the browser window is unavailable', () => {
    const start = jest.fn(
      async () => ({ done: Promise.resolve() }) as never,
    ) as unknown as BrowserEntryStartFunction;

    installBrowserEntryGlobals(start, null);

    expect(jest.mocked(start).mock.calls.length).toBe(0);
  });

  it('skips auto-start when the default host element is missing', () => {
    document.body.innerHTML = '';
    const start = jest.fn(
      async () => ({ done: Promise.resolve() }) as never,
    ) as unknown as BrowserEntryStartFunction;

    installBrowserEntryGlobals(start);
    jest.advanceTimersByTime(C.AUTO_START_DELAY_MS);

    expect(jest.mocked(start).mock.calls.length).toBe(0);
  });
});
