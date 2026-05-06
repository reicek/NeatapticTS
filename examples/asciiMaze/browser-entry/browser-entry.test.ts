/** @jest-environment jsdom */

import {
  composeBrowserEntryAbortSignal,
  createBrowserEntryEvolutionHostAdapter,
  createBrowserEntryHostServices,
  installBrowserEntryGlobals,
  runBrowserEntryCurriculum,
} from './browser-entry.services';
import { start } from './browser-entry';
import { resolveBrowserEntryHostElements } from './browser-entry.utils';

jest.mock('./browser-entry.services', () => ({
  composeBrowserEntryAbortSignal: jest.fn(
    (internalController: AbortController, externalSignal?: AbortSignal) =>
      externalSignal ?? internalController.signal,
  ),
  createBrowserEntryEvolutionHostAdapter: jest.fn(() => ({
    isPauseRequested: () => false,
  })),
  createBrowserEntryHostServices: jest.fn(() => ({
    dashboard: {},
    runtimeDashboard: { getLastTelemetry: jest.fn() },
    telemetryHub: { add: jest.fn(() => () => {}) },
    disposeResizeHandling: jest.fn(),
  })),
  installBrowserEntryGlobals: jest.fn(),
  runBrowserEntryCurriculum: jest.fn(),
}));

jest.mock('./browser-entry.utils', () => ({
  resolveBrowserEntryHostElements: jest.fn(),
}));

const mockedResolveBrowserEntryHostElements = jest.mocked(
  resolveBrowserEntryHostElements,
);
const mockedRunBrowserEntryCurriculum = jest.mocked(runBrowserEntryCurriculum);
const disposeResizeHandling = jest.fn();
const getLastTelemetry = jest.fn(() => ({
  bestFitness: null,
  details: null,
  elapsedMs: 0,
  generation: 12,
  gensPerSec: 0,
  progress: null,
  speciesCount: null,
}));
const addTelemetryListener = jest.fn(() => () => {});
const dispatchTelemetry = jest.fn();

describe('asciiMaze browser entry start()', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="ascii-maze-output"></div>';
    mockedResolveBrowserEntryHostElements.mockReturnValue({
      hostElement: document.getElementById('ascii-maze-output'),
      archiveElement: null,
      liveElement: null,
      networkCanvasElement: null,
      observeTarget: null,
    });
    mockedRunBrowserEntryCurriculum.mockClear();
    disposeResizeHandling.mockClear();
    getLastTelemetry.mockClear();
    addTelemetryListener.mockClear();
    dispatchTelemetry.mockClear();
    jest.mocked(createBrowserEntryHostServices).mockReturnValue({
      dashboard: {} as never,
      runtimeDashboard: { getLastTelemetry },
      telemetryHub: { add: addTelemetryListener, dispatch: dispatchTelemetry },
      disposeResizeHandling,
    } as never);
    jest.mocked(createBrowserEntryEvolutionHostAdapter).mockClear();
    jest.mocked(composeBrowserEntryAbortSignal).mockClear();
    jest.mocked(installBrowserEntryGlobals).mockClear();
  });

  it('reuses the active run when start is called again for the same host', async () => {
    const firstHandle = await start('ascii-maze-output');
    const secondHandle = await start('ascii-maze-output');

    expect({
      adapterInput: jest.mocked(createBrowserEntryEvolutionHostAdapter).mock
        .calls[0]?.[0],
      sameHandle: firstHandle === secondHandle,
      curriculumRunCount: mockedRunBrowserEntryCurriculum.mock.calls.length,
    }).toEqual({
      adapterInput: { liveElement: null },
      sameHandle: true,
      curriculumRunCount: 1,
    });

    firstHandle.stop();
  });

  it('allows a fresh run after the active handle stops', async () => {
    const firstHandle = await start('ascii-maze-output');
    firstHandle.stop();
    await firstHandle.done;
    const secondHandle = await start('ascii-maze-output');

    expect({
      differentHandle: firstHandle !== secondHandle,
      curriculumRunCount: mockedRunBrowserEntryCurriculum.mock.calls.length,
    }).toEqual({
      differentHandle: true,
      curriculumRunCount: 2,
    });

    secondHandle.stop();
    await secondHandle.done;
  });

  it('exposes telemetry helpers on the returned run handle', async () => {
    const listener = jest.fn();
    const handle = await start('ascii-maze-output');

    const unsubscribe = handle.onTelemetry(listener);

    expect({
      snapshot: handle.getTelemetry(),
      telemetryAddCallCount: addTelemetryListener.mock.calls.length,
      unsubscribeType: typeof unsubscribe,
    }).toEqual({
      snapshot: {
        bestFitness: null,
        details: null,
        elapsedMs: 0,
        generation: 12,
        gensPerSec: 0,
        progress: null,
        speciesCount: null,
      },
      telemetryAddCallCount: 1,
      unsubscribeType: 'function',
    });

    handle.stop();
    await handle.done;
  });

  it('finalizes immediately and skips curriculum start when the composed signal is already aborted', async () => {
    const abortedController = new AbortController();
    abortedController.abort();
    jest
      .mocked(composeBrowserEntryAbortSignal)
      .mockReturnValueOnce(abortedController.signal);

    const handle = await start('ascii-maze-output', {
      signal: abortedController.signal,
    });
    await handle.done;

    expect({
      disposeCallCount: disposeResizeHandling.mock.calls.length,
      isRunning: handle.isRunning(),
      runCount: mockedRunBrowserEntryCurriculum.mock.calls.length,
    }).toEqual({
      disposeCallCount: 1,
      isRunning: false,
      runCount: 0,
    });
  });

  it('keeps the run alive when abort-listener registration throws and still starts the curriculum', async () => {
    jest.mocked(composeBrowserEntryAbortSignal).mockReturnValueOnce({
      aborted: false,
      addEventListener: () => {
        throw new Error('listener registration failed');
      },
      dispatchEvent: () => true,
      onabort: null,
      reason: undefined,
      removeEventListener: () => {},
      throwIfAborted: () => {},
    } as unknown as AbortSignal);

    const handle = await start();

    expect({
      adapterInput: jest
        .mocked(createBrowserEntryEvolutionHostAdapter)
        .mock.calls.at(-1)?.[0],
      isRunning: handle.isRunning(),
      runCount: mockedRunBrowserEntryCurriculum.mock.calls.length,
    }).toEqual({
      adapterInput: { liveElement: null },
      isRunning: true,
      runCount: 1,
    });

    handle.stop();
    await handle.done;
  });

  it('updates the curriculum cancellation callback after the run handle stops', async () => {
    const handle = await start('ascii-maze-output');
    const curriculumContext =
      mockedRunBrowserEntryCurriculum.mock.calls.at(-1)?.[0];

    if (!curriculumContext) {
      throw new Error('Expected the curriculum runtime context to be captured');
    }

    const cancelledBeforeStop = curriculumContext.isCancelled();
    handle.stop();
    await handle.done;

    expect({
      cancelledAfterStop: curriculumContext.isCancelled(),
      cancelledBeforeStop,
    }).toEqual({
      cancelledAfterStop: true,
      cancelledBeforeStop: false,
    });
  });

  it('keeps the newer active handle registered when an older handle finishes cleanup later', async () => {
    const firstHandle = await start('ascii-maze-output');
    firstHandle.stop();
    const secondHandle = await start('ascii-maze-output');

    await Promise.resolve();
    const thirdHandle = await start('ascii-maze-output');

    expect({
      secondHandleReused: secondHandle === thirdHandle,
    }).toEqual({
      secondHandleReused: true,
    });

    secondHandle.stop();
    await firstHandle.done;
    await secondHandle.done;
  });
});
