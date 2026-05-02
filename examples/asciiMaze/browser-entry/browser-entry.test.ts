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

describe('asciiMaze browser entry start()', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="ascii-maze-output"></div>';
    mockedResolveBrowserEntryHostElements.mockReturnValue({
      hostElement: document.getElementById('ascii-maze-output'),
      archiveElement: null,
      liveElement: null,
      observeTarget: null,
    });
    mockedRunBrowserEntryCurriculum.mockClear();
    jest.mocked(createBrowserEntryHostServices).mockClear();
    jest.mocked(createBrowserEntryEvolutionHostAdapter).mockClear();
    jest.mocked(composeBrowserEntryAbortSignal).mockClear();
    jest.mocked(installBrowserEntryGlobals).mockClear();
  });

  it('reuses the active run when start is called again for the same host', async () => {
    const firstHandle = await start('ascii-maze-output');
    const secondHandle = await start('ascii-maze-output');

    expect({
      sameHandle: firstHandle === secondHandle,
      curriculumRunCount: mockedRunBrowserEntryCurriculum.mock.calls.length,
    }).toEqual({
      sameHandle: true,
      curriculumRunCount: 1,
    });

    firstHandle.stop();
  });

  it('allows a fresh run after the active handle stops', async () => {
    const firstHandle = await start('ascii-maze-output');
    firstHandle.stop();
    const secondHandle = await start('ascii-maze-output');

    expect({
      differentHandle: firstHandle !== secondHandle,
      curriculumRunCount: mockedRunBrowserEntryCurriculum.mock.calls.length,
    }).toEqual({
      differentHandle: true,
      curriculumRunCount: 2,
    });

    secondHandle.stop();
  });
});
