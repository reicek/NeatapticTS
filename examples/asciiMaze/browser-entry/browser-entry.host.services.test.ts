/** @jest-environment jsdom */

import { createBrowserEntryHostServices } from './browser-entry.host.services';
import { drawMazeNetworkVisualization } from './network-view/network-view';

jest.mock('./network-view/network-view', () => ({
  drawMazeNetworkVisualization: jest.fn(() => ({
    frame: {} as never,
    hitAreas: [],
  })),
}));

jest.mock('../../../src/architecture/network', () => ({
  __esModule: true,
  default: class MockVisualizationNetwork {},
  exportVisualizationGraph: jest.fn(() => ({ nodes: [], connections: [] })),
}));

const mockedDrawMazeNetworkVisualization = jest.mocked(
  drawMazeNetworkVisualization,
);

describe('createBrowserEntryHostServices', () => {
  beforeEach(() => {
    mockedDrawMazeNetworkVisualization.mockClear();
  });

  it('renders only the maze into the browser live pane instead of the framed terminal dashboard', () => {
    document.body.innerHTML = `
      <div id="live-output"></div>
      <div id="archive-output"></div>
      <div id="maze-network-tooltip"></div>
    `;

    const liveElement = document.getElementById('live-output');
    const archiveElement = document.getElementById('archive-output');
    if (!liveElement || !archiveElement) {
      throw new Error('Expected the browser host elements to exist');
    }

    const hostServices = createBrowserEntryHostServices({
      hostElement: null,
      archiveElement,
      liveElement,
      networkCanvasElement: null,
      archButtonsElement: null,
      observeTarget: null,
    });

    hostServices.dashboard.update(
      ['S.E'],
      {
        fitness: 1,
        path: [[0, 0]],
        progress: 0,
        steps: 1,
        success: false,
      } as never,
      null,
      1,
    );

    expect({
      containsHeaderFrame:
        liveElement.textContent?.includes('ASCII maze') ?? false,
      containsEvolvingHeader:
        liveElement.textContent?.includes('EVOLVING') ?? false,
      containsBoxBorder: liveElement.textContent?.includes('╔') ?? false,
      containsExit: liveElement.textContent?.includes('E') ?? false,
    }).toEqual({
      containsHeaderFrame: false,
      containsEvolvingHeader: false,
      containsBoxBorder: false,
      containsExit: true,
    });

    hostServices.disposeResizeHandling();
  });

  it('clears stale live and archive output when a new browser host session starts', () => {
    document.body.innerHTML = `
      <div id="live-output">stale-live</div>
      <div id="archive-output">stale-archive</div>
      <div id="maze-network-tooltip" style="display:block"></div>
    `;

    const liveElement = document.getElementById('live-output');
    const archiveElement = document.getElementById('archive-output');
    const tooltipElement = document.getElementById('maze-network-tooltip');
    if (!liveElement || !archiveElement || !tooltipElement) {
      throw new Error('Expected the browser host elements to exist');
    }

    const hostServices = createBrowserEntryHostServices({
      hostElement: null,
      archiveElement,
      liveElement,
      networkCanvasElement: null,
      archButtonsElement: null,
      observeTarget: null,
    });

    expect({
      archiveText: archiveElement.textContent,
      liveText: liveElement.textContent,
      tooltipDisplay: tooltipElement.style.display,
    }).toEqual({
      archiveText: '',
      liveText: '',
      tooltipDisplay: 'none',
    });

    hostServices.disposeResizeHandling();
  });

  it('redraws the network visualization when the observed pane height changes without a width change', () => {
    document.body.innerHTML = `
      <div id="observe-target"></div>
      <canvas id="network-canvas"></canvas>
      <div id="maze-network-tooltip"></div>
    `;

    const observeTarget = document.getElementById(
      'observe-target',
    ) as HTMLElement | null;
    const networkCanvasElement = document.getElementById(
      'network-canvas',
    ) as HTMLCanvasElement | null;
    if (!observeTarget || !networkCanvasElement) {
      throw new Error('Expected the observed pane and network canvas to exist');
    }

    Object.defineProperty(observeTarget, 'clientWidth', {
      configurable: true,
      value: 640,
    });
    Object.defineProperty(observeTarget, 'clientHeight', {
      configurable: true,
      value: 420,
    });

    let resizeObserverCallback: ResizeObserverCallback | undefined;
    const originalResizeObserver = global.ResizeObserver;

    global.ResizeObserver = class MockResizeObserver {
      constructor(callback: ResizeObserverCallback) {
        resizeObserverCallback = callback;
      }

      disconnect(): void {}

      observe(): void {}

      unobserve(): void {}
    } as unknown as typeof ResizeObserver;

    const hostServices = createBrowserEntryHostServices({
      hostElement: null,
      archiveElement: null,
      liveElement: null,
      networkCanvasElement,
      archButtonsElement: null,
      observeTarget,
    });

    hostServices.dashboard.update(
      ['SE'],
      {
        fitness: 1,
        path: [[0, 0]],
        progress: 0,
        steps: 1,
        success: false,
      } as never,
      {
        activate: () => [0, 0, 0, 0],
        connections: [],
        getTopologyIntent: () => 'feed-forward',
        inputNodeIds: [],
        nodes: [],
        outputNodeIds: [],
      } as never,
      1,
    );

    resizeObserverCallback?.(
      [
        {
          contentRect: {
            bottom: 620,
            height: 620,
            left: 0,
            right: 640,
            toJSON: () => ({}),
            top: 0,
            width: 640,
            x: 0,
            y: 0,
          },
          borderBoxSize: [],
          contentBoxSize: [],
          devicePixelContentBoxSize: [],
          target: observeTarget,
        } as unknown as ResizeObserverEntry,
      ],
      {} as ResizeObserver,
    );

    expect({
      drawCallCount: mockedDrawMazeNetworkVisualization.mock.calls.length,
    }).toEqual({
      drawCallCount: 2,
    });

    hostServices.disposeResizeHandling();
    global.ResizeObserver = originalResizeObserver;
  });
});
