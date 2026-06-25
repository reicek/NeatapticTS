/** @jest-environment jsdom */

import { createRacingHost, RACING_NARROW_VIEWPORT_THRESHOLD_PX } from './host';
import { Network } from '../../../../src/browser-entry.ts';
import * as networkViewModule from '../network-view/network-view';

describe('racing host boundary', () => {
  describe('createRacingHost', () => {
    beforeEach(() => {
      document.body.innerHTML = '';
    });

    it('switches the shell to the narrow layout below the viewport threshold', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);
      racingHost.applyViewportLayout(RACING_NARROW_VIEWPORT_THRESHOLD_PX - 1);

      expect(racingHost.rootElement.dataset.racingLayout).toBe('narrow');
    });

    it('does not create a visualizer-bottom region', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(
        racingHost.rootElement.querySelector(
          '.racing-host__region--visualizer',
        ),
      ).toBeNull();
    });

    it('returns a dedicated network canvas host in the right sidebar', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(
        racingHost.networkCanvasHost instanceof HTMLDivElement &&
          racingHost.networkRegionElement.contains(
            racingHost.networkCanvasHost,
          ),
      ).toBe(true);
    });

    it('returns a network canvas for the right sidebar', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(racingHost.networkCanvas instanceof HTMLCanvasElement).toBe(true);
    });

    it('returns a 2D context for the network canvas', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(racingHost.networkContext).toBeTruthy();
    });

    it('returns a resize/redraw controller', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(racingHost.resizeRedrawController).toBeDefined();
    });

    it('exposes a renderNetworkArchitecture hook for the right sidebar', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(typeof racingHost.renderNetworkArchitecture).toBe('function');
    });

    it('returns live HUD text nodes for the right sidebar', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(racingHost.networkHud).toEqual(
        expect.objectContaining({
          titleValue: expect.any(Text),
          sizeValue: expect.any(Text),
          lastChangeValue: expect.any(Text),
          statusValue: expect.any(Text),
        }),
      );
    });

    it('mounts the neon HUD strip above the network canvas', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(
        racingHost.networkCanvasHost.querySelector('.racing-network-hud'),
      ).not.toBeNull();
    });

    it('returns the static help chip strip for the right sidebar', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(racingHost.networkHelpStrip.className).toBe(
        'racing-network-help-strip',
      );
    });

    it('mounts help chips below the network canvas', () => {
      const hostElement = document.createElement('div');
      document.body.append(hostElement);

      const racingHost = createRacingHost(hostElement);

      expect(
        racingHost.networkCanvasHost.querySelector('.racing-network-help-chip'),
      ).not.toBeNull();
    });
  });

  describe('racing host network panel overlay width', () => {
    const buildRecordingContext = (): CanvasRenderingContext2D & {
      fillTextCalls: string[];
    } => {
      const canvas = document.createElement('canvas');
      const fillTextCalls: string[] = [];
      return {
        canvas,
        fillTextCalls,
        clearRect: jest.fn(),
        beginPath: jest.fn(),
        moveTo: jest.fn(),
        lineTo: jest.fn(),
        bezierCurveTo: jest.fn(),
        quadraticCurveTo: jest.fn(),
        ellipse: jest.fn(),
        stroke: jest.fn(),
        rect: jest.fn(),
        fill: jest.fn(),
        fillRect: jest.fn(),
        arc: jest.fn(),
        fillText: jest.fn((text: string) => {
          fillTextCalls.push(text);
        }),
        strokeText: jest.fn(),
        measureText: jest.fn().mockReturnValue({ width: 0 }),
        drawImage: jest.fn(),
        save: jest.fn(),
        restore: jest.fn(),
        translate: jest.fn(),
        scale: jest.fn(),
        rotate: jest.fn(),
        closePath: jest.fn(),
        clip: jest.fn(),
        setLineDash: jest.fn(),
        getLineDash: jest.fn().mockReturnValue([]),
        setTransform: jest.fn(),
        strokeRect: jest.fn(),
        strokeStyle: '',
        lineWidth: 1,
        fillStyle: '',
        font: '',
        textAlign: 'center',
        textBaseline: 'middle',
      } as unknown as CanvasRenderingContext2D & { fillTextCalls: string[] };
    };

    it('produces input-description scenes when the panel is wide despite a narrow viewport', () => {
      const originalInnerWidth = window.innerWidth;
      Object.defineProperty(window, 'innerWidth', {
        value: 600,
        configurable: true,
      });

      const container = document.createElement('div');
      document.body.append(container);

      const recordingContext = buildRecordingContext();
      const getContextSpy = jest
        .spyOn(HTMLCanvasElement.prototype, 'getContext')
        .mockImplementation(function mockGetContext(
          this: HTMLCanvasElement,
          contextId,
        ) {
          if (contextId === '2d') {
            (
              recordingContext as unknown as { canvas: HTMLCanvasElement }
            ).canvas = this;
            return recordingContext as unknown as CanvasRenderingContext2D;
          }
          return null;
        });

      const host = createRacingHost(container);

      Object.defineProperty(host.networkCanvasHost, 'clientWidth', {
        value: 900,
        configurable: true,
      });
      Object.defineProperty(host.networkCanvasHost, 'clientHeight', {
        value: 600,
        configurable: true,
      });

      const networkCanvasWrapper = host.networkCanvas
        .parentElement as HTMLDivElement;
      Object.defineProperty(networkCanvasWrapper, 'clientWidth', {
        value: 900,
        configurable: true,
      });
      Object.defineProperty(networkCanvasWrapper, 'clientHeight', {
        value: 600,
        configurable: true,
      });

      const drawFromFrameSpy = jest.spyOn(
        networkViewModule,
        'drawRacingNetworkVisualizationFromFrame',
      );

      const network = new Network(70, 2);
      host.renderNetworkArchitecture(network);

      const lastResult =
        drawFromFrameSpy.mock.results[drawFromFrameSpy.mock.results.length - 1];
      const positionedScene = lastResult?.value;

      Object.defineProperty(window, 'innerWidth', {
        value: originalInnerWidth,
        configurable: true,
      });
      getContextSpy.mockRestore();
      drawFromFrameSpy.mockRestore();
      container.remove();

      expect(
        positionedScene?.inputDescriptionScenes?.length ?? 0,
      ).toBeGreaterThan(0);
    });
  });

  describe('racing network visualizer live refresh', () => {
    const buildStubContext = (): CanvasRenderingContext2D => {
      const canvas = document.createElement('canvas');
      return {
        canvas,
        clearRect: jest.fn(),
        beginPath: jest.fn(),
        moveTo: jest.fn(),
        lineTo: jest.fn(),
        bezierCurveTo: jest.fn(),
        quadraticCurveTo: jest.fn(),
        ellipse: jest.fn(),
        stroke: jest.fn(),
        rect: jest.fn(),
        fill: jest.fn(),
        fillRect: jest.fn(),
        arc: jest.fn(),
        fillText: jest.fn(),
        strokeText: jest.fn(),
        measureText: jest.fn().mockReturnValue({ width: 0 }),
        drawImage: jest.fn(),
        save: jest.fn(),
        restore: jest.fn(),
        translate: jest.fn(),
        scale: jest.fn(),
        rotate: jest.fn(),
        closePath: jest.fn(),
        clip: jest.fn(),
        setLineDash: jest.fn(),
        getLineDash: jest.fn().mockReturnValue([]),
        setTransform: jest.fn(),
        strokeRect: jest.fn(),
        strokeStyle: '',
        lineWidth: 1,
        fillStyle: '',
        font: '',
        textAlign: 'center',
        textBaseline: 'middle',
      } as unknown as CanvasRenderingContext2D;
    };

    const setNetworkCanvasWrapperSize = (
      host: ReturnType<typeof createRacingHost>,
      widthPx: number,
      heightPx: number,
    ): void => {
      const wrapper = host.networkCanvas.parentElement as HTMLDivElement;
      Object.defineProperty(wrapper, 'clientWidth', {
        value: widthPx,
        configurable: true,
      });
      Object.defineProperty(wrapper, 'clientHeight', {
        value: heightPx,
        configurable: true,
      });
      Object.defineProperty(host.networkCanvasHost, 'clientWidth', {
        value: widthPx,
        configurable: true,
      });
      Object.defineProperty(host.networkCanvasHost, 'clientHeight', {
        value: heightPx,
        configurable: true,
      });
    };

    let getContextSpy: jest.SpyInstance | undefined;

    beforeEach(() => {
      document.body.innerHTML = '';
    });

    afterEach(() => {
      getContextSpy?.mockRestore();
    });

    it('re-resolves the frame after in-place network state mutation', () => {
      const container = document.createElement('div');
      document.body.append(container);

      const stubContext = buildStubContext();
      getContextSpy = jest
        .spyOn(HTMLCanvasElement.prototype, 'getContext')
        .mockImplementation(function mockGetContext(
          this: HTMLCanvasElement,
          contextId,
        ) {
          if (contextId === '2d') {
            (
              stubContext as unknown as { canvas: HTMLCanvasElement }
            ).canvas = this;
            return stubContext as unknown as CanvasRenderingContext2D;
          }
          return null;
        });

      const host = createRacingHost(container);
      setNetworkCanvasWrapperSize(host, 640, 480);

      const network = new Network(2, 1, { seed: 42 });
      const resolveSpy = jest.spyOn(
        networkViewModule,
        'resolveRacingNetworkVisualizationFrame',
      );
      host.renderNetworkArchitecture(network);
      resolveSpy.mockClear();

      network.connections[0].weight += 1;
      host.resizeRedrawController.redraw();

      const callCount = resolveSpy.mock.calls.length;
      resolveSpy.mockRestore();
      container.remove();

      expect(callCount).toBe(1);
    });

    it('resolves the frame only once for two synchronous render calls', () => {
      const container = document.createElement('div');
      document.body.append(container);

      const stubContext = buildStubContext();
      getContextSpy = jest
        .spyOn(HTMLCanvasElement.prototype, 'getContext')
        .mockImplementation(function mockGetContext(
          this: HTMLCanvasElement,
          contextId,
        ) {
          if (contextId === '2d') {
            (
              stubContext as unknown as { canvas: HTMLCanvasElement }
            ).canvas = this;
            return stubContext as unknown as CanvasRenderingContext2D;
          }
          return null;
        });

      const host = createRacingHost(container);
      setNetworkCanvasWrapperSize(host, 640, 480);

      const network = new Network(2, 1, { seed: 42 });
      const resolveSpy = jest.spyOn(
        networkViewModule,
        'resolveRacingNetworkVisualizationFrame',
      );
      host.renderNetworkArchitecture(network);
      host.renderNetworkArchitecture(network);

      const callCount = resolveSpy.mock.calls.length;
      resolveSpy.mockRestore();
      container.remove();

      expect(callCount).toBe(1);
    });

    it('does not draw when the network panel is hidden', () => {
      const container = document.createElement('div');
      document.body.append(container);

      const stubContext = buildStubContext();
      getContextSpy = jest
        .spyOn(HTMLCanvasElement.prototype, 'getContext')
        .mockImplementation(function mockGetContext(
          this: HTMLCanvasElement,
          contextId,
        ) {
          if (contextId === '2d') {
            (
              stubContext as unknown as { canvas: HTMLCanvasElement }
            ).canvas = this;
            return stubContext as unknown as CanvasRenderingContext2D;
          }
          return null;
        });

      const host = createRacingHost(container);
      setNetworkCanvasWrapperSize(host, 640, 480);

      const network = new Network(2, 1, { seed: 42 });
      const drawSpy = jest.spyOn(
        networkViewModule,
        'drawRacingNetworkVisualizationFromFrame',
      );
      host.renderNetworkArchitecture(network);
      drawSpy.mockClear();

      host.networkRegionElement.style.display = 'none';
      host.renderNetworkArchitecture(network);

      const wasCalled = drawSpy.mock.calls.length > 0;
      drawSpy.mockRestore();
      container.remove();

      expect(wasCalled).toBe(false);
    });
  });
});
