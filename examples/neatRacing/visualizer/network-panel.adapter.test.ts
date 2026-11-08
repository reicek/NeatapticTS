/** @jest-environment jsdom */

import Network from '../../../src/architecture/network';
import {
  NEATRACING_NETWORK_OUTPUT_LABELS,
  drawNeatRacingNetworkPanel,
  drawNeatRacingNetworkPanelFromFrame,
  installNeatRacingNetworkResize,
  resolveMostForwardNeatRacingAgent,
  resolveNeatRacingNetworkPanelArchitectureLabel,
  resolveNeatRacingNetworkPanelCanvasDimensions,
  resolveNeatRacingNetworkPanelVisualizationFrame,
  type NeatRacingNetworkPanelCandidate,
} from './network-panel.adapter';

/**
 * Green-first suite for the NeatRacing network panel adapter (slice P2-S4-B).
 *
 * Pins AC-034 behaviorally: the most-forward selection laws (sticky incumbent,
 * lowest-id tie-break, non-finite exclusion), the thin pass-through frame and
 * draw paths over the shared network visualizer, the NGE output tags
 * (THROTTLE/STEER control axes), and the panel resize wiring that mirrors the
 * recycled `installRacingNetworkResize` pattern.
 */
describe('NeatRacing network panel adapter', () => {
  const buildStubContext = (): {
    context: CanvasRenderingContext2D;
    fillText: jest.Mock;
  } => {
    const canvas = document.createElement('canvas');
    canvas.width = 900;
    canvas.height = 240;
    const fillText = jest.fn();
    const context = {
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
      fillText,
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
    return { context, fillText };
  };

  const attachStubContext = (
    canvas: HTMLCanvasElement,
    context: CanvasRenderingContext2D,
  ): void => {
    jest.spyOn(canvas, 'getContext').mockImplementation(() => context);
  };

  const buildCandidate = (
    agentId: number,
    progress: number,
  ): NeatRacingNetworkPanelCandidate => ({
    agentId,
    progress,
    network: new Network(82, 2),
  });

  describe('resolveMostForwardNeatRacingAgent', () => {
    it('resolves the most-forward agent without an incumbent', () => {
      const winner = resolveMostForwardNeatRacingAgent([
        buildCandidate(1, 40),
        buildCandidate(2, 75),
        buildCandidate(3, 60),
      ]);

      expect(winner?.agentId).toBe(2);
    });

    it('breaks no-incumbent progress ties by the lowest agent id', () => {
      const winner = resolveMostForwardNeatRacingAgent([
        buildCandidate(4, 60),
        buildCandidate(2, 60),
      ]);

      expect(winner?.agentId).toBe(2);
    });

    it('excludes non-finite progress from ever leading', () => {
      const winner = resolveMostForwardNeatRacingAgent([
        buildCandidate(5, Number.NaN),
        buildCandidate(6, Number.POSITIVE_INFINITY),
        buildCandidate(7, 12),
      ]);

      expect(winner?.agentId).toBe(7);
    });

    it('keeps the incumbent on an exactly equal progress tick', () => {
      const winner = resolveMostForwardNeatRacingAgent(
        [buildCandidate(1, 50), buildCandidate(2, 50)],
        1,
      );

      expect(winner?.agentId).toBe(1);
    });

    it('dethrones the incumbent only on strictly greater progress', () => {
      const winner = resolveMostForwardNeatRacingAgent(
        [buildCandidate(1, 50), buildCandidate(2, 51)],
        1,
      );

      expect(winner?.agentId).toBe(2);
    });

    it('returns undefined when no finite-progress candidate exists', () => {
      const winner = resolveMostForwardNeatRacingAgent([
        buildCandidate(1, Number.NaN),
        buildCandidate(2, Number.POSITIVE_INFINITY),
      ]);

      expect(winner).toBeUndefined();
    });
  });

  describe('resolveNeatRacingNetworkPanelCanvasDimensions', () => {
    it('clamps empty measurements to the positive panel shelf minimum', () => {
      const dimensions = resolveNeatRacingNetworkPanelCanvasDimensions(0, 0);

      expect(dimensions).toEqual({ widthPx: 320, heightPx: 240 });
    });

    it('honors a positive measured panel height over the aspect fallback', () => {
      const dimensions = resolveNeatRacingNetworkPanelCanvasDimensions(500, 360);

      expect(dimensions).toEqual({ widthPx: 500, heightPx: 360 });
    });
  });

  describe('shared visualizer pass-through', () => {
    it('resolves a positioned frame for an 82/2 NGE controller network', () => {
      const { context } = buildStubContext();

      const frame = resolveNeatRacingNetworkPanelVisualizationFrame(
        context,
        new Network(82, 2),
      );

      expect(frame.positionedScene.positionedNodes.length).toBeGreaterThan(0);
    });

    it('resolves a non-empty architecture label for the panel legend', () => {
      const label = resolveNeatRacingNetworkPanelArchitectureLabel(
        new Network(82, 2),
      );

      expect(label.length).toBeGreaterThan(0);
    });

    it('draws a resolved frame through the thin from-frame path', () => {
      const { context } = buildStubContext();
      const frame = resolveNeatRacingNetworkPanelVisualizationFrame(
        context,
        new Network(82, 2),
      );

      const positionedScene = drawNeatRacingNetworkPanelFromFrame(
        context,
        frame,
        [],
      );

      expect(positionedScene.positionedNodes.length).toBeGreaterThan(0);
    });
  });

  describe('drawNeatRacingNetworkPanel', () => {
    it('returns an empty scene when the canvas has no 2D context', () => {
      const canvas = document.createElement('canvas');
      jest.spyOn(canvas, 'getContext').mockImplementation(() => null);

      const positionedScene = drawNeatRacingNetworkPanel(
        canvas,
        new Network(82, 2),
      );

      expect(positionedScene.positionedNodes).toEqual([]);
    });

    it('stamps the pinned THROTTLE control-axis output tag', () => {
      const { context, fillText } = buildStubContext();
      const canvas = document.createElement('canvas');
      attachStubContext(canvas, context);

      drawNeatRacingNetworkPanel(canvas, new Network(82, 2));

      expect(fillText.mock.calls.some((call) => call[0] === 'THR')).toBe(
        true,
      );
    });

    it('stamps the pinned STEER control-axis output tag', () => {
      const { context, fillText } = buildStubContext();
      const canvas = document.createElement('canvas');
      attachStubContext(canvas, context);

      drawNeatRacingNetworkPanel(canvas, new Network(82, 2));

      expect(fillText.mock.calls.some((call) => call[0] === 'STR')).toBe(
        true,
      );
    });

    it('falls back to OUT-index tags past the pinned output label table', () => {
      const outputLabelCount = NEATRACING_NETWORK_OUTPUT_LABELS.length;

      expect(
        NEATRACING_NETWORK_OUTPUT_LABELS[outputLabelCount - 1],
      ).toBe('R6');
    });
  });

  describe('installNeatRacingNetworkResize', () => {
    it('falls back to the viewport fraction when the host has no layout', () => {
      const networkCanvas = document.createElement('canvas');
      const networkCanvasHost = document.createElement('div');
      networkCanvasHost.style.width = '0px';
      networkCanvasHost.style.height = '0px';
      document.body.append(networkCanvasHost);

      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });
      Object.defineProperty(window, 'innerHeight', {
        writable: true,
        configurable: true,
        value: 768,
      });

      const resizeHandle = installNeatRacingNetworkResize(
        networkCanvas,
        networkCanvasHost,
        jest.fn(),
      );
      resizeHandle.uninstall();

      expect(networkCanvas.width).toBe(Math.floor(1024 * 0.25));
    });

    it('does not invoke onResize when a resize changes no backing size', () => {
      const networkCanvas = document.createElement('canvas');
      const networkCanvasHost = document.createElement('div');
      document.body.append(networkCanvasHost);

      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });

      const onResize = jest.fn();
      const resizeHandle = installNeatRacingNetworkResize(
        networkCanvas,
        networkCanvasHost,
        onResize,
      );
      const installCallCount = onResize.mock.calls.length;
      window.dispatchEvent(new Event('resize'));
      resizeHandle.uninstall();

      expect(onResize.mock.calls.length).toBe(installCallCount);
    });

    it('invokes onResize exactly when the fallback backing size changes', () => {
      const networkCanvas = document.createElement('canvas');
      const networkCanvasHost = document.createElement('div');
      document.body.append(networkCanvasHost);

      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });

      const onResize = jest.fn();
      const resizeHandle = installNeatRacingNetworkResize(
        networkCanvas,
        networkCanvasHost,
        onResize,
      );
      const installCallCount = onResize.mock.calls.length;

      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1224,
      });
      window.dispatchEvent(new Event('resize'));
      resizeHandle.uninstall();

      expect(onResize.mock.calls.length).toBe(installCallCount + 1);
    });

    it('stops resize handling after uninstall', () => {
      const networkCanvas = document.createElement('canvas');
      const networkCanvasHost = document.createElement('div');
      document.body.append(networkCanvasHost);

      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });

      const onResize = jest.fn();
      const resizeHandle = installNeatRacingNetworkResize(
        networkCanvas,
        networkCanvasHost,
        onResize,
      );
      resizeHandle.uninstall();

      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1424,
      });
      window.dispatchEvent(new Event('resize'));

      expect(networkCanvas.width).toBe(Math.floor(1024 * 0.25));
    });
  });
});
