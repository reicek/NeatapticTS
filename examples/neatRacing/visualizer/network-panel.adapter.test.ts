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
  resolveNeatRacingNetworkPanelLiveCarrierFrame,
  resolveNeatRacingNetworkPanelVisualizationFrame,
  type NeatRacingNetworkPanelCandidate,
  type NetworkVisualizationResolvedFrame,
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
      const dimensions = resolveNeatRacingNetworkPanelCanvasDimensions(
        500,
        360,
      );

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

      expect(fillText.mock.calls.some((call) => call[0] === 'THR')).toBe(true);
    });

    it('stamps the pinned STEER control-axis output tag', () => {
      const { context, fillText } = buildStubContext();
      const canvas = document.createElement('canvas');
      attachStubContext(canvas, context);

      drawNeatRacingNetworkPanel(canvas, new Network(82, 2));

      expect(fillText.mock.calls.some((call) => call[0] === 'STR')).toBe(true);
    });

    it('falls back to OUT-index tags past the pinned output label table', () => {
      const outputLabelCount = NEATRACING_NETWORK_OUTPUT_LABELS.length;

      expect(NEATRACING_NETWORK_OUTPUT_LABELS[outputLabelCount - 1]).toBe('R6');
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

/**
 * RED contracts for the docked-right panel SHELF (slice P4-S2C-red — Phase
 * 4, Step 24C of plans/neatracing-voxel-racer.plans.md, RED-P4-20 +
 * RED-P4-21).
 *
 * Educational note — the 1a defect these contracts pin down. The panel shelf
 * is sized from RAW window/host fractions today: the resize service sizes a
 * measurable host from its raw client bounds, and a layout-less host from
 * `innerWidth * 0.25` / `innerHeight * 0.35`. The recorded defect measured
 * the F3 backing store at the FULL viewport 1689x1325 — the panel fills the
 * viewport and every live layer beneath it, instead of a docked-right
 * bounded shelf. These RED contracts never touch the frozen suite above
 * (RED-P4-SHAPE: no frozen assertion is edited; the fallback its stay green
 * on the zero-measure host path). The NEW laws live here:
 *
 * - RED-P4-20 — the SHELF SLOT LAW: a pure, DOM-free slot resolver maps the
 *   viewport dimensions onto the deterministic docked-right bounded slot
 *   (right-anchored x, inside the viewport, strictly bounded — never a
 *   full-viewport box, never wider than half the viewport). The resolver
 *   does not exist today (this is the missing 1a export).
 * - RED-P4-21 — the IDLE-PATH SLOT LAW: the resize service must size the
 *   panel through the SAME resolver (the bounded-host branch rides the
 *   slot; the zero-measure fallback pinned above keeps jsdom windowless
 *   parity, it is NOT the resolver's surface). Today the bounded host is
 *   sized from its raw full-viewport bounds — the recorded defect.
 *
 * Failure discipline (the promo-live red law): every failure must NAME the
 * missing behaviour. The gate below asserts the missing exported resolver
 * by reading it off the runtime module namespace — never a static import of
 * a not-yet-existing symbol — so the red reason is the NAMED missing export
 * plus the slot law, never a compile error.
 */
describe('NeatRacing network panel docked-right shelf (RED-P4-20/21)', () => {
  /** The shelf slot shape the resolver must return. */
  type RedShelfSlotResolver = (
    viewportWidthPx: number,
    viewportHeightPx: number,
  ) => { leftPx: number; topPx: number; widthPx: number; heightPx: number };

  /** The recorded 1a defect viewport (the measured backing 1689x1325). */
  const RED_SHELF_RECORDED_DEFECT_VIEWPORT = {
    label: 'recorded-defect-viewport',
    widthPx: 1689,
    heightPx: 1325,
  } as const;

  /** Two stable companion viewports for the slot-law walk. */
  const RED_SHELF_FIXTURE_VIEWPORTS: ReadonlyArray<{
    label: string;
    widthPx: number;
    heightPx: number;
  }> = [
    RED_SHELF_RECORDED_DEFECT_VIEWPORT,
    { label: 'demo-viewport', widthPx: 1280, heightPx: 720 },
    { label: 'small-window-viewport', widthPx: 640, heightPx: 480 },
  ];

  /** Captured `innerWidth` descriptor before this describe stubs it. */
  let redShelfWindowWidthDescriptor: PropertyDescriptor | undefined;
  /** Captured `innerHeight` descriptor before this describe stubs it. */
  let redShelfWindowHeightDescriptor: PropertyDescriptor | undefined;

  beforeEach(() => {
    redShelfWindowWidthDescriptor = Object.getOwnPropertyDescriptor(
      window,
      'innerWidth',
    );
    redShelfWindowHeightDescriptor = Object.getOwnPropertyDescriptor(
      window,
      'innerHeight',
    );
  });

  afterEach(() => {
    if (redShelfWindowWidthDescriptor !== undefined) {
      Object.defineProperty(
        window,
        'innerWidth',
        redShelfWindowWidthDescriptor,
      );
    }
    if (redShelfWindowHeightDescriptor !== undefined) {
      Object.defineProperty(
        window,
        'innerHeight',
        redShelfWindowHeightDescriptor,
      );
    }
    document.body.innerHTML = '';
  });

  /**
   * Stubs the jsdom window dimensions (the frozen suite's own pattern).
   *
   * @param widthPx - The window inner width to publish.
   * @param heightPx - The window inner height to publish.
   */
  const stubRedShelfWindowDimensions = (
    widthPx: number,
    heightPx: number,
  ): void => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: widthPx,
    });
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: heightPx,
    });
  };

  /**
   * Stubs the host element's client measures (jsdom performs no layout).
   *
   * @param host - The host element whose measures to publish.
   * @param widthPx - The client width to publish.
   * @param heightPx - The client height to publish.
   */
  const stubRedShelfHostMeasure = (
    host: HTMLElement,
    widthPx: number,
    heightPx: number,
  ): void => {
    jest.spyOn(host, 'clientWidth', 'get').mockReturnValue(widthPx);
    jest.spyOn(host, 'clientHeight', 'get').mockReturnValue(heightPx);
  };

  /**
   * Loads the (not-yet-existing) exported slot resolver behind the red gate.
   *
   * @returns The resolver once the implementing slice exports it.
   */
  const gateRedShelfSlotResolver = async (): Promise<RedShelfSlotResolver> => {
    const adapterModule =
      (await import('./network-panel.adapter')) as unknown as {
        resolveNeatRacingNetworkPanelDockedRightShelfSlot?: unknown;
      };
    const resolver =
      adapterModule.resolveNeatRacingNetworkPanelDockedRightShelfSlot;
    if (typeof resolver !== 'function') {
      throw new Error(
        'RED-P4-20/21 red gate: the exported pure docked-right shelf-slot resolver resolveNeatRacingNetworkPanelDockedRightShelfSlot does not exist yet on examples/neatRacing/visualizer/network-panel.adapter.ts (the 1a bounded-shelf sizing law cannot be observed until it is added)',
      );
    }
    return resolver as RedShelfSlotResolver;
  };

  describe('RED-P4-20 — the shelf slot law', () => {
    it('resolves the deterministic docked-right bounded slot and never a full-viewport box', async () => {
      // Arrange: the resolver gate (the 1a red: the export does not exist
      // yet).
      const resolver = await gateRedShelfSlotResolver();

      // Act + Assert: every fixture viewport maps onto the docked-right law
      // (the received object names the failing facet AND the fixture).
      for (const viewport of RED_SHELF_FIXTURE_VIEWPORTS) {
        const slot = resolver(viewport.widthPx, viewport.heightPx);
        expect({
          fixture: viewport.label,
          rightAnchored: slot.leftPx + slot.widthPx === viewport.widthPx,
          insideViewport:
            slot.leftPx >= 0 &&
            slot.topPx >= 0 &&
            slot.leftPx + slot.widthPx <= viewport.widthPx &&
            slot.topPx + slot.heightPx <= viewport.heightPx,
          boundedWidth: slot.widthPx >= 1 && slot.widthPx < viewport.widthPx,
          boundedHeight:
            slot.heightPx >= 1 && slot.heightPx < viewport.heightPx,
          halfViewportWidthCap: slot.widthPx <= Math.ceil(viewport.widthPx / 2),
        }).toEqual({
          fixture: viewport.label,
          rightAnchored: true,
          insideViewport: true,
          boundedWidth: true,
          boundedHeight: true,
          halfViewportWidthCap: true,
        });
      }
    });

    it('is deterministic and DOM-free: the same arguments always resolve the same slot', async () => {
      // Arrange: the resolver gate, then two calls with the SAME arguments —
      // with the jsdom window retuned BETWEEN them.
      const resolver = await gateRedShelfSlotResolver();
      const firstSlot = resolver(1689, 1325);
      stubRedShelfWindowDimensions(4096, 3000);

      // Act: the second same-argument resolution.
      const secondSlot = resolver(1689, 1325);

      // Assert: the slot follows its arguments only — a window-reading
      // resolver would drift.
      expect(secondSlot).toEqual(firstSlot);
    });
  });

  describe('RED-P4-21 — the idle-path shelf law', () => {
    it('sizes a measurable host from the docked-right slot through the same resolver at install', async () => {
      // Arrange: the resolver gate + the recorded defect viewport, with the
      // panel host stubbed to the full-viewport hud-output measure (the 1a
      // defect's host element).
      const resolver = await gateRedShelfSlotResolver();
      stubRedShelfWindowDimensions(1689, 1325);
      const networkCanvas = document.createElement('canvas');
      const networkCanvasHost = document.createElement('div');
      document.body.append(networkCanvasHost);
      stubRedShelfHostMeasure(networkCanvasHost, 1689, 1325);

      // Act: the idle-path installer applies its sizing once on install.
      const resizeHandle = installNeatRacingNetworkResize(
        networkCanvas,
        networkCanvasHost,
        jest.fn(),
      );
      resizeHandle.uninstall();
      const slot = resolver(1689, 1325);

      // Assert: the backing store carries the resolver's docked-right slot
      // (NEVER the full-viewport box the host itself measures).
      expect({
        backingWidth: networkCanvas.width,
        backingHeight: networkCanvas.height,
      }).toEqual({
        backingWidth: slot.widthPx,
        backingHeight: slot.heightPx,
      });
      expect(networkCanvas.width).toBeLessThan(1689);
    });

    it('re-applies the docked-right slot through the same resolver on a window resize', async () => {
      // Arrange: the resolver gate + the recorded defect viewport, then the
      // installer; afterwards the window AND the host measure change.
      const resolver = await gateRedShelfSlotResolver();
      stubRedShelfWindowDimensions(1689, 1325);
      const networkCanvas = document.createElement('canvas');
      const networkCanvasHost = document.createElement('div');
      document.body.append(networkCanvasHost);
      stubRedShelfHostMeasure(networkCanvasHost, 1689, 1325);
      const resizeHandle = installNeatRacingNetworkResize(
        networkCanvas,
        networkCanvasHost,
        jest.fn(),
      );

      // Act: retune the window AND the host measure, then fire the resize
      // event the idle path listens on.
      stubRedShelfWindowDimensions(2029, 1325);
      stubRedShelfHostMeasure(networkCanvasHost, 2029, 1325);
      window.dispatchEvent(new Event('resize'));
      const slotAfterResize = resolver(2029, 1325);

      // Assert: the backing store carries the NEWLY resolved slot.
      expect({
        backingWidth: networkCanvas.width,
        backingHeight: networkCanvas.height,
      }).toEqual({
        backingWidth: slotAfterResize.widthPx,
        backingHeight: slotAfterResize.heightPx,
      });
      resizeHandle.uninstall();
    });
  });
});

/**
 * GREEN companions for the docked-right panel SHELF (slice P4-S2C-shelf).
 *
 * The RED homes above (RED-P4-20/21) stay BYTE-FROZEN; this block APPENDS
 * ONLY. The red homes pin the slot law and the idle-path slot law
 * behaviorally; the companions pin the two implementation-level laws those
 * homes leave implicit:
 *
 * - the EXACT deterministic recorded-viewport slot — the 1a design
 *   constants (the shelf viewport fraction, the panel aspect lock, and the
 *   top clearance under the top HUD band) so the deterministic design is
 *   pinned, not just the law shape;
 * - the measurable-host branch sizes from the VIEWPORT slot, never the raw
 *   host bounds — the companion detunes the host measure from the window so
 *   a raw-bounds sizing (the recorded 1a defect lane) cannot sneak back.
 *
 * Failure discipline stays the red-gate pattern: the exported resolver is
 * read off the runtime module namespace, never a static import, so a
 * missing export names itself instead of failing as a compile error.
 */
describe('NeatRacing network panel shelf companions (P4-S2C-shelf)', () => {
  /** The shelf slot shape the resolver returns (declared locally). */
  type CompanionShelfSlotResolver = (
    viewportWidthPx: number,
    viewportHeightPx: number,
  ) => { leftPx: number; topPx: number; widthPx: number; heightPx: number };

  /** Captured `innerWidth` descriptor before this describe stubs it. */
  let companionShelfWindowWidthDescriptor: PropertyDescriptor | undefined;
  /** Captured `innerHeight` descriptor before this describe stubs it. */
  let companionShelfWindowHeightDescriptor: PropertyDescriptor | undefined;

  beforeEach(() => {
    companionShelfWindowWidthDescriptor = Object.getOwnPropertyDescriptor(
      window,
      'innerWidth',
    );
    companionShelfWindowHeightDescriptor = Object.getOwnPropertyDescriptor(
      window,
      'innerHeight',
    );
  });

  afterEach(() => {
    if (companionShelfWindowWidthDescriptor !== undefined) {
      Object.defineProperty(
        window,
        'innerWidth',
        companionShelfWindowWidthDescriptor,
      );
    }
    if (companionShelfWindowHeightDescriptor !== undefined) {
      Object.defineProperty(
        window,
        'innerHeight',
        companionShelfWindowHeightDescriptor,
      );
    }
    document.body.innerHTML = '';
  });

  /**
   * Stubs the jsdom window dimensions (the red-home pattern).
   *
   * @param widthPx - The window inner width to publish.
   * @param heightPx - The window inner height to publish.
   */
  const stubCompanionShelfWindowDimensions = (
    widthPx: number,
    heightPx: number,
  ): void => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: widthPx,
    });
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: heightPx,
    });
  };

  /**
   * Loads the exported slot resolver behind the companion gate.
   *
   * @returns The exported resolver (the slice implementation ships it).
   */
  const companionShelfSlotResolver =
    async (): Promise<CompanionShelfSlotResolver> => {
      const adapterModule =
        (await import('./network-panel.adapter')) as unknown as {
          resolveNeatRacingNetworkPanelDockedRightShelfSlot?: unknown;
        };
      const resolver =
        adapterModule.resolveNeatRacingNetworkPanelDockedRightShelfSlot;
      if (typeof resolver !== 'function') {
        throw new Error(
          'P4-S2C-shelf companion gate: the exported pure docked-right shelf-slot resolver resolveNeatRacingNetworkPanelDockedRightShelfSlot is missing from examples/neatRacing/visualizer/network-panel.adapter.ts',
        );
      }
      return resolver as CompanionShelfSlotResolver;
    };

  it('pins the exact deterministic 1a slot at the recorded viewport', async () => {
    // Arrange: the companion gate at the recorded defect viewport.
    const resolver = await companionShelfSlotResolver();

    // Act + Assert: the deterministic 1a design constants hold exactly —
    // docked right (1689 - 506), clear of the top HUD band (top 48), a
    // third-viewport shelf (506) with the panel aspect lock (379).
    expect(resolver(1689, 1325)).toEqual({
      leftPx: 1183,
      topPx: 48,
      widthPx: 506,
      heightPx: 379,
    });
  });

  it('sizes a measurable host from the VIEWPORT slot, never the raw host bounds', async () => {
    // Arrange: the companion gate with the window at the recorded defect
    // viewport while the host measure is DETUNED to 2029x1325 — a
    // raw-bounds sizing would size the backing store from 2029x1325 (the
    // recorded 1a defect lane), never from the viewport slot.
    const resolver = await companionShelfSlotResolver();
    stubCompanionShelfWindowDimensions(1689, 1325);
    const networkCanvas = document.createElement('canvas');
    const networkCanvasHost = document.createElement('div');
    document.body.append(networkCanvasHost);
    jest.spyOn(networkCanvasHost, 'clientWidth', 'get').mockReturnValue(2029);
    jest.spyOn(networkCanvasHost, 'clientHeight', 'get').mockReturnValue(1325);

    // Act: the idle-path installer applies its sizing once on install.
    const resizeHandle = installNeatRacingNetworkResize(
      networkCanvas,
      networkCanvasHost,
      jest.fn(),
    );
    resizeHandle.uninstall();
    const viewportSlot = resolver(1689, 1325);

    // Assert: the backing store carries the VIEWPORT slot.
    expect({
      backingWidth: networkCanvas.width,
      backingHeight: networkCanvas.height,
    }).toEqual({
      backingWidth: viewportSlot.widthPx,
      backingHeight: viewportSlot.heightPx,
    });
  });
});

/**
 * GREEN its for the LIVE-CARRIER panel feed (slice P4-S4-live-feed, AC-102).
 *
 * The frozen Step-26 RED home (examples/neatRacing/visualizer/
 * ai-mode.visualizer.test.ts) pins the LIVE-FEED contract behaviorally and
 * is NEVER edited; this block appends the adapter-side green companions:
 * the live-carrier resolver answers the carrier's OWN node / connection /
 * activation data, keeps activations LIVE between resolutions, answers the
 * elected candidate riding in alone, prefers the carrier's OWN network
 * over its candidate view, and NEVER degrades into the retired idle-path
 * stub frame (the named error, not a silent 84-node placeholder).
 */
describe('NeatRacing network panel live-carrier feed (P4-S4-live-feed)', () => {
  /** Live evolved fixture input count (mirrors the frozen RED home). */
  const LIVE_CARRIER_INPUT_COUNT = 4;
  /** Live evolved fixture output count (mirrors the frozen RED home). */
  const LIVE_CARRIER_OUTPUT_COUNT = 2;
  /** Live evolved fixture seed (mirrors the frozen RED home). */
  const LIVE_CARRIER_NETWORK_SEED = 4242;
  /** Live evolved fixture hidden node minimum (mirrors the RED home). */
  const LIVE_CARRIER_HIDDEN_COUNT = 3;
  /** The retired idle stub's node count (Tier A 82 channels + 2 outputs). */
  const RETIRED_IDLE_STUB_NODE_COUNT = 84;

  /**
   * Builds the deterministic live evolved network.
   *
   * Educational note — the fixture is the REAL evolved topology recipe: a
   * seeded controller whose nodes carry distinct live activations (the
   * nodeIndex-derived rounded values the frozen RED home overwrites), so
   * the activation witness cannot pass vacuously.
   */
  const buildLiveCarrierNetwork = (): Network => {
    const network = new Network(
      LIVE_CARRIER_INPUT_COUNT,
      LIVE_CARRIER_OUTPUT_COUNT,
      {
        seed: LIVE_CARRIER_NETWORK_SEED,
        minHidden: LIVE_CARRIER_HIDDEN_COUNT,
      },
    );
    network.nodes.forEach((node, nodeIndex) => {
      node.activation = Math.round(((nodeIndex + 1) / 10) * 1e4) / 1e4;
    });
    return network;
  };

  /**
   * Builds the DOM-free 2D context stub (the frozen RED home's recipe).
   */
  const buildLiveCarrierStubContext = (): CanvasRenderingContext2D =>
    ({
      canvas: { width: 900, height: 240 },
      measureText: (): { width: number } => ({ width: 0 }),
      font: '10px sans-serif',
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 1,
      textAlign: 'center',
      textBaseline: 'middle',
      globalAlpha: 1,
      save: (): void => undefined,
      restore: (): void => undefined,
      clearRect: (): void => undefined,
      beginPath: (): void => undefined,
      closePath: (): void => undefined,
      moveTo: (): void => undefined,
      lineTo: (): void => undefined,
      stroke: (): void => undefined,
      fill: (): void => undefined,
      fillRect: (): void => undefined,
      strokeRect: (): void => undefined,
      fillText: (): void => undefined,
      strokeText: (): void => undefined,
      rect: (): void => undefined,
      arc: (): void => undefined,
      translate: (): void => undefined,
      scale: (): void => undefined,
      rotate: (): void => undefined,
      clip: (): void => undefined,
      setLineDash: (): void => undefined,
      getLineDash: (): number[] => [],
      setTransform: (): void => undefined,
      drawImage: (): void => undefined,
    }) as unknown as CanvasRenderingContext2D;

  /**
   * Reads the sorted rounded activation multiset off a resolved frame (the
   * frozen RED home's reader law: nested node.activation first, direct
   * activation fallback).
   */
  const readFrameActivations = (
    frame: NetworkVisualizationResolvedFrame,
  ): number[] => {
    const activations = frame.positionedScene.positionedNodes.map(
      (positionedNode): number => {
        const nodeHolder = positionedNode as unknown as {
          node?: { activation?: number };
          activation?: number;
        };
        const nested = nodeHolder.node?.activation;
        const activation =
          typeof nested === 'number' ? nested : nodeHolder.activation;
        return typeof activation === 'number' && Number.isFinite(activation)
          ? Math.round(activation * 1e4) / 1e4
          : 0;
      },
    );
    return [...activations].sort((left, right) => left - right);
  };

  /** Reads the network's own live activation multiset (sorted, rounded). */
  const readNetworkActivations = (network: Network): number[] =>
    [...network.nodes.map((node) => node.activation ?? 0)].sort(
      (left, right) => left - right,
    );

  it('resolves the carrier OWN network into the shared frame (live node data)', () => {
    const network = buildLiveCarrierNetwork();
    const frame = resolveNeatRacingNetworkPanelLiveCarrierFrame(
      buildLiveCarrierStubContext(),
      { agentId: 2, progress: 120, network },
    );
    expect(frame.positionedScene.positionedNodes.length).toBe(
      network.nodes.length,
    );
    expect(frame.runtimeConnections.length).toBe(network.connections.length);
  });

  it('carries the live activation data (multiset equality, never vacuous)', () => {
    const network = buildLiveCarrierNetwork();
    const frame = resolveNeatRacingNetworkPanelLiveCarrierFrame(
      buildLiveCarrierStubContext(),
      { agentId: 2, progress: 120, network },
    );
    expect(readFrameActivations(frame)).toEqual(
      readNetworkActivations(network),
    );
  });

  it('never resolves the retired idle stub (the 84-node placeholder)', () => {
    const network = buildLiveCarrierNetwork();
    const frame = resolveNeatRacingNetworkPanelLiveCarrierFrame(
      buildLiveCarrierStubContext(),
      { agentId: 2, progress: 120, network },
    );
    expect(frame.positionedScene.positionedNodes.length).not.toBe(
      RETIRED_IDLE_STUB_NODE_COUNT,
    );
  });

  it('keeps the activation data LIVE across resolutions', () => {
    const network = buildLiveCarrierNetwork();
    const context = buildLiveCarrierStubContext();
    resolveNeatRacingNetworkPanelLiveCarrierFrame(context, {
      agentId: 2,
      progress: 120,
      network,
    });
    // The lane evolves: one node's live activation advances mid-round.
    const evolvedValue = 0.9999;
    network.nodes[0].activation = evolvedValue;
    const evolvedFrame = resolveNeatRacingNetworkPanelLiveCarrierFrame(
      context,
      { agentId: 2, progress: 120, network },
    );
    expect(readFrameActivations(evolvedFrame)).toContain(evolvedValue);
  });

  it('answers an elected candidate riding in as the carrier alone', () => {
    const network = buildLiveCarrierNetwork();
    const frame = resolveNeatRacingNetworkPanelLiveCarrierFrame(
      buildLiveCarrierStubContext(),
      { agentId: 2, progress: 120, network } as NeatRacingNetworkPanelCandidate,
    );
    expect(frame.positionedScene.positionedNodes.length).toBe(
      network.nodes.length,
    );
    expect(frame.runtimeConnections.length).toBe(network.connections.length);
  });

  it('prefers the carrier OWN network when both network slots differ', () => {
    const ownNetwork = buildLiveCarrierNetwork();
    const candidateNetwork = buildLiveCarrierNetwork();
    candidateNetwork.nodes[0].activation = 0.5;
    const frame = resolveNeatRacingNetworkPanelLiveCarrierFrame(
      buildLiveCarrierStubContext(),
      {
        agentId: 2,
        progress: 120,
        network: ownNetwork,
        candidate: { agentId: 2, progress: 120, network: candidateNetwork },
      },
    );
    expect(frame.positionedScene.positionedNodes.length).toBe(
      ownNetwork.nodes.length,
    );
    expect(readFrameActivations(frame)).toEqual(
      readNetworkActivations(ownNetwork),
    );
  });

  it('raises the named idle-stub error for a network-less carrier', () => {
    expect(() =>
      resolveNeatRacingNetworkPanelLiveCarrierFrame(
        buildLiveCarrierStubContext(),
        { agentId: 2, progress: 120 },
      ),
    ).toThrow(/NeatRacingNetworkPanelLiveCarrierError[\s\S]*idle-path stub/);
  });

  it('falls back to the candidate view network when the own slot is absent', () => {
    const candidateNetwork = buildLiveCarrierNetwork();
    const frame = resolveNeatRacingNetworkPanelLiveCarrierFrame(
      buildLiveCarrierStubContext(),
      {
        agentId: 2,
        progress: 120,
        candidate: { agentId: 2, progress: 120, network: candidateNetwork },
      },
    );
    expect(frame.positionedScene.positionedNodes.length).toBe(
      candidateNetwork.nodes.length,
    );
    expect(frame.runtimeConnections.length).toBe(
      candidateNetwork.connections.length,
    );
  });
});
