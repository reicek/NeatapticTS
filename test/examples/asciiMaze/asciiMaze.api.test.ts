/**
 * @jest-environment jsdom
 */

import { start } from './browser-entry';
import { pollUntil } from '../../utils/pollUntil';

/**
 * Comprehensive, educational API contract tests for the ASCII Maze example.
 *
 * Goals:
 * 1. Demonstrate single-expectation test style for precise failure diagnostics.
 * 2. Use AAA (Arrange, Act, Assert) explicitly so new contributors learn the pattern.
 * 3. Showcase deterministic async verification via polling rather than arbitrary sleeps.
 * 4. Validate cancellation semantics (AbortSignal) and graceful shutdown.
 *
 * Neuro‑evolution contexts emit telemetry streams; this suite ensures at least one
 * telemetry emission and validates surface API invariants without exhaustive behavior
 * (intensive loops are intentionally avoided for fast CI).
 */

/** Public example handle (opaque here for black‑box contract validation) */
interface IExampleHandle {
  stop: () => void;
  isRunning: () => boolean;
  done: Promise<unknown>;
  onTelemetry: (cb: (t: any) => void) => () => void;
  getTelemetry: () => any;
}

/** Shared context for the main (non-abort) lifecycle scenario */
interface IMainScenarioContext {
  host: HTMLElement;
  handle: IExampleHandle;
  telemetry: any[];
  unsubscribe: () => void;
}

/** Scenario root: Primary lifecycle without external cancellation */
describe('ASCII Maze Example – Primary Lifecycle', () => {
  const ctx: IMainScenarioContext = {
    host: (undefined as unknown) as HTMLElement,
    handle: (undefined as unknown) as IExampleHandle,
    telemetry: [],
    unsubscribe: () => {},
  };

  /** Arrange & Act once: start example and collect first telemetry event */
  beforeAll(async () => {
    // Arrange: create DOM host container replicating production embedding.
    const host = document.createElement('div');
    host.id = 'ascii-maze-output';
    host.innerHTML =
      '<div id="ascii-maze-live"></div><div id="ascii-maze-archive"></div>';
    document.body.appendChild(host);
    ctx.host = host;

    // Act: start the example (async initialization of evolution + telemetry loop).
    ctx.handle = (await start(host)) as IExampleHandle;

    // Act: subscribe to telemetry stream; store unsubscribe for later cleanup.
    ctx.unsubscribe = ctx.handle.onTelemetry((t) => ctx.telemetry.push(t));

    // Act: wait deterministically until at least one telemetry event is received.
    await pollUntil(() => ctx.telemetry.length > 0, {
      timeoutMs: 3000,
      intervalMs: 25,
    });
  });

  /** Teardown: stop and ensure quiescence */
  afterAll(async () => {
    try {
      ctx.unsubscribe?.();
      if (ctx.handle?.isRunning()) {
        ctx.handle.stop();
        await pollUntil(() => !ctx.handle.isRunning(), {
          timeoutMs: 3000,
          intervalMs: 25,
        });
      }
    } catch {
      /* ignore teardown errors */
    }
  });

  // --- Handle Structural Contract ---
  describe('handle structure', () => {
    it('returns a truthy handle instance', () => {
      // Assert
      expect(!!ctx.handle).toBe(true);
    });
    it('exposes stop() method', () => {
      expect(typeof ctx.handle.stop).toBe('function');
    });
    it('exposes isRunning() method', () => {
      expect(typeof ctx.handle.isRunning).toBe('function');
    });
    it('exposes done Promise', () => {
      expect(ctx.handle.done instanceof Promise).toBe(true);
    });
    it('exposes onTelemetry() subscription method', () => {
      expect(typeof ctx.handle.onTelemetry).toBe('function');
    });
    it('exposes getTelemetry() accessor', () => {
      expect(typeof ctx.handle.getTelemetry).toBe('function');
    });
  });

  // --- Telemetry Liveness ---
  describe('telemetry liveness', () => {
    it('emits at least one telemetry event', () => {
      expect(ctx.telemetry.length > 0).toBe(true);
    });
    it('snapshot generation is numeric', () => {
      const snap = ctx.handle.getTelemetry();
      expect(typeof snap.generation === 'number').toBe(true);
    });
    it('snapshot details field is present (may be null early)', () => {
      const snap = ctx.handle.getTelemetry();
      expect(snap.details !== undefined).toBe(true);
    });
    it('reports running state true prior to stop()', () => {
      expect(ctx.handle.isRunning()).toBe(true);
    });
  });

  // --- Graceful Stop ---
  describe('graceful stop()', () => {
    it('transitions isRunning() to false after stop()', async () => {
      // Act: issue stop then poll for idle state.
      ctx.handle.stop();
      await pollUntil(() => !ctx.handle.isRunning(), {
        timeoutMs: 3000,
        intervalMs: 25,
      });
      // Assert
      expect(ctx.handle.isRunning()).toBe(false);
    });
  });
});

/** Scenario: External cancellation via AbortSignal */
describe('ASCII Maze Example – AbortSignal cancellation', () => {
  let abortedHandle: IExampleHandle;

  beforeAll(async () => {
    // Arrange: independent host container for abort scenario.
    const host = document.createElement('div');
    host.id = 'ascii-maze-output-abort';
    host.innerHTML =
      '<div id="ascii-maze-live"></div><div id="ascii-maze-archive"></div>';
    document.body.appendChild(host);

    // Arrange: create controller & start with signal.
    const controller = new AbortController();
    abortedHandle = (await start(host, {
      signal: controller.signal,
    })) as IExampleHandle;

    // Act: schedule abort soon after start to exercise cooperative cancellation.
    setTimeout(() => controller.abort(), 50);

    // Act: wait for example to observe abort & halt.
    await pollUntil(() => !abortedHandle.isRunning(), {
      timeoutMs: 3000,
      intervalMs: 25,
    });
  });

  it('isRunning() becomes false after external abort', () => {
    expect(abortedHandle.isRunning()).toBe(false);
  });
});
