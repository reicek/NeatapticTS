/**
 * Red-phase contract tests for the NeatRacing WebGL2 context service
 * (slice P1-S1-C; AC-002 graceful fallback + context loss G5, AC-040 shader
 * warmup sweep with a named program-count constant).
 *
 * The Jest project runs in a plain Node environment, so a mock canvas
 * (listener registry + getContext spy) and a mock WebGL2 context drive every
 * service path without a real GPU or DOM.
 */
import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_FALLBACK_MESSAGE,
  NEATRACING_WARMED_PROGRAM_COUNT,
  NEATRACING_WARMED_PROGRAM_KEYS,
  initRenderingContext,
  runShaderWarmup,
} from './context.service';
import type { ShaderWarmupProgram } from './context.service';

/** `gl.POINTS` constant value used by the mock off-screen draw. */
const MOCK_DRAW_MODE_POINTS = 0;

/** `gl.COLOR_BUFFER_BIT` mask value used by the mock off-screen clear. */
const MOCK_CLEAR_MASK_COLOR_BUFFER = 0x4000;

/** Vertex count for the mock off-screen triangle draw. */
const MOCK_TRIANGLE_VERTEX_COUNT = 3;

/** Shape of the context-life-cycle events the mock canvas dispatches. */
interface RecordedContextEvent {
  readonly type: 'webglcontextlost' | 'webglcontextrestored';
  readonly preventDefault: () => void;
}

/** Mutable harness the tests use to drive and inspect the mock canvas. */
interface MockCanvasHarness {
  readonly canvas: HTMLCanvasElement;
  getContextCallCount(): number;
  dispatchContextEvent(event: RecordedContextEvent): void;
}

/**
 * Creates a mock canvas exposing a controllable `getContext` result and a
 * listener registry so context-loss events can be simulated without a DOM.
 */
function createMockCanvas(
  getContextResult: WebGL2RenderingContext | null,
): MockCanvasHarness {
  let getContextCalls = 0;
  const listeners = new Map<
    string,
    Array<(event: RecordedContextEvent) => void>
  >();
  const canvas = {
    getContext(type: string): WebGL2RenderingContext | null {
      getContextCalls += 1;
      return type === 'webgl2' ? getContextResult : null;
    },
    addEventListener(
      type: string,
      listener: (event: RecordedContextEvent) => void,
    ): void {
      const registered = listeners.get(type) ?? [];
      registered.push(listener);
      listeners.set(type, registered);
    },
    removeEventListener(): void {
      // Retained for DOM shape parity; the service contract never calls it.
    },
  };
  return {
    canvas: canvas as unknown as HTMLCanvasElement,
    getContextCallCount: () => getContextCalls,
    dispatchContextEvent(event: RecordedContextEvent): void {
      for (const listener of listeners.get(event.type) ?? []) {
        listener(event);
      }
    },
  };
}

/** Creates a mock WebGL2 context with the methods the warmup sweep uses. */
function createMockWebGl2(): WebGL2RenderingContext {
  const gl = {
    viewport: (): void => undefined,
    clearColor: (): void => undefined,
    clear: (): void => undefined,
    drawArrays: (): void => undefined,
  };
  return gl as unknown as WebGL2RenderingContext;
}

/** Minimal document stub able to create fallback notice elements. */
function createMockDocumentRef(): Pick<Document, 'createElement'> {
  const documentRef = {
    createElement(tag: string): HTMLElement {
      return { tagName: tag, textContent: '' } as unknown as HTMLElement;
    },
  };
  return documentRef as unknown as Pick<Document, 'createElement'>;
}

describe('neatRacing engine context service', () => {
  describe('context acquisition (AC-002)', () => {
    it('initializes the WebGL2 context exactly once per canvas', () => {
      const gl = createMockWebGl2();
      const harness = createMockCanvas(gl);

      initRenderingContext({ canvas: harness.canvas });
      initRenderingContext({ canvas: harness.canvas });

      expect(harness.getContextCallCount()).toBe(1);
    });

    it('reuses the memoized context across repeated init calls', () => {
      const gl = createMockWebGl2();
      const harness = createMockCanvas(gl);

      const first = initRenderingContext({ canvas: harness.canvas });
      const second = initRenderingContext({ canvas: harness.canvas });

      expect(first.ok).toBe(true);
      expect(second.ok).toBe(true);
      expect(first.gl === gl && second.gl === gl).toBe(true);
    });

    it('returns a graceful fallback result when WebGL2 is unavailable', () => {
      const harness = createMockCanvas(null);
      const documentRef = createMockDocumentRef();

      const result = initRenderingContext({
        canvas: harness.canvas,
        documentRef,
      });

      expect(result.ok).toBe(false);
      expect(result.message).toBe(NEATRACING_FALLBACK_MESSAGE);
      expect(result.fallbackElement?.textContent).toBe(
        NEATRACING_FALLBACK_MESSAGE,
      );
    });

    it('calls preventDefault on webglcontextlost (G5)', () => {
      const gl = createMockWebGl2();
      const harness = createMockCanvas(gl);
      initRenderingContext({ canvas: harness.canvas });

      let preventDefaultCalls = 0;
      harness.dispatchContextEvent({
        type: 'webglcontextlost',
        preventDefault: () => {
          preventDefaultCalls += 1;
        },
      });

      expect(preventDefaultCalls).toBe(1);
    });

    it('suppresses rendering while the context is lost', () => {
      const gl = createMockWebGl2();
      const harness = createMockCanvas(gl);
      const result = initRenderingContext({ canvas: harness.canvas });

      harness.dispatchContextEvent({
        type: 'webglcontextlost',
        preventDefault: () => undefined,
      });

      expect(result.ok).toBe(true);
      expect(result.isContextLost()).toBe(true);
    });

    it('reinitializes GPU resources through the restore hook on restore', () => {
      const gl = createMockWebGl2();
      const harness = createMockCanvas(gl);
      let restoreHookCalls = 0;
      initRenderingContext({
        canvas: harness.canvas,
        onContextRestored: () => {
          restoreHookCalls += 1;
        },
      });

      harness.dispatchContextEvent({
        type: 'webglcontextlost',
        preventDefault: () => undefined,
      });
      harness.dispatchContextEvent({
        type: 'webglcontextrestored',
        preventDefault: () => undefined,
      });

      expect(restoreHookCalls).toBe(1);
    });

    it('clears the lost flag once the context is restored', () => {
      const gl = createMockWebGl2();
      const harness = createMockCanvas(gl);
      const result = initRenderingContext({ canvas: harness.canvas });

      harness.dispatchContextEvent({
        type: 'webglcontextlost',
        preventDefault: () => undefined,
      });
      harness.dispatchContextEvent({
        type: 'webglcontextrestored',
        preventDefault: () => undefined,
      });

      expect(result.ok).toBe(true);
      expect(result.isContextLost()).toBe(false);
    });
  });

  describe('shader warmup sweep (AC-040)', () => {
    it('compiles, links, and draws every registered program exactly once', () => {
      const gl = createMockWebGl2();
      const compiledKeys: string[] = [];
      const programs: ShaderWarmupProgram[] =
        NEATRACING_WARMED_PROGRAM_KEYS.map((key) => ({
          key,
          compileAndLink: (): void => {
            compiledKeys.push(key);
          },
          drawOffscreen: (context: WebGL2RenderingContext): void => {
            context.clear(MOCK_CLEAR_MASK_COLOR_BUFFER);
            context.drawArrays(
              MOCK_DRAW_MODE_POINTS,
              0,
              MOCK_TRIANGLE_VERTEX_COUNT,
            );
          },
        }));

      const result = runShaderWarmup(gl, programs);

      expect(result.warmedProgramCount).toBe(NEATRACING_WARMED_PROGRAM_COUNT);
      expect(result.offscreenDrawCount).toBe(NEATRACING_WARMED_PROGRAM_COUNT);
      expect(compiledKeys.length).toBe(NEATRACING_WARMED_PROGRAM_COUNT);
    });

    it('pins the warmed program count to a named constant within the spec range', () => {
      expect(NEATRACING_WARMED_PROGRAM_COUNT).toBe(
        NEATRACING_WARMED_PROGRAM_KEYS.length,
      );
      expect(NEATRACING_WARMED_PROGRAM_COUNT).toBeGreaterThanOrEqual(7);
      expect(NEATRACING_WARMED_PROGRAM_COUNT).toBeLessThanOrEqual(9);
    });
  });
});
