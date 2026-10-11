/**
 * @jest-environment jsdom
 */

/**
 * RED → IMPLEMENT → GREEN contract tests for the Neat Racing editor shell and
 * browser entry (plans/neatracing-voxel-racer.plans.md — phase 1, step 08,
 * slice P1-S5-C; spec section 8.3, LD-28).
 *
 * The suite pins the slice's two partial acceptance criteria:
 *
 * - AC-019 (partial): the editor page mounts the WebGL2 preview canvas into
 *   the preview host beside the tool frame while the shell keeps the page
 *   scroll-free (`overflow: hidden`).
 * - AC-020 (partial): the boot path initializes the editor state container
 *   (`{ model: null }`) and exposes it to the later tool slices.
 *
 * The editor boot mirrors the game boot (browser-entry.ts): it dogfoods the
 * shared WebGL2 context service — the entry never touches `canvas.getContext`
 * directly beyond what `initRenderingContext` does — and mounts the context
 * service's fallback notice instead of throwing when WebGL2 is unavailable
 * (spec principle from section 8.1; LD-1). The shell source text (editor.html)
 * is asserted through `readFileSync` so the page contract is enforced in CI
 * without a browser.
 */

import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { NEATRACING_FALLBACK_MESSAGE } from '../engine/context.service';

// Side-effect import: evaluating the entry module registers
// `window.neatRacingEditorStart` once on the jsdom host.
// RED (pre-implementation): resolving './editor-entry' fails with
// "Cannot find module './editor-entry'" until the shell exists.
import './editor-entry';
import type {
  NeatRacingEditorSession,
  NeatRacingEditorStart,
} from './editor-entry';

const EDITOR_HTML_PATH = resolve(
  process.cwd(),
  'examples',
  'neatRacing',
  'editor',
  'editor.html',
);

/** The preview host id the editor page and the boot contract share. */
const PREVIEW_HOST_ID = 'neatracing-editor-preview';

/** The editor tool frame container id pinned by the shell markup. */
const EDITOR_TOOLS_ID = 'neatracing-editor-tools';

/** Returns the window global after the entry module side effect ran. */
function getEditorStart(): NeatRacingEditorStart {
  const start = window.neatRacingEditorStart;
  if (typeof start !== 'function') {
    throw new TypeError(
      'window.neatRacingEditorStart was not registered by editor-entry.ts',
    );
  }
  return start;
}

/** Boots a fresh editor session against a live jsdom preview host. */
function bootEditorInHost(): NeatRacingEditorSession {
  const start = getEditorStart();
  return start(PREVIEW_HOST_ID);
}

/** Reads the committed editor.html source text for markup contracts. */
function readEditorHtml(): string {
  return readFileSync(EDITOR_HTML_PATH, 'utf8');
}

describe('Neat Racing editor shell + entry (slice P1-S5-C)', () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <main id="neatracing-editor">
        <section id="${PREVIEW_HOST_ID}" aria-label="3D model preview"></section>
        <aside id="${EDITOR_TOOLS_ID}" aria-label="Editor tools"></aside>
      </main>
    `;
    // jsdom does not ship a WebGL implementation: stub the canvas context so
    // the boot path exercises the context service's fallback branch the same
    // way a browser without WebGL2 would (LD-1), without "Not implemented"
    // console noise.
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  });

  describe('AC-019 (partial): preview canvas mounts into the tool-frame shell', () => {
    it('registers window.neatRacingEditorStart on import', () => {
      expect(typeof getEditorStart()).toBe('function');
    });

    it('mounts the WebGL2 preview canvas into the preview host', () => {
      bootEditorInHost();
      expect(
        document.getElementById(PREVIEW_HOST_ID)?.querySelector('canvas'),
      ).toBeInstanceOf(HTMLCanvasElement);
    });

    it('acquires the WebGL2 context exactly once per boot on the mounted canvas', () => {
      const getContextSpy = HTMLCanvasElement.prototype.getContext as jest.Mock;
      // History is not auto-cleared between tests (clearMocks is false):
      // earlier boots leave counts on the reused prototype mock, so pin the
      // once-per-boot budget to this test's own boot.
      getContextSpy.mockClear();
      bootEditorInHost();
      expect(getContextSpy).toHaveBeenCalledTimes(1);
    });

    it('never throws when WebGL2 is unavailable and mounts the fallback notice', () => {
      let session: NeatRacingEditorSession | null = null;
      expect(() => {
        session = bootEditorInHost();
      }).not.toThrow();
      expect(document.getElementById(PREVIEW_HOST_ID)?.innerHTML).toContain(
        NEATRACING_FALLBACK_MESSAGE,
      );
      expect(session).not.toBeNull();
    });

    it('mounts cleanly without a fallback notice when WebGL2 is available', () => {
      jest
        .spyOn(HTMLCanvasElement.prototype, 'getContext')
        .mockReturnValue({} as unknown as WebGL2RenderingContext);
      bootEditorInHost();
      expect(
        document.getElementById(PREVIEW_HOST_ID)?.querySelector('canvas'),
      ).toBeInstanceOf(HTMLCanvasElement);
      expect(document.getElementById(PREVIEW_HOST_ID)?.innerHTML).not.toContain(
        NEATRACING_FALLBACK_MESSAGE,
      );
    });

    it('returns a stop handle that removes the preview canvas and fallback notice', () => {
      const session = bootEditorInHost();
      session.stop();
      const host = document.getElementById(PREVIEW_HOST_ID);
      expect(host?.querySelector('canvas')).toBeNull();
      expect(
        Array.from(host?.children ?? []).some((child) =>
          child.textContent?.includes(NEATRACING_FALLBACK_MESSAGE),
        ),
      ).toBe(false);
    });

    it('sizes the preview canvas drawing buffer to the preview host at boot', () => {
      const host = document.getElementById(PREVIEW_HOST_ID);
      Object.defineProperty(host, 'clientWidth', {
        value: 800,
        configurable: true,
      });
      Object.defineProperty(host, 'clientHeight', {
        value: 600,
        configurable: true,
      });
      bootEditorInHost();
      const canvas = host?.querySelector('canvas');
      expect(canvas?.width).toBe(800);
      expect(canvas?.height).toBe(600);
    });

    it('throws a descriptive error when the preview host is missing', () => {
      const start = getEditorStart();
      expect(() => start('missing-editor-preview-host')).toThrow(
        /Preview host #missing-editor-preview-host not found/u,
      );
    });
  });

  describe('AC-020 (partial): editor state container is initialized and exposed', () => {
    it('exposes an initialized editor state container on the session', () => {
      const session = bootEditorInHost();
      expect(session.state).toEqual({ model: null });
      expect(session.state).toBeInstanceOf(Object);
    });

    it('gives each boot a fresh state container instance', () => {
      const first = bootEditorInHost();
      const second = bootEditorInHost();
      expect(second.state).not.toBe(first.state);
    });
  });

  describe('editor.html shell source-text contract', () => {
    it('commits a non-empty editor page', () => {
      expect(readEditorHtml().trim().length).toBeGreaterThan(0);
    });

    it('keeps the editor page scroll-free via overflow hidden', () => {
      expect(readEditorHtml()).toContain('overflow: hidden');
    });

    it('provides the preview host container', () => {
      expect(readEditorHtml()).toContain(`id="${PREVIEW_HOST_ID}"`);
    });

    it('provides the editor tool frame container', () => {
      expect(readEditorHtml()).toContain(`id="${EDITOR_TOOLS_ID}"`);
    });

    it('loads the editor entry bundle by its stable dev-server name', () => {
      expect(readEditorHtml()).toContain('editor.bundle.js');
    });

    it('boots through window.neatRacingEditorStart on load', () => {
      expect(readEditorHtml()).toContain('neatRacingEditorStart');
    });

    it('routes the boot call through the committed preview host id', () => {
      expect(readEditorHtml()).toContain(PREVIEW_HOST_ID);
    });
  });

  describe('surfaces preview pass build failures observably (fix-packet-P1-composition-green)', () => {
    it('logs the labeled adapter build failure from the RAF lane and never throws from boot', async () => {
      // Arrange — a context whose createShader is callable but fails the
      // lazy CarPass build (jsdom cannot compile GLSL; the adapter's catch
      // exists for exactly this), plus a fetch chain that delivers a
      // decodable 1x1x1 document so the loop reaches the adapter's draw.
      const crashDocument = {
        meta: {
          name: 'preview-crash-model',
          dims: [1, 1, 1],
          symmetric: false,
        },
        palette: [
          [0, 0, 0, 0, 0],
          [255, 0, 0, 255, 0],
        ],
        slices: [['0101']],
      };
      const consoleError = jest
        .spyOn(console, 'error')
        .mockImplementation(() => undefined);
      const originalFetch = window.fetch;
      const originalRaf = window.requestAnimationFrame;
      const originalCancelRaf = window.cancelAnimationFrame;
      const frameQueue = new Map<number, (timeMs: number) => void>();
      let nextFrameId = 1;
      window.requestAnimationFrame = ((
        callback: (timeMs: number) => void,
      ): number => {
        const frameId = nextFrameId;
        nextFrameId += 1;
        frameQueue.set(frameId, callback);
        return frameId;
      }) as typeof window.requestAnimationFrame;
      window.cancelAnimationFrame = ((frameId: number): void => {
        frameQueue.delete(frameId);
      }) as typeof window.cancelAnimationFrame;
      window.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
        const url = String(input);
        if (url.includes('/api/models/')) {
          return {
            ok: true,
            json: async () => ({ document: crashDocument }),
          } as unknown as Response;
        }
        return {
          ok: true,
          json: async () => ({ models: ['preview-crash-model'] }),
        } as unknown as Response;
      }) as typeof window.fetch;
      const failingContext = { createShader: (): null => null };
      const contextSpy = jest
        .spyOn(HTMLCanvasElement.prototype, 'getContext')
        .mockReturnValue(failingContext as unknown as WebGL2RenderingContext);
      let session: NeatRacingEditorSession | null = null;
      try {
        // Act — boot the real entry; let the fire-and-forget fetch chain
        // land the document; then pump two RAF frames through the loop.
        session = getEditorStart()('neatracing-editor-preview');
        await new Promise<void>((resolve) => {
          setTimeout(resolve, 0);
        });
        await new Promise<void>((resolve) => {
          setTimeout(resolve, 0);
        });
        const pumpAll = (): void => {
          const queued = [...frameQueue.values()];
          frameQueue.clear();
          for (const callback of queued) {
            callback(16.6);
          }
        };
        pumpAll(); // document frame: decode -> bins -> adapter -> build throw
        pumpAll(); // next frame: same bins identity -> no rebuild, no new surface
        session.stop();
        // Assert — exactly one labeled failure; lane stayed scheduled until
        // stop; and stop fully drained the queue through cancelAnimationFrame.
        const buildFailures = consoleError.mock.calls.filter((call) =>
          String(call[0]).includes(
            'neatRacing editor preview: preview car pass build failed',
          ),
        );
        expect(buildFailures).toHaveLength(1);
        expect(consoleError.mock.calls).toHaveLength(1);
        expect(frameQueue.size).toBe(0);
      } finally {
        contextSpy.mockRestore();
        consoleError.mockRestore();
        window.fetch = originalFetch;
        window.requestAnimationFrame = originalRaf;
        window.cancelAnimationFrame = originalCancelRaf;
        session?.stop();
      }
    });
  });
});
