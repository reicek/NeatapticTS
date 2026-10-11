/// <reference lib="dom" />

/**
 * WebGL2 context service for the Neat Racing voxel racer engine (slice
 * P1-S1-C).
 *
 * The module owns three responsibilities behind one small seam so later
 * engine slices can rely on a stable contract:
 *
 * 1. Context acquisition (AC-002): acquire the `webgl2` context exactly once
 *    per canvas — WeakMap memoization keyed by the canvas element — and
 *    degrade gracefully, with a human-readable fallback notice element,
 *    instead of throwing when WebGL2 is unavailable.
 * 2. Context-loss handling (G5): `preventDefault()` is called on
 *    `webglcontextlost` so the browser keeps the context restorable, a lost
 *    flag lets the render loop suppress rendering while the context is gone,
 *    and an `onContextRestored` hook is where callers rebuild GPU resources.
 * 3. Shader warmup sweep (AC-040, P8): every registered program is compiled,
 *    linked, and drawn exactly once against a 1×1 offscreen target before the
 *    first visible frame, eliminating first-draw shader-compilation hitches.
 *    The registry is pinned by {@link NEATRACING_WARMED_PROGRAM_COUNT}.
 *
 * @module
 */

/**
 * Message surfaced to players when WebGL2 is unavailable (AC-002).
 *
 * Used both as the init result's `message` and as the text content of the
 * fallback notice element, so the DOM notice and the programmatic result can
 * never drift apart.
 */
export const NEATRACING_FALLBACK_MESSAGE =
  'WebGL2 is unavailable — Neat Racing is falling back to a reduced view.';

/**
 * Registry keys for every shader program warmed before the first frame.
 *
 * Six world passes (starfield, grid, pulse, track, car, glass) plus the three
 * bloom stages from G8 (threshold, separable blur, additive composite) give
 * the pinned nine-key sweep. Later engine slices register their programs under
 * exactly these keys, so the warmup sweep and the pass registry cannot drift
 * apart (AC-040).
 */
export const NEATRACING_WARMED_PROGRAM_KEYS = [
  'starfield',
  'grid',
  'pulse',
  'track',
  'car',
  'glass',
  'bloom-threshold',
  'bloom-blur',
  'bloom-additive',
] as const;

/**
 * Number of shader programs warmed by the pre-first-frame sweep (AC-040).
 *
 * Derived from {@link NEATRACING_WARMED_PROGRAM_KEYS} so the count and the
 * registry can never disagree; the spec pins the value to the 7–9 range.
 */
export const NEATRACING_WARMED_PROGRAM_COUNT: number =
  NEATRACING_WARMED_PROGRAM_KEYS.length;

/**
 * A single GPU program registered for the warmup sweep (AC-040).
 *
 * @property key - Registry key from {@link NEATRACING_WARMED_PROGRAM_KEYS};
 *   later pass slices use the same key for their draw pass.
 * @property compileAndLink - Builds the GPU program (compile + link) exactly
 *   once per sweep.
 * @property drawOffscreen - Issues a tiny offscreen draw so the driver
 *   compiles the render pipeline before the first visible frame.
 */
export interface ShaderWarmupProgram {
  key: string;
  compileAndLink: () => void;
  drawOffscreen: (context: WebGL2RenderingContext) => void;
}

/**
 * Counters returned by the shader warmup sweep.
 *
 * @property warmedProgramCount - Programs compiled and linked by the sweep.
 * @property offscreenDrawCount - Offscreen draws issued by the sweep; equals
 *   `warmedProgramCount` when every registered program drew successfully.
 */
export interface ShaderWarmupResult {
  warmedProgramCount: number;
  offscreenDrawCount: number;
}

/**
 * Options for {@link initRenderingContext}.
 *
 * @property canvas - Canvas element that will own the WebGL2 context.
 * @property documentRef - Document factory used to build the fallback notice
 *   element; omit in headless environments where no DOM exists.
 * @property onContextRestored - Hook invoked once after `webglcontextrestored`
 *   so callers can reinitialize GPU resources (programs, buffers, meshes).
 */
export interface InitRenderingContextOptions {
  canvas: HTMLCanvasElement;
  documentRef?: Pick<Document, 'createElement'>;
  onContextRestored?: () => void;
}

/**
 * Outcome of the WebGL2 context acquisition attempt (AC-002).
 *
 * @property ok - true when a live WebGL2 context was acquired.
 * @property gl - The acquired context, or null when WebGL2 is unavailable.
 * @property message - Fallback explanation; present only when ok is false.
 * @property fallbackElement - DOM notice carrying the fallback message, or
 *   null when no `documentRef` was supplied; present only when ok is false.
 * @property isContextLost - Whether rendering is currently suppressed by a
 *   context-loss event; the render loop polls this every frame.
 */
export interface RenderingContextInitResult {
  readonly ok: boolean;
  readonly gl: WebGL2RenderingContext | null;
  readonly message?: string;
  readonly fallbackElement?: HTMLElement | null;
  readonly isContextLost: () => boolean;
}

/** Tag used for the DOM fallback notice element. */
const NEATRACING_FALLBACK_NOTICE_TAG = 'div';

/** Origin corner of the 1×1 offscreen warmup viewport (AC-040). */
const WARMUP_VIEWPORT_ORIGIN = 0;

/** Offscreen warmup target is a single pixel; only pipeline warmup matters. */
const WARMUP_VIEWPORT_SIZE = 1;

/** Opaque-black clear color channels for the offscreen warmup target. */
const WARMUP_CLEAR_RED = 0;
const WARMUP_CLEAR_GREEN = 0;
const WARMUP_CLEAR_BLUE = 0;
const WARMUP_CLEAR_ALPHA = 1;

/** Mutable lost-flag shared between the loss listeners and the predicate. */
interface ContextLostState {
  lost: boolean;
}

/** Per-canvas memoization store backing the once-per-canvas contract. */
const contextResults = new WeakMap<
  HTMLCanvasElement,
  RenderingContextInitResult
>();

/**
 * Acquire the WebGL2 rendering context for a canvas, exactly once per canvas.
 *
 * The result is memoized in a WeakMap keyed by the canvas element, so repeated
 * boot calls never re-request the context. When WebGL2 is unavailable the
 * service degrades gracefully: the result carries the fallback message plus a
 * DOM notice element (when a `documentRef` is supplied) instead of throwing.
 *
 * On success the service registers the G5 context-loss listeners:
 * `webglcontextlost` calls `preventDefault()` and sets the lost flag (which
 * {@link RenderingContextInitResult.isContextLost} reports), and
 * `webglcontextrestored` clears the flag and fires the `onContextRestored`
 * hook where callers rebuild GPU resources.
 *
 * @param options - Canvas plus optional document factory and restore hook.
 * @returns The memoized acquisition result; inspect `ok` before using `gl`.
 *
 * @example
 * ```ts
 * const init = initRenderingContext({ canvas, documentRef: document });
 * if (init.ok && init.gl) {
 *   const warmup = runShaderWarmup(init.gl, buildWarmupPrograms());
 *   console.log(`warmed ${warmup.warmedProgramCount} programs`);
 * }
 * ```
 */
export function initRenderingContext(
  options: InitRenderingContextOptions,
): RenderingContextInitResult {
  const { canvas, documentRef, onContextRestored } = options;

  // Step 1: Reuse the memoized result — a canvas owns exactly one context.
  const memoized = contextResults.get(canvas);
  if (memoized !== undefined) {
    return memoized;
  }

  // Step 2: Acquire the context and prepare the shared lost-flag.
  // preserveDrawingBuffer is enabled so the composed frame stays readable
  // for screenshots, browser smoke tests, and end-user presentation without
  // relying on a racing readback before the compositor clears the buffer.
  const gl = canvas.getContext('webgl2', {
    alpha: true,
    premultipliedAlpha: true,
    preserveDrawingBuffer: true,
  });
  const lostState: ContextLostState = { lost: false };
  const isContextLost = (): boolean => lostState.lost;

  // Step 3: Build the graceful-fallback or live-context result.
  const result: RenderingContextInitResult =
    gl === null
      ? createFallbackResult(documentRef, isContextLost)
      : createLiveResult(
          canvas,
          gl,
          onContextRestored,
          lostState,
          isContextLost,
        );

  // Step 4: Memoize so later boot calls skip re-acquisition entirely.
  contextResults.set(canvas, result);
  return result;
}

/**
 * Run the shader warmup sweep against a 1×1 offscreen target (AC-040, P8).
 *
 * Every registered program is compiled, linked, and drawn exactly once so the
 * driver finishes pipeline compilation before the first visible frame; the
 * offscreen target keeps warmup pixels invisible to players. Call this right
 * after {@link initRenderingContext} succeeds and again from the
 * `onContextRestored` hook so restored contexts re-warm their pipelines.
 *
 * @param gl - Live WebGL2 context to warm.
 * @param programs - Programs registered for the sweep, keyed per
 *   {@link NEATRACING_WARMED_PROGRAM_KEYS}.
 * @returns Warmup counters: programs warmed and offscreen draws issued.
 *
 * @example
 * ```ts
 * const warmup = runShaderWarmup(gl, registeredPrograms);
 * console.log(warmup.warmedProgramCount === NEATRACING_WARMED_PROGRAM_COUNT);
 * ```
 */
export function runShaderWarmup(
  gl: WebGL2RenderingContext,
  programs: readonly ShaderWarmupProgram[],
): ShaderWarmupResult {
  // Step 1: Configure the invisible 1×1 offscreen warmup target.
  prepareWarmupRenderTarget(gl);

  // Step 2: Compile, link, and draw each registered program exactly once.
  let offscreenDrawCount = 0;
  for (const program of programs) {
    program.compileAndLink();
    program.drawOffscreen(gl);
    offscreenDrawCount += 1;
  }

  return { warmedProgramCount: programs.length, offscreenDrawCount };
}

/**
 * Build the graceful-fallback result used when WebGL2 is unavailable (AC-002).
 *
 * @param documentRef - Document factory for the notice element; when omitted
 *   (headless environments) the result carries `null` for `fallbackElement`.
 * @param isContextLost - Predicate bound to the shared lost-flag; always
 *   reports false because no context exists to lose.
 * @returns Result with `ok: false`, the fallback message, and the notice.
 */
function createFallbackResult(
  documentRef: Pick<Document, 'createElement'> | undefined,
  isContextLost: () => boolean,
): RenderingContextInitResult {
  const fallbackElement =
    documentRef?.createElement(NEATRACING_FALLBACK_NOTICE_TAG) ?? null;
  if (fallbackElement !== null) {
    fallbackElement.textContent = NEATRACING_FALLBACK_MESSAGE;
  }
  return {
    ok: false,
    gl: null,
    message: NEATRACING_FALLBACK_MESSAGE,
    fallbackElement,
    isContextLost,
  };
}

/**
 * Register the G5 context-loss listeners and build the live-context result.
 *
 * @param canvas - Canvas the loss/restore events fire on.
 * @param gl - Acquired WebGL2 context carried on the result.
 * @param onContextRestored - Caller hook that reinitializes GPU resources.
 * @param lostState - Shared mutable lost-flag toggled by the listeners.
 * @param isContextLost - Predicate bound to the shared lost-flag.
 * @returns Result with `ok: true` and the acquired context.
 */
function createLiveResult(
  canvas: HTMLCanvasElement,
  gl: WebGL2RenderingContext,
  onContextRestored: (() => void) | undefined,
  lostState: ContextLostState,
  isContextLost: () => boolean,
): RenderingContextInitResult {
  canvas.addEventListener('webglcontextlost', (event: Event): void => {
    // G5: preventDefault keeps the context restorable instead of letting the
    // browser discard it permanently.
    event.preventDefault();
    lostState.lost = true;
  });
  canvas.addEventListener('webglcontextrestored', (): void => {
    lostState.lost = false;
    // Rebuild GPU resources (programs, buffers, meshes) via the restore hook.
    onContextRestored?.();
  });
  return { ok: true, gl, isContextLost };
}

/**
 * Configure the invisible offscreen render target used by the warmup sweep.
 *
 * @param gl - Live WebGL2 context to configure before the sweep draws.
 */
function prepareWarmupRenderTarget(gl: WebGL2RenderingContext): void {
  gl.viewport(
    WARMUP_VIEWPORT_ORIGIN,
    WARMUP_VIEWPORT_ORIGIN,
    WARMUP_VIEWPORT_SIZE,
    WARMUP_VIEWPORT_SIZE,
  );
  gl.clearColor(
    WARMUP_CLEAR_RED,
    WARMUP_CLEAR_GREEN,
    WARMUP_CLEAR_BLUE,
    WARMUP_CLEAR_ALPHA,
  );
}
