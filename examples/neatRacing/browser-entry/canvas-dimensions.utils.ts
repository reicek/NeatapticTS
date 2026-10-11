/**
 * @module neatRacing/browser-entry/canvas-dimensions.utils
 *
 * Fixed-canvas strategy helpers for the neatRacing browser host (spec §6.2).
 *
 * The strategy renders into a fixed 720-pixel-tall backing store whose width
 * is derived from the client aspect ratio and clamped for ultrawide displays,
 * while CSS stretches the backing store into a letterboxed layout box so the
 * page never scrolls. The pure resolvers here are node-testable: every input
 * dimension is injected through parameters instead of reading `window`, and
 * the only DOM interaction is the thin `applyCanvasBackingStore` adapter,
 * which mirrors the Neatenstein parity shape.
 */

import {
  NEATRACING_FALLBACK_CANVAS_WIDTH,
  NEATRACING_FIXED_RENDER_HEIGHT,
  NEATRACING_MAX_BACKING_WIDTH,
} from './constants';

/** Reference dimensions available to the backing-store resolution ladder. */
export interface CanvasDimensionSources {
  /** Live client width of the canvas layout box in pixels; 0 when unmeasurable. */
  clientWidth: number;
  /** Live client height of the canvas layout box in pixels; 0 when unmeasurable. */
  clientHeight: number;
  /** Fallback viewport width in pixels used when client dimensions are missing. */
  viewportWidth: number;
  /** Fallback viewport height in pixels used when client dimensions are missing. */
  viewportHeight: number;
}

/** Resolved backing-store dimensions in pixels. */
export interface CanvasDimensions {
  /** Backing-store width in pixels, at most {@link NEATRACING_MAX_BACKING_WIDTH}. */
  width: number;
  /** Backing-store height in pixels, always {@link NEATRACING_FIXED_RENDER_HEIGHT}. */
  height: number;
}

/** Reference dimensions available to the letterbox layout resolution. */
export interface LetterboxLayoutSources {
  /** Available layout width in pixels (the viewport or CSS container width). */
  viewportWidth: number;
  /** Available layout height in pixels (the viewport or CSS container height). */
  viewportHeight: number;
  /** Rendered content width in pixels, usually the backing-store width. */
  contentWidth: number;
  /** Rendered content height in pixels, usually the backing-store height. */
  contentHeight: number;
}

/** Letterboxed layout box in pixels, sized to fit inside the viewport. */
export interface LetterboxLayout {
  /** Layout box width in pixels; never exceeds the viewport width. */
  width: number;
  /** Layout box height in pixels; never exceeds the viewport height. */
  height: number;
  /** Horizontal offset of the box inside the viewport in pixels; never negative. */
  offsetX: number;
  /** Vertical offset of the box inside the viewport in pixels; never negative. */
  offsetY: number;
}

/** Minimal structural surface the strategy reads and writes (a canvas or a stub). */
export interface ResizableCanvasSurface {
  /** Live client width of the layout box in pixels; 0 before layout or in stubs. */
  clientWidth: number;
  /** Live client height of the layout box in pixels; 0 before layout or in stubs. */
  clientHeight: number;
  /** Backing-store width in pixels, written by {@link applyCanvasBackingStore}. */
  width: number;
  /** Backing-store height in pixels, written by {@link applyCanvasBackingStore}. */
  height: number;
}

/** Inputs for the {@link updateRendererSize} orchestration. */
export interface RendererSizeUpdateSources {
  /** Canvas (or structural stub) whose backing store is resized. */
  canvas: ResizableCanvasSurface;
  /** Viewport width in pixels used when client dimensions are missing. */
  viewportWidth: number;
  /** Viewport height in pixels used when client dimensions are missing. */
  viewportHeight: number;
}

/** Result of a renderer resize: the applied backing dimensions and the letterbox layout. */
export interface RendererSizeUpdate {
  /** Backing-store dimensions that were applied to the canvas surface. */
  dimensions: CanvasDimensions;
  /** Letterboxed layout box matching the viewport for CSS positioning. */
  letterbox: LetterboxLayout;
}

/**
 * Resolves the backing-store dimensions from the available reference sizes.
 *
 * The fallback ladder mirrors the Neatenstein canvas strategy: proportional
 * scaling from the live client dimensions when measurable, otherwise
 * proportional scaling from the viewport dimensions, and finally the
 * {@link NEATRACING_FALLBACK_CANVAS_WIDTH} constant at the fixed render
 * height when nothing is measurable.
 *
 * @param sources - Client and viewport reference dimensions; zeros mark
 *   unmeasurable sources and advance the ladder.
 * @returns Backing-store dimensions at {@link NEATRACING_FIXED_RENDER_HEIGHT}
 *   height with the width clamped to {@link NEATRACING_MAX_BACKING_WIDTH}.
 *
 * @example
 * resolveCanvasBackingDimensions({ clientWidth: 3840, clientHeight: 1080, viewportWidth: 0, viewportHeight: 0 });
 * // → { width: 1920, height: 720 } (ultrawide 32:9 clamped)
 */
export function resolveCanvasBackingDimensions(
  sources: CanvasDimensionSources,
): CanvasDimensions {
  if (sources.clientWidth > 0 && sources.clientHeight > 0) {
    return proportionalBackingDimensions(
      sources.clientWidth,
      sources.clientHeight,
    );
  }
  if (sources.viewportWidth > 0 && sources.viewportHeight > 0) {
    return proportionalBackingDimensions(
      sources.viewportWidth,
      sources.viewportHeight,
    );
  }
  return {
    width: NEATRACING_FALLBACK_CANVAS_WIDTH,
    height: NEATRACING_FIXED_RENDER_HEIGHT,
  };
}

/**
 * Resolves the letterboxed layout box for the given viewport and content.
 *
 * The box scales the content uniformly to fit inside the viewport and is
 * centered with floor arithmetic so offsets stay non-negative and the box
 * never exceeds the viewport bounds (no page scroll). Degenerate inputs
 * (non-positive dimensions) collapse to a zero-size box at the origin.
 *
 * @param sources - Viewport bounds and content dimensions in pixels.
 * @returns Letterbox layout box with centered, non-negative offsets.
 *
 * @example
 * resolveLetterboxLayout({ viewportWidth: 2560, viewportHeight: 720, contentWidth: 1920, contentHeight: 720 });
 * // → { width: 1920, height: 720, offsetX: 320, offsetY: 0 } (horizontal letterbox)
 */
export function resolveLetterboxLayout(
  sources: LetterboxLayoutSources,
): LetterboxLayout {
  const { viewportWidth, viewportHeight, contentWidth, contentHeight } =
    sources;
  if (
    viewportWidth <= 0 ||
    viewportHeight <= 0 ||
    contentWidth <= 0 ||
    contentHeight <= 0
  ) {
    return { width: 0, height: 0, offsetX: 0, offsetY: 0 };
  }
  const scale = Math.min(
    viewportWidth / contentWidth,
    viewportHeight / contentHeight,
  );
  const width = Math.floor(contentWidth * scale);
  const height = Math.floor(contentHeight * scale);
  return {
    width,
    height,
    offsetX: Math.floor((viewportWidth - width) / 2),
    offsetY: Math.floor((viewportHeight - height) / 2),
  };
}

/**
 * Writes resolved dimensions onto the canvas backing store.
 *
 * Thin DOM adapter in the Neatenstein parity shape: the host assigns the
 * backing-store width and height so the render loop can draw at the fixed
 * strategy resolution while CSS stretches the element to the layout box.
 *
 * @param surface - Canvas element (or structural stub) to resize.
 * @param dimensions - Backing-store dimensions to apply.
 */
export function applyCanvasBackingStore(
  surface: ResizableCanvasSurface,
  dimensions: CanvasDimensions,
): void {
  surface.width = dimensions.width;
  surface.height = dimensions.height;
}

/**
 * Applies the letterbox layout to the canvas CSS box.
 *
 * The canvas backing store is fixed at 720 px height, so CSS owns the
 * on-screen display box. The layout box is centered with absolute positioning
 * inside the full-viewport output container; the computed offsets keep the
 * rendered box inside the viewport bounds without page scroll.
 *
 * @param canvas - Canvas element whose style box is positioned.
 * @param letterbox - Layout box produced by {@link resolveLetterboxLayout}.
 */
export function applyLetterboxLayoutStyle(
  canvas: HTMLCanvasElement,
  letterbox: LetterboxLayout,
): void {
  const style = canvas.style;
  style.position = 'absolute';
  style.left = `${String(letterbox.offsetX)}px`;
  style.top = `${String(letterbox.offsetY)}px`;
  style.width = `${String(letterbox.width)}px`;
  style.height = `${String(letterbox.height)}px`;
}

/**
 * Applies a full-window stretch style to the canvas.
 *
 * The canvas fills the entire output container (which is itself sized to the
 * viewport), so the rendered image stretches edge-to-edge like Neateinstein
 * rather than being letterboxed. Backing-store dimensions are matched to the
 * CSS client size so the projection aspect ratio stays correct.
 *
 * @param canvas - Canvas element whose style box is stretched to the container.
 */
export function applyFullWindowStretchStyle(canvas: HTMLCanvasElement): void {
  const style = canvas.style;
  style.position = 'absolute';
  style.left = '0';
  style.top = '0';
  style.width = '100%';
  style.height = '100%';
}

/**
 * Resolves backing-store dimensions that match the CSS client size.
 *
 * Falls back to the viewport dimensions when the canvas has not been laid out
 * yet, then to the fixed fallback constants when both sources are missing.
 *
 * @param sources - Client and viewport reference dimensions.
 * @returns Backing-store dimensions equal to the client size or fallback.
 */
export function resolveFullWindowBackingDimensions(
  sources: CanvasDimensionSources,
): CanvasDimensions {
  if (sources.clientWidth > 0 && sources.clientHeight > 0) {
    return { width: sources.clientWidth, height: sources.clientHeight };
  }
  if (sources.viewportWidth > 0 && sources.viewportHeight > 0) {
    return { width: sources.viewportWidth, height: sources.viewportHeight };
  }
  return {
    width: NEATRACING_FALLBACK_CANVAS_WIDTH,
    height: NEATRACING_FIXED_RENDER_HEIGHT,
  };
}

/** Inputs for the {@link updateFullWindowRendererSize} orchestration. */
export interface FullWindowRendererSizeUpdateSources {
  /** Canvas (or structural stub) whose backing store is resized. */
  canvas: ResizableCanvasSurface;
  /** Viewport width in pixels used when client dimensions are missing. */
  viewportWidth: number;
  /** Viewport height in pixels used when client dimensions are missing. */
  viewportHeight: number;
}

/**
 * Applies a full-window stretch resize to the renderer.
 *
 * Sets the canvas CSS box to fill its container, then resolves the backing
 * store from the live client dimensions (with viewport fallback) and applies
 * it. The returned dimensions match the pixel size the GPU will render into.
 *
 * @param sources - Canvas surface plus viewport fallback dimensions.
 * @returns The applied backing-store dimensions.
 */
export function updateFullWindowRendererSize(
  sources: FullWindowRendererSizeUpdateSources,
): CanvasDimensions {
  const dimensions = resolveFullWindowBackingDimensions({
    clientWidth: sources.canvas.clientWidth,
    clientHeight: sources.canvas.clientHeight,
    viewportWidth: sources.viewportWidth,
    viewportHeight: sources.viewportHeight,
  });
  applyCanvasBackingStore(sources.canvas, dimensions);
  return dimensions;
}

/**
 * Records a deterministic, human-readable size annotation for evidence
 * capture. The message includes the backing-store resolution, the CSS
 * letterbox box, and the host viewport dimensions so full-page screenshots
 * can be cross-checked against the source of truth.
 *
 * @param canvas - Resized canvas.
 * @param dimensions - Backing-store dimensions.
 * @param letterbox - CSS layout box.
 * @param viewportWidth - Host viewport width in pixels.
 * @param viewportHeight - Host viewport height in pixels.
 */
export function logRendererSizeAnnotation(
  canvas: HTMLCanvasElement,
  dimensions: CanvasDimensions,
  letterbox: LetterboxLayout,
  viewportWidth: number,
  viewportHeight: number,
): void {
  const isPreview = canvas.classList.contains(
    'neatracing-editor-preview-canvas',
  );
  const pageLabel = isPreview ? 'editor preview' : 'game page';
  console.info(`[neatRacing ${pageLabel}] renderer size annotation:`, {
    backingStore: { width: canvas.width, height: canvas.height },
    computedBacking: dimensions,
    letterbox,
    viewport: { width: viewportWidth, height: viewportHeight },
  });
}

/**
 * Applies a resize to the renderer: resolves, applies, and reports the layout.
 *
 * Orchestration entry in the Neatenstein parity shape. The backing-store
 * dimensions are resolved from the canvas client dimensions with a viewport
 * fallback, applied to the canvas surface, and the matching letterbox layout
 * is resolved so the host can position the canvas CSS box without scrolling.
 *
 * @param sources - Canvas surface plus viewport fallback dimensions.
 * @returns The applied backing-store dimensions and the letterbox layout.
 *
 * @example
 * updateRendererSize({ canvas, viewportWidth: window.innerWidth, viewportHeight: window.innerHeight });
 * // → { dimensions: { width: 1280, height: 720 }, letterbox: { ... } }
 */
export function updateRendererSize(
  sources: RendererSizeUpdateSources,
): RendererSizeUpdate {
  const dimensions = resolveCanvasBackingDimensions({
    clientWidth: sources.canvas.clientWidth,
    clientHeight: sources.canvas.clientHeight,
    viewportWidth: sources.viewportWidth,
    viewportHeight: sources.viewportHeight,
  });
  applyCanvasBackingStore(sources.canvas, dimensions);
  const letterbox = resolveLetterboxLayout({
    viewportWidth: sources.viewportWidth,
    viewportHeight: sources.viewportHeight,
    contentWidth: dimensions.width,
    contentHeight: dimensions.height,
  });
  return { dimensions, letterbox };
}

/**
 * Derives the proportional backing dimensions for a measurable reference size.
 *
 * The backing width is the reference aspect ratio applied to the fixed render
 * height, rounded to the nearest pixel and clamped between one pixel and the
 * {@link NEATRACING_MAX_BACKING_WIDTH} ultrawide guard.
 *
 * @param referenceWidth - Measurable reference width in pixels.
 * @param referenceHeight - Measurable reference height in pixels.
 * @returns Backing-store dimensions at the fixed render height.
 */
function proportionalBackingDimensions(
  referenceWidth: number,
  referenceHeight: number,
): CanvasDimensions {
  const proportionalWidth = Math.round(
    NEATRACING_FIXED_RENDER_HEIGHT * (referenceWidth / referenceHeight),
  );
  return {
    width: Math.min(
      NEATRACING_MAX_BACKING_WIDTH,
      Math.max(1, proportionalWidth),
    ),
    height: NEATRACING_FIXED_RENDER_HEIGHT,
  };
}
