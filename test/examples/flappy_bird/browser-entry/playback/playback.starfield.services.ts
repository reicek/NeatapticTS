import {
  FLAPPY_STARFIELD_CANVAS_CONTEXT_ID,
  FLAPPY_STARFIELD_CYAN_FILL_STYLE,
  FLAPPY_STARFIELD_COMPOSITE_SOURCE_OVER,
  FLAPPY_STARFIELD_FULL_ALPHA,
  FLAPPY_STARFIELD_INCLUSIVE_RANGE_OFFSET,
  FLAPPY_STARFIELD_MIN_DIMENSION_PX,
  FLAPPY_STARFIELD_NO_BLUR_PX,
  FLAPPY_STARFIELD_ORIGIN_PX,
  FLAPPY_STARFIELD_TRANSPARENT_SHADOW_COLOR,
} from '../../constants/constants';
import { createSeededRandom } from './playback.starfield.utils';
import type {
  CreateStarTileCanvasOptions,
  StarfieldCanvasDimensions,
  StarPlacement,
} from './playback.starfield.types';

/**
 * Pre-renders a deterministic tile that can be reused across animation frames.
 *
 * @param options - Declarative drawing recipe for one parallax layer.
 * @returns Canvas image source containing the rendered star strip.
 */
export function createStarTileCanvas(
  options: CreateStarTileCanvasOptions,
): CanvasImageSource {
  // Step 1: Allocate a compatible canvas for the requested tile dimensions.
  const canvas = createCompatibleCanvas(
    options.tileWidthPx,
    options.tileHeightPx,
  );

  // Step 2: Resolve a 2D context when the runtime can actually render.
  const tileContext = resolveStarTileContext(canvas);
  if (!tileContext) {
    return canvas;
  }

  // Step 3: Initialize drawing state and render deterministic stars.
  const seededRandom = createSeededRandom(options.seed);
  initializeStarTileContext({
    tileContext,
    canvas,
    blurPx: options.blurPx,
  });
  renderSeededStars({
    tileContext,
    seededRandom,
    canvasOptions: options,
  });

  // Step 4: Restore neutral drawing state before handing the canvas back.
  resetStarTileContext(tileContext);
  return canvas;
}

/**
 * Creates a browser-compatible canvas with clamped integer dimensions.
 *
 * @param widthPx - Requested tile width in pixels.
 * @param heightPx - Requested tile height in pixels.
 * @returns Offscreen canvas when supported, otherwise a DOM canvas fallback.
 */
function createCompatibleCanvas(
  widthPx: number,
  heightPx: number,
): HTMLCanvasElement | OffscreenCanvas {
  const canvasDimensions = normalizeCanvasDimensions(widthPx, heightPx);
  const offscreenCanvas = createOffscreenCanvasIfSupported(canvasDimensions);
  if (offscreenCanvas) {
    return offscreenCanvas;
  }

  const documentCanvas = createDocumentCanvasIfSupported(canvasDimensions);
  if (documentCanvas) {
    return documentCanvas;
  }

  return createCanvasSizeFallback(canvasDimensions);
}

/**
 * Resolves the rendering context used for star tile pre-rendering.
 *
 * @param canvas - Compatible canvas returned by the runtime-specific factory.
 * @returns A 2D drawing context when rendering is supported.
 */
function resolveStarTileContext(
  canvas: HTMLCanvasElement | OffscreenCanvas,
): OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null {
  return canvas.getContext(FLAPPY_STARFIELD_CANVAS_CONTEXT_ID);
}

/**
 * Clears the canvas and applies the glow settings shared by all rendered stars.
 *
 * @param options - Context initialization dependencies.
 * @returns Nothing. The provided context is mutated in place.
 */
function initializeStarTileContext(options: {
  tileContext: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
  canvas: HTMLCanvasElement | OffscreenCanvas;
  blurPx: number;
}): void {
  options.tileContext.clearRect(
    FLAPPY_STARFIELD_ORIGIN_PX,
    FLAPPY_STARFIELD_ORIGIN_PX,
    options.canvas.width,
    options.canvas.height,
  );
  options.tileContext.globalCompositeOperation =
    FLAPPY_STARFIELD_COMPOSITE_SOURCE_OVER;
  options.tileContext.shadowColor = FLAPPY_STARFIELD_CYAN_FILL_STYLE;
  options.tileContext.shadowBlur = options.blurPx;
}

/**
 * Draws all stars for one tile using a seeded random source.
 *
 * @param options - Drawing context, seed source, and tile recipe.
 * @returns Nothing. The provided context is mutated in place.
 */
function renderSeededStars(options: {
  tileContext: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
  seededRandom: () => number;
  canvasOptions: CreateStarTileCanvasOptions;
}): void {
  for (
    let starIndex = 0;
    starIndex < options.canvasOptions.starCount;
    starIndex += 1
  ) {
    const starPlacement = resolveStarPlacement({
      seededRandom: options.seededRandom,
      tileWidthPx: options.canvasOptions.tileWidthPx,
      tileHeightPx: options.canvasOptions.tileHeightPx,
      minSizePx: options.canvasOptions.minSizePx,
      maxSizePx: options.canvasOptions.maxSizePx,
      minAlpha: options.canvasOptions.minAlpha,
      maxAlpha: options.canvasOptions.maxAlpha,
    });

    options.tileContext.globalAlpha = starPlacement.alpha;
    options.tileContext.fillStyle = FLAPPY_STARFIELD_CYAN_FILL_STYLE;
    options.tileContext.fillRect(
      starPlacement.xPx,
      starPlacement.yPx,
      starPlacement.sizePx,
      starPlacement.sizePx,
    );
  }
}

/**
 * Restores neutral drawing state so later canvas consumers start from defaults.
 *
 * @param tileContext - 2D context used to render the star tile.
 * @returns Nothing. The provided context is mutated in place.
 */
function resetStarTileContext(
  tileContext: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
): void {
  tileContext.shadowBlur = FLAPPY_STARFIELD_NO_BLUR_PX;
  tileContext.shadowColor = FLAPPY_STARFIELD_TRANSPARENT_SHADOW_COLOR;
  tileContext.globalAlpha = FLAPPY_STARFIELD_FULL_ALPHA;
}

/**
 * Normalizes requested canvas dimensions into positive integer pixel sizes.
 *
 * @param widthPx - Requested width in pixels.
 * @param heightPx - Requested height in pixels.
 * @returns Clamped integer dimensions safe for canvas allocation.
 */
function normalizeCanvasDimensions(
  widthPx: number,
  heightPx: number,
): StarfieldCanvasDimensions {
  return {
    width: Math.max(FLAPPY_STARFIELD_MIN_DIMENSION_PX, Math.round(widthPx)),
    height: Math.max(FLAPPY_STARFIELD_MIN_DIMENSION_PX, Math.round(heightPx)),
  };
}

/**
 * Creates an offscreen canvas when the current runtime supports it.
 *
 * @param canvasDimensions - Already-normalized pixel dimensions.
 * @returns Offscreen canvas instance or `null` when unavailable.
 */
function createOffscreenCanvasIfSupported(
  canvasDimensions: StarfieldCanvasDimensions,
): OffscreenCanvas | null {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(canvasDimensions.width, canvasDimensions.height);
  }

  return null;
}

/**
 * Creates a DOM canvas when document APIs are available.
 *
 * @param canvasDimensions - Already-normalized pixel dimensions.
 * @returns DOM canvas instance or `null` when unavailable.
 */
function createDocumentCanvasIfSupported(
  canvasDimensions: StarfieldCanvasDimensions,
): HTMLCanvasElement | null {
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = canvasDimensions.width;
    canvas.height = canvasDimensions.height;
    return canvas;
  }

  return null;
}

/**
 * Creates a size-only fallback so non-browser tests can skip rendering safely.
 *
 * @param canvasDimensions - Already-normalized pixel dimensions.
 * @returns Minimal canvas-shaped object cast to the compatible return type.
 */
function createCanvasSizeFallback(
  canvasDimensions: StarfieldCanvasDimensions,
): HTMLCanvasElement {
  const canvas = {
    width: canvasDimensions.width,
    height: canvasDimensions.height,
  } as Partial<HTMLCanvasElement> as HTMLCanvasElement;
  return canvas;
}

/**
 * Resolves one deterministic star placement and appearance from the seeded RNG.
 *
 * @param options - Random source and star placement bounds.
 * @returns Pixel location, square size, and alpha for one rendered star.
 */
function resolveStarPlacement(options: {
  seededRandom: () => number;
  tileWidthPx: number;
  tileHeightPx: number;
  minSizePx: number;
  maxSizePx: number;
  minAlpha: number;
  maxAlpha: number;
}): StarPlacement {
  const xPx = Math.floor(options.seededRandom() * options.tileWidthPx);
  const yPx = Math.floor(options.seededRandom() * options.tileHeightPx);
  const sizePx =
    options.minSizePx +
    Math.floor(
      options.seededRandom() *
        (options.maxSizePx -
          options.minSizePx +
          FLAPPY_STARFIELD_INCLUSIVE_RANGE_OFFSET),
    );
  const alpha =
    options.minAlpha +
    options.seededRandom() * (options.maxAlpha - options.minAlpha);

  return {
    xPx,
    yPx,
    sizePx,
    alpha,
  };
}
