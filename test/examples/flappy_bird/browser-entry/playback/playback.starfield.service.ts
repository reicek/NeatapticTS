import {
  FLAPPY_STARFIELD_CYAN_FILL_STYLE,
  FLAPPY_STARFIELD_FAR_SCROLL_RATIO,
  FLAPPY_STARFIELD_MID_SCROLL_RATIO,
  FLAPPY_STARFIELD_NEAR_SCROLL_RATIO,
  FLAPPY_STARFIELD_TILE_WIDTH_PX,
} from '../../constants/constants';
import { cachedStarfieldTilesByHeight } from './playback.constants';
import { createSeededRandom } from './playback.starfield.utils';
import type { StarTile } from './playback.types';

/**
 * Resolves (and lazily creates) cached starfield tile layers for the viewport.
 *
 * @param visibleWorldHeightPx - Viewport height in world pixels.
 * @returns Ordered far/mid/near starfield tiles.
 */
export function resolveStarfieldTiles(
  visibleWorldHeightPx: number,
): readonly StarTile[] {
  const tileHeightPx = Math.max(1, Math.round(visibleWorldHeightPx));
  const cachedTilesForHeight = cachedStarfieldTilesByHeight.get(tileHeightPx);
  if (cachedTilesForHeight) {
    return cachedTilesForHeight;
  }

  const farTile: StarTile = {
    image: createStarTileCanvas({
      seed: 1_337,
      tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
      tileHeightPx,
      starCount: 35,
      minSizePx: 1,
      maxSizePx: 2,
      minAlpha: 0.08,
      maxAlpha: 0.22,
      blurPx: 4,
    }),
    tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
    tileHeightPx,
    scrollRatio: FLAPPY_STARFIELD_FAR_SCROLL_RATIO,
  };
  const midTile: StarTile = {
    image: createStarTileCanvas({
      seed: 2_777,
      tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
      tileHeightPx,
      starCount: 28,
      minSizePx: 1,
      maxSizePx: 3,
      minAlpha: 0.1,
      maxAlpha: 0.28,
      blurPx: 6,
    }),
    tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
    tileHeightPx,
    scrollRatio: FLAPPY_STARFIELD_MID_SCROLL_RATIO,
  };
  const nearTile: StarTile = {
    image: createStarTileCanvas({
      seed: 4_242,
      tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
      tileHeightPx,
      starCount: 23,
      minSizePx: 2,
      maxSizePx: 4,
      minAlpha: 0.12,
      maxAlpha: 0.34,
      blurPx: 8,
    }),
    tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
    tileHeightPx,
    scrollRatio: FLAPPY_STARFIELD_NEAR_SCROLL_RATIO,
  };

  const resolvedTiles = [farTile, midTile, nearTile] as const;
  cachedStarfieldTilesByHeight.set(tileHeightPx, resolvedTiles);
  return resolvedTiles;
}

function createStarTileCanvas(options: {
  seed: number;
  tileWidthPx: number;
  tileHeightPx: number;
  starCount: number;
  minSizePx: number;
  maxSizePx: number;
  minAlpha: number;
  maxAlpha: number;
  blurPx: number;
}): CanvasImageSource {
  const canvas = createCompatibleCanvas(
    options.tileWidthPx,
    options.tileHeightPx,
  );
  const tileContext = canvas.getContext('2d');
  if (!tileContext) {
    return canvas;
  }

  const seededRandom = createSeededRandom(options.seed);

  tileContext.clearRect(0, 0, canvas.width, canvas.height);
  tileContext.globalCompositeOperation = 'source-over';
  tileContext.shadowColor = FLAPPY_STARFIELD_CYAN_FILL_STYLE;
  tileContext.shadowBlur = options.blurPx;

  for (let starIndex = 0; starIndex < options.starCount; starIndex += 1) {
    const xPx = Math.floor(seededRandom() * options.tileWidthPx);
    const yPx = Math.floor(seededRandom() * options.tileHeightPx);
    const sizePx =
      options.minSizePx +
      Math.floor(seededRandom() * (options.maxSizePx - options.minSizePx + 1));
    const alpha =
      options.minAlpha + seededRandom() * (options.maxAlpha - options.minAlpha);

    tileContext.globalAlpha = alpha;
    tileContext.fillStyle = FLAPPY_STARFIELD_CYAN_FILL_STYLE;
    tileContext.fillRect(xPx, yPx, sizePx, sizePx);
  }

  tileContext.shadowBlur = 0;
  tileContext.shadowColor = 'transparent';
  tileContext.globalAlpha = 1;
  return canvas;
}

function createCompatibleCanvas(
  widthPx: number,
  heightPx: number,
): HTMLCanvasElement | OffscreenCanvas {
  const width = Math.max(1, Math.round(widthPx));
  const height = Math.max(1, Math.round(heightPx));

  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height);
  }

  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }

  // Fallback: non-browser environments won't render, but should not crash.
  const canvas = { width, height } as unknown as HTMLCanvasElement;
  return canvas;
}
