import {
  FLAPPY_STARFIELD_LAYER_SPECS,
  FLAPPY_STARFIELD_MIN_DIMENSION_PX,
  FLAPPY_STARFIELD_TILE_WIDTH_PX,
} from '../../constants/constants';
import { cachedStarfieldTilesByHeight } from './playback.constants';
import { createStarTileCanvas } from './playback.starfield.services';
import type {
  PlaybackStarfieldLayerSpec,
  StarTile,
} from './playback.starfield.types';

/**
 * Resolves (and lazily creates) cached starfield tile layers for the viewport.
 *
 * @param visibleWorldHeightPx - Viewport height in world pixels.
 * @returns Ordered far/mid/near starfield tiles.
 */
export function resolveStarfieldTiles(
  visibleWorldHeightPx: number,
): readonly StarTile[] {
  const tileHeightPx = Math.max(
    FLAPPY_STARFIELD_MIN_DIMENSION_PX,
    Math.round(visibleWorldHeightPx),
  );
  const cachedTilesForHeight = cachedStarfieldTilesByHeight.get(tileHeightPx);
  if (cachedTilesForHeight) {
    return cachedTilesForHeight;
  }

  const resolvedTiles = FLAPPY_STARFIELD_LAYER_SPECS.map((layerSpec) =>
    createStarTile(layerSpec, tileHeightPx),
  );
  cachedStarfieldTilesByHeight.set(tileHeightPx, resolvedTiles);
  return resolvedTiles;
}

/**
 * Creates one cached tile layer from a declarative layer specification.
 *
 * @param layerSpec - Density and motion contract for a starfield layer.
 * @param tileHeightPx - Height of the visible sky band in pixels.
 * @returns Cached tile metadata for parallax drawing.
 */
function createStarTile(
  layerSpec: PlaybackStarfieldLayerSpec,
  tileHeightPx: number,
): StarTile {
  // Step 1: Pre-render one deterministic tile for the requested layer.
  return {
    image: createStarTileCanvas({
      seed: layerSpec.seed,
      tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
      tileHeightPx,
      starCount: layerSpec.starCount,
      minSizePx: layerSpec.minSizePx,
      maxSizePx: layerSpec.maxSizePx,
      minAlpha: layerSpec.minAlpha,
      maxAlpha: layerSpec.maxAlpha,
      blurPx: layerSpec.blurPx,
    }),
    tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
    tileHeightPx,
    scrollRatio: layerSpec.scrollRatio,
  };
}
