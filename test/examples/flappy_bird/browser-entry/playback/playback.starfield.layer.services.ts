import {
  FLAPPY_STARFIELD_TILE_WIDTH_PX,
} from '../../constants/constants';
import { createStarTileCanvas } from './playback.starfield.services';
import type {
  PlaybackStarfieldLayerSpec,
  StarTile,
} from './playback.starfield.types';

/**
 * Creates one cached tile layer from a declarative layer specification.
 *
 * @param layerSpec - Density and motion contract for a starfield layer.
 * @param tileHeightPx - Height of the visible sky band in pixels.
 * @returns Cached tile metadata for parallax drawing.
 */
export function createStarTile(
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