import {
  FLAPPY_STARFIELD_LAYER_SPECS,
  FLAPPY_STARFIELD_MIN_DIMENSION_PX,
} from '../../constants/constants';
import { cachedStarfieldTilesByHeight } from './playback.constants';
import { createStarTile } from './playback.starfield.layer.services';
import type { StarTile } from './playback.starfield.types';

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
