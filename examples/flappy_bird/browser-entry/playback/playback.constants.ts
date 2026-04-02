import type { StarTile } from './playback.starfield.types';

/**
 * Shared in-memory cache for pre-rendered parallax starfield tiles.
 *
 * This cache is keyed by world height to avoid re-rendering identical
 * offscreen tile strips across playback frames.
 */
export const cachedStarfieldTilesByHeight = new Map<
  number,
  readonly StarTile[]
>();
