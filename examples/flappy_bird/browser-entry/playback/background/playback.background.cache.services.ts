import type {
  PlaybackBackgroundLayout,
  PlaybackBackgroundLayoutFactory,
} from './playback.background.types';

let cachedViewportSizeKey: string | null = null;

const cachedLayoutsByHeightPx = new Map<number, PlaybackBackgroundLayout>();
const cachedTileCoverageCountsByTileWidthPx = new Map<number, number>();

/**
 * Ensures background caches only retain entries for the current viewport size.
 *
 * When the page size changes, cached geometry and coverage counts become
 * obsolete because the background bands and tile coverage both depend on the
 * current viewport dimensions.
 *
 * @param visibleWorldWidthPx - Current visible world width in pixels.
 * @param visibleWorldHeightPx - Current visible world height in pixels.
 * @returns Stable viewport-size cache key for the current frame.
 */
export function ensurePlaybackBackgroundViewportCacheValidity(
  visibleWorldWidthPx: number,
  visibleWorldHeightPx: number,
): string {
  const viewportSizeKey = resolvePlaybackBackgroundViewportCacheKey(
    visibleWorldWidthPx,
    visibleWorldHeightPx,
  );

  if (cachedViewportSizeKey === viewportSizeKey) {
    return viewportSizeKey;
  }

  cachedViewportSizeKey = viewportSizeKey;
  cachedLayoutsByHeightPx.clear();
  cachedTileCoverageCountsByTileWidthPx.clear();
  return viewportSizeKey;
}

/**
 * Resolves the stable viewport-size cache key used by background caches.
 *
 * @param visibleWorldWidthPx - Current visible world width in pixels.
 * @param visibleWorldHeightPx - Current visible world height in pixels.
 * @returns Cache key that changes whenever the page size changes.
 */
export function resolvePlaybackBackgroundViewportCacheKey(
  visibleWorldWidthPx: number,
  visibleWorldHeightPx: number,
): string {
  return `${visibleWorldWidthPx}x${visibleWorldHeightPx}`;
}

/**
 * Resolves cached background layout for the current viewport height.
 *
 * @param visibleWorldHeightPx - Current visible world height in pixels.
 * @param factory - Lazy layout builder used when the cache misses.
 * @returns Cached background layout for the current viewport size.
 */
export function resolveCachedPlaybackBackgroundLayout(
  visibleWorldHeightPx: number,
  factory: PlaybackBackgroundLayoutFactory,
): PlaybackBackgroundLayout {
  const cachedLayout = cachedLayoutsByHeightPx.get(visibleWorldHeightPx);
  if (cachedLayout) {
    return cachedLayout;
  }

  const resolvedLayout = factory();
  cachedLayoutsByHeightPx.set(visibleWorldHeightPx, resolvedLayout);
  return resolvedLayout;
}

/**
 * Resolves cached tile coverage count for one tile width.
 *
 * @param tileWidthPx - Width of one repeated starfield tile in pixels.
 * @param factory - Lazy coverage builder used when the cache misses.
 * @returns Cached tile coverage count for the active viewport width.
 */
export function resolveCachedPlaybackTileCoverageCount(
  tileWidthPx: number,
  factory: () => number,
): number {
  const cachedCoverageCount =
    cachedTileCoverageCountsByTileWidthPx.get(tileWidthPx);
  if (cachedCoverageCount !== undefined) {
    return cachedCoverageCount;
  }

  const resolvedCoverageCount = factory();
  cachedTileCoverageCountsByTileWidthPx.set(tileWidthPx, resolvedCoverageCount);
  return resolvedCoverageCount;
}
