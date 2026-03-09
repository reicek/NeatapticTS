import { FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO } from './playback.background.ground-grid.constants';
import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridHorizontalGeometry,
  PlaybackGroundGridHorizontalGeometryFactory,
  PlaybackGroundGridVerticalGeometry,
  PlaybackGroundGridVerticalGeometryFactory,
} from './playback.background.ground-grid.types';

let cachedViewportSizeKey: string | null = null;
let cachedFogGradientsByCanvas = new WeakMap<
  HTMLCanvasElement,
  Map<string, CanvasGradient>
>();

const cachedHorizontalGeometryBySceneKey = new Map<
  string,
  PlaybackGroundGridHorizontalGeometry
>();
const cachedVerticalGeometryByCycleKey = new Map<
  string,
  PlaybackGroundGridVerticalGeometry
>();

/**
 * Ensures ground-grid caches only retain entries for the current viewport size.
 *
 * The ground grid is derived from viewport width and total scene height, so a
 * page resize invalidates every cached geometry variant and fog gradient.
 *
 * @param sceneContext - Current lower-band scene geometry.
 * @returns Stable viewport-size cache key for the current frame.
 */
export function ensureGroundGridViewportCacheValidity(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): string {
  const viewportSizeKey = resolveGroundGridViewportCacheKey(sceneContext);
  if (cachedViewportSizeKey === viewportSizeKey) {
    return viewportSizeKey;
  }

  cachedViewportSizeKey = viewportSizeKey;
  cachedHorizontalGeometryBySceneKey.clear();
  cachedVerticalGeometryByCycleKey.clear();
  cachedFogGradientsByCanvas = new WeakMap<
    HTMLCanvasElement,
    Map<string, CanvasGradient>
  >();
  return viewportSizeKey;
}

/**
 * Resolves the viewport-size cache key used by the ground-grid caches.
 *
 * @param sceneContext - Current lower-band scene geometry.
 * @returns Cache key that changes whenever the page size changes.
 */
export function resolveGroundGridViewportCacheKey(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): string {
  return `${sceneContext.visibleWorldWidthPx}x${sceneContext.lowerBandBottomYPx}`;
}

/**
 * Resolves the stable local-scene cache key for ground-grid geometry.
 *
 * @param sceneContext - Current lower-band scene geometry.
 * @returns Scene key suitable for static horizontal and vertical cache entries.
 */
export function resolveGroundGridSceneCacheKey(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): string {
  return [
    sceneContext.visibleWorldWidthPx,
    sceneContext.alignedHorizonYPx,
    sceneContext.lowerBandTopYPx,
    sceneContext.lowerBandHeightPx,
    sceneContext.lowerBandBottomYPx,
    sceneContext.vanishingPointXPx,
    sceneContext.vanishingPointYPx,
  ].join(':');
}

/**
 * Resolves the cache key for one wrapped vertical-geometry cycle.
 *
 * @param sceneCacheKey - Stable scene key for the active viewport.
 * @param wrappedOffsetPx - Quantized wrapped offset within one lane cycle.
 * @returns Cycle key used for vertical geometry reuse.
 */
export function resolveGroundGridVerticalCycleCacheKey(
  sceneCacheKey: string,
  wrappedOffsetPx: number,
): string {
  return `${sceneCacheKey}|offset:${wrappedOffsetPx}`;
}

/**
 * Resolves cached horizontal geometry for one scene.
 *
 * @param sceneCacheKey - Stable scene key for the active viewport.
 * @param factory - Lazy geometry builder used when the cache misses.
 * @returns Cached horizontal geometry bundle for the scene.
 */
export function resolveCachedGroundGridHorizontalGeometry(
  sceneCacheKey: string,
  factory: PlaybackGroundGridHorizontalGeometryFactory,
): PlaybackGroundGridHorizontalGeometry {
  const cachedGeometry = cachedHorizontalGeometryBySceneKey.get(sceneCacheKey);
  if (cachedGeometry) {
    return cachedGeometry;
  }

  const resolvedGeometry = factory();
  cachedHorizontalGeometryBySceneKey.set(sceneCacheKey, resolvedGeometry);
  return resolvedGeometry;
}

/**
 * Resolves cached vertical geometry for one scene and wrapped offset cycle.
 *
 * @param cycleCacheKey - Scene-and-offset cache key for the active frame.
 * @param factory - Lazy geometry builder used when the cache misses.
 * @returns Cached vertical geometry bundle for the cycle.
 */
export function resolveCachedGroundGridVerticalGeometry(
  cycleCacheKey: string,
  factory: PlaybackGroundGridVerticalGeometryFactory,
): PlaybackGroundGridVerticalGeometry {
  const cachedGeometry = cachedVerticalGeometryByCycleKey.get(cycleCacheKey);
  if (cachedGeometry) {
    return cachedGeometry;
  }

  const resolvedGeometry = factory();
  cachedVerticalGeometryByCycleKey.set(cycleCacheKey, resolvedGeometry);
  return resolvedGeometry;
}

/**
 * Resolves a cached fog gradient for one canvas and local scene.
 *
 * @param context - Canvas 2D drawing context.
 * @param sceneCacheKey - Stable scene key for the active viewport.
 * @param sceneContext - Current lower-band scene geometry.
 * @param fogColor - Theme-owned fog color token.
 * @returns Cached fog gradient aligned to the lower-band scene.
 */
export function resolveCachedGroundGridFogGradient(
  context: CanvasRenderingContext2D,
  sceneCacheKey: string,
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  fogColor: string,
): CanvasGradient {
  const fogGradientKey = `${sceneCacheKey}|fog:${fogColor}`;
  const canvas = context.canvas;
  const cachedGradientsForCanvas =
    cachedFogGradientsByCanvas.get(canvas) ?? new Map<string, CanvasGradient>();
  const cachedGradient = cachedGradientsForCanvas.get(fogGradientKey);
  if (cachedGradient) {
    return cachedGradient;
  }

  const fogHeightPx =
    sceneContext.lowerBandHeightPx * FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO;
  const fogGradient = context.createLinearGradient(
    0,
    sceneContext.lowerBandTopYPx,
    0,
    sceneContext.lowerBandTopYPx + fogHeightPx,
  );
  fogGradient.addColorStop(0, fogColor);
  fogGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

  cachedGradientsForCanvas.set(fogGradientKey, fogGradient);
  cachedFogGradientsByCanvas.set(canvas, cachedGradientsForCanvas);
  return fogGradient;
}
