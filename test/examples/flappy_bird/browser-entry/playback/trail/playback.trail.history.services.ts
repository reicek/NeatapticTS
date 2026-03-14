import {
  FLAPPY_CHAMPION_TRAIL_MAX_POINTS,
  FLAPPY_TRAIL_MAX_POINTS,
} from '../../../constants/constants';
import type { TrailPoint } from '../../browser-entry.types';

/**
 * Trail-history helpers for the playback renderer.
 *
 * This module owns the small rolling histories that create the flock's neon
 * afterimage. The policy is intentionally simple: keep only a short recent
 * window, and keep the champion trail even shorter and denser.
 *
 * Minimal usage sketch:
 * ```ts
 * const trailPoints = [{ frameIndex: 10, yPx: 140 }];
 * pushTrailPoint(trailPoints, 11, 136, 2);
 * ```
 */

/**
 * Appends one trail point while enforcing the maximum retained history length.
 *
 * Playback trails are intentionally modeled as short rolling histories rather
 * than unbounded path logs. That keeps the neon afterimage readable, prevents
 * old turns from dominating the current frame, and avoids per-frame growth in a
 * long-running browser session.
 *
 * @param trailPoints - Mutable trail collection.
 * @param frameIndex - Source frame index.
 * @param yPosition - Bird y position.
 * @param maxRetainedPoints - Optional maximum retained trail history length.
 * @returns Nothing.
 * @example
 * ```ts
 * const trailPoints = [{ frameIndex: 10, yPx: 140 }];
 * pushTrailPoint(trailPoints, 11, 136, 2);
 * ```
 */
export function pushTrailPoint(
  trailPoints: TrailPoint[],
  frameIndex: number,
  yPosition: number,
  maxRetainedPoints: number = FLAPPY_TRAIL_MAX_POINTS,
): void {
  trailPoints.push({ frameIndex, yPx: yPosition });
  if (trailPoints.length > maxRetainedPoints) {
    trailPoints.splice(0, trailPoints.length - maxRetainedPoints);
  }
}

/**
 * Appends one point to the champion-only short trail history.
 *
 * The browser highlights the current leader with a shorter, denser trail than
 * the rest of the flock. Using a dedicated helper keeps that policy explicit in
 * the call site instead of scattering champion-specific retention numbers
 * through the playback renderer.
 *
 * @param trailPoints - Mutable champion trail collection.
 * @param frameIndex - Source frame index.
 * @param yPosition - Bird y position.
 * @returns Nothing.
 */
export function pushChampionTrailPoint(
  trailPoints: TrailPoint[],
  frameIndex: number,
  yPosition: number,
): void {
  pushTrailPoint(
    trailPoints,
    frameIndex,
    yPosition,
    FLAPPY_CHAMPION_TRAIL_MAX_POINTS,
  );
}
