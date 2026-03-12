import {
  FLAPPY_CHAMPION_TRAIL_MAX_POINTS,
  FLAPPY_TRAIL_MAX_POINTS,
} from '../../../constants/constants';
import type { TrailPoint } from '../../browser-entry.types';

/**
 * Appends one trail point while enforcing the maximum retained history length.
 *
 * @param trailPoints - Mutable trail collection.
 * @param frameIndex - Source frame index.
 * @param yPosition - Bird y position.
 * @param maxRetainedPoints - Optional maximum retained trail history length.
 * @returns Nothing.
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