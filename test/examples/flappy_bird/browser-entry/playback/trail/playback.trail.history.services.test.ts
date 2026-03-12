import type { TrailPoint } from '../../browser-entry.types';
import {
  pushChampionTrailPoint,
  pushTrailPoint,
} from './playback.trail.history.services';

describe('pushTrailPoint', () => {
  it('trims the oldest points when the trail exceeds the retained maximum', () => {
    const trailPoints = createTrailPoints();

    pushTrailPoint(trailPoints, 3, 30, 2);

    expect(trailPoints).toEqual([
      { frameIndex: 2, yPx: 20 },
      { frameIndex: 3, yPx: 30 },
    ]);
  });
});

describe('pushChampionTrailPoint', () => {
  it('appends the latest champion point to the trail history', () => {
    const trailPoints: TrailPoint[] = [];

    pushChampionTrailPoint(trailPoints, 7, 42);

    expect(trailPoints.at(-1)).toEqual({ frameIndex: 7, yPx: 42 });
  });
});

function createTrailPoints(): TrailPoint[] {
  return [
    { frameIndex: 1, yPx: 10 },
    { frameIndex: 2, yPx: 20 },
  ];
}