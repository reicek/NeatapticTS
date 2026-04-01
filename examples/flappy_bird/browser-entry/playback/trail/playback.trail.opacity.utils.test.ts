import type { PlaybackEdgeBounds } from '../playback.types';
import {
  clamp01,
  resolveEdgeOpacityFactor,
  resolveTrailLifetimeOpacityFactor,
} from './playback.trail.opacity.utils';

describe('resolveEdgeOpacityFactor', () => {
  it('returns full opacity when the point is safely inside the fade band', () => {
    expect(resolveEdgeOpacityFactor(100, 100, createEdgeBounds())).toBe(1);
  });

  it('returns zero opacity when the point is on the edge', () => {
    expect(resolveEdgeOpacityFactor(0, 100, createEdgeBounds())).toBe(0);
  });
});

describe('resolveTrailLifetimeOpacityFactor', () => {
  it('returns half opacity for a midpoint trail age', () => {
    expect(resolveTrailLifetimeOpacityFactor(5, 10)).toBe(0.5);
  });
});

describe('clamp01', () => {
  it('caps values above one to the upper bound', () => {
    expect(clamp01(2)).toBe(1);
  });
});

function createEdgeBounds(): PlaybackEdgeBounds {
  return {
    leftXPx: 0,
    rightXPx: 200,
    topYPx: 0,
    bottomYPx: 200,
  };
}
