import { computeWorldTransform } from './racing.renderer';
import { generateTrack } from '../track/track.generator';

describe('racing renderer sibling seam', () => {
  describe('computeWorldTransform', () => {
    it('returns a positive scale for a generated track', () => {
      const canvasElement = {
        width: 960,
        height: 540,
      } as HTMLCanvasElement;
      const trackSpec = generateTrack({
        seed: 42,
        layoutVersion: 1,
        sizeBucket: 'medium',
      });
      const worldTransform = computeWorldTransform(canvasElement, trackSpec);

      expect(worldTransform.scale).toBeGreaterThan(0);
    });
  });
});
