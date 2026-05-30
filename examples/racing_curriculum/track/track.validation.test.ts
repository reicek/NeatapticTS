import type { TrackSpec } from './track.generator.types';
import { validateTrackSpec } from './track.validation';

describe('track validation sibling seam', () => {
  describe('validateTrackSpec', () => {
    it('accepts a simple closed square loop with positive widths', () => {
      const trackSpec: TrackSpec = {
        seed: 1,
        layoutVersion: 1,
        sizeBucket: 'unit-test',
        segments: [
          { startX: 0, startY: 0, endX: 10, endY: 0, width: 20 },
          { startX: 10, startY: 0, endX: 10, endY: 10, width: 20 },
          { startX: 10, startY: 10, endX: 0, endY: 10, width: 20 },
          { startX: 0, startY: 10, endX: 0, endY: 0, width: 20 },
        ],
        splineSamples: [],
      };

      expect(validateTrackSpec(trackSpec)).toBe(true);
    });
  });
});