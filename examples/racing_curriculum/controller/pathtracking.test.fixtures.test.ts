import { createCurvedTrackSpec } from './pathtracking.test.fixtures';

describe('pathtracking test fixtures', () => {
  describe('createCurvedTrackSpec', () => {
    it('precomputes a denser spline sample strip than the raw chord list', () => {
      const trackSpec = createCurvedTrackSpec();

      expect(trackSpec.splineSamples.length > trackSpec.segments.length).toBe(
        true,
      );
    });
  });
});
