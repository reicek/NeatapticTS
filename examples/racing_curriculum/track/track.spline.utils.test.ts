import { generateTrack } from './track.generator';
import {
  buildTrackSplineSamples,
  resolveSplineSampleFrame,
  TRACK_SPLINE_SAMPLES_PER_SEGMENT,
} from './track.spline.utils';

describe('track.spline.utils', () => {
  describe('buildTrackSplineSamples', () => {
    it('emits one stable sample strip per segment with ordered metadata', () => {
      const trackSpec = generateTrack({
        seed: 42,
        layoutVersion: 1,
        sizeBucket: 'medium',
      });
      const splineSamples = buildTrackSplineSamples(trackSpec.segments);

      expect({
        sampleCount: splineSamples.length,
        firstSampleSegmentIndex: splineSamples[0]?.segmentIndex,
        firstSampleIndexWithinSegment:
          splineSamples[0]?.sampleIndexWithinSegment,
        lastSampleSegmentIndex: splineSamples.at(-1)?.segmentIndex,
        lastSampleIndexWithinSegment:
          splineSamples.at(-1)?.sampleIndexWithinSegment,
      }).toEqual({
        sampleCount:
          trackSpec.segments.length * TRACK_SPLINE_SAMPLES_PER_SEGMENT,
        firstSampleSegmentIndex: 0,
        firstSampleIndexWithinSegment: 0,
        lastSampleSegmentIndex: trackSpec.segments.length - 1,
        lastSampleIndexWithinSegment: TRACK_SPLINE_SAMPLES_PER_SEGMENT - 1,
      });
    });
  });

  describe('resolveSplineSampleFrame', () => {
    it('points the left normal toward the interior of a counter-clockwise generated track', () => {
      const trackSpec = generateTrack({
        seed: 42,
        layoutVersion: 1,
        sizeBucket: 'medium',
      });
      const samples = trackSpec.splineSamples;
      const centroid = samples.reduce(
        (sum, sample) => ({
          x: sum.x + sample.x,
          y: sum.y + sample.y,
        }),
        { x: 0, y: 0 },
      );
      centroid.x /= samples.length;
      centroid.y /= samples.length;

      const inwardChecks = samples.map((sample) => {
        const frame = resolveSplineSampleFrame(samples, sample.globalIndex);
        const toCenterX = centroid.x - sample.x;
        const toCenterY = centroid.y - sample.y;

        return toCenterX * frame.normalX + toCenterY * frame.normalY > 0;
      });
      const allNormalsPointInward = inwardChecks.every(Boolean);

      expect({ allNormalsPointInward }).toEqual({
        allNormalsPointInward: true,
      });
    });
  });
});
