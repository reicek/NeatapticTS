import { generateTrack } from './track.generator';
import {
  buildTrackSplineSamples,
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
        firstSampleIndexWithinSegment: splineSamples[0]?.sampleIndexWithinSegment,
        lastSampleSegmentIndex: splineSamples.at(-1)?.segmentIndex,
        lastSampleIndexWithinSegment: splineSamples.at(-1)?.sampleIndexWithinSegment,
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
});