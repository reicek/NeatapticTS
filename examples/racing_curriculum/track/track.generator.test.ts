import { generateTrack, freezeTrackSpec } from './track.generator';
import { validateTrackSpec } from './track.validation';
import type { TrackSegment } from './track.generator.types';

/**
 * Red-phase contracts for the deterministic procedural track generator.
 *
 * These tests target the Tier-0 Visual Driving Harness acceptance criterion:
 * `same seed + layoutVersion + sizeBucket → byte-identical TrackSpec` with all
 * geometry invariants satisfied.
 *
 * All tests in this file intentionally fail until `track.generator.ts` and
 * `track.validation.ts` are implemented.
 */
describe('track.generator', () => {
  describe('generateTrack', () => {
    describe('non-trivial output', () => {
      it('produces at least 3 closed-loop segments for seed 42 size-bucket medium', () => {
        // Arrange
        const input = { seed: 42, layoutVersion: 1, sizeBucket: 'medium' };

        // Act
        const spec = generateTrack(input);

        // Assert — stub returns segments:[], so 0 is NOT > 2 → red
        expect(spec.segments.length).toBeGreaterThan(2);
      });

      it('returns byte-identical non-empty segment geometry across two runs with the same input', () => {
        // Arrange
        const input = { seed: 42, layoutVersion: 1, sizeBucket: 'medium' };

        // Act
        const firstSpec = generateTrack(input);
        const secondSpec = generateTrack(input);
        const firstGeometry = JSON.stringify(firstSpec.segments);
        const secondGeometry = JSON.stringify(secondSpec.segments);

        // Assert — stub returns empty geometry for both runs, so the equality holds
        // but the non-empty determinism contract still fails honestly → red
        expect({
          hasSegments: firstSpec.segments.length > 0,
          firstGeometry,
          secondGeometry,
        }).toEqual({
          hasSegments: true,
          firstGeometry,
          secondGeometry: firstGeometry,
        });
      });
    });

    describe('closed-loop invariant', () => {
      it('has last segment endX matching first segment startX for seed 42', () => {
        // Arrange
        const input = { seed: 42, layoutVersion: 1, sizeBucket: 'medium' };

        // Act
        const spec = generateTrack(input);
        const firstSegment = spec.segments[0] ?? { startX: Infinity };
        const lastSegment = spec.segments.at(-1) ?? { endX: NaN };

        // Assert — stub returns empty segments, NaN is not close to Infinity → red
        expect(lastSegment.endX).toBeCloseTo(firstSegment.startX, 2);
      });

      it('has last segment endY matching first segment startY for seed 42', () => {
        // Arrange
        const input = { seed: 42, layoutVersion: 1, sizeBucket: 'medium' };

        // Act
        const spec = generateTrack(input);
        const firstSegment = spec.segments[0] ?? { startY: Infinity };
        const lastSegment = spec.segments.at(-1) ?? { endY: NaN };

        // Assert — stub returns empty segments, NaN is not close to Infinity → red
        expect(lastSegment.endY).toBeCloseTo(firstSegment.startY, 2);
      });
    });

    describe('minimum width constraint', () => {
      it('all segments have positive width AND the track is non-empty for seed 42', () => {
        // Arrange
        const input = { seed: 42, layoutVersion: 1, sizeBucket: 'medium' };

        // Act
        const spec = generateTrack(input);
        const hasSegments = spec.segments.length > 0;
        const allPositiveWidth = spec.segments.every(
          (segment) => segment.width > 0,
        );

        // Assert — stub returns empty, hasSegments is false → red
        expect(hasSegments && allPositiveWidth).toBe(true);
      });
    });
  });

  describe('freezeTrackSpec', () => {
    it('throws TypeError when segments array is mutated after freeze', () => {
      // Arrange
      const spec = generateTrack({
        seed: 1,
        layoutVersion: 1,
        sizeBucket: 'medium',
      });
      const frozenSpec = freezeTrackSpec(spec);
      const dummySegment: TrackSegment = {
        startX: 0,
        startY: 0,
        endX: 10,
        endY: 0,
        width: 20,
      };

      // Act + Assert — stub returns unfrozen spec, push succeeds → expected throw never arrives → red
      expect(() => {
        (frozenSpec.segments as TrackSegment[]).push(dummySegment);
      }).toThrow(TypeError);
    });
  });

  describe('validateTrackSpec', () => {
    it('throws RangeError when validating a spec with zero segments', () => {
      // Arrange — an empty spec that must be rejected by the validator
      const emptySpec = {
        seed: 0,
        layoutVersion: 1,
        sizeBucket: 'medium',
        segments: [],
        splineSamples: [],
      };

      // Act + Assert — stub returns true without checking → expected throw never arrives → red
      expect(() => validateTrackSpec(emptySpec)).toThrow(RangeError);
    });

    it('throws RangeError when validating a self-intersecting bow-tie layout', () => {
      // Arrange — this four-segment loop crosses itself in the middle and must
      // be rejected by the validator
      const selfIntersectingSpec = {
        seed: 99,
        layoutVersion: 1,
        sizeBucket: 'medium',
        segments: [
          { startX: 0, startY: 0, endX: 10, endY: 10, width: 20 },
          { startX: 10, startY: 10, endX: 0, endY: 10, width: 20 },
          { startX: 0, startY: 10, endX: 10, endY: 0, width: 20 },
          { startX: 10, startY: 0, endX: 0, endY: 0, width: 20 },
        ],
        splineSamples: [],
      };

      // Act + Assert — stub returns true without checking crossings → red
      expect(() => validateTrackSpec(selfIntersectingSpec)).toThrow(RangeError);
    });
  });
});
