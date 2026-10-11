/**
 * Fixture track for the autopilot green tests: one straight closed loop
 * (N by the sample stride) of lane-center samples along +x at y = 0. The
 * closed loop is formed by the wrap from the last sample back to the
 * first, matching the frozen `SplineSample` order contract.
 *
 * @module
 */

import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';

/** Number of fixture samples along the straight loop. */
const FIXTURE_SAMPLE_COUNT = 40;

/** Fixture loop stride (m) between consecutive samples along +x. */
const FIXTURE_SAMPLE_STRIDE_M = 1;

/** Fixture lane-center width (m) shared by every sample. */
const FIXTURE_WIDTH_M = 12;

/**
 * Builds the straight-loop spline fixture: 40 samples, x = 0..39 at
 * y = 0, `globalIndex` equal to the array index.
 *
 * @returns The frozen sample list for direct `stepAutopilotInput` calls.
 */
export function createTrackSpecForAutopilotFixture(): readonly SplineSample[] {
  const samples: SplineSample[] = [];
  for (let index = 0; index < FIXTURE_SAMPLE_COUNT; index += 1) {
    samples.push({
      x: index * FIXTURE_SAMPLE_STRIDE_M,
      y: 0,
      width: FIXTURE_WIDTH_M,
      segmentIndex: 0,
      sampleIndexWithinSegment: index,
      globalIndex: index,
    });
  }
  return samples;
}
