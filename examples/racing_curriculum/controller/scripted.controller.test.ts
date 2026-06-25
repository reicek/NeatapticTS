import { computeScriptedControl } from './scripted.controller';
import {
  buildTrackSplineSamples,
  resolveSplineSampleFrame,
} from '../track/track.spline.utils';
import type { TrackSpec } from '../track/track.generator.types';
import {
  createCurvedTrackSpec,
  createEnvironmentState,
  selectControllerFocalSample,
} from './pathtracking.test.fixtures';

describe('scripted.controller', () => {
  describe('computeScriptedControl', () => {
    it('keeps steer near the lane tangent instead of pulling toward a raw chord endpoint on curved samples', () => {
      const trackSpec = createCurvedTrackSpec();
      const { focalSample, focalFrame } =
        selectControllerFocalSample(trackSpec);
      const laneCount = 2;
      const laneWidthWorld = focalSample.width / laneCount;
      const innerOffsetWorld = focalSample.width / 2 - laneWidthWorld / 2;
      const envState = createEnvironmentState({
        carX: focalSample.x + focalFrame.normalX * innerOffsetWorld,
        carY: focalSample.y + focalFrame.normalY * innerOffsetWorld,
        carHeading: focalFrame.tangentHeadingRadians,
      });
      const controllerState = { targetSegmentIndex: focalSample.segmentIndex };

      const controlOutput = computeScriptedControl(
        envState,
        trackSpec,
        controllerState,
      );

      expect({
        keepsSteerNearLaneTangent: Math.abs(controlOutput.steer) < 0.1,
      }).toEqual({
        keepsSteerNearLaneTangent: true,
      });
    });

    it('steers near zero when the car is already on the inner-lane centerline', () => {
      const trackSpec = createStraightTrackSpec();
      const focalSampleIndex = 2;
      const focalSample = trackSpec.splineSamples[focalSampleIndex]!;
      const frame = resolveSplineSampleFrame(
        trackSpec.splineSamples,
        focalSample.globalIndex,
      );
      const laneCount = 2;
      const laneWidthWorld = focalSample.width / laneCount;
      const innerOffsetWorld = focalSample.width / 2 - laneWidthWorld / 2;
      const envState = createEnvironmentState({
        carX: focalSample.x + frame.normalX * innerOffsetWorld,
        carY: focalSample.y + frame.normalY * innerOffsetWorld,
        carHeading: frame.tangentHeadingRadians,
      });
      const controllerState = { targetSegmentIndex: focalSample.segmentIndex };

      const controlOutput = computeScriptedControl(
        envState,
        trackSpec,
        controllerState,
      );

      expect({
        steerNearZero: Math.abs(controlOutput.steer) < 0.05,
      }).toEqual({
        steerNearZero: true,
      });
    });
  });
});

function createStraightTrackSpec(): TrackSpec {
  const segments = [
    { startX: 0, startY: 0, endX: 100, endY: 0, width: 24 },
    { startX: 100, startY: 0, endX: 100, endY: 100, width: 24 },
    { startX: 100, startY: 100, endX: 0, endY: 100, width: 24 },
    { startX: 0, startY: 100, endX: 0, endY: 0, width: 24 },
  ];

  return {
    seed: 1,
    layoutVersion: 1,
    sizeBucket: 'straight-red',
    segments,
    splineSamples: [...buildTrackSplineSamples(segments)],
  };
}
