import { computeScriptedControl } from './scripted.controller';
import {
  createCurvedTrackSpec,
  createEnvironmentState,
  selectControllerFocalSample,
} from './pathtracking.test.fixtures';

describe('scripted.controller', () => {
  describe('computeScriptedControl', () => {
    it('keeps steer near the lane tangent instead of pulling toward a raw chord endpoint on curved samples', () => {
      const trackSpec = createCurvedTrackSpec();
      const { focalSample, focalFrame } = selectControllerFocalSample(trackSpec);
      const envState = createEnvironmentState({
        carX: focalSample.x,
        carY: focalSample.y,
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
  });
});