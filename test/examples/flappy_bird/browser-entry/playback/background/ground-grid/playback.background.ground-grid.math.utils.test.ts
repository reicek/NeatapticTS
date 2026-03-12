import { FLAPPY_PIPE_SPEED_PX_PER_FRAME } from '../../../../constants/constants.pipes';
import { FLAPPY_GROUND_GRID_SCROLL_RATIO } from './playback.background.ground-grid.constants';
import { resolvePlaybackGroundGridPipeConnectionProfile } from './playback.background.ground-grid.math.utils';

describe('resolvePlaybackGroundGridPipeConnectionProfile', () => {
  it('places the lower-pipe floor above the bottom edge', () => {
    const pipeConnectionProfile =
      resolvePlaybackGroundGridPipeConnectionProfile(512);

    expect(pipeConnectionProfile.pipeFloorYPx).toBeLessThan(512);
  });

  it('matches perspective-ray motion to pipe speed at the connection line', () => {
    const visibleWorldHeightPx = 512;
    const pipeConnectionProfile =
      resolvePlaybackGroundGridPipeConnectionProfile(visibleWorldHeightPx);
    const lowerBandBottomYPx = visibleWorldHeightPx;
    const vanishingPointYPx = visibleWorldHeightPx * 0.5;
    const interpolationRatio =
      (pipeConnectionProfile.pipeFloorYPx - lowerBandBottomYPx) /
      (vanishingPointYPx - lowerBandBottomYPx);
    const localRaySpeedPxPerFrame =
      FLAPPY_PIPE_SPEED_PX_PER_FRAME *
      FLAPPY_GROUND_GRID_SCROLL_RATIO *
      pipeConnectionProfile.matchedRayScrollRatio *
      (1 - interpolationRatio);

    expect(localRaySpeedPxPerFrame).toBeCloseTo(
      FLAPPY_PIPE_SPEED_PX_PER_FRAME,
      6,
    );
  });
});