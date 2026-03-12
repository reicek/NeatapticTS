import {
  FLAPPY_MAX_DIFFICULTY_PIPE_PITCH_PX,
  FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME,
  FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES,
} from '../../../../constants/constants.difficulty';
import {
  buildPlaybackGroundGridVerticalSceneMetrics,
  resolvePlaybackGroundGridVerticalCycleContext,
} from './playback.background.ground-grid.geometry.layout.utils';
import { resolvePlaybackGroundGridPipeConnectionProfile } from './playback.background.ground-grid.math.utils';
import type { PlaybackBackgroundGroundGridSceneContext } from './playback.background.ground-grid.types';

describe('buildPlaybackGroundGridVerticalSceneMetrics', () => {
  it('matches pipe-floor lane spacing to the actual max-difficulty pipe pitch', () => {
    const sceneContext = createSceneContext();
    const verticalSceneMetrics =
      buildPlaybackGroundGridVerticalSceneMetrics(sceneContext);
    const pipeConnectionProfile =
      resolvePlaybackGroundGridPipeConnectionProfile(
        sceneContext.lowerBandBottomYPx,
      );
    const retainedWidthRatioAtPipeConnection =
      (sceneContext.vanishingPointYPx - pipeConnectionProfile.pipeFloorYPx) /
      (sceneContext.vanishingPointYPx - sceneContext.lowerBandBottomYPx);
    const projectedLaneSpacingPx =
      verticalSceneMetrics.safeLaneSpacingPx *
      retainedWidthRatioAtPipeConnection;

    expect(projectedLaneSpacingPx).toBeCloseTo(
      FLAPPY_MAX_DIFFICULTY_PIPE_PITCH_PX,
      6,
    );
  });

  it('repeats the same grid phase after one max-difficulty spawn interval', () => {
    const sceneContext = createSceneContext();
    const verticalSceneMetrics =
      buildPlaybackGroundGridVerticalSceneMetrics(sceneContext);
    const pipeConnectionProfile =
      resolvePlaybackGroundGridPipeConnectionProfile(
        sceneContext.lowerBandBottomYPx,
      );
    const retainedWidthRatioAtPipeConnection =
      (sceneContext.vanishingPointYPx - pipeConnectionProfile.pipeFloorYPx) /
      (sceneContext.vanishingPointYPx - sceneContext.lowerBandBottomYPx);
    const projectedLaneSpacingPx =
      verticalSceneMetrics.safeLaneSpacingPx *
      retainedWidthRatioAtPipeConnection;
    const projectedTravelPerSpawnPx =
      FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME *
      FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES;

    expect(projectedTravelPerSpawnPx % projectedLaneSpacingPx).toBeCloseTo(
      0,
      6,
    );
  });

  it('retains enough rays to cover the full visible horizon span', () => {
    const sceneContext = createSceneContext();
    const verticalSceneMetrics =
      buildPlaybackGroundGridVerticalSceneMetrics(sceneContext);
    const coveredAnchorSpanPx =
      (verticalSceneMetrics.totalVisibleLaneCount - 1) *
      verticalSceneMetrics.safeLaneSpacingPx;

    expect(coveredAnchorSpanPx).toBeGreaterThanOrEqual(
      verticalSceneMetrics.visibleAnchorBounds.anchorSpanPx,
    );
  });
});

describe('resolvePlaybackGroundGridVerticalCycleContext', () => {
  it('preserves the lane spacing that the viewport metrics resolved', () => {
    const verticalCycleContext = resolvePlaybackGroundGridVerticalCycleContext(
      120,
      512,
      960,
    );

    expect(verticalCycleContext.safeLaneSpacingPx).toBe(120);
  });
});

function createSceneContext(): PlaybackBackgroundGroundGridSceneContext {
  return {
    viewportOffsetXPx: 0,
    visibleWorldWidthPx: 288,
    alignedHorizonYPx: 339,
    lowerBandTopYPx: 341,
    lowerBandHeightPx: 171,
    lowerBandBottomYPx: 512,
    vanishingPointXPx: 144,
    vanishingPointYPx: 256,
  };
}
