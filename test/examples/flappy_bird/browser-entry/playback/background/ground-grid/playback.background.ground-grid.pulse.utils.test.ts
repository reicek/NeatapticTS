import {
  FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS,
  FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS,
  FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO,
  FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO,
} from './playback.background.ground-grid.constants';
import { resolvePlaybackGroundGridPulse } from './playback.background.ground-grid.pulse.utils';
import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridPulsePath,
} from './playback.background.ground-grid.types';

describe('resolvePlaybackGroundGridPulse', () => {
  it('keeps a vertical spark on the same ray when visible vertical candidates shift', () => {
    const sceneContext = createSceneContext();
    const verticalPulsePaths = createVerticalPulsePaths();
    const startingFrameIndex = findVerticalPulseFrameIndex(
      sceneContext,
      verticalPulsePaths,
    );
    const firstPulse = resolvePlaybackGroundGridPulse({
      frameIndex: startingFrameIndex,
      horizontalPulsePaths: [],
      sceneContext,
      verticalPulsePaths,
      visibleVerticalPulsePaths: verticalPulsePaths.slice(1, 3),
    });
    resolvePlaybackGroundGridPulse({
      frameIndex: startingFrameIndex + 1,
      horizontalPulsePaths: [],
      sceneContext,
      verticalPulsePaths,
      visibleVerticalPulsePaths: verticalPulsePaths.slice(2, 4),
    });

    expect(firstPulse).not.toBeNull();
  });

  it('continues the same vertical spark trajectory when edge visibility changes', () => {
    const sceneContext = createSceneContext();
    const verticalPulsePaths = createVerticalPulsePaths();
    const startingFrameIndex = findVerticalPulseFrameIndex(
      sceneContext,
      verticalPulsePaths,
    );
    resolvePlaybackGroundGridPulse({
      frameIndex: startingFrameIndex,
      horizontalPulsePaths: [],
      sceneContext,
      verticalPulsePaths,
      visibleVerticalPulsePaths: verticalPulsePaths.slice(1, 3),
    });
    const secondPulse = resolvePlaybackGroundGridPulse({
      frameIndex: startingFrameIndex + 1,
      horizontalPulsePaths: [],
      sceneContext,
      verticalPulsePaths,
      visibleVerticalPulsePaths: verticalPulsePaths.slice(2, 4),
    });

    expect(secondPulse).not.toBeNull();
  });

  it('moves the vertical spark only by the expected per-frame delta', () => {
    const sceneContext = createSceneContext();
    const verticalPulsePaths = createVerticalPulsePaths();
    const startingFrameIndex = findVerticalPulseFrameIndex(
      sceneContext,
      verticalPulsePaths,
    );
    const firstPulse = resolvePlaybackGroundGridPulse({
      frameIndex: startingFrameIndex,
      horizontalPulsePaths: [],
      sceneContext,
      verticalPulsePaths,
      visibleVerticalPulsePaths: verticalPulsePaths.slice(1, 3),
    });
    const secondPulse = resolvePlaybackGroundGridPulse({
      frameIndex: startingFrameIndex + 1,
      horizontalPulsePaths: [],
      sceneContext,
      verticalPulsePaths,
      visibleVerticalPulsePaths: verticalPulsePaths.slice(2, 4),
    });

    const stablePath = resolveClosestPulsePath(
      firstPulse!,
      verticalPulsePaths,
      resolveVerticalTravelProgressRatio(startingFrameIndex),
    );
    const expectedNextCenterXPx = interpolatePathXPx(
      stablePath,
      resolveVerticalTravelProgressRatio(startingFrameIndex + 1),
    );

    expect(secondPulse!.centerXPx).toBeCloseTo(expectedNextCenterXPx, 6);
  });

  it('keeps a vertical spark continuous across wrapped ray lists', () => {
    const sceneContext = createSceneContext();
    const startingFrameIndex = findVerticalPulseFrameIndex(sceneContext, [
      createVerticalPulsePath(40),
      createVerticalPulsePath(140),
      createVerticalPulsePath(240),
      createVerticalPulsePath(340),
      createVerticalPulsePath(440),
    ]);
    const firstVerticalPulsePaths = [
      createVerticalPulsePath(-260),
      createVerticalPulsePath(-160),
      createVerticalPulsePath(-60),
      createVerticalPulsePath(40),
      createVerticalPulsePath(140),
    ];
    const secondVerticalPulsePaths = [
      createVerticalPulsePath(60),
      createVerticalPulsePath(160),
      createVerticalPulsePath(260),
      createVerticalPulsePath(360),
      createVerticalPulsePath(460),
    ];
    const firstPulse = resolvePlaybackGroundGridPulse({
      frameIndex: startingFrameIndex,
      horizontalPulsePaths: [],
      sceneContext,
      verticalPulsePaths: firstVerticalPulsePaths,
      visibleVerticalPulsePaths: firstVerticalPulsePaths,
    });
    const secondPulse = resolvePlaybackGroundGridPulse({
      frameIndex: startingFrameIndex + 1,
      horizontalPulsePaths: [],
      sceneContext,
      verticalPulsePaths: secondVerticalPulsePaths,
      visibleVerticalPulsePaths: secondVerticalPulsePaths,
    });

    expect(
      Math.abs(secondPulse!.centerXPx - firstPulse!.centerXPx),
    ).toBeLessThan(140);
  });
});

function findVerticalPulseFrameIndex(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[],
): number {
  for (let frameIndex = 0; frameIndex < 4_000; frameIndex += 1) {
    const pulse = resolvePlaybackGroundGridPulse({
      frameIndex,
      horizontalPulsePaths: [],
      sceneContext,
      verticalPulsePaths,
      visibleVerticalPulsePaths: verticalPulsePaths,
    });
    if (pulse) {
      return frameIndex;
    }
  }

  throw new Error(
    'Expected to find a vertical pulse frame for the test fixture.',
  );
}

function resolveVerticalTravelProgressRatio(frameIndex: number): number {
  const currentTimeMs =
    frameIndex * FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS;
  const pulseSlotIndex = Math.floor(
    currentTimeMs / FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS,
  );
  const pulseElapsedMs =
    currentTimeMs - pulseSlotIndex * FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS;
  const lifetimeProgressRatio =
    pulseElapsedMs / FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS;
  const directionIsForward = resolveTestUnitHash(pulseSlotIndex, 29) >= 0.5;
  const baseTravelProgressRatio = directionIsForward
    ? lifetimeProgressRatio
    : 1 - lifetimeProgressRatio;

  return (
    FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO +
    (FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO -
      FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO) *
      baseTravelProgressRatio
  );
}

function resolveClosestPulsePath(
  pulse: { centerXPx: number; centerYPx: number },
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[],
  travelProgressRatio: number,
): PlaybackGroundGridPulsePath {
  let closestPath = verticalPulsePaths[0];
  let closestDistanceSq = Number.POSITIVE_INFINITY;

  for (const verticalPulsePath of verticalPulsePaths) {
    const centerXPx = interpolatePathXPx(
      verticalPulsePath,
      travelProgressRatio,
    );
    const centerYPx = interpolatePathYPx(
      verticalPulsePath,
      travelProgressRatio,
    );
    const deltaXPx = centerXPx - pulse.centerXPx;
    const deltaYPx = centerYPx - pulse.centerYPx;
    const distanceSq = deltaXPx * deltaXPx + deltaYPx * deltaYPx;
    if (distanceSq < closestDistanceSq) {
      closestDistanceSq = distanceSq;
      closestPath = verticalPulsePath;
    }
  }

  return closestPath;
}

function interpolatePathXPx(
  pulsePath: PlaybackGroundGridPulsePath,
  travelProgressRatio: number,
): number {
  return (
    pulsePath.startXPx +
    (pulsePath.endXPx - pulsePath.startXPx) * travelProgressRatio
  );
}

function interpolatePathYPx(
  pulsePath: PlaybackGroundGridPulsePath,
  travelProgressRatio: number,
): number {
  return (
    pulsePath.startYPx +
    (pulsePath.endYPx - pulsePath.startYPx) * travelProgressRatio
  );
}

function resolveTestUnitHash(seed: number, salt: number): number {
  let hashedValue = (seed + 1) ^ Math.imul(salt + 1, 374_761_393);
  hashedValue = Math.imul(hashedValue ^ (hashedValue >>> 13), 1_274_126_177);
  hashedValue ^= hashedValue >>> 16;
  return (hashedValue >>> 0) / 0x1_0000_0000;
}

function createSceneContext(): PlaybackBackgroundGroundGridSceneContext {
  return {
    viewportOffsetXPx: 0,
    visibleWorldWidthPx: 640,
    alignedHorizonYPx: 220,
    lowerBandTopYPx: 220,
    lowerBandHeightPx: 260,
    lowerBandBottomYPx: 480,
    vanishingPointXPx: 320,
    vanishingPointYPx: 220,
  };
}

function createVerticalPulsePaths(): readonly PlaybackGroundGridPulsePath[] {
  return [
    createVerticalPulsePath(60),
    createVerticalPulsePath(160),
    createVerticalPulsePath(260),
    createVerticalPulsePath(360),
    createVerticalPulsePath(460),
  ];
}

function createVerticalPulsePath(
  startXPx: number,
): PlaybackGroundGridPulsePath {
  return {
    orientation: 'vertical',
    startXPx,
    startYPx: 480,
    endXPx: 320,
    endYPx: 220,
    thicknessPx: 2,
  };
}
