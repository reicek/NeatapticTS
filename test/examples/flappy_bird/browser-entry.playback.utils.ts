import {
  resolveFramePrimaryWinnerIndex,
  resolveLeaderPipesPassed,
} from './browser-entry.observation.utils';
import {
  resolveVisibleWorldWidthPx,
  resolveWorldViewport,
} from './browser-entry.viewport.utils';
import { requestWorkerPlaybackStep } from './browser-entry.worker-channel.utils';
import {
  FLAPPY_EMULATION_SPEED_MULTIPLIER,
  FLAPPY_BIRD_SHINE_FILL_STYLE,
  FLAPPY_BIRD_SHINE_INSET_RATIO,
  FLAPPY_BIRD_SHINE_SIZE_RATIO,
  FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR,
  FLAPPY_BIRD_WHITE_SHINE_GLOW_BLUR_PX,
  FLAPPY_BIRD_BODY_GLOW_BLUR_PX,
  FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX,
  FLAPPY_BIRD_CHAMPION_RED_GLOW_ALPHA,
  FLAPPY_BIRD_CHAMPION_RED_GLOW_EXPAND_PX,
  FLAPPY_LEADER_RING_LINE_WIDTH_PX,
  FLAPPY_LEADER_RING_GLOW_BLUR_PX,
  FLAPPY_LEADER_RING_RADIUS_OFFSET_PX,
  FLAPPY_NON_CHAMPION_OPACITY,
  FLAPPY_PIPE_OUTLINE_INNER_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_INNER_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_INSET_PX,
  FLAPPY_PIPE_OUTLINE_AURA_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_AURA_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_CORE_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_CORE_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_OUTER_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_OUTER_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_SEPARATOR_COLOR,
  FLAPPY_TRAIL_LINE_WIDTH_PX,
  FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX,
  FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX,
  FLAPPY_TRAIL_MAX_POINTS,
  FLAPPY_NEON_PALETTE,
} from './browser-entry.constants';
import type {
  EvolutionPlaybackStepSnapshot,
  PlaybackFrameStats,
  PopulationRenderState,
  TrailPoint,
  TrailState,
} from './browser-entry.types';
import {
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_TRAIL_OPACITY_FACTOR,
} from './constants';

/**
 * Renders all birds from the current generation in one shared world.
 *
 * @param canvas - Target playback canvas.
 * @param context - Canvas 2D context.
 * @param evolutionWorker - Worker owning playback state.
 * @param onFrameStats - Frame telemetry callback.
 * @returns Aggregate playback summary.
 */
export async function animatePopulationEpisode(
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  evolutionWorker: Worker,
  onFrameStats: (stats: PlaybackFrameStats) => void,
): Promise<{
  averagePipesPassed: number;
  p90FramesSurvived: number;
  winnerPipesPassed: number;
  winnerFramesSurvived: number;
}> {
  // Step 1: Initialize worker playback state with current viewport width.
  evolutionWorker.postMessage({
    type: 'start-playback',
    payload: {
      visibleWorldWidthPx: resolveVisibleWorldWidthPx(canvas),
    },
  });

  // Step 2: Initialize local render/trail state mirrors.
  const renderState: PopulationRenderState = {
    frameIndex: 0,
    visibleWorldWidthPx: resolveVisibleWorldWidthPx(canvas),
    nextPipeId: 0,
    lastSpawnedPipeGapPx: 0,
    lastSpawnedPipeGapCenterYPx: 0,
    lastSpawnedPipeSpawnIntervalFrames: 0,
    framesUntilNextPipeSpawn: 0,
    pipes: [],
    birds: [],
  };
  const trailState: TrailState = {
    birdTrailsY: [],
  };
  let simulationFrameBudget = 0;
  let finished = false;
  let averagePipesPassed = 0;
  let p90FramesSurvived = 0;
  let winnerPipesPassed = 0;
  let winnerFramesSurvived = 0;
  let latestLeaderPipesPassed = 0;
  let latestLeaderFramesSurvived = 0;

  // Step 3: Run playback batches until worker reports completion.
  while (!finished) {
    // Step 3.1: Resolve frame budget and request one playback step batch.
    renderState.visibleWorldWidthPx = resolveVisibleWorldWidthPx(canvas);
    simulationFrameBudget += FLAPPY_EMULATION_SPEED_MULTIPLIER;
    const simulationStepsThisRender = Math.max(
      1,
      Math.floor(simulationFrameBudget),
    );
    simulationFrameBudget -= simulationStepsThisRender;

    const playbackStepPayload = await requestWorkerPlaybackStep(
      evolutionWorker,
      {
        simulationSteps: simulationStepsThisRender,
        visibleWorldWidthPx: renderState.visibleWorldWidthPx,
      },
    );

    // Step 3.2: Apply state snapshot and update visual trail cache.
    applyPlaybackSnapshot(renderState, playbackStepPayload.snapshot);
    updateTrailState(trailState, renderState);

    // Step 3.3: Resolve leader metrics and emit frame telemetry.
    const leaderPipesPassed = resolveLeaderPipesPassed(renderState.birds);
    const leaderFramesSurvived = resolveLeaderFramesSurvived(renderState);
    latestLeaderPipesPassed = leaderPipesPassed;
    latestLeaderFramesSurvived = leaderFramesSurvived;
    onFrameStats({
      frameIndex: renderState.frameIndex,
      leaderPipesPassed,
      leaderFramesSurvived,
      activationCallsPerFrame:
        playbackStepPayload.instrumentation?.activationCallsPerFrame ?? 0,
      simulationStepsPerRaf:
        playbackStepPayload.instrumentation?.simulationStepsPerRaf ?? 0,
    });

    // Step 3.4: Fold end-of-playback aggregates when worker signals done.
    if (playbackStepPayload.done) {
      finished = true;
      averagePipesPassed = playbackStepPayload.averagePipesPassed ?? 0;
      p90FramesSurvived = playbackStepPayload.p90FramesSurvived ?? 0;
      winnerPipesPassed =
        playbackStepPayload.winnerPipesPassed ?? latestLeaderPipesPassed;
      winnerFramesSurvived =
        playbackStepPayload.winnerFramesSurvived ?? latestLeaderFramesSurvived;
    }

    // Step 3.5: Render frame and yield to browser RAF when still active.
    renderPopulationFrame(context, renderState, trailState);
    if (!finished) {
      await nextAnimationFrame();
    }
  }

  // Step 4: Render final settled frame and return aggregate summary.
  renderPopulationFrame(context, renderState, trailState);

  return {
    averagePipesPassed,
    p90FramesSurvived,
    winnerPipesPassed,
    winnerFramesSurvived,
  };
}

/**
 * Applies worker snapshot data to the local render state.
 *
 * @param renderState - Mutable render state.
 * @param snapshot - Worker playback snapshot.
 * @returns Nothing.
 */
function applyPlaybackSnapshot(
  renderState: PopulationRenderState,
  snapshot: EvolutionPlaybackStepSnapshot,
): void {
  renderState.frameIndex = snapshot.frameIndex;
  renderState.visibleWorldWidthPx = snapshot.visibleWorldWidthPx;
  renderState.pipes = snapshot.pipes;
  renderState.birds = snapshot.birds.map((birdSnapshot) => ({
    color: birdSnapshot.color,
    yPx: birdSnapshot.yPx,
    pipesPassed: birdSnapshot.pipesPassed,
    framesSurvived: birdSnapshot.framesSurvived,
    done: birdSnapshot.done,
  }));
}

/**
 * Draws one simulation frame for the current population state.
 *
 * @param context - Canvas 2D drawing context.
 * @param renderState - Mutable simulation state snapshot.
 * @param trailState - Leader trail render cache.
 * @returns Nothing.
 */
function renderPopulationFrame(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
  trailState: TrailState,
): void {
  // Step 1: Resolve viewport transform and clear target canvas.
  const viewport = resolveWorldViewport(context.canvas);
  const cameraLeftPx = 0;

  context.clearRect(0, 0, context.canvas.width, context.canvas.height);

  context.save();
  context.translate(viewport.offsetXPx, viewport.offsetYPx);
  context.scale(viewport.scale, viewport.scale);
  context.translate(-cameraLeftPx, 0);

  // Step 2: Draw pipes with neon multi-pass outlines.
  for (const pipe of renderState.pipes) {
    const gapHalf = pipe.gapSizePx * 0.5;
    const pipeLeft = pipe.xPx;
    const gapTop = pipe.gapCenterYPx - gapHalf;
    const gapBottom = pipe.gapCenterYPx + gapHalf;

    context.fillStyle = FLAPPY_NEON_PALETTE.pipeFill;
    context.fillRect(pipeLeft, 0, FLAPPY_PIPE_WIDTH_PX, gapTop);
    drawPipeNeonOutline(context, pipeLeft, 0, FLAPPY_PIPE_WIDTH_PX, gapTop);
    context.fillRect(
      pipeLeft,
      gapBottom,
      FLAPPY_PIPE_WIDTH_PX,
      FLAPPY_WORLD_HEIGHT_PX - gapBottom,
    );
    drawPipeNeonOutline(
      context,
      pipeLeft,
      gapBottom,
      FLAPPY_PIPE_WIDTH_PX,
      FLAPPY_WORLD_HEIGHT_PX - gapBottom,
    );
  }

  // Step 3: Resolve champion bird and render bird bodies/shine/rings.
  const leaderBirdIndex = resolveLeaderBirdIndex(renderState);
  const fallbackAliveBirdIndex = renderState.birds.findIndex(
    (bird) => !bird.done,
  );
  const championBirdIndex =
    leaderBirdIndex >= 0 ? leaderBirdIndex : fallbackAliveBirdIndex;

  renderState.birds.forEach((bird, birdIndex) => {
    if (bird.done) {
      return;
    }

    const birdSideLengthPx = Math.max(1, Math.round(FLAPPY_BIRD_RADIUS_PX * 2));
    const birdLeftPx = Math.round(FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX);
    const birdTopPx = Math.round(bird.yPx - FLAPPY_BIRD_RADIUS_PX);
    const birdOpacity =
      birdIndex === championBirdIndex ? 1 : FLAPPY_NON_CHAMPION_OPACITY;
    const birdRenderColor =
      birdIndex === championBirdIndex
        ? FLAPPY_NEON_PALETTE.championBird
        : FLAPPY_NEON_PALETTE.nonChampionBird;
    const isChampionBird = birdIndex === championBirdIndex;

    if (isChampionBird) {
      const expandedGlowInsetPx = Math.round(
        FLAPPY_BIRD_CHAMPION_RED_GLOW_EXPAND_PX,
      );
      context.globalAlpha = birdOpacity * FLAPPY_BIRD_CHAMPION_RED_GLOW_ALPHA;
      context.fillStyle = FLAPPY_NEON_PALETTE.championBird;
      context.shadowColor = FLAPPY_NEON_PALETTE.championBird;
      context.shadowBlur =
        FLAPPY_BIRD_BODY_GLOW_BLUR_PX + FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX;
      context.fillRect(
        birdLeftPx - expandedGlowInsetPx,
        birdTopPx - expandedGlowInsetPx,
        birdSideLengthPx + expandedGlowInsetPx * 2,
        birdSideLengthPx + expandedGlowInsetPx * 2,
      );
    }

    context.globalAlpha = birdOpacity;
    context.fillStyle = birdRenderColor;
    context.shadowColor = birdRenderColor;
    context.shadowBlur =
      FLAPPY_BIRD_BODY_GLOW_BLUR_PX +
      (isChampionBird ? FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX : 0);
    context.fillRect(birdLeftPx, birdTopPx, birdSideLengthPx, birdSideLengthPx);

    const shineInsetPx = birdSideLengthPx * FLAPPY_BIRD_SHINE_INSET_RATIO;
    const shineSideLengthPx = Math.max(
      1,
      birdSideLengthPx * FLAPPY_BIRD_SHINE_SIZE_RATIO,
    );
    context.fillStyle = FLAPPY_BIRD_SHINE_FILL_STYLE;
    context.shadowColor = FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR;
    context.shadowBlur = FLAPPY_BIRD_WHITE_SHINE_GLOW_BLUR_PX;
    context.fillRect(
      Math.round(birdLeftPx + shineInsetPx),
      Math.round(birdTopPx + shineInsetPx),
      Math.round(shineSideLengthPx),
      Math.round(shineSideLengthPx),
    );
    context.shadowBlur = 0;
    context.shadowColor = 'transparent';

    if (isChampionBird) {
      context.strokeStyle = FLAPPY_NEON_PALETTE.leaderRing;
      context.lineWidth = FLAPPY_LEADER_RING_LINE_WIDTH_PX;
      context.shadowColor = FLAPPY_NEON_PALETTE.leaderRing;
      context.shadowBlur = FLAPPY_LEADER_RING_GLOW_BLUR_PX;
      const leaderRingInsetPx = Math.round(FLAPPY_LEADER_RING_RADIUS_OFFSET_PX);
      context.strokeRect(
        birdLeftPx - leaderRingInsetPx,
        birdTopPx - leaderRingInsetPx,
        birdSideLengthPx + leaderRingInsetPx * 2,
        birdSideLengthPx + leaderRingInsetPx * 2,
      );
      context.shadowBlur = 0;
      context.shadowColor = 'transparent';
    }
  });

  // Step 4: Draw stepped trails for active birds.
  renderState.birds.forEach((bird, birdIndex) => {
    if (bird.done) {
      return;
    }

    const birdTrailPoints = trailState.birdTrailsY[birdIndex];
    if (!birdTrailPoints || birdTrailPoints.length === 0) {
      return;
    }

    const parentBirdOpacity =
      birdIndex === championBirdIndex ? 1 : FLAPPY_NON_CHAMPION_OPACITY;
    const birdTrailColor =
      birdIndex === championBirdIndex
        ? FLAPPY_NEON_PALETTE.championBird
        : FLAPPY_NEON_PALETTE.nonChampionBird;
    context.globalAlpha = parentBirdOpacity * FLAPPY_TRAIL_OPACITY_FACTOR;
    drawTrail(
      context,
      birdTrailPoints,
      birdTrailColor,
      FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX,
    );
  });

  context.globalAlpha = 1;
  // Step 5: Restore context to pre-viewport transform state.
  context.restore();
}

/**
 * Draws multi-pass neon outline around a pipe rectangle.
 *
 * @param context - Canvas 2D context.
 * @param rectangleLeftPx - Rectangle left position.
 * @param rectangleTopPx - Rectangle top position.
 * @param rectangleWidthPx - Rectangle width.
 * @param rectangleHeightPx - Rectangle height.
 * @returns Nothing.
 */
function drawPipeNeonOutline(
  context: CanvasRenderingContext2D,
  rectangleLeftPx: number,
  rectangleTopPx: number,
  rectangleWidthPx: number,
  rectangleHeightPx: number,
): void {
  // Step 1: Guard degenerate rectangles.
  if (rectangleWidthPx <= 0 || rectangleHeightPx <= 0) {
    return;
  }

  const alignedLeftPx = Math.round(rectangleLeftPx);
  const alignedTopPx = Math.round(rectangleTopPx);
  const alignedWidthPx = Math.max(1, Math.round(rectangleWidthPx));
  const alignedHeightPx = Math.max(1, Math.round(rectangleHeightPx));

  // Step 2: Define pixel-aligned stroke helper to avoid blurry edges.
  const strokeAlignedRect = (
    leftPx: number,
    topPx: number,
    widthPx: number,
    heightPx: number,
    lineWidthPx: number,
  ): void => {
    const oddLineAlignmentOffsetPx = lineWidthPx % 2 === 1 ? 0.5 : 0;
    context.lineWidth = lineWidthPx;
    context.strokeRect(
      Math.round(leftPx) + oddLineAlignmentOffsetPx,
      Math.round(topPx) + oddLineAlignmentOffsetPx,
      Math.max(1, Math.round(widthPx)),
      Math.max(1, Math.round(heightPx)),
    );
  };

  // Step 3: Draw outer aura pass.
  context.strokeStyle = FLAPPY_PIPE_OUTLINE_AURA_GLOW_COLOR;
  context.shadowColor = FLAPPY_PIPE_OUTLINE_AURA_GLOW_COLOR;
  context.shadowBlur = FLAPPY_PIPE_OUTLINE_AURA_GLOW_BLUR_PX;
  strokeAlignedRect(
    alignedLeftPx - FLAPPY_PIPE_OUTLINE_INSET_PX * 2,
    alignedTopPx - FLAPPY_PIPE_OUTLINE_INSET_PX * 2,
    alignedWidthPx + FLAPPY_PIPE_OUTLINE_INSET_PX * 4,
    alignedHeightPx + FLAPPY_PIPE_OUTLINE_INSET_PX * 4,
    1,
  );

  // Step 4: Draw outer bright pass.
  context.strokeStyle = FLAPPY_NEON_PALETTE.pipeEdgeOuter;
  context.shadowColor = FLAPPY_PIPE_OUTLINE_OUTER_GLOW_COLOR;
  context.shadowBlur = FLAPPY_PIPE_OUTLINE_OUTER_GLOW_BLUR_PX;
  strokeAlignedRect(
    alignedLeftPx - FLAPPY_PIPE_OUTLINE_INSET_PX,
    alignedTopPx - FLAPPY_PIPE_OUTLINE_INSET_PX,
    alignedWidthPx + FLAPPY_PIPE_OUTLINE_INSET_PX * 2,
    alignedHeightPx + FLAPPY_PIPE_OUTLINE_INSET_PX * 2,
    1,
  );

  // Step 5: Draw separator/core structure pass.
  context.strokeStyle = FLAPPY_PIPE_OUTLINE_SEPARATOR_COLOR;
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  strokeAlignedRect(
    alignedLeftPx,
    alignedTopPx,
    alignedWidthPx,
    alignedHeightPx,
    1,
  );

  // Step 6: Draw inner inset pass when enough area remains.
  if (
    alignedWidthPx > FLAPPY_PIPE_OUTLINE_INSET_PX * 2 &&
    alignedHeightPx > FLAPPY_PIPE_OUTLINE_INSET_PX * 2
  ) {
    context.strokeStyle = FLAPPY_NEON_PALETTE.pipeEdgeInner;
    context.shadowColor = FLAPPY_PIPE_OUTLINE_INNER_GLOW_COLOR;
    context.shadowBlur = FLAPPY_PIPE_OUTLINE_INNER_GLOW_BLUR_PX;
    strokeAlignedRect(
      alignedLeftPx + FLAPPY_PIPE_OUTLINE_INSET_PX,
      alignedTopPx + FLAPPY_PIPE_OUTLINE_INSET_PX,
      alignedWidthPx - FLAPPY_PIPE_OUTLINE_INSET_PX * 2,
      alignedHeightPx - FLAPPY_PIPE_OUTLINE_INSET_PX * 2,
      1,
    );
  }

  // Step 7: Draw final core glow pass and reset shadow state.
  context.strokeStyle = FLAPPY_PIPE_OUTLINE_CORE_GLOW_COLOR;
  context.shadowColor = FLAPPY_PIPE_OUTLINE_CORE_GLOW_COLOR;
  context.shadowBlur = FLAPPY_PIPE_OUTLINE_CORE_GLOW_BLUR_PX;
  strokeAlignedRect(
    alignedLeftPx,
    alignedTopPx,
    alignedWidthPx,
    alignedHeightPx,
    1,
  );
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
}

/**
 * Updates the trail cache from the latest frame snapshot.
 *
 * @param trailState - Mutable trail state.
 * @param renderState - Current render state.
 * @returns Nothing.
 */
function updateTrailState(
  trailState: TrailState,
  renderState: PopulationRenderState,
): void {
  renderState.birds.forEach((bird, birdIndex) => {
    if (!trailState.birdTrailsY[birdIndex]) {
      trailState.birdTrailsY[birdIndex] = [];
    }
    const birdTrail = trailState.birdTrailsY[birdIndex];

    if (bird.done) {
      birdTrail.length = 0;
      return;
    }

    pushTrailPoint(birdTrail, renderState.frameIndex, bird.yPx);
  });
}

/**
 * Appends one trail point while enforcing max history length.
 *
 * @param trailPoints - Mutable trail collection.
 * @param frameIndex - Source frame index.
 * @param yPosition - Bird y position.
 * @returns Nothing.
 */
function pushTrailPoint(
  trailPoints: TrailPoint[],
  frameIndex: number,
  yPosition: number,
): void {
  trailPoints.push({ frameIndex, yPx: yPosition });
  const maxTrailPoints = FLAPPY_TRAIL_MAX_POINTS;
  if (trailPoints.length > maxTrailPoints) {
    trailPoints.splice(0, trailPoints.length - maxTrailPoints);
  }
}

/**
 * Renders a stepped trail polyline.
 *
 * @param context - Canvas 2D context.
 * @param trailPoints - Ordered trail points.
 * @param color - Trail color.
 * @param anchorX - Latest point x-anchor.
 * @returns Nothing.
 */
function drawTrail(
  context: CanvasRenderingContext2D,
  trailPoints: TrailPoint[],
  color: string,
  anchorX: number,
): void {
  // Step 1: Guard empty trails.
  if (trailPoints.length === 0) {
    return;
  }

  const latestTrailFrameIndex = trailPoints.at(-1)?.frameIndex ?? 0;

  const firstTrailPoint = trailPoints[0];
  const firstTrailFrameOffset = Math.max(
    0,
    latestTrailFrameIndex - firstTrailPoint.frameIndex,
  );
  let previousXPosition =
    anchorX - firstTrailFrameOffset * FLAPPY_PIPE_SPEED_PX_PER_FRAME;
  let previousYPosition = firstTrailPoint.yPx;

  // Step 2: Start stepped polyline at earliest trail point.
  context.strokeStyle = color;
  context.lineWidth = FLAPPY_TRAIL_LINE_WIDTH_PX;
  context.beginPath();
  context.moveTo(previousXPosition, previousYPosition);

  // Step 3: Append horizontal-then-vertical stepped segments for each point.
  trailPoints.slice(1).forEach((trailPoint) => {
    const frameOffset = Math.max(
      0,
      latestTrailFrameIndex - trailPoint.frameIndex,
    );
    const nextXPosition =
      anchorX - frameOffset * FLAPPY_PIPE_SPEED_PX_PER_FRAME;
    const nextYPosition = trailPoint.yPx;

    const horizontalDeltaPx = nextXPosition - previousXPosition;
    const horizontalDirection = Math.sign(horizontalDeltaPx) || 1;
    const steppedHorizontalLengthPx = Math.max(
      Math.abs(horizontalDeltaPx),
      FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX,
    );
    const steppedHorizontalXPosition =
      previousXPosition + horizontalDirection * steppedHorizontalLengthPx;
    context.lineTo(steppedHorizontalXPosition, previousYPosition);

    const verticalDeltaPx = nextYPosition - previousYPosition;
    if (verticalDeltaPx !== 0) {
      const verticalDirection = Math.sign(verticalDeltaPx);
      const steppedVerticalLengthPx = Math.max(
        Math.abs(verticalDeltaPx),
        FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX,
      );
      const steppedVerticalYPosition =
        previousYPosition + verticalDirection * steppedVerticalLengthPx;
      context.lineTo(steppedHorizontalXPosition, steppedVerticalYPosition);
    }

    context.lineTo(steppedHorizontalXPosition, nextYPosition);
    context.lineTo(nextXPosition, nextYPosition);
    previousXPosition = nextXPosition;
    previousYPosition = nextYPosition;
  });

  // Step 4: Flush stroke path.
  context.stroke();
}

/**
 * Resolves the current leader index among alive birds.
 *
 * @param renderState - Current render state.
 * @returns Leader index or `-1`.
 */
function resolveLeaderBirdIndex(renderState: PopulationRenderState): number {
  return resolveFramePrimaryWinnerIndex(renderState.birds, true);
}

/**
 * Resolves the maximum survived-frame count in the current render state.
 *
 * @param renderState - Current render state.
 * @returns Maximum frames survived.
 */
function resolveLeaderFramesSurvived(
  renderState: PopulationRenderState,
): number {
  return renderState.birds.reduce(
    (maximumFramesSurvived, bird) =>
      Math.max(maximumFramesSurvived, bird.framesSurvived),
    0,
  );
}

/**
 * Yields until the next browser animation frame.
 *
 * @returns Promise resolved on next animation frame.
 */
function nextAnimationFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}
