import {
  resolveAliveBirdCount,
  resolveFramePrimaryWinnerIndex,
  resolveLeaderPipesPassed,
} from './browser-entry.observation.utils';
import {
  resolveVisibleWorldHeightPx,
  resolveVisibleWorldWidthPx,
  resolveWorldViewport,
} from './browser-entry.viewport.utils';
import { requestWorkerPlaybackStep } from './browser-entry.worker-channel.utils';
import {
  FLAPPY_EMULATION_SPEED_MULTIPLIER,
  FLAPPY_BIRD_VIEWPORT_X_RATIO,
  FLAPPY_BIRD_CHAMPION_SHINE_FILL_STYLE,
  FLAPPY_BIRD_CHAMPION_SHINE_GLOW_COLOR,
  FLAPPY_BIRD_SHINE_FILL_STYLE,
  FLAPPY_BIRD_SHINE_INSET_RATIO,
  FLAPPY_BIRD_SHINE_SIZE_RATIO,
  FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR,
  FLAPPY_BIRD_WHITE_SHINE_GLOW_BLUR_PX,
  FLAPPY_BIRD_AURA_ALPHA,
  FLAPPY_BIRD_AURA_BLUR_MULTIPLIER,
  FLAPPY_BIRD_AURA_EXPAND_PX,
  FLAPPY_BIRD_BODY_GLOW_BLUR_PX,
  FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX,
  FLAPPY_BIRD_CHAMPION_RED_GLOW_ALPHA,
  FLAPPY_BIRD_CHAMPION_RED_GLOW_EXPAND_PX,
  FLAPPY_LEADER_RING_LINE_WIDTH_PX,
  FLAPPY_LEADER_RING_GLOW_BLUR_PX,
  FLAPPY_LEADER_RING_RADIUS_OFFSET_PX,
  FLAPPY_NON_CHAMPION_OPACITY,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX,
  FLAPPY_PIPE_OUTLINE_GLOW_ALPHA,
  FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX,
  FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX,
  FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX,
  FLAPPY_TRAIL_LINE_WIDTH_PX,
  FLAPPY_TRAIL_EDGE_FADE_DISTANCE_PX,
  FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX,
  FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX,
  FLAPPY_TRAIL_MAX_POINTS,
  FLAPPY_NEON_PALETTE,
  FLAPPY_STARFIELD_CYAN_FILL_STYLE,
  FLAPPY_STARFIELD_FAR_SCROLL_RATIO,
  FLAPPY_STARFIELD_MID_SCROLL_RATIO,
  FLAPPY_STARFIELD_NEAR_SCROLL_RATIO,
  FLAPPY_STARFIELD_TILE_WIDTH_PX,
} from '../constants/constants';
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
  FLAPPY_BIRD_X_PX,
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_TRAIL_OPACITY_FACTOR,
} from '../constants/constants';

type StarTile = {
  image: CanvasImageSource;
  tileWidthPx: number;
  tileHeightPx: number;
  scrollRatio: number;
};

const cachedStarfieldTilesByHeight = new Map<number, readonly StarTile[]>();

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
      visibleWorldHeightPx: resolveVisibleWorldHeightPx(canvas),
    },
  });

  // Step 2: Initialize local render/trail state mirrors.
  const renderState: PopulationRenderState = {
    frameIndex: 0,
    visibleWorldWidthPx: resolveVisibleWorldWidthPx(canvas),
    visibleWorldHeightPx: resolveVisibleWorldHeightPx(canvas),
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
    renderState.visibleWorldHeightPx = resolveVisibleWorldHeightPx(canvas);
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
        visibleWorldHeightPx: renderState.visibleWorldHeightPx,
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
      activeBirdCount: resolveAliveBirdCount(renderState.birds),
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
  renderState.visibleWorldHeightPx = snapshot.visibleWorldHeightPx;
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
  const visibleWorldWidthPx = Math.max(1, renderState.visibleWorldWidthPx);
  const visibleWorldHeightPx = Math.max(1, renderState.visibleWorldHeightPx);
  const desiredBirdScreenXPx =
    visibleWorldWidthPx * FLAPPY_BIRD_VIEWPORT_X_RATIO;
  const cameraLeftPx = FLAPPY_BIRD_X_PX - desiredBirdScreenXPx;

  // Reset state that can suppress shadows between frames.
  context.globalAlpha = 1;
  context.globalCompositeOperation = 'source-over';
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;

  context.clearRect(0, 0, context.canvas.width, context.canvas.height);

  context.save();
  context.translate(viewport.offsetXPx, viewport.offsetYPx);
  context.scale(viewport.scale, viewport.scale);
  context.translate(-cameraLeftPx, 0);

  // Step 2: Draw Radiant-style parallax background.
  drawParallaxBackground(context, renderState);

  // Step 3: Draw pipes with neon outlines.
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
      visibleWorldHeightPx - gapBottom,
    );
    drawPipeNeonOutline(
      context,
      pipeLeft,
      gapBottom,
      FLAPPY_PIPE_WIDTH_PX,
      visibleWorldHeightPx - gapBottom,
    );
  }

  // Step 4: Resolve champion bird and render bird bodies/shine/rings.
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

    // Step 4.1: Add a soft Radiant-style aura plate behind the champion bird.
    if (isChampionBird) {
      const auraExpandPx = Math.round(FLAPPY_BIRD_AURA_EXPAND_PX);
      const previousCompositeOperation = context.globalCompositeOperation;
      context.globalCompositeOperation = 'lighter';
      context.globalAlpha = birdOpacity * FLAPPY_BIRD_AURA_ALPHA;
      context.fillStyle = birdRenderColor;
      context.shadowColor = birdRenderColor;
      context.shadowBlur = Math.round(
        FLAPPY_BIRD_BODY_GLOW_BLUR_PX * FLAPPY_BIRD_AURA_BLUR_MULTIPLIER,
      );
      context.fillRect(
        birdLeftPx - auraExpandPx,
        birdTopPx - auraExpandPx,
        birdSideLengthPx + auraExpandPx * 2,
        birdSideLengthPx + auraExpandPx * 2,
      );
      context.shadowBlur = 0;
      context.shadowColor = 'transparent';
      context.globalCompositeOperation = previousCompositeOperation;
    }

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
    context.fillStyle = isChampionBird
      ? FLAPPY_BIRD_CHAMPION_SHINE_FILL_STYLE
      : FLAPPY_BIRD_SHINE_FILL_STYLE;
    context.shadowColor = isChampionBird
      ? FLAPPY_BIRD_CHAMPION_SHINE_GLOW_COLOR
      : FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR;
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

  // Step 5: Draw stepped trails for active birds.
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
        ? FLAPPY_NEON_PALETTE.trail
        : FLAPPY_NEON_PALETTE.nonChampionBird;

    drawTrail(
      context,
      birdTrailPoints,
      birdTrailColor,
      FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX,
      parentBirdOpacity * FLAPPY_TRAIL_OPACITY_FACTOR,
      {
        leftXPx: cameraLeftPx,
        rightXPx: cameraLeftPx + visibleWorldWidthPx,
        topYPx: 0,
        bottomYPx: visibleWorldHeightPx,
      },
    );
  });

  context.globalAlpha = 1;
  // Step 6: Restore context to pre-viewport transform state.
  context.restore();
}

function drawParallaxBackground(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
): void {
  const visibleWorldWidthPx = Math.max(
    1,
    Math.round(renderState.visibleWorldWidthPx),
  );
  const scrollBasePx = renderState.frameIndex * FLAPPY_PIPE_SPEED_PX_PER_FRAME;

  // Step 1: Paint the background fill.
  context.globalAlpha = 1;
  context.globalCompositeOperation = 'source-over';
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.fillStyle = FLAPPY_NEON_PALETTE.background;
  const visibleWorldHeightPx = Math.max(
    1,
    Math.round(renderState.visibleWorldHeightPx),
  );
  context.fillRect(0, 0, visibleWorldWidthPx, visibleWorldHeightPx);

  // Step 2: Draw cached square-particle starfield layers with subtle parallax.
  // Each layer is pre-rendered into a tile and repeated via drawImage, which is
  // much faster than drawing dozens of blurred particles every frame.
  context.globalCompositeOperation = 'lighter';
  const starfieldTiles = resolveStarfieldTiles(visibleWorldHeightPx);
  for (const tile of starfieldTiles) {
    const scrollOffsetPx = scrollBasePx * tile.scrollRatio;
    drawTiledImageRow(context, {
      tile: tile.image,
      tileWidthPx: tile.tileWidthPx,
      visibleWidthPx: visibleWorldWidthPx,
      offsetPx: scrollOffsetPx,
    });
  }
  context.globalCompositeOperation = 'source-over';
}

function resolveStarfieldTiles(
  visibleWorldHeightPx: number,
): readonly StarTile[] {
  const tileHeightPx = Math.max(1, Math.round(visibleWorldHeightPx));
  const cachedTilesForHeight = cachedStarfieldTilesByHeight.get(tileHeightPx);
  if (cachedTilesForHeight) {
    return cachedTilesForHeight;
  }

  const farTile: StarTile = {
    image: createStarTileCanvas({
      seed: 1_337,
      tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
      tileHeightPx,
      starCount: 35,
      minSizePx: 1,
      maxSizePx: 2,
      minAlpha: 0.08,
      maxAlpha: 0.22,
      blurPx: 4,
    }),
    tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
    tileHeightPx,
    scrollRatio: FLAPPY_STARFIELD_FAR_SCROLL_RATIO,
  };
  const midTile: StarTile = {
    image: createStarTileCanvas({
      seed: 2_777,
      tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
      tileHeightPx,
      starCount: 28,
      minSizePx: 1,
      maxSizePx: 3,
      minAlpha: 0.1,
      maxAlpha: 0.28,
      blurPx: 6,
    }),
    tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
    tileHeightPx,
    scrollRatio: FLAPPY_STARFIELD_MID_SCROLL_RATIO,
  };
  const nearTile: StarTile = {
    image: createStarTileCanvas({
      seed: 4_242,
      tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
      tileHeightPx,
      starCount: 23,
      minSizePx: 2,
      maxSizePx: 4,
      minAlpha: 0.12,
      maxAlpha: 0.34,
      blurPx: 8,
    }),
    tileWidthPx: FLAPPY_STARFIELD_TILE_WIDTH_PX,
    tileHeightPx,
    scrollRatio: FLAPPY_STARFIELD_NEAR_SCROLL_RATIO,
  };

  const resolvedTiles = [farTile, midTile, nearTile] as const;
  cachedStarfieldTilesByHeight.set(tileHeightPx, resolvedTiles);
  return resolvedTiles;
}

function drawTiledImageRow(
  context: CanvasRenderingContext2D,
  layer: {
    tile: CanvasImageSource;
    tileWidthPx: number;
    visibleWidthPx: number;
    offsetPx: number;
  },
): void {
  const normalizedOffsetPx = positiveModulo(layer.offsetPx, layer.tileWidthPx);
  const maxTileIndex = Math.ceil(layer.visibleWidthPx / layer.tileWidthPx) + 1;

  for (let tileIndex = -1; tileIndex <= maxTileIndex; tileIndex += 1) {
    const tileLeftPx = tileIndex * layer.tileWidthPx - normalizedOffsetPx;
    context.drawImage(layer.tile, tileLeftPx, 0);
  }
}

function createStarTileCanvas(options: {
  seed: number;
  tileWidthPx: number;
  tileHeightPx: number;
  starCount: number;
  minSizePx: number;
  maxSizePx: number;
  minAlpha: number;
  maxAlpha: number;
  blurPx: number;
}): CanvasImageSource {
  const canvas = createCompatibleCanvas(
    options.tileWidthPx,
    options.tileHeightPx,
  );
  const tileContext = canvas.getContext('2d');
  if (!tileContext) {
    return canvas;
  }

  const seededRandom = createSeededRandom(options.seed);

  tileContext.clearRect(0, 0, canvas.width, canvas.height);
  tileContext.globalCompositeOperation = 'source-over';
  tileContext.shadowColor = FLAPPY_STARFIELD_CYAN_FILL_STYLE;
  tileContext.shadowBlur = options.blurPx;

  for (let starIndex = 0; starIndex < options.starCount; starIndex += 1) {
    const xPx = Math.floor(seededRandom() * options.tileWidthPx);
    const yPx = Math.floor(seededRandom() * options.tileHeightPx);
    const sizePx =
      options.minSizePx +
      Math.floor(seededRandom() * (options.maxSizePx - options.minSizePx + 1));
    const alpha =
      options.minAlpha + seededRandom() * (options.maxAlpha - options.minAlpha);

    tileContext.globalAlpha = alpha;
    tileContext.fillStyle = FLAPPY_STARFIELD_CYAN_FILL_STYLE;
    tileContext.fillRect(xPx, yPx, sizePx, sizePx);
  }

  tileContext.shadowBlur = 0;
  tileContext.shadowColor = 'transparent';
  tileContext.globalAlpha = 1;
  return canvas;
}

function createCompatibleCanvas(
  widthPx: number,
  heightPx: number,
): HTMLCanvasElement | OffscreenCanvas {
  const width = Math.max(1, Math.round(widthPx));
  const height = Math.max(1, Math.round(heightPx));

  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height);
  }

  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }

  // Fallback: non-browser environments won't render, but should not crash.
  const canvas = { width, height } as unknown as HTMLCanvasElement;
  return canvas;
}

function createSeededRandom(seed: number): () => number {
  let randomState = seed >>> 0;
  return () => {
    // xorshift32
    randomState ^= randomState << 13;
    randomState >>>= 0;
    randomState ^= randomState >> 17;
    randomState >>>= 0;
    randomState ^= randomState << 5;
    randomState >>>= 0;
    return randomState / 0x1_0000_0000;
  };
}

function positiveModulo(value: number, modulo: number): number {
  const remainder = value % modulo;
  return remainder < 0 ? remainder + modulo : remainder;
}

/**
 * Draws a simplified neon outline around a pipe rectangle.
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
  context.save();
  // Step 1: Guard degenerate rectangles.
  if (rectangleWidthPx <= 0 || rectangleHeightPx <= 0) {
    context.restore();
    return;
  }

  const alignedLeftPx = Math.round(rectangleLeftPx);
  const alignedTopPx = Math.round(rectangleTopPx);
  const alignedWidthPx = Math.max(1, Math.round(rectangleWidthPx));
  const alignedHeightPx = Math.max(1, Math.round(rectangleHeightPx));

  // Step 2: Compute outline geometry.
  // We want a small side gap and a larger "entrance" gap (top for bottom pipe,
  // bottom for top pipe) to suggest a pipe rim.
  const outlineStrokeWidthPx = FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX;
  const outlineStrokeHalfPx = outlineStrokeWidthPx / 2;
  const sideExpandPx = Math.round(
    FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX + outlineStrokeHalfPx,
  );
  const entranceExpandPx = Math.round(
    FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX + outlineStrokeHalfPx,
  );

  const isTopPipeSegment = alignedTopPx === 0;
  const outlineTopExpandPx = isTopPipeSegment ? sideExpandPx : entranceExpandPx;
  const outlineBottomExpandPx = isTopPipeSegment
    ? entranceExpandPx
    : sideExpandPx;

  const outlineLeftPx = alignedLeftPx - sideExpandPx;
  const outlineTopPx = alignedTopPx - outlineTopExpandPx;
  const outlineWidthPx = alignedWidthPx + sideExpandPx * 2;
  const outlineHeightPx =
    alignedHeightPx + outlineTopExpandPx + outlineBottomExpandPx;

  // Step 3: Define pixel-aligned stroke helper to avoid blurry edges.
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

  // Step 4: Draw neon-green outline with cyan neon glow.
  const previousCompositeOperation = context.globalCompositeOperation;
  context.globalCompositeOperation = 'lighter';
  context.strokeStyle = FLAPPY_NEON_PALETTE.pipeFill;

  // Step 4.1: Soft glow pass (thicker stroke + cyan shadow).
  context.globalAlpha = FLAPPY_PIPE_OUTLINE_GLOW_ALPHA;
  context.shadowColor = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR;
  context.shadowBlur = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX;
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  strokeAlignedRect(
    outlineLeftPx,
    outlineTopPx,
    outlineWidthPx,
    outlineHeightPx,
    FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX,
  );

  // Step 4.2: Crisp outline pass (no shadow).
  context.globalAlpha = 1;
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  strokeAlignedRect(
    outlineLeftPx,
    outlineTopPx,
    outlineWidthPx,
    outlineHeightPx,
    outlineStrokeWidthPx,
  );

  context.globalCompositeOperation = previousCompositeOperation;
  context.restore();
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
  baseOpacity: number,
  edgeBounds: {
    leftXPx: number;
    rightXPx: number;
    topYPx: number;
    bottomYPx: number;
  },
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
  let previousFrameOffset = firstTrailFrameOffset;
  const maxTrailFrameOffset = Math.max(1, firstTrailFrameOffset);

  // Step 2: Configure trail stroke style.
  const previousGlobalAlpha = context.globalAlpha;
  context.strokeStyle = color;
  context.lineWidth = FLAPPY_TRAIL_LINE_WIDTH_PX;

  // Step 3: Render stepped segments with edge-proximity alpha fading.
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
    drawTrailSegmentWithEdgeFade(
      context,
      previousXPosition,
      previousYPosition,
      steppedHorizontalXPosition,
      previousYPosition,
      baseOpacity,
      edgeBounds,
      previousFrameOffset,
      frameOffset,
      maxTrailFrameOffset,
    );
    previousXPosition = steppedHorizontalXPosition;

    const verticalDeltaPx = nextYPosition - previousYPosition;
    if (verticalDeltaPx !== 0) {
      const verticalDirection = Math.sign(verticalDeltaPx);
      const steppedVerticalLengthPx = Math.max(
        Math.abs(verticalDeltaPx),
        FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX,
      );
      const steppedVerticalYPosition =
        previousYPosition + verticalDirection * steppedVerticalLengthPx;
      drawTrailSegmentWithEdgeFade(
        context,
        previousXPosition,
        previousYPosition,
        steppedHorizontalXPosition,
        steppedVerticalYPosition,
        baseOpacity,
        edgeBounds,
        previousFrameOffset,
        frameOffset,
        maxTrailFrameOffset,
      );
      previousYPosition = steppedVerticalYPosition;
    }

    drawTrailSegmentWithEdgeFade(
      context,
      previousXPosition,
      previousYPosition,
      steppedHorizontalXPosition,
      nextYPosition,
      baseOpacity,
      edgeBounds,
      previousFrameOffset,
      frameOffset,
      maxTrailFrameOffset,
    );
    previousYPosition = nextYPosition;

    drawTrailSegmentWithEdgeFade(
      context,
      previousXPosition,
      previousYPosition,
      nextXPosition,
      nextYPosition,
      baseOpacity,
      edgeBounds,
      previousFrameOffset,
      frameOffset,
      maxTrailFrameOffset,
    );
    previousXPosition = nextXPosition;
    previousYPosition = nextYPosition;
    previousFrameOffset = frameOffset;
  });

  // Step 4: Restore caller alpha state.
  context.globalAlpha = previousGlobalAlpha;
}

/**
 * Draws one trail segment with edge-aware alpha attenuation.
 *
 * @param context - Canvas 2D context.
 * @param startXPx - Segment start x position.
 * @param startYPx - Segment start y position.
 * @param endXPx - Segment end x position.
 * @param endYPx - Segment end y position.
 * @param baseOpacity - Base trail opacity before edge fading.
 * @param edgeBounds - Visible world bounds used for edge distance checks.
 * @param startFrameOffset - Age offset at segment start (frames from latest).
 * @param endFrameOffset - Age offset at segment end (frames from latest).
 * @param maxTrailFrameOffset - Oldest age offset currently retained by the trail.
 * @returns Nothing.
 */
function drawTrailSegmentWithEdgeFade(
  context: CanvasRenderingContext2D,
  startXPx: number,
  startYPx: number,
  endXPx: number,
  endYPx: number,
  baseOpacity: number,
  edgeBounds: {
    leftXPx: number;
    rightXPx: number;
    topYPx: number;
    bottomYPx: number;
  },
  startFrameOffset: number,
  endFrameOffset: number,
  maxTrailFrameOffset: number,
): void {
  const segmentLengthPx = Math.hypot(endXPx - startXPx, endYPx - startYPx);
  if (segmentLengthPx === 0 || baseOpacity <= 0) {
    return;
  }

  // Step 1: Resolve edge-fade factor from both segment endpoints.
  const startOpacityFactor = resolveEdgeOpacityFactor(
    startXPx,
    startYPx,
    edgeBounds,
  );
  const endOpacityFactor = resolveEdgeOpacityFactor(endXPx, endYPx, edgeBounds);
  const edgeOpacityFactor = Math.min(startOpacityFactor, endOpacityFactor);

  // Step 2: Resolve lifetime fade so older trail history fades near cutoff.
  const startLifetimeOpacityFactor = resolveTrailLifetimeOpacityFactor(
    startFrameOffset,
    maxTrailFrameOffset,
  );
  const endLifetimeOpacityFactor = resolveTrailLifetimeOpacityFactor(
    endFrameOffset,
    maxTrailFrameOffset,
  );
  const lifetimeOpacityFactor = Math.min(
    startLifetimeOpacityFactor,
    endLifetimeOpacityFactor,
  );

  const segmentOpacity =
    baseOpacity * edgeOpacityFactor * lifetimeOpacityFactor;
  if (segmentOpacity <= 0) {
    return;
  }

  // Step 3: Draw the segment with resolved opacity.
  context.globalAlpha = segmentOpacity;
  context.beginPath();
  context.moveTo(startXPx, startYPx);
  context.lineTo(endXPx, endYPx);
  context.stroke();
}

/**
 * Converts distance-to-edge into a normalized opacity factor.
 *
 * Returns 0 exactly on or beyond an edge and rises to 1 once distance exceeds
 * the configured fade band.
 *
 * @param pointXPx - Point x position.
 * @param pointYPx - Point y position.
 * @param edgeBounds - Visible world bounds used for edge distance checks.
 * @returns Opacity multiplier in [0, 1].
 */
function resolveEdgeOpacityFactor(
  pointXPx: number,
  pointYPx: number,
  edgeBounds: {
    leftXPx: number;
    rightXPx: number;
    topYPx: number;
    bottomYPx: number;
  },
): number {
  const distanceToLeftEdgePx = pointXPx - edgeBounds.leftXPx;
  const distanceToRightEdgePx = edgeBounds.rightXPx - pointXPx;
  const distanceToTopEdgePx = pointYPx - edgeBounds.topYPx;
  const distanceToBottomEdgePx = edgeBounds.bottomYPx - pointYPx;

  const nearestEdgeDistancePx = Math.min(
    distanceToLeftEdgePx,
    distanceToRightEdgePx,
    distanceToTopEdgePx,
    distanceToBottomEdgePx,
  );
  const fadeProgress =
    nearestEdgeDistancePx / FLAPPY_TRAIL_EDGE_FADE_DISTANCE_PX;
  return clamp01(fadeProgress);
}

/**
 * Converts trail age into a normalized opacity factor.
 *
 * Oldest retained history approaches 0 opacity; newest approaches 1.
 *
 * @param frameOffset - Frames between this point and newest trail point.
 * @param maxTrailFrameOffset - Oldest age offset currently retained by trail.
 * @returns Opacity multiplier in [0, 1].
 */
function resolveTrailLifetimeOpacityFactor(
  frameOffset: number,
  maxTrailFrameOffset: number,
): number {
  const normalizedLifetimeProgress =
    1 - frameOffset / Math.max(1, maxTrailFrameOffset);
  return clamp01(normalizedLifetimeProgress);
}

/**
 * Clamps a number to the inclusive [0, 1] range.
 *
 * @param value - Candidate value.
 * @returns Clamped value.
 */
function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
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
