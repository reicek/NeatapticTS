import { Neat, methods } from '../../../src/neataptic';
import Architect from '../../../src/architecture/architect';
import type Network from '../../../src/architecture/network';
import { evaluateFlappyFitness, rolloutEpisode } from './flappyEvaluation';
import { createXorshift32 } from './rng';
import {
  clamp,
  createBirdColor,
  hasAliveBirds,
  resolveDifficultyProfile,
  resolveFlapDecision,
  resolveFramePrimaryWinnerIndex as resolveFramePrimaryWinnerIndexFromUtils,
  resolveLeaderPipesPassed as resolveLeaderPipesPassedFromUtils,
  resolveNextSpawnGapCenterY,
  resolveNextSpawnGapSize,
  resolveNextSpawnIntervalFrames,
  resolveObservationVector,
  resolvePipeSpawnXPx,
  resolveVisibleWorldWidthPx,
  resolveWorldViewport,
  sampleGapCenterY,
} from './browser-entry.utils';
import {
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
  FLAPPY_NETWORK_OUTPUT_SIZE,
  FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
  FLAPPY_GRAVITY_PX_PER_FRAME2,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_TRAIL_OPACITY_FACTOR,
} from './constants';

/** Runtime window contract for the Flappy Bird browser demo. */
interface RuntimeWindow extends Window {
  flappyBird?: {
    start?: typeof start;
    _autoStarted?: boolean;
    [key: string]: unknown;
  };
  flappyBirdStart?: (containerElement?: unknown) => unknown;
}

/** Default host container id for the demo. */
const DEFAULT_CONTAINER_ID = 'flappy-bird-output';

/** Emulation speed multiplier for browser playback (1.5 => 50% faster). */
const FLAPPY_EMULATION_SPEED_MULTIPLIER = 1.5;

/** Screen-edge padding between viewport border and demo frame. */
const FLAPPY_SCREEN_PADDING_PX = 24;

/** TRON-like neon palette matching asciiMaze style. */
const FLAPPY_NEON_PALETTE = {
  background: '#060b14',
  pipeFill: '#00e5ff',
  pipeEdge: '#0fb5ff',
  leaderRing: '#9fffff',
  trail: '#00e5ff',
  currentRunText: '#00ff66',
  bestRunText: '#ff9a2e',
  hudText: '#9fdcff',
  hudAccent: '#ff9a2e',
  hudPanelBackground: '#000000',
  hudPanelBorder: '#0fb5ff',
} as const;

const FLAPPY_STATS_KEYS = [
  'currentHeader',
  'currentFrames',
  'currentPipes',
  'currentMaxFrames',
  'currentMaxPipes',
  'currentArchitecture',
  'bestHeader',
  'bestFrames',
  'bestPipes',
  'bestMaxFrames',
  'bestMaxPipes',
  'bestArchitecture',
  'status',
] as const;

type FlappyStatsKey = (typeof FLAPPY_STATS_KEYS)[number];

type FlappyStatsTableCells = Record<FlappyStatsKey, HTMLTableCellElement>;

interface PlaybackFrameStats {
  frameIndex: number;
  leaderPipesPassed: number;
  leaderFramesSurvived: number;
}

/**
 * Handle returned by {@link start}.
 * Exposes a minimal stop/isRunning API for embedders.
 */
export interface FlappyBirdRunHandle {
  stop: () => void;
  isRunning: () => boolean;
  done: Promise<void>;
}

type BirdDoneReason = 'collision' | 'out_of_bounds';

interface PopulationPipe {
  id: number;
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

interface PopulationBird {
  network: Network;
  color: string;
  yPx: number;
  velocityYPxPerFrame: number;
  pipesPassed: number;
  framesSurvived: number;
  passedPipeIds: Set<number>;
  done: boolean;
  doneReason?: BirdDoneReason;
}

interface PopulationRenderState {
  frameIndex: number;
  visibleWorldWidthPx: number;
  nextPipeId: number;
  lastSpawnedPipeGapPx: number;
  lastSpawnedPipeGapCenterYPx: number;
  lastSpawnedPipeSpawnIntervalFrames: number;
  framesUntilNextPipeSpawn: number;
  pipes: PopulationPipe[];
  birds: PopulationBird[];
}

interface TrailState {
  birdTrailsY: number[][];
}

/**
 * Start the Flappy Bird NEAT demo in the browser.
 *
 * This is intentionally minimal: it continuously evolves generations and
 * animates the current best genome on a canvas.
 *
 * @param container - Element id or HTMLElement to host the demo.
 * @returns A small run handle for stopping the loop.
 */
export const start = async (
  container: string | HTMLElement = DEFAULT_CONTAINER_ID,
): Promise<FlappyBirdRunHandle> => {
  const hostElement =
    typeof container === 'string'
      ? document.getElementById(container)
      : container;

  if (!hostElement) {
    throw new Error(`Flappy demo container not found: ${String(container)}`);
  }

  // Smallish defaults so the demo stays responsive in the browser.
  const inputSize = FLAPPY_NETWORK_INPUT_SIZE;
  const outputSize = FLAPPY_NETWORK_OUTPUT_SIZE;
  const populationSize = 200;
  const elitismCount = 20;

  const { canvas, context, statsValueByKey } = createCanvasHost(hostElement);

  const neat = new Neat(inputSize, outputSize, () => 0, {
    popsize: populationSize,
    elitism: elitismCount,
    mutationRate: 0.75,
    mutationAmount: 2,
    mutation: methods.mutation.FFW,
    network: Architect.perceptron(
      inputSize,
      ...FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
      outputSize,
    ),
    speciation: true,
    multiObjective: { enabled: false },
    novelty: { enabled: false },
  });

  neat.fitness = (network) => evaluateFlappyFitness(network);

  // Deterministic run baseline.
  neat.restoreRNGState(0x1234abcd);

  let stopped = false;
  let resolveDone: (() => void) | undefined;
  const done = new Promise<void>((resolve) => {
    resolveDone = resolve;
  });

  const stop = () => {
    stopped = true;
    resolveDone?.();
  };

  const isRunning = () => !stopped;

  // Run evolution + render loop.
  (async () => {
    let lastBestNetwork: Network | undefined = undefined;
    let bestRunFrames = 0;
    let bestRunPipes = 0;
    let bestRunArchitecture = '-';

    while (!stopped) {
      const best = await neat.evolve();
      lastBestNetwork = best as Network;

      const episode = rolloutEpisode(lastBestNetwork);
      const bestFitness = (best.score ?? episode.fitness) as number;
      bestRunFrames = Math.max(bestRunFrames, episode.framesSurvived);
      bestRunPipes = Math.max(bestRunPipes, episode.pipesPassed);
      bestRunArchitecture = resolveNetworkArchitectureLabel(
        lastBestNetwork,
        inputSize,
        outputSize,
      );
      const generationPopulation = resolveGenerationPopulation(
        neat,
        lastBestNetwork,
      );
      updateStatsTableValues(statsValueByKey, {
        currentHeader: `Current run · Gen ${neat.generation}`,
        currentFrames: '0',
        currentPipes: '0',
        currentMaxFrames: '0',
        currentMaxPipes: '0',
        currentArchitecture: resolveNetworkArchitectureLabel(
          lastBestNetwork,
          inputSize,
          outputSize,
        ),
        bestFrames: String(episode.framesSurvived),
        bestPipes: String(episode.pipesPassed),
        bestMaxFrames: String(bestRunFrames),
        bestMaxPipes: String(bestRunPipes),
        bestArchitecture: bestRunArchitecture,
        bestHeader: 'Best run',
        status: 'playing',
      });
      const playbackSummary = await animatePopulationEpisode(
        canvas,
        context,
        generationPopulation,
        (frameStats) => {
          updateStatsTableValues(statsValueByKey, {
            currentFrames: String(frameStats.frameIndex),
            currentPipes: String(frameStats.leaderPipesPassed),
            currentMaxFrames: String(frameStats.leaderFramesSurvived),
            currentMaxPipes: String(frameStats.leaderPipesPassed),
          });
        },
      );
      seedNextGenerationWithPlaybackWinner(neat, playbackSummary.winnerNetwork);

      updateStatsTableValues(statsValueByKey, { status: 'evolving' });
      try {
        console.log(
          `[flappy_bird] gen=${neat.generation} fitness=${bestFitness.toFixed(0)} pipes=${episode.pipesPassed} frames=${episode.framesSurvived} avgPipes=${playbackSummary.averagePipesPassed.toFixed(2)} p90Frames=${playbackSummary.p90FramesSurvived}`,
        );
      } catch {
        // ignore console write issues
      }

      // Yield to the browser before the next generation.
      await nextAnimationFrame();
    }
  })().catch((error: unknown) => {
    updateStatsTableValues(statsValueByKey, {
      status: `error: ${String((error as Error)?.message ?? error)}`,
    });
    stop();
  });

  return { stop, isRunning, done };

  /**
   * Builds the demo host DOM tree (frame, canvas, and stats table).
   *
   * @param containerElement - Root container that receives the demo UI.
   * @returns Canvas rendering objects and writable stats cell map.
   */
  function createCanvasHost(containerElement: HTMLElement): {
    canvas: HTMLCanvasElement;
    context: CanvasRenderingContext2D;
    statsValueByKey: FlappyStatsTableCells;
  } {
    // Step 1: Ensure container is empty-ish.
    containerElement.innerHTML = '';

    const outerFrame = document.createElement('section');
    outerFrame.style.width = '100%';
    outerFrame.style.height = '100%';
    outerFrame.style.boxSizing = 'border-box';
    outerFrame.style.padding = `${FLAPPY_SCREEN_PADDING_PX}px`;
    outerFrame.style.border = `4px double ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
    outerFrame.style.background = '#050a12';
    outerFrame.style.boxShadow =
      '0 0 0 1px rgba(159,220,255,0.35), inset 0 0 0 1px rgba(159,220,255,0.2), 0 0 26px rgba(15,181,255,0.18)';

    const contentColumn = document.createElement('div');
    contentColumn.style.width = '100%';
    contentColumn.style.height = '100%';
    contentColumn.style.display = 'flex';
    contentColumn.style.flexDirection = 'column';
    contentColumn.style.gap = '8px';
    contentColumn.style.alignItems = 'stretch';
    contentColumn.style.overflow = 'hidden';

    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    canvas.style.width = '100%';
    canvas.style.height = '1px';
    canvas.style.display = 'block';
    canvas.style.maxWidth = '100%';
    canvas.style.boxSizing = 'border-box';
    canvas.style.border = '1px solid #0fb5ff';
    canvas.style.background = FLAPPY_NEON_PALETTE.background;

    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas 2D context unavailable');

    const statsContainer = document.createElement('div');
    statsContainer.style.marginTop = '8px';
    statsContainer.style.boxSizing = 'border-box';
    statsContainer.style.border = `1px solid ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
    statsContainer.style.borderBottom = `1px solid ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
    statsContainer.style.background = FLAPPY_NEON_PALETTE.hudPanelBackground;
    statsContainer.style.boxShadow = '0 0 14px rgba(15,181,255,0.22) inset';
    statsContainer.style.padding = '8px 10px';
    statsContainer.style.overflow = 'hidden';

    const statsTable = document.createElement('table');
    statsTable.style.borderCollapse = 'collapse';
    statsTable.style.width = '100%';
    statsTable.style.maxWidth = '100%';
    statsTable.style.tableLayout = 'fixed';
    statsTable.style.fontFamily = 'Consolas, Menlo, Monaco, monospace';
    statsTable.style.fontSize = '12px';
    statsTable.style.borderBottom = '1px solid rgba(15,181,255,0.2)';

    const statsValueByKey = createStatsTableRows(statsTable);
    updateStatsTableValues(statsValueByKey, {
      currentHeader: 'Current run · Gen -',
      currentFrames: '0',
      currentPipes: '0',
      currentMaxFrames: '0',
      currentMaxPipes: '0',
      currentArchitecture: '-',
      bestHeader: 'Best run',
      bestFrames: '0',
      bestPipes: '0',
      bestMaxFrames: '0',
      bestMaxPipes: '0',
      bestArchitecture: '-',
      status: 'initializing',
    });
    statsContainer.appendChild(statsTable);

    const statsBottomRule = document.createElement('div');
    statsBottomRule.style.height = '1px';
    statsBottomRule.style.marginTop = '6px';
    statsBottomRule.style.background = FLAPPY_NEON_PALETTE.hudPanelBorder;
    statsContainer.appendChild(statsBottomRule);

    contentColumn.appendChild(canvas);
    contentColumn.appendChild(statsContainer);
    outerFrame.appendChild(contentColumn);
    containerElement.appendChild(outerFrame);

    installResponsiveViewportSizing(canvas, contentColumn, statsContainer);

    return { canvas, context, statsValueByKey };
  }
};

/**
 * Keeps the canvas backing resolution synchronized with container size.
 *
 * @param canvas - Simulation canvas to resize.
 * @param containerElement - Width source for responsive sizing.
 * @returns Nothing.
 */
function installResponsiveViewportSizing(
  canvas: HTMLCanvasElement,
  containerElement: HTMLElement,
  statsContainer: HTMLElement,
): void {
  const applyCanvasSize = (): void => {
    const availableWidthPx = Math.max(1, containerElement.clientWidth);
    const availableHeightPx = Math.max(
      1,
      containerElement.clientHeight - statsContainer.offsetHeight - 8,
    );

    const nextWidthPx = Math.max(1, Math.floor(availableWidthPx));
    const nextHeightPx = Math.max(1, Math.floor(availableHeightPx));

    if (canvas.width !== nextWidthPx || canvas.height !== nextHeightPx) {
      canvas.width = nextWidthPx;
      canvas.height = nextHeightPx;
      canvas.style.width = `${nextWidthPx}px`;
      canvas.style.height = `${nextHeightPx}px`;
    }
  };

  applyCanvasSize();
  window.addEventListener('resize', applyCanvasSize);

  if (typeof ResizeObserver === 'function') {
    const resizeObserver = new ResizeObserver(() => applyCanvasSize());
    resizeObserver.observe(containerElement);
  }
}

/**
 * Renders all birds from the current generation in one shared world.
 */
async function animatePopulationEpisode(
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  networks: Network[],
  onFrameStats: (stats: PlaybackFrameStats) => void,
): Promise<{
  averagePipesPassed: number;
  p90FramesSurvived: number;
  winnerNetwork?: Network;
}> {
  const rng = createXorshift32(0xabcdef01);
  const renderState = createPopulationRenderState(
    networks,
    rng,
    resolveVisibleWorldWidthPx(canvas),
  );
  const trailState: TrailState = {
    birdTrailsY: renderState.birds.map(() => []),
  };
  let simulationFrameBudget = 0;

  while (hasAliveBirds(renderState.birds)) {
    renderState.visibleWorldWidthPx = resolveVisibleWorldWidthPx(canvas);
    simulationFrameBudget += FLAPPY_EMULATION_SPEED_MULTIPLIER;
    const simulationStepsThisRender = Math.max(
      1,
      Math.floor(simulationFrameBudget),
    );
    simulationFrameBudget -= simulationStepsThisRender;

    for (
      let simulationStepIndex = 0;
      simulationStepIndex < simulationStepsThisRender && hasAliveBirds(renderState.birds);
      simulationStepIndex++
    ) {
      const difficultyProfile = resolveDifficultyProfile(
        resolveLeaderPipesPassedFromUtils(renderState.birds),
      );
      stepPopulationFrame(renderState, rng, difficultyProfile);

      updateTrailState(trailState, renderState);

      const leaderPipesPassed = resolveLeaderPipesPassedFromUtils(
        renderState.birds,
      );
      const leaderFramesSurvived = resolveLeaderFramesSurvived(renderState);
      onFrameStats({
        frameIndex: renderState.frameIndex,
        leaderPipesPassed,
        leaderFramesSurvived,
      });
    }

    renderPopulationFrame(context, renderState, trailState);
    await nextAnimationFrame();
  }

  renderPopulationFrame(context, renderState, trailState);

  const winnerBirdIndex = resolveWinnerBirdIndex(renderState);
  const winnerNetwork =
    winnerBirdIndex >= 0 ? renderState.birds[winnerBirdIndex]?.network : undefined;
  const averagePipesPassed =
    renderState.birds.reduce(
      (totalPipesPassed, bird) => totalPipesPassed + bird.pipesPassed,
      0,
    ) / Math.max(1, renderState.birds.length);
  const sortedFramesSurvived = renderState.birds
    .map((bird) => bird.framesSurvived)
    .toSorted((leftFrames, rightFrames) => leftFrames - rightFrames);
  const p90FrameIndex = Math.min(
    sortedFramesSurvived.length - 1,
    Math.floor(sortedFramesSurvived.length * 0.9),
  );
  const p90FramesSurvived =
    sortedFramesSurvived.length > 0
      ? sortedFramesSurvived[p90FrameIndex]
      : 0;

  return {
    averagePipesPassed,
    p90FramesSurvived,
    winnerNetwork,
  };
}

/**
 * Seeds the next generation by injecting the playback winner clone.
 */
function seedNextGenerationWithPlaybackWinner(
  neat: Neat,
  winnerNetwork: Network | undefined,
): void {
  if (!winnerNetwork) return;

  const runtimeNeat = neat as unknown as { population?: Network[] };
  if (!Array.isArray(runtimeNeat.population) || runtimeNeat.population.length === 0) {
    return;
  }

  runtimeNeat.population[0] = winnerNetwork.clone();
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
  const viewport = resolveWorldViewport(context.canvas);
  const cameraLeftPx = 0;

  // Step 1: Clear.
  context.clearRect(0, 0, context.canvas.width, context.canvas.height);

  // Step 2: Draw world left-aligned in a dynamic viewport (no stretching).
  context.save();
  context.translate(viewport.offsetXPx, viewport.offsetYPx);
  context.scale(viewport.scale, viewport.scale);
  context.translate(-cameraLeftPx, 0);

  // Step 3: Draw pipes.
  for (const pipe of renderState.pipes) {
    const gapHalf = pipe.gapSizePx * 0.5;
    const pipeLeft = pipe.xPx;
    const gapTop = pipe.gapCenterYPx - gapHalf;
    const gapBottom = pipe.gapCenterYPx + gapHalf;

    context.fillStyle = FLAPPY_NEON_PALETTE.pipeFill;
    context.strokeStyle = FLAPPY_NEON_PALETTE.pipeEdge;
    context.lineWidth = 2;
    // Top pipe
    context.fillRect(pipeLeft, 0, FLAPPY_PIPE_WIDTH_PX, gapTop);
    context.strokeRect(pipeLeft, 0, FLAPPY_PIPE_WIDTH_PX, gapTop);
    // Bottom pipe
    context.fillRect(
      pipeLeft,
      gapBottom,
      FLAPPY_PIPE_WIDTH_PX,
      FLAPPY_WORLD_HEIGHT_PX - gapBottom,
    );
    context.strokeRect(
      pipeLeft,
      gapBottom,
      FLAPPY_PIPE_WIDTH_PX,
      FLAPPY_WORLD_HEIGHT_PX - gapBottom,
    );
  }

  // Step 4: Draw all birds with unique colors.
  const leaderBirdIndex = resolveLeaderBirdIndex(renderState);
  renderState.birds.forEach((bird, birdIndex) => {
    if (bird.done) return;

    const birdOpacity = birdIndex === leaderBirdIndex ? 1 : 0.1;
    context.globalAlpha = birdOpacity;
    context.fillStyle = bird.color;
    context.beginPath();
    context.arc(
      FLAPPY_BIRD_X_PX,
      bird.yPx,
      FLAPPY_BIRD_RADIUS_PX,
      0,
      Math.PI * 2,
    );
    context.fill();

    if (birdIndex === leaderBirdIndex && !bird.done) {
      context.strokeStyle = FLAPPY_NEON_PALETTE.leaderRing;
      context.lineWidth = 2;
      context.beginPath();
      context.arc(
        FLAPPY_BIRD_X_PX,
        bird.yPx,
        FLAPPY_BIRD_RADIUS_PX + 2,
        0,
        Math.PI * 2,
      );
      context.stroke();
    }
  });

  // Step 5: Draw per-bird trails using matching bird color and opacity.
  renderState.birds.forEach((bird, birdIndex) => {
    if (bird.done) return;
    const birdTrailPoints = trailState.birdTrailsY[birdIndex];
    if (!birdTrailPoints || birdTrailPoints.length === 0) return;

    const parentBirdOpacity = birdIndex === leaderBirdIndex ? 1 : 0.1;
    context.globalAlpha = parentBirdOpacity * FLAPPY_TRAIL_OPACITY_FACTOR;
    drawTrail(
      context,
      birdTrailPoints,
      bird.color,
      FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX,
    );
  });
  context.globalAlpha = 1;
  context.restore();
}

/**
 * Creates aligned key/value rows for the neon stats panel.
 *
 * @param statsTable - Table element that receives the rows.
 * @returns Map of stats keys to writable value cells.
 */
function createStatsTableRows(statsTable: HTMLTableElement): FlappyStatsTableCells {
  const statsValueByKey = {} as FlappyStatsTableCells;

  const statsRows: Array<{ key: FlappyStatsKey; label: string }> = [
    { key: 'currentHeader', label: 'Current run' },
    { key: 'currentFrames', label: 'Frames' },
    { key: 'currentPipes', label: 'Pipes' },
    { key: 'currentMaxFrames', label: 'Max frames' },
    { key: 'currentMaxPipes', label: 'Max pipes' },
    { key: 'currentArchitecture', label: 'NN architecture' },
    { key: 'bestHeader', label: 'Best run' },
    { key: 'bestFrames', label: 'Frames' },
    { key: 'bestPipes', label: 'Pipes' },
    { key: 'bestMaxFrames', label: 'Max frames' },
    { key: 'bestMaxPipes', label: 'Max pipes' },
    { key: 'bestArchitecture', label: 'NN architecture' },
    { key: 'status', label: 'Status' },
  ];

  const resolveCategoryColor = (
    statsKey: FlappyStatsKey,
  ): { keyColor: string; valueColor: string } => {
    if (statsKey.startsWith('current')) {
      return {
        keyColor: FLAPPY_NEON_PALETTE.currentRunText,
        valueColor: FLAPPY_NEON_PALETTE.currentRunText,
      };
    }
    if (statsKey.startsWith('best')) {
      return {
        keyColor: FLAPPY_NEON_PALETTE.bestRunText,
        valueColor: FLAPPY_NEON_PALETTE.bestRunText,
      };
    }
    return {
      keyColor: FLAPPY_NEON_PALETTE.hudText,
      valueColor: FLAPPY_NEON_PALETTE.hudAccent,
    };
  };

  statsRows.forEach((statsRow) => {
    const rowElement = document.createElement('tr');
    const { keyColor, valueColor } = resolveCategoryColor(statsRow.key);
    const isSectionHeader =
      statsRow.key === 'currentHeader' || statsRow.key === 'bestHeader';

    if (isSectionHeader) {
      const sectionCell = document.createElement('th');
      sectionCell.colSpan = 2;
      sectionCell.textContent = statsRow.label;
      sectionCell.style.color = keyColor;
      sectionCell.style.textAlign = 'left';
      sectionCell.style.padding = '6px 0 4px 0';
      sectionCell.style.fontWeight = '700';
      sectionCell.style.textTransform = 'uppercase';
      sectionCell.style.letterSpacing = '0.06em';
      sectionCell.style.borderBottom = '1px solid rgba(15,181,255,0.35)';
      rowElement.appendChild(sectionCell);
      statsTable.appendChild(rowElement);
      statsValueByKey[statsRow.key] = sectionCell as unknown as HTMLTableCellElement;
      return;
    }

    const keyCell = document.createElement('th');
    keyCell.textContent = statsRow.label;
    keyCell.style.color = keyColor;
    keyCell.style.textAlign = 'left';
    keyCell.style.padding = '2px 8px 2px 0';
    keyCell.style.fontWeight = '600';
    keyCell.style.textTransform = 'uppercase';
    keyCell.style.letterSpacing = '0.04em';
    keyCell.style.width = '35%';
    keyCell.style.borderBottom = '1px solid rgba(15,181,255,0.2)';

    const valueCell = document.createElement('td');
    valueCell.textContent = '-';
    valueCell.style.color = valueColor;
    valueCell.style.textAlign = 'left';
    valueCell.style.padding = '2px 0';
    valueCell.style.borderBottom = '1px solid rgba(15,181,255,0.2)';
    valueCell.style.whiteSpace = 'nowrap';
    valueCell.style.overflow = 'hidden';
    valueCell.style.textOverflow = 'ellipsis';

    rowElement.appendChild(keyCell);
    rowElement.appendChild(valueCell);
    statsTable.appendChild(rowElement);
    statsValueByKey[statsRow.key] = valueCell;
  });

  return statsValueByKey;
}

/**
 * Applies partial stat updates to the rendered stats table.
 *
 * @param statsValueByKey - Lookup of stat keys to value cells.
 * @param partialValues - Subset of values to write this tick.
 * @returns Nothing.
 */
function updateStatsTableValues(
  statsValueByKey: FlappyStatsTableCells,
  partialValues: Partial<Record<FlappyStatsKey, string>>,
): void {
  FLAPPY_STATS_KEYS.forEach((statsKey) => {
    const nextValue = partialValues[statsKey];
    if (nextValue == null) return;
    statsValueByKey[statsKey].textContent = nextValue;
  });
}

/**
 * Advances the full population simulation by one logical frame.
 *
 * @param renderState - Mutable population simulation state.
 * @param rng - Deterministic random generator used for pipe sampling.
 * @param difficultyProfile - Active difficulty values for this frame.
 * @returns Nothing.
 */
function stepPopulationFrame(
  renderState: PopulationRenderState,
  rng: ReturnType<typeof createXorshift32>,
  difficultyProfile: {
    pipeGapPx: number;
    pipeSpeedPxPerFrame: number;
    pipeSpawnIntervalFrames: number;
  },
): void {
  const controlSubstepCount = Math.max(1, FLAPPY_CONTROL_SUBSTEPS_PER_FRAME);
  const controlSubstepDelta = 1 / controlSubstepCount;

  // Step 1: Mark frame survival for all alive birds.
  renderState.birds.forEach((bird) => {
    if (bird.done) return;
    bird.framesSurvived += 1;
  });

  // Step 2: Run high-frequency control/physics substeps.
  for (
    let controlSubstepIndex = 0;
    controlSubstepIndex < controlSubstepCount;
    controlSubstepIndex++
  ) {
    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      const observation = resolveObservationVector(
        bird.yPx,
        bird.velocityYPxPerFrame,
        renderState.pipes,
        renderState.visibleWorldWidthPx,
        difficultyProfile,
        renderState.lastSpawnedPipeSpawnIntervalFrames,
      );
      const outputs = bird.network.activate(observation) as unknown;
      const shouldFlap = resolveFlapDecision(outputs);

      if (shouldFlap) {
        bird.velocityYPxPerFrame = FLAPPY_FLAP_VELOCITY_PX_PER_FRAME;
      }
    });

    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      bird.velocityYPxPerFrame = clamp(
        bird.velocityYPxPerFrame +
          FLAPPY_GRAVITY_PX_PER_FRAME2 * controlSubstepDelta,
        -Infinity,
        FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
      );
      bird.yPx += bird.velocityYPxPerFrame * controlSubstepDelta;
    });

    renderState.pipes.forEach((pipe) => {
      pipe.xPx -= difficultyProfile.pipeSpeedPxPerFrame * controlSubstepDelta;
    });
    renderState.pipes = renderState.pipes.filter(
      (pipe) => pipe.xPx + FLAPPY_PIPE_WIDTH_PX > 0,
    );

    renderState.framesUntilNextPipeSpawn -= controlSubstepDelta;
    if (renderState.framesUntilNextPipeSpawn <= 0) {
      const nextGapSizePx = resolveNextSpawnGapSize(
        renderState.lastSpawnedPipeGapPx,
        difficultyProfile,
        rng,
      );
      const nextSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
        renderState.lastSpawnedPipeSpawnIntervalFrames,
        difficultyProfile,
      );
      const nextGapCenterYPx = resolveNextSpawnGapCenterY(
        renderState.lastSpawnedPipeGapCenterYPx,
        rng,
      );
      renderState.pipes.push({
        id: renderState.nextPipeId++,
        xPx: resolvePipeSpawnXPx(renderState.visibleWorldWidthPx),
        gapCenterYPx: nextGapCenterYPx,
        gapSizePx: nextGapSizePx,
      });
      renderState.lastSpawnedPipeGapPx = nextGapSizePx;
      renderState.lastSpawnedPipeGapCenterYPx = nextGapCenterYPx;
      renderState.lastSpawnedPipeSpawnIntervalFrames = nextSpawnIntervalFrames;
      renderState.framesUntilNextPipeSpawn += nextSpawnIntervalFrames;
    }

    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      const birdTop = bird.yPx - FLAPPY_BIRD_RADIUS_PX;
      const birdBottom = bird.yPx + FLAPPY_BIRD_RADIUS_PX;

      if (birdTop <= 0 || birdBottom >= FLAPPY_WORLD_HEIGHT_PX) {
        bird.done = true;
        bird.doneReason = 'out_of_bounds';
        return;
      }

      const birdLeft = FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX;
      const birdRight = FLAPPY_BIRD_X_PX + FLAPPY_BIRD_RADIUS_PX;

      for (const pipe of renderState.pipes) {
        const pipeLeft = pipe.xPx;
        const pipeRight = pipe.xPx + FLAPPY_PIPE_WIDTH_PX;
        const overlapsHorizontally = birdRight >= pipeLeft && birdLeft <= pipeRight;

        if (overlapsHorizontally) {
          const gapHalf = pipe.gapSizePx * 0.5;
          const gapTop = pipe.gapCenterYPx - gapHalf;
          const gapBottom = pipe.gapCenterYPx + gapHalf;
          const isInsideGap = birdTop >= gapTop && birdBottom <= gapBottom;

          if (!isInsideGap) {
            bird.done = true;
            bird.doneReason = 'collision';
            break;
          }
        }

        if (pipeRight < FLAPPY_BIRD_X_PX && !bird.passedPipeIds.has(pipe.id)) {
          bird.passedPipeIds.add(pipe.id);
          bird.pipesPassed += 1;
        }
      }
    });
  }

  // Step 3: Advance frame counter.
  renderState.frameIndex += 1;
}

/**
 * Creates initial simulation state for a generation playback.
 *
 * @param networks - Networks participating in playback.
 * @param rng - Random source for deterministic initial pipe placement.
 * @returns Initialized mutable render state.
 */
function createPopulationRenderState(
  networks: Network[],
  rng: ReturnType<typeof createXorshift32>,
  initialVisibleWorldWidthPx: number,
): PopulationRenderState {
  const initialDifficultyProfile = resolveDifficultyProfile(0);
  const initialGapCenterYPx = sampleGapCenterY(rng);
  const initialGapSizePx = resolveNextSpawnGapSize(
    undefined,
    initialDifficultyProfile,
    rng,
  );
  const initialSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
    undefined,
    initialDifficultyProfile,
  );
  const birds = networks.map((network, networkIndex) => ({
    network,
    color: createBirdColor(networkIndex, networks.length),
    yPx: FLAPPY_WORLD_HEIGHT_PX * 0.5,
    velocityYPxPerFrame: 0,
    pipesPassed: 0,
    framesSurvived: 0,
    passedPipeIds: new Set<number>(),
    done: false,
  }));

  return {
    frameIndex: 0,
    visibleWorldWidthPx: initialVisibleWorldWidthPx,
    nextPipeId: 2,
    lastSpawnedPipeGapPx: initialGapSizePx,
    lastSpawnedPipeGapCenterYPx: initialGapCenterYPx,
    lastSpawnedPipeSpawnIntervalFrames: initialSpawnIntervalFrames,
    framesUntilNextPipeSpawn: initialSpawnIntervalFrames,
    pipes: [
      {
        id: 1,
        xPx: resolvePipeSpawnXPx(initialVisibleWorldWidthPx),
        gapCenterYPx: initialGapCenterYPx,
        gapSizePx: initialGapSizePx,
      },
    ],
    birds,
  };
}

/**
 * Resolves the current generation population with safe fallback.
 *
 * @param neat - Runtime NEAT instance.
 * @param fallbackBest - Champion fallback when population is unavailable.
 * @returns Networks to animate for playback.
 */
function resolveGenerationPopulation(neat: Neat, fallbackBest: Network): Network[] {
  const runtimeNeat = neat as unknown as { population?: Network[] };
  if (Array.isArray(runtimeNeat.population) && runtimeNeat.population.length > 0) {
    return runtimeNeat.population;
  }
  return [fallbackBest];
}

/**
 * Updates the leader trail cache after a simulation step.
 *
 * @param trailState - Mutable trail state buffer.
 * @param renderState - Current population render state.
 * @returns Nothing.
 */
function updateTrailState(
  trailState: TrailState,
  renderState: PopulationRenderState,
): void {
  renderState.birds.forEach((bird, birdIndex) => {
    const birdTrail = trailState.birdTrailsY[birdIndex];
    if (!birdTrail) return;

    if (bird.done) {
      birdTrail.length = 0;
      return;
    }

    pushTrailPoint(birdTrail, bird.yPx);
  });
}

/**
 * Appends a leader trail point and keeps a fixed maximum length.
 *
 * @param trailPoints - Mutable trail y-position collection.
 * @param yPosition - New y-position to append.
 * @returns Nothing.
 */
function pushTrailPoint(trailPoints: number[], yPosition: number): void {
  trailPoints.push(yPosition);
  const maxTrailPoints = 40;
  if (trailPoints.length > maxTrailPoints) {
    trailPoints.splice(0, trailPoints.length - maxTrailPoints);
  }
}

/**
 * Renders the trailing line behind the current leader.
 *
 * @param context - Canvas 2D drawing context.
 * @param trailPoints - Ordered y positions of the recent trail.
 * @param color - Stroke color used for the trail.
 * @param anchorX - Right-side anchor x-position for the latest point.
 * @returns Nothing.
 */
function drawTrail(
  context: CanvasRenderingContext2D,
  trailPoints: number[],
  color: string,
  anchorX: number,
): void {
  if (trailPoints.length === 0) return;

  context.strokeStyle = color;
  context.lineWidth = 1.5;
  context.beginPath();
  trailPoints.forEach((trailY, trailIndex) => {
    const xOffset = trailPoints.length - 1 - trailIndex;
    const xPosition = anchorX - xOffset;
    if (trailIndex === 0) {
      context.moveTo(xPosition, trailY);
      return;
    }
    context.lineTo(xPosition, trailY);
  });
  context.stroke();
}

/**
 * Resolves leader index for rendering (alive birds only).
 *
 * @param renderState - Current population render state.
 * @returns Index of current leader or -1 when unavailable.
 */
function resolveLeaderBirdIndex(renderState: PopulationRenderState): number {
  return resolveFramePrimaryWinnerIndexFromUtils(renderState.birds, true);
}

/**
 * Resolves generation winner index at playback end.
 *
 * @param renderState - Current population render state.
 * @returns Winner index or -1 when unavailable.
 */
function resolveWinnerBirdIndex(renderState: PopulationRenderState): number {
  return resolveFramePrimaryWinnerIndexFromUtils(renderState.birds, false);
}

/**
 * Resolves the highest survived-frame count among birds in current playback.
 *
 * @param renderState - Current population render state.
 * @returns Maximum frames survived value.
 */
function resolveLeaderFramesSurvived(renderState: PopulationRenderState): number {
  return renderState.birds.reduce(
    (maximumFramesSurvived, bird) =>
      Math.max(maximumFramesSurvived, bird.framesSurvived),
    0,
  );
}

/**
 * Produces a concise neural-network architecture label.
 *
 * @param network - Network instance to describe.
 * @param inputSize - Configured input size.
 * @param outputSize - Configured output size.
 * @returns Readable architecture string like `12 | 18 - 10 | 2 (40 nodes, 112 connections)`.
 */
function resolveNetworkArchitectureLabel(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): string {
  if (!network) {
    return formatArchitectureLabel(
      inputSize,
      '-',
      outputSize,
      inputSize + outputSize,
      0,
    );
  }

  const architectureDescriptor = network.describeArchitecture();
  const hiddenLayersLabel = resolveHiddenLayersLabel(
    architectureDescriptor.hiddenLayerSizes,
    architectureDescriptor.source,
  );

  return formatArchitectureLabel(
    inputSize,
    hiddenLayersLabel,
    outputSize,
    architectureDescriptor.totalNodes,
    architectureDescriptor.totalConnections,
  );

  /**
   * @param architectureInputSize - Input-layer size.
   * @param hiddenLayersLabel - Hidden-layer label section.
   * @param architectureOutputSize - Output-layer size.
   * @param totalNodeCount - Total node count shown in suffix.
   * @param totalConnectionCount - Total connection count shown in suffix.
   * @returns Formatted architecture label.
   */
  function formatArchitectureLabel(
    architectureInputSize: number,
    hiddenLayersLabel: string,
    architectureOutputSize: number,
    totalNodeCount: number,
    totalConnectionCount: number,
  ): string {
    return `${architectureInputSize} | ${hiddenLayersLabel} | ${architectureOutputSize} (${totalNodeCount} nodes, ${totalConnectionCount} connections)`;
  }

  /**
   * @param hiddenLayerSizes - Hidden-layer widths.
   * @param architectureSource - Architecture descriptor provenance.
   * @returns Label section for hidden layers.
   */
  function resolveHiddenLayersLabel(
    hiddenLayerSizes: number[],
    architectureSource: 'layer-metadata' | 'graph-topology' | 'inferred',
  ): string {
    if (hiddenLayerSizes.length === 0) {
      return '-';
    }

    if (architectureSource === 'inferred') {
      return hiddenLayerSizes
        .map((hiddenLayerSize) => `~${hiddenLayerSize}`)
        .join(' - ');
    }

    return hiddenLayerSizes.join(' - ');
  }
}

/**
 * Yields execution until the next browser animation frame.
 *
 * @returns Promise that resolves on the next animation frame.
 */
function nextAnimationFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

// Expose a friendly global entry point (mirrors asciiMaze publishing style).
try {
  const runtimeWindow = window as unknown as RuntimeWindow;
  runtimeWindow.flappyBird = runtimeWindow.flappyBird ?? {};
  runtimeWindow.flappyBird.start = start;
  runtimeWindow.flappyBirdStart = (containerElement?: unknown) =>
    start(containerElement as never);

  if (!runtimeWindow.flappyBird._autoStarted) {
    runtimeWindow.flappyBird._autoStarted = true;
    setTimeout(() => {
      try {
        // Best-effort auto-start when loaded via docs/examples.
        start().catch(() => undefined);
      } catch {
        // ignore
      }
    }, 20);
  }
} catch {
  // ignore global wiring failures (non-browser environments).
}
