import type {
  PopulationBird,
  PopulationPipe,
} from './browser-entry.worker.types';

/**
 * Simulation-facing browser contracts shared by playback helpers.
 *
 * These types describe the minimum world state the browser needs while it is
 * reconstructing, rendering, or summarizing worker-produced frames.
 */

/** Bird shape used by utility winner/leader resolver helpers. */
export interface BrowserPopulationBirdLike {
  done: boolean;
  pipesPassed: number;
  framesSurvived: number;
}

/** Pipe shape used by utility observation-vector helpers. */
export interface BrowserPopulationPipeLike {
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

/**
 * Difficulty profile consumed by simulation observation helpers.
 *
 * This bundles the three variables that define how demanding a stretch of the
 * course is: corridor width, pipe speed, and spawn cadence.
 */
export interface BrowserDifficultyProfile {
  pipeGapPx: number;
  pipeSpeedPxPerFrame: number;
  pipeSpawnIntervalFrames: number;
}

/**
 * Mutable render-state model consumed by the population frame renderer.
 *
 * The playback layer incrementally updates this state as worker snapshots
 * arrive, which lets rendering stay deterministic without re-deriving world
 * history from scratch each frame.
 */
export interface PopulationRenderState {
  frameIndex: number;
  cumulativePipeTravelPx: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  nextPipeId: number;
  lastSpawnedPipeGapPx: number;
  lastSpawnedPipeGapCenterYPx: number;
  lastSpawnedPipeSpawnIntervalFrames: number;
  framesUntilNextPipeSpawn: number;
  pipes: PopulationPipe[];
  birds: PopulationBird[];
}

/**
 * Trail point used by playback trail rendering cache.
 *
 * A trail point stores where one bird was at one frame so the UI can draw a
 * short motion history behind active agents.
 */
export interface TrailPoint {
  frameIndex: number;
  yPx: number;
}

/**
 * Mutable trail cache keyed by bird index for frame rendering.
 *
 * This cache exists purely for visualization ergonomics; it is not part of the
 * worker simulation state.
 */
export interface TrailState {
  birdTrailsY: TrailPoint[][];
}

/**
 * Minimal random source contract used by utility random helpers.
 *
 * The narrow contract keeps deterministic spawn utilities portable across
 * browser and test contexts.
 */
export interface RngLike {
  nextInt: (min: number, max: number) => number;
}
