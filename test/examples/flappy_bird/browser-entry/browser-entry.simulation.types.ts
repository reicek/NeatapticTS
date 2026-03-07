import type {
  PopulationBird,
  PopulationPipe,
} from './browser-entry.worker.types';

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

/** Difficulty profile consumed by simulation observation helpers. */
export interface BrowserDifficultyProfile {
  pipeGapPx: number;
  pipeSpeedPxPerFrame: number;
  pipeSpawnIntervalFrames: number;
}

/** Mutable render-state model consumed by the population frame renderer. */
export interface PopulationRenderState {
  frameIndex: number;
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

/** Trail point used by playback trail rendering cache. */
export interface TrailPoint {
  frameIndex: number;
  yPx: number;
}

/** Mutable trail cache keyed by bird index for frame rendering. */
export interface TrailState {
  birdTrailsY: TrailPoint[][];
}

/** Minimal random source contract used by utility random helpers. */
export interface RngLike {
  nextInt: (min: number, max: number) => number;
}
