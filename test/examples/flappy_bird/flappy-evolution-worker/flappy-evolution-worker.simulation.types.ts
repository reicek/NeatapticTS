import type { RngLike } from '../browser-entry/browser-entry.types';
import type { SharedDifficultyProfile } from '../flappy.simulation.shared.utils';
import type { WorkerPlaybackState } from './flappy-evolution-worker.types';

/** Shared mutable inputs for one worker playback frame simulation pass. */
export interface WorkerPlaybackFrameContext {
  renderState: WorkerPlaybackState;
  rng: RngLike;
  difficultyProfile: SharedDifficultyProfile;
  controlSubstepDelta: number;
  cameraLeftXPx: number;
  birdLeftXPx: number;
  birdRightXPx: number;
}
