import type { RngLike } from '../browser-entry/browser-entry.types';
import type { SharedDifficultyProfile } from '../flappy.simulation.shared.utils';
import type { WorkerPlaybackState } from './flappy-evolution-worker.types';

/**
 * Shared mutable inputs for one worker playback frame simulation pass.
 *
 * Educational note:
 * The frame service computes several derived geometry values once per logical
 * frame and threads them through the substep helpers in this context object.
 * That keeps the top-level simulation flow declarative while avoiding repeated
 * argument sprawl across helper calls.
 */
export interface WorkerPlaybackFrameContext {
  renderState: WorkerPlaybackState;
  rng: RngLike;
  difficultyProfile: SharedDifficultyProfile;
  controlSubstepDelta: number;
  cameraLeftXPx: number;
  birdLeftXPx: number;
  birdRightXPx: number;
}
