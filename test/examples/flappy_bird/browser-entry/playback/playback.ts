import type { PlaybackFrameStats } from '../browser-entry.types';
import { animatePopulationEpisodeInternal } from '../browser-entry.playback.utils';

/**
 * Public playback entry point used by browser runtime orchestration.
 *
 * @param canvas - Target playback canvas.
 * @param context - Canvas 2D context.
 * @param evolutionWorker - Worker owning playback simulation state.
 * @param onFrameStats - Callback receiving per-frame playback telemetry.
 * @returns Aggregate playback summary for the current episode.
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
  return animatePopulationEpisodeInternal(
    canvas,
    context,
    evolutionWorker,
    onFrameStats,
  );
}
