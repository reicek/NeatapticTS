import type { PlaybackFrameStats } from '../browser-entry.types';
import { renderPopulationFrame } from './frame-render/playback.frame-render.service';
import { runPlaybackLoop } from './playback.iteration.services';
import type { PlaybackEpisodeSummary } from './playback.orchestration.types';
import {
  initializePlaybackSessionContext,
  resolvePlaybackEpisodeSummary,
} from './playback.session.services';

export type { PlaybackEpisodeSummary } from './playback.orchestration.types';

/**
 * Public playback orchestration for the Flappy Bird browser demo.
 *
 * Playback is the bridge between off-thread simulation and on-screen
 * visualization. The worker advances the world and streams packed snapshots;
 * this layer mirrors enough state locally to animate those snapshots at browser
 * frame cadence, render trails and backgrounds, and emit HUD telemetry.
 */

/**
 * Public playback entry point used by browser runtime orchestration.
 *
 * Conceptually, this answers: "play one worker-produced episode on the canvas
 * until it is done, and tell me what happened along the way".
 *
 * @param canvas - Target playback canvas.
 * @param context - Canvas 2D context.
 * @param evolutionWorker - Worker owning playback simulation state.
 * @param onFrameStats - Callback receiving per-frame playback telemetry.
 * @returns Aggregate playback summary for the current episode.
 * @example
 * ```ts
 * const summary = await animatePopulationEpisode(
 *   canvas,
 *   context,
 *   evolutionWorker,
 *   (stats) => updateHud(stats),
 * );
 * ```
 */
export async function animatePopulationEpisode(
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  evolutionWorker: Worker,
  onFrameStats: (stats: PlaybackFrameStats) => void,
): Promise<PlaybackEpisodeSummary> {
  return animatePopulationEpisodeInternal(
    canvas,
    context,
    evolutionWorker,
    onFrameStats,
  );
}

/**
 * Internal playback orchestration entry retained for compatibility re-exports.
 *
 * The implementation is shared with the public entry so legacy imports and the
 * newer folderized surface behave identically.
 *
 * @param canvas - Target playback canvas.
 * @param context - Canvas 2D context.
 * @param evolutionWorker - Worker owning playback simulation state.
 * @param onFrameStats - Callback receiving per-frame playback telemetry.
 * @returns Aggregate playback summary for the current episode.
 */
export async function animatePopulationEpisodeInternal(
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  evolutionWorker: Worker,
  onFrameStats: (stats: PlaybackFrameStats) => void,
): Promise<PlaybackEpisodeSummary> {
  // Step 1: Initialize worker playback state and local render mirrors.
  const sessionContext = initializePlaybackSessionContext(
    canvas,
    evolutionWorker,
  );

  // Step 2: Run playback iterations until the worker reports completion.
  await runPlaybackLoop({
    canvas,
    context,
    evolutionWorker,
    onFrameStats,
    sessionContext,
  });

  // Step 3: Render the final settled frame for the completed episode.
  renderPopulationFrame(
    context,
    sessionContext.renderState,
    sessionContext.trailState,
  );

  // Step 4: Fold the mutable loop summary into the public return shape.
  return resolvePlaybackEpisodeSummary(sessionContext.loopState.summary);
}
