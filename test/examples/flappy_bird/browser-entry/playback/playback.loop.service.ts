import { PlaybackAnimationFrameUnavailableError } from './playback.errors';

/**
 * Browser frame-pacing helpers for playback animation.
 *
 * The playback loop advances worker simulation in batches but still presents
 * frames at browser animation cadence. This module isolates the
 * `requestAnimationFrame` dependency so playback orchestration can read more
 * clearly and fail with a targeted error when RAF is unavailable.
 */

/**
 * Yields until the next browser animation frame.
 *
 * In browser rendering terms, this is the pacing boundary between simulation
 * work and visible painting.
 *
 * @returns Promise resolved on next animation frame.
 */
export function nextAnimationFrame(): Promise<void> {
  if (typeof requestAnimationFrame !== 'function') {
    throw new PlaybackAnimationFrameUnavailableError();
  }

  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}
