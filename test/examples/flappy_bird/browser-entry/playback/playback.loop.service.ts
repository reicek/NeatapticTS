import { PlaybackAnimationFrameUnavailableError } from './playback.errors';

/**
 * Yields until the next browser animation frame.
 *
 * @returns Promise resolved on next animation frame.
 */
export function nextAnimationFrame(): Promise<void> {
  if (typeof requestAnimationFrame !== 'function') {
    throw new PlaybackAnimationFrameUnavailableError();
  }

  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}
