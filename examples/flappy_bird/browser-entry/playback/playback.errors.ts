/**
 * Error message emitted when playback requires RAF but it is unavailable.
 */
export const PLAYBACK_ANIMATION_FRAME_UNAVAILABLE_ERROR_MESSAGE =
  'Playback animation frame API is unavailable in this environment.';

/**
 * Error thrown when a playback frame wait is requested without RAF support.
 */
export class PlaybackAnimationFrameUnavailableError extends Error {
  public constructor() {
    super(PLAYBACK_ANIMATION_FRAME_UNAVAILABLE_ERROR_MESSAGE);
    this.name = 'PlaybackAnimationFrameUnavailableError';
  }
}
