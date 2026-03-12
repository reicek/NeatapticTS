/**
 * Compatibility facade for playback snapshot helpers.
 *
 * Legacy imports still reach snapshot synchronization through this file while
 * the implementation now lives in the dedicated snapshot folder.
 */
export { applyPlaybackSnapshot } from './snapshot/playback.snapshot.services';
export { resolveLeaderFramesSurvived } from './snapshot/playback.snapshot.summary.utils';
