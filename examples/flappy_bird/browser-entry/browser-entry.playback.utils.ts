/**
 * Compatibility facade for the browser-entry playback boundary.
 *
 * Older imports still reach playback through this file, while the real
 * implementation now lives in the dedicated playback folder. Keeping the facade
 * explicit preserves stable imports while letting the playback subsystem grow
 * into a clearer module boundary.
 */
export { animatePopulationEpisodeInternal } from './playback/playback';
