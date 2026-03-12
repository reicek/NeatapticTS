/**
 * Compatibility facade for playback frame rendering.
 *
 * Older imports still reach the frame renderer through this file while the
 * implementation now lives in the dedicated frame-render folder.
 */
export {
  renderPopulationFrame,
  updateTrailState,
} from './frame-render/playback.frame-render.service';
