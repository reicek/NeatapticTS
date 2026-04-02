export type {
  PlaybackStarfieldLayerSpec,
  StarTile,
} from './playback.starfield.types';

/**
 * Shared playback utility types for the browser-entry subsystem.
 *
 * These types support rendering concerns that cut across multiple playback
 * helpers, such as edge-aware trail fading and cached parallax backgrounds.
 */

/**
 * Axis-aligned visible world bounds used for edge-aware trail fading.
 *
 * Trail rendering needs a quick answer to "is this point still visually inside
 * the active world rectangle?" so fading logic can taper paths near the edges
 * instead of drawing abrupt cutoffs.
 */
export type PlaybackEdgeBounds = {
  leftXPx: number;
  rightXPx: number;
  topYPx: number;
  bottomYPx: number;
};
