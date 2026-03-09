export type {
  PlaybackStarfieldLayerSpec,
  StarTile,
} from './playback.starfield.types';

/**
 * Axis-aligned visible world bounds used for edge-aware trail fading.
 */
export type PlaybackEdgeBounds = {
  leftXPx: number;
  rightXPx: number;
  topYPx: number;
  bottomYPx: number;
};
