/**
 * HUD mugshot head-crop decoder for the Neatenstein robot.
 *
 * Exports the head-only crop decoder and mouse-look direction selector used
 * by the HUD overlay. The head crop is extracted from the `stand` pose of each
 * front-facing direction, tinted with a health-based eye-stripe color, and
 * decoded at the standard robot sprite scale.
 *
 * @module
 */

import {
  ROBOT_SPRITE_FRAMES,
  ROBOT_SPRITE_PALETTE,
  ROBOT_SPRITE_SCALE,
} from '../../robot-sprite-data.js';
import {
  NEATENSTEIN_CANVAS_2D_CONTEXT,
  RGBA_OPAQUE_ALPHA,
} from '../constants';
import {
  CSS_FLEX_0_0_AUTO,
  CSS_HEIGHT_AUTO,
  CSS_IMAGE_PIXELATED,
  CSS_WIDTH_100PCT,
} from './hud.constants';
import {
  EYE_STRIPE_TINT_INDEX,
  MUGSHOT_COLOR_CYAN,
  MUGSHOT_COLOR_GREY,
  MUGSHOT_CROP_END_COL,
  MUGSHOT_CROP_END_ROW,
  MUGSHOT_CROP_START_COL,
  MUGSHOT_CROP_START_ROW,
  MUGSHOT_EYE_STRIPE_ROW,
} from './game/constants';
import {
  type EncodedRobotSpriteFrame,
  decodeRobotSpriteFrame,
} from '../renderer/robot-sprite-decode';
import type {
  MugshotDirection,
  MugshotHeadCrop,
  MugshotLook,
  MugshotOverlay,
} from './types';

// Re-export consolidated types so existing imports from this module remain valid.
export type {
  MugshotDirection,
  MugshotHeadCrop,
  MugshotLook,
  MugshotOverlay,
} from './types';

/**
 * Select the mugshot direction from the current mouse-look yaw delta.
 *
 * Negative `yawDelta` (turning left) yields `frontLeft`; positive (turning
 * right) yields `frontRight`; zero (mouse still) yields `front`.
 *
 * @param look - Mouse-look state containing the per-frame yaw delta.
 * @returns Selected mugshot direction.
 */
export function selectMugshotDirection(look: MugshotLook): MugshotDirection {
  if (look.yawDelta < 0) return 'frontLeft';
  if (look.yawDelta > 0) return 'frontRight';
  return 'front';
}

/**
 * Compute the eye-stripe tint color by lerping from NEON_GRAY (zero health)
 * to NEON_TEAL (full health).
 *
 * @param healthRatio - Health fraction from 0.0 to 1.0.
 * @returns Tinted RGBA color tuple.
 */
function computeEyeStripeTint(
  healthRatio: number,
): readonly [number, number, number, number] {
  return [
    MUGSHOT_COLOR_GREY[0] + (MUGSHOT_COLOR_CYAN[0] - MUGSHOT_COLOR_GREY[0]) * healthRatio,
    MUGSHOT_COLOR_GREY[1] + (MUGSHOT_COLOR_CYAN[1] - MUGSHOT_COLOR_GREY[1]) * healthRatio,
    MUGSHOT_COLOR_GREY[2] + (MUGSHOT_COLOR_CYAN[2] - MUGSHOT_COLOR_GREY[2]) * healthRatio,
    RGBA_OPAQUE_ALPHA,
  ];
}

/**
 * Determine whether a row-7 index-1 pixel qualifies as an eye-stripe outline
 * pixel — adjacent to an index-5 on one side and an index-0 on the other.
 *
 * @param frame - Original encoded frame (48×48).
 * @param col - Column index within the original frame.
 * @returns True if the pixel is an eye-stripe outline pixel.
 */
function isEyeStripeOutline(
  frame: EncodedRobotSpriteFrame,
  col: number,
): boolean {
  /* istanbul ignore next -- defensive: col±1 always within 48-col frame */
  const left = frame[MUGSHOT_EYE_STRIPE_ROW][col - 1] ?? 0;
  /* istanbul ignore next -- defensive: col±1 always within 48-col frame */
  const right = frame[MUGSHOT_EYE_STRIPE_ROW][col + 1] ?? 0;
  return (left === 5 && right === 0) || (left === 0 && right === 5);
}

/**
 * Build a modified crop frame with eye-stripe tint pixels remapped to a
 * custom palette index.
 *
 * In row 7 of the crop, all index-5 pixels and qualifying index-1 outline
 * pixels are replaced with a new palette index (9) that maps to the
 * health-based tint color.
 *
 * @param frame - Original 48×48 encoded frame.
 * @returns Modified crop frame (15 rows × 11 cols) with tint indices applied.
 */
function buildTintedCropFrame(frame: EncodedRobotSpriteFrame): number[][] {
  const crop: number[][] = [];
  for (let row = MUGSHOT_CROP_START_ROW; row < MUGSHOT_CROP_END_ROW; row += 1) {
    const originalRow = frame[row] as readonly number[];
    const cropRow: number[] = [];
    for (
      let col = MUGSHOT_CROP_START_COL;
      col < MUGSHOT_CROP_END_COL;
      col += 1
    ) {
      const value = originalRow[col];
      if (row === MUGSHOT_EYE_STRIPE_ROW) {
        if (value === 5) {
          cropRow.push(EYE_STRIPE_TINT_INDEX);
          continue;
        }
        if (value === 1 && isEyeStripeOutline(frame, col)) {
          cropRow.push(EYE_STRIPE_TINT_INDEX);
          continue;
        }
      }
      cropRow.push(value);
    }
    crop.push(cropRow);
  }
  return crop;
}

/**
 * Decode the head-only mugshot crop for a given direction.
 *
 * The crop spans rows 0–14 and cols 19–29 of the `stand` pose, scaled up by
 * `ROBOT_SPRITE_SCALE` to 44×60 physical pixels. The eye-stripe (row 7
 * index-5 pixels and adjacent index-1 outline pixels) is tinted with a
 * health-based color lerped from neon gray (dead) to neon teal (healthy).
 *
 * @param direction - One of 'front', 'frontLeft', or 'frontRight'.
 * @param healthRatio - Health fraction from 0.0 to 1.0. Defaults to 1.0.
 * @returns Decoded head crop with width, height, and RGBA data.
 */
export function decodeMugshotHeadCrop(
  direction: MugshotDirection,
  healthRatio?: number,
): MugshotHeadCrop {
  const ratio = healthRatio ?? 1.0;
  const frame = ROBOT_SPRITE_FRAMES[direction].stand as EncodedRobotSpriteFrame;
  const tintedCrop = buildTintedCropFrame(frame);
  const tint = computeEyeStripeTint(ratio);
  const extendedPalette = [
    ...ROBOT_SPRITE_PALETTE,
    tint,
  ] as readonly (readonly [number, number, number, number])[];
  const decoded = decodeRobotSpriteFrame(tintedCrop, extendedPalette);
  return {
    width: decoded.width,
    height: decoded.height,
    data: decoded.data,
  };
}

/** Physical width of the mugshot crop in pixels (11 cols × scale). */
const MUGSHOT_CROP_WIDTH =
  (MUGSHOT_CROP_END_COL - MUGSHOT_CROP_START_COL) * ROBOT_SPRITE_SCALE;

/** Physical height of the mugshot crop in pixels (15 rows × scale). */
const MUGSHOT_CROP_HEIGHT =
  (MUGSHOT_CROP_END_ROW - MUGSHOT_CROP_START_ROW) * ROBOT_SPRITE_SCALE;

/** Mugshot canvas overlay instance returned by {@link createMugshotOverlay}. */
// Type is defined in ./types and re-exported above.

/**
 * Create a host-DOM canvas overlay that renders the robot mugshot head crop.
 *
 * The canvas is sized to the physical crop dimensions (44×60) and uses
 * `image-rendering: pixelated` for crisp upscaling. The `update` callback
 * decodes the head crop for the given direction and health ratio, then draws
 * the pixel data onto the canvas via a 2D context. When the 2D context is
 * unavailable (e.g. in jsdom without the `canvas` package), the draw is a
 * no-op so the overlay remains safe to construct in test environments.
 *
 * @returns Mugshot overlay instance with `canvas` and `update`.
 */
export function createMugshotOverlay(): MugshotOverlay {
  const canvas = document.createElement('canvas');
  canvas.width = MUGSHOT_CROP_WIDTH;
  canvas.height = MUGSHOT_CROP_HEIGHT;
  canvas.style.imageRendering = CSS_IMAGE_PIXELATED;
  canvas.style.height = CSS_WIDTH_100PCT;
  canvas.style.width = CSS_HEIGHT_AUTO;
  canvas.style.flex = CSS_FLEX_0_0_AUTO;

  const update = (direction: MugshotDirection, healthRatio?: number): void => {
    const crop = decodeMugshotHeadCrop(direction, healthRatio);
    const ctx = canvas.getContext(NEATENSTEIN_CANVAS_2D_CONTEXT);
    if (ctx) {
      const imageData = ctx.createImageData(crop.width, crop.height);
      imageData.data.set(crop.data);
      ctx.putImageData(imageData, 0, 0);
    }
  };

  // Initialize with front-facing, full-health mugshot.
  update('front', 1.0);

  return { canvas, update };
}
