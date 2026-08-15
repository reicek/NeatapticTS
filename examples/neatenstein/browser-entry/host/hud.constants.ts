/**
 * HUD-local constants for the Neatenstein host overlay.
 *
 * Constants that are shared with other host subsystems (health/ammo labels,
 * hive-density colors, thresholds, etc.) live in `../constants` and are NOT
 * duplicated here. This module owns only constants that are internal to the
 * HUD layer: death-feedback label, neon status-bar geometry, and wave
 * announcement styling.
 *
 * @module
 */

/** Static label prefix shown on the death feedback indicator. */
export const NEATENSTEIN_DEATH_FEEDBACK_LABEL_TEXT = 'DEATH FEEDBACK' as const;

/** Number of segments in each segmented track (health and ammo). */
export const NEON_STATUS_BAR_SEGMENT_COUNT = 10;

/** Background color for inactive (unlit) segments. */
export const NEON_STATUS_BAR_INACTIVE_COLOR = 'rgba(0, 0, 0, 0.2)' as const;

/** Fixed height of the neon status bar overlay, in CSS pixels. */
export const NEON_STATUS_BAR_HEIGHT_PX = 48;

/**
 * Monospace font family matching the flappy_bird neon generation display.
 * Note: no outer quotes — the browser must parse this as a CSS font fallback
 * list, not a single font name.
 */
export const NEATENSTEIN_WAVE_FONT_FAMILY =
  'Consolas, Menlo, Monaco, monospace' as const;

/** Neon green fill color matching flappy_bird generation text. */
export const NEATENSTEIN_WAVE_TEXT_COLOR = '#00ff66' as const;

/** Cyan glow color matching flappy_bird pipe outline glow. */
export const NEATENSTEIN_WAVE_GLOW_COLOR = 'rgba(95, 255, 255, 0.95)' as const;

/** Inner glow blur radius in px. */
export const NEATENSTEIN_WAVE_GLOW_INNER_PX = 20 as const;

/** Mid glow blur radius in px. */
export const NEATENSTEIN_WAVE_GLOW_MID_PX = 40 as const;

/** Outer glow blur radius in px. */
export const NEATENSTEIN_WAVE_GLOW_OUTER_PX = 80 as const;

/** Far glow blur radius in px. */
export const NEATENSTEIN_WAVE_GLOW_FAR_PX = 120 as const;

/** Responsive font-size ratio — matches flappy_bird's 0.075. */
export const NEATENSTEIN_WAVE_FONT_SIZE_RATIO = 0.075 as const;

/** Minimum font size in px — matches flappy_bird's 18px. */
export const NEATENSTEIN_WAVE_MIN_FONT_SIZE_PX = 18 as const;

/** Maximum font size in px — matches flappy_bird's 40px. */
export const NEATENSTEIN_WAVE_MAX_FONT_SIZE_PX = 40 as const;

/** Wave announcement fade-in/out duration in milliseconds (linear ramp). */
export const NEATENSTEIN_WAVE_FADE_MS = 500 as const;

/** Wave announcement hold duration in milliseconds. */
export const NEATENSTEIN_WAVE_HOLD_MS = 600 as const;

// ---------------------------------------------------------------------------
// CSS value constants (single source for magic strings used in style setters)
// ---------------------------------------------------------------------------

/** CSS `position: absolute` value. */
export const CSS_POSITION_ABSOLUTE = 'absolute' as const;

/** CSS `display: flex` value. */
export const CSS_DISPLAY_FLEX = 'flex' as const;

/** CSS `justify-content: center` value. */
export const CSS_JUSTIFY_CENTER = 'center' as const;

/** CSS `font-family: monospace` value. */
export const CSS_FONT_MONOSPACE = 'monospace' as const;

/** CSS `width: 100%` value. */
export const CSS_WIDTH_100PCT = '100%' as const;

/** CSS `height: 0px` value. */
export const CSS_HEIGHT_0PX = '0px' as const;

/** CSS `height: 0%` value. */
export const CSS_HEIGHT_0PCT = '0%' as const;

/** CSS font-size `14px` value. */
export const CSS_FONT_14PX = '14px' as const;

/** CSS font-size `16px` value. */
export const CSS_FONT_16PX = '16px' as const;

/** CSS `padding: 4px 8px` value. */
export const CSS_PADDING_4PX_8PX = '4px 8px' as const;

/** Background color for HUD overlay containers. */
export const CSS_OVERLAY_BG = 'rgba(6, 11, 20, 0.85)' as const;

/** CSS `image-rendering: pixelated` value for crisp sprite upscaling. */
export const CSS_IMAGE_PIXELATED = 'pixelated' as const;

/** CSS `height: auto` value. */
export const CSS_HEIGHT_AUTO = 'auto' as const;

/** CSS `flex: 0 0 auto` value. */
export const CSS_FLEX_0_0_AUTO = '0 0 auto' as const;

// ---------------------------------------------------------------------------
// Error and visibility constants
// ---------------------------------------------------------------------------

/** DOM exception name used to detect unsupported pointer/touch APIs. */
export const ERROR_NOT_SUPPORTED = 'NotSupportedError' as const;

/** CSS `visibility: visible` value. */
export const VISIBILITY_VISIBLE = 'visible' as const;
