/**
 * Shared constants for responsive host sizing.
 *
 * These values keep the resize policy readable by naming the split ratio, the
 * minimum positive dimension, and the CSS token bundle used by layout appliers.
 */

/**
 * Shared CSS tokens used by the host resize layout appliers.
 *
 * Centralizing these string tokens reduces repetition across the responsive DOM
 * style appliers.
 */
export const FLAPPY_HOST_RESIZE_STYLE_TOKENS = {
  rowDirection: 'row' as CSSStyleDeclaration['flexDirection'],
  columnDirection: 'column' as CSSStyleDeclaration['flexDirection'],
  stretchAlignment: 'stretch' as CSSStyleDeclaration['alignItems'],
  flexStartAlignment: 'flex-start' as CSSStyleDeclaration['alignItems'],
  hiddenDisplay: 'none' as CSSStyleDeclaration['display'],
  blockDisplay: 'block' as CSSStyleDeclaration['display'],
  autoSize: 'auto',
  zeroSizePx: '0px',
  fullSize: '100%',
  hiddenOverflow: 'hidden' as CSSStyleDeclaration['overflowY'],
  autoOverflow: 'auto' as CSSStyleDeclaration['overflowY'],
  primaryOrder: '0',
  secondaryOrder: '1',
  autoFlex: '0 0 auto',
  fillFlex: '1 1 0',
  fullFlex: '1 1 100%',
} as const;

/**
 * Split ratio used when dividing the viewport between simulation and stats.
 *
 * A value of `0.5` means the standard stacked layout begins from an even split
 * before clamping against minimum-height constraints.
 */
export const FLAPPY_HOST_RESIZE_SPLIT_RATIO = 0.5;

/**
 * Minimum positive dimension enforced by resize math.
 *
 * This prevents zero or negative canvas sizes from leaking into downstream
 * layout and backing-store calculations.
 */
export const FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX = 1;
