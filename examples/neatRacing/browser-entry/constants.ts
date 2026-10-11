/**
 * @module neatRacing/browser-entry/constants
 *
 * Shared constants for the neatRacing browser host entry. This module holds
 * the fixed-canvas strategy numbers for Phase 1 (spec §6.2 and §10.2): the
 * fixed backing-store render height, the maximum backing-store width clamp,
 * and the final fallback canvas width used when no measurable reference
 * dimensions exist. Palette tokens and other host constants arrive with
 * their owning slices.
 */

/**
 * Fixed backing-store render height in pixels for the neatRacing canvas.
 *
 * The strategy (spec §6.2) always renders into a 720-pixel-tall backing
 * store and derives the backing width from the client aspect ratio, so the
 * render loop and the letterbox layout can be computed against a known
 * vertical resolution regardless of the window size.
 *
 * @example
 * resolveCanvasBackingDimensions({ clientWidth: 1920, clientHeight: 1080, ... });
 * // → { width: 1280, height: NEATRACING_FIXED_RENDER_HEIGHT }
 */
export const NEATRACING_FIXED_RENDER_HEIGHT = 720;

/**
 * Maximum backing-store width in pixels for the neatRacing canvas.
 *
 * Ultrawide displays (for example 32:9) would otherwise double the fill
 * rate compared with a 16:9 backing store; the backing width is therefore
 * clamped at this constant and the CSS letterbox absorbs the difference.
 */
export const NEATRACING_MAX_BACKING_WIDTH = 1920;

/**
 * Fallback backing-store width in pixels used when no client or viewport
 * dimensions are measurable (the final step of the resolution ladder).
 *
 * 1280x720 keeps the fallback at a 16:9 aspect ratio matching the fixed
 * render height, mirroring the Neatenstein fallback ladder shape.
 */
export const NEATRACING_FALLBACK_CANVAS_WIDTH = 1280;
