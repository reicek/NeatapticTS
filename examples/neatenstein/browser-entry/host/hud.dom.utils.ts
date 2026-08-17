/**
 * Shared DOM element builders for the Neatenstein HUD overlay.
 *
 * Pure leaf functions that create and style DOM elements commonly used across
 * multiple HUD factories (container resolution, segmented tracks, status-bar
 * prefixes and labels).
 *
 * @module
 */

import {
  CSS_DISPLAY_FLEX,
  CSS_FONT_16PX,
  CSS_FONT_MONOSPACE,
  CSS_JUSTIFY_CENTER,
  CSS_WIDTH_100PCT,
} from './hud.constants';

/**
 * Resolve a host container element by its DOM id, throwing if it is missing.
 *
 * Shared by every `create*Hud` factory to avoid duplicated
 * `getElementById` + throw boilerplate.
 *
 * @param outputId - Host container element id reserved for HUD output.
 * @param hudName - Human-readable HUD name used in the error message.
 * @returns The resolved container element.
 * @throws {Error} When no element with `id === outputId` exists in the
 *   document.
 */
export function resolveHudContainer(
  outputId: string,
  hudName: string,
): HTMLElement {
  const container = document.getElementById(outputId);
  if (!container) {
    throw new Error(
      `HUD container #${outputId} not found; cannot create ${hudName}`,
    );
  }
  return container;
}

/**
 * Create an array of flexible segment `<div>` elements for a segmented track.
 *
 * Each segment is given the provided `className`, `flex: '1'`, full height,
 * and the inactive background color. The caller appends them to the parent
 * bar and later updates their `backgroundColor` from an update callback.
 *
 * @param segmentCount - Number of segments to create.
 * @param className - CSS class name applied to each segment.
 * @param inactiveColor - Initial background color for unlit segments.
 * @returns Array of segment elements ready to append.
 */
export function createSegmentedTrack(
  segmentCount: number,
  className: string,
  inactiveColor: string,
): HTMLElement[] {
  const segments: HTMLElement[] = [];
  for (let i = 0; i < segmentCount; i++) {
    const seg = document.createElement('div');
    seg.className = className;
    seg.style.flex = '1';
    seg.style.height = CSS_WIDTH_100PCT;
    seg.style.backgroundColor = inactiveColor;
    segments.push(seg);
  }
  return segments;
}

/**
 * Create a styled prefix `<span>` element (e.g. `"K:"`, `"D:"`, `"GEN:"`).
 *
 * The span uses monospace font, 16px size, flex layout with center alignment,
 * and the provided text and color.
 *
 * @param text - Text content for the prefix.
 * @param color - CSS color string for the text.
 * @returns A configured `<span>` element ready to append.
 */
export function createStatusPrefix(
  text: string,
  color: string,
): HTMLSpanElement {
  const prefix = document.createElement('span');
  prefix.textContent = text;
  prefix.style.color = color;
  prefix.style.fontFamily = CSS_FONT_MONOSPACE;
  prefix.style.fontSize = CSS_FONT_16PX;
  prefix.style.padding = '0 2px';
  prefix.style.display = CSS_DISPLAY_FLEX;
  prefix.style.alignItems = CSS_JUSTIFY_CENTER;
  return prefix;
}

/**
 * Create a styled status-bar label `<div>` element (e.g. kill/death counts).
 *
 * The label uses monospace font, 16px size, flex layout with center alignment,
 * and the provided initial text and color.
 *
 * @param text - Initial text content for the label.
 * @param color - CSS color string for the text.
 * @returns A configured `<div>` element ready to append.
 */
export function createStatusLabel(text: string, color: string): HTMLElement {
  const label = document.createElement('div');
  label.textContent = text;
  label.style.color = color;
  label.style.fontFamily = CSS_FONT_MONOSPACE;
  label.style.fontSize = CSS_FONT_16PX;
  label.style.padding = '0 4px';
  label.style.display = CSS_DISPLAY_FLEX;
  label.style.alignItems = CSS_JUSTIFY_CENTER;
  return label;
}
