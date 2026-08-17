/**
 * Status-bar update executors for the Neatenstein neon HUD.
 *
 * Pure leaf functions that update the segmented health/ammo tracks and
 * kill/death/generation readout labels from a render-state snapshot.
 *
 * @module
 */

import { NEATENSTEIN_HEALTH_COLOR_CYAN } from '../constants';
import { resolveHealthColor } from './hud.color.utils';
import { NEON_STATUS_BAR_INACTIVE_COLOR } from './hud.constants';
import type { NeonStatusBarState } from './types';

// Re-export consolidated type so existing imports from this module remain valid.
export type { NeonStatusBarState } from './types';

/**
 * Update health segment background colors from a health fraction.
 *
 * Segments at index `< activeCount` receive the threshold-dependent health
 * color; the rest receive the inactive color.
 *
 * @param segments - Health segment DOM elements to update.
 * @param fraction - Current health fraction (health / maxHealth), expected in
 *   [0, 1].
 */
export function computeHealthSegments(
  segments: HTMLElement[],
  fraction: number,
): void {
  const activeCount = Math.round(fraction * segments.length);
  const healthColor = resolveHealthColor(fraction);
  for (let i = 0; i < segments.length; i++) {
    segments[i].style.backgroundColor =
      i < activeCount ? healthColor : NEON_STATUS_BAR_INACTIVE_COLOR;
  }
}

/**
 * Update ammo segment background colors from an ammo fraction.
 *
 * Segments at index `< activeCount` receive the cyan active color; the rest
 * receive the inactive color.
 *
 * @param segments - Ammo segment DOM elements to update.
 * @param fraction - Current ammo fraction (ammo / maxAmmo), expected in
 *   [0, 1].
 */
export function computeAmmoSegments(
  segments: HTMLElement[],
  fraction: number,
): void {
  const activeCount = Math.round(fraction * segments.length);
  for (let i = 0; i < segments.length; i++) {
    segments[i].style.backgroundColor =
      i < activeCount
        ? NEATENSTEIN_HEALTH_COLOR_CYAN
        : NEON_STATUS_BAR_INACTIVE_COLOR;
  }
}

/**
 * Update kill, death, and generation readout labels from a state snapshot.
 *
 * @param killsLabel - Label element for the kill count.
 * @param deathsLabel - Label element for the death count.
 * @param generationLabel - Label element for the generation number.
 * @param state - Render-state snapshot with optional kill/death/generation.
 */
export function resolveStatusBarReadouts(
  killsLabel: HTMLElement,
  deathsLabel: HTMLElement,
  generationLabel: HTMLElement,
  state: NeonStatusBarState,
): void {
  killsLabel.textContent = `${state.playerKills ?? 0}`;
  deathsLabel.textContent = `${state.playerDeaths ?? 0}`;
  generationLabel.textContent = `${state.generation ?? 0}`;
}
