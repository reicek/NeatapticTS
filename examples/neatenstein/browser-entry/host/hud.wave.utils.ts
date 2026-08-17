/**
 * Wave announcement HUD for the Neatenstein host overlay.
 *
 * Renders a centered "WAVE N" announcement with multi-layer cyan glow,
 * responsive font sizing, and fade-in/hold/fade-out animation.
 *
 * @module
 */

import { resolveHudContainer } from './hud.dom.utils';
import {
  NEATENSTEIN_WAVE_FADE_MS,
  NEATENSTEIN_WAVE_FONT_FAMILY,
  NEATENSTEIN_WAVE_FONT_SIZE_RATIO,
  NEATENSTEIN_WAVE_GLOW_COLOR,
  NEATENSTEIN_WAVE_GLOW_FAR_PX,
  NEATENSTEIN_WAVE_GLOW_INNER_PX,
  NEATENSTEIN_WAVE_GLOW_MID_PX,
  NEATENSTEIN_WAVE_GLOW_OUTER_PX,
  NEATENSTEIN_WAVE_HOLD_MS,
  NEATENSTEIN_WAVE_MAX_FONT_SIZE_PX,
  NEATENSTEIN_WAVE_MIN_FONT_SIZE_PX,
  NEATENSTEIN_WAVE_TEXT_COLOR,
  CSS_POSITION_ABSOLUTE,
  CSS_JUSTIFY_CENTER,
  CSS_WIDTH_100PCT,
} from './hud.constants';
import type { WaveAnnouncementHud } from './types';

// Re-export consolidated type so existing imports from this module remain valid.
export type { WaveAnnouncementHud } from './types';

/**
 * Create a centered "WAVE N" announcement overlay inside the host container.
 *
 * The overlay reproduces the flappy_bird generation display exactly:
 * - Monospace font (Consolas, Menlo, Monaco, monospace), weight 700
 * - Responsive font size: `round(clamp(min(w,h) * 0.075, 18, 40))`
 * - Neon-green fill (#00ff66)
 * - Two-pass rendering: multi-layer cyan glow pass (4 stacked text-shadows at
 *   20/40/80/120px blur for intense, diffuse halo) + crisp core pass (no shadow,
 *   full opacity)
 * - 500ms linear fade-in, 600ms hold, 500ms linear fade-out
 *
 * @param outputId - DOM id of the host container (e.g. `'neatenstein-output'`).
 * @returns HUD handle with a `show(waveNumber)` method.
 */
export function createWaveAnnouncement(outputId: string): WaveAnnouncementHud {
  const container = resolveHudContainer(outputId, 'wave announcement');

  // Wrapper div — handles positioning and opacity transition.
  const overlay = document.createElement('div');
  overlay.setAttribute('data-role', 'wave-announcement');
  overlay.style.position = CSS_POSITION_ABSOLUTE;
  overlay.style.top = '50%';
  overlay.style.left = '50%';
  overlay.style.transform = 'translate(-50%, -50%)';
  overlay.style.zIndex = '20';
  overlay.style.pointerEvents = 'none';
  overlay.style.opacity = '0';
  overlay.style.transition = `opacity ${NEATENSTEIN_WAVE_FADE_MS}ms linear`;
  overlay.style.whiteSpace = 'nowrap';
  overlay.style.textAlign = CSS_JUSTIFY_CENTER;

  // Glow pass element — multi-layer cyan text-shadow for intense, diffuse glow.
  // Stacks 4 shadow layers at increasing blur radii to reproduce the wide,
  // bright halo that canvas additive blending creates in flappy_bird.
  const glowLayer = document.createElement('div');
  glowLayer.style.fontFamily = NEATENSTEIN_WAVE_FONT_FAMILY;
  glowLayer.style.fontWeight = '700';
  glowLayer.style.color = NEATENSTEIN_WAVE_TEXT_COLOR;
  glowLayer.style.textShadow = [
    `0 0 ${NEATENSTEIN_WAVE_GLOW_INNER_PX}px ${NEATENSTEIN_WAVE_GLOW_COLOR}`,
    `0 0 ${NEATENSTEIN_WAVE_GLOW_MID_PX}px rgba(95, 255, 255, 0.75)`,
    `0 0 ${NEATENSTEIN_WAVE_GLOW_OUTER_PX}px rgba(95, 255, 255, 0.5)`,
    `0 0 ${NEATENSTEIN_WAVE_GLOW_FAR_PX}px rgba(95, 255, 255, 0.3)`,
  ].join(', ');

  // Core pass element — normal blend, full opacity, no shadow.
  // Reproduces canvas globalCompositeOperation='source-over' crisp text.
  const coreLayer = document.createElement('div');
  coreLayer.style.fontFamily = NEATENSTEIN_WAVE_FONT_FAMILY;
  coreLayer.style.fontWeight = '700';
  coreLayer.style.color = NEATENSTEIN_WAVE_TEXT_COLOR;
  coreLayer.style.position = CSS_POSITION_ABSOLUTE;
  coreLayer.style.top = '0';
  coreLayer.style.left = '0';
  coreLayer.style.width = CSS_WIDTH_100PCT;

  overlay.appendChild(glowLayer);
  overlay.appendChild(coreLayer);
  container.appendChild(overlay);

  /**
   * Compute responsive font size matching flappy_bird's formula:
   * `round(clamp(min(w,h) * 0.075, 18, 40))`.
   */
  const resolveFontSizePx = (): string => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    const smaller = Math.max(1, Math.min(w, h));
    const raw = smaller * NEATENSTEIN_WAVE_FONT_SIZE_RATIO;
    const clamped = Math.min(
      Math.max(raw, NEATENSTEIN_WAVE_MIN_FONT_SIZE_PX),
      NEATENSTEIN_WAVE_MAX_FONT_SIZE_PX,
    );
    return `${Math.round(clamped)}px`;
  };

  let hideTimer: ReturnType<typeof setTimeout> | null = null;

  const show = (waveNumber: number): void => {
    if (hideTimer !== null) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    const text = `Wave ${waveNumber}`;
    const fontSize = resolveFontSizePx();
    glowLayer.textContent = text;
    coreLayer.textContent = text;
    glowLayer.style.fontSize = fontSize;
    coreLayer.style.fontSize = fontSize;
    overlay.style.opacity = '1';
    hideTimer = setTimeout(() => {
      overlay.style.opacity = '0';
      hideTimer = setTimeout(() => {
        hideTimer = null;
      }, NEATENSTEIN_WAVE_FADE_MS);
    }, NEATENSTEIN_WAVE_HOLD_MS);
  };

  return { container, overlay, show };
}
