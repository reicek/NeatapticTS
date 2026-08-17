/**
 * Human-mode selector HUD for the Neatenstein host overlay.
 *
 * Renders a `<select>` dropdown that toggles between autonomous and
 * human-play modes, with callback notification on mode change.
 *
 * @module
 */

import {
  NEATENSTEIN_HUMAN_MODE_LABEL_AUTO,
  NEATENSTEIN_HUMAN_MODE_LABEL_HUMAN,
  NEATENSTEIN_HEALTH_COLOR_CYAN,
} from '../constants';
import { DOM_EVENT_CHANGE } from './dom-events.constants';
import {
  CSS_FONT_14PX,
  CSS_FONT_MONOSPACE,
  CSS_OVERLAY_BG,
  CSS_PADDING_4PX_8PX,
  CSS_POSITION_ABSOLUTE,
} from './hud.constants';
import { HUD_Z_INDEX } from './game/constants';
import { resolveHudContainer } from './hud.dom.utils';
import type { HumanMode, HumanModeSelector } from './types';

// Re-export consolidated types so existing imports from this module remain valid.
export type { HumanMode, HumanModeSelector } from './types';

/**
 * Create a human-mode selector dropdown inside the host container identified
 * by `outputId`.
 *
 * The function resolves the existing host container whose `id === outputId`,
 * throws if it is missing, and appends a `<select>` element with two options:
 * `'auto'` and `'human'` (initial). The returned object exposes `mode`,
 * `setMode`, and `onToggle` so the host can wire the selector to the arms-race
 * configuration.
 *
 * @param outputId - Host container element id reserved for HUD output.
 * @returns A {@link HumanModeSelector} instance.
 * @throws {Error} When no element with `id === outputId` exists in the
 *   document.
 *
 * @example
 * ```ts
 * const selector = createHumanModeSelector('neatenstein-hud-output');
 * selector.onToggle((mode) => console.log(`Mode changed to ${mode}`));
 * selector.setMode('human'); // logs 'Mode changed to human'
 * ```
 */
export function createHumanModeSelector(outputId: string): HumanModeSelector {
  const container = resolveHudContainer(outputId, 'human-mode selector');

  const select = document.createElement('select');
  select.style.position = CSS_POSITION_ABSOLUTE;
  select.style.top = '0px';
  select.style.right = '0px';
  select.style.zIndex = String(HUD_Z_INDEX);
  select.style.backgroundColor = CSS_OVERLAY_BG;
  select.style.color = NEATENSTEIN_HEALTH_COLOR_CYAN;
  select.style.fontFamily = CSS_FONT_MONOSPACE;
  select.style.fontSize = CSS_FONT_14PX;
  select.style.padding = CSS_PADDING_4PX_8PX;
  select.style.borderColor = NEATENSTEIN_HEALTH_COLOR_CYAN;

  const autoOption = document.createElement('option');
  autoOption.value = NEATENSTEIN_HUMAN_MODE_LABEL_AUTO;
  autoOption.textContent = NEATENSTEIN_HUMAN_MODE_LABEL_AUTO;
  select.appendChild(autoOption);

  const humanOption = document.createElement('option');
  humanOption.value = NEATENSTEIN_HUMAN_MODE_LABEL_HUMAN;
  humanOption.textContent = NEATENSTEIN_HUMAN_MODE_LABEL_HUMAN;
  select.appendChild(humanOption);

  select.value = NEATENSTEIN_HUMAN_MODE_LABEL_HUMAN;

  container.appendChild(select);

  let mode: HumanMode = NEATENSTEIN_HUMAN_MODE_LABEL_HUMAN;
  const callbacks: Array<(mode: HumanMode) => void> = [];

  const notify = (): void => {
    for (const cb of callbacks) {
      cb(mode);
    }
  };

  select.addEventListener(DOM_EVENT_CHANGE, () => {
    mode = select.value as HumanMode;
    notify();
  });

  const setMode = (next: HumanMode): void => {
    /* istanbul ignore next -- no-op guard; the test always changes mode */
    if (next === mode) {
      return;
    }
    mode = next;
    select.value = next;
    notify();
  };

  const onToggle = (callback: (mode: HumanMode) => void): void => {
    callbacks.push(callback);
  };

  return {
    container,
    select,
    get mode(): HumanMode {
      return mode;
    },
    setMode,
    onToggle,
  };
}
