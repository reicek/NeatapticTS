/**
 * Host HUD overlay for the Neatenstein neon raycasting demo.
 *
 * Renders a HIVE DENSITY meter into a reserved host container. The meter
 * displays the current enemy population density as a colored fill bar with
 * threshold-dependent styling and a percentage label, giving the player an
 * at-a-glance read on swarm pressure.
 *
 * @module
 */

import {
  NEATENSTEIN_HUD_LABEL_TEXT,
  NEATENSTEIN_HUD_METER_HEIGHT_PX,
  NEATENSTEIN_HUD_METER_WIDTH_PX,
  NEATENSTEIN_HEALTH_AMMO_LABEL_AMMO,
  NEATENSTEIN_HEALTH_AMMO_LABEL_HEALTH,
  NEATENSTEIN_HEALTH_COLOR_CYAN,
  NEATENSTEIN_HEALTH_COLOR_MAGENTA,
} from '../constants';
import { createMugshotOverlay } from './hud-mugshot';

import {
  NEATENSTEIN_DEATH_FEEDBACK_LABEL_TEXT,
  NEON_STATUS_BAR_HEIGHT_PX,
  NEON_STATUS_BAR_INACTIVE_COLOR,
  NEON_STATUS_BAR_SEGMENT_COUNT,
  CSS_DISPLAY_FLEX,
  CSS_FONT_14PX,
  CSS_FONT_MONOSPACE,
  CSS_HEIGHT_0PCT,
  CSS_OVERLAY_BG,
  CSS_PADDING_4PX_8PX,
  CSS_POSITION_ABSOLUTE,
  CSS_WIDTH_100PCT,
} from './hud.constants';
import { HUD_Z_INDEX } from './game/constants';
import { resolveHiveDensityColor, resolveHealthColor } from './hud.color.utils';
import {
  resolveHudContainer,
  createSegmentedTrack,
  createStatusPrefix,
  createStatusLabel,
} from './hud.dom.utils';
import {
  computeHealthSegments,
  computeAmmoSegments,
  resolveStatusBarReadouts,
} from './hud.status-bar.update.utils';
import type {
  DeathFeedbackIndicator,
  DeathFeedbackSignal,
  HealthAmmoHud,
  HealthAmmoHudState,
  HiveDensityHud,
  HiveDensityHudState,
  NeonStatusBarHud,
  NeonStatusBarState,
} from './types';

// Re-exports for previously-public symbols that moved to util files and types.
export {
  type HumanMode,
  type HumanModeSelector,
  createHumanModeSelector,
} from './hud.human-mode.utils';
export {
  type WaveAnnouncementHud,
  createWaveAnnouncement,
} from './hud.wave.utils';
export type {
  DeathFeedbackIndicator,
  DeathFeedbackSignal,
  HealthAmmoHud,
  HealthAmmoHudState,
  HiveDensityHud,
  HiveDensityHudState,
  NeonStatusBarHud,
  NeonStatusBarState,
} from './types';

/**
 * Create a HIVE DENSITY HUD overlay inside the host container identified by
 * `outputId`.
 *
 * The function resolves the existing host container whose `id === outputId`,
 * throws if it is missing, and appends a fixed-size meter track, a fill bar,
 * and a label element. The returned `update` callback drives the fill width,
 * threshold color, and label percentage from render-state snapshots posted on
 * each animation frame.
 *
 * @param outputId - Host container element id reserved for HUD output.
 * @returns HUD instance with `container`, `meter`, `fill`, `label`, and
 *   `update`.
 * @throws {Error} When no element with `id === outputId` exists in the
 *   document.
 *
 * @example
 * ```ts
 * const hud = createHiveDensityHud('neatenstein-hud-output');
 * hud.update({ hiveDensity: 0.75 });
 * console.log(hud.fill.style.backgroundColor); // 'rgb(255, 0, 85)'
 * ```
 */
export function createHiveDensityHud(outputId: string): HiveDensityHud {
  const container = resolveHudContainer(outputId, 'HIVE DENSITY overlay');

  const meter = document.createElement('div');
  meter.style.width = `${NEATENSTEIN_HUD_METER_WIDTH_PX}px`;
  meter.style.height = `${NEATENSTEIN_HUD_METER_HEIGHT_PX}px`;

  const fill = document.createElement('div');
  fill.style.height = CSS_WIDTH_100PCT;
  fill.style.width = CSS_HEIGHT_0PCT;
  meter.appendChild(fill);

  const label = document.createElement('div');
  label.textContent = NEATENSTEIN_HUD_LABEL_TEXT;

  container.appendChild(meter);
  container.appendChild(label);

  const update = (state: HiveDensityHudState): void => {
    const density = state.hiveDensity;
    const percentage = Math.round(density * 100);
    fill.style.width = `${percentage}%`;
    fill.style.backgroundColor = resolveHiveDensityColor(density);
    label.textContent = `${NEATENSTEIN_HUD_LABEL_TEXT} ${percentage}%`;
  };

  return { container, meter, fill, label, update };
}

// ---------------------------------------------------------------------------
// Death feedback indicator
// ---------------------------------------------------------------------------

/**
 * Create a death feedback indicator inside the host container identified by
 * `outputId`.
 *
 * The function resolves the existing host container whose `id === outputId`,
 * throws if it is missing, and appends an indicator wrapper element and a label
 * element. The returned `update` callback drives the label text from
 * adaptation-signal snapshots posted at each generation boundary.
 *
 * @param outputId - Host container element id reserved for HUD output.
 * @returns Indicator instance with `container`, `indicator`, `label`, and
 *   `update`.
 * @throws {Error} When no element with `id === outputId` exists in the
 *   document.
 *
 * @example
 * ```ts
 * const indicator = createDeathFeedbackIndicator('neatenstein-hud-output');
 * indicator.update({ direction: 'stronger', aggressionDelta: 0.4, movementDelta: -0.1, positioningDelta: 0.3 });
 * console.log(indicator.label.textContent); // 'DEATH FEEDBACK: STRONGER'
 * ```
 */
export function createDeathFeedbackIndicator(
  outputId: string,
): DeathFeedbackIndicator {
  const container = resolveHudContainer(outputId, 'death feedback indicator');

  const indicator = document.createElement('div');
  indicator.setAttribute('data-role', 'death-feedback');
  indicator.style.position = CSS_POSITION_ABSOLUTE;
  indicator.style.top = '0px';
  indicator.style.left = '0px';
  indicator.style.zIndex = String(HUD_Z_INDEX);
  indicator.style.color = NEATENSTEIN_HEALTH_COLOR_CYAN;
  indicator.style.fontFamily = CSS_FONT_MONOSPACE;
  indicator.style.fontSize = CSS_FONT_14PX;
  indicator.style.padding = CSS_PADDING_4PX_8PX;
  indicator.style.backgroundColor = CSS_OVERLAY_BG;

  const label = document.createElement('div');
  label.textContent = NEATENSTEIN_DEATH_FEEDBACK_LABEL_TEXT;
  label.style.color = 'inherit';

  indicator.appendChild(label);
  container.appendChild(indicator);

  const update = (signal: DeathFeedbackSignal): void => {
    label.textContent = `${NEATENSTEIN_DEATH_FEEDBACK_LABEL_TEXT}: ${signal.direction.toUpperCase()}`;
  };

  return { container, indicator, label, update };
}

// ---------------------------------------------------------------------------
// Health/ammo HUD overlay
// ---------------------------------------------------------------------------

/**
 * Create a health/ammo HUD overlay inside the host container identified by
 * `outputId`.
 *
 * The function resolves the existing host container whose `id === outputId`,
 * throws if it is missing, and appends a health meter track, a fill bar, a
 * health label, and an ammo label element. The returned `update` callback
 * drives the fill width, threshold color, health label, and ammo label from
 * render-state snapshots posted on each animation frame.
 *
 * @param outputId - Host container element id reserved for HUD output.
 * @returns HUD instance with `container`, `healthTrack`, `healthFill`,
 *   `healthLabel`, `ammoLabel`, and `update`.
 * @throws {Error} When no element with `id === outputId` exists in the
 *   document.
 *
 * @example
 * ```ts
 * const hud = createHealthAmmoHud('neatenstein-hud-output');
 * hud.update({ health: 75, maxHealth: 100, ammo: 30, maxAmmo: 50 });
 * console.log(hud.healthFill.style.width); // '75%'
 * console.log(hud.ammoLabel.textContent); // 'AMMO 30/50'
 * ```
 */
export function createHealthAmmoHud(outputId: string): HealthAmmoHud {
  const container = resolveHudContainer(outputId, 'health/ammo overlay');

  const healthTrack = document.createElement('div');
  healthTrack.style.width = `${NEATENSTEIN_HUD_METER_WIDTH_PX}px`;
  healthTrack.style.height = `${NEATENSTEIN_HUD_METER_HEIGHT_PX}px`;

  const healthFill = document.createElement('div');
  healthFill.style.height = CSS_WIDTH_100PCT;
  healthFill.style.width = CSS_HEIGHT_0PCT;
  healthTrack.appendChild(healthFill);

  const healthLabel = document.createElement('div');
  healthLabel.textContent = NEATENSTEIN_HEALTH_AMMO_LABEL_HEALTH;

  const ammoLabel = document.createElement('div');
  ammoLabel.textContent = NEATENSTEIN_HEALTH_AMMO_LABEL_AMMO;

  container.appendChild(healthTrack);
  container.appendChild(healthLabel);
  container.appendChild(ammoLabel);

  const update = (state: HealthAmmoHudState): void => {
    const fraction = state.maxHealth > 0 ? state.health / state.maxHealth : 0;
    const percentage = Math.round(fraction * 100);
    healthFill.style.width = `${percentage}%`;
    healthFill.style.backgroundColor = resolveHealthColor(fraction);
    healthLabel.textContent = `${NEATENSTEIN_HEALTH_AMMO_LABEL_HEALTH} ${percentage}%`;
    ammoLabel.textContent = `${NEATENSTEIN_HEALTH_AMMO_LABEL_AMMO} ${state.ammo}/${state.maxAmmo}`;
  };

  return { container, healthTrack, healthFill, healthLabel, ammoLabel, update };
}

// ---------------------------------------------------------------------------
// Neon status bar overlay
// ---------------------------------------------------------------------------

/**
 * Create a neon Wolfenstein-style status bar overlay inside the host
 * container identified by `outputId`.
 *
 * The function resolves the existing host container whose `id === outputId`,
 * throws if it is missing, and appends an absolutely-positioned bottom bar
 * with neon cyan borders, segmented health and ammo tracks, a HIVE density
 * fill bar, and kill/death readout labels. The returned `update` callback
 * drives the segment colors, fill width, and label text from render-state
 * snapshots posted on each animation frame.
 *
 * @param outputId - Host container element id reserved for HUD output.
 * @returns Status bar instance with `container`, `bar`, `healthSegments`,
 *   `ammoSegments`, `hiveFill`, `killsLabel`, `deathsLabel`, and `update`.
 * @throws {Error} When no element with `id === outputId` exists in the
 *   document.
 *
 * @example
 * ```ts
 * const statusBar = createNeonStatusBar('neatenstein-hud-output');
 * statusBar.update({ playerHealth: 70, playerMaxHealth: 100, playerAmmo: 50, playerMaxAmmo: 100, playerKills: 3, playerDeaths: 1 });
 * console.log(statusBar.healthSegments[0].style.backgroundColor); // 'rgb(0, 240, 255)'
 * ```
 */
export function createNeonStatusBar(outputId: string): NeonStatusBarHud {
  // Step 1: Resolve the host container.
  const container = resolveHudContainer(outputId, 'neon status bar');

  // Step 2: Build the absolutely-positioned bottom bar element.
  const bar = document.createElement('div');
  bar.style.position = CSS_POSITION_ABSOLUTE;
  bar.style.bottom = '0px';
  bar.style.left = '0px';
  bar.style.width = CSS_WIDTH_100PCT;
  bar.style.height = `${NEON_STATUS_BAR_HEIGHT_PX}px`;
  bar.style.borderColor = NEATENSTEIN_HEALTH_COLOR_CYAN;
  bar.style.borderStyle = 'solid';
  bar.style.borderWidth = '1px';
  bar.style.display = CSS_DISPLAY_FLEX;
  bar.style.gap = '4px';
  bar.style.padding = '4px';
  bar.style.boxSizing = 'border-box';
  bar.style.alignItems = 'stretch';
  bar.style.backgroundColor = CSS_OVERLAY_BG;

  // Step 3: Create segmented health track (leftmost bar on the status edge).
  const healthSegments = createSegmentedTrack(
    NEON_STATUS_BAR_SEGMENT_COUNT,
    'health-segment',
    NEON_STATUS_BAR_INACTIVE_COLOR,
  );
  for (const seg of healthSegments) {
    bar.appendChild(seg);
  }

  // Step 4: Create kill prefix + label (left of the centered portrait).
  const killsPrefix = createStatusPrefix('K:', NEATENSTEIN_HEALTH_COLOR_CYAN);
  const killsLabel = createStatusLabel('0', NEATENSTEIN_HEALTH_COLOR_CYAN);
  bar.appendChild(killsPrefix);
  bar.appendChild(killsLabel);

  // Step 5: Create mugshot canvas overlay (absolutely centered on screen).
  const mugshot = createMugshotOverlay();
  mugshot.canvas.style.position = CSS_POSITION_ABSOLUTE;
  mugshot.canvas.style.left = '50%';
  mugshot.canvas.style.top = '0';
  mugshot.canvas.style.transform = 'translateX(-50%)';
  mugshot.canvas.style.zIndex = '1';
  bar.appendChild(mugshot.canvas);

  // Step 6: Create HIVE density fill bar (flex spacer + density indicator).
  const hiveTrack = document.createElement('div');
  hiveTrack.className = 'hive-density-track';
  hiveTrack.style.flex = '2';
  hiveTrack.style.height = CSS_WIDTH_100PCT;
  hiveTrack.style.position = 'relative';
  hiveTrack.style.backgroundColor = NEON_STATUS_BAR_INACTIVE_COLOR;
  const hiveFill = document.createElement('div');
  hiveFill.className = 'hive-density-fill';
  hiveFill.style.height = CSS_WIDTH_100PCT;
  hiveFill.style.width = CSS_HEIGHT_0PCT;
  hiveTrack.appendChild(hiveFill);
  bar.appendChild(hiveTrack);

  // Step 7: Create death prefix + label (right of the centered portrait).
  const deathsPrefix = createStatusPrefix(
    'D:',
    NEATENSTEIN_HEALTH_COLOR_MAGENTA,
  );
  const deathsLabel = createStatusLabel('0', NEATENSTEIN_HEALTH_COLOR_MAGENTA);
  bar.appendChild(deathsPrefix);
  bar.appendChild(deathsLabel);

  // Step 8: Create generation prefix + label (right of the deaths label).
  const generationPrefix = createStatusPrefix(
    'GEN:',
    NEATENSTEIN_HEALTH_COLOR_CYAN,
  );
  const generationLabel = createStatusLabel('0', NEATENSTEIN_HEALTH_COLOR_CYAN);
  bar.appendChild(generationPrefix);
  bar.appendChild(generationLabel);

  // Step 9: Create segmented ammo track (rightmost bar on the status edge).
  const ammoSegments = createSegmentedTrack(
    NEON_STATUS_BAR_SEGMENT_COUNT,
    'ammo-segment',
    NEON_STATUS_BAR_INACTIVE_COLOR,
  );
  for (const seg of ammoSegments) {
    bar.appendChild(seg);
  }

  container.appendChild(bar);

  // Step 10: Wire the update callback using imported executors.
  let internalState: NeonStatusBarState = {
    hiveDensity: 0,
    playerHealth: 0,
    playerMaxHealth: 100,
    playerAmmo: 0,
    playerMaxAmmo: 50,
    playerKills: 0,
    playerDeaths: 0,
    generation: 0,
  };

  const update = (state: NeonStatusBarState): void => {
    internalState = { ...internalState, ...state };

    // Compute and apply health segment colors.
    const healthFraction =
      internalState.playerMaxHealth > 0
        ? internalState.playerHealth / internalState.playerMaxHealth
        : 0;
    computeHealthSegments(healthSegments, healthFraction);

    // Compute and apply ammo segment colors.
    const ammoFraction =
      internalState.playerMaxAmmo > 0
        ? internalState.playerAmmo / internalState.playerMaxAmmo
        : 0;
    computeAmmoSegments(ammoSegments, ammoFraction);

    // Update HIVE density fill.
    const density = internalState.hiveDensity ?? 0;
    const percentage = Math.round(density * 100);
    hiveFill.style.width = `${percentage}%`;
    hiveFill.style.backgroundColor = resolveHiveDensityColor(density);

    // Update kill/death/generation readout labels.
    resolveStatusBarReadouts(
      killsLabel,
      deathsLabel,
      generationLabel,
      internalState,
    );
  };

  return {
    container,
    bar,
    healthSegments,
    ammoSegments,
    hiveFill,
    killsLabel,
    deathsLabel,
    generationLabel,
    mugshot,
    update,
  };
}
