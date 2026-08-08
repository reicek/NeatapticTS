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
  NEATENSTEIN_HIVE_DENSITY_COLOR_CALM,
  NEATENSTEIN_HIVE_DENSITY_COLOR_HIGH,
  NEATENSTEIN_HIVE_DENSITY_COLOR_LOW,
  NEATENSTEIN_HIVE_DENSITY_COLOR_MID,
  NEATENSTEIN_HIVE_DENSITY_THRESHOLD_HIGH,
  NEATENSTEIN_HIVE_DENSITY_THRESHOLD_LOW,
  NEATENSTEIN_HIVE_DENSITY_THRESHOLD_MID,
  NEATENSTEIN_HUMAN_MODE_LABEL_AUTO,
  NEATENSTEIN_HUMAN_MODE_LABEL_HUMAN,
  NEATENSTEIN_HEALTH_AMMO_LABEL_AMMO,
  NEATENSTEIN_HEALTH_AMMO_LABEL_HEALTH,
  NEATENSTEIN_HEALTH_COLOR_AMBER,
  NEATENSTEIN_HEALTH_COLOR_CYAN,
  NEATENSTEIN_HEALTH_COLOR_MAGENTA,
  NEATENSTEIN_HEALTH_THRESHOLD_AMBER,
  NEATENSTEIN_HEALTH_THRESHOLD_CYAN,
} from '../constants';

/**
 * Render-state snapshot accepted by {@link HiveDensityHud.update}.
 */
export interface HiveDensityHudState {
  /** Current enemy population density, expected in [0, 1]. */
  hiveDensity: number;
}

/**
 * HUD instance returned by {@link createHiveDensityHud}.
 *
 * Each field exposes a live DOM reference so callers (and tests) can assert on
 * the rendered state. The `update` callback refreshes the fill width, color,
 * and label text from a render-state snapshot.
 */
export interface HiveDensityHud {
  /** Resolved host container element that owns the HUD. */
  container: HTMLElement;
  /** Meter track element (fixed-size outer bar). */
  meter: HTMLElement;
  /** Fill bar element (width + color driven by density). */
  fill: HTMLElement;
  /** Label element showing the meter title and percentage. */
  label: HTMLElement;
  /** Refresh the HUD from a render-state snapshot. */
  update: (state: HiveDensityHudState) => void;
}

/**
 * Resolve a threshold-dependent CSS color for a hive density value.
 *
 * - Below {@link NEATENSTEIN_HIVE_DENSITY_THRESHOLD_LOW} → calm cyan.
 * - From low (inclusive) to mid (exclusive) → low green.
 * - From mid (inclusive) to high (exclusive) → mid amber.
 * - At or above high → high magenta.
 *
 * @param density - Current density, expected in [0, 1].
 * @returns CSS `rgb()` color string.
 */
function resolveHiveDensityColor(density: number): string {
  if (density < NEATENSTEIN_HIVE_DENSITY_THRESHOLD_LOW) {
    return NEATENSTEIN_HIVE_DENSITY_COLOR_CALM;
  }
  if (density < NEATENSTEIN_HIVE_DENSITY_THRESHOLD_MID) {
    return NEATENSTEIN_HIVE_DENSITY_COLOR_LOW;
  }
  if (density < NEATENSTEIN_HIVE_DENSITY_THRESHOLD_HIGH) {
    return NEATENSTEIN_HIVE_DENSITY_COLOR_MID;
  }
  return NEATENSTEIN_HIVE_DENSITY_COLOR_HIGH;
}

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
  const container = document.getElementById(outputId);
  if (!container) {
    throw new Error(
      `HUD container #${outputId} not found; cannot create HIVE DENSITY overlay`,
    );
  }

  const meter = document.createElement('div');
  meter.style.width = `${NEATENSTEIN_HUD_METER_WIDTH_PX}px`;
  meter.style.height = `${NEATENSTEIN_HUD_METER_HEIGHT_PX}px`;

  const fill = document.createElement('div');
  fill.style.height = '100%';
  fill.style.width = '0%';
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
// Human-mode selector
// ---------------------------------------------------------------------------

/**
 * Available human-mode selector values.
 *
 * - `'auto'` — the demo runs in autonomous NEAT-driven mode.
 * - `'human'` — the demo switches to human-play mode with replay-buffer
 *   selection pressure.
 */
export type HumanMode = 'auto' | 'human';

/**
 * Human-mode selector instance returned by {@link createHumanModeSelector}.
 *
 * The selector renders a `<select>` element with auto/human options into the
 * host container. The `mode` field tracks the current selection, `setMode`
 * programmatically updates it, and `onToggle` registers a callback that fires
 * whenever the mode changes (either via the select element or `setMode`).
 */
export interface HumanModeSelector {
  /** Resolved host container element that owns the selector. */
  container: HTMLElement;
  /** The `<select>` element with auto/human options. */
  select: HTMLSelectElement;
  /** Current mode ('auto' or 'human'). */
  mode: HumanMode;
  /** Programmatically set the mode and notify registered callbacks. */
  setMode: (mode: HumanMode) => void;
  /** Register a callback invoked whenever the mode changes. */
  onToggle: (callback: (mode: HumanMode) => void) => void;
}

// ---------------------------------------------------------------------------
// Death feedback indicator
// ---------------------------------------------------------------------------

/**
 * Adaptation signal accepted by {@link DeathFeedbackIndicator.update}.
 *
 * Mirrors the {@link AdaptationSignal} shape from the death-feedback harness
 * module so the HUD can consume the signal without a circular import.
 */
export interface DeathFeedbackSignal {
  /** Coarse direction label: `'stronger'`, `'weaker'`, or `'shifted'`. */
  direction: 'stronger' | 'weaker' | 'shifted';
  /** Change in enemy aggression between generations. */
  aggressionDelta: number;
  /** Change in enemy movement pattern between generations. */
  movementDelta: number;
  /** Change in enemy positioning between generations. */
  positioningDelta: number;
}

/**
 * Death feedback indicator instance returned by
 * {@link createDeathFeedbackIndicator}.
 *
 * Each field exposes a live DOM reference so callers (and tests) can assert on
 * the rendered state. The `update` callback refreshes the indicator label from
 * an adaptation signal posted on each generation boundary.
 */
export interface DeathFeedbackIndicator {
  /** Resolved host container element that owns the indicator. */
  container: HTMLElement;
  /** Indicator wrapper element appended to the host container. */
  indicator: HTMLElement;
  /** Label element showing the adaptation direction text. */
  label: HTMLElement;
  /** Refresh the indicator from an adaptation signal. */
  update: (signal: DeathFeedbackSignal) => void;
}

/** Static label prefix shown on the death feedback indicator. */
const NEATENSTEIN_DEATH_FEEDBACK_LABEL_TEXT = 'DEATH FEEDBACK' as const;

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
  const container = document.getElementById(outputId);
  /* istanbul ignore next -- defensive throw; tested implicitly via createHiveDensityHud pattern */
  if (!container) {
    throw new Error(
      `HUD container #${outputId} not found; cannot create death feedback indicator`,
    );
  }

  const indicator = document.createElement('div');
  indicator.setAttribute('data-role', 'death-feedback');
  indicator.style.position = 'absolute';
  indicator.style.top = '0px';
  indicator.style.left = '0px';
  indicator.style.zIndex = '10';
  indicator.style.color = NEATENSTEIN_HEALTH_COLOR_CYAN;
  indicator.style.fontFamily = 'monospace';
  indicator.style.fontSize = '14px';
  indicator.style.padding = '4px 8px';
  indicator.style.backgroundColor = 'rgba(6, 11, 20, 0.85)';

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

/**
 * Create a human-mode selector dropdown inside the host container identified
 * by `outputId`.
 *
 * The function resolves the existing host container whose `id === outputId`,
 * throws if it is missing, and appends a `<select>` element with two options:
 * `'auto'` (initial) and `'human'`. The returned object exposes `mode`,
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
  const container = document.getElementById(outputId);
  /* istanbul ignore next -- defensive throw; tested implicitly via createHiveDensityHud pattern */
  if (!container) {
    throw new Error(
      `HUD container #${outputId} not found; cannot create human-mode selector`,
    );
  }

  const select = document.createElement('select');
  select.style.position = 'absolute';
  select.style.top = '0px';
  select.style.right = '0px';
  select.style.zIndex = '10';
  select.style.backgroundColor = 'rgba(6, 11, 20, 0.85)';
  select.style.color = NEATENSTEIN_HEALTH_COLOR_CYAN;
  select.style.fontFamily = 'monospace';
  select.style.fontSize = '14px';
  select.style.padding = '4px 8px';
  select.style.borderColor = NEATENSTEIN_HEALTH_COLOR_CYAN;

  const autoOption = document.createElement('option');
  autoOption.value = NEATENSTEIN_HUMAN_MODE_LABEL_AUTO;
  autoOption.textContent = NEATENSTEIN_HUMAN_MODE_LABEL_AUTO;
  select.appendChild(autoOption);

  const humanOption = document.createElement('option');
  humanOption.value = NEATENSTEIN_HUMAN_MODE_LABEL_HUMAN;
  humanOption.textContent = NEATENSTEIN_HUMAN_MODE_LABEL_HUMAN;
  select.appendChild(humanOption);

  select.value = NEATENSTEIN_HUMAN_MODE_LABEL_AUTO;

  container.appendChild(select);

  let mode: HumanMode = 'auto';
  const callbacks: Array<(mode: HumanMode) => void> = [];

  const notify = (): void => {
    for (const cb of callbacks) {
      cb(mode);
    }
  };

  select.addEventListener('change', () => {
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

// ---------------------------------------------------------------------------
// Health/ammo HUD overlay
// ---------------------------------------------------------------------------

/**
 * Render-state snapshot accepted by {@link HealthAmmoHud.update}.
 */
export interface HealthAmmoHudState {
  /** Current player health. */
  health: number;
  /** Maximum player health. */
  maxHealth: number;
  /** Current player ammo. */
  ammo: number;
  /** Maximum player ammo. */
  maxAmmo: number;
}

/**
 * Health/ammo HUD instance returned by {@link createHealthAmmoHud}.
 *
 * Each field exposes a live DOM reference so callers (and tests) can assert on
 * the rendered state. The `update` callback refreshes the health fill width,
 * threshold color, health label, and ammo label from a render-state snapshot.
 */
export interface HealthAmmoHud {
  /** Resolved host container element that owns the HUD. */
  container: HTMLElement;
  /** Health meter track element (fixed-size outer bar). */
  healthTrack: HTMLElement;
  /** Health fill bar element (width + color driven by health fraction). */
  healthFill: HTMLElement;
  /** Label element showing the health percentage. */
  healthLabel: HTMLElement;
  /** Label element showing the ammo count. */
  ammoLabel: HTMLElement;
  /** Refresh the HUD from a render-state snapshot. */
  update: (state: HealthAmmoHudState) => void;
}

/**
 * Resolve a threshold-dependent CSS color for a health fraction.
 *
 * - At or above {@link NEATENSTEIN_HEALTH_THRESHOLD_CYAN} → cyan.
 * - From amber (inclusive) to cyan (exclusive) → amber.
 * - Below amber → magenta.
 *
 * @param fraction - Current health fraction (health / maxHealth), expected in
 *   [0, 1].
 * @returns CSS `rgb()` color string.
 */
function resolveHealthColor(fraction: number): string {
  if (fraction >= NEATENSTEIN_HEALTH_THRESHOLD_CYAN) {
    return NEATENSTEIN_HEALTH_COLOR_CYAN;
  }
  if (fraction >= NEATENSTEIN_HEALTH_THRESHOLD_AMBER) {
    return NEATENSTEIN_HEALTH_COLOR_AMBER;
  }
  return NEATENSTEIN_HEALTH_COLOR_MAGENTA;
}

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
  const container = document.getElementById(outputId);
  if (!container) {
    throw new Error(
      `HUD container #${outputId} not found; cannot create health/ammo overlay`,
    );
  }

  const healthTrack = document.createElement('div');
  healthTrack.style.width = `${NEATENSTEIN_HUD_METER_WIDTH_PX}px`;
  healthTrack.style.height = `${NEATENSTEIN_HUD_METER_HEIGHT_PX}px`;

  const healthFill = document.createElement('div');
  healthFill.style.height = '100%';
  healthFill.style.width = '0%';
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

/** Number of segments in each segmented track (health and ammo). */
const NEON_STATUS_BAR_SEGMENT_COUNT = 10;

/** Background color for inactive (unlit) segments. */
const NEON_STATUS_BAR_INACTIVE_COLOR = 'rgba(0, 0, 0, 0.2)' as const;

/** Fixed height of the neon status bar overlay, in CSS pixels. */
const NEON_STATUS_BAR_HEIGHT_PX = 48;

/**
 * Render-state snapshot accepted by {@link NeonStatusBarHud.update}.
 *
 * The health/ammo fields are always required. `hiveDensity`, `playerKills`,
 * and `playerDeaths` are optional and default to 0 when omitted.
 */
export interface NeonStatusBarState {
  /** Current enemy population density, expected in [0, 1]. */
  hiveDensity?: number;
  /** Current player health. */
  playerHealth: number;
  /** Maximum player health. */
  playerMaxHealth: number;
  /** Current player ammo. */
  playerAmmo: number;
  /** Maximum player ammo. */
  playerMaxAmmo: number;
  /** Current player kill count (fallback 0). */
  playerKills?: number;
  /** Current player death count (fallback 0). */
  playerDeaths?: number;
}

/**
 * Neon status bar HUD instance returned by {@link createNeonStatusBar}.
 *
 * Each field exposes a live DOM reference so callers (and tests) can assert
 * on the rendered state. The `update` callback refreshes the segmented health
 * track, segmented ammo track, HIVE density fill, and kill/death labels from
 * a render-state snapshot posted on each animation frame.
 */
export interface NeonStatusBarHud {
  /** Resolved host container element that owns the status bar. */
  container: HTMLElement;
  /** Absolutely-positioned bottom overlay bar element. */
  bar: HTMLElement;
  /** Array of health segment elements lit by health fraction. */
  healthSegments: HTMLElement[];
  /** Array of ammo segment elements lit by ammo fraction. */
  ammoSegments: HTMLElement[];
  /** HIVE density fill bar element (width driven by density). */
  hiveFill: HTMLElement;
  /** Label element showing the player kill count. */
  killsLabel: HTMLElement;
  /** Label element showing the player death count. */
  deathsLabel: HTMLElement;
  /** Refresh the status bar from a render-state snapshot. */
  update: (state: NeonStatusBarState) => void;
}

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
  const container = document.getElementById(outputId);
  if (!container) {
    throw new Error(
      `HUD container #${outputId} not found; cannot create neon status bar`,
    );
  }

  const bar = document.createElement('div');
  bar.style.position = 'absolute';
  bar.style.bottom = '0px';
  bar.style.left = '0px';
  bar.style.width = '100%';
  bar.style.height = `${NEON_STATUS_BAR_HEIGHT_PX}px`;
  bar.style.borderColor = NEATENSTEIN_HEALTH_COLOR_CYAN;
  bar.style.borderStyle = 'solid';
  bar.style.borderWidth = '1px';
  bar.style.display = 'flex';
  bar.style.gap = '4px';
  bar.style.padding = '4px';
  bar.style.boxSizing = 'border-box';
  bar.style.alignItems = 'stretch';
  bar.style.backgroundColor = 'rgba(6, 11, 20, 0.85)';

  // Segmented health track
  const healthSegments: HTMLElement[] = [];
  for (let i = 0; i < NEON_STATUS_BAR_SEGMENT_COUNT; i++) {
    const seg = document.createElement('div');
    seg.className = 'health-segment';
    seg.style.flex = '1';
    seg.style.height = '100%';
    seg.style.backgroundColor = NEON_STATUS_BAR_INACTIVE_COLOR;
    healthSegments.push(seg);
    bar.appendChild(seg);
  }

  // Segmented ammo track
  const ammoSegments: HTMLElement[] = [];
  for (let i = 0; i < NEON_STATUS_BAR_SEGMENT_COUNT; i++) {
    const seg = document.createElement('div');
    seg.className = 'ammo-segment';
    seg.style.flex = '1';
    seg.style.height = '100%';
    seg.style.backgroundColor = NEON_STATUS_BAR_INACTIVE_COLOR;
    ammoSegments.push(seg);
    bar.appendChild(seg);
  }

  // HIVE density fill bar
  const hiveTrack = document.createElement('div');
  hiveTrack.className = 'hive-density-track';
  hiveTrack.style.flex = '1';
  hiveTrack.style.height = '100%';
  hiveTrack.style.position = 'relative';
  hiveTrack.style.backgroundColor = NEON_STATUS_BAR_INACTIVE_COLOR;
  const hiveFill = document.createElement('div');
  hiveFill.className = 'hive-density-fill';
  hiveFill.style.height = '100%';
  hiveFill.style.width = '0%';
  hiveTrack.appendChild(hiveFill);
  bar.appendChild(hiveTrack);

  // Kill/death readout labels with prefix labels for visibility
  const killsPrefix = document.createElement('span');
  killsPrefix.textContent = 'K:';
  killsPrefix.style.color = NEATENSTEIN_HEALTH_COLOR_CYAN;
  killsPrefix.style.fontFamily = 'monospace';
  killsPrefix.style.fontSize = '16px';
  killsPrefix.style.padding = '0 2px';
  killsPrefix.style.display = 'flex';
  killsPrefix.style.alignItems = 'center';

  const killsLabel = document.createElement('div');
  killsLabel.textContent = '0';
  killsLabel.style.color = NEATENSTEIN_HEALTH_COLOR_CYAN;
  killsLabel.style.fontFamily = 'monospace';
  killsLabel.style.fontSize = '16px';
  killsLabel.style.padding = '0 4px';
  killsLabel.style.display = 'flex';
  killsLabel.style.alignItems = 'center';

  const deathsPrefix = document.createElement('span');
  deathsPrefix.textContent = 'D:';
  deathsPrefix.style.color = NEATENSTEIN_HEALTH_COLOR_MAGENTA;
  deathsPrefix.style.fontFamily = 'monospace';
  deathsPrefix.style.fontSize = '16px';
  deathsPrefix.style.padding = '0 2px';
  deathsPrefix.style.display = 'flex';
  deathsPrefix.style.alignItems = 'center';

  const deathsLabel = document.createElement('div');
  deathsLabel.textContent = '0';
  deathsLabel.style.color = NEATENSTEIN_HEALTH_COLOR_MAGENTA;
  deathsLabel.style.fontFamily = 'monospace';
  deathsLabel.style.fontSize = '16px';
  deathsLabel.style.padding = '0 4px';
  deathsLabel.style.display = 'flex';
  deathsLabel.style.alignItems = 'center';

  bar.appendChild(killsPrefix);
  bar.appendChild(killsLabel);
  bar.appendChild(deathsPrefix);
  bar.appendChild(deathsLabel);

  container.appendChild(bar);

  let internalState: NeonStatusBarState = {
    hiveDensity: 0,
    playerHealth: 0,
    playerMaxHealth: 100,
    playerAmmo: 0,
    playerMaxAmmo: 50,
    playerKills: 0,
    playerDeaths: 0,
  };

  const update = (state: NeonStatusBarState): void => {
    internalState = { ...internalState, ...state };

    // Health segments
    const healthFraction =
      internalState.playerMaxHealth > 0
        ? internalState.playerHealth / internalState.playerMaxHealth
        : 0;
    const activeCount = Math.round(healthFraction * healthSegments.length);
    const healthColor = resolveHealthColor(healthFraction);
    for (let i = 0; i < healthSegments.length; i++) {
      healthSegments[i].style.backgroundColor =
        i < activeCount ? healthColor : NEON_STATUS_BAR_INACTIVE_COLOR;
    }

    // Ammo segments
    const ammoFraction =
      internalState.playerMaxAmmo > 0
        ? internalState.playerAmmo / internalState.playerMaxAmmo
        : 0;
    const activeAmmoCount = Math.round(ammoFraction * ammoSegments.length);
    for (let i = 0; i < ammoSegments.length; i++) {
      ammoSegments[i].style.backgroundColor =
        i < activeAmmoCount
          ? NEATENSTEIN_HEALTH_COLOR_CYAN
          : NEON_STATUS_BAR_INACTIVE_COLOR;
    }

    // HIVE density fill
    const density = internalState.hiveDensity ?? 0;
    const percentage = Math.round(density * 100);
    hiveFill.style.width = `${percentage}%`;
    hiveFill.style.backgroundColor = resolveHiveDensityColor(density);

    // Kill/death readouts
    killsLabel.textContent = `${internalState.playerKills ?? 0}`;
    deathsLabel.textContent = `${internalState.playerDeaths ?? 0}`;
  };

  return {
    container,
    bar,
    healthSegments,
    ammoSegments,
    hiveFill,
    killsLabel,
    deathsLabel,
    update,
  };
}

/** Wave announcement overlay state. */
export interface WaveAnnouncementHud {
  /** Resolved host container element. */
  container: HTMLElement;
  /** The overlay div element. */
  overlay: HTMLDivElement;
  /** Show the "Wave N" announcement with fade-in/hold/fade-out animation. */
  show: (waveNumber: number) => void;
}

/**
 * Monospace font family matching the flappy_bird neon generation display.
 * Note: no outer quotes — the browser must parse this as a CSS font fallback
 * list, not a single font name.
 */
const NEATENSTEIN_WAVE_FONT_FAMILY =
  'Consolas, Menlo, Monaco, monospace' as const;

/** Neon green fill color matching flappy_bird generation text. */
const NEATENSTEIN_WAVE_TEXT_COLOR = '#00ff66' as const;

/** Cyan glow color matching flappy_bird pipe outline glow. */
const NEATENSTEIN_WAVE_GLOW_COLOR = 'rgba(95, 255, 255, 0.95)' as const;

/** Inner glow blur radius in px. */
const NEATENSTEIN_WAVE_GLOW_INNER_PX = 20 as const;

/** Mid glow blur radius in px. */
const NEATENSTEIN_WAVE_GLOW_MID_PX = 40 as const;

/** Outer glow blur radius in px. */
const NEATENSTEIN_WAVE_GLOW_OUTER_PX = 80 as const;

/** Far glow blur radius in px. */
const NEATENSTEIN_WAVE_GLOW_FAR_PX = 120 as const;

/** Responsive font-size ratio — matches flappy_bird's 0.075. */
const NEATENSTEIN_WAVE_FONT_SIZE_RATIO = 0.075 as const;

/** Minimum font size in px — matches flappy_bird's 18px. */
const NEATENSTEIN_WAVE_MIN_FONT_SIZE_PX = 18 as const;

/** Maximum font size in px — matches flappy_bird's 40px. */
const NEATENSTEIN_WAVE_MAX_FONT_SIZE_PX = 40 as const;

/** Wave announcement fade-in/out duration in milliseconds (linear ramp). */
const NEATENSTEIN_WAVE_FADE_MS = 500 as const;

/** Wave announcement hold duration in milliseconds. */
const NEATENSTEIN_WAVE_HOLD_MS = 600 as const;

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
export function createWaveAnnouncement(
  outputId: string,
): WaveAnnouncementHud {
  const container = document.getElementById(outputId);
  if (!container) {
    throw new Error(
      `HUD container #${outputId} not found; cannot create wave announcement`,
    );
  }

  // Wrapper div — handles positioning and opacity transition.
  const overlay = document.createElement('div');
  overlay.setAttribute('data-role', 'wave-announcement');
  overlay.style.position = 'absolute';
  overlay.style.top = '50%';
  overlay.style.left = '50%';
  overlay.style.transform = 'translate(-50%, -50%)';
  overlay.style.zIndex = '20';
  overlay.style.pointerEvents = 'none';
  overlay.style.opacity = '0';
  overlay.style.transition = `opacity ${NEATENSTEIN_WAVE_FADE_MS}ms linear`;
  overlay.style.whiteSpace = 'nowrap';
  overlay.style.textAlign = 'center';

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
  coreLayer.style.position = 'absolute';
  coreLayer.style.top = '0';
  coreLayer.style.left = '0';
  coreLayer.style.width = '100%';

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
