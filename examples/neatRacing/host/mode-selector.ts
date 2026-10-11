/**
 * NeatRacing host race-mode selector — plan slice P1-S7-E (plan
 * `neatracing-voxel-racer.plans.md`, phase 1, step 12), implementing the
 * AC-024 selector overlay ("top-right; ai, manual options; initial
 * manual mode"; spec sections 4.2, 10.2, 11.2, 13.1) and pre-verifying
 * the AC-033 restart-on-switch surface: `onSwitch` is the single hook
 * the host shell will connect to a race restart, so the "toggle the
 * race mode" step of the slice is delivered as a notify contract here
 * rather than as an owned restart call (spec section 4.2 keeps mode
 * authority in the shell, not the selector).
 *
 * Structural precedent (cited in the slice contract header):
 * `examples/neatenstein/browser-entry/host/hud.human-mode.utils.ts` —
 * resolve the reserved container or throw; build the `<select>` once and
 * style it from named constants; render both options with ai FIRST;
 * keep the selected mode in a closure guarded by an early same-value
 * exit; notify subscribers exactly once per actual switch (both for a
 * user `change` event and a programmatic `setMode`); never rebuild DOM
 * on mode change (allocation policy: only allocation is mount-time).
 *
 * Phase-1 reconciliation (batch 108 user-gap patch): the overlay still
 * exposes both 'ai' and 'manual' options, but the INITIAL mode is 'ai' —
 * the served page must open as a running race (the AI autopilot seam in
 * race-session samples the deterministic driver), while the manual
 * option keeps the keyboard lap flow pickable; a user switch to
 * 'manual' fires the AC-033 restart hook. Same-value switches are
 * documented no-ops (and covered
 * by the slice contract's jest non-call assertion) rather than silent
 * restarts: the host shell must not reset a race because a user
 * re-picked the already-active mode.
 *
 * Cyan/magenta discipline (spec section 11.2 "same CSS tokens,
 * cyan/magenta"): the border recycles `NEATRACING_HUD_COLOR_CYAN`
 * imported from './hud' (single source of truth for the tokens; the
 * magenta token stays owned by hud.ts since the selector only uses the
 * cyan border in this slice).
 *
 * Determinism section: every executor here is pure or mount-only —
 * identical inputs produce identical DOM/notify behavior with no
 * Math.random, no clock reads, no timers. Not claimed (same-runtime
 * rung only): cross-environment byte equality (jsdom vs browsers agree
 * on the style strings this module assigns, not on layout metrics).
 */

import { NEATRACING_HUD_COLOR_CYAN } from './hud';

/** The two race modes supported by the selector overlay. */
export type NeatRacingRaceMode = 'ai' | 'manual';

/** Human-readable option value for the AI race-mode boot. */
const MODE_OPTION_AI: NeatRacingRaceMode = 'ai';

/** Human-readable option value for the manual (keyboard) race boot. */
const MODE_OPTION_MANUAL: NeatRacingRaceMode = 'manual';

/** First option in the select (arcade affordance reads ai, manual). */
const MODE_OPTIONS: readonly NeatRacingRaceMode[] = [
  MODE_OPTION_AI,
  MODE_OPTION_MANUAL,
];

/** Initial race mode: the page opens as a running AI-driven race (batch 108). */
const INITIAL_RACE_MODE: NeatRacingRaceMode = MODE_OPTION_AI;

/** Selector placement mode (fixed overlay corner, Neatenstein pattern). */
const SELECT_POSITION_ABSOLUTE: string = 'absolute';

/** Selector top offset (pinned to the overlay's top edge). */
const SELECT_TOP_0PX: string = '0px';

/** Selector right offset (pinned to the overlay's right edge). */
const SELECT_RIGHT_0PX: string = '0px';

/** Selector stacking order (matches the Neatenstein overlay layer). */
const SELECT_Z_INDEX: number = 10;

/** Selector background (Neatenstein death-feedback palette token). */
const SELECT_BG: string = 'rgba(6, 11, 20, 0.85)';

/** Selector border color (recycled cyan token from hud.ts). */
const SELECT_BORDER_COLOR: string = NEATRACING_HUD_COLOR_CYAN;

/** Selector typeface (arcade terminal look, Neatenstein precedent). */
const SELECT_FONT_MONOSPACE: string = 'monospace';

/** Selector type size (Neatenstein precedent 14px). */
const SELECT_FONT_14PX: string = '14px';

/** Selector padding (Neatenstein precedent 4px 8px). */
const SELECT_PADDING_4PX_8PX: string = '4px 8px';

/**
 * Human-readable host name used in the missing-container error banner
 * (Neatenstein precedent surfaces which HUD piece failed to mount).
 */
const SELECTOR_NAME: string = 'NeatRacing race mode selector';

/** Race-mode change subscriber shape. */
export type NeatRacingRaceModeListener = (mode: NeatRacingRaceMode) => void;

/** The constructed race-mode selector handed to the host shell. */
export interface NeatRacingRaceModeSelector {
  /** The mounted `<select>` overlay element. */
  readonly select: HTMLSelectElement;
  /** The currently selected race mode (kept in sync with `select`). */
  readonly mode: NeatRacingRaceMode;
  /**
   * Programmatically switches the race mode (host shell hook; keeps
   * `select` synchronized and notifies exactly once; same-value calls
   * are documented no-ops so a re-pick never restarts the race).
   */
  setMode(next: NeatRacingRaceMode): void;
  /**
   * Subscribes a listener to actual mode switches (AC-033 restart
   * hook; fired exactly once per switch with the switched-to mode).
   */
  onSwitch(listener: NeatRacingRaceModeListener): void;
}

/**
 * What: resolves the reserved host output container by id, matching the
 * Neatenstein `resolveHudContainer` precedent (`hud.dom.utils.ts`)
 * flattened into this slice per the P1-S7-E file contract (no extra
 * shared files inside the slice boundary; hud.ts keeps its own copy of
 * this eight-line resolver).
 *
 * @param outputId - Reserved host DOM id (spec section 10.2 folder
 * layout; `neatracing-hud-output` in the host shell and fixtures).
 * @param surfaceName - Human-readable surface name for the error body.
 * @returns The DOM container to append the selector into.
 * @throws When `document.getElementById(outputId)` misses (host shell
 * forgot the container or the call predates DOM mount).
 *
 * @example
 * ```ts
 * resolveSelectorContainer('neatracing-hud-output', SELECTOR_NAME);
 * ```
 */
function resolveSelectorContainer(
  outputId: string,
  surfaceName: string,
): HTMLElement {
  const container: HTMLElement | null = document.getElementById(outputId);
  if (container === null) {
    throw new Error(
      `HUD container #${outputId} not found; cannot create ${surfaceName}`,
    );
  }
  return container;
}

/**
 * What: builds the `<select>` with both race-mode options in the pinned
 * order (`ai` first) and the Neatenstein top-right overlay styling from
 * named constants (spec section 11.2 recycling inventory), without
 * attaching it to the document yet.
 *
 * Why: option order and initial value are part of the AC-024 acceptance
 * contract and the DOM is built exactly once at mount (allocation
 * policy: no per-switch DOM construction).
 *
 * @returns The styled, unattached `<select>` element.
 */
function createModeSelect(): HTMLSelectElement {
  const select: HTMLSelectElement = document.createElement('select');
  select.id = 'neatracing-race-mode-selector';
  select.name = 'neatracing-race-mode-selector';
  select.style.position = SELECT_POSITION_ABSOLUTE;
  select.style.top = SELECT_TOP_0PX;
  select.style.right = SELECT_RIGHT_0PX;
  select.style.zIndex = String(SELECT_Z_INDEX);
  select.style.backgroundColor = SELECT_BG;
  select.style.borderColor = SELECT_BORDER_COLOR;
  select.style.fontFamily = SELECT_FONT_MONOSPACE;
  select.style.fontSize = SELECT_FONT_14PX;
  select.style.padding = SELECT_PADDING_4PX_8PX;
  for (const mode of MODE_OPTIONS) {
    const option: HTMLOptionElement = document.createElement('option');
    option.value = mode;
    option.textContent = mode;
    select.appendChild(option);
  }
  return select;
}

/**
 * What: creates the race-mode selector overlay inside the reserved host
 * output container with the Neatenstein `hud.human-mode.utils.ts`
 * structure: a single `<select>` (ai, manual — initial 'ai'), a
 * closure-held current mode, an `onSwitch` subscription list, and one
 * `change` listener plus one `setMode` entry point that both notify
 * subscribers exactly once per actual switch and never notify on
 * same-value switches.
 *
 * Why (AC-033 pre-verify): the host shell will connect `onSwitch` to a
 * race restart (later-phase slice); this slice delivers and verifies the
 * hook so the restart wiring cannot be blocked by a missing selector
 * surface. Batch 108 opens the race in the 'ai' mode (running page);
 * the boot glue never calls `setMode`; the 'manual' option remains
 * rendered
 * and selectable to keep the toggle contract fully exercisable (same
 * CSS tokens, cyan/magenta; option order `ai, manual` from AC-024).
 *
 * When: called once during host boot after the HUD mounts into the same
 * reserved container id (`neatracing-hud-output`). Throws instead of
 * returning null so boot-order bugs surface immediately.
 *
 * @param outputId - Reserved host DOM id; `neatracing-hud-output` in
 * the host shell and the slice fixture suite.
 * @returns The selector controller (select element + mode API).
 * @throws When the output container is missing from the document.
 *
 * @example
 * ```ts
 * const selector = createRaceModeSelector('neatracing-hud-output');
 * selector.mode; // 'ai'
 * selector.setMode('manual'); // notifies onSwitch listeners once with 'manual'
 * selector.setMode('ai'); // same-value: documented no-op, not notified
 * selector.onSwitch((mode) => restartRace(mode));
 * ```
 */
export function createRaceModeSelector(
  outputId: string,
): NeatRacingRaceModeSelector {
  const container: HTMLElement = resolveSelectorContainer(
    outputId,
    SELECTOR_NAME,
  );
  const select: HTMLSelectElement = createModeSelect();

  let mode: NeatRacingRaceMode = INITIAL_RACE_MODE;
  const listeners: NeatRacingRaceModeListener[] = [];

  const notifySwitch: () => void = (): void => {
    for (const listener of listeners) {
      listener(mode);
    }
  };

  const setMode: (next: NeatRacingRaceMode) => void = (
    next: NeatRacingRaceMode,
  ): void => {
    /* istanbul ignore if -- no-op guard; the test always changes mode */
    if (next === mode) {
      return;
    }
    mode = next;
    select.value = next;
    notifySwitch();
  };

  const onSwitch: (listener: NeatRacingRaceModeListener) => void = (
    listener: NeatRacingRaceModeListener,
  ): void => {
    listeners.push(listener);
  };

  // One change listener for the selector's lifetime; a switch is only
  // reported when the picked value actually differs from the current
  // mode, mirroring the setMode same-value guard (host shells and
  // fixtures can dispatch 'change' repeatedly without restart storms).
  select.addEventListener('change', (): void => {
    const picked: string = select.value;
    /* istanbul ignore if -- same-value change events are non-domain */
    if (picked === mode) {
      return;
    }
    // Domain-safe cast: the select only offers MODE_OPTIONS members,
    // and user picks cannot inject values outside that element's
    // option list (Neatenstein precedent casts the same way).
    mode = picked as NeatRacingRaceMode;
    notifySwitch();
  });

  // Mirror the initial mode into the DOM control after the listener is
  // attached so user picks and programmatic switches share one state.
  select.value = mode;
  container.appendChild(select);

  return {
    select,
    setMode,
    onSwitch,
    get mode(): NeatRacingRaceMode {
      return mode;
    },
  };
}
