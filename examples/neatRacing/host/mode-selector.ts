/**
 * NeatRacing host race-mode selector — the closed `<select>` overlay that
 * owns the race-mode CHOICE and the switch NOTIFICATION, but never the
 * race itself.
 *
 * WHAT the overlay does: it mounts a single `<select>` (the `ai` and
 * `manual` options, `ai` first, pinned to the top-right of the reserved
 * host output container) and keeps the current mode in a closure. Both a
 * user pick and a programmatic `setMode` notify subscribers EXACTLY ONCE
 * per ACTUAL switch; a same-value switch is a documented no-op, so a
 * re-pick of the already-active mode can never restart a race.
 *
 * WHY the restart lives elsewhere: `onSwitch` is the single hook the
 * host shell connects to a race restart, so "toggle the race mode" is
 * delivered as a NOTIFY CONTRACT here rather than as an owned restart
 * call — the shell keeps the mode authority, and this module owns no
 * race state.
 *
 * THE AI-MODE LIFECYCLE this selector switches into
 * (`../ai/ai-mode.goals.ts`): AI mode runs a minimum 20-lap session and
 * then continues until the current level's promotion-threshold-driven
 * growth goal is met; level 3 continues until the user stops the
 * session. A mode switch into AI mode makes the HOST shell boot that
 * fresh AI goal session at the first ladder rung
 * (`NEATRACING_AI_MODE_START_LEVEL`).
 *
 * BOOT MODE: the overlay still exposes both 'ai' and 'manual' options,
 * but the INITIAL mode is 'manual'. WHY: the served page IS the
 * acceptance surface (the user judges the driving feel on the page that
 * is actually served) and the ai-mode page is never acceptable evidence
 * — the AI autopilot crosses the 0.5 engage cliff on only 176 of 7200
 * ticks (2.44 percent), so an AI boot masks the very slide the user has
 * to judge. The 'ai' option stays SELECTABLE through the same switch,
 * which still notifies its listeners and fires the restart hook.
 *
 * Structural precedent: `examples/neatenstein/browser-entry/host/hud.human-mode.utils.ts`
 * — resolve the reserved container or throw; build the `<select>` once and
 * style it from named constants; render both options with ai FIRST; keep
 * the selected mode in a closure guarded by an early same-value exit;
 * notify subscribers exactly once per actual switch (both for a user
 * `change` event and a programmatic `setMode`); never rebuild DOM on mode
 * change (allocation policy: only allocation is mount-time).
 *
 * Cyan/magenta discipline: the border recycles `NEATRACING_HUD_COLOR_CYAN`
 * imported from './hud' (single source of truth for the tokens; the
 * magenta token stays owned by hud.ts since the selector only uses the
 * cyan border here).
 *
 * NEON DISCIPLINE: the native `<option>` popup is OS-rendered and IGNORES
 * `text-shadow` and `filter`, so the neon lands on the CLOSED control —
 * the neon CLASS hook and the two-pass neon text style both come from the
 * ONE helper (`./hud.neon`), the single source of truth for the
 * neatRacing neon. No neon literal is inlined here, the select keeps its
 * id and its name, and the option loop below is untouched (the popup
 * stays OS-rendered; no custom listbox is introduced).
 *
 * Determinism section: every executor here is pure or mount-only —
 * identical inputs produce identical DOM/notify behavior with no
 * Math.random, no clock reads, no timers. Not claimed (same-runtime
 * rung only): cross-environment byte equality (jsdom vs browsers agree
 * on the style strings this module assigns, not on layout metrics).
 */

import { NEATRACING_HUD_COLOR_CYAN } from './hud';
import { applyNeonTreatment } from './hud.neon';

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

/**
 * Initial race mode: the game page boots in MANUAL mode.
 *
 * WHY (the served page is the acceptance surface): the served page is the
 * acceptance surface — the user judges the drift on the page that is
 * actually served — and the ai-mode page is never acceptable evidence.
 * The AI autopilot only crosses the 0.5 engage cliff on 176 of 7200
 * ticks (2.44 percent), so an AI boot masks the very slide the user
 * has to judge. This is a BOOT DEFAULT and nothing else: the 'ai'
 * option stays selectable through the mode selector and a switch still
 * notifies its listeners.
 */
export const INITIAL_RACE_MODE: NeatRacingRaceMode = MODE_OPTION_MANUAL;

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
   * Subscribes a listener to actual mode switches (the restart hook;
   * fired exactly once per switch with the switched-to mode).
   *
   * The host restarts the race in BOTH directions on this notification;
   * the boot of the fresh AI goal session that an AI-mode restart needs
   * belongs to the host wiring, not to this selector.
   */
  onSwitch(listener: NeatRacingRaceModeListener): void;
}

/**
 * What: resolves the reserved host output container by id, matching the
 * Neatenstein `resolveHudContainer` precedent (`hud.dom.utils.ts`)
 * (hud.ts keeps its own copy of this eight-line resolver).
 *
 * @param outputId - Reserved host DOM id (`neatracing-hud-output` in the
 * host shell and the fixture suite).
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
 * named constants, without attaching it to the document yet.
 *
 * Why: the option order and the initial value are part of the overlay's
 * acceptance contract, and the DOM is built exactly once at mount
 * (allocation policy: no per-switch DOM construction).
 *
 * The closed control also carries the neon treatment: the neon CLASS
 * hook and the two-pass neon text style, both read from the ONE helper
 * (`./hud.neon`).
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
  // The NEON treatment: the native `<option>` popup is OS-rendered and
  // IGNORES `text-shadow`, so the neon lands on the CLOSED control —
  // the neon CLASS hook plus the two-pass neon text style, both read
  // from the ONE helper (`./hud.neon`), never inlined.
  applyNeonTreatment(select);
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
 * structure: a single `<select>` (ai, manual — initial 'manual'), a
 * closure-held current mode, an `onSwitch` subscription list, and one
 * `change` listener plus one `setMode` entry point that both notify
 * subscribers exactly once per actual switch and never notify on
 * same-value switches.
 *
 * Why: the host connects `onSwitch` to a race restart — the restart runs
 * in BOTH directions and the host shell handles it, including the boot
 * of the fresh AI goal session at the first ladder rung
 * (`../ai/ai-mode.goals.ts`); the hook is delivered and verified here so
 * the restart wiring cannot be blocked by a missing selector surface.
 * The race opens in the 'manual' mode (the served page is the acceptance
 * surface and the ai-mode page is never acceptable evidence — see
 * `INITIAL_RACE_MODE`); the boot glue never calls `setMode`; the 'ai'
 * option remains rendered and selectable to keep the toggle contract
 * fully exercisable (same CSS tokens, cyan/magenta; option order
 * `ai, manual`).
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
 * selector.mode; // 'manual' (the boot default)
 * selector.setMode('ai'); // notifies onSwitch listeners once with 'ai'
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
    // Same-value guard: re-picking the already-active mode is a
    // documented no-op, never a silent restart. This branch is
    // domain-tested, so no coverage exemption is needed.
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
