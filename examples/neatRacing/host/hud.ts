/**
 * NeatRacing host HUD overlay — plan slice P1-S7-E (plan
 * `neatracing-voxel-racer.plans.md`, phase 1, step 12), implementing spec
 * AC-024: the speed/lap/time DOM overlay into the reserved host output
 * container plus the mode-switch control surface (spec sections 4.2,
 * 10.2, 11.2, 13.1). The cyan/magenta palette is recycled verbatim from
 * the Neatenstein host HUD (spec section 11.2 recycling inventory: "same
 * CSS tokens, cyan/magenta"), pinned here as `NEATRACING_HUD_COLOR_CYAN`
 * and `NEATRACING_HUD_COLOR_MAGENTA` (rgb() literal strings, matching the
 * Neatenstein `hud.constants.ts` precedent).
 *
 * RED -> GREEN contract: `examples/neatRacing/host/mode-selector.test.ts`
 * (the header contract of this slice) pins the display string grammar —
 * speed = whole km/h (TP-01 physical anchor: 1 world unit = 1 meter, so
 * ~89 units/s ≈ ~320 km/h arcade feel; never a decimal part), lap =
 * `Lap c/20` clamped at `NEATRACING_HUMAN_RACE_LAPS` (a completed race
 * never displays 21/20; the selector restart contract AC-033 is
 * pre-verified by `mode-selector.ts`), and time = `M:SS.d` (unpadded
 * minutes, zero-padded seconds, one tenths digit).
 *
 * The arcade reconciliation is deliberate and documented rather than
 * incidental: spec section 11.1 wants an arcade-speed feel while the
 * physics layer (spec section 5) is intentionally NOT scale-accurate
 * (TP-01). The HUD is a display-only read model — it never mutates race
 * state — so the km/h conversion constant lives here at the seam where
 * the decision is made, mirroring how `race-rules.ts` keeps its own
 * budget constants instead of importing physics timing.
 *
 * Allocation policy: `createHud` is the only DOM allocator (three labels
 * created once inside one wrapper), and `update()` performs no per-frame
 * DOM/geometry allocation beyond the unavoidable display strings passed
 * to `textContent` (zero allocations is not technically achievable for
 * a text repaint; this is the minimal allocation form — one string per
 * label, no arrays/closures/new objects per call). Formatting never
 * touches the DOM: it is pure so the same numbers produce the same
 * strings on any tick, enabling deterministic replay in tests.
 *
 * Time formatting note: tenths are computed via
 * `Math.round(seconds * TENTHS_PER_SECOND)` and then integer modulo
 * arithmetic (tenths per minute = 600). Floating-point seconds such as
 * 3.04, 75.3, or 9.9999 do not accumulate a double drift error larger
 * than one rounding of the tenths (the double is 3 units of ulp at
 * ~1e14 tenths, a range far beyond any race duration); negative seconds
 * are non-domain — race-rules guarantees `elapsedS >= 0` (zero starts,
 * 60s checkpoint budget) and the formatters document that domain instead
 * of carrying an unreachable guard branch.
 *
 * Determinism section: every executor here is pure — identical inputs
 * produce identical strings with no Math.random, no clock reads, no
 * timers. Not claimed (same-runtime rung only): cross-environment byte
 * equality (DOM layout differences across browsers do not affect text
 * content or the pinned style token strings).
 */

import {
  type RaceRulesState,
  createRaceRulesState,
  NEATRACING_HUMAN_RACE_LAPS,
} from '../physics/race-rules';

/** Cyan token used for the speed label, time label, and selector border. */
export const NEATRACING_HUD_COLOR_CYAN: string = 'rgb(0, 240, 255)';

/**
 * Magenta token used for the lap label (the second accent of the
 * Neatenstein palette kept as a distinct token so future HUD elements
 * can share one import source).
 */
export const NEATRACING_HUD_COLOR_MAGENTA: string = 'rgb(255, 0, 85)';

/** Speed conversion constant (TP-01: 1 world unit = 1 meter). */
const KMH_PER_SIM_UNIT: number = 3.6;

/** Speed unit label rendered in the speed label. */
const HUD_SPEED_UNIT_LABEL: string = 'km/h';

/** Lap word prefix for the lap label. */
const HUD_LAP_WORD: string = 'Lap';

/** Tenths of a second per unit second (time formatting sub-second unit). */
const TENTHS_PER_SECOND: number = 10;

/** Tenths of a second per whole minute (60 seconds * 10 tenths). */
const TENTHS_PER_MINUTE: number = 600;

/** Digit width of the zero-padded seconds segment (SS in M:SS.d). */
const HUD_SECONDS_PAD_WIDTH: number = 2;

/** HUD overlay placement mode. */
const HUD_POSITION_ABSOLUTE: string = 'absolute';

/** Zero-position strings shared by the overlay edges (top and left). */
const HUD_OFFSET_0PX: string = '0px';

/** HUD overlay stacking order (matches the Neatenstein overlay layer). */
const HUD_Z_INDEX: number = 10;

/** HUD overlay background (Neatenstein death-feedback palette token). */
const HUD_OVERLAY_BG: string = 'rgba(6, 11, 20, 0.85)';

/** HUD border thickness (Neatenstein 1px solid overlay border). */
const HUD_BORDER_WIDTH_1PX: string = '1px';

/** HUD border style (Neatenstein solid overlay border). */
const HUD_BORDER_STYLE_SOLID: string = 'solid';

/** HUD border color (recycled cyan token). */
const HUD_BORDER_COLOR: string = NEATRACING_HUD_COLOR_CYAN;

/** HUD typeface (arcade terminal look, Neatenstein precedent). */
const HUD_FONT_MONOSPACE: string = 'monospace';

/** HUD type size (Neatenstein precedent 14px). */
const HUD_FONT_14PX: string = '14px';

/** HUD padding (Neatenstein precedent 4px 8px). */
const HUD_PADDING_4PX_8PX: string = '4px 8px';

/** Speed label accent color: cyan (primary speed readout). */
const HUD_SPEED_LABEL_COLOR: string = NEATRACING_HUD_COLOR_CYAN;

/** Lap label accent color: magenta (secondary progress readout). */
const HUD_LAP_LABEL_COLOR: string = NEATRACING_HUD_COLOR_MAGENTA;

/** Time label accent color: cyan (matches speed as the primary tokens). */
const HUD_TIME_LABEL_COLOR: string = NEATRACING_HUD_COLOR_CYAN;

/** Controls-hint accent color: dim slate (a non-readout affordance). */
const HUD_HINT_LABEL_COLOR: string = 'rgb(140, 160, 175)';

/** Static controls hint shown under the readouts (AC-024 discoverability). */
const HUD_CONTROLS_HINT_TEXT: string = 'WASD / ARROWS to drive';

/** Human-readable host name used in the missing-container error banner. */
const HUD_NAME: string = 'NeatRacing HUD overlay';

/**
 * The race state slice the HUD reads from. Mirroring `input.ts`, this is
 * a named input contract instead of the whole `RaceRulesState` so the
 * render loop can pass exactly what the HUD needs.
 */
export interface NeatRacingHudState {
  /** Current forward speed, in world-units-per-second (TP-01 meters/s). */
  speedUnitsPerS: number;
  /** Authoritative race state the lap/time readouts are derived from. */
  raceState: RaceRulesState;
}

/** The constructed HUD controller handed to the host shell. */
export interface NeatRacingHud {
  /** The overlay wrapper mounted inside the output container. */
  readonly container: HTMLElement;
  /** Speed readout label (cyan accent). */
  readonly speedLabel: HTMLElement;
  /** Lap readout label (magenta accent). */
  readonly lapLabel: HTMLElement;
  /** Time readout label (cyan accent). */
  readonly timeLabel: HTMLElement;
  /**
   * Repaints the HUD readouts from live state (once per frame or once
   * per `race-rules` step, whichever the caller integrates).
   */
  update(state: NeatRacingHudState): void;
}

/**
 * What: formats forward speed for the speed label, converting world
 * units-per-second into whole km/h (TP-01: 1 unit = 1 meter, so the
 * conversion is a fixed 3.6 km/h per unit/s).
 *
 * Why: the arcade physics layer is deliberately not scale-accurate
 * (spec section 11.1) but the HUD readout is pinned to a physical anchor
 * in the slice contract so the displayed number is stable across
 * refactors of the physics constants; keeping the conversion here means
 * callers pass the raw sim speed without pre-scaling.
 *
 * When: called by `NeatRacingHud.update` per frame; also used directly
 * by tests and debug overlays.
 *
 * Format contract (slice header): whole number, no decimal part — a
 * half-unit bump is an imperceptible 1.8 km/h jump against a 320 km/h
 * readout, so rounding loses no arcade readability.
 *
 * @param speedUnitsPerS - Forward speed in world-units-per-second
 * (TP-01 meters/s); expected in [0, inf).
 * @returns The speed formatted as whole `km/h`, e.g. `'320'` for 89.
 *
 * @example
 * ```ts
 * formatHudSpeed(89); // '320 km/h'
 * formatHudSpeed(0); // '0 km/h'
 * formatHudSpeed(20); // '72 km/h'
 * ```
 */
export function formatHudSpeed(speedUnitsPerS: number): string {
  return `${String(Math.round(speedUnitsPerS * KMH_PER_SIM_UNIT))} ${HUD_SPEED_UNIT_LABEL}`;
}

/**
 * What: formats the completed-laps counter `lapsCompleted` for the lap
 * label as the race-day ordinal `Lap c/20` (the lap currently being
 * raced), clamped so a finished race still reads `Lap 20/20`.
 *
 * Why: `race-rules` counts completed crossings, so lapsCompleted === 0
 * during lap 1; the human-facing label is the current lap number. The
 * clamp uses the shared `NEATRACING_HUMAN_RACE_LAPS` constant (single
 * source of truth with the rules module, spec section 5.4); without it a
 * terminal out-of-budget tick could paint `21/20` for one frame.
 *
 * When: called by `NeatRacingHud.update` per frame and by tests.
 *
 * @param lapsCompleted - Number of lap lines fully crossed (race-rules
 * domain: integer >= 0, clamped at NEATRACING_HUMAN_RACE_LAPS).
 * @returns The lap label, e.g. `'Lap 1/20'` for 0 completed laps.
 *
 * @example
 * ```ts
 * formatHudLap(0); // 'Lap 1/20'
 * formatHudLap(1); // 'Lap 2/20'
 * formatHudLap(20); // 'Lap 20/20' (clamped, never 21/20)
 * ```
 */
export function formatHudLap(lapsCompleted: number): string {
  const currentLap: number = Math.min(
    lapsCompleted + 1,
    NEATRACING_HUMAN_RACE_LAPS,
  );
  return `${HUD_LAP_WORD} ${String(currentLap)}/${String(NEATRACING_HUMAN_RACE_LAPS)}`;
}

/**
 * What: formats race elapsed seconds as `M:SS.d` — unpadded minutes,
 * zero-padded 2-digit seconds, one tenths digit — for the time label.
 *
 * Why: tenths are resolved first via `Math.round(seconds *
 * TENTHS_PER_SECOND)` and then by integer arithmetic (minutes =
 * floor of tenths / TENTHS_PER_MINUTE; remainder split into seconds and
 * tenths), which is immune to the classic double pitfall of decomposing
 * fractional seconds directly — e.g. 75.3s stored as
 * 75.299999999999997... would floor to the wrong `:15.2` but rounds to
 * the contract-pinned `1:15.3`, and 3.04s pins to `0:03.0` not `0:03.1`
 * even though 3.04 * TENTHS_PER_SECOND lands on 30.4.
 *
 * When: called by `NeatRacingHud.update` per frame and by tests.
 *
 * @param seconds - Elapsed race time in seconds (race-rules domain:
 * non-negative; zero starts, checkpoint budget keeps it at or above 0).
 * @returns The race clock, e.g. `'1:15.3'` for 75.3.
 *
 * @example
 * ```ts
 * formatHudTime(0); // '0:00.0'
 * formatHudTime(75.3); // '1:15.3'
 * formatHudTime(3.04); // '0:03.0'
 * formatHudTime(60); // '1:00.0'
 * ```
 */
export function formatHudTime(seconds: number): string {
  const totalTenths: number = Math.round(seconds * TENTHS_PER_SECOND);
  const minutes: number = Math.floor(totalTenths / TENTHS_PER_MINUTE);
  const remainingTenths: number = totalTenths - minutes * TENTHS_PER_MINUTE;
  const wholeSeconds: number = Math.floor(remainingTenths / TENTHS_PER_SECOND);
  const tenths: number = remainingTenths % TENTHS_PER_SECOND;
  return `${String(minutes)}:${String(wholeSeconds).padStart(HUD_SECONDS_PAD_WIDTH, '0')}.${String(tenths)}`;
}

/**
 * What: resolves the reserved host output container by id, matching the
 * Neatenstein `resolveHudContainer` precedent (`hud.dom.utils.ts`)
 * flattened into this slice per the P1-S7-E file contract (no extra
 * shared files inside the slice boundary).
 *
 * @param outputId - Reserved host DOM id (spec section 10.2 folder
 * layout; `neatracing-hud-output` in the host shell and fixtures).
 * @param hudName - Human-readable surface name used in the banner text.
 * @returns The DOM container to append the HUD overlay into.
 * @throws When `document.getElementById(outputId)` misses (host shell
 * forgot the container or the call predates DOM mount).
 *
 * @example
 * ```ts
 * resolveOutputContainer('neatracing-hud-output', HUD_NAME);
 * ```
 */
function resolveOutputContainer(
  outputId: string,
  hudName: string,
): HTMLElement {
  const container: HTMLElement | null = document.getElementById(outputId);
  if (container === null) {
    throw new Error(
      `HUD container #${outputId} not found; cannot create ${hudName}`,
    );
  }
  return container;
}

/**
 * What: builds the HUD wrapper element styled like the Neatenstein
 * death-feedback overlay (absolute top-left, z-index 10, translucent
 * dark bg, 1px cyan border, monospace) so both examples share one
 * readable arcade palette (spec section 11.2).
 *
 * @param container - The host container to mount the wrapper into.
 * @returns The styled, appended wrapper element.
 */
function createHudWrapper(container: HTMLElement): HTMLElement {
  const wrapper: HTMLElement = document.createElement('div');
  wrapper.style.position = HUD_POSITION_ABSOLUTE;
  wrapper.style.top = HUD_OFFSET_0PX;
  wrapper.style.left = HUD_OFFSET_0PX;
  wrapper.style.zIndex = String(HUD_Z_INDEX);
  wrapper.style.backgroundColor = HUD_OVERLAY_BG;
  wrapper.style.borderStyle = HUD_BORDER_STYLE_SOLID;
  wrapper.style.borderWidth = HUD_BORDER_WIDTH_1PX;
  wrapper.style.borderColor = HUD_BORDER_COLOR;
  wrapper.style.fontFamily = HUD_FONT_MONOSPACE;
  wrapper.style.fontSize = HUD_FONT_14PX;
  wrapper.style.padding = HUD_PADDING_4PX_8PX;
  container.appendChild(wrapper);
  return wrapper;
}

/**
 * What: builds one HUD readout label inside the wrapper with a pinned
 * accent color (speed = cyan, lap = magenta, time = cyan — the labels as
 * primary/progress/primary readouts of the recycled palette). Labels
 * share the wrapper's typography (monospace 14px) so each label only
 * needs its own color.
 *
 * @param wrapper - The HUD wrapper the label is appended to.
 * @param color - The accent token from the Neatenstein palette.
 * @returns The created label element.
 */
function createHudLabel(wrapper: HTMLElement, color: string): HTMLElement {
  const label: HTMLElement = document.createElement('div');
  label.style.color = color;
  label.style.fontFamily = HUD_FONT_MONOSPACE;
  label.style.fontSize = HUD_FONT_14PX;
  wrapper.appendChild(label);
  return label;
}

/**
 * What: binds the three HUD labels to a live `RaceRulesState` slice and
 * mounts them into the reserved host output container (spec section
 * 10.2 folder layout; AC-024). This is the only place that reads
 * `document` for the HUD display (the selector module resolves its own
 * container the same way, Neatenstein precedent).
 *
 * Why: the host shell owns the container id; the HUD never assumes a
 * particular host DOM beyond that one hook, keeping the module portable
 * between the dev-server shell and the jsdom test fixture.
 *
 * When: called once during host boot (slice P1-S8 wires the render
 * loop's `update` call; AC-033's restart hook pre-verified by
 * `createRaceModeSelector`). Throws instead of returning a null HUD so
 * boot order bugs surface immediately instead of as silent blank HUDs.
 *
 * Allocation contract: this function performs all HUD DOM allocation
 * for the page lifetime (one wrapper + three labels, each once);
 * `update()` then only writes textContent strings per frame.
 *
 * @param outputId - Reserved host DOM id; `neatracing-hud-output` in
 * the host shell and the slice fixture suite.
 * @returns The HUD controller (labels + stateless update binding).
 * @throws When the output container is missing from the document.
 *
 * @example
 * ```ts
 * const hud = createHud('neatracing-hud-output');
 * hud.update({
 *     speedUnitsPerS: 20,
 *     raceState: createRaceRulesState(),
 * });
 * // speedLabel.textContent === '72 km/h', lapLabel === 'Lap 1/20',
 * // timeLabel === '0:00.0'
 * ```
 */
export function createHud(outputId: string): NeatRacingHud {
  const container: HTMLElement = resolveOutputContainer(outputId, HUD_NAME);
  const wrapper: HTMLElement = createHudWrapper(container);
  const speedLabel: HTMLElement = createHudLabel(
    wrapper,
    HUD_SPEED_LABEL_COLOR,
  );
  const lapLabel: HTMLElement = createHudLabel(wrapper, HUD_LAP_LABEL_COLOR);
  const timeLabel: HTMLElement = createHudLabel(wrapper, HUD_TIME_LABEL_COLOR);
  // Static affordance label (never repainted by `update`): manual mode is
  // static until keyboard input, so the page must tell the viewer the
  // car is drivable or the race reads as broken on first load.
  const hintLabel: HTMLElement = createHudLabel(wrapper, HUD_HINT_LABEL_COLOR);
  hintLabel.textContent = HUD_CONTROLS_HINT_TEXT;

  const hud: NeatRacingHud = {
    container: wrapper,
    speedLabel,
    lapLabel,
    timeLabel,
    update(state: NeatRacingHudState): void {
      speedLabel.textContent = formatHudSpeed(state.speedUnitsPerS);
      lapLabel.textContent = formatHudLap(state.raceState.lapsCompleted);
      timeLabel.textContent = formatHudTime(state.raceState.elapsedS);
    },
  };

  // Paint the mount once at boot so the HUD never shows blank labels
  // during the first frame (a zeroed read model is the deterministic
  // initial state a fresh race also starts from).
  hud.update({ speedUnitsPerS: 0, raceState: createRaceRulesState() });

  return hud;
}
