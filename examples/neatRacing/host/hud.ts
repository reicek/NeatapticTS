/**
 * NeatRacing host HUD overlay: the speed/lap/time DOM overlay into the
 * reserved host output container plus the mode-switch control surface
 * (spec sections 4.2, 10.2, 11.2, 13.1). The cyan/magenta palette is
 * recycled verbatim from the Neatenstein host HUD (spec section 11.2
 * recycling inventory: "same CSS tokens, cyan/magenta"), pinned here as
 * `NEATRACING_HUD_COLOR_CYAN` and `NEATRACING_HUD_COLOR_MAGENTA`
 * (rgb() literal strings, matching the Neatenstein `hud.constants.ts`
 * precedent).
 *
 * Display contract: `examples/neatRacing/host/mode-selector.test.ts`
 * pins the display string grammar —
 * speed = whole km/h (TP-01 physical anchor: 1 world unit = 1 meter, so
 * ~89 units/s ≈ ~320 km/h arcade feel; never a decimal part), lap =
 * `Lap c/20` clamped at `NEATRACING_HUMAN_RACE_LAPS` (a completed race
 * never displays 21/20; the selector restart contract is pre-verified
 * by `mode-selector.ts`), and time = `M:SS.d` (unpadded
 * minutes, zero-padded seconds, one tenths digit).
 *
 * The TRACTION READOUT is a DISPLAY of the ONE published traction severity
 * (`PhysicsCarState.tractionSeverity`, published by the physics step):
 * the HUD SHOWS the number it is handed and NEVER re-derives it, so the
 * overlay, the car pass tires and the browser debug seam can never
 * disagree about how much grip is left. The severity reaches the read
 * model as an OPTIONAL field (a read model without it paints the idle
 * zero), so the sibling suites that build the classic speed/lap/time
 * model keep compiling and keep their legacy look.
 *
 * The arcade reconciliation is deliberate and documented rather than
 * incidental: spec section 11.1 wants an arcade-speed feel while the
 * physics layer (spec section 5) is intentionally NOT scale-accurate
 * (TP-01). The HUD is a display-only read model — it never mutates race
 * state — so the km/h conversion constant lives here at the seam where
 * the decision is made, mirroring how `race-rules.ts` keeps its own
 * budget constants instead of importing physics timing.
 *
 * The COUNTDOWN OVERLAY is the Flappy-style 3-2-1-GO start signal: the
 * HUD paints the phase the RACE clock publishes (`raceState.elapsedS`,
 * NEVER the wall clock) with the two-pass neon recipe (the glow pass
 * behind, the crisp core on top). The phase law lives in
 * `hud.countdown.ts` — ONE resolver, TWO consumers — so the gate never
 * releases before the overlay shows GO. The overlay MAY paint GO up to ONE
 * FRAME before the gate releases: the frame's throttle must ride the
 * request BEFORE that frame's reply exists, so the gate reads the
 * previous reply's published clock while the overlay paints the current
 * one.
 *
 * The DEPARTURE HINT is an ADVISORY display of the ONE published FINE
 * surface classification (`TrackSurfaceClassification`,
 * published by `physics/collision.utils.ts`): the hint fires when the
 * published classification says the car left the drivable surface (the
 * grass BAND is still drivable, the OPEN grass is not). The law lives in
 * `hud.departure.ts` — ONE pure resolver — and the hint CHANGES NO pose,
 * NO speed, NO penalty and NO physics state: it is a DISPLAY, in the
 * same spirit as the traction readout.
 *
 * Allocation policy: `createHud` is the only DOM allocator (the five
 * readout labels plus the countdown overlay and the departure hint are
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

import { type TrackSurfaceClassification } from '../physics/collision.utils';
import {
  type RaceRulesState,
  createRaceRulesState,
  NEATRACING_HUMAN_RACE_LAPS,
} from '../physics/race-rules';
import {
  NEATRACING_HUD_COUNTDOWN_CORE_COLOR,
  NEATRACING_HUD_COUNTDOWN_GLOW_SHADOW,
  NEATRACING_HUD_COUNTDOWN_PHASE_INFO,
  resolveHudCountdownPhase,
} from './hud.countdown';
import { resolveHudDepartureHint } from './hud.departure';

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

/**
 * Traction label accent color: magenta (the reserved second accent of
 * the recycled palette, kept distinct so the grip readout is visually
 * separable from the speed/lap/time readouts).
 */
const HUD_TRACTION_LABEL_COLOR: string = NEATRACING_HUD_COLOR_MAGENTA;

/** Traction word prefix for the traction label. */
const HUD_TRACTION_WORD: string = 'Traction';

/**
 * Percent scale of the traction severity (the published severity is a
 * [0, 1] fraction, the readout shows it as a whole percent).
 */
const HUD_TRACTION_PERCENT_SCALE: number = 100;

/** Controls-hint accent color: dim slate (a non-readout affordance). */
const HUD_HINT_LABEL_COLOR: string = 'rgb(140, 160, 175)';

/** Static controls hint shown under the readouts (discoverability). */
const HUD_CONTROLS_HINT_TEXT: string = 'WASD / ARROWS to drive';

/**
 * The countdown overlay's positioning mode: the countdown is a viewport
 * overlay, not a flow child of the HUD wrapper (the wrapper is the
 * top-left readout box; the countdown floats over the track).
 */
const HUD_COUNTDOWN_POSITION_FIXED: string = 'fixed';

/** The countdown overlay's viewport-centring offset (both axes). */
const HUD_COUNTDOWN_CENTER_50_PERCENT: string = '50%';

/** The countdown overlay's centring transform (the element is top/left anchored). */
const HUD_COUNTDOWN_TRANSFORM_CENTER: string = 'translate(-50%, -50%)';

/** The countdown overlay's font size (the start signal must read at a glance). */
const HUD_COUNTDOWN_FONT_96PX: string = '96px';

/** The countdown overlay's font weight (the bold weight of the neon recipe). */
const HUD_COUNTDOWN_FONT_WEIGHT_700: string = '700';

/** The countdown overlay's text alignment. */
const HUD_COUNTDOWN_TEXT_ALIGN_CENTER: string = 'center';

/**
 * The countdown overlay's pointer behaviour: the host output container
 * re-enables pointer events for its direct children, so the countdown
 * opts out explicitly and never swallows a click on the track.
 */
const HUD_COUNTDOWN_POINTER_EVENTS_NONE: string = 'none';

/**
 * The departure hint's accent color: magenta (the reserved second accent
 * of the recycled palette — the hint is a WARNING line, so it takes the
 * accent the lap readout does not own, and the palette stays the
 * two-token cyan/magenta set the module header pins).
 */
const HUD_DEPARTURE_LABEL_COLOR: string = NEATRACING_HUD_COLOR_MAGENTA;

/**
 * The departure hint's positioning mode: the hint is an OVERLAY that
 * floats just under the readout box (the box must NOT grow and shrink
 * as the hint comes and goes), so it is positioned instead of flowing.
 */
const HUD_DEPARTURE_POSITION_ABSOLUTE: string = 'absolute';

/**
 * The departure hint's offset below the readout box (the wrapper's
 * bottom edge).
 */
const HUD_DEPARTURE_TOP_100_PERCENT: string = '100%';

/** The departure hint's font weight (the warning must read at a glance). */
const HUD_DEPARTURE_FONT_WEIGHT_700: string = '700';

/**
 * The departure hint's pointer behaviour: the host output container
 * re-enables pointer events for its direct children, so the hint opts
 * out explicitly and never swallows a click on the track.
 */
const HUD_DEPARTURE_POINTER_EVENTS_NONE: string = 'none';

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
  /**
   * The PUBLISHED traction severity in [0, 1]: how much grip
   * the tires have lost. The HUD SHOWS this number and NEVER re-derives
   * it — the physics step publishes the ONE shared scalar, so the
   * overlay, the car pass and the browser seam can never disagree.
   *
   * OPTIONAL: a read model without the field paints the IDLE zero (a
   * straight line), so the classic speed/lap/time model stays valid for
   * the callers that publish no severity.
   */
  tractionSeverity?: number;
  /**
   * The PUBLISHED FINE surface classification: the ONE
   * record the physics publishes for the surface under the car. The HUD
   * reads its `isDrivable` flag to publish the ADVISORY departure hint
   * and NEVER re-derives a surface of its own.
   *
   * OPTIONAL: a read model without the field publishes NO hint (the
   * hint needs published evidence), so the classic speed/lap/time model
   * stays valid for the callers that publish no classification.
   */
  surfaceClassification?: TrackSurfaceClassification;
}

/**
 * The constructed HUD controller handed to the host shell.
 *
 * The interface exposes the overlay wrapper plus the speed, lap and time
 * readout labels; `update` repaints every published label, including the
 * traction readout, the controls hint, the countdown overlay and the
 * departure hint.
 */
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
 * @returns The speed formatted as whole `km/h`, e.g. `'320 km/h'` for 89.
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
 * What: formats the PUBLISHED traction severity for the traction
 * readout as a whole percent (e.g. `'Traction 42%'`).
 *
 * Why: the severity is the ONE shared scalar the physics step publishes,
 * so the HUD only CONVERTS it for display — it never
 * re-derives it from the speed, the slip or any other HUD input, and
 * the overlay therefore cannot disagree with the car pass tires. A
 * missing severity (the OPTIONAL read-model field) is the IDLE zero:
 * the readout is always on screen, so the player never has to guess
 * whether the overlay lost the value.
 *
 * When: called by `NeatRacingHud.update` per frame and by the boot
 * paint.
 *
 * @param tractionSeverity - The published severity in [0, 1];
 * `undefined` when the read model carries no severity (the idle zero).
 * @returns The traction readout, e.g. `'Traction 42%'` for 0.42.
 *
 * @example
 * ```ts
 * formatHudTraction(0.42); // 'Traction 42%'
 * formatHudTraction(0); // 'Traction 0%'
 * formatHudTraction(undefined); // 'Traction 0%' (the idle zero)
 * ```
 */
function formatHudTraction(tractionSeverity: number | undefined): string {
  const severity: number = tractionSeverity ?? 0;
  const percent: number = Math.round(severity * HUD_TRACTION_PERCENT_SCALE);
  return `${HUD_TRACTION_WORD} ${String(percent)}%`;
}

/**
 * What: resolves the reserved host output container by id, matching the
 * Neatenstein `resolveHudContainer` precedent (`hud.dom.utils.ts`)
 * flattened into this module (no extra shared files inside the host
 * boundary).
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
 * What: builds the COUNTDOWN overlay element inside the HUD wrapper —
 * the Flappy-style 3-2-1-GO start token, painted with the two-pass neon
 * recipe (the GLOW pass as a `text-shadow` behind, the CRISP core colour
 * on top) and centred in the viewport.
 *
 * Why: the countdown is a START signal for the whole race, so it must
 * read over the track instead of inside the top-left readout box; the
 * element is `position: fixed` and therefore viewport-relative, while it
 * stays a DESCENDANT of the HUD wrapper so the overlay surface keeps
 * owning it.
 *
 * Why the pointer opt-out: the host output container re-enables pointer
 * events for its direct children, so the countdown (a grandchild) would
 * otherwise sit on top of the track and swallow clicks.
 *
 * @param wrapper - The HUD wrapper the countdown overlay is appended to.
 * @returns The created countdown overlay element.
 */
function createHudCountdownLabel(wrapper: HTMLElement): HTMLElement {
  const label: HTMLElement = document.createElement('div');
  label.style.position = HUD_COUNTDOWN_POSITION_FIXED;
  label.style.top = HUD_COUNTDOWN_CENTER_50_PERCENT;
  label.style.left = HUD_COUNTDOWN_CENTER_50_PERCENT;
  label.style.transform = HUD_COUNTDOWN_TRANSFORM_CENTER;
  label.style.fontFamily = HUD_FONT_MONOSPACE;
  label.style.fontSize = HUD_COUNTDOWN_FONT_96PX;
  label.style.fontWeight = HUD_COUNTDOWN_FONT_WEIGHT_700;
  label.style.textAlign = HUD_COUNTDOWN_TEXT_ALIGN_CENTER;
  // The CRISP PASS (the solid core colour) over the GLOW PASS (the
  // bloom behind it): the same two-pass recipe the Flappy neon pipes
  // use, translated from the canvas API to CSS.
  label.style.color = NEATRACING_HUD_COUNTDOWN_CORE_COLOR;
  label.style.textShadow = NEATRACING_HUD_COUNTDOWN_GLOW_SHADOW;
  label.style.pointerEvents = HUD_COUNTDOWN_POINTER_EVENTS_NONE;
  wrapper.appendChild(label);
  return label;
}

/**
 * What: builds the DEPARTURE hint overlay element inside the HUD wrapper
 * — the ADVISORY "return to the track" line.
 *
 * Why the overlay sits INSIDE the wrapper: the wrapper owns the HUD
 * surface, so the hint is mounted with the readouts and the overlay
 * surface keeps owning it (the countdown precedent). The hint is
 * POSITIONED (absolute, just under the readout box) so the box never
 * grows and shrinks while the hint comes and goes.
 *
 * Why the pointer opt-out: the host output container re-enables pointer
 * events for its direct children, so the hint (a grandchild) would
 * otherwise float over the track and swallow clicks.
 *
 * Why the hint starts EMPTY: an unpainted hint contributes NO HUD text,
 * so the boot paint and every on-surface frame leave the readout box
 * exactly as it was.
 *
 * @param wrapper - The HUD wrapper the departure overlay is appended to.
 * @returns The created departure overlay element.
 */
function createHudDepartureLabel(wrapper: HTMLElement): HTMLElement {
  const label: HTMLElement = document.createElement('div');
  label.style.position = HUD_DEPARTURE_POSITION_ABSOLUTE;
  label.style.top = HUD_DEPARTURE_TOP_100_PERCENT;
  label.style.left = HUD_OFFSET_0PX;
  label.style.color = HUD_DEPARTURE_LABEL_COLOR;
  label.style.fontFamily = HUD_FONT_MONOSPACE;
  label.style.fontSize = HUD_FONT_14PX;
  label.style.fontWeight = HUD_DEPARTURE_FONT_WEIGHT_700;
  label.style.pointerEvents = HUD_DEPARTURE_POINTER_EVENTS_NONE;
  wrapper.appendChild(label);
  return label;
}

/**
 * What: binds the HUD labels to a live `RaceRulesState` slice and
 * mounts them into the reserved host output container (spec section
 * 10.2 folder layout). This is the only place that reads
 * `document` for the HUD display (the selector module resolves its own
 * container the same way, Neatenstein precedent).
 *
 * Why: the host shell owns the container id; the HUD never assumes a
 * particular host DOM beyond that one hook, keeping the module portable
 * between the dev-server shell and the jsdom test fixture.
 *
 * When: called once during host boot (the render loop's `update` call is
 * wired by the host shell; the restart hook is pre-verified by
 * `createRaceModeSelector`). Throws instead of returning a null HUD so
 * boot order bugs surface immediately instead of as silent blank HUDs.
 *
 * Allocation contract: this function performs all HUD DOM allocation
 * for the page lifetime (one wrapper + seven labels — speed, lap, time,
 * traction, the controls hint, the countdown overlay and the departure
 * hint — each once); `update()` then only writes textContent strings
 * per frame.
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
 * // timeLabel === '0:00.0', traction readout === 'Traction 0%'
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
  // Traction readout label (AC-088): a live readout that SHOWS the
  // published traction severity. It is repainted by `update` like the
  // other readouts, and it is not exposed on the public controller
  // interface (the read model is the contract, not the label element).
  const tractionLabel: HTMLElement = createHudLabel(
    wrapper,
    HUD_TRACTION_LABEL_COLOR,
  );
  // Static affordance label (never repainted by `update`): manual mode is
  // static until keyboard input, so the page must tell the viewer the
  // car is drivable or the race reads as broken on first load.
  const hintLabel: HTMLElement = createHudLabel(wrapper, HUD_HINT_LABEL_COLOR);
  hintLabel.textContent = HUD_CONTROLS_HINT_TEXT;
  // Countdown overlay: the Flappy-style 3-2-1-GO start signal.
  // It is repainted by `update` from the RACE clock and it is not exposed
  // on the public controller interface (the read model is the contract,
  // not the label element).
  const countdownLabel: HTMLElement = createHudCountdownLabel(wrapper);
  // Departure hint overlay (AC-107-LAW): the ADVISORY "return to the
  // track" line. It is repainted by `update` from the PUBLISHED FINE
  // surface classification and it is not exposed on the public
  // controller interface (the read model is the contract, not the label
  // element).
  const departureLabel: HTMLElement = createHudDepartureLabel(wrapper);

  const hud: NeatRacingHud = {
    container: wrapper,
    speedLabel,
    lapLabel,
    timeLabel,
    update(state: NeatRacingHudState): void {
      speedLabel.textContent = formatHudSpeed(state.speedUnitsPerS);
      lapLabel.textContent = formatHudLap(state.raceState.lapsCompleted);
      timeLabel.textContent = formatHudTime(state.raceState.elapsedS);
      tractionLabel.textContent = formatHudTraction(state.tractionSeverity);
      // The countdown token comes from the RACE clock (`elapsedS`), NEVER
      // from the wall clock, and from the SHARED phase law in
      // `hud.countdown.ts` — the same law the host gate reads before it
      // releases the throttle. The 'done' phase carries the empty token,
      // which clears the overlay when the race is under way.
      const countdownPhase = resolveHudCountdownPhase(state.raceState.elapsedS);
      countdownLabel.textContent =
        NEATRACING_HUD_COUNTDOWN_PHASE_INFO[countdownPhase].label;
      // The ADVISORY departure hint comes from the ONE published FINE
      // surface classification and from NOTHING else (no clock, no
      // re-derived surface model): the law lives in `hud.departure.ts`.
      // The on-surface record carries the EMPTY token, so the hint
      // clears itself while the car drives.
      const departureHint = resolveHudDepartureHint(
        state.surfaceClassification,
      );
      departureLabel.textContent = departureHint.label;
    },
  };

  // Paint the mount once at boot so the HUD never shows blank labels
  // during the first frame (a zeroed read model is the deterministic
  // initial state a fresh race also starts from). The boot paint
  // carries no severity, so the traction readout starts at the idle
  // zero until the first physics tick publishes one.
  hud.update({ speedUnitsPerS: 0, raceState: createRaceRulesState() });

  return hud;
}
