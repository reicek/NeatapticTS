/**
 * NeatRacing host track selector — plan slice P4-S3-select-restart (plan
 * `neatracing-voxel-racer.plans.md`, phase 4, step 25), implementing the
 * AC-101 track-selection control over the retained mode-selector
 * notify/restart law (spec sections 4.4 and 13.5; LD-08 / LD-12): switching
 * the selection restarts the race through the host's restart carrier in
 * EITHER direction, and the selection is retained.
 *
 * The module carries TWO surfaces and pins their EXPORT ORDER:
 *
 * 1. `createNeatRacingTrackSelection` (FIRST) — the DOM-free selection
 *    core. The frozen RED-P4-8 contract red home
 *    (`../track/track.catalog.test.ts`) runs in the NODE environment and
 *    discovers the selection surface with a capability probe that iterates
 *    the module's exported functions in declaration order, calling each
 *    one with the catalog and with `undefined`, and adopts the FIRST
 *    produced record that carries a switch + subscribe surface. This core
 *    is therefore the FIRST export and it must never touch the DOM: the
 *    contract witnesses the switch + notify/restart law with no browser
 *    environment at all.
 * 2. `createNeatRacingTrackSelector` (SECOND) — the `<select>` overlay
 *    composition over the mode-selector pattern (resolve the reserved
 *    container or throw; one `<select>` built once at mount; Neatenstein
 *    overlay styling; user `change` picks and programmatic switches share
 *    the SAME core law). In a DOM-free environment the factory throws
 *    (`document` is undefined) and the capability probe skips it — the
 *    FIRST-export ordering above keeps the probe deterministic in ANY
 *    environment regardless.
 *
 * AC-101 wiring law (the browser-entry boot composition carries it): the
 * demo boots on the catalog default (`NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID`
 * — the retirement home of the single pinned `NEATRACING_DEMO_TRACK_ID` boot
 * constant) and a switch notifies the restart hook EXACTLY once per actual
 * switch in BOTH directions with the SELECTED track id; the wiring then
 * re-resolves the round's geometry THROUGH the catalog resolver
 * (`resolveTrackVariantSpec` — the sole spec-resolution path), so every
 * selected variant inherits the catalog's SIGNED left-normal lane-point
 * convention: the generator emits a positive magnitude on the optional
 * per-sample `innerOffsetWorld` and each consumer applies its own team
 * sign, and the catalog's `reverse` / `mirrorX` transforms NEGATE the
 * stored sample value (a sign flip, never a magnitude mirroring) so each
 * team keeps its physical lane on the same road (cf. the lane-point math in
 * `examples/racing_curriculum/renderer/racing.renderer.ts`).
 *
 * Same-value switches are documented no-ops (mirroring the mode selector:
 * the host shell must not reboot a race because a user re-picked the
 * already-active track — a same-track restart must come from the round
 * lifecycle, never from a re-pick).
 *
 * Membership law: the selection is a control OVER the catalog, so an id the
 * catalog does not carry is a named membership error that mirrors the
 * catalog resolver's error shape (unknown id + the known ids). The guard
 * reads the catalog KEYS — never a generated spec — so validating a pick
 * never pays a track generation, and `resolveTrackVariantSpec` stays the
 * SOLE spec-resolution path of the demo.
 *
 * Determinism: the selection law is pure closure state mutation; the
 * overlay is mount-only — identical inputs produce identical state
 * transitions with no Math.random, no clock reads, no timers.
 */

import { NEATRACING_HUD_COLOR_CYAN } from './hud';
import {
  NEATRACING_DEMO_TRACK_CATALOG,
  NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
} from '../track/track.catalog';
import type { NeatRacingDemoTrackCatalogEntry } from '../track/track.catalog';

/** Track-selector placement mode (fixed overlay corner, Neatenstein pattern). */
const SELECT_POSITION_ABSOLUTE: string = 'absolute';

/**
 * Track-selector top offset — pinned BELOW the race-mode selector overlay
 * (top `0px`, ~36px tall with the 14px type + 4px 8px padding) so both
 * overlay controls stay visible in the reserved HUD container.
 */
const SELECT_TOP_44PX: string = '44px';

/** Track-selector right offset (pinned to the overlay's right edge). */
const SELECT_RIGHT_0PX: string = '0px';

/** Track-selector stacking order (shared with the mode selector overlay). */
const SELECT_Z_INDEX: number = 10;

/** Track-selector background (Neatenstein death-feedback palette token). */
const SELECT_BG: string = 'rgba(6, 11, 20, 0.85)';

/** Track-selector border color (recycled cyan token from hud.ts). */
const SELECT_BORDER_COLOR: string = NEATRACING_HUD_COLOR_CYAN;

/** Track-selector typeface (arcade terminal look, Neatenstein precedent). */
const SELECT_FONT_MONOSPACE: string = 'monospace';

/** Track-selector type size (Neatenstein precedent 14px). */
const SELECT_FONT_14PX: string = '14px';

/** Track-selector padding (Neatenstein precedent 4px 8px). */
const SELECT_PADDING_4PX_8PX: string = '4px 8px';

/**
 * Human-readable host name used in the missing-container error banner
 * (Neatenstein precedent surfaces which HUD piece failed to mount).
 */
const SELECTOR_NAME: string = 'NeatRacing track selector';

/** DOM id/name of the track-selector `<select>` element. */
const TRACK_SELECTOR_DOM_ID: string = 'neatracing-track-selector';

/** Track-switch subscriber shape: carries the SELECTED track id. */
export type NeatRacingTrackSwitchListener = (trackId: string) => void;

/**
 * The DOM-free track-selection controller (the surface the frozen RED-P4-8
 * contract discovers): it switches the selection to a named catalog member
 * and notifies subscribers with the selected track id, exactly once per
 * actual switch, in either direction.
 */
export interface NeatRacingTrackSelection {
  /** The currently selected catalog track id. */
  readonly trackId: string;
  /**
   * Switches the selection to the named track (host wiring hook; notifies
   * exactly once; same-value calls are documented no-ops so a re-pick
   * never restarts the race).
   */
  setSelectedTrack(nextTrackId: string): void;
  /**
   * Subscribes a listener to actual track switches (the restart hook; fired
   * exactly once per switch with the switched-to track id — the notify/
   * restart contract of the race-mode selector, extended to tracks).
   */
  onSwitch(listener: NeatRacingTrackSwitchListener): void;
}

/** Optional arguments of the DOM-free selection core factory (all default). */
export interface NeatRacingTrackSelectionOptions {
  /**
   * The catalog the selection works over; defaults to the demo catalog.
   * The capability probe calls the factory with the catalog itself and
   * with `undefined` — both shapes must boot the same law, and the probe
   * argument (a catalog keyed by variant ids) carries none of these option
   * fields, so both shapes fall to the same defaults.
   */
  readonly catalog?: Readonly<Record<string, NeatRacingDemoTrackCatalogEntry>>;
  /** The selected track id at creation; defaults to the catalog default. */
  readonly initialTrackId?: string;
}

/**
 * The constructed track-selector overlay handed to the host shell (the
 * `<select>` composition over {@link NeatRacingTrackSelection}).
 */
export interface NeatRacingTrackSelector {
  /** The mounted `<select>` overlay element. */
  readonly select: HTMLSelectElement;
  /** The currently selected track id (kept in sync with `select`). */
  readonly trackId: string;
  /**
   * Programmatically switches the selection (host shell hook; keeps
   * `select` synchronized and notifies exactly once; same-value calls are
   * documented no-ops).
   */
  setSelectedTrack(nextTrackId: string): void;
  /** Subscribes the restart hook (exactly once per actual switch). */
  onSwitch(listener: NeatRacingTrackSwitchListener): void;
}

/**
 * What: validates a track id against the configured catalog's KEYS — the
 * same membership law `resolveTrackVariantSpec` enforces — and throws the
 * catalog's named membership error when the id is unknown.
 *
 * Why: the selection is a control over the catalog, so a typo can never
 * select a track the demo does not ship. The guard reads the catalog keys
 * (never a generated spec), so validating a pick never pays a track
 * generation and the catalog resolver stays the sole spec-resolution path.
 * The error message is built from the SAME `Object.keys` shape the
 * resolver builds its message from, so both surfaces speak one membership
 * vocabulary.
 *
 * @param catalog - The catalog the selection works over.
 * @param trackId - The candidate variant id.
 * @throws When `trackId` is not a member of the catalog.
 */
function assertCatalogMemberTrackId(
  catalog: Readonly<Record<string, NeatRacingDemoTrackCatalogEntry>>,
  trackId: string,
): void {
  if (!Object.prototype.hasOwnProperty.call(catalog, trackId)) {
    throw new Error(
      `NEATRACING_DEMO_TRACK_CATALOG has no variant id "${trackId}" — known ids: ` +
        Object.keys(catalog).join(', ') +
        '.',
    );
  }
}

/**
 * What: creates the DOM-free track-selection core (the FIRST export, per
 * the frozen capability probe's declaration-order law): a closure-held
 * current track id, a same-value early-exit guard, a membership guard over
 * the configured catalog, and an `onSwitch` subscription list that notifies
 * EXACTLY once per actual switch with the selected track id — never on a
 * same-value switch, in BOTH switch directions.
 *
 * Why DOM-free: the frozen RED-P4-8 contract witnesses the notify/restart
 * law in the NODE environment (no browser); the capability probe calls
 * this factory with the demo catalog and with `undefined`, and both shapes
 * must boot the identical law (the probe argument carries none of the
 * option fields, so both fall to the same defaults).
 *
 * When: called by the browser-entry boot composition through the overlay
 * factory, and directly by the frozen contract. Never touches the DOM, so
 * a node-environment probe always succeeds.
 *
 * @param options - Optional catalog / initial-id overrides (all default to
 *   the demo catalog's own law).
 * @returns The DOM-free selection controller.
 * @throws When a provided `initialTrackId` is not a catalog member, and
 *   when {@link NeatRacingTrackSelection.setSelectedTrack} receives an id
 *   the catalog does not carry.
 *
 * @example
 * ```ts
 * // The frozen contract probes BOTH call shapes:
 * createNeatRacingTrackSelection(NEATRACING_DEMO_TRACK_CATALOG);
 * createNeatRacingTrackSelection(undefined);
 * createNeatRacingTrackSelection();
 * const selection = createNeatRacingTrackSelection();
 * selection.trackId; // 'neatracing-demo-loop' (the catalog default)
 * selection.setSelectedTrack('neatracing-demo-loop-reverse'); // notifies once
 * selection.setSelectedTrack('neatracing-demo-loop-reverse'); // no-op
 * ```
 */
export function createNeatRacingTrackSelection(
  options: NeatRacingTrackSelectionOptions = {},
): NeatRacingTrackSelection {
  const catalog: Readonly<Record<string, NeatRacingDemoTrackCatalogEntry>> =
    options.catalog ?? NEATRACING_DEMO_TRACK_CATALOG;
  // A provided initial id is validated against the CONFIGURED catalog (a
  // fixture catalog must boot from its OWN members); the default id is the
  // catalog's own pinned member and can never miss.
  if (options.initialTrackId !== undefined) {
    assertCatalogMemberTrackId(catalog, options.initialTrackId);
  }
  const initialTrackId: string =
    options.initialTrackId ?? NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID;

  let trackId: string = initialTrackId;
  const listeners: NeatRacingTrackSwitchListener[] = [];

  const notifySwitch: () => void = (): void => {
    for (const listener of listeners) {
      listener(trackId);
    }
  };

  const setSelectedTrack: (nextTrackId: string) => void = (
    nextTrackId: string,
  ): void => {
    // Same-value guard: re-picking the already-active track is a
    // documented no-op, never a silent restart. The re-pick branch is
    // domain-tested (the same-value pins), so no coverage exemption is
    // needed.
    if (nextTrackId === trackId) {
      return;
    }
    // Membership guard: an id the catalog does not carry is a named
    // membership error (the catalog resolver's law mirrored at the
    // selection surface).
    assertCatalogMemberTrackId(catalog, nextTrackId);
    trackId = nextTrackId;
    notifySwitch();
  };

  const onSwitch: (listener: NeatRacingTrackSwitchListener) => void = (
    listener: NeatRacingTrackSwitchListener,
  ): void => {
    listeners.push(listener);
  };

  return {
    setSelectedTrack,
    onSwitch,
    get trackId(): string {
      return trackId;
    },
  };
}

/**
 * What: resolves the reserved host output container by id, matching the
 * mode-selector resolver (`host/mode-selector.ts`) — resolve-or-throw so
 * boot-order bugs surface immediately.
 *
 * @param outputId - Reserved host DOM id (`neatracing-hud-output` in the
 * host shell and fixtures).
 * @param surfaceName - Human-readable surface name for the error body.
 * @returns The DOM container to append the track selector into.
 * @throws When `document.getElementById(outputId)` misses (host shell
 * forgot the container or the call predates DOM mount).
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
 * What: builds the `<select>` with every catalog member rendered in
 * CATALOG order (the pinned demo loop first) — the catalog's `displayName`
 * is the visible label and the variant id is the value — styled from the
 * named Neatenstein overlay constants, offset BELOW the race-mode selector
 * so both controls stay visible.
 *
 * Why: option order and initial value are part of the selection
 * acceptance contract, the DOM is built exactly once at mount (allocation
 * policy: no per-switch DOM construction), and the option list is a VIEW
 * over the catalog — never a second source of the track list.
 *
 * @returns The styled, unattached `<select>` element.
 */
function createTrackSelect(): HTMLSelectElement {
  const select: HTMLSelectElement = document.createElement('select');
  select.id = TRACK_SELECTOR_DOM_ID;
  select.name = TRACK_SELECTOR_DOM_ID;
  select.style.position = SELECT_POSITION_ABSOLUTE;
  select.style.top = SELECT_TOP_44PX;
  select.style.right = SELECT_RIGHT_0PX;
  select.style.zIndex = String(SELECT_Z_INDEX);
  select.style.backgroundColor = SELECT_BG;
  select.style.borderColor = SELECT_BORDER_COLOR;
  select.style.fontFamily = SELECT_FONT_MONOSPACE;
  select.style.fontSize = SELECT_FONT_14PX;
  select.style.padding = SELECT_PADDING_4PX_8PX;
  for (const entry of Object.values(NEATRACING_DEMO_TRACK_CATALOG)) {
    const option: HTMLOptionElement = document.createElement('option');
    option.value = entry.id;
    option.textContent = entry.displayName;
    select.appendChild(option);
  }
  return select;
}

/**
 * What: creates the track-selector overlay inside the reserved host output
 * container over the mode-selector structure: one `<select>` listing every
 * catalog member (pinned demo loop first), a closure-held selection via the
 * DOM-free core, one `change` listener plus one `setSelectedTrack` entry
 * point that both notify subscribers exactly once per actual switch and
 * never notify on same-value switches.
 *
 * Why (AC-101): the host connects `onSwitch` to the track-switch restart
 * carrier — the restart runs in BOTH directions (LD-12) on the SELECTED
 * track and the selection is retained. Same-value picks and same-value
 * programmatic switches ride the core's same-value guard (one law, one
 * place), and programmatic switches are mirrored into the DOM control so
 * user picks and programmatic switches share one state.
 *
 * When: called once during host boot after the race-mode selector mounts
 * into the same reserved container id (`neatracing-hud-output`). Throws
 * instead of returning null so boot-order bugs surface immediately; the
 * boot composition (browser-entry) degrades to an inert fallback when the
 * container cannot be resolved, so the demo keeps racing on the catalog
 * default.
 *
 * @param outputId - Reserved host DOM id; `neatracing-hud-output` in
 * the host shell and the slice fixture suite.
 * @returns The track-selector controller (select element + selection API).
 * @throws When the output container is missing from the document. In a
 * DOM-free environment (node) the factory throws (`ReferenceError:
 * document is not defined`) BEFORE any container resolution — the exact
 * condition the frozen capability probe depends on to skip this overlay.
 *
 * @example
 * ```ts
 * const trackSelector = createNeatRacingTrackSelector('neatracing-hud-output');
 * trackSelector.trackId; // 'neatracing-demo-loop' (the catalog default)
 * trackSelector.setSelectedTrack('neatracing-demo-loop-reverse'); // notifies
 * trackSelector.setSelectedTrack('neatracing-demo-loop-reverse'); // no-op
 * trackSelector.onSwitch((trackId) => restartRaceOn(trackId));
 * ```
 */
export function createNeatRacingTrackSelector(
  outputId: string,
): NeatRacingTrackSelector {
  const container: HTMLElement = resolveSelectorContainer(
    outputId,
    SELECTOR_NAME,
  );
  // One core per overlay: the DOM composition drives the SAME law the
  // frozen contract discovers — user picks ride the core's same-value
  // guard and exactly-once notify.
  const selection: NeatRacingTrackSelection = createNeatRacingTrackSelection();
  const select: HTMLSelectElement = createTrackSelect();

  // The DOM mirror is the FIRST listener on the core: a switch syncs the
  // `<select>` BEFORE any later-registered carrier runs, so a throwing
  // host carrier cannot desync the control from the core state (the
  // mode-selector `setMode` precedent writes the DOM before it notifies).
  selection.onSwitch((switchedTrackId: string): void => {
    select.value = switchedTrackId;
  });

  const setSelectedTrack: (nextTrackId: string) => void = (
    nextTrackId: string,
  ): void => {
    // Same-value guard (mirrors the mode-selector setMode law): a re-pick
    // never notifies and never touches the DOM (the DOM already shows the
    // current selection; the mirror above owns every DOM write).
    if (nextTrackId === selection.trackId) {
      return;
    }
    selection.setSelectedTrack(nextTrackId);
  };

  const onSwitch: (listener: NeatRacingTrackSwitchListener) => void = (
    listener: NeatRacingTrackSwitchListener,
  ): void => {
    selection.onSwitch(listener);
  };

  // One change listener for the selector's lifetime: the pick delegates to
  // the core, so a replayed same-value change event never notifies (the
  // core's same-value guard owns the law in one place).
  select.addEventListener('change', (): void => {
    selection.setSelectedTrack(select.value);
  });

  // Mirror the initial selection into the DOM control after the listener
  // is attached so user picks and programmatic switches share one state.
  select.value = selection.trackId;
  container.appendChild(select);

  return {
    select,
    setSelectedTrack,
    onSwitch,
    get trackId(): string {
      return selection.trackId;
    },
  };
}
