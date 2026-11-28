/**
 * The ADVISORY departure resolver: the HUD's "return to the track" hint.
 *
 * WHAT: ONE pure resolver over the ONE published FINE surface
 * classification (`TrackSurfaceClassification`, published by
 * `physics/collision.utils.ts`). The resolver publishes the ADVISORY
 * departure state: the car has LEFT the drivable surface when the
 * published classification says the surface is not drivable.
 *
 * WHY the published classification and never a second surface model:
 * the physics already publishes ONE FINE classification per car pose
 * (the same surface family the coarse penalty path consumes), so the
 * HUD reads THAT record instead of re-deriving a surface of its own. A
 * second surface model would be a second law: it could disagree with the
 * published one about where the drivable surface ends, and the hint
 * would then fire while the car still drives on the grass apron.
 *
 * WHY the law reads `isDrivable` alone: the grass BAND (the apron
 * between the curb and the open grass) IS drivable, so only the OPEN
 * grass counts as a departure. `isRoadOnly` is deliberately NOT read:
 * the curb and the grass apron are drivable surfaces, and the hint is
 * about being OFF the drivable surface, not about being off the road
 * core.
 *
 * WHY the hint is ADVISORY: it changes NO pose, NO speed, NO penalty and
 * NO physics state. It is a DISPLAY of the published classification, in
 * the same spirit as the traction readout: the overlay shows the
 * published fact and never re-derives it, so the hint, the coarse
 * penalty counter and the browser debug seam can never disagree.
 *
 * WHY there is no clock: the hint is a function of the published
 * classification ALONE, so a replayed race repaints the same hint.
 * `Date.now()` is NEVER read in this module.
 *
 * The hint table is the whole law, and it is a TABLE (not a branch
 * tree): a module-level constant, so resolving a hint allocates
 * nothing — the render loop reads the token and the HUD writes it to the
 * DOM without building a per-frame object.
 */

import { type TrackSurfaceClassification } from '../physics/collision.utils';

/**
 * The ADVISORY hint the HUD paints while the car is off the drivable
 * surface.
 *
 * WHY this wording: the hint is a RETURN instruction, not a failure
 * message — the car is never punished for a departure, the hint only
 * tells the driver how to get back onto the road.
 *
 * @example
 * ```ts
 * NEATRACING_HUD_DEPARTURE_HINT_TEXT; // 'RETURN TO THE TRACK'
 * ```
 */
export const NEATRACING_HUD_DEPARTURE_HINT_TEXT: string = 'RETURN TO THE TRACK';

/**
 * The departure state the resolver publishes.
 *
 * The names are the STABLE vocabulary of the departure law: the HUD
 * paints the state's token and nothing else consumes the state. The
 * names carry the SEMANTICS, never the rendered text (the token lives
 * in `NEATRACING_HUD_DEPARTURE_HINTS`).
 *
 * @example
 * ```ts
 * const state: NeatRacingHudDepartureState = 'departed';
 * ```
 */
export type NeatRacingHudDepartureState = 'onSurface' | 'departed';

/**
 * What each departure state renders.
 *
 * `label` is the token the HUD paints (`''` clears the hint), so the
 * on-surface state contributes NO HUD text at all.
 *
 * @example
 * ```ts
 * const hint: NeatRacingHudDepartureHint = {
 *     state: 'departed',
 *     label: NEATRACING_HUD_DEPARTURE_HINT_TEXT,
 * };
 * ```
 */
export interface NeatRacingHudDepartureHint {
  /** The departure state this record describes. */
  readonly state: NeatRacingHudDepartureState;
  /** The token the HUD paints for this state ('' clears the hint). */
  readonly label: string;
}

/**
 * The departure hint table (the ONE law the HUD reads).
 *
 * The table is a module-level constant, so resolving a hint allocates
 * nothing: the render loop reads the token and the HUD writes it to the
 * DOM without building a per-frame object.
 *
 * @example
 * ```ts
 * NEATRACING_HUD_DEPARTURE_HINTS.onSurface.label; // '' (the hint paints nothing)
 * NEATRACING_HUD_DEPARTURE_HINTS.departed.label;  // 'RETURN TO THE TRACK'
 * ```
 */
export const NEATRACING_HUD_DEPARTURE_HINTS: Readonly<
  Record<NeatRacingHudDepartureState, NeatRacingHudDepartureHint>
> = {
  onSurface: { state: 'onSurface', label: '' },
  departed: { state: 'departed', label: NEATRACING_HUD_DEPARTURE_HINT_TEXT },
};

/**
 * What: resolves the ADVISORY departure hint for a published FINE
 * surface classification.
 *
 * Why: the hint must agree with the physics about where the drivable
 * surface ends, so the law reads the ONE published classification
 * (`isDrivable`) and NOTHING else — no clock, no pose, no speed, no
 * re-derived surface. The grass apron (the BAND between the curb and
 * the open grass) IS drivable, so only the OPEN grass counts as a
 * departure.
 *
 * The resolver is TOTAL: it never throws and never reads a clock, so an
 * absent classification is answered with the `onSurface` record rather
 * than with an error.
 *
 * When: called by `NeatRacingHud.update` once per frame (to paint the
 * hint token).
 *
 * @param classification - The PUBLISHED FINE surface classification for
 * the car's pose; `undefined` when the read model carries no
 * classification (the hint needs published evidence, so it stays off).
 * @returns The departure hint record for the classification: the
 * `onSurface` record when the classification is absent or reports a
 * drivable surface, and the `departed` record when it reports a
 * non-drivable surface. The returned record is always one of the shared
 * `NEATRACING_HUD_DEPARTURE_HINTS` entries.
 *
 * @example
 * ```ts
 * resolveHudDepartureHint({ surface: 'road', isRoadOnly: true, isDrivable: true }).label; // ''
 * resolveHudDepartureHint({ surface: 'grass', isRoadOnly: false, isDrivable: false }).label; // 'RETURN TO THE TRACK'
 * resolveHudDepartureHint(undefined).label; // '' (no published evidence)
 * ```
 */
export function resolveHudDepartureHint(
  classification: TrackSurfaceClassification | undefined,
): NeatRacingHudDepartureHint {
  if (classification === undefined) {
    return NEATRACING_HUD_DEPARTURE_HINTS.onSurface;
  }
  if (classification.isDrivable) {
    return NEATRACING_HUD_DEPARTURE_HINTS.onSurface;
  }
  return NEATRACING_HUD_DEPARTURE_HINTS.departed;
}
