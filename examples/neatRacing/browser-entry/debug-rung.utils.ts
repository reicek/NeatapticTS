/**
 * @file The PURE debug-rung seam of the W2 step (Step 24A) — the OPTIONAL
 * `?level` query parameter the host seeds its OPENING boot level from, and
 * the boot-level precedence that resolves it
 * (plans/neatracing-voxel-racer.plans.md, slice `P4-S2A-debug-rung-seam`).
 *
 * Educational note — THE SEAM IS PURE AND DOM-FREE. Both functions take
 * their inputs as arguments and read no global, no DOM node and no clock,
 * so the NODE test environment exercises exactly the logic the browser
 * boots with, without a browser.
 * Educational note — THE DEFAULT IS BYTE-IDENTICAL. With no `?level`
 * parameter the boot level is exactly what the host booted with before
 * this seam existed (`nextRaceLevel` else the session level), so every
 * pre-existing boot — the shipped demo included — is untouched.
 * Educational note — THE RUNG IS A KEY, NOT A GRANT. A debug rung seeds
 * ONE opening boot level; it never writes the promotion ladder's state
 * (`nextRaceLevel` stays ladder-owned and STILL WINS) and it grants no rung
 * — the seeded boot level still maps through the ladder's own tier and
 * fleet resolvers.
 * Educational note — THE FLEET STAYS LADDER-OWNED. This module exports no
 * fleet-size table and no car-count constant: the car count for a seeded
 * rung comes from the ladder's `resolveNeatRacingLevelCarCount` alone.
 *
 * @module
 */

import { NEATRACING_PROMOTION_MAX_LEVEL } from '../ai/promotion.rules';

/**
 * The ladder's first rung level. The promotion ladder clamps to this rung
 * below it (`resolveNeatRacingLevelCarCount(0)` answers the rung-1 fleet),
 * so the debug rung accepts EXACTLY the integer band the ladder will map —
 * no name below it is coerced into a rung.
 */
const NEATRACING_LADDER_FIRST_LEVEL = 1;

/**
 * A debug rung name must be pure digits (`2`): a name carrying a fraction,
 * a sign or any non-digit (`2.5`, `-1`, `abc`, an empty value) is IGNORED,
 * never coerced.
 */
const NEATRACING_DEBUG_RUNG_INTEGER_PATTERN = /^\d+$/;

/**
 * The boot-level sources the precedence resolver reads, in the exact shape
 * the host passes at the round boot.
 */
interface NeatRacingDebugRungBootLevelSources {
  /** The promotion ladder's pending next-race level, or null when the ladder has none. */
  readonly nextRaceLevel: number | null;
  /** The debug rung resolved from the optional `?level` parameter, or null. */
  readonly debugRungLevel: number | null;
  /** The session's own level (the byte-identical boot-level fallback). */
  readonly sessionLevel: number;
}

/**
 * Resolves the debug rung the OPTIONAL `level` query parameter names, or
 * null when the parameter is absent, malformed or outside the ladder.
 *
 * Educational note — WHAT and WHY. The W2 debug rung lets a developer boot
 * the racer straight at a rung (`?level=2`) so the rival-bearing manual
 * rung is reachable in ONE boot — no promotion driving needed. The rung is
 * a debug/harness affordance: it is read ONCE at the opening boot, it
 * seeds a boot level, and it never writes the promotion ladder's state.
 * THE DEFAULT: with no `level` parameter this resolver answers null and
 * the host boots the byte-identical default (`nextRaceLevel` else the
 * session level), so the shipped demo is unaffected. THE ACCEPTED RUNGS:
 * exactly the ladder band, level 1 to `NEATRACING_PROMOTION_MAX_LEVEL`
 * (1, 2, 3) — the ceiling is IMPORTED from the promotion ladder so a
 * ladder change moves the seam. NOTHING IS COERCED: a fraction (`2.5`),
 * a signed name (`-1`), a non-numeric name (`abc`) and an out-of-band name
 * (`0` and above the ceiling) all answer null even though the LADDER'S own
 * clamp would have coerced some of them — the debug rung is a key to a
 * rung, not a re-derivation of it.
 * WHEN: the host calls this exactly once per page boot, inside the live
 * drive loop's round setup; the resolver may also be called from tests,
 * scripts and embeddings because it reads nothing global.
 *
 * @param queryString - The boot query string, sampled from
 * `window.location.search` by the host (a leading `?` is fine).
 * @returns The ladder rung (1..3) the parameter names, or null when the
 * parameter is absent, malformed or out of the ladder band.
 *
 * @example
 * // The seeded rival-bearing rung
 * expect(resolveNeatRacingDebugRungLevel('?level=2')).toBe(2);
 * // The absent-parameter default stays byte-identical (null)
 * expect(resolveNeatRacingDebugRungLevel('?track=demo')).toBeNull();
 * // Malformed and out-of-ladder names are ignored, never coerced
 * expect(resolveNeatRacingDebugRungLevel('?level=2.5')).toBeNull();
 * expect(resolveNeatRacingDebugRungLevel('?level=9')).toBeNull();
 */
export function resolveNeatRacingDebugRungLevel(
  queryString: string,
): number | null {
  const requestedLevel = new URLSearchParams(queryString).get('level');
  if (
    requestedLevel === null ||
    !NEATRACING_DEBUG_RUNG_INTEGER_PATTERN.test(requestedLevel)
  ) {
    return null;
  }
  const rung = Number.parseInt(requestedLevel, 10);
  if (
    rung >= NEATRACING_LADDER_FIRST_LEVEL &&
    rung <= NEATRACING_PROMOTION_MAX_LEVEL
  ) {
    return rung;
  }
  return null;
}

/**
 * Resolves the level the host boots a round at, from the sources in
 * precedence order: the promotion ladder's pending level first, the debug
 * rung second, the session level last.
 *
 * Educational note — THE LADDER KEEPS PRECEDENCE. A debug rung only seeds
 * the boot when the promotion ladder is silent: a pending `nextRaceLevel`
 * (a race-completion boundary set it) always boots first, so `?level` can
 * never fight a freshly promoted round.
 * Educational note — THE RUNG SEEDS A BOOT, IT GRANTS NO RUNG. The seeded
 * level feeds the SAME boot path every round takes, so the observation
 * tier and the fleet size still resolve through the ladder's own
 * resolvers — a debug boot races the identical rules, it only opens the
 * door to a rung.
 * Educational note — THE DEFAULT IS BYTE-IDENTICAL. With a null debug rung
 * this is exactly the host's former boot expression
 * (`nextRaceLevel ?? sessionLevel`).
 *
 * @param sources - The boot-level sources: `nextRaceLevel` (the ladder's
 * pending level, null until a race-completion boundary sets it),
 * `debugRungLevel` (the rung resolved from the optional `?level`
 * parameter, null when absent) and `sessionLevel` (the session level the
 * boot falls back to).
 * @returns The boot level.
 *
 * @example
 * // The default boot (no ?level parameter) is byte-identical
 * expect(resolveNeatRacingBootLevel({
 *   nextRaceLevel: null,
 *   debugRungLevel: null,
 *   sessionLevel: 1,
 * })).toBe(1);
 * // A debug rung seeds only a silent ladder's boot
 * expect(resolveNeatRacingBootLevel({
 *   nextRaceLevel: null,
 *   debugRungLevel: 3,
 *   sessionLevel: 1,
 * })).toBe(3);
 */
export function resolveNeatRacingBootLevel(
  sources: NeatRacingDebugRungBootLevelSources,
): number {
  return (
    sources.nextRaceLevel ?? sources.debugRungLevel ?? sources.sessionLevel
  );
}
