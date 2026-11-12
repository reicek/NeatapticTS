/**
 * @file RED contracts for the W2 DEBUG RUNG SEAM (`?level=2`).
 *
 * Slice `P4-S2A-debug-rung-seam-red` — Phase 4, Step 24A of
 * `plans/neatracing-voxel-racer.plans.md`.
 *
 * Educational note — WHAT THIS FILE OWNS:
 * The W2 gap the Step-23B close recorded as NOT DISCHARGED is a
 * REACHABILITY gap. The demo's default boot is LEVEL 1 and
 * `resolveNeatRacingLevelCarCount(1) === 1` — a LONE car 0 — and the sim
 * worker's agent-car branch is reached ONLY for `carIndex >= 1`, so a
 * level-1 manual smoke is VACUOUS for the seam: it can never observe an
 * agent car. The ladder maps 2 → 2 cars and 3 → 5 cars, so `?level=2` is
 * the SMALLEST rung that puts a rival on the track — reachable in ONE boot,
 * with no pose seam and no vision lane.
 *
 * Educational note — THE THREE CONTRACTS:
 * - RED-P4-17 — THE PURE DEBUG-RUNG RESOLVER. `?level=2` resolves the rung
 *   2, `?level=3` the rung 3 and `?level=1` the rung 1; an ABSENT parameter
 *   resolves NULL (that IS the byte-identical default) and a malformed
 *   parameter (`0`, `4`, `-1`, `2.5`, `abc`) is IGNORED — NEVER coerced the
 *   way the promotion ladder clamps it.
 * - RED-P4-18 — THE BOOT-LEVEL PRECEDENCE. `nextRaceLevel` FIRST (the
 *   ladder still wins), THEN the debug rung, THEN the session level. Pinned
 *   with NO DOM dependency: the resolver is pure, so the contract needs no
 *   browser environment (this file runs in the NODE environment — there is
 *   deliberately no `@jest-environment jsdom` docblock here).
 * - RED-P4-19 — THE FLEET-SIZE PROVENANCE. The fleet size of a debug rung
 *   comes from `resolveNeatRacingLevelCarCount` (2 → 2 cars, 3 → 5 cars) and
 *   NOT from a parallel table inside the seam — the seam exposes NO
 *   fleet-size constant of its own, so a ladder change moves the seam.
 *
 * Educational note — THE EXPORT CONTRACT (the implementing slice MUST
 * satisfy this; the RED home is byte-frozen and the implementing slice
 * NEVER edits it). `examples/neatRacing/browser-entry/debug-rung.utils.ts`
 * (NEW) exports TWO pure functions:
 * - `resolveNeatRacingDebugRungLevel(queryString: string): number | null`
 *   — parses the OPTIONAL `level` query parameter and resolves the rung, or
 *   `null` when the parameter is absent or malformed.
 * - `resolveNeatRacingBootLevel(sources: { nextRaceLevel: number | null;
 *   debugRungLevel: number | null; sessionLevel: number }): number` — the
 *   boot-level precedence resolver.
 *
 * Educational note — THE RED GATE (why the seam is NEVER statically
 * imported). At HEAD the module does not exist, so a static import would
 * fail the suite at COMPILE time (TS 2307) — the WRONG red reason. The
 * module is therefore loaded through a WIDENED dynamic import inside the
 * gate helper, and the gate answers with a NAMED error that names the
 * MISSING BEHAVIOUR (the `?level=2` rival-bearing rung is not reachable),
 * never a bare "module not found" (the house pattern of
 * `examples/neatRacing/ai/panel.network.transport.test.ts`).
 *
 * Educational note — THE LADDER IS THE ONLY FLEET-SIZE SOURCE. The ladder's
 * fleet-size resolver is spied (the house `jest.mock` shape of
 * `examples/neatRacing/host/promotion-live.red.test.ts`) so this contract
 * reads the fleet size THROUGH the ladder resolver — the contract never
 * hard-codes a seam-side fleet table.
 *
 * @module
 */

/**
 * The ladder's fleet-size resolver is the ONLY fleet-size source the seam may
 * use. It is spied (and still answers with the REAL ladder's numbers) so the
 * RED-P4-19 contract reads the fleet size THROUGH the ladder resolver.
 */
jest.mock('../ai/promotion.rules', () => {
  const actual = jest.requireActual<typeof import('../ai/promotion.rules')>(
    '../ai/promotion.rules',
  );
  return {
    ...actual,
    // The `__esModule` marker is re-declared on the mock: the spread above
    // drops the actual module's marker (it is not enumerable), and a
    // namespace import of a mock WITHOUT the marker is an interop COPY.
    __esModule: true,
    resolveNeatRacingLevelCarCount: jest.fn(
      actual.resolveNeatRacingLevelCarCount,
    ),
  };
});

import { describe, expect, it } from '@jest/globals';
import { resolveNeatRacingLevelCarCount } from '../ai/promotion.rules';

/** The seam module path (WIDENED on purpose — see the red gate below). */
const RED_DEBUG_RUNG_MODULE_PATH = './debug-rung.utils';

/** The seam's pure rung resolver (the `?level=` parser). */
type RedDebugRungLevelResolver = (queryString: string) => number | null;

/** The sources the seam's boot-level precedence resolver reads. */
interface RedDebugRungBootLevelSources {
  /** The ladder's carried next-race level (null when the ladder has none). */
  readonly nextRaceLevel: number | null;
  /** The resolved debug rung (null when no debug rung was requested). */
  readonly debugRungLevel: number | null;
  /** The session's own level (the byte-identical default). */
  readonly sessionLevel: number;
}

/** The seam's pure boot-level precedence resolver. */
type RedDebugRungBootLevelResolver = (
  sources: RedDebugRungBootLevelSources,
) => number;

/** The seam's module surface, as the red gate hands it back. */
interface RedDebugRungSeamSurface {
  /** The seam's pure rung resolver. */
  readonly resolveNeatRacingDebugRungLevel: RedDebugRungLevelResolver;
  /** The seam's pure boot-level precedence resolver. */
  readonly resolveNeatRacingBootLevel: RedDebugRungBootLevelResolver;
  /** The seam's exported values (the fleet-size constant check reads them). */
  readonly exports: Readonly<Record<string, unknown>>;
}

/**
 * The vocabulary a seam-side fleet-size table would carry in its export
 * names. A seam that keeps its own fleet table would expose a constant
 * named after the fleet (or the car count) — the seam must NOT.
 */
const RED_FLEET_SIZE_EXPORT_NAME_PATTERN = /fleet|car_?count|cars/i;

/**
 * Loads the (not-yet-existing) debug-rung seam behind the red gate.
 *
 * Educational note — the red gate. At HEAD the file
 * `examples/neatRacing/browser-entry/debug-rung.utils.ts` does not exist, so
 * the import throws and the gate answers the NAMED missing-behaviour error.
 * At GREEN the import resolves and the pinned exports are checked. The file
 * is NEVER statically imported (a static missing import would fail the suite
 * at compile time — the wrong red reason); the widened string specifier
 * keeps the dynamic import compilable both at HEAD and at GREEN.
 *
 * @param contractId - The RED contract the gate speaks for (RED-P4-17/18/19).
 * @param requiredExportNames - The seam exports the contract needs.
 * @returns The seam surface once every required export exists.
 */
async function gateRedDebugRungSeam(
  contractId: string,
  requiredExportNames: readonly string[],
): Promise<RedDebugRungSeamSurface> {
  // Arranged with a widened variable on purpose: the literal path does not
  // exist yet, and TS would reject the literal specifier at HEAD (TS 2307) —
  // a compile failure is NOT the red reason.
  const seamModulePath: string = RED_DEBUG_RUNG_MODULE_PATH;
  let seamModule: Record<string, unknown> | undefined;
  try {
    seamModule = (await import(seamModulePath)) as Record<string, unknown>;
  } catch {
    seamModule = undefined;
  }
  const missingExports = requiredExportNames.filter(
    (name) => typeof seamModule?.[name] !== 'function',
  );
  if (seamModule === undefined || missingExports.length > 0) {
    throw new Error(
      `${contractId} red gate: the debug-rung seam (${requiredExportNames.join(
        ' + ',
      )}) does not exist yet at examples/neatRacing/browser-entry/debug-rung.utils.ts — the \`?level=2\` rival-bearing rung is NOT reachable: the host still boots the lone level-1 car, so a manual smoke can never witness an agent car`,
    );
  }
  return {
    resolveNeatRacingDebugRungLevel: seamModule[
      'resolveNeatRacingDebugRungLevel'
    ] as RedDebugRungLevelResolver,
    resolveNeatRacingBootLevel: seamModule[
      'resolveNeatRacingBootLevel'
    ] as RedDebugRungBootLevelResolver,
    exports: seamModule,
  };
}

/**
 * Reads the seam's resolved rung and FAILS LOUDLY when it is absent.
 *
 * Educational note — the naming aid. A fleet-size question needs a rung to
 * ask about; when the resolver answers `null` the contract names the missing
 * behaviour instead of letting `null` flow into the ladder resolver.
 *
 * @param seam - The gated seam surface.
 * @param queryString - The query string the resolver is asked about.
 * @param contractId - The RED contract the helper speaks for.
 * @returns The resolved rung (never null).
 */
function requireRedDebugRung(
  seam: RedDebugRungSeamSurface,
  queryString: string,
  contractId: string,
): number {
  const rung = seam.resolveNeatRacingDebugRungLevel(queryString);
  if (rung === null) {
    throw new Error(
      `${contractId} red gate: the debug-rung resolver answered null for \`${queryString}\` — the \`?level=2\` rival-bearing rung is still unreachable`,
    );
  }
  return rung;
}

/**
 * Collects the seam's fleet-size CONSTANTS (its exported values that are not
 * functions and whose names speak of a fleet or a car count).
 *
 * Educational note — the provenance probe. A seam that kept its own
 * fleet-size table would expose it as a constant; a seam that reads the
 * ladder exposes no such constant at all.
 *
 * @param exports - The seam's exported values.
 * @returns The sorted names of the seam's fleet-size constants.
 */
function redFleetSizeConstantNames(
  exports: Readonly<Record<string, unknown>>,
): string[] {
  return Object.entries(exports)
    .filter(
      ([name, value]) =>
        typeof value !== 'function' &&
        RED_FLEET_SIZE_EXPORT_NAME_PATTERN.test(name),
    )
    .map(([name]) => name)
    .sort();
}

describe('RED-P4-17 — the pure debug-rung resolver', () => {
  it('resolves the rival-bearing rung 2 for `?level=2`', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-17', [
      'resolveNeatRacingDebugRungLevel',
    ]);

    // Act: the query string of the rival-bearing rung.
    const rung = seam.resolveNeatRacingDebugRungLevel('?level=2');

    // Assert: the rung is 2 — the smallest rung that puts a rival on the
    // track (car 0 human + car 1 rival).
    expect(rung).toBe(2);
  });

  it('resolves the five-car rung 3 for `?level=3`', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-17', [
      'resolveNeatRacingDebugRungLevel',
    ]);

    // Act: the query string of the five-car rung.
    const rung = seam.resolveNeatRacingDebugRungLevel('?level=3');

    // Assert: the rung is 3 — the ladder's five-car rung.
    expect(rung).toBe(3);
  });

  it('resolves the solo rung 1 for `?level=1`', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-17', [
      'resolveNeatRacingDebugRungLevel',
    ]);

    // Act: the query string of the solo rung.
    const rung = seam.resolveNeatRacingDebugRungLevel('?level=1');

    // Assert: the rung is 1 — the ladder's solo rung (the default boot).
    expect(rung).toBe(1);
  });

  it('returns null for an absent level parameter (the byte-identical default)', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-17', [
      'resolveNeatRacingDebugRungLevel',
    ]);

    // Act + Assert: an empty query string carries no level parameter, and a
    // query string that carries only another parameter carries no level
    // parameter either — both answer null, so the shipped default boot
    // stands byte-identically.
    expect(seam.resolveNeatRacingDebugRungLevel('')).toBeNull();
    expect(seam.resolveNeatRacingDebugRungLevel('?track=demo')).toBeNull();
  });

  it('ignores a level parameter outside the ladder rungs (0, 4 and -1)', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-17', [
      'resolveNeatRacingDebugRungLevel',
    ]);

    // Act + Assert: the ladder owns the rungs 1, 2 and 3, so a rung below
    // or above them is IGNORED (never clamped into the ladder).
    expect(seam.resolveNeatRacingDebugRungLevel('?level=0')).toBeNull();
    expect(seam.resolveNeatRacingDebugRungLevel('?level=4')).toBeNull();
    expect(seam.resolveNeatRacingDebugRungLevel('?level=-1')).toBeNull();
  });

  it('ignores a malformed level parameter instead of coercing it (2.5 and abc)', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-17', [
      'resolveNeatRacingDebugRungLevel',
    ]);

    // Act + Assert: a fractional and a non-numeric parameter are IGNORED —
    // a debug rung is never a truncated or a defaulted number.
    expect(seam.resolveNeatRacingDebugRungLevel('?level=2.5')).toBeNull();
    expect(seam.resolveNeatRacingDebugRungLevel('?level=abc')).toBeNull();
  });

  it('never coerces a malformed rung the way the promotion ladder clamps it', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-17', [
      'resolveNeatRacingDebugRungLevel',
    ]);

    // Act + Assert: the seam IGNORES the malformed values, while the
    // promotion ladder's own clamp WOULD have coerced them into a real rung
    // (2.5 → the level-2 rung, 0 → the level-1 rung). The debug rung never
    // borrows that coercion.
    expect(seam.resolveNeatRacingDebugRungLevel('?level=2.5')).toBeNull();
    expect(resolveNeatRacingLevelCarCount(2.5)).toBe(2);
    expect(resolveNeatRacingLevelCarCount(0)).toBe(1);
  });
});

describe('RED-P4-18 — the boot-level precedence', () => {
  it('keeps the ladder’s nextRaceLevel ahead of the debug rung', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-18', [
      'resolveNeatRacingBootLevel',
    ]);

    // Act: the ladder carries a next race level while the debug rung asks
    // for another rung.
    const bootLevel = seam.resolveNeatRacingBootLevel({
      nextRaceLevel: 2,
      debugRungLevel: 3,
      sessionLevel: 1,
    });

    // Assert: the ladder still wins — the debug rung SEEDS a boot and can
    // never GRANT a rung the human has not earned.
    expect(bootLevel).toBe(2);
  });

  it('seeds the boot level from the debug rung when the ladder has no next race level', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-18', [
      'resolveNeatRacingBootLevel',
    ]);

    // Act: the ladder carries no next race level, the debug rung asks for
    // the five-car rung.
    const bootLevel = seam.resolveNeatRacingBootLevel({
      nextRaceLevel: null,
      debugRungLevel: 3,
      sessionLevel: 1,
    });

    // Assert: the debug rung seeds the boot (the ladder is silent).
    expect(bootLevel).toBe(3);
  });

  it('falls back to the session level when neither the ladder nor the debug rung speaks', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-18', [
      'resolveNeatRacingBootLevel',
    ]);

    // Act: no ladder level, no debug rung — the session's own level stands.
    const bootLevel = seam.resolveNeatRacingBootLevel({
      nextRaceLevel: null,
      debugRungLevel: null,
      sessionLevel: 1,
    });

    // Assert: the session level is the byte-identical default.
    expect(bootLevel).toBe(1);
  });

  it('seeds the opening boot level 2 from the `?level=2` debug rung', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-18', [
      'resolveNeatRacingDebugRungLevel',
      'resolveNeatRacingBootLevel',
    ]);
    const debugRungLevel = seam.resolveNeatRacingDebugRungLevel('?level=2');

    // Act: the debug rung the query string asked for seeds a fresh boot.
    const bootLevel = seam.resolveNeatRacingBootLevel({
      nextRaceLevel: null,
      debugRungLevel,
      sessionLevel: 1,
    });

    // Assert: the opening boot level is 2 — the rival-bearing rung in ONE
    // boot.
    expect(bootLevel).toBe(2);
  });
});

describe('RED-P4-19 — the fleet-size provenance', () => {
  it('exposes no fleet-size constant of its own (the ladder stays the single fleet-size source)', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet).
    const seam = await gateRedDebugRungSeam('RED-P4-19', [
      'resolveNeatRacingDebugRungLevel',
    ]);

    // Act: the seam's exported surface.
    const fleetSizeConstants = redFleetSizeConstantNames(seam.exports);

    // Assert: the seam carries NO fleet-size constant of its own — a ladder
    // change moves the seam, because the seam has no table to move.
    expect(fleetSizeConstants).toEqual([]);
  });

  it('derives the debug-rung fleet size through the ladder resolver (2 → 2 cars, 3 → 5 cars)', async () => {
    // Arrange: the seam behind the red gate (it does not exist yet) plus
    // the two rungs the debug seam is asked about.
    const seam = await gateRedDebugRungSeam('RED-P4-19', [
      'resolveNeatRacingDebugRungLevel',
    ]);
    const levelTwo = requireRedDebugRung(seam, '?level=2', 'RED-P4-19');
    const levelThree = requireRedDebugRung(seam, '?level=3', 'RED-P4-19');

    // Act + Assert: the fleet size of a debug rung is the LADDER's answer,
    // read through `resolveNeatRacingLevelCarCount` — the rival-bearing
    // rung materializes two cars and the five-car rung materializes five.
    expect(resolveNeatRacingLevelCarCount(levelTwo)).toBe(2);
    expect(resolveNeatRacingLevelCarCount(levelThree)).toBe(5);
  });
});
