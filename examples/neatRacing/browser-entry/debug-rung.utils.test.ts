/**
 * @file The permanent green contract of the W2 debug-rung seam (Step 24A):
 * the pure `?level` rung resolver and the boot-level precedence the host
 * seeds its OPENING round from (slice `P4-S2A-debug-rung-seam`).
 *
 * This home deliberately runs in the NODE test environment (there is NO
 * jsdom docblock on purpose): the seam is pure and DOM-free, so every
 * exercise here doubles as the purity witness — the exact resolver the
 * browser boots with runs headless, and the composition contract proves
 * the full ?level=2 chain without touching `window.location`.
 *
 * Educational note — THE RED TWINS. The contracts RED-P4-17 / RED-P4-18 /
 * RED-P4-19 live in the BYTE-FROZEN `debug-rung.red.test.ts`; this green
 * home mirrors their behaviour as the PERMANENT contract with static
 * imports and no gating helpers — the seam is simply, plainly exercised.
 * If a future edit re-freezes a divergence between the two homes, the red
 * home escalates to the orchestrator (a frozen-byte divergence), it is
 * never edited silently.
 * Educational note — THE LADDER STAYS THE FLEET'S OWNER. The fleet-size
 * expectations pin the REAL `resolveNeatRacingLevelCarCount` answers (2
 * cars at rung 2, 5 at rung 3) with NO mock: provenance means the seam's
 * numbers match the ladder's, not a spy's. The seam must also expose NO
 * fleet-size constant of its own, so a ladder change moves the seam.
 * Educational note — THE DEFAULT IS BYTE-IDENTICAL. With no `?level`
 * parameter (the ABSENT-parameter contract) the precedence resolver
 * boots exactly the former `nextRaceLevel ?? sessionLevel` default, so
 * the shipped demo is unaffected by the seam.
 * @module
 */

import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_PROMOTION_MAX_LEVEL,
  resolveNeatRacingLevelCarCount,
} from '../ai/promotion.rules';
import {
  resolveNeatRacingBootLevel,
  resolveNeatRacingDebugRungLevel,
} from './debug-rung.utils';

describe('resolveNeatRacingDebugRungLevel — the pure ?level rung resolver', () => {
  it('resolves the ladder rung a ?level parameter names', () => {
    expect(resolveNeatRacingDebugRungLevel('?level=1')).toBe(1);
    expect(resolveNeatRacingDebugRungLevel('?level=2')).toBe(2);
    expect(resolveNeatRacingDebugRungLevel('?level=3')).toBe(3);
  });

  it('keeps the absent-parameter default byte-identical (null)', () => {
    expect(resolveNeatRacingDebugRungLevel('')).toBeNull();
    expect(resolveNeatRacingDebugRungLevel('?track=demo')).toBeNull();
  });

  it('ignores a rung name outside the ladder band instead of coercing it', () => {
    expect(resolveNeatRacingDebugRungLevel('?level=0')).toBeNull();
    expect(resolveNeatRacingDebugRungLevel('?level=-1')).toBeNull();
    expect(
      resolveNeatRacingDebugRungLevel(
        `?level=${NEATRACING_PROMOTION_MAX_LEVEL + 1}`,
      ),
    ).toBeNull();
  });

  it('ignores a malformed rung name where the ladder clamp would have coerced', () => {
    expect(resolveNeatRacingDebugRungLevel('?level=2.5')).toBeNull();
    expect(resolveNeatRacingDebugRungLevel('?level=abc')).toBeNull();
  });

  it('pins the ladder difference: the clamp coerces where the seam must not', () => {
    expect(resolveNeatRacingLevelCarCount(2.5)).toBe(2);
    expect(resolveNeatRacingLevelCarCount(0)).toBe(1);
  });

  it('seeds one boot level straight from ?level=2 (the composition contract)', () => {
    const debugRungLevel = resolveNeatRacingDebugRungLevel('?level=2');
    expect(debugRungLevel).toBe(2);
    expect(
      resolveNeatRacingBootLevel({
        nextRaceLevel: null,
        debugRungLevel,
        sessionLevel: 1,
      }),
    ).toBe(2);
  });
});

describe('resolveNeatRacingBootLevel — the boot-level precedence', () => {
  it('keeps the pending ladder level FIRST when a debug rung is named', () => {
    expect(
      resolveNeatRacingBootLevel({
        nextRaceLevel: 2,
        debugRungLevel: 3,
        sessionLevel: 1,
      }),
    ).toBe(2);
  });

  it('seeds a boot from the debug rung when the ladder is silent', () => {
    expect(
      resolveNeatRacingBootLevel({
        nextRaceLevel: null,
        debugRungLevel: 3,
        sessionLevel: 1,
      }),
    ).toBe(3);
  });

  it('falls back to the session level byte-identically when nothing is seeded', () => {
    expect(
      resolveNeatRacingBootLevel({
        nextRaceLevel: null,
        debugRungLevel: null,
        sessionLevel: 1,
      }),
    ).toBe(1);
  });
});

describe('the fleet-size provenance — the ladder stays the only fleet source', () => {
  it('exposes no fleet-size constant of its own', async () => {
    const seamModule = (await import('./debug-rung.utils')) as Record<
      string,
      unknown
    >;
    const fleetSizeConstantNames = Object.entries(seamModule)
      .filter(([constName, exportValue]) => {
        return (
          typeof exportValue !== 'function' &&
          /fleet|car_?count|cars/i.test(constName)
        );
      })
      .map(([constName]) => constName)
      .sort();
    expect(fleetSizeConstantNames).toEqual([]);
  });

  it('sizes rung 2 at the rival two-car fleet through the ladder resolver', () => {
    const debugRungLevel = resolveNeatRacingDebugRungLevel('?level=2');
    expect(debugRungLevel).toBe(2);
    expect(resolveNeatRacingLevelCarCount(debugRungLevel as number)).toBe(2);
  });

  it('sizes rung 3 at the five-car fleet through the ladder resolver', () => {
    const debugRungLevel = resolveNeatRacingDebugRungLevel('?level=3');
    expect(debugRungLevel).toBe(3);
    expect(resolveNeatRacingLevelCarCount(debugRungLevel as number)).toBe(5);
  });
});
