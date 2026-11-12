/** @jest-environment jsdom */

/**
 * RED -> GREEN contract tests for the NeatRacing host HUD traction readout
 * (slice P1-drift-presentation-red, RED-2; plan
 * `neatracing-voxel-racer.plans.md`, step 14R; AC-088 + AC-089).
 *
 * The HUD is a DISPLAY read model: it SHOWS the ONE published traction
 * severity (Step 14Q, `PhysicsCarState.tractionSeverity`) and NEVER
 * re-derives it. This file is the ONLY owner of the HUD contract for the
 * traction wave: `examples/neatRacing/host/hud.test.ts` did not exist
 * before this slice and is created here (the plan's RED-2).
 *
 * RED PREMISE: `NeatRacingHudState` (hud.ts:142-147) carries ONLY the
 * speed and the race state - there is no traction field, no traction
 * readout, and the overlay text carries NO percentage at all today. Every
 * contract below therefore fails on its ASSERTION (a missing readout), not
 * on a compile error: the missing member is a TYPE MEMBER, and a type
 * member cannot produce a MISSING IMPORT. The name of the new field is
 * pinned by the FIXTURE TYPE (`NeatRacingHudState & { tractionSeverity:
 * number }`): the green must read EXACTLY `tractionSeverity` for the
 * readout to appear, and no placeholder export is created to make an
 * import resolve.
 *
 * RED CONTRACTS (the four failing assertions):
 * 1. [RED-14R-HUD-READOUT] the overlay renders the PUBLISHED severity.
 * 2. [RED-14R-HUD-NO-RECOMPUTE] the HUD never recomputes the severity
 *    from its OWN inputs: the same published severity renders the same
 *    readout at rest and at speed.
 * 3. [RED-14R-HUD-IDLE-READOUT] the readout is the IDLE zero on a
 *    straight line (an explicit idle text is accepted as well).
 * 4. [RED-14R-HUD-REPAINT] the readout repaints when the published
 *    severity changes.
 *
 * RED DECISIONS (the shape the green slice `P1-drift-presentation-hud`
 * must honour):
 * - the HUD state field is NAMED `tractionSeverity` - the SAME name the
 *   physics state publishes and the car pass reads - and it carries the
 *   PUBLISHED severity in [0, 1], never a re-derived one;
 * - the field is OPTIONAL in the production type: the sibling suite
 *   `host/mode-selector.test.ts` (NOT owned by the green slice) calls
 *   `hud.update({ speedUnitsPerS, raceState })` without it, so a
 *   REQUIRED field would break an unowned suite;
 * - the readout is observed through the HOST OUTPUT CONTAINER's
 *   `textContent`, so the green may name the label, its color and its
 *   wording freely - the contract pins the VALUE, never the label;
 * - the VALUE is accepted in either rendering: a whole percent ('42%',
 *   '42 %') or the raw published severity ('0.42'). The contract pins
 *   the number the player reads, not the format around it;
 * - NO new formatter export is required (`formatHudTraction` is NOT
 *   mandated): the green keeps its API surface free.
 *
 * Fixture rules: the per-file jsdom pragma turns on the DOM for this file
 * only (the project default is node) - the exact fixture precedent of
 * `host/mode-selector.test.ts` for the same recycled HUD pattern. Each
 * test resets `document.body` and appends a single
 * `#neatracing-hud-output` container, then mounts the HUD on it, so no
 * DOM state crosses a test boundary. No timers, no RNG, no browser: the
 * overlay is a pure text surface.
 */

import { describe, expect, it } from '@jest/globals';
import {
  createRaceRulesState,
  NEATRACING_HUMAN_RACE_LAPS,
} from '../physics/race-rules';
import { createHud, type NeatRacingHud, type NeatRacingHudState } from './hud';

/** Host container element id reserved for the neatRacing HUD output. */
const HUD_OUTPUT_ID = 'neatracing-hud-output';

/** Published severity fixture: a car mid-slide (42 percent of the grip gone). */
const DRIFTING_TRAACTION_SEVERITY = 0.42;

/** Published severity fixture: a deeper slide (90 percent of the grip gone). */
const DEEP_TRAACTION_SEVERITY = 0.9;

/** The straight-line published severity: the published zero. */
const IDLE_TRAACTION_SEVERITY = 0;

/** Speed fixture at rest. */
const REST_SPEED_UNITS_PER_S = 0;

/** Speed fixture at speed (the 320 km/h arcade readout of the TP-01 anchor). */
const MOVING_SPEED_UNITS_PER_S = 89;

/** The HUD fixture: the host output container plus the mounted HUD. */
interface HudFixture {
  /** The host output container the HUD is mounted into. */
  readonly container: HTMLElement;
  /** The mounted HUD controller. */
  readonly hud: NeatRacingHud;
}

/**
 * Resets the DOM and mounts a fresh HUD container for one test (per-test
 * reset; no state crosses a test boundary).
 */
function createHudFixture(): HudFixture {
  document.body.innerHTML = '';
  const container = document.createElement('div');
  container.id = HUD_OUTPUT_ID;
  document.body.appendChild(container);
  return { container, hud: createHud(HUD_OUTPUT_ID) };
}

/**
 * Adds the PUBLISHED traction severity to a HUD state.
 *
 * The seam does not carry the channel yet, so the fixture cannot spell it
 * as a plain object literal (the excess-property check would reject the
 * unknown property). The helper keeps the channel NAME in the fixture's
 * type - the returned state is `NeatRacingHudState & { tractionSeverity:
 * number }` - so the production must read EXACTLY `tractionSeverity` for
 * the contracts to turn green. The name is pinned by the BEHAVIOUR, never
 * by a placeholder export.
 *
 * @param base - The HUD state without the traction channel.
 * @param tractionSeverity - The published traction severity in [0, 1].
 * @returns The HUD state carrying the published severity.
 */
function withTractionSeverity(
  base: NeatRacingHudState,
  tractionSeverity: number,
): NeatRacingHudState & { tractionSeverity: number } {
  return Object.assign({}, base, { tractionSeverity });
}

/**
 * Builds a HUD state fixture with the given speed and published severity.
 *
 * @param speedUnitsPerS - The forward speed in world-units-per-second.
 * @param tractionSeverity - The published traction severity in [0, 1].
 * @returns The HUD state fixture.
 */
function hudStateFixture(
  speedUnitsPerS: number,
  tractionSeverity: number,
): NeatRacingHudState & { tractionSeverity: number } {
  return withTractionSeverity(
    { speedUnitsPerS, raceState: createRaceRulesState() },
    tractionSeverity,
  );
}

/**
 * The overlay text as the host sees it (the whole output container, so
 * the contract never depends on WHICH label carries the readout).
 *
 * @param container - The host output container.
 * @returns The concatenated overlay text.
 */
function overlayText(container: HTMLElement): string {
  return container.textContent ?? '';
}

/**
 * Every percentage readout in the overlay text, as numbers.
 *
 * @param text - The overlay text.
 * @returns The percentage values the overlay shows.
 */
function percentReadouts(text: string): number[] {
  const values: number[] = [];
  for (const match of text.matchAll(/(\d+(?:\.\d+)?)\s*%/g)) {
    values.push(Number.parseFloat(match[1]));
  }
  return values;
}

/**
 * The traction readout the overlay shows, canonicalised to a string, or
 * `null` when the overlay shows no traction readout at all.
 *
 * Two renderings are accepted (the contract pins the VALUE, never the
 * wording): a whole percent (canonicalised to `'42%'`) and the raw
 * published severity (canonicalised to `'0.42'`).
 *
 * @param text - The overlay text.
 * @returns The canonicalised traction readout, or `null`.
 */
function tractionReadout(text: string): string | null {
  const readouts = percentReadouts(text);
  if (readouts.length > 0) {
    return `${String(Math.round(readouts[0]))}%`;
  }
  const raw = /(?:^|[^\d.])(0\.\d{2})(?:[^\d]|$)/.exec(text);
  return raw === null ? null : raw[1];
}

/**
 * True when the overlay shows the IDLE traction state on a straight line:
 * a zero readout ('0%' or '0.00') or an EXPLICIT idle text. A MISSING
 * readout is NOT idle - the traction readout must always be on screen.
 *
 * @param text - The overlay text.
 * @returns True when the overlay shows the idle traction state.
 */
function showsIdleTraction(text: string): boolean {
  const readouts = percentReadouts(text);
  if (readouts.length > 0) {
    return readouts.every((value) => Math.round(value) === 0);
  }
  if (/(?:^|[^\d.])0\.00(?:[^\d]|$)/.test(text)) {
    return true;
  }
  return /\b(?:idle|steady|attached|normal|none|ok)\b/i.test(text);
}

describe('NeatRacing HUD traction readout (P1-drift-presentation-red, RED-2)', () => {
  describe('the published traction severity reaches the overlay (AC-088)', () => {
    // [RED-14R-HUD-READOUT]
    it('renders the published traction severity on the overlay', () => {
      // Arrange
      const fixture = createHudFixture();
      // Act
      fixture.hud.update(
        hudStateFixture(MOVING_SPEED_UNITS_PER_S, DRIFTING_TRAACTION_SEVERITY),
      );
      // Assert - the overlay shows the PUBLISHED severity (42 percent).
      expect(['42%', '0.42']).toContain(
        tractionReadout(overlayText(fixture.container)),
      );
    });

    // [RED-14R-HUD-NO-RECOMPUTE]
    it('never recomputes the traction severity from its own inputs', () => {
      // Arrange - the SAME published severity, two very different speeds.
      const fixture = createHudFixture();
      // Act
      fixture.hud.update(
        hudStateFixture(REST_SPEED_UNITS_PER_S, DRIFTING_TRAACTION_SEVERITY),
      );
      const atRest = tractionReadout(overlayText(fixture.container));
      fixture.hud.update(
        hudStateFixture(MOVING_SPEED_UNITS_PER_S, DRIFTING_TRAACTION_SEVERITY),
      );
      const atSpeed = tractionReadout(overlayText(fixture.container));
      // Assert - the readout IS the published severity and does NOT move
      // with the speed (a recomputed severity would change here).
      expect(['42%', '0.42']).toContain(atRest);
      expect(atSpeed).toBe(atRest);
    });

    // [RED-14R-HUD-IDLE-READOUT]
    it('shows the idle traction state on a straight line', () => {
      // Arrange
      const fixture = createHudFixture();
      // The boot paint is the zeroed read model, so the traction readout is
      // already on screen before the first update.
      const atBoot = showsIdleTraction(overlayText(fixture.container));
      // Act - the straight-line published severity is the published zero.
      fixture.hud.update(
        hudStateFixture(MOVING_SPEED_UNITS_PER_S, IDLE_TRAACTION_SEVERITY),
      );
      const atStraightLine = showsIdleTraction(overlayText(fixture.container));
      // Assert - the readout is zero (or an explicit idle text), never a
      // blank traction readout.
      expect(atBoot).toBe(true);
      expect(atStraightLine).toBe(true);
    });

    // [RED-14R-HUD-REPAINT]
    it('repaints the traction readout when the published severity changes', () => {
      // Arrange
      const fixture = createHudFixture();
      fixture.hud.update(
        hudStateFixture(MOVING_SPEED_UNITS_PER_S, DRIFTING_TRAACTION_SEVERITY),
      );
      // Act - a deeper slide publishes a bigger severity.
      fixture.hud.update(
        hudStateFixture(MOVING_SPEED_UNITS_PER_S, DEEP_TRAACTION_SEVERITY),
      );
      // Assert - the overlay carries the NEW value, not the stale one.
      expect(['90%', '0.9']).toContain(
        tractionReadout(overlayText(fixture.container)),
      );
    });
  });

  describe('preserved HUD behaviour (AC-089, no regression)', () => {
    it('keeps the speed, lap and time readouts unchanged', () => {
      // Arrange - the traction field is OPTIONAL, so the classic read
      // model still paints without it.
      const fixture = createHudFixture();
      // Act
      fixture.hud.update({
        speedUnitsPerS: MOVING_SPEED_UNITS_PER_S,
        raceState: createRaceRulesState(),
      });
      // Assert - the three existing readouts are untouched.
      expect({
        speed: fixture.hud.speedLabel.textContent,
        lap: fixture.hud.lapLabel.textContent,
        time: fixture.hud.timeLabel.textContent,
      }).toEqual({
        speed: '320 km/h',
        lap: `Lap 1/${NEATRACING_HUMAN_RACE_LAPS}`,
        time: '0:00.0',
      });
    });
  });
});
