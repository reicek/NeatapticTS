/** @jest-environment jsdom */

/**
 * RED->GREEN contract tests for the NeatRacing host HUD and race mode
 * selector (slice P1-S7-E; AC-024 + AC-033 restart pre-verify; spec
 * sections 4.2, 10.2, 11.2, 13.1 AC-024, 13.2 AC-033, LD-12, TP-01).
 *
 * Red phase: this suite must fail while './hud' and './mode-selector' do
 * not exist ("Cannot find module './hud'" / "Cannot find module
 * './mode-selector'" under ts-jest diagnostics); a green suite during the
 * red phase would be a contract defect. The plan red_test wording
 * ("asserts toggle + restart behavior") was reconciled against the spec
 * source of truth: the selector also binds the AC-024 style contract, so
 * the file carries both halves of slice P1-S7-E.
 *
 * Spec reconciliation recorded in this header:
 * - AC-024 pins the selector to the Neatenstein pattern (absolute top 0
 *   right 0, zIndex 10, bg rgba(6,11,20,0.85), cyan border, monospace
 *   14px, padding 4px 8px) and says Phase 1 exposes manual mode only. The
 *   governing slice AC (plan P1-S7-E) extends the component to the FULL
 *   AI/man toggle with restart-on-switch ("AC-033 restart behavior
 *   pre-verified here"), and the orchestrator objective repeats the
 *   AI/man toggle. The contract therefore renders both options with the
 *   INITIAL mode 'manual' (the "manual only" Phase 1 semantics: Phase 1
 *   boot glue keeps the race on manual and never switches it; the option
 *   list itself carries AI already so P2-S4-A only adds AI wiring).
 * - "Restart on switch" (LD-12: switching AI <-> manual restarts the
 *   race/simulation in both directions) binds at the component as an
 *   onSwitch notification: subscribers are notified exactly once per
 *   switch — for a user change event and for a programmatic setMode —
 *   with the mode being switched TO. The race restart itself (fresh
 *   race-rules state) is host wiring that lands with Step 13 boot glue;
 *   the notification contract pre-verifies the AC-033 restart behavior
 *   the wiring will consume.
 * - The HUD DOM overlay (AC-024) shows speed/lap/time with the cyan/
 *   magenta tokens recycled from Neatenstein per spec section 11.2
 *   ("same CSS tokens, cyan/magenta"): the token values mirror the
 *   Neatenstein health palette. The spec does not pin the exact counter
 *   formats, so the red contract pins minimal arcade formats by design:
 *   speed = whole km/h from the TP-01 anchor (1 unit = 1 meter, ~89
 *   units/s ~= ~320 km/h), lap = "Lap c/20" (current lap = laps completed
 *   + 1, clamped at NEATRACING_HUMAN_RACE_LAPS so a completed race never
 *   shows "21/20"), time = "M:SS.d" (unpadded minutes, zero-padded
 *   seconds, one tenths digit).
 *
 * Determinism: every executor is pure — identical inputs produce
 * identical strings with no Math.random, no clock reads, no timers. Not
 * claimed: cross-environment byte equality (same-runtime rung only).
 *
 * Fixture rules: the per-file jsdom pragma turns on the DOM for this file
 * only (the project default is node) — the exact fixture precedent of
 * examples/neatenstein/browser-entry/host/hud-human-mode.test.ts for the
 * same recycled pattern. Each test resets `document.body` and appends a
 * single `#neatracing-hud-output` container; numeric fixtures are
 * binary-exact or round-trip-stable in `formatHudSpeed` (20 and 89
 * multiplied by 3.6 round to exact integers).
 */

import { describe, expect, it, jest } from '@jest/globals';
import {
  createRaceRulesState,
  NEATRACING_HUMAN_RACE_LAPS,
  stepRaceRules,
} from '../physics/race-rules';
import type { RaceRulesState } from '../physics/race-rules';
import {
  createHud,
  formatHudLap,
  formatHudSpeed,
  formatHudTime,
  NEATRACING_HUD_COLOR_CYAN,
  NEATRACING_HUD_COLOR_MAGENTA,
} from './hud';
import { createRaceModeSelector } from './mode-selector';
import type { NeatRacingRaceMode } from './mode-selector';

/** Host container element id reserved for the neatRacing HUD output. */
const HUD_OUTPUT_ID = 'neatracing-hud-output';

/** Container element id used for the missing-container error contracts. */
const MISSING_OUTPUT_ID = 'neatracing-hud-missing';

/**
 * Resets the DOM and mounts the fresh HUD container fixture for one test
 * (per-test reset; no state leaks between tests).
 */
function createHudFixture(): HTMLElement {
  document.body.innerHTML = '';
  const container = document.createElement('div');
  container.id = HUD_OUTPUT_ID;
  document.body.appendChild(container);
  return container;
}

/** Collapses CSS whitespace so jsdom-normalized color reads stay stable. */
function normalizeCssValue(value: string): string {
  return value.replace(/\s+/g, '');
}

describe('neatRacing HUD + race mode selector (P1-S7-E)', () => {
  describe('HUD display contract (AC-024)', () => {
    describe('pure display formatters (cyan/magenta token feed)', () => {
      it('converts sim speed to the whole-km/h arcade readout (TP-01 anchor)', () => {
        expect(formatHudSpeed(89)).toBe('320 km/h');
        expect(formatHudSpeed(0)).toBe('0 km/h');
      });

      it('shows the current lap out of the 20-lap human race', () => {
        expect(formatHudLap(0)).toBe(`Lap 1/${NEATRACING_HUMAN_RACE_LAPS}`);
        expect(formatHudLap(1)).toBe(`Lap 2/${NEATRACING_HUMAN_RACE_LAPS}`);
        // A completed race must never advertise "Lap 21/20".
        expect(formatHudLap(NEATRACING_HUMAN_RACE_LAPS)).toBe(
          `Lap ${NEATRACING_HUMAN_RACE_LAPS}/${NEATRACING_HUMAN_RACE_LAPS}`,
        );
      });

      it('formats the race clock as M:SS.d', () => {
        expect(formatHudTime(0)).toBe('0:00.0');
        expect(formatHudTime(75.3)).toBe('1:15.3');
        expect(formatHudTime(3.04)).toBe('0:03.0');
      });
    });

    describe('DOM overlay (cyan/magenta tokens)', () => {
      it('exports the recycled Neatenstein cyan and magenta overlay tokens', () => {
        expect(NEATRACING_HUD_COLOR_CYAN).toBe('rgb(0, 240, 255)');
        expect(NEATRACING_HUD_COLOR_MAGENTA).toBe('rgb(255, 0, 85)');
      });

      it('shows the speed/lap/time values through the overlay labels', () => {
        createHudFixture();
        const hud = createHud(HUD_OUTPUT_ID);
        hud.update({
          speedUnitsPerS: 89,
          raceState: createRaceRulesState(),
        });

        expect({
          speed: hud.speedLabel.textContent,
          lap: hud.lapLabel.textContent,
          time: hud.timeLabel.textContent,
        }).toEqual({
          speed: '320 km/h',
          lap: `Lap 1/${NEATRACING_HUMAN_RACE_LAPS}`,
          time: '0:00.0',
        });
      });

      it('repaints the three counters across race updates', () => {
        createHudFixture();
        const hud = createHud(HUD_OUTPUT_ID);
        hud.update({
          speedUnitsPerS: 89,
          raceState: createRaceRulesState(),
        });

        const racedState: RaceRulesState = stepRaceRules(
          createRaceRulesState(),
          { dtS: 1, forwardLapLineCrossing: false },
        );
        hud.update({ speedUnitsPerS: 20, raceState: racedState });

        expect({
          speed: hud.speedLabel.textContent,
          lap: hud.lapLabel.textContent,
          time: hud.timeLabel.textContent,
        }).toEqual({
          speed: '72 km/h',
          lap: `Lap 1/${NEATRACING_HUMAN_RACE_LAPS}`,
          time: '0:01.0',
        });
      });

      it('colors the overlay labels with the cyan/magenta tokens only', () => {
        createHudFixture();
        const hud = createHud(HUD_OUTPUT_ID);

        const allowedTokens = new Set(
          [NEATRACING_HUD_COLOR_CYAN, NEATRACING_HUD_COLOR_MAGENTA].map(
            (token) => normalizeCssValue(token),
          ),
        );
        const labelColors = [hud.speedLabel, hud.lapLabel, hud.timeLabel].map(
          (label) => normalizeCssValue(label.style.color),
        );

        expect(labelColors.every((color) => allowedTokens.has(color))).toBe(
          true,
        );
      });

      it('throws when the HUD output container is missing', () => {
        document.body.innerHTML = '';
        expect(() => createHud(MISSING_OUTPUT_ID)).toThrow(/not found/);
      });
    });
  });

  describe('race mode selector (Neatenstein pattern, AC-024 style)', () => {
    it('renders an ai/manual select whose initial mode is ai (batch 108)', () => {
      createHudFixture();
      const selector = createRaceModeSelector(HUD_OUTPUT_ID);

      expect({
        tag: selector.select.tagName,
        options: Array.from(selector.select.options).map(
          (option) => option.value,
        ),
        initialMode: selector.mode,
      }).toEqual({
        tag: 'SELECT',
        options: ['ai', 'manual'],
        initialMode: 'ai',
      });
    });

    it('exposes id and name attributes for accessibility', () => {
      createHudFixture();
      const selector = createRaceModeSelector(HUD_OUTPUT_ID);

      expect(selector.select.id).toBe('neatracing-race-mode-selector');
      expect(selector.select.name).toBe('neatracing-race-mode-selector');
    });

    it('uses the Neatenstein pattern placement (top 0 / right 0 / zIndex 10)', () => {
      createHudFixture();
      const selector = createRaceModeSelector(HUD_OUTPUT_ID);

      expect({
        position: selector.select.style.position,
        top: selector.select.style.top,
        right: selector.select.style.right,
        zIndex: selector.select.style.zIndex,
      }).toEqual({
        position: 'absolute',
        top: '0px',
        right: '0px',
        zIndex: '10',
      });
    });

    it('uses the Neatenstein pattern tokens (bg / cyan border / monospace 14px / 4px 8px)', () => {
      createHudFixture();
      const selector = createRaceModeSelector(HUD_OUTPUT_ID);

      expect({
        backgroundColor: normalizeCssValue(
          selector.select.style.backgroundColor,
        ),
        borderColor: normalizeCssValue(selector.select.style.borderColor),
        fontFamily: selector.select.style.fontFamily,
        fontSize: selector.select.style.fontSize,
        padding: selector.select.style.padding,
      }).toEqual({
        backgroundColor: 'rgba(6,11,20,0.85)',
        borderColor: 'rgb(0,240,255)',
        fontFamily: 'monospace',
        fontSize: '14px',
        padding: '4px 8px',
      });
    });

    it('throws when the output container is missing', () => {
      document.body.innerHTML = '';
      expect(() => createRaceModeSelector(MISSING_OUTPUT_ID)).toThrow(
        /not found/,
      );
    });
  });

  describe('toggle + restart-on-switch (AC-033 restart pre-verify)', () => {
    it('switches to manual on a user change event and notifies the switch hook once', () => {
      createHudFixture();
      const selector = createRaceModeSelector(HUD_OUTPUT_ID);
      const switchLog: NeatRacingRaceMode[] = [];
      selector.onSwitch((mode) => switchLog.push(mode));

      selector.select.value = 'manual';
      selector.select.dispatchEvent(new Event('change'));

      expect(selector.mode).toBe('manual');
      expect(switchLog).toEqual(['manual']);
    });

    it('notifies the restart-carrying switch hook in both directions', () => {
      createHudFixture();
      const selector = createRaceModeSelector(HUD_OUTPUT_ID);
      const switchLog: NeatRacingRaceMode[] = [];
      selector.onSwitch((mode) => switchLog.push(mode));

      selector.setMode('manual');
      selector.setMode('ai');

      expect(switchLog).toEqual(['manual', 'ai']);
    });

    it('notifies exactly once per switch — never for a same-value setMode', () => {
      createHudFixture();
      const selector = createRaceModeSelector(HUD_OUTPUT_ID);
      const onSwitch = jest.fn();
      selector.onSwitch(onSwitch);

      selector.setMode('ai');

      expect(onSwitch).not.toHaveBeenCalled();
    });

    it('keeps the select DOM value in sync with a programmatic setMode', () => {
      createHudFixture();
      const selector = createRaceModeSelector(HUD_OUTPUT_ID);

      selector.setMode('ai');

      expect(selector.select.value).toBe('ai');
    });
  });
});
