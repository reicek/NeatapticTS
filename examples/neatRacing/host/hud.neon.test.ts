/** @jest-environment jsdom */

/**
 * RED home for the NEON CLOSED-SELECT contract of the NeatRacing host lane
 * (slice `P5-S29-red` — Phase 5, Step 29 of
 * `plans/neatracing-voxel-racer.plans.md`, AC-108-RED).
 *
 * Educational note — what this file owns:
 * The FOUR RED-S29 contracts of the "neon dropdown text" packet, expressed
 * against the EXISTING race-mode selector seam:
 *
 * - RED-S29-1 — the NEON CLASS HOOK. The CLOSED race-mode select (the
 *   `<select>` element itself, rendered CLOSED — NOT an option popup) must
 *   carry at least one class whose name matches /neon/i. The neon helper
 *   module owns the class-hook NAME, so the assertion is deliberately
 *   NAME-AGNOSTIC: it asks "is there a neon class on the closed select?",
 *   never "is the class called X?" — the implementer may pick any name.
 * - RED-S29-2 — the NEON GLOW. The CLOSED select must declare a
 *   `text-shadow` (the neon glow the helper builds). The declaration is
 *   read from the INLINE style attribute (the surface the selector
 *   already writes) and from the parsed `style.textShadow`; the check is
 *   case-insensitive and whitespace-tolerant and never pins a colour value.
 * - RED-S29-3 — the NATIVE `<option>` POPUP IS UNTOUCHED (a PASSING
 *   invariant). The option popup is OS-rendered, so the neon lands on the
 *   CLOSED control only: the option list, the option values and the option
 *   text stay exactly as they are and the control stays a native
 *   `<select>` (NO custom listbox replaces it).
 * - RED-S29-4 — the IDENTITY LAW (a PASSING invariant). The select keeps
 *   its `id` and its `name`.
 *
 * Educational note — the RED shape (why these assertions, and not others):
 * This file is RED BY ASSERTION and NEVER by an import failure. It imports
 * ONLY the EXISTING `./mode-selector` module (the sibling seam in this
 * directory); it NEVER imports the NEW `./hud.neon` helper module, so a
 * missing module can never be the failure reason. The two RED contracts
 * fail because the CLOSED select carries no neon class and no text-shadow
 * TODAY; the two passing invariants pass today and MUST keep passing once
 * the neon lands (the neon is a DOM treatment on the EXISTING select — it
 * never rewrites the option text and never replaces the native select).
 *
 * Educational note — the MEASURED BASELINE (recorded 2026-11-22, BEFORE any
 * production edit; the green lane compares against THIS measurement and
 * never against a hand-written expectation):
 * - the closed select's `classList` is EMPTY and it carries NO class
 *   attribute at all;
 * - its inline `style` attribute declares NINE properties (position, top,
 *   right, z-index, background-color, border-color, font-family, font-size,
 *   padding) and NO `text-shadow`;
 * - `select.style.textShadow` is the empty string;
 * - `id` and `name` are both `neatracing-race-mode-selector`;
 * - the option list is `ai`, `manual` and each option's text is its value.
 *
 * Educational note — the fixture contract:
 * DETERMINISTIC (no RNG, no timers, no wall-clock reads), RESETTABLE (every
 * test rebuilds the DOM container through the fixture helper) and
 * SINGLE-PURPOSE (one contract per `it`; the two RED contracts fail on
 * their own assertion and the two passing invariants are separate `it`
 * blocks).
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';
import { createRaceModeSelector } from './mode-selector';

/** Host container element id reserved for the neatRacing HUD output. */
const HUD_OUTPUT_ID = 'neatracing-hud-output';

/** The identity the closed race-mode select keeps (the identity law). */
const RACE_MODE_SELECTOR_ID = 'neatracing-race-mode-selector';

/**
 * Resets the DOM and mounts the fresh race-mode selector fixture for one
 * test (per-test reset; no state leaks between tests).
 *
 * @returns The mounted CLOSED `<select>` element of the race-mode selector.
 */
function createHudFixture(): HTMLSelectElement {
  document.body.innerHTML = '';
  const container = document.createElement('div');
  container.id = HUD_OUTPUT_ID;
  document.body.appendChild(container);
  return createRaceModeSelector(HUD_OUTPUT_ID).select;
}

/** Collapses CSS whitespace so jsdom-normalized declarations read stable. */
function normalizeCssValue(value: string): string {
  return value.replace(/\s+/g, '');
}

describe('the neon closed-select contract (AC-108-RED, P5-S29-red)', () => {
  it('RED-S29-1: the closed race-mode select carries at least one class matching /neon/i (the neon class hook)', () => {
    const select = createHudFixture();
    const classNames = Array.from(select.classList);

    expect({
      contract:
        'RED-S29-1: the closed select must carry at least one class matching /neon/i',
      neonClassHookPresent: classNames.some((name) => /neon/i.test(name)),
      observedClassNames: classNames,
    }).toEqual({
      contract:
        'RED-S29-1: the closed select must carry at least one class matching /neon/i',
      neonClassHookPresent: true,
      observedClassNames: expect.any(Array),
    });
  });

  it('RED-S29-2: the closed race-mode select declares a text-shadow glow (the neon glow)', () => {
    const select = createHudFixture();
    const inlineStyle = select.getAttribute('style') ?? '';
    const normalizedInlineStyle = normalizeCssValue(inlineStyle).toLowerCase();
    const declaredTextShadow = select.style.textShadow.trim();

    expect({
      contract: 'RED-S29-2: the closed select must declare a text-shadow glow',
      neonGlowDeclared:
        normalizedInlineStyle.includes('text-shadow:') ||
        declaredTextShadow !== '',
      observedTextShadow: declaredTextShadow,
    }).toEqual({
      contract: 'RED-S29-2: the closed select must declare a text-shadow glow',
      neonGlowDeclared: true,
      observedTextShadow: expect.any(String),
    });
  });

  it('RED-S29-3: the native option popup stays untouched (the option list, the option values and the option text are unchanged)', () => {
    const select = createHudFixture();
    const options = Array.from(select.options);

    expect({
      tagName: select.tagName,
      optionValues: options.map((option) => option.value),
      optionTexts: options.map((option) => option.textContent),
    }).toEqual({
      tagName: 'SELECT',
      optionValues: ['ai', 'manual'],
      optionTexts: ['ai', 'manual'],
    });
  });

  it('RED-S29-4: the closed select keeps its id and its name (the identity law)', () => {
    const select = createHudFixture();

    expect({
      id: select.id,
      name: select.name,
    }).toEqual({
      id: RACE_MODE_SELECTOR_ID,
      name: RACE_MODE_SELECTOR_ID,
    });
  });
});
