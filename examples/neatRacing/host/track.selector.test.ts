/** @jest-environment jsdom */

/**
 * GREEN contract tests for the NeatRacing host track selector (slice
 * P4-S3-select-restart; plan `neatracing-voxel-racer.plans.md`, phase 4,
 * step 25; AC-101 + LD-08/LD-12; spec sections 4.4 and 13.5).
 *
 * This file is the slice's OWN green home: it pins the CHOSEN surface
 * names (`setSelectedTrack` + `onSwitch` + the `trackId` field) precisely,
 * while the frozen red home (`../track/track.catalog.test.ts`, untouched —
 * RED-P4-FROZEN) discovers the DOM-free core via its capability probe and
 * witnesses the 5 RED-P4-8 its against those names.
 *
 * Law provenance (mirroring `host/mode-selector.test.ts`):
 * - The notify/restart contract: subscribe is notified EXACTLY once per
 *   actual switch with the SELECTED track id, in BOTH directions (LD-12);
 *   a same-value re-pick is a documented no-op (never a silent restart).
 * - The membership law: the selection is a control OVER the catalog, so an
 *   id the catalog does not carry is the catalog resolver's named
 *   membership error (unknown id + the known ids).
 * - The retirement law (AC-101): the demo boots on
 *   `NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID` — the catalog's pinned
 *   member — so the single pinned `NEATRACING_DEMO_TRACK_ID` boot literal
 *   has no dual path anywhere.
 * - The SIGNED left-normal lane-point convention rides the catalog:
 *   selection re-resolves geometry THROUGH `resolveTrackVariantSpec`, and
 *   the catalog's reverse/mirrorX transforms negate the stored signed
 *   `innerOffsetWorld` (a sign flip, never a magnitude mirroring), so the
 *   selection law never needs to care about lane-point signs.
 *
 * The per-file jsdom pragma turns on the DOM for the overlay halves (the
 * exact fixture precedent of `host/mode-selector.test.ts`); the DOM-free
 * core law itself needs no DOM and never touches one. Each test resets
 * `document.body` and appends a single `#neatracing-hud-output` container
 * (per-test reset; no state leaks between tests).
 *
 * Determinism: every executor here is pure or mount-only — identical
 * inputs produce identical state transitions with no Math.random, no clock
 * reads, no timers.
 */

import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_DEMO_TRACK_CATALOG,
  NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
  NEATRACING_DEMO_TRACK_VARIANT_COUNT,
} from '../track/track.catalog';
import {
  createNeatRacingTrackSelection,
  createNeatRacingTrackSelector,
} from './track.selector';
import type {
  NeatRacingTrackSelectionOptions,
  NeatRacingTrackSwitchListener,
} from './track.selector';

/** Host container element id reserved for the neatRacing HUD output. */
const HUD_OUTPUT_ID = 'neatracing-hud-output';

/** Container element id used for the missing-container error contracts. */
const MISSING_OUTPUT_ID = 'neatracing-hud-missing';

/**
 * Collapses CSS whitespace so jsdom-normalized color reads stay stable
 * (the mode-selector fixture precedent).
 */
function normalizeCssValue(value: string): string {
  return value.replace(/\s+/g, '');
}

/**
 * Resets the DOM and mounts the fresh HUD container fixture for one test
 * (per-test reset; no state leaks between tests).
 */
function createSelectorFixture(): HTMLElement {
  document.body.innerHTML = '';
  const container = document.createElement('div');
  container.id = HUD_OUTPUT_ID;
  document.body.appendChild(container);
  return container;
}

/**
 * The catalog order of member ids (the overlay must render exactly this
 * order, pinned member first — the view never re-sorts the catalog).
 */
const CATALOG_TRACK_IDS: readonly string[] = Object.keys(
  NEATRACING_DEMO_TRACK_CATALOG,
);

/**
 * The FIRST member id beside the pinned demo loop (the switch target the
 * frozen contract uses for the "other" circuit).
 */
const OTHER_TRACK_ID: string = CATALOG_TRACK_IDS.filter(
  (trackId) => trackId !== NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
)[0];

/** Collects every notified track id through one subscriber harness. */
function createNotifyRecorder(): {
  readonly notified: string[];
  readonly listener: NeatRacingTrackSwitchListener;
} {
  const notified: string[] = [];
  return {
    notified,
    listener: (payload): void => {
      notified.push(payload);
    },
  };
}

describe('neatRacing track selector (P4-S3-select-restart, AC-101)', () => {
  describe('DOM-free selection core (the frozen RED-P4-8 discovery target)', () => {
    it('boots BOTH capability-probe call shapes on the catalog default', () => {
      // The frozen contract calls each exported factory with the catalog
      // and with `undefined`, then adopts the first switch surface. Both
      // shapes — plus the bare call — must boot the same law: the demo
      // catalog, pinned member first, catalog default selected.
      const fromCatalog = createNeatRacingTrackSelection(
        NEATRACING_DEMO_TRACK_CATALOG,
      );
      const fromUndefined = createNeatRacingTrackSelection(undefined);
      const bare = createNeatRacingTrackSelection();

      expect(fromCatalog.trackId).toBe(
        NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
      );
      expect(fromUndefined.trackId).toBe(
        NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
      );
      expect(bare.trackId).toBe(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);
    });

    it('boots on the catalog default — the retired demo literal is a catalog member', () => {
      const selection = createNeatRacingTrackSelection();

      expect(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID).toBe(
        'neatracing-demo-loop',
      );
      expect(
        Object.prototype.hasOwnProperty.call(
          NEATRACING_DEMO_TRACK_CATALOG,
          NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
        ),
      ).toBe(true);
      expect(selection.trackId).toBe(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);
    });

    it('boots a fixture catalog with an explicit initial track id', () => {
      const options: NeatRacingTrackSelectionOptions = {
        catalog: NEATRACING_DEMO_TRACK_CATALOG,
        initialTrackId: OTHER_TRACK_ID,
      };

      expect(createNeatRacingTrackSelection(options).trackId).toBe(
        OTHER_TRACK_ID,
      );
    });

    it('notifies the restart hook exactly once with the SELECTED track id', () => {
      const selection = createNeatRacingTrackSelection();
      const { notified, listener } = createNotifyRecorder();
      selection.onSwitch(listener);

      selection.setSelectedTrack(OTHER_TRACK_ID);

      expect(notified).toEqual([OTHER_TRACK_ID]);
      expect(selection.trackId).toBe(OTHER_TRACK_ID);
    });

    it('restarts in EITHER direction — a switch back notifies the first track again', () => {
      const selection = createNeatRacingTrackSelection();
      const { notified, listener } = createNotifyRecorder();
      selection.onSwitch(listener);

      selection.setSelectedTrack(OTHER_TRACK_ID);
      selection.setSelectedTrack(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);

      expect(notified).toEqual([
        OTHER_TRACK_ID,
        NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
      ]);
      expect(selection.trackId).toBe(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);
    });

    it('never notifies a same-value re-pick (the documented no-op law)', () => {
      const selection = createNeatRacingTrackSelection();
      const { notified, listener } = createNotifyRecorder();
      selection.onSwitch(listener);

      selection.setSelectedTrack(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);
      selection.setSelectedTrack(OTHER_TRACK_ID);
      selection.setSelectedTrack(OTHER_TRACK_ID);

      expect(notified).toEqual([OTHER_TRACK_ID]);
    });

    it('notifies every subscriber exactly once per switch (multi-subscribe law)', () => {
      const selection = createNeatRacingTrackSelection();
      const first = createNotifyRecorder();
      const second = createNotifyRecorder();
      selection.onSwitch(first.listener);
      selection.onSwitch(second.listener);

      selection.setSelectedTrack(OTHER_TRACK_ID);

      expect(first.notified).toEqual([OTHER_TRACK_ID]);
      expect(second.notified).toEqual([OTHER_TRACK_ID]);
    });

    it('round-trips every catalog member and retains the last selection', () => {
      const selection = createNeatRacingTrackSelection();
      const { notified, listener } = createNotifyRecorder();
      selection.onSwitch(listener);

      // Walk the WHOLE catalog round-robin (pinned loop -> reverse ->
      // mirrored -> alt circuit) and end back on the pinned loop: the law
      // must hold for every shipped member, and the final selection is
      // retained exactly where it was last set. The opening walk step is
      // the boot default (a same-value no-op), so the notified sequence is
      // the pinned loop picked LAST instead of first.
      for (const trackId of CATALOG_TRACK_IDS) {
        selection.setSelectedTrack(trackId);
      }
      selection.setSelectedTrack(CATALOG_TRACK_IDS[0]);

      expect(notified).toEqual([
        ...CATALOG_TRACK_IDS.slice(1),
        CATALOG_TRACK_IDS[0],
      ]);
      expect(notified[notified.length - 1]).toBe(CATALOG_TRACK_IDS[0]);
      expect(selection.trackId).toBe(CATALOG_TRACK_IDS[0]);
    });

    it('throws the catalog resolver\u2019s named membership error for an unknown pick', () => {
      const selection = createNeatRacingTrackSelection();
      const { notified, listener } = createNotifyRecorder();
      selection.onSwitch(listener);
      const unknownTrackId = 'neatracing-not-a-shipped-track';
      const knownIds = CATALOG_TRACK_IDS.join(', ');

      expect(() => selection.setSelectedTrack(unknownTrackId)).toThrow(
        `NEATRACING_DEMO_TRACK_CATALOG has no variant id "${unknownTrackId}" — known ids: ${knownIds}.`,
      );
      // A rejected pick changes nothing: the selection (and the restart
      // hook) stay exactly where they were.
      expect(selection.trackId).toBe(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);
      expect(notified).toEqual([]);
    });

    it('rejects an unknown initial track id outright', () => {
      const options: NeatRacingTrackSelectionOptions = {
        catalog: NEATRACING_DEMO_TRACK_CATALOG,
        initialTrackId: 'neatracing-not-a-shipped-track',
      };

      expect(() => createNeatRacingTrackSelection(options)).toThrow(
        /no variant id "neatracing-not-a-shipped-track"/,
      );
    });
  });

  describe('track-selector overlay (the AC-101 selection control)', () => {
    it('renders every catalog member in catalog order (pinned member first)', () => {
      createSelectorFixture();
      const selector = createNeatRacingTrackSelector(HUD_OUTPUT_ID);

      const options = Array.from(selector.select.options);
      expect(options.length).toBe(NEATRACING_DEMO_TRACK_VARIANT_COUNT);
      expect(options.map((option) => option.value)).toEqual(
        CATALOG_TRACK_IDS.slice(),
      );
      // The visible labels are the catalog's own displayName values — the
      // overlay is a view over the catalog, never a second label source.
      expect(options.map((option) => option.textContent)).toEqual(
        CATALOG_TRACK_IDS.map((trackId) => {
          const entry = NEATRACING_DEMO_TRACK_CATALOG[trackId];
          return entry === undefined ? null : entry.displayName;
        }),
      );
    });

    it('exposes id and name attributes and boots on the catalog default', () => {
      createSelectorFixture();
      const selector = createNeatRacingTrackSelector(HUD_OUTPUT_ID);

      expect(selector.select.id).toBe('neatracing-track-selector');
      expect(selector.select.name).toBe('neatracing-track-selector');
      expect(selector.select.value).toBe(
        NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
      );
      expect(selector.trackId).toBe(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);
    });

    it('switches on a user change event and notifies the restart hook once per switch', () => {
      createSelectorFixture();
      const selector = createNeatRacingTrackSelector(HUD_OUTPUT_ID);
      const { notified, listener } = createNotifyRecorder();
      selector.onSwitch(listener);

      selector.select.value = OTHER_TRACK_ID;
      selector.select.dispatchEvent(new Event('change'));

      expect(notified).toEqual([OTHER_TRACK_ID]);
      expect(selector.trackId).toBe(OTHER_TRACK_ID);
      expect(selector.select.value).toBe(OTHER_TRACK_ID);
    });

    it('absorbs a replayed same-value change event (the core\u2019s no-op law)', () => {
      createSelectorFixture();
      const selector = createNeatRacingTrackSelector(HUD_OUTPUT_ID);
      const { notified, listener } = createNotifyRecorder();
      selector.onSwitch(listener);

      // Two same-value change events: user agents (and fixtures) can
      // replay them; the core's same-value guard owns the law in one
      // place, so neither notifies and the selection is retained.
      selector.select.dispatchEvent(new Event('change'));
      selector.select.dispatchEvent(new Event('change'));

      expect(notified).toEqual([]);
      expect(selector.trackId).toBe(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);
    });

    it('syncs the DOM value with a programmatic setSelectedTrack', () => {
      createSelectorFixture();
      const selector = createNeatRacingTrackSelector(HUD_OUTPUT_ID);
      const { notified, listener } = createNotifyRecorder();
      selector.onSwitch(listener);

      selector.setSelectedTrack(OTHER_TRACK_ID);

      expect(notified).toEqual([OTHER_TRACK_ID]);
      expect(selector.select.value).toBe(OTHER_TRACK_ID);
      expect(selector.trackId).toBe(OTHER_TRACK_ID);
    });

    it('fires the restart hook in both directions through the overlay', () => {
      createSelectorFixture();
      const selector = createNeatRacingTrackSelector(HUD_OUTPUT_ID);
      const { notified, listener } = createNotifyRecorder();
      selector.onSwitch(listener);

      // Out: a user change pick. Back: a programmatic switch (the host
      // wiring and the user both ride the ONE restart hook).
      selector.select.value = OTHER_TRACK_ID;
      selector.select.dispatchEvent(new Event('change'));
      selector.setSelectedTrack(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);

      expect(notified).toEqual([
        OTHER_TRACK_ID,
        NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
      ]);
      expect(selector.trackId).toBe(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);
      // The DOM control stays synchronized with the programmatic switch.
      expect(selector.select.value).toBe(
        NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
      );
    });

    it('keeps the same-value programmatic switch silent (no notify, no DOM churn)', () => {
      createSelectorFixture();
      const selector = createNeatRacingTrackSelector(HUD_OUTPUT_ID);
      const { notified, listener } = createNotifyRecorder();
      selector.onSwitch(listener);

      selector.setSelectedTrack(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);

      expect(notified).toEqual([]);
      expect(selector.trackId).toBe(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID);
      expect(selector.select.value).toBe(
        NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
      );
    });

    it('offsets below the race-mode selector overlay (top 44px / right 0 / zIndex 10)', () => {
      createSelectorFixture();
      const selector = createNeatRacingTrackSelector(HUD_OUTPUT_ID);

      expect(selector.select.style.position).toBe('absolute');
      expect(selector.select.style.top).toBe('44px');
      expect(selector.select.style.right).toBe('0px');
      expect(selector.select.style.zIndex).toBe('10');
    });

    it('uses the Neatenstein pattern tokens (bg / cyan border / monospace 14px / 4px 8px)', () => {
      createSelectorFixture();
      const selector = createNeatRacingTrackSelector(HUD_OUTPUT_ID);

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

    it('throws when the output container is missing (the mode-selector resolver law)', () => {
      document.body.innerHTML = '';

      expect(() => createNeatRacingTrackSelector(MISSING_OUTPUT_ID)).toThrow(
        `HUD container #${MISSING_OUTPUT_ID} not found; cannot create NeatRacing track selector`,
      );
    });
  });
});
