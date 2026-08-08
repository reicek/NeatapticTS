/** @jest-environment jsdom */

import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for the neon Wolfenstein-style status bar in
 * examples/neatenstein/browser-entry/host/hud.ts.
 *
 * Covers:
 * - AC-004: status bar DOM structure, neon colors, and segmented bar geometry
 *   before implementation.
 */

const HUD_OUTPUT_ID = 'neatenstein-hud-output';

function createHudFixture(): { container: HTMLElement } {
  document.body.innerHTML = '';
  const container = document.createElement('div');
  container.id = HUD_OUTPUT_ID;
  document.body.appendChild(container);
  return { container };
}

interface NeonStatusBarState {
  hiveDensity?: number;
  playerHealth: number;
  playerMaxHealth: number;
  playerAmmo: number;
  playerMaxAmmo: number;
  playerKills?: number;
  playerDeaths?: number;
}

interface NeonStatusBarHud {
  container: HTMLElement;
  bar: HTMLElement;
  healthSegments: HTMLElement[];
  ammoSegments: HTMLElement[];
  hiveFill: HTMLElement;
  killsLabel: HTMLElement;
  deathsLabel: HTMLElement;
  update: (state: NeonStatusBarState) => void;
}

interface NeonStatusBarModule {
  createNeonStatusBar: (outputId: string) => NeonStatusBarHud;
}

const NEON_CYAN = 'rgb(0, 240, 255)';
const NEON_MAGENTA = 'rgb(255, 0, 85)';

describe('Neatenstein neon Wolfenstein-style HUD status bar', () => {
  describe('AC-004: status bar factory contract', () => {
    it('exports createNeonStatusBar as a function', async () => {
      const mod = (await import('./hud.ts')) as unknown as Record<
        string,
        unknown
      >;

      expect(typeof mod.createNeonStatusBar).toBe('function');
    });

    it('throws when the reserved outputId container is missing', async () => {
      document.body.innerHTML = '';
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;

      expect(() => createNeonStatusBar(HUD_OUTPUT_ID)).toThrow(
        /outputId|HUD container|status bar/,
      );
    });

    it('returns the host container, bar overlay, segmented tracks, and update callback', async () => {
      const { container } = createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      expect({
        container: hud.container,
        bar: hud.bar,
        healthSegments: hud.healthSegments.length,
        ammoSegments: hud.ammoSegments.length,
        hiveFill: hud.hiveFill,
        killsLabel: hud.killsLabel,
        deathsLabel: hud.deathsLabel,
        update: typeof hud.update,
      }).toEqual({
        container,
        bar: expect.any(HTMLElement),
        healthSegments: expect.any(Number),
        ammoSegments: expect.any(Number),
        hiveFill: expect.any(HTMLElement),
        killsLabel: expect.any(HTMLElement),
        deathsLabel: expect.any(HTMLElement),
        update: 'function',
      });
    });
  });

  describe('AC-004: neon status bar geometry and colors', () => {
    it('positions the bar as an absolutely-positioned bottom overlay', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      expect({
        position: hud.bar.style.position,
        bottom: hud.bar.style.bottom,
        left: hud.bar.style.left,
        width: hud.bar.style.width,
      }).toEqual({
        position: 'absolute',
        bottom: '0px',
        left: '0px',
        width: '100%',
      });
    });

    it('renders a neon border color on the status bar', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      expect(hud.bar.style.borderColor).toMatch(
        /rgb\(0,\s*240,\s*255\)|#00f0ff/,
      );
    });

    it('renders segmented health and ammo tracks with multiple segments', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      expect({
        healthSegmentCount: hud.healthSegments.length,
        ammoSegmentCount: hud.ammoSegments.length,
      }).toEqual({
        healthSegmentCount: expect.any(Number),
        ammoSegmentCount: expect.any(Number),
      });

      expect(hud.healthSegments.length).toBeGreaterThan(1);
      expect(hud.ammoSegments.length).toBeGreaterThan(1);
    });
  });

  describe('AC-004: segmented bar state updates', () => {
    it('lights the expected number of health segments by health fraction', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        playerHealth: 70,
        playerMaxHealth: 100,
        playerAmmo: 50,
        playerMaxAmmo: 100,
      });

      const activeHealth = hud.healthSegments.filter(
        (segment) => segment.style.backgroundColor === NEON_CYAN,
      ).length;

      expect(activeHealth).toBeGreaterThan(0);
      expect(activeHealth).toBeLessThanOrEqual(hud.healthSegments.length);
    });

    it('lights all health segments when health is full', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        playerHealth: 100,
        playerMaxHealth: 100,
        playerAmmo: 0,
        playerMaxAmmo: 100,
      });

      const activeHealth = hud.healthSegments.filter(
        (segment) => segment.style.backgroundColor === NEON_CYAN,
      ).length;

      expect(activeHealth).toBe(hud.healthSegments.length);
    });

    it('lights no health segments when health is zero', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        playerHealth: 0,
        playerMaxHealth: 100,
        playerAmmo: 50,
        playerMaxAmmo: 100,
      });

      const activeHealth = hud.healthSegments.filter(
        (segment) => segment.style.backgroundColor === NEON_CYAN,
      ).length;

      expect(activeHealth).toBe(0);
    });

    it('switches health segment color to magenta when health is critically low', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        playerHealth: 20,
        playerMaxHealth: 100,
        playerAmmo: 50,
        playerMaxAmmo: 100,
      });

      const activeHealth = hud.healthSegments.filter(
        (segment) => segment.style.backgroundColor === NEON_MAGENTA,
      ).length;

      expect(activeHealth).toBeGreaterThan(0);
    });
  });

  describe('AC-004: HIVE density and kill/death readouts', () => {
    it('updates the HIVE density fill width from density fraction', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        hiveDensity: 0.37,
        playerHealth: 100,
        playerMaxHealth: 100,
        playerAmmo: 50,
        playerMaxAmmo: 100,
      });

      expect(hud.hiveFill.style.width).toBe('37%');
    });

    it('displays kill and death counts from the update state', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        playerHealth: 100,
        playerMaxHealth: 100,
        playerAmmo: 50,
        playerMaxAmmo: 100,
        playerKills: 42,
        playerDeaths: 7,
      });

      expect({
        killsText: hud.killsLabel.textContent,
        deathsText: hud.deathsLabel.textContent,
      }).toEqual({
        killsText: expect.stringContaining('42'),
        deathsText: expect.stringContaining('7'),
      });
    });
  });

  describe('division-by-zero guards: zero max values', () => {
    it('lights no health segments when playerMaxHealth is zero', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        playerHealth: 50,
        playerMaxHealth: 0,
        playerAmmo: 50,
        playerMaxAmmo: 100,
      });

      const activeHealth = hud.healthSegments.filter(
        (segment) => segment.style.backgroundColor === NEON_CYAN,
      ).length;

      expect(activeHealth).toBe(0);
    });

    it('lights no ammo segments when playerMaxAmmo is zero', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        playerHealth: 100,
        playerMaxHealth: 100,
        playerAmmo: 50,
        playerMaxAmmo: 0,
      });

      const activeAmmo = hud.ammoSegments.filter(
        (segment) => segment.style.backgroundColor === NEON_CYAN,
      ).length;

      expect(activeAmmo).toBe(0);
    });
  });

  describe('nullish coalescing fallbacks: undefined optional fields', () => {
    it('falls back to 0% HIVE density when hiveDensity is undefined', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        hiveDensity: undefined,
        playerHealth: 100,
        playerMaxHealth: 100,
        playerAmmo: 50,
        playerMaxAmmo: 100,
        playerKills: 5,
        playerDeaths: 2,
      });

      expect(hud.hiveFill.style.width).toBe('0%');
    });

    it('falls back to 0 kills when playerKills is undefined', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        playerHealth: 100,
        playerMaxHealth: 100,
        playerAmmo: 50,
        playerMaxAmmo: 100,
        playerKills: undefined,
        playerDeaths: 2,
      });

      expect(hud.killsLabel.textContent).toBe('0');
    });

    it('falls back to 0 deaths when playerDeaths is undefined', async () => {
      createHudFixture();
      const { createNeonStatusBar } =
        (await import('./hud.ts')) as unknown as NeonStatusBarModule;
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);

      hud.update({
        playerHealth: 100,
        playerMaxHealth: 100,
        playerAmmo: 50,
        playerMaxAmmo: 100,
        playerKills: 5,
        playerDeaths: undefined,
      });

      expect(hud.deathsLabel.textContent).toBe('0');
    });
  });
});
