/** @jest-environment jsdom */

import {
  describe,
  expect,
  it,
  jest,
  beforeEach,
  afterEach,
} from '@jest/globals';

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
const loadModule = (path: string): Promise<any> => import(path);

const HUD_OUTPUT_ID = 'neatenstein-hud-output';

function createHudFixture(): { container: HTMLElement } {
  document.body.innerHTML = '';
  const container = document.createElement('div');
  container.id = HUD_OUTPUT_ID;
  document.body.appendChild(container);
  return { container };
}

describe('Neatenstein host HUD overlay', () => {
  describe('AC-501-S05-001: HUD container resolution', () => {
    it('resolves the host HUD container from the reserved outputId', async () => {
      const { container } = createHudFixture();
      const { createHiveDensityHud } = await loadModule('./hud.ts');
      const hud = createHiveDensityHud(HUD_OUTPUT_ID);

      expect(hud.container).toBe(container);
    });

    it('throws when the reserved outputId container is missing', async () => {
      document.body.innerHTML = '';
      const { createHiveDensityHud } = await loadModule('./hud.ts');

      expect(() => createHiveDensityHud(HUD_OUTPUT_ID)).toThrow(
        /outputId|HUD container/,
      );
    });
  });

  describe('AC-501-S05-002: HIVE DENSITY meter rendering', () => {
    it('renders a meter with the expected dimensions and label text', async () => {
      createHudFixture();
      const { createHiveDensityHud } = await loadModule('./hud.ts');
      const hud = createHiveDensityHud(HUD_OUTPUT_ID);

      expect({
        meterWidth: hud.meter.style.width,
        meterHeight: hud.meter.style.height,
        labelText: hud.label.textContent,
      }).toEqual({
        meterWidth: '160px',
        meterHeight: '8px',
        labelText: 'HIVE DENSITY',
      });
    });

    it('uses the calm color for densities below the first threshold', async () => {
      createHudFixture();
      const { createHiveDensityHud } = await loadModule('./hud.ts');
      const hud = createHiveDensityHud(HUD_OUTPUT_ID);
      hud.update({ hiveDensity: 0.1 });

      expect(hud.fill.style.backgroundColor).toBe('rgb(0, 240, 255)');
    });

    it('uses the low-threshold color between 0.25 and 0.50', async () => {
      createHudFixture();
      const { createHiveDensityHud } = await loadModule('./hud.ts');
      const hud = createHiveDensityHud(HUD_OUTPUT_ID);
      hud.update({ hiveDensity: 0.37 });

      expect(hud.fill.style.backgroundColor).toBe('rgb(160, 240, 0)');
    });

    it('uses the mid-threshold color between 0.50 and 0.75', async () => {
      createHudFixture();
      const { createHiveDensityHud } = await loadModule('./hud.ts');
      const hud = createHiveDensityHud(HUD_OUTPUT_ID);
      hud.update({ hiveDensity: 0.62 });

      expect(hud.fill.style.backgroundColor).toBe('rgb(240, 160, 0)');
    });

    it('uses the high-threshold color at or above 0.75', async () => {
      createHudFixture();
      const { createHiveDensityHud } = await loadModule('./hud.ts');
      const hud = createHiveDensityHud(HUD_OUTPUT_ID);
      hud.update({ hiveDensity: 0.9 });

      expect(hud.fill.style.backgroundColor).toBe('rgb(255, 0, 85)');
    });
  });

  describe('AC-501-S05-003: overlay render-state updates', () => {
    it('updates the meter fill width from a render-state density field', async () => {
      createHudFixture();
      const { createHiveDensityHud } = await loadModule('./hud.ts');
      const hud = createHiveDensityHud(HUD_OUTPUT_ID);
      hud.update({ hiveDensity: 0.25 });

      expect(hud.fill.style.width).toBe('25%');
    });

    it('updates the label percentage from a render-state density field', async () => {
      createHudFixture();
      const { createHiveDensityHud } = await loadModule('./hud.ts');
      const hud = createHiveDensityHud(HUD_OUTPUT_ID);
      hud.update({ hiveDensity: 0.75 });

      expect(hud.label.textContent).toContain('75%');
    });
  });

  describe('AC-501-S05-003b: Neon status bar rendering', () => {
    it('exports createNeonStatusBar as a function', async () => {
      const mod = (await loadModule('./hud.ts')) as Record<string, unknown>;
      expect(typeof mod.createNeonStatusBar).toBe('function');
    });

    it('renders a default generation of 0 when generation is omitted', async () => {
      createHudFixture();
      const { createNeonStatusBar } = await loadModule('./hud.ts');
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);
      hud.update({
        playerHealth: 50,
        playerMaxHealth: 100,
        playerAmmo: 25,
        playerMaxAmmo: 50,
      });
      expect(hud.generationLabel.textContent).toBe('0');
    });

    it('renders an explicit generation value', async () => {
      createHudFixture();
      const { createNeonStatusBar } = await loadModule('./hud.ts');
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);
      hud.update({
        playerHealth: 50,
        playerMaxHealth: 100,
        playerAmmo: 25,
        playerMaxAmmo: 50,
        generation: 7,
      });
      expect(hud.generationLabel.textContent).toBe('7');
    });

    it('falls back to 0 when generation is explicitly undefined', async () => {
      createHudFixture();
      const { createNeonStatusBar } = await loadModule('./hud.ts');
      const hud = createNeonStatusBar(HUD_OUTPUT_ID);
      hud.update({
        playerHealth: 50,
        playerMaxHealth: 100,
        playerAmmo: 25,
        playerMaxAmmo: 50,
        generation: undefined,
      });
      expect(hud.generationLabel.textContent).toBe('0');
    });
  });

  describe('AC-501-S05-004: Wave announcement overlay', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('throws when the reserved outputId container is missing', async () => {
      document.body.innerHTML = '';
      const { createWaveAnnouncement } = await loadModule('./hud.ts');

      expect(() => createWaveAnnouncement(HUD_OUTPUT_ID)).toThrow(
        /outputId|wave announcement/,
      );
    });

    it('clears the pending hide timer when show is called twice', async () => {
      createHudFixture();
      const { createWaveAnnouncement } = await loadModule('./hud.ts');
      const { overlay, show } = createWaveAnnouncement(HUD_OUTPUT_ID);

      show(1);
      expect(overlay.style.opacity).toBe('1');

      // Spy on clearTimeout to verify the first timer is cancelled.
      const clearTimeoutSpy = jest.spyOn(globalThis, 'clearTimeout');

      // Second call should clear the first timer and schedule a new one.
      show(2);
      expect(clearTimeoutSpy).toHaveBeenCalled();

      // Advance past HOLD_MS (600ms) — the second timer fires, not the first.
      jest.advanceTimersByTime(600);
      expect(overlay.style.opacity).toBe('0');

      // The glow layer text should reflect the second call.
      const glowLayer = overlay.firstChild as HTMLElement;
      expect(glowLayer.textContent).toBe('Wave 2');
    });

    it('completes the full fade-out cycle and clears the nested timer', async () => {
      createHudFixture();
      const { createWaveAnnouncement } = await loadModule('./hud.ts');
      const { overlay, show } = createWaveAnnouncement(HUD_OUTPUT_ID);

      show(1);
      expect(overlay.style.opacity).toBe('1');

      // Advance past HOLD_MS (600ms) — first setTimeout fires, starts fade.
      jest.advanceTimersByTime(600);
      expect(overlay.style.opacity).toBe('0');

      // Advance past FADE_MS (500ms) — nested setTimeout fires, hideTimer
      // is set to null.  We verify by calling show() again: it should NOT
      // call clearTimeout because hideTimer is already null.
      const clearTimeoutSpy = jest.spyOn(globalThis, 'clearTimeout');
      jest.advanceTimersByTime(500);

      show(2);
      expect(clearTimeoutSpy).not.toHaveBeenCalled();
      expect(overlay.style.opacity).toBe('1');
    });
  });
});
