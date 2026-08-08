/** @jest-environment jsdom */

import { describe, expect, it } from '@jest/globals';

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
});
