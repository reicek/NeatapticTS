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

describe('Neatenstein health/ammo HUD overlay', () => {
  describe('AC-801-S02-003: createHealthAmmoHud factory contract', () => {
    it('returns an object with container, healthTrack, healthFill, healthLabel, ammoLabel, and update', async () => {
      const { container } = createHudFixture();
      const { createHealthAmmoHud } = await loadModule('./hud.ts');
      const hud = createHealthAmmoHud(HUD_OUTPUT_ID);

      expect(hud.container).toBe(container);
      expect(hud.healthTrack).toBeDefined();
      expect(hud.healthFill).toBeDefined();
      expect(hud.healthLabel).toBeDefined();
      expect(hud.ammoLabel).toBeDefined();
      expect(typeof hud.update).toBe('function');
    });

    it('throws when the reserved outputId container is missing', async () => {
      document.body.innerHTML = '';
      const { createHealthAmmoHud } = await loadModule('./hud.ts');

      expect(() => createHealthAmmoHud(HUD_OUTPUT_ID)).toThrow(
        /outputId|HUD container/,
      );
    });
  });

  describe('AC-801-S02-004: health fill width and color thresholds', () => {
    it('sets health fill width to health/maxHealth percentage', async () => {
      createHudFixture();
      const { createHealthAmmoHud } = await loadModule('./hud.ts');
      const hud = createHealthAmmoHud(HUD_OUTPUT_ID);

      hud.update({ health: 75, maxHealth: 100, ammo: 30, maxAmmo: 50 });

      expect(hud.healthFill.style.width).toBe('75%');
    });

    it('uses cyan color when health fraction is at or above 70%', async () => {
      createHudFixture();
      const { createHealthAmmoHud } = await loadModule('./hud.ts');
      const hud = createHealthAmmoHud(HUD_OUTPUT_ID);

      hud.update({ health: 70, maxHealth: 100, ammo: 30, maxAmmo: 50 });

      expect(hud.healthFill.style.backgroundColor).toBe('rgb(0, 240, 255)');
    });

    it('uses amber color when health fraction is between 30% and 69%', async () => {
      createHudFixture();
      const { createHealthAmmoHud } = await loadModule('./hud.ts');
      const hud = createHealthAmmoHud(HUD_OUTPUT_ID);

      hud.update({ health: 50, maxHealth: 100, ammo: 30, maxAmmo: 50 });

      expect(hud.healthFill.style.backgroundColor).toBe('rgb(240, 160, 0)');
    });

    it('uses magenta color when health fraction is below 30%', async () => {
      createHudFixture();
      const { createHealthAmmoHud } = await loadModule('./hud.ts');
      const hud = createHealthAmmoHud(HUD_OUTPUT_ID);

      hud.update({ health: 20, maxHealth: 100, ammo: 30, maxAmmo: 50 });

      expect(hud.healthFill.style.backgroundColor).toBe('rgb(255, 0, 85)');
    });
  });

  describe('AC-801-S02-005: ammo label format', () => {
    it('displays ammo label in AMMO ammo/maxAmmo format', async () => {
      createHudFixture();
      const { createHealthAmmoHud } = await loadModule('./hud.ts');
      const hud = createHealthAmmoHud(HUD_OUTPUT_ID);

      hud.update({ health: 100, maxHealth: 100, ammo: 30, maxAmmo: 50 });

      expect(hud.ammoLabel.textContent).toBe('AMMO 30/50');
    });

    it('updates ammo label text on each update call', async () => {
      createHudFixture();
      const { createHealthAmmoHud } = await loadModule('./hud.ts');
      const hud = createHealthAmmoHud(HUD_OUTPUT_ID);

      hud.update({ health: 100, maxHealth: 100, ammo: 45, maxAmmo: 50 });
      expect(hud.ammoLabel.textContent).toBe('AMMO 45/50');

      hud.update({ health: 100, maxHealth: 100, ammo: 10, maxAmmo: 50 });
      expect(hud.ammoLabel.textContent).toBe('AMMO 10/50');
    });
  });

  describe('division-by-zero guard: maxHealth = 0', () => {
    it('falls back to 0% health fill when maxHealth is zero', async () => {
      createHudFixture();
      const { createHealthAmmoHud } = await loadModule('./hud.ts');
      const hud = createHealthAmmoHud(HUD_OUTPUT_ID);

      hud.update({ health: 50, maxHealth: 0, ammo: 30, maxAmmo: 50 });

      expect(hud.healthFill.style.width).toBe('0%');
      expect(hud.healthLabel.textContent).toBe('HEALTH 0%');
    });
  });
});
