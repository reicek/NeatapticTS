/** @jest-environment jsdom */

import { describe, expect, it, jest } from '@jest/globals';

/**
 * Red-phase contract tests for the human-mode selector in
 * examples/neatenstein/browser-entry/host/hud.ts.
 *
 * Covers:
 * - AC-601-S02-004: HUD exposes a human-mode selector that toggles between
 *   auto-play and human-play without requiring a real browser.
 */

const HUD_OUTPUT_ID = 'neatenstein-hud-output';

function createHudFixture(): { container: HTMLElement } {
  document.body.innerHTML = '';
  const container = document.createElement('div');
  container.id = HUD_OUTPUT_ID;
  document.body.appendChild(container);
  return { container };
}

interface HumanModeHudModule {
  createHumanModeSelector: (outputId: string) => {
    container: HTMLElement;
    select: HTMLSelectElement;
    mode: 'auto' | 'human';
    setMode: (mode: 'auto' | 'human') => void;
    onToggle: (callback: (mode: 'auto' | 'human') => void) => void;
  };
}

describe('Neatenstein host HUD human-mode selector', () => {
  describe('AC-601-S02-004: auto-play / human-play toggle', () => {
    it('exports createHumanModeSelector as a function', async () => {
      const mod = (await import('./hud.ts')) as unknown as Record<
        string,
        unknown
      >;

      expect(typeof mod.createHumanModeSelector).toBe('function');
    });

    it('renders a select element with auto and human options', async () => {
      createHudFixture();
      const { createHumanModeSelector } =
        (await import('./hud.ts')) as unknown as HumanModeHudModule;
      const hud = createHumanModeSelector(HUD_OUTPUT_ID);

      expect({
        tag: hud.select.tagName,
        options: Array.from(hud.select.options).map((option) => option.value),
        initialMode: hud.mode,
      }).toEqual({
        tag: 'SELECT',
        options: ['auto', 'human'],
        initialMode: 'human',
      });
    });

    it('toggles the mode when the select value changes', async () => {
      createHudFixture();
      const { createHumanModeSelector } =
        (await import('./hud.ts')) as unknown as HumanModeHudModule;
      const hud = createHumanModeSelector(HUD_OUTPUT_ID);

      hud.select.value = 'human';
      hud.select.dispatchEvent(new Event('change'));

      expect(hud.mode).toBe('human');
    });

    it('notifies onToggle callbacks when the mode changes', async () => {
      createHudFixture();
      const { createHumanModeSelector } =
        (await import('./hud.ts')) as unknown as HumanModeHudModule;
      const hud = createHumanModeSelector(HUD_OUTPUT_ID);
      const callback = jest.fn();

      hud.onToggle(callback);
      hud.setMode('auto');

      expect(callback).toHaveBeenCalledWith('auto');
    });
  });
});
