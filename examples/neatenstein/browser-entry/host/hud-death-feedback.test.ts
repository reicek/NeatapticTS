/** @jest-environment jsdom */

import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for the death feedback indicator in
 * examples/neatenstein/browser-entry/host/hud.ts.
 *
 * Covers:
 * - AC-601-S05-004: HUD displays a death feedback indicator showing adaptation
 *   direction (stronger/WEAKER/shifted) without requiring a real browser.
 */

const HUD_OUTPUT_ID = 'neatenstein-hud-output';

function createHudFixture(): { container: HTMLElement } {
  document.body.innerHTML = '';
  const container = document.createElement('div');
  container.id = HUD_OUTPUT_ID;
  document.body.appendChild(container);
  return { container };
}

interface AdaptationSignal {
  direction: 'stronger' | 'weaker' | 'shifted';
  aggressionDelta: number;
  movementDelta: number;
  positioningDelta: number;
}

interface DeathFeedbackHudModule {
  createDeathFeedbackIndicator: (outputId: string) => {
    container: HTMLElement;
    indicator: HTMLElement;
    label: HTMLElement;
    update: (signal: AdaptationSignal) => void;
  };
}

describe('Neatenstein host HUD death feedback indicator', () => {
  describe('AC-601-S05-004: adaptation direction indicator', () => {
    it('exports createDeathFeedbackIndicator as a function', async () => {
      const mod = (await import('./hud.ts')) as unknown as Record<
        string,
        unknown
      >;

      expect(typeof mod.createDeathFeedbackIndicator).toBe('function');
    });

    it('renders an indicator element into the host container', async () => {
      createHudFixture();
      const { createDeathFeedbackIndicator } =
        (await import('./hud.ts')) as unknown as DeathFeedbackHudModule;
      const hud = createDeathFeedbackIndicator(HUD_OUTPUT_ID);

      expect(hud.indicator).toBeDefined();
      expect(hud.indicator.tagName).toBeDefined();
    });

    it('displays the adaptation direction label from an adaptation signal', async () => {
      createHudFixture();
      const { createDeathFeedbackIndicator } =
        (await import('./hud.ts')) as unknown as DeathFeedbackHudModule;
      const hud = createDeathFeedbackIndicator(HUD_OUTPUT_ID);

      hud.update({
        direction: 'stronger',
        aggressionDelta: 0.4,
        movementDelta: -0.1,
        positioningDelta: 0.3,
      });

      expect(hud.label.textContent).toMatch(/stronger/i);
    });
  });
});
