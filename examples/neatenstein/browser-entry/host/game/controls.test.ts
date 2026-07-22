import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/controls.ts.
 *
 * Covers AC-206: pointer-lock mouse look, keyboard/touch fallbacks, and
 * worker-tier input forwarding.
 */

describe('Neatenstein game controls', () => {
  describe('AC-206: input binding contract', () => {
    it('exports bindPointerLock', async () => {
      const mod = (await import('./controls.ts')) as Record<string, unknown>;
      expect(typeof mod.bindPointerLock).toBe('function');
    });

    it('exports bindMouseLook', async () => {
      const mod = (await import('./controls.ts')) as Record<string, unknown>;
      expect(typeof mod.bindMouseLook).toBe('function');
    });

    it('exports bindKeyboardLook', async () => {
      const mod = (await import('./controls.ts')) as Record<string, unknown>;
      expect(typeof mod.bindKeyboardLook).toBe('function');
    });

    it('exports bindTouchLook', async () => {
      const mod = (await import('./controls.ts')) as Record<string, unknown>;
      expect(typeof mod.bindTouchLook).toBe('function');
    });

    it('exports forwardWorkerInput', async () => {
      const mod = (await import('./controls.ts')) as Record<string, unknown>;
      expect(typeof mod.forwardWorkerInput).toBe('function');
    });
  });
});
