import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/cadence.ts.
 *
 * Covers AC-209: the fixed-timestep design must support at least 2 generations
 * per minute in AI modes.
 */

describe('Neatenstein game cadence', () => {
  describe('AC-209: generation cadence contract', () => {
    it('exports estimateGenerationsPerMinute', async () => {
      const mod = (await import('./cadence.ts')) as Record<string, unknown>;
      expect(typeof mod.estimateGenerationsPerMinute).toBe('function');
    });

    it('exports a target minimum generations per minute', async () => {
      const mod = (await import('./cadence.ts')) as Record<string, unknown>;
      expect(typeof mod.NEATENSTEIN_TARGET_MIN_GENERATIONS_PER_MINUTE).toBe(
        'number',
      );
    });

    it('reports at least 2 generations per minute for default settings', async () => {
      const {
        estimateGenerationsPerMinute,
        NEATENSTEIN_TARGET_MIN_GENERATIONS_PER_MINUTE,
      } = (await import('./cadence.ts')) as Record<string, any>;
      const rate = estimateGenerationsPerMinute({
        episodeDurationMs: 20000,
        evaluationOverheadMs: 2000,
      });
      expect(rate).toBeGreaterThanOrEqual(
        NEATENSTEIN_TARGET_MIN_GENERATIONS_PER_MINUTE,
      );
    });

    it('target minimum is greater than or equal to 2', async () => {
      const mod = (await import('./cadence.ts')) as Record<string, unknown>;
      expect(
        (mod.NEATENSTEIN_TARGET_MIN_GENERATIONS_PER_MINUTE as number) >= 2,
      ).toBe(true);
    });
  });
});
