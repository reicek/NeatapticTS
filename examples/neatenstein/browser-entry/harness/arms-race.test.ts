import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/arms-race.ts.
 *
 * Covers:
 * - AC-408: ARMS RACE mode advances at interactive rates with a deterministic
 *   generation barrier.
 * - AC-409: Main agent fitness is evaluated against a frozen MLP snapshot, not
 *   the live enemy MLP population.
 */

describe('Neatenstein harness arms-race', () => {
  describe('AC-408: interactive deterministic generation barrier', () => {
    it('exports runArmsRaceGeneration as a function', async () => {
      const mod = (await import('./arms-race.ts')) as Record<string, unknown>;
      expect(typeof mod.runArmsRaceGeneration).toBe('function');
    });

    it('advances the generation number', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as Record<string, any>;
      const first = runArmsRaceGeneration({ seed: 1, generation: 1 });
      const second = runArmsRaceGeneration({ seed: 1, generation: 2 });
      expect(second.generation).toBe(first.generation + 1);
    });

    it('returns within interactive-rate headroom', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as Record<string, any>;
      const start = Date.now();
      runArmsRaceGeneration({ seed: 1, generation: 1 });
      expect(Date.now() - start).toBeLessThan(1000);
    });

    it('produces deterministic barrier state for the same seed', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as Record<string, any>;
      const first = runArmsRaceGeneration({ seed: 123, generation: 1 });
      const second = runArmsRaceGeneration({ seed: 123, generation: 1 });
      expect({
        generation: first.generation,
        mainSnapshot: first.mainSnapshot,
        enemySnapshot: first.enemySnapshot,
      }).toEqual({
        generation: second.generation,
        mainSnapshot: second.mainSnapshot,
        enemySnapshot: second.enemySnapshot,
      });
    });
  });

  describe('AC-409: frozen MLP snapshot evaluation', () => {
    it('evaluates main fitness against an enemy snapshot', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as Record<string, any>;
      const result = runArmsRaceGeneration({
        seed: 5,
        generation: 1,
        enemySnapshot: { kind: 'mlp', weights: new Float32Array(80) },
      });
      expect(typeof result.quality.survivalTicks).toBe('number');
    });

    it('returns the supplied frozen snapshot instead of generating a new one', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as Record<string, any>;
      const frozen = { kind: 'mlp', weights: new Float32Array(80) };
      frozen.weights[0] = 0.5;
      const result = runArmsRaceGeneration({
        seed: 5,
        generation: 5,
        enemySnapshot: frozen,
      });
      expect(result.enemySnapshot.weights).toEqual(frozen.weights);
    });
  });
});
