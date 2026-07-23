import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/arms-race.ts
 * focused on the frozen MLP snapshot guarantee.
 *
 * Covers AC-409: Main agent fitness is evaluated against a frozen MLP snapshot,
 * not the live enemy MLP population.
 */

describe('Neatenstein harness enemy-mlp snapshot', () => {
  describe('AC-409: frozen snapshot vs live population', () => {
    it('exports runArmsRaceGeneration from the arms-race module', async () => {
      const mod = (await import('./arms-race.ts')) as Record<string, unknown>;
      expect(typeof mod.runArmsRaceGeneration).toBe('function');
    });

    it('evaluates main fitness against the supplied enemy snapshot', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as Record<string, any>;
      const snapshot = { kind: 'mlp', weights: new Float32Array(80) };
      const result = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: snapshot,
      });
      expect(result.enemySnapshot.weights).toEqual(snapshot.weights);
    });

    it('returns the same main fitness when replayed with the same frozen snapshot', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as Record<string, any>;
      const snapshot = { kind: 'mlp', weights: new Float32Array(80) };
      const first = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: snapshot,
      });
      const second = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: snapshot,
      });
      expect(first.quality).toEqual(second.quality);
    });

    it('produces different main fitness for different frozen snapshots', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as Record<string, any>;
      const firstSnapshot = { kind: 'mlp', weights: new Float32Array(80) };
      firstSnapshot.weights[0] = 0.1;
      const secondSnapshot = { kind: 'mlp', weights: new Float32Array(80) };
      secondSnapshot.weights[0] = 0.9;
      const first = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: firstSnapshot,
      });
      const second = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: secondSnapshot,
      });
      expect(first.quality).not.toEqual(second.quality);
    });

    it('does not mutate the live MLP population when evaluating from a snapshot', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as Record<string, any>;
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as Record<string, any>;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const liveBefore = population.snapshot().weights[0];
      const frozen = population.snapshot();
      runArmsRaceGeneration({
        seed: 1,
        generation: 5,
        enemySnapshot: frozen,
      });
      expect(population.snapshot().weights[0]).toBe(liveBefore);
    });
  });
});
