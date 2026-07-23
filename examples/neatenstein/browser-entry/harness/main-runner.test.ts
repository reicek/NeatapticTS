import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/main-runner.ts.
 *
 * Covers AC-307: a single main agent lifecycle runner advances a deterministic
 * generation against a frozen enemy snapshot and emits a CombatQualitySignal.
 */

describe('Neatenstein harness main-runner', () => {
  describe('AC-307: single main agent lifecycle runner', () => {
    it('exports runMainGeneration as a function', async () => {
      const mod = (await import('./main-runner.ts')) as Record<string, unknown>;
      expect(typeof mod.runMainGeneration).toBe('function');
    });

    it('returns a CombatQualitySignal for a valid config', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as Record<string, any>;
      const result = runMainGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: { kind: 'mlp', weights: new Float32Array(8) },
      });
      expect({
        hasSurvivalTicks: typeof result.survivalTicks === 'number',
        hasDamageDealt: typeof result.damageDealt === 'number',
        hasKills: typeof result.kills === 'number',
        hasDamageTaken: typeof result.damageTaken === 'number',
        hasAimMissRate: typeof result.aimMissRate === 'number',
        hasComplexityBonus: typeof result.complexityBonus === 'number',
        hasParsimonyPenalty: typeof result.parsimonyDensityPenalty === 'number',
      }).toEqual({
        hasSurvivalTicks: true,
        hasDamageDealt: true,
        hasKills: true,
        hasDamageTaken: true,
        hasAimMissRate: true,
        hasComplexityBonus: true,
        hasParsimonyPenalty: true,
      });
    });

    it('produces deterministic output for the same config', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as Record<string, any>;
      const config = {
        seed: 7,
        generation: 2,
        enemySnapshot: { kind: 'mlp', weights: new Float32Array(8) },
      };
      const first = runMainGeneration(config);
      const second = runMainGeneration(config);
      expect(first).toEqual(second);
    });

    it('falls back to a generated MLP enemy snapshot when none is supplied', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as Record<string, any>;
      const result = runMainGeneration({ seed: 3, generation: 1 });
      expect(typeof result.survivalTicks).toBe('number');
    });

    it('refreshes the fallback MLP enemy snapshot on refresh generations', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as Record<string, any>;
      const result = runMainGeneration({ seed: 3, generation: 5 });
      expect(typeof result.survivalTicks).toBe('number');
    });

    it('produces deterministic output for a swarm enemy snapshot', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as Record<string, any>;
      const config = {
        seed: 5,
        generation: 1,
        enemySnapshot: {
          kind: 'swarm',
          dna: 'swarm-dna',
          coordinates: [{ x: 0.1, y: 0.2 }],
        },
      };
      const first = runMainGeneration(config);
      const second = runMainGeneration(config);
      expect(first).toEqual(second);
    });
  });
});
