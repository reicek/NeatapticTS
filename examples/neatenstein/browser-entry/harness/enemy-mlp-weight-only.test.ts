import { describe, expect, it } from '@jest/globals';

import type * as EnemyMlp from './enemy-mlp';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/enemy-mlp.ts
 * weight-only mutation guard (Phase 4 slice 04-enemy-mlp-weight-only).
 *
 * Covers AC-404: MLP enemies use a fixed 8->6->4->2 topology, weight-only
 * mutation, and a runtime guard that rejects structural mutation operators.
 */

interface EnemyMlpModule {
  createMlpEnemyPopulation: typeof EnemyMlp.createMlpEnemyPopulation;
  guardMlpStructuralMutation: typeof EnemyMlp.guardMlpStructuralMutation;
}

describe('Neatenstein harness enemy-mlp weight-only', () => {
  describe('AC-404: fixed topology', () => {
    it('produces weight vectors sized for the 8->6->4->2 topology', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const variant = population.sample(0) as { weights: Float32Array };
      expect(variant.weights.length).toBe(8 * 6 + 6 * 4 + 4 * 2);
    });
  });

  describe('AC-404: structural mutation guard', () => {
    it('exports guardMlpStructuralMutation as a function', async () => {
      const mod = (await import('./enemy-mlp.ts')) as Record<string, unknown>;
      expect(typeof mod.guardMlpStructuralMutation).toBe('function');
    });

    it('allows weight mutation operators', async () => {
      const { guardMlpStructuralMutation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      expect(guardMlpStructuralMutation({ type: 'weight' })).toBe(true);
    });

    it('rejects add-node structural mutation operators', async () => {
      const { guardMlpStructuralMutation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      expect(guardMlpStructuralMutation({ type: 'add-node' })).toBe(false);
    });

    it('rejects add-connection structural mutation operators', async () => {
      const { guardMlpStructuralMutation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      expect(guardMlpStructuralMutation({ type: 'add-connection' })).toBe(
        false,
      );
    });

    it('rejects remove-node structural mutation operators', async () => {
      const { guardMlpStructuralMutation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      expect(guardMlpStructuralMutation({ type: 'remove-node' })).toBe(false);
    });
  });

  describe('AC-404: main-agent integration', () => {
    it('wires the main-agent runner to MLP enemies', async () => {
      const { runMainAgentGeneration } =
        (await import('./main-agent.ts')) as Record<string, unknown>;
      expect(typeof runMainAgentGeneration).toBe('function');
    });
  });
});
