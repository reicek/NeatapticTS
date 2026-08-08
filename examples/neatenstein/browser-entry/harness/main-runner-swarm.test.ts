import { describe, expect, it } from '@jest/globals';

import type { MainGenerationResult } from './main-runner';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/main-runner.ts
 * SWARM default resolution (Phase 5 Step 02).
 *
 * Covers AC-501-S02-001: the runner resolves a SWARM enemy snapshot when
 * enemy.kind is 'swarm' and returns it in evaluatedEnemySnapshot.
 */

interface RunMainGenerationWithEnemy {
  seed: number;
  generation: number;
  enemy?: { kind: 'swarm' | 'mlp' };
}

interface MainRunnerSwarmModule {
  runMainGeneration: (
    options: RunMainGenerationWithEnemy,
  ) => MainGenerationResult;
}

describe('Neatenstein harness main-runner SWARM default', () => {
  it('resolves a SWARM enemy snapshot when enemy.kind is "swarm"', async () => {
    const { runMainGeneration } =
      (await import('./main-runner.ts')) as unknown as MainRunnerSwarmModule;

    const result = runMainGeneration({
      seed: 42,
      generation: 0,
      enemy: { kind: 'swarm' },
    });

    expect(result.evaluatedEnemySnapshot.kind).toBe('swarm');
  });

  it('returns a deterministic SWARM snapshot for the same seed and generation', async () => {
    const { runMainGeneration } =
      (await import('./main-runner.ts')) as unknown as MainRunnerSwarmModule;

    const options = {
      seed: 42,
      generation: 0,
      enemy: { kind: 'swarm' },
    } as const;

    const first = runMainGeneration(options);
    const second = runMainGeneration(options);

    expect(first.evaluatedEnemySnapshot.kind).toBe('swarm');
    expect(first.evaluatedEnemySnapshot).toEqual(second.evaluatedEnemySnapshot);
  });
});
