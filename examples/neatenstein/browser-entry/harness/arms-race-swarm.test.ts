import { describe, expect, it } from '@jest/globals';

import type { ArmsRaceGenerationResult } from './arms-race';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/arms-race.ts
 * SWARM default enemy snapshot (Phase 5 Step 02).
 *
 * Covers AC-501-S02-002: the arms-race runner defaults to a SWARM enemy
 * snapshot and advances the generation deterministically.
 */

interface RunArmsRaceGenerationWithEnemy {
  seed: number;
  generation: number;
  enemySnapshot?: unknown;
}

interface ArmsRaceSwarmModule {
  runArmsRaceGeneration: (
    options: RunArmsRaceGenerationWithEnemy,
  ) => ArmsRaceGenerationResult;
}

describe('Neatenstein harness arms-race SWARM default', () => {
  it('resolves a deterministic SWARM enemy snapshot by default and advances the generation', async () => {
    const { runArmsRaceGeneration } =
      (await import('./arms-race.ts')) as unknown as ArmsRaceSwarmModule;

    const first = runArmsRaceGeneration({ seed: 7, generation: 2 });
    const second = runArmsRaceGeneration({ seed: 7, generation: 2 });

    expect(first.enemySnapshot.kind).toBe('swarm');
    expect(first.generation).toBe(3);
    expect(first.enemySnapshot).toEqual(second.enemySnapshot);
  });
});
