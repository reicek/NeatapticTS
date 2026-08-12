import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for per-death evolution in
 * examples/neatenstein/browser-entry/harness/arms-race.ts.
 *
 * Covers:
 * - AC-601-S05-001: runArmsRaceGeneration with a non-empty replay buffer
 *   produces a generation whose enemy fitness is influenced by the replayed
 *   death contexts.
 * - AC-601-S05-002: per-death evolution pulse produces a measurable shift in
 *   enemy behavior metrics compared to a baseline generation without replay.
 */

interface ReplayBuffer {
  size: () => number;
  push: (ctx: unknown) => void;
}

interface EnemyBehaviorMetrics {
  aggression: number;
  movementPattern: number;
  positioning: number;
}

interface ArmsRacePerDeathModule {
  runArmsRaceGeneration: (options: {
    seed: number;
    generation: number;
    humanMode?: boolean;
    replayBuffer?: ReplayBuffer;
  }) => {
    generation: number;
    replayDriven: boolean;
    replayPressure: number;
    enemyBehaviorMetrics: EnemyBehaviorMetrics;
  };
}

function createReplayBufferFixture(entries: unknown[]): ReplayBuffer {
  return {
    size: () => entries.length,
    push: (ctx) => {
      entries.push(ctx);
    },
  };
}

const deathContextA = {
  hero: { position: { x: 3, y: 5 }, angleRad: 1.2, health: 0 },
  enemies: [{ kind: 'swarm', dna: 'alpha', coordinates: [{ x: 2, y: 2 }] }],
  damageSource: 'enemy-swarm',
};

const deathContextB = {
  hero: { position: { x: 7, y: 1 }, angleRad: 0.5, health: 0 },
  enemies: [{ kind: 'swarm', dna: 'beta', coordinates: [{ x: 6, y: 4 }] }],
  damageSource: 'enemy-swarm',
};

describe('Neatenstein harness per-death evolution', () => {
  describe('AC-601-S05-001: replay-driven enemy fitness', () => {
    it('produces a replayPressure field quantifying replay influence on enemy fitness', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as unknown as ArmsRacePerDeathModule;
      const buffer = createReplayBufferFixture([deathContextA, deathContextB]);

      const result = runArmsRaceGeneration({
        seed: 42,
        generation: 1,
        humanMode: true,
        replayBuffer: buffer,
      });

      expect(typeof result.replayPressure).toBe('number');
    });

    it('applies non-zero replay pressure when the replay buffer is non-empty', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as unknown as ArmsRacePerDeathModule;
      const buffer = createReplayBufferFixture([deathContextA, deathContextB]);

      const result = runArmsRaceGeneration({
        seed: 42,
        generation: 1,
        humanMode: true,
        replayBuffer: buffer,
      });

      expect(result.replayPressure).toBeGreaterThan(0);
    });

    it('applies zero replay pressure when the replay buffer is empty', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as unknown as ArmsRacePerDeathModule;
      const buffer = createReplayBufferFixture([]);

      const result = runArmsRaceGeneration({
        seed: 42,
        generation: 1,
        humanMode: true,
        replayBuffer: buffer,
      });

      expect(result.replayPressure).toBe(0);
    });
  });

  describe('AC-601-S05-002: measurable behavior shift vs baseline', () => {
    it('produces enemy behavior metrics on the generation result', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as unknown as ArmsRacePerDeathModule;
      const buffer = createReplayBufferFixture([deathContextA, deathContextB]);

      const result = runArmsRaceGeneration({
        seed: 42,
        generation: 1,
        humanMode: true,
        replayBuffer: buffer,
      });

      expect(result.enemyBehaviorMetrics).toBeDefined();
    });

    it('produces a measurable shift in enemy behavior metrics compared to a baseline without replay', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as unknown as ArmsRacePerDeathModule;

      const baselineResult = runArmsRaceGeneration({
        seed: 42,
        generation: 1,
        humanMode: false,
      });

      const replayResult = runArmsRaceGeneration({
        seed: 42,
        generation: 1,
        humanMode: true,
        replayBuffer: createReplayBufferFixture([deathContextA, deathContextB]),
      });

      expect(replayResult.enemyBehaviorMetrics).not.toEqual(
        baselineResult.enemyBehaviorMetrics,
      );
    });
  });
});
