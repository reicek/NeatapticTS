import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for human-mode integration in
 * examples/neatenstein/browser-entry/harness/arms-race.ts.
 *
 * Covers:
 * - AC-601-S02-003: arms-race runner accepts a human-mode flag and a replay
 *   buffer, and triggers a replay-driven generation when the buffer is
 *   non-empty.
 */

interface ReplayBuffer {
  size: () => number;
  push: (ctx: unknown) => void;
}

interface ArmsRaceHumanModule {
  runArmsRaceGeneration: (options: {
    seed: number;
    generation: number;
    humanMode?: boolean;
    replayBuffer?: ReplayBuffer;
  }) => {
    generation: number;
    replayDriven: boolean;
    quality: { survivalTicks: number };
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

const minimalDeathContext = {
  hero: { position: { x: 0, y: 0 }, angleRad: 0, health: 0 },
  enemies: [],
  damageSource: 'enemy',
};

describe('Neatenstein harness arms-race human mode', () => {
  describe('AC-601-S02-003: human-mode flag and replay buffer', () => {
    it('returns a replayDriven flag when humanMode is enabled', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as unknown as ArmsRaceHumanModule;
      const buffer = createReplayBufferFixture([]);

      const result = runArmsRaceGeneration({
        seed: 42,
        generation: 1,
        humanMode: true,
        replayBuffer: buffer,
      });

      expect(typeof result.replayDriven).toBe('boolean');
    });

    it('triggers a replay-driven generation when the replay buffer is non-empty', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as unknown as ArmsRaceHumanModule;
      const buffer = createReplayBufferFixture([minimalDeathContext]);

      const result = runArmsRaceGeneration({
        seed: 42,
        generation: 1,
        humanMode: true,
        replayBuffer: buffer,
      });

      expect(result.replayDriven).toBe(true);
    });

    it('does not trigger a replay-driven generation when the replay buffer is empty', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as unknown as ArmsRaceHumanModule;
      const buffer = createReplayBufferFixture([]);

      const result = runArmsRaceGeneration({
        seed: 42,
        generation: 1,
        humanMode: true,
        replayBuffer: buffer,
      });

      expect(result.replayDriven).toBe(false);
    });
  });
});
