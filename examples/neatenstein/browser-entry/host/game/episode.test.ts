import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/episode.ts.
 *
 * Covers AC-208: a default episode ends within 15-25 seconds and replays
 * deterministically for the same seed.
 */

describe('Neatenstein game episode', () => {
  describe('AC-208: deterministic episode contract', () => {
    it('exports createEpisode', async () => {
      const mod = (await import('./episode.ts')) as Record<string, unknown>;
      expect(typeof mod.createEpisode).toBe('function');
    });

    it('exports runEpisode', async () => {
      const mod = (await import('./episode.ts')) as Record<string, unknown>;
      expect(typeof mod.runEpisode).toBe('function');
    });

    it('completes a default episode within 15-25 seconds', async () => {
      const { createEpisode, runEpisode } =
        (await import('./episode.ts')) as typeof import('./episode.ts');
      const episode = createEpisode({ seed: 1 });
      const final = runEpisode(episode);
      expect({
        withinMin: final.episodeTimeMs >= 15000,
        withinMax: final.episodeTimeMs <= 25000,
      }).toEqual({
        withinMin: true,
        withinMax: true,
      });
    });

    it('produces identical final state when replayed with the same seed', async () => {
      const { createEpisode, runEpisode } =
        (await import('./episode.ts')) as typeof import('./episode.ts');
      const run1 = runEpisode(createEpisode({ seed: 7 }));
      const run2 = runEpisode(createEpisode({ seed: 7 }));
      expect({
        health: run1.player.health,
        ammo: run1.player.ammo,
        kills: run1.kills,
        enemyCount: run1.enemies.length,
      }).toEqual({
        health: run2.player.health,
        ammo: run2.player.ammo,
        kills: run2.kills,
        enemyCount: run2.enemies.length,
      });
    });
  });
});
