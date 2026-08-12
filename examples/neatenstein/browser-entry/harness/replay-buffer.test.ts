import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/replay-buffer.ts.
 *
 * Covers:
 * - AC-601-S02-001: ReplayBuffer stores death contexts with a configurable
 *   capacity and returns entries in insertion order.
 * - AC-601-S02-002: DeathContext captures hero pose, enemy state snapshot, and
 *   damage source at death time.
 */

interface ReplayBufferModule {
  createReplayBuffer: (capacity: number) => {
    push: (ctx: unknown) => void;
    entries: () => unknown[];
  };
}

interface DeathContext {
  hero: {
    position: { x: number; y: number };
    angleRad: number;
    health: number;
  };
  enemies: unknown[];
  damageSource: string;
}

function makeDeathContext(seed: number): DeathContext {
  return {
    hero: {
      position: { x: seed, y: seed * 2 },
      angleRad: 0,
      health: 0,
    },
    enemies: [
      {
        kind: 'swarm',
        dna: `seed-${seed}`,
        coordinates: [{ x: 1, y: 1 }],
      },
    ],
    damageSource: 'enemy-swarm',
  };
}

describe('Neatenstein harness replay buffer', () => {
  describe('AC-601-S02-001: capacity and insertion order', () => {
    it('exports createReplayBuffer as a function', async () => {
      const mod = (await import('./replay-buffer.ts')) as ReplayBufferModule;

      expect(typeof mod.createReplayBuffer).toBe('function');
    });

    it('retains at most capacity entries', async () => {
      const { createReplayBuffer } =
        (await import('./replay-buffer.ts')) as ReplayBufferModule;
      const buffer = createReplayBuffer(2);

      buffer.push(makeDeathContext(1));
      buffer.push(makeDeathContext(2));
      buffer.push(makeDeathContext(3));

      expect(buffer.entries().length).toBe(2);
    });

    it('returns entries in insertion order', async () => {
      const { createReplayBuffer } =
        (await import('./replay-buffer.ts')) as ReplayBufferModule;
      const buffer = createReplayBuffer(3);
      const first = makeDeathContext(1);
      const second = makeDeathContext(2);
      const third = makeDeathContext(3);

      buffer.push(first);
      buffer.push(second);
      buffer.push(third);

      expect(buffer.entries()).toEqual([first, second, third]);
    });
  });

  describe('AC-601-S02-002: death context shape', () => {
    it('captures hero pose, enemy snapshot, and damage source', async () => {
      const { createReplayBuffer } =
        (await import('./replay-buffer.ts')) as ReplayBufferModule;
      const buffer = createReplayBuffer(1);
      const ctx = makeDeathContext(42);

      buffer.push(ctx);

      const [entry] = buffer.entries() as DeathContext[];
      expect({
        hero: entry.hero,
        enemies: entry.enemies,
        damageSource: entry.damageSource,
      }).toEqual({
        hero: ctx.hero,
        enemies: ctx.enemies,
        damageSource: ctx.damageSource,
      });
    });
  });
});
