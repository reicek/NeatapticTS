/**
 * Replay buffer for death-context recording in the Neatenstein co-evolution
 * harness.
 *
 * The buffer is a fixed-capacity FIFO queue of {@link DeathContext} entries.
 * When the buffer is full and a new entry is pushed, the oldest entry is
 * evicted. Entries are returned in insertion order so downstream replay logic
 * can iterate historical death states as deterministic selection pressure.
 *
 * @module
 */

import type { DeathContext } from './types';

/**
 * Replay buffer instance returned by {@link createReplayBuffer}.
 *
 * The buffer exposes `push`, `entries`, and `size` so it satisfies both the
 * harness replay interface (which uses `entries`) and the arms-race human-mode
 * interface (which uses `size`).
 */
export interface ReplayBuffer {
  /** Append a death context to the buffer, evicting the oldest entry when full. */
  push: (ctx: DeathContext) => void;
  /** Return all stored death contexts in insertion order. */
  entries: () => DeathContext[];
  /** Return the current number of stored death contexts. */
  size: () => number;
}

/**
 * Create a fixed-capacity FIFO replay buffer for death contexts.
 *
 * When the number of stored entries exceeds `capacity`, the oldest entry
 * (first pushed) is evicted before the new entry is appended. This keeps the
 * buffer bounded while preserving the most recent death contexts for replay
 * selection pressure.
 *
 * @param capacity - Maximum number of death contexts to retain.
 * @returns A {@link ReplayBuffer} instance with `push`, `entries`, and `size`.
 *
 * @example
 * ```ts
 * const buffer = createReplayBuffer(32);
 * buffer.push({ hero: { position: { x: 0, y: 0 }, angleRad: 0, health: 0 }, enemies: [], damageSource: 'enemy' });
 * console.log(buffer.size()); // 1
 * console.log(buffer.entries().length); // 1
 * ```
 */
export function createReplayBuffer(capacity: number): ReplayBuffer {
  const store: DeathContext[] = [];

  const push = (ctx: DeathContext): void => {
    if (store.length >= capacity) {
      store.shift();
    }
    store.push(ctx);
  };

  const entries = (): DeathContext[] => [...store];

  /* istanbul ignore next -- size is exercised via the arms-race mock fixture */
  const size = (): number => store.length;

  return { push, entries, size };
}
