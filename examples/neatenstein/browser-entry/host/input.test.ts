import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/input.ts.
 *
 * Covers AC-206: the host input layer exposes a router that can be attached to
 * DOM events, detached, and queried for a snapshot consumed by controls.ts and
 * forwarded to the worker tier.
 */

describe('Neatenstein host input', () => {
  describe('AC-206: input router contract', () => {
    it('exports createInputRouter', async () => {
      const mod = (await import('./input.ts')) as Record<string, unknown>;
      expect(typeof mod.createInputRouter).toBe('function');
    });

    it('returns a router with attach, detach, and getSnapshot methods', async () => {
      const { createInputRouter } = (await import('./input.ts')) as Record<
        string,
        any
      >;
      const router = createInputRouter();
      expect({
        hasAttach: typeof router.attach === 'function',
        hasDetach: typeof router.detach === 'function',
        hasGetSnapshot: typeof router.getSnapshot === 'function',
      }).toEqual({
        hasAttach: true,
        hasDetach: true,
        hasGetSnapshot: true,
      });
    });
  });
});
