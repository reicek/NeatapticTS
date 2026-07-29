import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/select.ts.
 *
 * Covers AC-301: deterministic variant selection uses an index-stable argmax
 * and breaks ties by the lowest variant id.
 */

interface SelectModule {
  selectVariant: <TVariant>(variants: readonly TVariant[]) => TVariant;
}

describe('Neatenstein harness select', () => {
  describe('AC-301: deterministic variant selection', () => {
    it('exports selectVariant as a function', async () => {
      const mod = (await import('./select.ts')) as Record<string, unknown>;
      expect(typeof mod.selectVariant).toBe('function');
    });

    it('selects the variant with the highest fitness', async () => {
      const { selectVariant } = (await import('./select.ts')) as SelectModule;
      const variants = [
        { id: 0, fitness: 10 },
        { id: 1, fitness: 30 },
        { id: 2, fitness: 20 },
      ];
      expect(selectVariant(variants)).toBe(variants[1]);
    });

    it('tie-breaks by the lowest variant id', async () => {
      const { selectVariant } = (await import('./select.ts')) as SelectModule;
      const variants = [
        { id: 2, fitness: 50 },
        { id: 0, fitness: 50 },
        { id: 1, fitness: 50 },
      ];
      expect(selectVariant(variants)).toBe(variants[1]);
    });

    it('is stable for identical input order', async () => {
      const { selectVariant } = (await import('./select.ts')) as SelectModule;
      const variants = [
        { id: 0, fitness: 5 },
        { id: 1, fitness: 5 },
      ];
      const first = selectVariant(variants);
      const second = selectVariant(variants);
      expect(first.id).toBe(second.id);
    });

    it('throws when given an empty population', async () => {
      const { selectVariant } = (await import('./select.ts')) as SelectModule;
      expect(() => selectVariant([])).toThrow(
        'Cannot select a variant from an empty population.',
      );
    });

    it('treats missing fitness as negative infinity', async () => {
      const { selectVariant } = (await import('./select.ts')) as SelectModule;
      const unscored = { id: 0 };
      const scored = { id: 1, fitness: 10 };
      expect(selectVariant([unscored, scored])).toBe(scored);
    });

    it('ignores later variants without a fitness score', async () => {
      const { selectVariant } = (await import('./select.ts')) as SelectModule;
      const scored = { id: 0, fitness: 10 };
      const unscored = { id: 1 };
      expect(selectVariant([scored, unscored])).toBe(scored);
    });
  });
});
