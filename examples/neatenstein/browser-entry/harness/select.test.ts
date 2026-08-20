import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/select.ts.
 *
 * Covers AC-301: deterministic variant selection uses an index-stable argmax
 * and breaks ties by the lowest variant id.
 */

interface SelectModule {
  selectVariant: <TVariant>(variants: readonly TVariant[]) => TVariant;
  selectParentProportional: (
    population: Array<{ id: number; fitness?: number }>,
    fitnessRecords: Map<
      number,
      {
        damageDealt: number;
        survivalTicks: number;
        kills: number;
        deaths: number;
        damageTaken: number;
      }
    >,
    mutationSeed: number,
  ) => number;
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

  describe('AC-A4-002b: proportional parent selection for per-death evolution', () => {
    it('exports selectParentProportional as a function', async () => {
      const mod = (await import('./select.ts')) as Record<string, unknown>;
      expect(typeof mod.selectParentProportional).toBe('function');
    });

    it('returns a valid variant index within population bounds', async () => {
      const { selectParentProportional } =
        (await import('./select.ts')) as SelectModule;
      const population = [
        { id: 0, fitness: 10 },
        { id: 1, fitness: 30 },
        { id: 2, fitness: 20 },
        { id: 3, fitness: 5 },
      ];
      const fitnessRecords = new Map([
        [
          0,
          {
            damageDealt: 10,
            survivalTicks: 100,
            kills: 0,
            deaths: 1,
            damageTaken: 5,
          },
        ],
        [
          1,
          {
            damageDealt: 30,
            survivalTicks: 200,
            kills: 1,
            deaths: 1,
            damageTaken: 10,
          },
        ],
        [
          2,
          {
            damageDealt: 20,
            survivalTicks: 150,
            kills: 0,
            deaths: 1,
            damageTaken: 3,
          },
        ],
        [
          3,
          {
            damageDealt: 5,
            survivalTicks: 50,
            kills: 0,
            deaths: 1,
            damageTaken: 1,
          },
        ],
      ]);
      const result = selectParentProportional(population, fitnessRecords, 42);
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(4);
    });

    it('falls back to uniform selection when total fitness is zero', async () => {
      const { selectParentProportional } =
        (await import('./select.ts')) as SelectModule;
      const population = [
        { id: 0, fitness: 0 },
        { id: 1, fitness: 0 },
        { id: 2, fitness: 0 },
        { id: 3, fitness: 0 },
      ];
      const fitnessRecords = new Map<
        number,
        {
          damageDealt: number;
          survivalTicks: number;
          kills: number;
          deaths: number;
          damageTaken: number;
        }
      >([
        [
          0,
          {
            damageDealt: 0,
            survivalTicks: 0,
            kills: 0,
            deaths: 1,
            damageTaken: 0,
          },
        ],
        [
          1,
          {
            damageDealt: 0,
            survivalTicks: 0,
            kills: 0,
            deaths: 1,
            damageTaken: 0,
          },
        ],
        [
          2,
          {
            damageDealt: 0,
            survivalTicks: 0,
            kills: 0,
            deaths: 1,
            damageTaken: 0,
          },
        ],
        [
          3,
          {
            damageDealt: 0,
            survivalTicks: 0,
            kills: 0,
            deaths: 1,
            damageTaken: 0,
          },
        ],
      ]);
      const result = selectParentProportional(population, fitnessRecords, 42);
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(4);
    });

    it('is deterministic for the same mutation seed', async () => {
      const { selectParentProportional } =
        (await import('./select.ts')) as SelectModule;
      const population = [
        { id: 0, fitness: 10 },
        { id: 1, fitness: 30 },
        { id: 2, fitness: 20 },
        { id: 3, fitness: 5 },
      ];
      const fitnessRecords = new Map([
        [
          0,
          {
            damageDealt: 10,
            survivalTicks: 100,
            kills: 0,
            deaths: 1,
            damageTaken: 5,
          },
        ],
        [
          1,
          {
            damageDealt: 30,
            survivalTicks: 200,
            kills: 1,
            deaths: 1,
            damageTaken: 10,
          },
        ],
        [
          2,
          {
            damageDealt: 20,
            survivalTicks: 150,
            kills: 0,
            deaths: 1,
            damageTaken: 3,
          },
        ],
        [
          3,
          {
            damageDealt: 5,
            survivalTicks: 50,
            kills: 0,
            deaths: 1,
            damageTaken: 1,
          },
        ],
      ]);
      const first = selectParentProportional(population, fitnessRecords, 42);
      const second = selectParentProportional(population, fitnessRecords, 42);
      expect(first).toBe(second);
    });

    it('damageTaken reduces the fitness score used for selection', async () => {
      const { selectParentProportional } =
        (await import('./select.ts')) as SelectModule;
      // Two variants with identical positive telemetry except damageTaken.
      // The one with higher damageTaken should have lower fitness, so the
      // fitter (low-damageTaken) variant must be selected more often — verified
      // by clamping: when the penalty exceeds rewards, fitness falls to 0 and
      // uniform fallback kicks in, meaning every variant is selectable.
      const lowDamagePopulation = [
        { id: 0, fitness: 0 },
        { id: 1, fitness: 0 },
      ];
      const lowDamageRecords = new Map([
        [
          0,
          {
            damageDealt: 100,
            survivalTicks: 0,
            kills: 0,
            deaths: 1,
            damageTaken: 0,
          },
        ],
        [
          1,
          {
            damageDealt: 100,
            survivalTicks: 0,
            kills: 0,
            deaths: 1,
            damageTaken: 0,
          },
        ],
      ]);
      // Both have the same fitness (100 * 1.0 - 0 * 0.5 = 100), so selection
      // is proportional and both indices are valid.
      const result = selectParentProportional(
        lowDamagePopulation,
        lowDamageRecords,
        7,
      );
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(2);

      // Now give variant 1 a huge damageTaken penalty so its fitness clamps
      // to 0. Variant 0 keeps positive fitness. With only one positive
      // variant, roulette selection should always pick index 0.
      const penalizedRecords = new Map([
        [
          0,
          {
            damageDealt: 100,
            survivalTicks: 0,
            kills: 0,
            deaths: 1,
            damageTaken: 0,
          },
        ],
        [
          1,
          {
            damageDealt: 100,
            survivalTicks: 0,
            kills: 0,
            deaths: 1,
            damageTaken: 1000,
          },
        ],
      ]);
      for (let seed = 0; seed < 10; seed++) {
        const pick = selectParentProportional(
          lowDamagePopulation,
          penalizedRecords,
          seed,
        );
        expect(pick).toBe(0);
      }
    });
  });
});
