/**
 * benchmark.buildVariants.ts
 *
 * Placeholder planning utilities for build variant (src vs dist) benchmarks.
 */

/**
 * Planned run descriptor representing one element of the cartesian product across
 * build modes, scenarios and synthetic size points.
 */
export interface PlannedVariantRun {
  mode: 'src' | 'dist';
  scenario: string;
  size: number;
}

/**
 * Generate exhaustive cartesian product of provided axes (modes × scenarios × sizes).
 * Educational clarity > micro‑optimisation: explicit nested loops show intent to learners.
 *
 * Complexity: O(M * S * N) where M=#modes, S=#scenarios, N=#sizes.
 * No deduplication is performed: duplicate values in any axis produce duplicate rows (by design),
 * which allows upstream callers to intentionally weight certain modes/scenarios.
 */
export function planVariantRuns(
  modes: Array<'src' | 'dist'> = ['src', 'dist'],
  scenarios: string[] = ['build'],
  sizes: number[] = [1000]
): PlannedVariantRun[] {
  const plannedRuns: PlannedVariantRun[] = [];
  // Iterate axes in fixed order to guarantee stable output ordering for reproducibility.
  for (const variantMode of modes) {
    // outer: modes
    for (const scenarioKey of scenarios) {
      // middle: scenarios
      for (const sizePoint of sizes) {
        // inner: sizes
        // Push cartesian element (Act)
        plannedRuns.push({
          mode: variantMode,
          scenario: scenarioKey,
          size: sizePoint,
        });
      }
    }
  }
  return plannedRuns;
}

// Tests ------------------------------------------------------------------------------------------

describe('benchmark.buildVariants placeholder', () => {
  describe('planVariantRuns()', () => {
    // Default arrangement reused across tests
    const plannedDefault = planVariantRuns();
    it('returns an array of runs', () => {
      expect(Array.isArray(plannedDefault)).toBe(true);
    });
    it('produces runs covering both modes', () => {
      const modes = new Set(plannedDefault.map((r) => r.mode));
      expect(modes.has('src') && modes.has('dist')).toBe(true);
    });
    const custom = planVariantRuns(['src'], ['build', 'forward'], [1, 2]);
    it('computes expected cartesian size for custom axes', () => {
      const expected = 1 * 2 * 2; // modes × scenarios × sizes
      expect(custom.length).toBe(expected);
    });
    describe('edge cases', () => {
      it('returns empty array when modes empty', () => {
        const out = planVariantRuns([], ['build'], [1]);
        expect(out.length).toBe(0);
      });
      it('returns empty array when scenarios empty', () => {
        const out = planVariantRuns(['src'], [], [1]);
        expect(out.length).toBe(0);
      });
      it('returns empty array when sizes empty', () => {
        const out = planVariantRuns(['src'], ['build'], []);
        expect(out.length).toBe(0);
      });
    });
    describe('ordering & duplicates', () => {
      const sizes = [5, 3, 9];
      const ordered = planVariantRuns(['src'], ['build'], sizes);
      const duplicateModes = planVariantRuns(['src', 'src'], ['build'], [1]);
      it('preserves size iteration order (no internal sort)', () => {
        expect(ordered.map((r) => r.size).join(',')).toBe(sizes.join(','));
      });
      it('retains duplicate modes (no deduplication)', () => {
        const srcCount = duplicateModes.filter((r) => r.mode === 'src').length;
        expect(srcCount).toBe(2);
      });
    });
  });
});
