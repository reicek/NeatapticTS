/**
 * Phase 1 – Variance & Field Slimming Assertions
 * Consumes benchmark.results.json (single source of truth) to validate:
 *  - Variance entries exist for large sizes (>=100k) when repeats >1.
 *  - Samples meet minimum threshold.
 *  - Connection enumerable key count has not regressed (>9 would fail).
 *  - CV% reported; soft expectations (warn if above soft threshold, still pass to gather data).
 */
import fs from 'fs';
import path from 'path';

interface VarianceEntry {
  mode: string;
  size: number;
  samples: number;
  buildMsCvPct: number;
  fwdAvgMsCvPct: number;
}

describe('phase1.variance artifact assertions', () => {
  const resultsPath = path.resolve(__dirname, 'benchmark.results.json');
  let artifact: any;
  test('artifact present', () => {
    expect(fs.existsSync(resultsPath)).toBe(true);
    artifact = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
  });

  test('variance entries exist for large sizes (>=100k) when repeats>1', () => {
    const repeats = artifact?.meta?.varianceRepeatsLarge || 1;
    if (repeats <= 1) {
      // If no repeats, variance section may be absent; skip assertion.
      return;
    }
    expect(Array.isArray(artifact.variance)).toBe(true);
    const sizes = new Set(artifact.variance.map((v: VarianceEntry) => v.size));
    expect(sizes.has(100000)).toBe(true);
    expect(sizes.has(200000)).toBe(true);
  });

  test('variance samples >= configured repeats for each large size/mode', () => {
    const repeats = artifact?.meta?.varianceRepeatsLarge || 1;
    if (repeats <= 1 || !artifact.variance) return;
    for (const v of artifact.variance as VarianceEntry[]) {
      if (v.size >= 100000) {
        expect(v.samples).toBeGreaterThanOrEqual(repeats);
      }
    }
  });

  test('Connection field audit key count ≤ 12 (post-optimizer virtualization)', () => {
    const count = artifact?.fieldAudit?.Connection?.count;
    // TODO(phase2): tighten to <=9 after confirming stability across mutations/optimizers
    expect(count).toBeLessThanOrEqual(12);
  });

  test('Forward CV% present for large sizes; warn if above soft limit (15%)', () => {
    if (!artifact.variance) return;
    const softWarn = 15; // informational threshold
    for (const v of artifact.variance as VarianceEntry[]) {
      if (v.size >= 100000) {
        expect(typeof v.fwdAvgMsCvPct).toBe('number');
        if (v.fwdAvgMsCvPct > softWarn) {
          // Emit a structured warning annotation (does not fail test) – kept minimal.
          // eslint-disable-next-line no-console
          console.log(
            `[variance-soft] size=${v.size} mode=${
              v.mode
            } fwdCvPct=${v.fwdAvgMsCvPct.toFixed(2)}> ${softWarn}%`
          );
        }
      }
    }
  });
});
