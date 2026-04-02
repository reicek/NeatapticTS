/**
 * Phase 1 – Regression Annotation Assertions
 * Ensures regressionAnnotations (if present) are purely informational and structurally valid.
 */
import fs from 'fs';
import path from 'path';

describe('phase1.regression annotations', () => {
  const resultsPath = path.resolve(__dirname, 'benchmark.results.json');
  if (!fs.existsSync(resultsPath)) {
    test('artifact missing (skip)', () => {
      expect(true).toBe(true);
    });
    return;
  }
  const artifact = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));

  test('regressionAnnotations structure (if present)', () => {
    const anns = artifact.regressionAnnotations;
    if (!anns) return; // nothing to validate
    expect(Array.isArray(anns)).toBe(true);
    for (const a of anns) {
      expect(a.type).toBe('fwd-regression');
      expect(typeof a.size).toBe('number');
      expect(typeof a.deltaPct).toBe('number');
      expect(a.deltaPct).toBeGreaterThan(0);
      // CV fields are optional phase1; present only if baseline + dist CV collected
      if ('srcCvPct' in a) expect(typeof a.srcCvPct).toBe('number');
      if ('distCvPct' in a) expect(typeof a.distCvPct).toBe('number');
    }
  });
});
