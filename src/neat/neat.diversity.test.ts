/**
 * @module neat.diversity.barrel.test
 * @description Minimal coverage test for the neat.diversity barrel file.
 *
 * The real diversity logic is tested in the dedicated chapter tests; this file
 * only verifies that the barrel re-exports every public symbol expected by
 * downstream consumers.
 */

import {
  buildEmptyDiversityStats,
  computeDiversityStats,
  MAX_COMPATIBILITY_SAMPLE,
  MAX_LINEAGE_PAIR_SAMPLE,
  structuralEntropy,
  type DiversityStats,
} from './neat.diversity';

describe('neat.diversity barrel', () => {
  it('re-exports all named diversity helpers', () => {
    expect(buildEmptyDiversityStats).toBeDefined();
    expect(typeof buildEmptyDiversityStats).toBe('function');

    expect(computeDiversityStats).toBeDefined();
    expect(typeof computeDiversityStats).toBe('function');

    expect(structuralEntropy).toBeDefined();
    expect(typeof structuralEntropy).toBe('function');
  });

  it('re-exports diversity sampling constants', () => {
    expect(MAX_COMPATIBILITY_SAMPLE).toBeDefined();
    expect(typeof MAX_COMPATIBILITY_SAMPLE).toBe('number');
    expect(MAX_COMPATIBILITY_SAMPLE).toBeGreaterThan(0);

    expect(MAX_LINEAGE_PAIR_SAMPLE).toBeDefined();
    expect(typeof MAX_LINEAGE_PAIR_SAMPLE).toBe('number');
    expect(MAX_LINEAGE_PAIR_SAMPLE).toBeGreaterThan(0);
  });

  it('re-exports the DiversityStats type placeholder', () => {
    // Type-only export; this test exists to exercise the import statement line.
    const placeholder: DiversityStats | null = null;
    expect(placeholder).toBeNull();
  });
});
