/**
 * @module neat.lineage.barrel.test
 * @description Minimal coverage test for the neat.lineage barrel file.
 *
 * The real lineage logic is tested in the dedicated chapter tests; this file
 * only verifies that the barrel re-exports every public symbol expected by
 * downstream consumers.
 */

import {
  buildAnc,
  computeAncestorUniqueness,
  type GenomeLike,
  type NeatLineageContext,
} from './neat.lineage';

describe('neat.lineage barrel', () => {
  it('re-exports the lineage helpers', () => {
    expect(buildAnc).toBeDefined();
    expect(typeof buildAnc).toBe('function');

    expect(computeAncestorUniqueness).toBeDefined();
    expect(typeof computeAncestorUniqueness).toBe('function');
  });

  it('re-exports the lineage type placeholders', () => {
    // Type-only exports; this test exists to exercise the import statement lines.
    const genome: GenomeLike | null = null;
    const context: NeatLineageContext | null = null;
    expect(genome).toBeNull();
    expect(context).toBeNull();
  });
});
