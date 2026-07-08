/**
 * @module activationArrayPool.barrel.test
 * @description Minimal coverage test for the activation-array-pool barrel file.
 *
 * The real implementation is exercised elsewhere; this test only verifies that
 * the public barrel surface re-exports the expected symbols without breaking the
 * wiring.
 */

import {
  activationArrayPool,
  type ActivationArray,
} from './activationArrayPool';

describe('activationArrayPool barrel', () => {
  it('re-exports the activation array pool object', () => {
    expect(activationArrayPool).toBeDefined();
    expect(typeof activationArrayPool).toBe('object');
  });

  it('re-exports the ActivationArray type placeholder', () => {
    // Type-only export; this test exists to exercise the import statement line.
    const placeholder: ActivationArray | null = null;
    expect(placeholder).toBeNull();
  });
});
