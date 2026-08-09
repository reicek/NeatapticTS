/**
 * This test file previously contained tests for the voxel gun descriptor
 * (`buildVoxelGun`) and projector (`projectVoxelGunSprite`). Those modules
 * were removed in slice 04c-impl-renderer and replaced by the palette-indexed
 * 2D sprite decoder (`gun-sprite-decode.ts`). The tests are removed as part
 * of the No Deferred Cleanup Policy — no dead tests for removed code.
 *
 * The palette-indexed sprite contract is now tested in `gun.test.ts` and
 * `gun-sprite-data.test.ts`.
 */

import { describe, expect, it } from '@jest/globals';

describe('gun-voxel (removed in 04c-impl-renderer)', () => {
  it('preserves a passing placeholder so Jest does not fail on an empty suite', () => {
    expect(true).toBe(true);
  });
});
