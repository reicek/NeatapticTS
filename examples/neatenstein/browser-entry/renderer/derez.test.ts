/**
 * @jest-environment node
 */

import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase tests for the Tron-style pixel-by-pixel derez death animation.
 *
 * These tests define the contract for the new `derez.ts` module (AC-11e-003,
 * AC-11e-006, AC-11e-008) and the NEATENSTEIN_ENEMY_DEATH_COLOR constant
 * (AC-11e-005). They are expected to FAIL until the implementation phase
 * (Step 04) creates `derez.ts` and adds the constant to `constants.ts`.
 */

describe('derez module', () => {
  describe('derezHash', () => {
    it('returns the same value for identical inputs (determinism)', async () => {
      const { derezHash } = await import('./derez');
      const a = derezHash(10, 20, 42);
      const b = derezHash(10, 20, 42);
      expect(a).toBe(b);
    });

    it('produces values in the [0, 1) range for the full 48x48 logical grid', async () => {
      const { derezHash } = await import('./derez');
      let min = 1;
      let max = 0;
      for (let y = 0; y < 48; y += 1) {
        for (let x = 0; x < 48; x += 1) {
          const v = derezHash(x, y, 42);
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      expect(min).toBeGreaterThanOrEqual(0);
      expect(max).toBeLessThan(1);
    });

    it('returns different values for different seeds at the same coordinates', async () => {
      const { derezHash } = await import('./derez');
      const v1 = derezHash(10, 20, 1);
      const v2 = derezHash(10, 20, 2);
      expect(v1).not.toBe(v2);
    });
  });

  describe('voxelToLogical', () => {
    it('maps voxel coordinates to logical 0-47 grid via Math.floor(voxel / scale)', async () => {
      const { voxelToLogical } = await import('./derez');
      // ROBOT_SPRITE_SCALE = 4, so voxel 0-3 → logical 0, voxel 4-7 → logical 1, etc.
      expect(voxelToLogical(0)).toBe(0);
      expect(voxelToLogical(3)).toBe(0);
      expect(voxelToLogical(4)).toBe(1);
      expect(voxelToLogical(191)).toBe(47);
    });
  });

  describe('shouldDissolvePixel', () => {
    it('returns false for all pixels when t=0 (start of animation, nothing dissolved)', async () => {
      const { shouldDissolvePixel } = await import('./derez');
      const durationMs = 700;
      const elapsedMs = 0; // t = 0
      let dissolved = 0;
      for (let vy = 0; vy < 192; vy += 1) {
        for (let vx = 0; vx < 192; vx += 1) {
          if (shouldDissolvePixel(vx, vy, 42, elapsedMs, durationMs)) {
            dissolved += 1;
          }
        }
      }
      expect(dissolved).toBe(0);
    });

    it('returns true for all pixels when t=1 (end of animation, fully dissolved)', async () => {
      const { shouldDissolvePixel } = await import('./derez');
      const durationMs = 700;
      const elapsedMs = 700; // t = 1
      let survived = 0;
      for (let vy = 0; vy < 192; vy += 1) {
        for (let vx = 0; vx < 192; vx += 1) {
          if (!shouldDissolvePixel(vx, vy, 42, elapsedMs, durationMs)) {
            survived += 1;
          }
        }
      }
      // At t=1, noise < 1 for all pixels (since hash is [0,1)), so all dissolve.
      expect(survived).toBe(0);
    });

    it('returns mixed results at t=0.5 (some pixels dissolved, some surviving)', async () => {
      const { shouldDissolvePixel } = await import('./derez');
      const durationMs = 700;
      const elapsedMs = 350; // t = 0.5
      let dissolved = 0;
      let survived = 0;
      for (let vy = 0; vy < 192; vy += 1) {
        for (let vx = 0; vx < 192; vx += 1) {
          if (shouldDissolvePixel(vx, vy, 42, elapsedMs, durationMs)) {
            dissolved += 1;
          } else {
            survived += 1;
          }
        }
      }
      // At t=0.5, roughly half the pixels should dissolve and half survive.
      // We don't require an exact 50/50 split, but both counts must be > 0.
      expect(dissolved).toBeGreaterThan(0);
      expect(survived).toBeGreaterThan(0);
    });
  });

  describe('seed stability across calls', () => {
    it('produces identical dissolution results for the same seed across repeated calls', async () => {
      const { shouldDissolvePixel } = await import('./derez');
      const durationMs = 700;
      const elapsedMs = 350;
      const seed = 42;

      const results1: boolean[] = [];
      const results2: boolean[] = [];
      for (let vy = 0; vy < 192; vy += 1) {
        for (let vx = 0; vx < 192; vx += 1) {
          results1.push(
            shouldDissolvePixel(vx, vy, seed, elapsedMs, durationMs),
          );
          results2.push(
            shouldDissolvePixel(vx, vy, seed, elapsedMs, durationMs),
          );
        }
      }
      expect(results1).toEqual(results2);
    });

    it('produces different dissolution patterns for different seeds', async () => {
      const { shouldDissolvePixel } = await import('./derez');
      const durationMs = 700;
      const elapsedMs = 350;

      const pattern1: boolean[] = [];
      const pattern2: boolean[] = [];
      for (let vy = 0; vy < 192; vy += 1) {
        for (let vx = 0; vx < 192; vx += 1) {
          pattern1.push(shouldDissolvePixel(vx, vy, 1, elapsedMs, durationMs));
          pattern2.push(shouldDissolvePixel(vx, vy, 2, elapsedMs, durationMs));
        }
      }
      // Different seeds must produce different dissolution patterns.
      let differences = 0;
      for (let i = 0; i < pattern1.length; i += 1) {
        if (pattern1[i] !== pattern2[i]) {
          differences += 1;
        }
      }
      expect(differences).toBeGreaterThan(0);
    });
  });

  describe('scattered dissolution pattern', () => {
    it('dissolution is spatially scattered, not row-by-row', async () => {
      const { shouldDissolvePixel } = await import('./derez');
      const durationMs = 700;
      const elapsedMs = 175; // t = 0.25 — early in the animation

      // Sample a small 16x16 region in logical space (voxels 0-63 × 0-63).
      // If dissolution were row-by-row, entire rows would vanish together.
      // A scattered pattern has a mix of dissolved and surviving pixels
      // within the same row and the same column.
      let rowsWithMixedResults = 0;
      let colsWithMixedResults = 0;

      for (let vy = 0; vy < 64; vy += 1) {
        let hasDissolved = false;
        let hasSurvived = false;
        for (let vx = 0; vx < 64; vx += 1) {
          if (shouldDissolvePixel(vx, vy, 42, elapsedMs, durationMs)) {
            hasDissolved = true;
          } else {
            hasSurvived = true;
          }
        }
        if (hasDissolved && hasSurvived) {
          rowsWithMixedResults += 1;
        }
      }

      for (let vx = 0; vx < 64; vx += 1) {
        let hasDissolved = false;
        let hasSurvived = false;
        for (let vy = 0; vy < 64; vy += 1) {
          if (shouldDissolvePixel(vx, vy, 42, elapsedMs, durationMs)) {
            hasDissolved = true;
          } else {
            hasSurvived = true;
          }
        }
        if (hasDissolved && hasSurvived) {
          colsWithMixedResults += 1;
        }
      }

      // A scattered pattern has many rows and columns with mixed results.
      // A row-by-row pattern would have zero mixed rows.
      expect(rowsWithMixedResults).toBeGreaterThan(0);
      expect(colsWithMixedResults).toBeGreaterThan(0);
    });
  });
});

describe('NEATENSTEIN_ENEMY_DEATH_COLOR constant', () => {
  it('is defined as [180, 190, 210] in browser-entry/constants.ts', async () => {
    const constants = await import('../constants');
    expect(constants.NEATENSTEIN_ENEMY_DEATH_COLOR).toEqual([180, 190, 210]);
  });
});

describe('DE_REZ duration constants', () => {
  it('ENEMY_CONTROLLER_DE_REZ_DURATION_MS is 700 in enemy-controller.ts', async () => {
    const { ENEMY_CONTROLLER_DE_REZ_DURATION_MS } =
      await import('../../../neatenstein/scripts/enemy-controller');
    expect(ENEMY_CONTROLLER_DE_REZ_DURATION_MS).toBe(700);
  });

  it('ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS is 700 in enemy-sprite.ts (parity)', async () => {
    const { ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS } =
      await import('../../../neatenstein/scripts/enemy-sprite');
    expect(ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS).toBe(700);
  });
});
