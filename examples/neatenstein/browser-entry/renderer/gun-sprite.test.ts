import { describe, expect, it } from '@jest/globals';
import { projectGunSprite } from './gun-sprite';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/renderer/gun-sprite.ts.
 *
 * Covers AC-11a-003: a dedicated voxel/3D projection helper must exist and return
 * a non-empty screen-space polygon/pixel list for the gun overlay.
 */

describe('Neatenstein gun sprite projection', () => {
  describe('AC-11a: voxel/3D projection helper', () => {
    it('exports projectGunSprite and returns a non-empty projected list', () => {
      expect(typeof projectGunSprite).toBe('function');

      const projected = projectGunSprite({
        voxelGrid: [[1]],
        screenX: 320,
        screenY: 300,
        scale: 10,
      });
      expect(projected.length).toBeGreaterThan(0);
    });

    it('returns an empty list for an empty voxel grid', () => {
      const projected = projectGunSprite({
        voxelGrid: [],
        screenX: 320,
        screenY: 300,
        scale: 10,
      });
      expect(projected).toEqual([]);
    });

    it('projects a stacked voxel column with separate face and top shades', () => {
      const projected = projectGunSprite({
        voxelGrid: [[2]],
        screenX: 320,
        screenY: 300,
        scale: 10,
      });
      expect(projected.length).toBe(2);
      expect(projected[0].color).toBe('#eefcfd');
      expect(projected[1].color).toBe('#ffffff');
    });
  });
});
