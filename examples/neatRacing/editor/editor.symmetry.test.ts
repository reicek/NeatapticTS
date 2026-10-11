/**
 * Red-first contract tests for the NeatRacing editor symmetry helper
 * (slice P1-S6-C; spec section 7.2 / AC-021 / LD-28).
 *
 * Mirror symmetry painting must deterministically expand one paint voxel into
 * every mirrored position. The mapping is the same on every axis: a
 * coordinate c is mirrored to (127 - c) on the 128-cubed editor volume
 * (spec section 7.2: cols 0-63 mirror to cols 64-127, col c <-> col 127-c),
 * so each enabled axis contributes the mirror of the paint's own coordinate
 * and combined modes emit the full axis-power set without duplicates.
 *
 * Emission order is part of the deterministic contract: the primary voxel
 * itself first, then the mirrors in ascending subset order
 * (x < y < xy < z < xz < yz < xyz). X symmetry is enabled by default
 * (LD-28); Y and Z are opt-in, and symmetry fully off leaves only the
 * primary voxel. Input coordinates outside 0..127 are rejected (the same
 * RangeError guard editor.state.paint applies).
 */

import { describe, expect, it } from '@jest/globals';
import { createEditorSymmetry } from './editor.symmetry';

describe('editor.symmetry (P1-S6-C; spec AC-021)', () => {
  describe('default symmetry flags', () => {
    it('enables mirror-X by default and leaves Y and Z off', () => {
      const symmetry = createEditorSymmetry();
      expect(symmetry.mirrorX).toBe(true);
      expect(symmetry.mirrorY).toBe(false);
      expect(symmetry.mirrorZ).toBe(false);
    });

    it('accepts explicit overrides for all three axes', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: false,
        mirrorY: true,
        mirrorZ: true,
      });
      expect(symmetry.mirrorX).toBe(false);
      expect(symmetry.mirrorY).toBe(true);
      expect(symmetry.mirrorZ).toBe(true);
    });

    it('accepts an all-flags-off configuration to disable symmetry', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: false,
        mirrorY: false,
        mirrorZ: false,
      });
      expect(symmetry.mirrorX).toBe(false);
      expect(symmetry.mirrorY).toBe(false);
      expect(symmetry.mirrorZ).toBe(false);
    });
  });

  describe('single-axis mirroring uses the 127 - c rule', () => {
    it('expands x-only symmetry across the x axis', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: true,
        mirrorY: false,
        mirrorZ: false,
      });
      expect(symmetry.map(10, 20, 30)).toEqual([
        [10, 20, 30],
        [117, 20, 30],
      ]);
    });

    it('expands y-only symmetry across the y axis', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: false,
        mirrorY: true,
        mirrorZ: false,
      });
      expect(symmetry.map(10, 20, 30)).toEqual([
        [10, 20, 30],
        [10, 107, 30],
      ]);
    });

    it('expands z-only symmetry across the z axis', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: false,
        mirrorY: false,
        mirrorZ: true,
      });
      expect(symmetry.map(10, 20, 30)).toEqual([
        [10, 20, 30],
        [10, 20, 97],
      ]);
    });
  });

  describe('combined symmetry modes compose', () => {
    it('composes x and y mirrors in ascending subset order', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: true,
        mirrorY: true,
        mirrorZ: false,
      });
      expect(symmetry.map(10, 20, 30)).toEqual([
        [10, 20, 30],
        [117, 20, 30],
        [10, 107, 30],
        [117, 107, 30],
      ]);
    });

    it('returns all eight mirrored combinations when every axis is enabled', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: true,
        mirrorY: true,
        mirrorZ: true,
      });
      expect(symmetry.map(1, 2, 3)).toEqual([
        [1, 2, 3],
        [126, 2, 3],
        [1, 125, 3],
        [126, 125, 3],
        [1, 2, 124],
        [126, 2, 124],
        [1, 125, 124],
        [126, 125, 124],
      ]);
    });

    it('returns only the primary voxel when symmetry is fully off', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: false,
        mirrorY: false,
        mirrorZ: false,
      });
      expect(symmetry.map(10, 20, 30)).toEqual([[10, 20, 30]]);
    });
  });

  describe('mirror boundary values', () => {
    it('maps x boundary coordinates 0, 63, and 127 across the mirror', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: true,
        mirrorY: false,
        mirrorZ: false,
      });
      expect(symmetry.map(0, 50, 50)).toEqual([
        [0, 50, 50],
        [127, 50, 50],
      ]);
      expect(symmetry.map(63, 50, 50)).toEqual([
        [63, 50, 50],
        [64, 50, 50],
      ]);
      expect(symmetry.map(127, 50, 50)).toEqual([
        [0, 50, 50],
        [127, 50, 50],
      ]);
    });
  });

  describe('invalid input guard', () => {
    it('rejects coordinates outside 0..127 with a RangeError', () => {
      const symmetry = createEditorSymmetry({
        mirrorX: true,
        mirrorY: false,
        mirrorZ: false,
      });
      expect(() => symmetry.map(128, 50, 50)).toThrow(RangeError);
      expect(() => symmetry.map(10, 20, -1)).toThrow(RangeError);
    });
  });
});
