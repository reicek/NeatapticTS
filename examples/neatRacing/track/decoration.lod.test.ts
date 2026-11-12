/**
 * Red contract tests for the neatRacing right-world decoration LOD system
 * (slice P1-decoration-scale, Step 14D, AC-051).
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - {@link NEATRACING_DECORATION_LOD_BUDGET} caps per-frame decorations.
 * - {@link computeDecorationScale} returns a larger positive scale near the
 *   camera and a smaller positive scale far away, floored at the min scale.
 * - {@link computeDecorationLod} reports visibility, scale, and render mode.
 * - {@link resolveDecorationRenderMode} prefers '2d' in P1 and only selects
 *   'voxel' when an explicit fallback threshold is supplied and exceeded.
 * - {@link selectDecorationsForCamera} culls distant decorations and keeps
 *   the nearest ones up to the budget.
 */

import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_DECORATION_CULL_DISTANCE_M,
  NEATRACING_DECORATION_LOD_BUDGET,
  NEATRACING_DECORATION_MIN_SCALE,
  NEATRACING_DECORATION_NEAR_REFERENCE_M,
  NEATRACING_DECORATION_P1_DECISION,
  computeDecorationLod,
  computeDecorationScale,
  resolveDecorationRenderMode,
  selectDecorationsForCamera,
} from './decoration.lod';

/** Camera at the origin for deterministic distance tests. */
const FIXTURE_CAMERA = [0, 0, 0] as const;

/** Ten evenly spaced decorations receding down +z from the camera. */
const FIXTURE_CANDIDATES = Array.from({ length: 10 }, (_, index) => ({
  position: [0, 0, index * 10] as const,
}));

/** Explicit decision enabling the voxel fallback at 100 meters. */
const VOXEL_FALLBACK_DECISION = {
  preferredMode: '2d' as const,
  voxelFallbackThresholdM: 100,
  rationale: 'test threshold enabling voxel fallback branch',
};

describe('decoration.lod (P1-decoration-scale, AC-051)', () => {
  describe('budget cap', () => {
    it('exports a positive per-frame LOD budget', () => {
      expect(NEATRACING_DECORATION_LOD_BUDGET).toBeGreaterThan(0);
    });

    it('selects no more decorations than the default LOD budget', () => {
      const selected = selectDecorationsForCamera(
        FIXTURE_CANDIDATES,
        FIXTURE_CAMERA,
      );

      expect(selected.length).toBeLessThanOrEqual(
        NEATRACING_DECORATION_LOD_BUDGET,
      );
    });

    it('honors an explicit smaller budget', () => {
      const budget = 4;
      const selected = selectDecorationsForCamera(
        FIXTURE_CANDIDATES,
        FIXTURE_CAMERA,
        budget,
      );

      expect(selected.length).toBeLessThanOrEqual(budget);
    });
  });

  describe('distance scaling', () => {
    it('returns a larger scale near the camera than far away', () => {
      const near = computeDecorationScale(
        NEATRACING_DECORATION_NEAR_REFERENCE_M,
      );
      const far = computeDecorationScale(NEATRACING_DECORATION_CULL_DISTANCE_M);

      expect(near).toBeGreaterThan(far);
    });

    it('floors the scale at the minimum value for extreme distances', () => {
      const scale = computeDecorationScale(
        NEATRACING_DECORATION_CULL_DISTANCE_M * 2,
      );

      expect(scale).toBeGreaterThanOrEqual(NEATRACING_DECORATION_MIN_SCALE);
    });

    it('uses the near reference as a unit scale anchor', () => {
      const scale = computeDecorationScale(
        NEATRACING_DECORATION_NEAR_REFERENCE_M,
      );

      expect(scale).toBeCloseTo(1, 6);
    });
  });

  describe('visibility', () => {
    it('marks decorations within the cull distance as visible', () => {
      const lod = computeDecorationLod(NEATRACING_DECORATION_NEAR_REFERENCE_M);

      expect(lod.visible).toBe(true);
    });

    it('marks decorations beyond the cull distance as not visible', () => {
      const lod = computeDecorationLod(
        NEATRACING_DECORATION_CULL_DISTANCE_M + 1,
      );

      expect(lod.visible).toBe(false);
    });

    it('preserves distance and exposes a positive scale in the LOD record', () => {
      const distance = 55;
      const lod = computeDecorationLod(distance);

      expect(lod.distance).toBe(distance);
      expect(lod.scale).toBeGreaterThan(0);
    });
  });

  describe('2D-preferred decision', () => {
    it('resolves to 2D for the P1 decision at all visible distances', () => {
      const nearMode = resolveDecorationRenderMode(
        NEATRACING_DECORATION_NEAR_REFERENCE_M,
        NEATRACING_DECORATION_P1_DECISION,
      );
      const farMode = resolveDecorationRenderMode(
        NEATRACING_DECORATION_CULL_DISTANCE_M,
        NEATRACING_DECORATION_P1_DECISION,
      );

      expect(nearMode).toBe('2d');
      expect(farMode).toBe('2d');
    });

    it('documents the P1 decision as 2D-preferred with disabled voxel fallback', () => {
      expect(NEATRACING_DECORATION_P1_DECISION.preferredMode).toBe('2d');
      expect(
        NEATRACING_DECORATION_P1_DECISION.voxelFallbackThresholdM,
      ).toBeNull();
    });
  });

  describe('voxel fallback branch', () => {
    it('stays in 2D when distance is below the explicit voxel threshold', () => {
      const mode = resolveDecorationRenderMode(50, VOXEL_FALLBACK_DECISION);

      expect(mode).toBe('2d');
    });

    it('switches to voxel when distance exceeds the explicit voxel threshold', () => {
      const mode = resolveDecorationRenderMode(150, VOXEL_FALLBACK_DECISION);

      expect(mode).toBe('voxel');
    });

    it('reflects the explicit voxel mode in the full LOD record', () => {
      const lod = computeDecorationLod(150, VOXEL_FALLBACK_DECISION);

      expect(lod.mode).toBe('voxel');
    });
  });
});
