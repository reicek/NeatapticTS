/**
 * Contract tests for the Neat Racing LOD distance swap (slice P1-S3-E;
 * AC-012, spec §7.3.4).
 *
 * These tests pin the spec formula arithmetic (voxel world size from
 * carLength/128, focal length from the fixed render height + 60° FOV, swap
 * distance from the 0.5 px projected-error budget — each asserted against
 * the spec's documented values within tolerance), the spec-locked tier
 * names (`lod128` fine / `lod64` coarse — spec wins over tracker wording),
 * the ±10% hysteresis band edges, the inclusive swap boundaries, and the
 * anti-thrash behavior that holds the current tier for oscillating
 * overtaking distances inside the band.
 */
import { describe, expect, it } from '@jest/globals';
import {
  LOD_TIER_COARSE,
  LOD_TIER_FINE,
  NEATRACING_LOD_HYSTERESIS_BAND_FRACTION,
  NEATRACING_LOD_SWAP_DISTANCE_FAR_M,
  NEATRACING_LOD_SWAP_DISTANCE_M,
  NEATRACING_LOD_SWAP_DISTANCE_NEAR_M,
  NEATRACING_LOD_SWAP_FOCAL_PX,
  NEATRACING_VOXEL_WORLD_SIZE,
  selectLodTier,
  type LodTier,
} from './lod.distance.utils';

/** Spec §7.3.4 documented voxel world size (carLength 4.5 m / 128 axes). */
const SPEC_VOXEL_WORLD_SIZE_M = 0.03515625;

/** Spec §7.3.4 documented focal length (≈ 623 px at 720p / 60° FOV). */
const SPEC_FOCAL_PX = 623;

/** Spec §7.3.4 documented swap distance (≈ 44 m for a ~4.5 m car). */
const SPEC_SWAP_DISTANCE_M = 44;

/** Tolerance for asserting the spec-documented focal length (§7.3.4 "≈"). */
const SPEC_FOCAL_TOLERANCE_PX = 1;

/** Tolerance for asserting the spec-documented swap distance (§7.3.4 "≈"). */
const SPEC_SWAP_DISTANCE_TOLERANCE_M = 1;

/** Probe offset in meters used to step just inside or outside a band edge. */
const BAND_PROBE_M = 1;

/**
 * Overtaking oscillation offsets (m) around the base swap distance, all
 * strictly inside the ±10% hysteresis band.
 */
const OSCILLATION_OFFSETS_M = [0, -0.5, 0.5, -0.25, 0.75] as const;

/** Decimal places for the exact binary transcription of carLength/128. */
const VOXEL_WORLD_SIZE_PRECISION = 12;

/** Decimal places for the hysteresis-edge arithmetic assertions. */
const BAND_EDGE_PRECISION = 12;

/**
 * Folds a distance sequence through {@link selectLodTier} from a starting
 * tier, returning the tier after every step so transient toggles are
 * observable.
 *
 * @param startTier - Tier rendered before the sequence begins.
 * @param distancesM - Camera distances in meters, in visit order.
 * @returns One tier per distance, in order.
 */
function foldTierSequence(
  startTier: LodTier,
  distancesM: readonly number[],
): readonly LodTier[] {
  let tier = startTier;
  return distancesM.map((distanceM) => {
    tier = selectLodTier(distanceM, tier);
    return tier;
  });
}

describe('lod.distance.utils (slice P1-S3-E, AC-012)', () => {
  describe('spec §7.3.4 formula constants', () => {
    it('derives NEATRACING_VOXEL_WORLD_SIZE from carLength/128', () => {
      expect(NEATRACING_VOXEL_WORLD_SIZE).toBeCloseTo(
        SPEC_VOXEL_WORLD_SIZE_M,
        VOXEL_WORLD_SIZE_PRECISION,
      );
    });

    it('derives the focal length from the fixed render height and the 60° FOV', () => {
      expect(
        Math.abs(NEATRACING_LOD_SWAP_FOCAL_PX - SPEC_FOCAL_PX),
      ).toBeLessThan(SPEC_FOCAL_TOLERANCE_PX);
    });

    it('derives the swap distance from the projected 0.5 px error budget', () => {
      expect(
        Math.abs(NEATRACING_LOD_SWAP_DISTANCE_M - SPEC_SWAP_DISTANCE_M),
      ).toBeLessThan(SPEC_SWAP_DISTANCE_TOLERANCE_M);
    });

    it('places the hysteresis edges symmetrically around the swap distance', () => {
      expect(NEATRACING_LOD_SWAP_DISTANCE_NEAR_M).toBeCloseTo(
        NEATRACING_LOD_SWAP_DISTANCE_M *
          (1 - NEATRACING_LOD_HYSTERESIS_BAND_FRACTION),
        BAND_EDGE_PRECISION,
      );
      expect(NEATRACING_LOD_SWAP_DISTANCE_FAR_M).toBeCloseTo(
        NEATRACING_LOD_SWAP_DISTANCE_M *
          (1 + NEATRACING_LOD_HYSTERESIS_BAND_FRACTION),
        BAND_EDGE_PRECISION,
      );
    });
  });

  describe('tier naming (spec §7.3.3 wins over tracker wording)', () => {
    it('names the tiers lod128 (fine) and lod64 (coarse)', () => {
      expect(LOD_TIER_FINE).toBe('lod128');
      expect(LOD_TIER_COARSE).toBe('lod64');
    });
  });

  describe('hysteresis tier selection', () => {
    it('downgrades lod128 to lod64 at the far hysteresis edge (inclusive)', () => {
      expect(
        selectLodTier(NEATRACING_LOD_SWAP_DISTANCE_FAR_M, LOD_TIER_FINE),
      ).toBe(LOD_TIER_COARSE);
    });

    it('keeps lod128 just inside the far hysteresis edge', () => {
      expect(
        selectLodTier(
          NEATRACING_LOD_SWAP_DISTANCE_FAR_M - BAND_PROBE_M,
          LOD_TIER_FINE,
        ),
      ).toBe(LOD_TIER_FINE);
    });

    it('upgrades lod64 to lod128 at the near hysteresis edge (inclusive)', () => {
      expect(
        selectLodTier(NEATRACING_LOD_SWAP_DISTANCE_NEAR_M, LOD_TIER_COARSE),
      ).toBe(LOD_TIER_FINE);
    });

    it('keeps lod64 just outside the near hysteresis edge', () => {
      expect(
        selectLodTier(
          NEATRACING_LOD_SWAP_DISTANCE_NEAR_M + BAND_PROBE_M,
          LOD_TIER_COARSE,
        ),
      ).toBe(LOD_TIER_COARSE);
    });

    it('downgrades lod128 well beyond the far hysteresis edge', () => {
      expect(
        selectLodTier(
          NEATRACING_LOD_SWAP_DISTANCE_FAR_M + BAND_PROBE_M,
          LOD_TIER_FINE,
        ),
      ).toBe(LOD_TIER_COARSE);
    });

    it('upgrades lod64 well inside the near hysteresis edge', () => {
      expect(
        selectLodTier(
          NEATRACING_LOD_SWAP_DISTANCE_NEAR_M - BAND_PROBE_M,
          LOD_TIER_COARSE,
        ),
      ).toBe(LOD_TIER_FINE);
    });

    it('holds both tiers at the base swap distance (hysteresis band is no-swap territory)', () => {
      expect(selectLodTier(NEATRACING_LOD_SWAP_DISTANCE_M, LOD_TIER_FINE)).toBe(
        LOD_TIER_FINE,
      );
      expect(
        selectLodTier(NEATRACING_LOD_SWAP_DISTANCE_M, LOD_TIER_COARSE),
      ).toBe(LOD_TIER_COARSE);
    });
  });

  describe('swap-thrash prevention (AC-012)', () => {
    it('holds the fine tier across an overtaking oscillation inside the band', () => {
      const distancesM = OSCILLATION_OFFSETS_M.map(
        (offset) => NEATRACING_LOD_SWAP_DISTANCE_M + offset,
      );
      const tiers = foldTierSequence(LOD_TIER_FINE, distancesM);
      expect(tiers.every((tier) => tier === LOD_TIER_FINE)).toBe(true);
    });

    it('holds the coarse tier across an overtaking oscillation inside the band', () => {
      const distancesM = OSCILLATION_OFFSETS_M.map(
        (offset) => NEATRACING_LOD_SWAP_DISTANCE_M + offset,
      );
      const tiers = foldTierSequence(LOD_TIER_COARSE, distancesM);
      expect(tiers.every((tier) => tier === LOD_TIER_COARSE)).toBe(true);
    });
  });

  describe('totality (deterministic fallback)', () => {
    it('returns the current tier for a non-finite distance', () => {
      expect(selectLodTier(Number.NaN, LOD_TIER_FINE)).toBe(LOD_TIER_FINE);
    });
  });
});
