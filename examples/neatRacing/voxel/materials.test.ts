/**
 * Contract tests for the Neat Racing material semantics (slice P1-S4-A;
 * AC-013 part 1, spec sections 7.2 / LD-23).
 *
 * These tests pin the material classification table (empty sentinel,
 * solid body, shining glass) and the AC-013 black-exception emission
 * rule: black cells (r === g === b === 0) yield zero emissive through
 * *every* material lookup — per-entry resolution and index-aligned
 * table lookup alike — regardless of glow magnitude or alpha, while
 * transparent colored entries stay glass and opaque colored entries
 * carry their glow verbatim.
 */
import { describe, expect, it } from '@jest/globals';
import {
  buildVoxelMaterialTable,
  resolveVoxelMaterial,
  VOXEL_MATERIAL_BLACK_EMISSIVE,
  VOXEL_MATERIAL_GLASS_ALPHA_MAX,
} from './materials';
import {
  VOXEL_EMPTY_PALETTE_INDEX,
  type VoxelPaletteEntry,
} from './model.schema';

/** Opaque team-red body voxel carrying no glow byte. */
const SOLID_RED_NO_GLOW: VoxelPaletteEntry = [200, 20, 20, 255, 0];

/** Opaque team-red body voxel carrying bloom glow (spec section 6.6). */
const SOLID_RED_GLOWING: VoxelPaletteEntry = [200, 20, 20, 255, 200];

/** Light-blue shining-glass tint (spec section 7.2 "typically light-blue"). */
const GLASS_BLUE_GLOWING: VoxelPaletteEntry = [190, 225, 255, 0, 160];

/** Transparent colored entry with a zero glow byte (no shine). */
const GLASS_DARK_NO_GLOW: VoxelPaletteEntry = [10, 10, 40, 0, 0];

/** Black solid body voxel with a large glow byte (LD-23 suppression probe). */
const BLACK_SOLID_GLOWING: VoxelPaletteEntry = [0, 0, 0, 255, 200];

/** Transparent black entry with a large glow byte (no color, no shine). */
const BLACK_TRANSPARENT_GLOWING: VoxelPaletteEntry = [0, 0, 0, 0, 200];

/** Glow magnitudes probed against the LD-23 black suppression. */
const BLACK_GLOW_PROBES = [1, 128, 255] as const;

/** Example palette covering every material class in one lookup. */
const EXAMPLE_PALETTE: readonly VoxelPaletteEntry[] = [
  [0, 0, 0, 0, 0],
  SOLID_RED_GLOWING,
  BLACK_SOLID_GLOWING,
  GLASS_BLUE_GLOWING,
  GLASS_DARK_NO_GLOW,
  BLACK_TRANSPARENT_GLOWING,
];

describe('materials (slice P1-S4-A, AC-013 part 1: spec 7.2 / LD-23)', () => {
  describe('resolveVoxelMaterial classification', () => {
    it('classifies the exact EMPTY sentinel entry as empty air', () => {
      const material = resolveVoxelMaterial([0, 0, 0, 0, 0]);
      expect(material.kind).toBe('empty');
      expect(material.emissive).toBe(VOXEL_MATERIAL_BLACK_EMISSIVE);
    });

    it('classifies opaque colored entries as solid body voxels', () => {
      const material = resolveVoxelMaterial(SOLID_RED_NO_GLOW);
      expect(material.kind).toBe('solid');
    });

    it('carries the glow byte verbatim as emissive on solid entries', () => {
      const material = resolveVoxelMaterial(SOLID_RED_GLOWING);
      expect(material.emissive).toBe(200);
    });

    it('classifies fully transparent colored entries as shining glass', () => {
      const material = resolveVoxelMaterial(GLASS_BLUE_GLOWING);
      expect(material.kind).toBe('glass');
      expect(material.emissive).toBe(160);
    });

    it('keeps glow-less transparent entries in the glass class with zero shine', () => {
      const material = resolveVoxelMaterial(GLASS_DARK_NO_GLOW);
      expect(material.kind).toBe('glass');
      expect(material.emissive).toBe(0);
    });

    it('classifies black entries as their own class distinct from glass', () => {
      const material = resolveVoxelMaterial(BLACK_SOLID_GLOWING);
      expect(material.kind).toBe('black');
    });

    it('classifies transparent black entries as black, not glass', () => {
      const material = resolveVoxelMaterial(BLACK_TRANSPARENT_GLOWING);
      expect(material.kind).toBe('black');
    });

    it('copies solid entry color and alpha bytes verbatim', () => {
      const solid = resolveVoxelMaterial(SOLID_RED_NO_GLOW);
      expect(solid.color).toEqual([200, 20, 20]);
      expect(solid.alpha).toBe(255);
    });

    it('copies glass entry color and alpha bytes verbatim', () => {
      const glass = resolveVoxelMaterial(GLASS_BLUE_GLOWING);
      expect(glass.color).toEqual([190, 225, 255]);
      expect(glass.alpha).toBe(0);
    });
  });

  describe('LD-23 black-exception emission rule (AC-013: black never emits shine)', () => {
    it('forces zero emissive on black entries with a large glow byte', () => {
      const material = resolveVoxelMaterial(BLACK_SOLID_GLOWING);
      expect(material.emissive).toBe(VOXEL_MATERIAL_BLACK_EMISSIVE);
    });

    it('suppresses shine at every glow magnitude for black cells', () => {
      for (const glow of BLACK_GLOW_PROBES) {
        const material = resolveVoxelMaterial([0, 0, 0, 255, glow]);
        expect(material.emissive).toBe(0);
      }
    });

    it('emits zero from transparent black entries carrying bloom glow', () => {
      const material = resolveVoxelMaterial(BLACK_TRANSPARENT_GLOWING);
      expect(material.emissive).toBe(VOXEL_MATERIAL_BLACK_EMISSIVE);
    });

    it('keeps the black suppression intact through table lookup', () => {
      const table = buildVoxelMaterialTable(EXAMPLE_PALETTE);
      const blackMaterial = table[EXAMPLE_PALETTE.indexOf(BLACK_SOLID_GLOWING)];
      expect(blackMaterial.emissive).toBe(VOXEL_MATERIAL_BLACK_EMISSIVE);
    });
  });

  describe('buildVoxelMaterialTable (index-aligned palette lookup)', () => {
    it('resolves one material per palette entry in palette order', () => {
      const table = buildVoxelMaterialTable(EXAMPLE_PALETTE);
      expect(table.length).toBe(EXAMPLE_PALETTE.length);
      for (let index = 0; index < EXAMPLE_PALETTE.length; index++) {
        expect(table[index]).toEqual(
          resolveVoxelMaterial(EXAMPLE_PALETTE[index]),
        );
      }
    });

    it('resolves the EMPTY palette index 0 as empty air in the table', () => {
      const table = buildVoxelMaterialTable(EXAMPLE_PALETTE);
      expect(table[VOXEL_EMPTY_PALETTE_INDEX].kind).toBe('empty');
    });

    it('keeps the glass classification intact through table lookup', () => {
      const table = buildVoxelMaterialTable(EXAMPLE_PALETTE);
      const glassMaterial = table[EXAMPLE_PALETTE.indexOf(GLASS_BLUE_GLOWING)];
      expect(glassMaterial.kind).toBe('glass');
      expect(glassMaterial.emissive).toBe(160);
    });

    it('defines glass as every non-black entry at the glass alpha bound', () => {
      expect(VOXEL_MATERIAL_GLASS_ALPHA_MAX).toBe(0);
    });
  });
});
