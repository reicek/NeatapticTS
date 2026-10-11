/**
 * Materials semantics for the Neat Racing voxel racer (slice P1-S4-A,
 * AC-013 part 1; spec sections 7.2 / LD-23 in
 * `plans/neatracing-voxel-racer.spec.md`).
 *
 * This module is the single shared owner of the palette-entry material
 * contract ("glass/glow/black semantics" per the layout comment in the
 * spec): it classifies every on-disk palette entry into one render class
 * and applies the LD-23 black exception, so mesher body shading, the LOD
 * derive, and the dedicated glass pass can share one lookup instead of
 * re-deriving the rule per consumer.
 *
 * The spec rules encoded here (section 7.2, normative per LD-26):
 *
 * 1. Palette entries are `[r, g, b, a, glow]` bytes; palette index 0 is
 *    reserved as the EMPTY sentinel (air; never rendered).
 * 2. `glow` 0-255 drives the bloom/shine pass (section 6.6).
 * 3. A fully transparent entry (`a === 0`) with a color renders as
 *    shining glass of that color (glow only, no body quads).
 * 4. Black (`r === g === b === 0`) is the exception: no color and no
 *    shine even if `glow` is set (LD-23 / AC-013). Black stays solid
 *    occluding body geometry at opaque alphas, and black cells yield
 *    zero emissive in *every* material lookup.
 *
 * Inputs are trusted palette entries as validated by `model.schema.ts`
 * the same way the mesher and LOD builder trust decoded models; this
 * module adds no I/O and performs one pure pass per entry.
 *
 * @module
 */

import {
  VOXEL_EMPTY_PALETTE_ENTRY,
  type VoxelPaletteEntry,
} from './model.schema';

/**
 * Emissive contribution forced onto black palette entries (LD-23:
 * black never emits shine), matching the mesher's
 * `VOXEL_BLACK_EXCEPTION_EMISSIVE` body-pass rule and the LOD
 * builder's derived-black glow rule (spec section 7.3.3 rule 7).
 */
export const VOXEL_MATERIAL_BLACK_EMISSIVE = 0;

/**
 * Fully transparent alpha at or below which a non-black palette entry
 * is glass (spec section 7.2: "a fully transparent entry (`a === 0`)"
 * renders as shining glass of its color, glow only, no body quads).
 */
export const VOXEL_MATERIAL_GLASS_ALPHA_MAX = 0;

/**
 * Render class of one palette entry after section 7.2 / LD-23
 * material semantics:
 *
 * - `empty`: the exact EMPTY sentinel entry (air; never rendered).
 * - `black`: `r === g === b === 0`; solid occluding body geometry at
 *   opaque alphas with emissive forced to 0 regardless of the glow
 *   byte; never glass and never shining even when transparent.
 * - `glass`: fully transparent colored entry rendered by the glow-only
 *   glass pass (no body quads).
 * - `solid`: opaque (`alpha >= 1`) colored body voxel carrying its
 *   glow byte for the bloom pass.
 */
export type VoxelMaterialKind = 'empty' | 'black' | 'glass' | 'solid';

/**
 * One resolved material-table record for a palette entry: verbatim
 * color and alpha bytes plus the emissive glow after the LD-23
 * black-exception rule, plus the render class consumers branch on.
 */
export interface VoxelMaterial {
  /** Render classification for the palette entry. */
  readonly kind: VoxelMaterialKind;
  /** Color bytes [r, g, b] copied verbatim from the palette entry. */
  readonly color: readonly [number, number, number];
  /** Alpha byte copied verbatim from the palette entry. */
  readonly alpha: number;
  /** Emissive glow contribution; 0 for empty/black entries (LD-23). */
  readonly emissive: number;
}

/**
 * Resolves the material semantics of one palette entry, as shown in the
 * material table (AC-013 part 1, spec section 7.2 / LD-23). This is the
 * single canonical classification; every consumer shares it so black
 * cells contribute no glow through any consumer.
 *
 * @param entry - Palette entry as `[r, g, b, a, glow]` with integer
 *   bytes 0-255, as validated by `model.schema.ts`.
 * @returns The resolved material: verbatim color/alpha bytes, the
 *   render class, and the emissive glow after the LD-23 black
 *   exception (empty and black entries never emit shine).
 *
 * @example
 * ```ts
 * resolveVoxelMaterial([200, 20, 20, 255, 40]); // { kind: 'solid', ..., emissive: 40 }
 * resolveVoxelMaterial([190, 225, 255, 0, 160]); // { kind: 'glass', ..., emissive: 160 }
 * resolveVoxelMaterial([0, 0, 0, 255, 200]); // { kind: 'black', ..., emissive: 0 }
 * resolveVoxelMaterial([0, 0, 0, 0, 0]); // { kind: 'empty', ..., emissive: 0 }
 * ```
 */
export function resolveVoxelMaterial(entry: VoxelPaletteEntry): VoxelMaterial {
  const [r, g, b, alpha, glow] = entry;
  const kind = classifyMaterialKind(entry);
  // AC-013 black exception: black (and empty) cells yield zero emissive
  // in every material lookup, regardless of their glow byte.
  const emissive =
    kind === 'black' || kind === 'empty' ? VOXEL_MATERIAL_BLACK_EMISSIVE : glow;
  return { kind, color: [r, g, b], alpha, emissive };
}

/**
 * Builds an index-aligned material table over a model palette (AC-013
 * part 1): entry `i` resolves to the material semantics of palette
 * index `i`, so meshing, LOD, and glass consumers perform one
 * O(palette) resolve pass and then O(1) per-voxel table lookups
 * instead of re-classifying every voxel.
 *
 * @param palette - Palette entries as validated by `model.schema.ts`;
 *   index 0 must hold the EMPTY sentinel per section 7.2.
 * @returns One {@link VoxelMaterial} per palette index, in palette
 *   order; the returned table is safe to share across passes.
 *
 * @example
 * ```ts
 * const table = buildVoxelMaterialTable(document.palette);
 * const glassy = table.at(glassPaletteIndex); // { kind: 'glass', ... }
 * ```
 */
export function buildVoxelMaterialTable(
  palette: readonly VoxelPaletteEntry[],
): readonly VoxelMaterial[] {
  return palette.map(resolveVoxelMaterial);
}

/**
 * Classifies one palette entry into its {@link VoxelMaterialKind} by
 * the section 7.2 / LD-23 priority order: the exact EMPTY sentinel
 * first, then the black exception, then glass, then opaque solid body.
 *
 * @param entry - Palette entry bytes `[r, g, b, a, glow]`.
 * @returns The palette entry's render class.
 */
function classifyMaterialKind(entry: VoxelPaletteEntry): VoxelMaterialKind {
  const [r, g, b, alpha] = entry;
  // The EMPTY sentinel is air on every axis of the decision, including
  // indices beyond 0 when a model happens to carry a duplicate.
  if (isEmptyPaletteEntry(entry)) {
    return 'empty';
  }
  // LD-23 black exception: black carries no color, so it can never be
  // shining glass; it stays solid body geometry at opaque alphas with
  // no shine, and transparent black renders nothing.
  if (isBlackColor(r, g, b)) {
    return 'black';
  }
  // Fully transparent colored entries render as shining glass of that
  // color (glow only, no body quads); entry with glow 0 contributes no
  // shine but stays in the glass class for consumer routing.
  if (alpha <= VOXEL_MATERIAL_GLASS_ALPHA_MAX) {
    return 'glass';
  }
  return 'solid';
}

/**
 * Detects black palette entries by their color bytes (LD-23),
 * regardless of alpha or glow.
 *
 * @param r - Red byte.
 * @param g - Green byte.
 * @param b - Blue byte.
 * @returns True when all three color bytes are 0.
 */
function isBlackColor(r: number, g: number, b: number): boolean {
  return r === 0 && g === 0 && b === 0;
}

/**
 * Checks whether a palette entry is the exact EMPTY sentinel entry.
 *
 * @param entry - The palette entry to check.
 * @returns True when every byte matches
 *   {@link VOXEL_EMPTY_PALETTE_ENTRY}.
 */
function isEmptyPaletteEntry(entry: VoxelPaletteEntry): boolean {
  return VOXEL_EMPTY_PALETTE_ENTRY.every(
    (component, index) => component === entry[index],
  );
}
