/**
 * Greedy mesher for the Neat Racing voxel pipeline (spec §7.3.1; slice
 * P1-S3-B, AC-009).
 *
 * Consumes a decoded voxel model ({@link DecodedVoxelModel}, produced by
 * `decodeVoxelModel` in `model.decode.utils.ts`) and emits greedy-merged quad
 * records. The merge key is (palette index, normal axis, face orientation):
 * faces merge along a plane run only when all three match, so every merged
 * quad shares one flat palette color and renders identically to unmerged
 * faces under the per-fragment shading model (palette color, signed normal,
 * depth — the G3 merge-contract shading invariant).
 *
 * Interior-face culling: a face is emitted only where exactly one side of the
 * face plane is a body voxel. Body voxels are palette entries with
 * alpha >= {@link VOXEL_OPAQUE_ALPHA_MIN}; fully transparent entries (alpha
 * 0, e.g. glow glass) and the EMPTY sentinel are non-occluding air, so body
 * faces stay visible through glass. Per §7.2 glass is glow only with no body
 * quads — glass-pass geometry is owned by the AC-013 materials/glass slices,
 * not by this mesher.
 *
 * The LD-23 black exception is enforced in the per-voxel shade lookup
 * ({@link resolveVoxelShade}): black (r === g === b === 0) never emits shine
 * regardless of its glow byte, while remaining solid occluding geometry.
 *
 * Output quad record is the normative §7.3.1 shape
 * `{ position, normalAxis, width, height, paletteIndex }`: `position` is the
 * quad's minimum lattice corner, `normalAxis` the face-plane axis, and
 * width/height the extents along the two tangent axes in ascending order
 * (axis 0 → tangents y/z, axis 1 → z/x, axis 2 → x/y). Faces merge only
 * within one plane, one axis, and one orientation. Emission order is
 * deterministic (ascending axis, ascending face plane, row-major mask scan
 * extending width before height) so build outputs are byte-stable for the
 * bin codec (AC-010).
 *
 * The mesher is the single shared implementation for car bodies, wheels,
 * track slabs, and the editor preview (dogfooding, LD-28).
 *
 * @module
 */

import {
  VOXEL_EMPTY_PALETTE_INDEX,
  type VoxelPaletteEntry,
} from './model.schema';
import type { DecodedVoxelModel } from './model.decode.utils';

/**
 * Alpha byte at or above which a palette entry renders as an opaque body
 * voxel; entries below it (including glass at alpha 0) are non-occluding air
 * for body meshing.
 */
export const VOXEL_OPAQUE_ALPHA_MIN = 1;

/** Emissive value forced onto black palette entries (LD-23: black never shines). */
export const VOXEL_BLACK_EXCEPTION_EMISSIVE = 0;

/** Canonical meshing axes in deterministic emission order. */
const VOXEL_MESH_AXES: readonly VoxelNormalAxis[] = [0, 1, 2];

/**
 * Tangent axes (width, height) per normal axis, both ascending; a quad's
 * `width` extends along the first tangent and `height` along the second
 * (fixed mapping per the §7.3.1 record layout).
 */
const FACE_TANGENT_AXES: Readonly<
  Record<VoxelNormalAxis, readonly [number, number]>
> = {
  0: [1, 2],
  1: [0, 2],
  2: [0, 1],
};

/** Face-mask value for a plane cell with no emitted face. */
const VOXEL_MASK_EMPTY = 0;

/** Sentinel for a face-plane neighbor that is not a body voxel. */
const VOXEL_NO_BODY_CELL = -1;

/** Axis a quad face lies normal to: 0 = x, 1 = y, 2 = z. */
export type VoxelNormalAxis = 0 | 1 | 2;

/**
 * Normative §7.3.1 output quad record. `position` is the quad's minimum
 * lattice corner, `normalAxis` the face-plane axis, `width`/`height` the
 * extents along the two tangent axes in ascending order, and
 * `paletteIndex` the shared palette entry of every merged cell.
 */
export interface VoxelQuad {
  /** Minimum lattice corner of the quad; coordinate on `normalAxis` is the face plane. */
  readonly position: readonly [number, number, number];
  /** Face-plane axis: 0 = x, 1 = y, 2 = z. */
  readonly normalAxis: VoxelNormalAxis;
  /** Extent along the first tangent axis (see {@link FACE_TANGENT_AXES}). */
  readonly width: number;
  /** Extent along the second tangent axis (see {@link FACE_TANGENT_AXES}). */
  readonly height: number;
  /** Palette entry shared by every cell merged into the quad. */
  readonly paletteIndex: number;
}

/**
 * Resolved per-voxel render shade for one palette entry: the color bytes
 * verbatim, the alpha byte, and the emissive glow after the LD-23 black
 * exception (black entries carry emissive 0 regardless of their glow byte).
 */
export interface VoxelShade {
  /** Color bytes [r, g, b] copied verbatim from the palette entry. */
  readonly color: readonly [number, number, number];
  /** Alpha byte copied verbatim from the palette entry. */
  readonly alpha: number;
  /** Emissive glow contribution; 0 for black entries (LD-23). */
  readonly emissive: number;
}

/** Greedy-meshed voxel mesh: merged quads plus the source model dimensions. */
export interface VoxelMesh {
  /** Grid dimensions [width, height, depth] echoed from the decoded model. */
  readonly dims: readonly [number, number, number];
  /** Merged quads in deterministic emission order (ascending axis, then plane). */
  readonly quads: readonly VoxelQuad[];
}

/** Face-plane sweep context for one (axis, plane) mask pass. */
interface FacePlaneContext {
  /** Normal axis of the face plane. */
  readonly axis: VoxelNormalAxis;
  /** Lattice coordinate of the face plane along `axis`. */
  readonly plane: number;
  /** First tangent axis (quad width direction). */
  readonly widthAxis: number;
  /** Second tangent axis (quad height direction). */
  readonly heightAxis: number;
}

/**
 * Greedy-meshes a decoded voxel model into merged face quads with interior
 * faces culled (AC-009, spec §7.3.1). Body voxels are palette entries with
 * alpha >= {@link VOXEL_OPAQUE_ALPHA_MIN}; glass and EMPTY cells are
 * non-occluding air, so body faces stay exposed behind glass.
 *
 * @param model - Decoded voxel model from `decodeVoxelModel`; its palette
 *   indices and slice grids are trusted as already validated by the decoder.
 * @returns The merged quad mesh with deterministic emission order (ascending
 *   axis, ascending face plane, row-major mask scan) and the model dims
 *   echoed for bin-header encoding.
 * @example
 * ```ts
 * import { decodeVoxelModel } from './model.decode.utils';
 * const decoded = decodeVoxelModel(document);
 * const mesh = meshVoxelModel(decoded);
 * // mesh.quads[0] -> { position, normalAxis, width, height, paletteIndex }
 * ```
 */
export function meshVoxelModel(model: DecodedVoxelModel): VoxelMesh {
  // Step 1: greedy-mesh each face-axis family in canonical axis order.
  const axisQuads = VOXEL_MESH_AXES.map((axis) => meshAxisFaces(model, axis));
  // Step 2: flatten in deterministic emission order for byte-stable bins.
  return { dims: model.dims, quads: axisQuads.flat() };
}

/**
 * Resolves the per-voxel render shade for one palette entry (AC-009
 * per-voxel color/emissive lookup). The LD-23 black exception is enforced
 * here: black (r === g === b === 0) never shines regardless of its glow
 * byte, while remaining solid geometry in the body pass.
 *
 * @param entry - Palette entry as [r, g, b, a, glow] with bytes 0-255.
 * @returns Color bytes verbatim, alpha verbatim, and the emissive glow
 *   contribution (0 for black entries).
 * @example
 * ```ts
 * resolveVoxelShade([0, 0, 0, 255, 200]); // { color: [0,0,0], alpha: 255, emissive: 0 }
 * resolveVoxelShade([0, 0, 255, 255, 40]); // { color: [0,0,255], alpha: 255, emissive: 40 }
 * ```
 */
export function resolveVoxelShade(entry: VoxelPaletteEntry): VoxelShade {
  const [r, g, b, alpha, glow] = entry;
  // LD-23 black exception: black carries no color and no shine regardless
  // of its glow byte.
  const isBlack = r === 0 && g === 0 && b === 0;
  return {
    color: [r, g, b],
    alpha,
    emissive: isBlack ? VOXEL_BLACK_EXCEPTION_EMISSIVE : glow,
  };
}

/**
 * Meshes one face-axis family: sweeps every face plane perpendicular to
 * `axis`, builds the signed face mask per plane, and greedy-merges maximal
 * equal-key rectangles.
 *
 * @param model - Decoded model to mesh.
 * @param axis - Face-plane normal axis for this sweep.
 * @returns Quads for this axis in ascending-plane emission order.
 */
function meshAxisFaces(
  model: DecodedVoxelModel,
  axis: VoxelNormalAxis,
): VoxelQuad[] {
  const [widthAxis, heightAxis] = FACE_TANGENT_AXES[axis];
  const planeCount = model.dims[axis] + 1;
  const quads: VoxelQuad[] = [];
  for (let plane = 0; plane < planeCount; plane++) {
    const context: FacePlaneContext = { axis, plane, widthAxis, heightAxis };
    const mask = buildFaceMask(model, context);
    quads.push(
      ...mergeMaskRectangles(
        mask,
        model.dims[widthAxis],
        model.dims[heightAxis],
        context,
      ),
    );
  }
  return quads;
}

/**
 * Builds the signed face mask for one face plane. A cell holds +paletteIndex
 * when exactly the behind-side voxel is a body voxel (+face), -paletteIndex
 * when exactly the ahead-side voxel is a body voxel (-face), and 0 when both
 * sides agree (both solid → interior face culled; both air → no face).
 *
 * @param model - Decoded model to probe.
 * @param context - Face-plane sweep context (axis, plane, tangent axes).
 * @returns Mask of dims[widthAxis] x dims[heightAxis] signed palette indices.
 */
function buildFaceMask(
  model: DecodedVoxelModel,
  context: FacePlaneContext,
): Int16Array {
  const maskWidth = model.dims[context.widthAxis];
  const maskHeight = model.dims[context.heightAxis];
  const mask = new Int16Array(maskWidth * maskHeight);
  for (let v = 0; v < maskHeight; v++) {
    for (let u = 0; u < maskWidth; u++) {
      mask[v * maskWidth + u] = faceMaskValue(model, context, u, v);
    }
  }
  return mask;
}

/**
 * Computes one mask cell's signed value from the two face-plane neighbors.
 *
 * @param model - Decoded model to probe.
 * @param context - Face-plane sweep context.
 * @param u - Mask cell coordinate along the width tangent axis.
 * @param v - Mask cell coordinate along the height tangent axis.
 * @returns Signed palette index for the emitted face, or {@link VOXEL_MASK_EMPTY}.
 */
function faceMaskValue(
  model: DecodedVoxelModel,
  context: FacePlaneContext,
  u: number,
  v: number,
): number {
  const behind = voxelBodyIndex(model, context, context.plane - 1, u, v);
  const ahead = voxelBodyIndex(model, context, context.plane, u, v);
  if (behind !== VOXEL_NO_BODY_CELL && ahead === VOXEL_NO_BODY_CELL) {
    return behind;
  }
  if (ahead !== VOXEL_NO_BODY_CELL && behind === VOXEL_NO_BODY_CELL) {
    return -ahead;
  }
  return VOXEL_MASK_EMPTY;
}

/**
 * Resolves the palette index of the body voxel at a face-plane neighbor, or
 * the no-body sentinel. Glass (alpha 0) and EMPTY cells are non-occluding
 * air, so they never return a body index.
 *
 * @param model - Decoded model to probe.
 * @param context - Face-plane sweep context.
 * @param planeCoord - Lattice coordinate along the normal axis (may lie just
 *   outside the grid, which yields no body).
 * @param u - Mask cell coordinate along the width tangent axis.
 * @param v - Mask cell coordinate along the height tangent axis.
 * @returns Body palette index, or {@link VOXEL_NO_BODY_CELL}.
 */
function voxelBodyIndex(
  model: DecodedVoxelModel,
  context: FacePlaneContext,
  planeCoord: number,
  u: number,
  v: number,
): number {
  const coords: [number, number, number] = [0, 0, 0];
  coords[context.axis] = planeCoord;
  coords[context.widthAxis] = u;
  coords[context.heightAxis] = v;
  const index = voxelIndexAt(model, coords);
  if (index === VOXEL_EMPTY_PALETTE_INDEX) {
    return VOXEL_NO_BODY_CELL;
  }
  return isVoxelBodyEntry(model.palette[index]) ? index : VOXEL_NO_BODY_CELL;
}

/**
 * Reads the palette index at lattice coordinates, treating anything outside
 * the grid as the EMPTY sentinel (faces at grid boundaries stay exposed).
 *
 * @param model - Decoded model to probe.
 * @param coords - Lattice coordinates [x, y, z].
 * @returns Palette index at the cell, or {@link VOXEL_EMPTY_PALETTE_INDEX}.
 */
function voxelIndexAt(
  model: DecodedVoxelModel,
  coords: readonly [number, number, number],
): number {
  const [x, y, z] = coords;
  const inGrid =
    x >= 0 &&
    y >= 0 &&
    z >= 0 &&
    x < model.dims[0] &&
    y < model.dims[1] &&
    z < model.dims[2];
  if (!inGrid) {
    return VOXEL_EMPTY_PALETTE_INDEX;
  }
  return model.slices[z][x + y * model.dims[0]];
}

/**
 * Classifies a palette entry as body geometry by its alpha byte.
 *
 * @param entry - Palette entry as [r, g, b, a, glow].
 * @returns True when the entry renders as an opaque body voxel.
 */
function isVoxelBodyEntry(entry: VoxelPaletteEntry): boolean {
  return entry[3] >= VOXEL_OPAQUE_ALPHA_MIN;
}

/**
 * Greedy-merges a face mask into maximal rectangles of equal signed value
 * (the merge key: palette index + orientation within one fixed axis/plane).
 * Scans row-major and extends width before height; consumed cells never
 * join a second rectangle.
 *
 * @param mask - Signed face mask from {@link buildFaceMask}.
 * @param maskWidth - Mask extent along the width tangent axis.
 * @param maskHeight - Mask extent along the height tangent axis.
 * @param context - Face-plane sweep context for record placement.
 * @returns Merged quads in row-major origin order.
 */
function mergeMaskRectangles(
  mask: Int16Array,
  maskWidth: number,
  maskHeight: number,
  context: FacePlaneContext,
): VoxelQuad[] {
  const consumed = new Uint8Array(mask.length);
  const quads: VoxelQuad[] = [];
  for (let v = 0; v < maskHeight; v++) {
    for (let u = 0; u < maskWidth; u++) {
      const origin = v * maskWidth + u;
      const value = mask[origin];
      if (value === VOXEL_MASK_EMPTY || consumed[origin] !== 0) {
        continue;
      }
      const width = measureMaskRunWidth(
        mask,
        consumed,
        origin,
        value,
        v,
        maskWidth,
      );
      const height = measureMaskBlockHeight(
        mask,
        consumed,
        value,
        v,
        u,
        maskWidth,
        maskHeight,
        width,
      );
      consumeMaskBlock(consumed, origin, width, height, maskWidth);
      quads.push(
        createQuadRecord(context, u, v, width, height, Math.abs(value)),
      );
    }
  }
  return quads;
}

/**
 * Measures the maximal run of equal-value unconsumed mask cells along the
 * width tangent axis starting at the rectangle origin (never crossing the
 * row end).
 *
 * @param mask - Signed face mask being merged.
 * @param consumed - Cells already claimed by earlier rectangles.
 * @param origin - Flat mask index of the rectangle origin.
 * @param value - Signed mask value to match.
 * @param v - Origin row (height tangent coordinate).
 * @param maskWidth - Mask extent along the width tangent axis.
 * @returns Run width in cells, at least 1.
 */
function measureMaskRunWidth(
  mask: Int16Array,
  consumed: Uint8Array,
  origin: number,
  value: number,
  v: number,
  maskWidth: number,
): number {
  const rowEnd = (v + 1) * maskWidth;
  let width = 1;
  while (
    origin + width < rowEnd &&
    mask[origin + width] === value &&
    consumed[origin + width] === 0
  ) {
    width++;
  }
  return width;
}

/**
 * Measures how many additional full-width rows of equal-value unconsumed
 * cells can extend the rectangle along the height tangent axis.
 *
 * @param mask - Signed face mask being merged.
 * @param consumed - Cells already claimed by earlier rectangles.
 * @param origin - Flat mask index of the rectangle origin.
 * @param value - Signed mask value to match.
 * @param v - Origin row (height tangent coordinate).
 * @param u - Origin column (width tangent coordinate).
 * @param maskWidth - Mask extent along the width tangent axis.
 * @param maskHeight - Mask extent along the height tangent axis.
 * @param width - Already-measured rectangle width.
 * @returns Rectangle height in rows, at least 1.
 */
function measureMaskBlockHeight(
  mask: Int16Array,
  consumed: Uint8Array,
  value: number,
  v: number,
  u: number,
  maskWidth: number,
  maskHeight: number,
  width: number,
): number {
  let height = 1;
  while (
    v + height < maskHeight &&
    canExtendMaskBlockRow(
      mask,
      consumed,
      value,
      v + height,
      u,
      maskWidth,
      width,
    )
  ) {
    height++;
  }
  return height;
}

/**
 * Checks whether one candidate row fully extends a rectangle: every cell in
 * the row segment must carry the rectangle's signed value and be unconsumed.
 *
 * @param mask - Signed face mask being merged.
 * @param consumed - Cells already claimed by earlier rectangles.
 * @param origin - Flat mask index of the rectangle origin.
 * @param value - Signed mask value to match.
 * @param row - Candidate height tangent row index.
 * @param u - Origin column (width tangent coordinate).
 * @param maskWidth - Mask extent along the width tangent axis.
 * @param width - Rectangle width to validate.
 * @returns True when the row segment can join the rectangle.
 */
function canExtendMaskBlockRow(
  mask: Int16Array,
  consumed: Uint8Array,
  value: number,
  row: number,
  u: number,
  maskWidth: number,
  width: number,
): boolean {
  const rowStart = row * maskWidth;
  for (let offset = 0; offset < width; offset++) {
    const probe = rowStart + u + offset;
    if (mask[probe] !== value || consumed[probe] !== 0) {
      return false;
    }
  }
  return true;
}

/**
 * Marks a merged rectangle's mask cells as consumed.
 *
 * @param consumed - Claimed-cell tracker to mutate.
 * @param origin - Flat mask index of the rectangle origin.
 * @param width - Rectangle width.
 * @param height - Rectangle height.
 * @param maskWidth - Mask extent along the width tangent axis.
 */
function consumeMaskBlock(
  consumed: Uint8Array,
  origin: number,
  width: number,
  height: number,
  maskWidth: number,
): void {
  for (let row = 0; row < height; row++) {
    const rowStart = origin + row * maskWidth;
    consumed.fill(1, rowStart, rowStart + width);
  }
}

/**
 * Assembles the normative quad record for one merged rectangle, mapping the
 * face-plane context and mask cell origin onto lattice coordinates: the
 * plane coordinate lands on `axis`, the mask origin on the tangent axes.
 *
 * @param context - Face-plane sweep context (axis, plane, tangent axes).
 * @param u - Rectangle origin along the width tangent axis.
 * @param v - Rectangle origin along the height tangent axis.
 * @param width - Rectangle width along the first tangent axis.
 * @param height - Rectangle height along the second tangent axis.
 * @param paletteIndex - Shared palette entry of the merged cells.
 * @returns The normative `{ position, normalAxis, width, height, paletteIndex }` record.
 */
function createQuadRecord(
  context: FacePlaneContext,
  u: number,
  v: number,
  width: number,
  height: number,
  paletteIndex: number,
): VoxelQuad {
  const position: [number, number, number] = [0, 0, 0];
  position[context.axis] = context.plane;
  position[context.widthAxis] = u;
  position[context.heightAxis] = v;
  return { position, normalAxis: context.axis, width, height, paletteIndex };
}
