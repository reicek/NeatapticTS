/**
 * Mirror symmetry painting for the NeatRacing editor (slice P1-S6-C; spec
 * sections 7.2 / 8.3, AC-021, LD-28): deterministic expansion of one paint
 * voxel into every mirrored position across the enabled axes.
 *
 * The mapping is identical on every axis: on the 128-cubed editor volume a
 * coordinate `c` mirrors to `{@linkcode NEATRACING_EDITOR_MIRROR_MAX_COORDINATE} - c`
 * (col c <-> col 127-c; spec section 7.2 stores only the canonical half,
 * cols 0..63, so a paint above the mirror line canonicalizes down first —
 * `map(127, y, z)` emits the same ordered set as `map(0, y, z)`).
 *
 * {@linkcode EditorSymmetry.map} emits the canonical primary tuple first,
 * then exactly one tuple per enabled-axis mirror subset in ascending subset
 * order (x < y < xy < z < xz < yz < xyz), so combined modes compose into the
 * full axis-power set (1–8 tuples) in a stable, duplicate-free order.
 * Mirror-X symmetry is enabled by default (LD-28 — only half is stored);
 * mirror-Y and mirror-Z are opt-in, and an all-flags-off configuration
 * leaves only the primary voxel (disabled axes pass through unchanged).
 * Coordinates outside 0..127 are rejected with the same
 * {@linkcode RangeError} guard the editor volume paint path applies.
 *
 * The module stays pure — no DOM, no WebGL, no random source — so the
 * expansion is exactly deterministic: same flags and same coordinates in,
 * the same ordered tuple list out on every call and runtime.
 */

import { NEATRACING_EDITOR_VOLUME_EDGE } from './editor.state';

/**
 * Highest inclusive voxel coordinate on any editor axis: 128 - 1 = 127.
 * The mirror mapping is `c <-> NEATRACING_EDITOR_MIRROR_MAX_COORDINATE - c`,
 * so a paint at 0 expands across the mirror to 127 and vice versa.
 */
export const NEATRACING_EDITOR_MIRROR_MAX_COORDINATE =
  NEATRACING_EDITOR_VOLUME_EDGE - 1;

/** Default mirror-X flag: x symmetry paints are on by default (LD-28). */
export const NEATRACING_EDITOR_DEFAULT_MIRROR_X = true;

/** Default mirror-Y flag: y symmetry is opt-in. */
export const NEATRACING_EDITOR_DEFAULT_MIRROR_Y = false;

/** Default mirror-Z flag: z symmetry is opt-in. */
export const NEATRACING_EDITOR_DEFAULT_MIRROR_Z = false;

/** One mirrored voxel position, addressed as `[x, y, z]`. */
export type EditorSymmetryTarget = [number, number, number];

/** Names of the axes mirror symmetry can expand a paint across. */
type MirroredAxisName = 'x' | 'y' | 'z';

/** Factory configuration for mirror symmetry painting. */
export interface EditorSymmetryConfig {
  /** Mirror paints across the x (column) axis; defaults to true (LD-28). */
  mirrorX?: boolean;
  /** Mirror paints across the y (row) axis; defaults to false. */
  mirrorY?: boolean;
  /** Mirror paints across the z (slice) axis; defaults to false. */
  mirrorZ?: boolean;
}

/** Public mirror-symmetry surface the editor paint path drives. */
export interface EditorSymmetry {
  /** Whether paints mirror across the x (column) axis. */
  readonly mirrorX: boolean;
  /** Whether paints mirror across the y (row) axis. */
  readonly mirrorY: boolean;
  /** Whether paints mirror across the z (slice) axis. */
  readonly mirrorZ: boolean;

  /**
   * Expands one paint voxel into every mirrored position deterministically.
   *
   * Emits the canonical primary voxel first (each enabled axis folded into
   * its stored mirror half), then one tuple per enabled-axis mirror subset
   * in ascending subset order (x < y < xy < z < xz < yz < xyz). Disabled
   * axes keep the painted coordinate unchanged.
   *
   * @param x - Voxel column, inclusive in 0..127.
   * @param y - Voxel row, inclusive in 0..127.
   * @param z - Voxel slice, inclusive in 0..127.
   * @returns Ordered mirrored positions (1 tuple when fully off, 2 per
   * single axis, up to 8 when every axis is enabled).
   * @throws RangeError when a coordinate falls outside the volume.
   * @example
   * ```ts
   * const symmetry = createEditorSymmetry(); // mirror-X on by default
   * symmetry.map(10, 20, 30); // [[10, 20, 30], [117, 20, 30]]
   * ```
   */
  map(x: number, y: number, z: number): EditorSymmetryTarget[];
}

/**
 * Validates one symmetry coordinate against the volume edge, reporting the
 * failing axis name in the thrown range error (mirrors the paint-path guard
 * in `editor.state.ts`).
 *
 * @param axisName - Axis label for the error message ('x', 'y', or 'z').
 * @param value - Coordinate value to validate.
 * @throws RangeError when `value` is not an integer in `0..127`.
 */
function validateSymmetryCoordinate(axisName: string, value: number): void {
  if (
    !Number.isInteger(value) ||
    value < 0 ||
    value > NEATRACING_EDITOR_MIRROR_MAX_COORDINATE
  ) {
    throw new RangeError(
      `voxel ${axisName} ${value} is outside the volume range 0..${NEATRACING_EDITOR_MIRROR_MAX_COORDINATE}`,
    );
  }
}

/**
 * Mirrors one coordinate across its axis under the 127 - c rule.
 *
 * @param value - Coordinate to mirror.
 * @returns The mirrored coordinate (c and the result are always distinct on
 * the 128-volume, so mirroring never produces a self-duplicate).
 */
function mirrorCoordinate(value: number): number {
  return NEATRACING_EDITOR_MIRROR_MAX_COORDINATE - value;
}

/**
 * Folds one coordinate into the stored mirror half of its axis pair.
 *
 * Spec section 7.2 stores only cols 0..63 (the canonical half); a paint
 * above the mirror line canonicalizes down so the primary tuple is
 * volume-canonical before mirror subsets are emitted.
 *
 * @param value - Coordinate to fold.
 * @returns The lower member of `{value, mirror(value)}`.
 */
function canonicalizeMirrorCoordinate(value: number): number {
  return Math.min(value, mirrorCoordinate(value));
}

/**
 * Mirrors one tuple across exactly one axis, leaving the others unchanged.
 *
 * @param axis - Axis to mirror ('x', 'y', or 'z').
 * @param x - Current x coordinate.
 * @param y - Current y coordinate.
 * @param z - Current z coordinate.
 * @returns The tuple with the named axis coordinate mirrored.
 */
function mirrorAlongAxis(
  axis: MirroredAxisName,
  x: number,
  y: number,
  z: number,
): EditorSymmetryTarget {
  if (axis === 'x') {
    return [mirrorCoordinate(x), y, z];
  }
  if (axis === 'y') {
    return [x, mirrorCoordinate(y), z];
  }
  return [x, y, mirrorCoordinate(z)];
}

/**
 * Collects the enabled mirror axes in ascending subset-bit order.
 *
 * @param mirrorX - Whether the x axis is mirrored.
 * @param mirrorY - Whether the y axis is mirrored.
 * @param mirrorZ - Whether the z axis is mirrored.
 * @returns Enabled axis names ordered x, then y, then z (mask bit 0..2).
 */
function collectEnabledAxes(
  mirrorX: boolean,
  mirrorY: boolean,
  mirrorZ: boolean,
): MirroredAxisName[] {
  const enabled: MirroredAxisName[] = [];
  if (mirrorX) {
    enabled.push('x');
  }
  if (mirrorY) {
    enabled.push('y');
  }
  if (mirrorZ) {
    enabled.push('z');
  }
  return enabled;
}

/**
 * Canonicalizes a paint coordinate per enabled axis, leaving disabled axes
 * untouched.
 *
 * @param x - Painted x coordinate.
 * @param y - Painted y coordinate.
 * @param z - Painted z coordinate.
 * @param mirrorX - Whether the x axis is mirrored.
 * @param mirrorY - Whether the y axis is mirrored.
 * @param mirrorZ - Whether the z axis is mirrored.
 * @returns The canonical primary tuple for the paint.
 */
function canonicalizePaint(
  x: number,
  y: number,
  z: number,
  mirrorX: boolean,
  mirrorY: boolean,
  mirrorZ: boolean,
): EditorSymmetryTarget {
  return [
    mirrorX ? canonicalizeMirrorCoordinate(x) : x,
    mirrorY ? canonicalizeMirrorCoordinate(y) : y,
    mirrorZ ? canonicalizeMirrorCoordinate(z) : z,
  ];
}

/**
 * Applies one mirror-subset bitmask to the canonical tuple.
 *
 * Bit `i` mirrors across `axes[i]`, so iterating masks 1..2^n-1 in order
 * yields exactly the ascending subset order (x < y < xy < z < xz < yz <
 * xyz) the emission contract promises.
 *
 * @param axes - Enabled mirror axes in mask-bit order.
 * @param mask - Subset bitmask selecting which axes to mirror.
 * @param base - Canonical tuple to mirror from.
 * @returns The tuple with every masked axis mirrored.
 */
function applyMirrorSubset(
  axes: MirroredAxisName[],
  mask: number,
  base: EditorSymmetryTarget,
): EditorSymmetryTarget {
  let [x, y, z] = base;
  for (const [bitIndex, axis] of axes.entries()) {
    if ((mask & (1 << bitIndex)) !== 0) {
      [x, y, z] = mirrorAlongAxis(axis, x, y, z);
    }
  }
  return [x, y, z];
}

/**
 * Emits the full ordered symmetric target list for a canonical paint.
 *
 * Collects the canonical primary first, then walks every enabled-axis
 * mirror subset mask in ascending order, producing 2^n duplicate-free
 * tuples (n = enabled axis count; n = 0 leaves only the primary).
 *
 * @param base - Canonical primary tuple.
 * @param axes - Enabled mirror axes in mask-bit order.
 * @returns Ordered mirrored positions, primary first.
 */
function collectMirrorSubsets(
  base: EditorSymmetryTarget,
  axes: MirroredAxisName[],
): EditorSymmetryTarget[] {
  const targets: EditorSymmetryTarget[] = [base];
  const mirrorSubsetCount = 2 ** axes.length;
  for (let mask = 1; mask < mirrorSubsetCount; mask++) {
    targets.push(applyMirrorSubset(axes, mask, base));
  }
  return targets;
}

/**
 * Creates the editor mirror-symmetry expansion policy.
 *
 * The expansion is pure and exactly deterministic (Level 3 at the call
 * boundary): identical flags and coordinates always produce the identical
 * ordered tuple list on every runtime, with the mirrors emitted in
 * ascending subset order (x < y < xy < z < xz < yz < xyz).
 *
 * @param config - Partial axis configuration; mirror-X defaults to `true`
 * (LD-28 — only half the volume is stored), mirror-Y and mirror-Z default
 * to `false` unless explicitly overridden.
 * @returns Ready-to-use symmetry expansion implementing
 * {@linkcode EditorSymmetry}.
 * @example
 * ```ts
 * import { createEditorSymmetry } from './editor.symmetry';
 * const xy = createEditorSymmetry({ mirrorY: true });
 * xy.map(10, 20, 30); // [[10, 20, 30], [117, 20, 30], [10, 107, 30], [117, 107, 30]]
 * const off = createEditorSymmetry({ mirrorX: false });
 * off.map(10, 20, 30); // [[10, 20, 30]]
 * ```
 */
export function createEditorSymmetry(
  config?: EditorSymmetryConfig,
): EditorSymmetry {
  const mirrorX = config?.mirrorX ?? NEATRACING_EDITOR_DEFAULT_MIRROR_X;
  const mirrorY = config?.mirrorY ?? NEATRACING_EDITOR_DEFAULT_MIRROR_Y;
  const mirrorZ = config?.mirrorZ ?? NEATRACING_EDITOR_DEFAULT_MIRROR_Z;
  const enabledAxes = collectEnabledAxes(mirrorX, mirrorY, mirrorZ);
  const canonicalize = (
    x: number,
    y: number,
    z: number,
  ): EditorSymmetryTarget =>
    canonicalizePaint(x, y, z, mirrorX, mirrorY, mirrorZ);

  return {
    mirrorX,
    mirrorY,
    mirrorZ,

    map(x, y, z): EditorSymmetryTarget[] {
      // Step 1: reject out-of-range coordinates with the paint-path guard.
      validateSymmetryCoordinate('x', x);
      validateSymmetryCoordinate('y', y);
      validateSymmetryCoordinate('z', z);

      // Step 2: fold the paint into the stored mirror half per enabled axis.
      const canonical = canonicalize(x, y, z);

      // Step 3: emit the primary, then every enabled mirror subset in order.
      return collectMirrorSubsets(canonical, enabledAxes);
    },
  };
}
