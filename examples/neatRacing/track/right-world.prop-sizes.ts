/**
 * Canonical axis-aligned half-sizes for every prop archetype emitted by the
 * right-world builder.
 *
 * One table drives three consumers that have to agree exactly:
 * - solid voxel box emission in the right-world render pass,
 * - surface anchoring of billboard sprites in `public.sprite.ts`,
 * - collision and elevation assertions in the right-world test suite.
 *
 * Sharing the table is what guarantees that a billboard's resolved foot
 * surface matches the rendered prop top, so a sprite never floats above - or
 * sinks into - the geometry it is supposed to stand on.
 *
 * The table covers the outer-ring scenery archetypes (pillar, gantry, sign)
 * and the spectator bleacher, plus the inner-ring people archetypes
 * (mechanic, crew, team-banner). Its y component is the vertical half-extent
 * measured from the base plane, so the prop top sits at
 * `position.y + 2 * halfHeight`.
 *
 * The BLEACHER row is the coarse exception: a bleacher is a stepped voxel
 * structure, so the render pass and the sprite anchoring read its size from
 * the real voxel geometry and use the row only as a fallback.
 */

import type { RightWorldProp } from './right-world.builder';

/**
 * Half-width, half-height, half-depth in meters for every right-world prop
 * kind. The y component is the vertical half-extent from the base plane; the
 * prop top is at `position.y + 2 * halfHeight`.
 *
 * @example
 * ```ts
 * const [halfWidth, halfHeight, halfDepth] = RIGHT_WORLD_PROP_HALF_SIZES.pillar;
 * console.log(halfWidth, halfHeight, halfDepth); // 0.4 2 0.4
 * ```
 */
export const RIGHT_WORLD_PROP_HALF_SIZES = {
  pillar: [0.4, 2, 0.4],
  gantry: [1.5, 0.3, 0.3],
  sign: [1, 1, 0.15],
  // Bleacher coarse bounding box of the flat-tier 3-level stadium footprint:
  // 13 m on the WIDTH axis, 3 m tall and 9 m on the DEPTH axis, with the
  // highest stand row at ~3.0 m.
  //
  // The row is a COARSE FALLBACK for bleachers. A bleacher is a stepped
  // voxel structure, so the render pass draws it from its voxel geometry
  // (emitBleacherVoxelBox in right-world.pass.ts) and falls back to this
  // row only for a bleacher that carries no bleacherGeometry, while the
  // sprite anchoring in public.sprite.ts resolves the half extent and the
  // seat top from the same voxel geometry (getBleacherAxisAlignedHalfSize
  // and getBleacherSeatTopY). The row therefore bounds the stand for the
  // consumers that have no geometry to read, and the rendered bleacher size
  // comes from the voxels.
  bleacher: [6.5, 1.5, 4.5],
  // Inner-ring pit props, with their recorded half-sizes.
  // The mechanic and the crew are low PEDESTALS: a 0.25 m vertical
  // half-size, so the pedestal top sits at 0.5 m and the agent billboard
  // is anchored on top of it without hoisting the agent above the crowd.
  // The team banner keeps its recorded panel half-size: a 1.2 m x 1.5 m
  // half-extent panel that is 0.1 m deep.
  mechanic: [0.5, 0.25, 0.5],
  crew: [0.4, 0.25, 0.4],
  'team-banner': [1.2, 1.5, 0.1],
} as const satisfies Record<
  RightWorldProp['kind'],
  readonly [number, number, number]
>;

/**
 * Resolves the canonical axis-aligned half-size of a scenery prop in meters.
 *
 * The lookup is a table read, so the answer is stable and cheap enough to
 * call per prop and per billboard.
 *
 * @param kind - Prop archetype emitted by the right-world builder.
 * @returns Half-width, half-height, half-depth in meters.
 *
 * @example
 * ```ts
 * const [halfWidth, halfHeight, halfDepth] = resolvePropHalfSize('pillar');
 * console.log(halfWidth, halfHeight, halfDepth); // 0.4 2 0.4
 * ```
 */
export function resolvePropHalfSize(
  kind: RightWorldProp['kind'],
): readonly [number, number, number] {
  return RIGHT_WORLD_PROP_HALF_SIZES[kind];
}
