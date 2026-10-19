/**
 * Canonical axis-aligned half-sizes for every prop archetype emitted by the
 * right-world builder. The same table drives:
 * - solid voxel box emission in the right-world render pass,
 * - surface-anchoring of billboard sprites in public.sprite and mechanic.sprite,
 * - collision/elevation assertions in the right-world test suite.
 *
 * Keeping one source of truth guarantees that a billboard's resolved foot
 * surface matches the rendered prop top (AC-052 + AC-053) and that inner-ring
 * mechanic/crew props are rendered as low pedestals that do not intersect the
 * sprites standing on them.
 */

import type { RightWorldProp } from './right-world.builder';

/**
 * Half-width, half-height, half-depth in meters for every right-world prop
 * kind. The y component is the vertical half-extent from the base plane; the
 * prop top is at `position.y + 2 * halfHeight`.
 */
export const RIGHT_WORLD_PROP_HALF_SIZES = {
  pillar: [0.4, 2, 0.4],
  gantry: [1.5, 0.3, 0.3],
  sign: [1, 1, 0.15],
  bleacher: [2, 0.6, 1],
  // Low inner-ring pedestals for mechanics/crew; the sprite stands on top so
  // the billboard never shares volume with a full-height cube (OBS-2).
  mechanic: [0.5, 0.25, 0.5],
  crew: [0.4, 0.25, 0.4],
  'team-banner': [1.2, 1.5, 0.1],
} as const satisfies Record<RightWorldProp['kind'], readonly [number, number, number]>;

/**
 * Resolves the canonical axis-aligned half-size of a scenery prop in meters.
 *
 * @param kind - Prop archetype emitted by the right-world builder.
 * @returns Half-width, half-height, half-depth in meters.
 */
export function resolvePropHalfSize(
  kind: RightWorldProp['kind'],
): readonly [number, number, number] {
  return RIGHT_WORLD_PROP_HALF_SIZES[kind];
}
