/**
 * Quality-tier rendering constants for the Neatenstein neon renderer.
 *
 * Centralises half-resolution raycasting toggles and layer-exemption flags so
 * that temporal coherence (C1.4) can be enabled or disabled at runtime without
 * breaking the floor/wall/ceiling grid alignment invariants (§1, §3, §8).
 *
 * @module
 */

/**
 * Default enabled state for half-resolution wall-color rendering.
 *
 * When `true`, the renderer casts every other wall column and interpolates the
 * rest. The floor/ceiling grid, spark overlay, and z-buffer are always
 * full-resolution (see exemption flags below).
 */
export const NEATENSTEIN_HALF_RES_DEFAULT_ENABLED = false as const;

/**
 * Flag indicating the z-buffer is cast at every column even when half-res is
 * enabled (Invariant §1).
 *
 * Half-res decimation applies ONLY to wall color — depth occlusion must stay
 * exact so sprites and sparks are correctly depth-tested.
 */
export const NEATENSTEIN_HALF_RES_ZBUFFER_FULL_RESOLUTION = 1 as const;

/**
 * Flag indicating the floor/ceiling grid is exempt from half-res decimation.
 *
 * The stroked grid MUST stay at full resolution so grid lines remain aligned
 * with wall bases (Invariant §1) and the traveling spark stays coupled to its
 * grid line (Invariant §3).
 */
export const NEATENSTEIN_HALF_RES_GRID_EXEMPT = 1 as const;

/**
 * Flag indicating the spark/pulse overlay is exempt from half-res decimation.
 *
 * The traveling spark slides along integer floor-grid lines and must remain at
 * full resolution so it does not detach from the grid (Invariant §3).
 */
export const NEATENSTEIN_HALF_RES_SPARK_EXEMPT = 1 as const;

/**
 * Quality settings object accepted by {@link resolveNeatensteinHalfResEnabled}.
 */
export interface NeatensteinHalfResSettings {
  /** Runtime toggle to enable or disable half-resolution wall rendering. */
  halfResEnabled?: boolean;
}

/**
 * Resolved half-resolution configuration.
 *
 * When half-res is disabled, returns `false`. When enabled, returns an object
 * describing which layers are exempt from decimation.
 */
export type NeatensteinHalfResConfig =
  | {
      enabled: true;
      gridExempt: boolean;
      sparkExempt: boolean;
      zbufferFullResolution: boolean;
    }
  | false;

/**
 * Resolve whether half-resolution wall rendering is enabled and which layers
 * are exempt from decimation (C1.4).
 *
 * When `halfResEnabled` is `false` (or unset and the default is disabled), the
 * function returns `false` — all layers render at full resolution. When
 * enabled, it returns a config object marking the grid, spark, and z-buffer as
 * exempt so only wall COLOR is decimated.
 *
 * @param settings - Runtime quality settings.
 * @returns `false` when half-res is off, or a config object when on.
 */
export function resolveNeatensteinHalfResEnabled(
  settings: NeatensteinHalfResSettings,
): NeatensteinHalfResConfig {
  const enabled =
    settings?.halfResEnabled ?? NEATENSTEIN_HALF_RES_DEFAULT_ENABLED;

  if (!enabled) {
    return false;
  }

  return {
    enabled: true,
    gridExempt: NEATENSTEIN_HALF_RES_GRID_EXEMPT === 1,
    sparkExempt: NEATENSTEIN_HALF_RES_SPARK_EXEMPT === 1,
    zbufferFullResolution: NEATENSTEIN_HALF_RES_ZBUFFER_FULL_RESOLUTION === 1,
  };
}
