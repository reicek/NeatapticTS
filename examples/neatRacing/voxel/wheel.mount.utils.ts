/**
 * Wheel mount contract for the Neat Racing voxel racer.
 *
 * Implements the spec §7.4 "G9 — pinned mount contract" (LD-21, AC-015, TP-14):
 *
 * 1. Wheels are separate small rotating voxel models authored in the compact
 *    voxel JSON format at the wheel's own voxel world scale. The wheel grid is
 *    the TP-14 normative 32-cubed grid and shares
 *    {@link NEATRACING_WHEEL_VOXEL_WORLD_SIZE} (== `NEATRACING_VOXEL_WORLD_SIZE`
 *    from `./lod.distance.utils`, which stays the single owner of the voxel
 *    scale constant).
 * 2. Hub height: a wheel is mounted at hub height equal to the wheel's world
 *    radius, so the contact patch sits exactly on the ground plane `y = 0`
 *    (spec §7.4: "mounted at hub height = the wheel's world radius so the
 *    contact patch sits at `y = 0`").
 * 3. Pivot: the wheel's rotation pivot is the disc's geometric center
 *    (the normalized grid center, fractions `[0.5, 0.5, 0.5]`).
 * 4. Authoring frame and mount: the disc plane contains the vertical (`y`)
 *    axis and the spin axis is the model's depth axis before mounting. The
 *    mount rotates the model by exactly +90° around the car's up axis, which
 *    maps the pre-mount depth axis onto the car's lateral (car-right) axis,
 *    so the spin axis becomes the lateral axle line shared by left and right
 *    wheels.
 * 5. Spin: the true rotation advances with ground speed as `ω = v / r`. The
 *    *displayed* spin is capped at
 *    {@link NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S} revolutions per second
 *    (spec §7.4: near 320 km/h a ~0.33 m wheel spins near 270 rad/s and
 *    strobes; the cap keeps the display alias-free). The cap is applied after
 *    the true ω is computed, so physics is unaffected.
 *
 * The car frame convention used by socket offsets is the car model frame:
 * `+x` = lateral (car-right), `+y` = up, `+z` = forward (car nose), with
 * offsets measured from the car's model center. Deterministic: every exported
 * function is a pure function of its inputs with safe no-spin fallbacks for
 * non-finite numbers, so browser and replay runs agree bit-for-bit.
 *
 * Row-0-is-down, RLE row decoding and palette validation stay owned by
 * `./model.schema` (and the decoder); this module only derives the wheel's
 * world geometry from the declared (validated) document shape.
 *
 * @module
 */

import { NEATRACING_VOXEL_WORLD_SIZE } from './lod.distance.utils';
import { VoxelModelError, type VoxelModelDocument } from './model.schema';

/** Radians per revolution (2·π); converts rev/s to rad/s. */
const REV_PER_S_TO_RAD_PER_S = 2 * Math.PI;

/** Divisor that turns an axle span (wheelbase, track) into its half-span. */
const HALF_AXLE_SPAN_DIVISOR = 2;

/**
 * Normative wheel grid side in voxels (TP-14: wheel models use the 32-cubed
 * grid while car bodies use 128).
 */
export const NEATRACING_WHEEL_GRID_SIZE = 32;

/**
 * Wheel voxel world size in meters per voxel. Shared with the car scale:
 * spec §7.4 requires wheel models at the car's voxel world scale, and the
 * scale constant stays owned by `./lod.distance.utils`.
 */
export const NEATRACING_WHEEL_VOXEL_WORLD_SIZE = NEATRACING_VOXEL_WORLD_SIZE;

/**
 * Wheel model world extent in meters (grid side × voxel world size), i.e. the
 * tire's outer diameter for a disc spanning the declared height axis:
 * 32 × 4.5/128 = 1.125 m.
 */
export const NEATRACING_WHEEL_WORLD_SIZE_M =
  NEATRACING_WHEEL_GRID_SIZE * NEATRACING_WHEEL_VOXEL_WORLD_SIZE;

/**
 * Wheel world radius in meters (half the disc's world diameter):
 * 1.125 / 2 = 0.5625 m (spec §7.4: the wheel's world radius).
 */
export const NEATRACING_WHEEL_WORLD_RADIUS_M =
  NEATRACING_WHEEL_WORLD_SIZE_M / HALF_AXLE_SPAN_DIVISOR;

/**
 * Hub height in meters above the ground plane. Spec-pinned to the wheel's
 * world radius (G9), which puts the contact patch exactly at `y = 0`:
 * 0.5625 − 0.5625 = 0.
 */
export const NEATRACING_WHEEL_HUB_HEIGHT_M = NEATRACING_WHEEL_WORLD_RADIUS_M;

/**
 * Normative wheel rotation pivot in voxel coordinates: the disc's geometric
 * center, i.e. half the grid on every axis (G9: pivot = disc center).
 */
export const NEATRACING_WHEEL_PIVOT_VOXEL =
  NEATRACING_WHEEL_GRID_SIZE / HALF_AXLE_SPAN_DIVISOR;

/**
 * Normative wheel rotation pivot in normalized [0, 1] grid fractions: the
 * geometric center of the cubic grid on every axis (G9).
 */
export const NEATRACING_WHEEL_PIVOT_FRACTIONS = [0.5, 0.5, 0.5] as const;

/**
 * Mount yaw in radians around the car's up axis. Spec §7.4: the mount rotates
 * the model 90° so the pre-mount depth (spin) axis becomes the lateral axis.
 * +π/2 maps the model +z axis onto the car's +x (car-right) axis, so the same
 * axle line serves the left and right wheels (spin sign is carried by ω).
 */
export const NEATRACING_WHEEL_MOUNT_YAW_RAD = Math.PI / HALF_AXLE_SPAN_DIVISOR;

/**
 * Displayed wheel spin cap in revolutions per second (G9 strobing cap): the
 * display never renders faster than 2 rev/s no matter the ground speed. The
 * cap is applied after the true ω, so physics is unaffected.
 */
export const NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S = 2;

/**
 * Displayed wheel spin cap expressed in radians per second
 * (2 rev/s × 2π ≈ 12.566 rad/s), the exact value returned by
 * {@link clampVisualWheelSpinOmega} for over-cap inputs.
 */
export const NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S =
  NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S * REV_PER_S_TO_RAD_PER_S;

/**
 * Wheelbase in meters (distance between the front and rear axle lines,
 * measured from the car's model center front-to-back). Spec-unpinned F1-style
 * tuning point for a ~4.5 m car (≈ 71 % of the car length); retune here, not
 * in geometry code.
 */
export const NEATRACING_WHEEL_WHEELBASE_M = 3.2;

/**
 * Track width in meters (distance between the left and right wheel center
 * lines). Spec-unpinned F1-style tuning point for a ~4.5 m car (≈ 44 % of the
 * car length); retune here, not in geometry code.
 */
export const NEATRACING_WHEEL_TRACK_M = 2;

/** Deterministic socket order: front-left, front-right, rear-left, rear-right. */
export const NEATRACING_WHEEL_SOCKET_ORDER = [
  'front-left',
  'front-right',
  'rear-left',
  'rear-right',
] as const;

/**
 * Mirror signs for each socket in the car frame: `[lateralSign,
 * longitudinalSign]`. Single fixed mapping table shared by the socket list
 * and the mount transform so both surfaces can never disagree.
 */
const WHEEL_SOCKET_AXIS_SIGNS: Readonly<
  Record<
    VoxelWheelSocketId,
    readonly [lateralSign: 1 | -1, longitudinalSign: 1 | -1]
  >
> = {
  'front-left': [-1, 1],
  'front-right': [1, 1],
  'rear-left': [-1, -1],
  'rear-right': [1, -1],
};

/** Pre-mount spin axis label (authoring frame): the model's depth axis. */
export type VoxelWheelSpinAxisPreMount = 'model-depth';

/** Mounted spin axis label (car frame): the lateral car-right axis. */
export type VoxelWheelSpinAxisMounted = 'car-lateral';

/** The four wheel sockets of the voxel racer. */
export type VoxelWheelSocketId =
  'front-left' | 'front-right' | 'rear-left' | 'rear-right';

/** One wheel socket: identity plus its hub-mount offset in the car frame. */
export interface VoxelWheelSocket {
  /** Deterministic socket identity. */
  readonly id: VoxelWheelSocketId;
  /**
   * Hub-mount offset in meters from the car's model center, as
   * `[lateral, hubHeight, longitudinal]` (`+x` car-right, `+y` up,
   * `+z` forward). `y` is the hub height, so the contact patch (one wheel
   * radius below the hub) sits at `y = 0`.
   */
  readonly offset: readonly [number, number, number];
}

/** World-space geometry derived from a wheel model document. */
export interface VoxelWheelMetrics {
  /** Wheel grid side in voxels (the declared, normative TP-14 grid). */
  readonly gridVoxels: number;
  /** Meters per voxel shared with the car scale. */
  readonly voxelWorldSize: number;
  /** Wheel disc radius in meters (half the declared height extent). */
  readonly discRadiusM: number;
  /** Hub height in meters equals the disc radius (G9), contact patch at y=0. */
  readonly hubHeightM: number;
  /** Disc rotation pivot as normalized grid-center fractions (G9). */
  readonly pivot: readonly [number, number, number];
}

/** Full mount transform for one wheel socket (G9: hub height + 90° yaw). */
export interface VoxelWheelMountTransform {
  /** Socket this transform mounts. */
  readonly socketId: VoxelWheelSocketId;
  /** Car-frame mount position `[lateral, hubHeight, longitudinal]` in meters. */
  readonly position: readonly [number, number, number];
  /** Disc rotation pivot as normalized grid-center fractions (G9). */
  readonly pivot: readonly [number, number, number];
  /** Yaw in radians around car up (+π/2): pre-mount depth → lateral. */
  readonly rotationYawRad: number;
  /** Spin axis in the authoring frame: the model's depth axis. */
  readonly spinAxisPreMount: VoxelWheelSpinAxisPreMount;
  /** Spin axis after mounting: the lateral (car-right) axle line. */
  readonly spinAxisMounted: VoxelWheelSpinAxisMounted;
}

/**
 * Named rejection for wheel documents that do not satisfy the wheel-scale
 * format contract (spec §7.4 / TP-14: normative 32-cubed wheel grid).
 */
export class VoxelWheelFormatError extends VoxelModelError {
  /** Stable error code shared by every wheel-format rejection. */
  static readonly code = 'voxel-wheel/format';

  constructor(message: string) {
    super(VoxelWheelFormatError.code, `voxel wheel format: ${message}`);
  }
}

const WHEEL_GRID_EXPECTED_AXES = 3;

/**
 * Resolves the wheel's world geometry from a compact wheel model document.
 *
 * Derives the disc radius from the declared height axis
 * (`height / 2 × voxel world size`), the hub height (G9: equal to the radius),
 * and the grid-center rotation pivot. Expects the normative TP-14 wheel grid
 * (see {@link NEATRACING_WHEEL_GRID_SIZE}); deeper row/RLE/palette validation
 * stays owned by `./model.schema` and the decoder.
 *
 * @param wheelModel - A compact voxel model document authored as a wheel.
 * @returns The wheel's world-space metrics.
 * @throws VoxelWheelFormatError when the document is not a wheel-scale model
 * with the normative 32-cubed grid, a palette, and one slice per depth index.
 */
export function resolveVoxelWheelMetrics(
  wheelModel: VoxelModelDocument,
): VoxelWheelMetrics {
  const dims = wheelModel?.meta?.dims;
  if (
    !Array.isArray(dims) ||
    dims.length !== WHEEL_GRID_EXPECTED_AXES ||
    dims[0] !== NEATRACING_WHEEL_GRID_SIZE ||
    dims[1] !== NEATRACING_WHEEL_GRID_SIZE ||
    dims[2] !== NEATRACING_WHEEL_GRID_SIZE
  ) {
    throw new VoxelWheelFormatError(
      `expected the normative ${NEATRACING_WHEEL_GRID_SIZE}-cubed wheel grid, got ${String(dims)}`,
    );
  }
  if (!Array.isArray(wheelModel.palette)) {
    throw new VoxelWheelFormatError('missing palette');
  }
  if (
    !Array.isArray(wheelModel.slices) ||
    wheelModel.slices.length !== NEATRACING_WHEEL_GRID_SIZE
  ) {
    throw new VoxelWheelFormatError(
      `expected ${NEATRACING_WHEEL_GRID_SIZE} slices (one per depth index), got ${String(wheelModel.slices?.length)}`,
    );
  }

  return {
    gridVoxels: NEATRACING_WHEEL_GRID_SIZE,
    voxelWorldSize: NEATRACING_WHEEL_VOXEL_WORLD_SIZE,
    discRadiusM:
      (dims[1] / HALF_AXLE_SPAN_DIVISOR) * NEATRACING_WHEEL_VOXEL_WORLD_SIZE,
    hubHeightM: NEATRACING_WHEEL_HUB_HEIGHT_M,
    pivot: NEATRACING_WHEEL_PIVOT_FRACTIONS,
  };
}

/**
 * Resolves the four wheel sockets at spec-defined offsets (plan AC: wheels
 * mount at 4 sockets; G9: hub height so the contact patch sits at y = 0).
 *
 * Deterministic order: front-left, front-right, rear-left, rear-right.
 * Longitudinal offsets are ±`NEATRACING_WHEEL_WHEELBASE_M / 2` and lateral
 * offsets are ±`NEATRACING_WHEEL_TRACK_M / 2` around the car's model center
 * (`+x` car-right, `+z` forward). Override `hubHeightM` only when mounting a
 * non-normative wheel radius; the default keeps the contact patch at y = 0.
 *
 * @param hubHeightM - Hub height in meters; defaults to the normative wheel hub.
 * @returns The four sockets in the pinned deterministic order.
 */
export function resolveVoxelWheelSockets(
  hubHeightM: number = NEATRACING_WHEEL_HUB_HEIGHT_M,
): readonly VoxelWheelSocket[] {
  const halfWheelbaseM = NEATRACING_WHEEL_WHEELBASE_M / HALF_AXLE_SPAN_DIVISOR;
  const halfTrackM = NEATRACING_WHEEL_TRACK_M / HALF_AXLE_SPAN_DIVISOR;

  return NEATRACING_WHEEL_SOCKET_ORDER.map((id) => {
    const [lateralSign, longitudinalSign] = WHEEL_SOCKET_AXIS_SIGNS[id];
    return {
      id,
      offset: [
        lateralSign * halfTrackM,
        hubHeightM,
        longitudinalSign * halfWheelbaseM,
      ] as const,
    };
  });
}

/**
 * Resolves the mount transform for one wheel socket (G9): position at the
 * socket's hub offset, rotation pivot at the disc's geometric center
 * (`NEATRACING_WHEEL_PIVOT_FRACTIONS`), and the spec +90° mount yaw that maps
 * the pre-mount depth (spin) axis onto the lateral (car-right) axle line.
 *
 * @param socketId - Socket to transform; signs come from the shared axis table.
 * @param hubHeightM - Hub height in meters; defaults to the normative wheel hub.
 * @returns The car-frame mount transform for the socket.
 */
export function resolveVoxelWheelMountTransform(
  socketId: VoxelWheelSocketId,
  hubHeightM: number = NEATRACING_WHEEL_HUB_HEIGHT_M,
): VoxelWheelMountTransform {
  const [lateralSign, longitudinalSign] = WHEEL_SOCKET_AXIS_SIGNS[socketId];
  const halfWheelbaseM = NEATRACING_WHEEL_WHEELBASE_M / HALF_AXLE_SPAN_DIVISOR;
  const halfTrackM = NEATRACING_WHEEL_TRACK_M / HALF_AXLE_SPAN_DIVISOR;

  return {
    socketId,
    position: [
      lateralSign * halfTrackM,
      hubHeightM,
      longitudinalSign * halfWheelbaseM,
    ],
    pivot: NEATRACING_WHEEL_PIVOT_FRACTIONS,
    rotationYawRad: NEATRACING_WHEEL_MOUNT_YAW_RAD,
    spinAxisPreMount: 'model-depth',
    spinAxisMounted: 'car-lateral',
  };
}

/**
 * Computes the true wheel spin ω = v / r in rad/s from the ground speed,
 * advancing with ground speed (spec §7.4: rotation advances with ground
 * speed, `ω = v / r`). Sign is preserved: positive speed rolls forward
 * (positive ω about the car-right axle under the pinned +π/2 mount), negative
 * speed rolls in reverse. This value is physical truth — feed it to physics;
 * feed only the clamped result of {@link clampVisualWheelSpinOmega} to the
 * display.
 *
 * Safe fallbacks (deterministic total function): a non-finite speed, or a
 * non-finite / non-positive radius, yields 0 (a stopped wheel) instead of
 * NaN or Infinity.
 *
 * @param groundSpeedMPerS - Signed ground speed in m/s (positive = forward).
 * @param wheelRadiusM - Wheel radius in meters; must be positive and finite.
 * @returns The true spin ω in rad/s, or 0 for unsafe inputs.
 */
export function trueWheelSpinOmega(
  groundSpeedMPerS: number,
  wheelRadiusM: number,
): number {
  if (
    !Number.isFinite(groundSpeedMPerS) ||
    !Number.isFinite(wheelRadiusM) ||
    wheelRadiusM <= 0
  ) {
    return 0;
  }
  return groundSpeedMPerS / wheelRadiusM;
}

/**
 * Clamps the true wheel spin to the visual strobing cap (G9): the displayed
 * ω never exceeds {@link NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S}
 * (2 rev/s). Sign is preserved so a reversing wheel still visibly reverses,
 * inputs at or below the cap pass through unchanged, and non-finite inputs
 * yield 0. Deterministic: a pure function of the input ω. Never feed the
 * clamped value to physics — pass true ω there (the cap is display-only and
 * applied after the true ω is computed, per spec §7.4).
 *
 * @param trueOmegaRadPerS - Signed true spin ω in rad/s from
 * {@link trueWheelSpinOmega}.
 * @returns The display-safe spin ω in rad/s, capped at 2 rev/s.
 */
export function clampVisualWheelSpinOmega(trueOmegaRadPerS: number): number {
  if (!Number.isFinite(trueOmegaRadPerS)) {
    return 0;
  }
  const magnitude = Math.abs(trueOmegaRadPerS);
  if (magnitude <= NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S) {
    return trueOmegaRadPerS;
  }
  return (
    Math.sign(trueOmegaRadPerS) * NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S
  );
}

/**
 * Advances the display wheel spin for one batch of stepped physics ticks:
 * the true ω (v / r) is clamped to the 2 rev/s visual strobing cap first,
 * then the clamped omega integrates over `tickCountDelta` timesteps of
 * `timestepSeconds` each. The frame composer accumulates the returned
 * deltas so the displayed spin stays smooth and never strobes past the
 * cap (AC-015), while physics keeps the true omega (spec §7.4).
 *
 * Safe fallbacks (deterministic total function): a non-finite ground speed
 * or radius yields the clamped zero omega, so the delta collapses to 0;
 * a non-positive tick delta returns 0 without touching the accumulator.
 *
 * @param tickCountDelta - Number of physics ticks elapsed (non-negative).
 * @param timestepSeconds - Fixed timestep in seconds (0.5³ world seconds per
 *   tick; 1/60 for the 60 Hz race loop).
 * @param groundSpeedMPerS - Signed ground speed in m/s (positive = forward).
 * @param wheelRadiusM - Wheel radius in meters; must be positive and finite.
 * @returns The display spin delta in radians for this tick batch.
 */
export function computeVisualWheelSpinDeltaRad(
  tickCountDelta: number,
  timestepSeconds: number,
  groundSpeedMPerS: number,
  wheelRadiusM: number,
): number {
  if (!(tickCountDelta > 0)) {
    return 0;
  }
  const omega = clampVisualWheelSpinOmega(
    trueWheelSpinOmega(groundSpeedMPerS, wheelRadiusM),
  );
  return tickCountDelta * timestepSeconds * omega;
}
