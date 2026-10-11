/**
 * Contract tests for the wheel mount contract (slice P1-S4-C; spec §7.4
 * "G9 — pinned mount contract", §7.3.4 world-scale derivation, AC-015, LD-21,
 * TP-14; plan AC: wheel sockets attach at spec-defined offsets and the visual
 * rotation is clamped at `NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S = 2`).
 *
 * Probe constants pin spec numbers (320 km/h strobe example, 60 fps display
 * cadence from §7.4) or spec-named module constants; every `it` asserts one
 * mount/spin contract.
 */
import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S,
  NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S,
  NEATRACING_WHEEL_GRID_SIZE,
  NEATRACING_WHEEL_HUB_HEIGHT_M,
  NEATRACING_WHEEL_MOUNT_YAW_RAD,
  NEATRACING_WHEEL_PIVOT_FRACTIONS,
  NEATRACING_WHEEL_PIVOT_VOXEL,
  NEATRACING_WHEEL_TRACK_M,
  NEATRACING_WHEEL_WHEELBASE_M,
  NEATRACING_WHEEL_WORLD_RADIUS_M,
  NEATRACING_WHEEL_WORLD_SIZE_M,
  NEATRACING_WHEEL_VOXEL_WORLD_SIZE,
  clampVisualWheelSpinOmega,
  computeVisualWheelSpinDeltaRad,
  resolveVoxelWheelMetrics,
  resolveVoxelWheelMountTransform,
  resolveVoxelWheelSockets,
  trueWheelSpinOmega,
  VoxelWheelFormatError,
} from './wheel.mount.utils';
import { NEATRACING_VOXEL_WORLD_SIZE } from './lod.distance.utils';
import type {
  VoxelModelDocument,
  VoxelModelMeta,
  VoxelPaletteEntry,
} from './model.schema';

/** Decimal precision for floating-point meter probes. */
const METER_PRECISION = 12;

/** Spec §7.4 strobe example: near 320 km/h a ~0.33 m wheel spins ~270 rad/s. */
const SPEC_STROBE_SPEED_KMH = 320;
const SPEC_STROBE_WHEEL_RADIUS_M = 0.33;

/** Display cadence assumed by the spec §7.4 strobe math (60 fps). */
const SPEC_DISPLAY_FRAME_RATE_HZ = 60;

/** A wheel document whose declared dims deviate from the TP-14 32-cubed grid. */
const WRONG_DIMS_WHEEL_META: VoxelModelMeta = {
  name: 'wheel-wrong',
  dims: [64, 64, 64],
  symmetric: false,
};

/**
 * Builds a minimal normative 32-cubed wheel document (shape-only payload;
 * row/RLE/palette content stays owned by `./model.schema`, not this contract).
 */
function buildNormativeWheelDocument(): VoxelModelDocument {
  return {
    meta: { name: 'wheel-f1', dims: [32, 32, 32], symmetric: false },
    palette: [
      [0, 0, 0, 0, 0],
      [20, 20, 20, 255, 0],
    ] as VoxelPaletteEntry[],
    slices: Array.from({ length: 32 }, () =>
      Array.from({ length: 32 }, () => '0201'),
    ),
  };
}

/** Rotates an axis vector by yawRad around the car's up (+y) axis. */
function rotateAxisAroundCarUp(
  axis: readonly [number, number, number],
  yawRad: number,
): [number, number, number] {
  const [x, y, z] = axis;
  return [
    x * Math.cos(yawRad) + z * Math.sin(yawRad),
    y,
    -x * Math.sin(yawRad) + z * Math.cos(yawRad),
  ];
}

describe('wheel scale format (spec §7.4 / TP-14; AC-015)', () => {
  it('keeps the TP-14 normative wheel grid at 32 voxels per side', () => {
    expect(NEATRACING_WHEEL_GRID_SIZE).toBe(32);
  });

  it('shares the car voxel world scale for wheel models', () => {
    expect(NEATRACING_WHEEL_VOXEL_WORLD_SIZE).toBe(NEATRACING_VOXEL_WORLD_SIZE);
  });

  it('pins the wheel world diameter at the TP-14 grid × shared world size', () => {
    expect(NEATRACING_WHEEL_WORLD_SIZE_M).toBeCloseTo(1.125, METER_PRECISION);
  });

  it('pins the wheel world radius at half the disc diameter', () => {
    expect(NEATRACING_WHEEL_WORLD_RADIUS_M).toBeCloseTo(
      NEATRACING_WHEEL_WORLD_SIZE_M / 2,
      METER_PRECISION,
    );
  });

  it('derives the disc radius from the declared height axis at the shared scale', () => {
    const metrics = resolveVoxelWheelMetrics(buildNormativeWheelDocument());
    expect(metrics.discRadiusM).toBeCloseTo(0.5625, METER_PRECISION);
  });

  it('rejects a wheel document deviating from the normative wheel grid', () => {
    const deviating = {
      ...buildNormativeWheelDocument(),
      meta: WRONG_DIMS_WHEEL_META,
    };
    expect(() => resolveVoxelWheelMetrics(deviating)).toThrow(
      VoxelWheelFormatError,
    );
  });

  it('rejects a wheel document whose slice count deviates from the grid depth', () => {
    const truncated = {
      ...buildNormativeWheelDocument(),
      slices: Array.from({ length: 31 }, () =>
        Array.from({ length: 32 }, () => '0201'),
      ),
    };
    expect(() => resolveVoxelWheelMetrics(truncated)).toThrow(
      VoxelWheelFormatError,
    );
  });

  it('rejects a wheel document missing its palette', () => {
    const paletteless = {
      ...buildNormativeWheelDocument(),
      palette: undefined as unknown as VoxelPaletteEntry[],
    };
    expect(() => resolveVoxelWheelMetrics(paletteless)).toThrow(
      VoxelWheelFormatError,
    );
  });

  it('names wheel-format rejections with the stable voxel-wheel/format code', () => {
    const deviating = {
      ...buildNormativeWheelDocument(),
      meta: WRONG_DIMS_WHEEL_META,
    };
    try {
      resolveVoxelWheelMetrics(deviating);
      throw new Error('expected VoxelWheelFormatError');
    } catch (error) {
      expect(error).toBeInstanceOf(VoxelWheelFormatError);
      expect((error as VoxelWheelFormatError).code).toBe('voxel-wheel/format');
    }
  });
});

describe('hub height and contact patch (spec §7.4 G9)', () => {
  it('mounts wheels at hub height equal to the wheel world radius', () => {
    expect(NEATRACING_WHEEL_HUB_HEIGHT_M).toBe(NEATRACING_WHEEL_WORLD_RADIUS_M);
  });

  it('sits the normative wheel contact patch exactly on the ground plane', () => {
    expect(
      NEATRACING_WHEEL_HUB_HEIGHT_M - NEATRACING_WHEEL_WORLD_RADIUS_M,
    ).toBe(0);
  });

  it('keeps the derived hub height equal to the derived disc radius', () => {
    const metrics = resolveVoxelWheelMetrics(buildNormativeWheelDocument());
    expect(metrics.hubHeightM).toBe(metrics.discRadiusM);
  });
});

describe('disc pivot (spec §7.4 G9: pivot = disc geometric center)', () => {
  it('pivots at the disc geometric center in voxel coordinates', () => {
    expect(NEATRACING_WHEEL_PIVOT_VOXEL).toBe(NEATRACING_WHEEL_GRID_SIZE / 2);
    expect(NEATRACING_WHEEL_PIVOT_VOXEL).toBe(16);
  });

  it('pivots at the normalized grid-center fractions', () => {
    expect(NEATRACING_WHEEL_PIVOT_FRACTIONS).toEqual([0.5, 0.5, 0.5]);
    expect(resolveVoxelWheelMetrics(buildNormativeWheelDocument()).pivot).toBe(
      NEATRACING_WHEEL_PIVOT_FRACTIONS,
    );
  });
});

describe('90° mount (spec §7.4 G9 authoring frame)', () => {
  it('rotates the wheel mount by exactly +90° around car up', () => {
    expect(NEATRACING_WHEEL_MOUNT_YAW_RAD).toBe(Math.PI / 2);
  });

  it('maps the pre-mount depth spin axis onto the lateral car-right axis', () => {
    const mounted = rotateAxisAroundCarUp(
      [0, 0, 1],
      NEATRACING_WHEEL_MOUNT_YAW_RAD,
    );
    expect(mounted[0]).toBeCloseTo(1, METER_PRECISION);
    expect(mounted[1]).toBe(0);
    expect(mounted[2]).toBeCloseTo(0, METER_PRECISION);
  });

  it('labels the spin axis depth before mounting and lateral after mounting', () => {
    const transform = resolveVoxelWheelMountTransform('front-left');
    expect(transform.spinAxisPreMount).toBe('model-depth');
    expect(transform.spinAxisMounted).toBe('car-lateral');
  });
});

describe('four sockets at spec-defined offsets (plan AC; spec §7.4 G9)', () => {
  it('exposes exactly the four sockets in the pinned deterministic order', () => {
    const sockets = resolveVoxelWheelSockets();
    expect(sockets).toHaveLength(4);
    expect(sockets.map((socket) => socket.id)).toEqual([
      'front-left',
      'front-right',
      'rear-left',
      'rear-right',
    ]);
  });

  it('mounts every socket at the hub height so the contact patch sits at y = 0', () => {
    for (const socket of resolveVoxelWheelSockets()) {
      expect(socket.offset[1]).toBe(NEATRACING_WHEEL_HUB_HEIGHT_M);
    }
  });

  it('mirrors sockets at half the wheelbase and half the track offsets', () => {
    const halfTrackM = NEATRACING_WHEEL_TRACK_M / 2;
    const halfWheelbaseM = NEATRACING_WHEEL_WHEELBASE_M / 2;
    const hubHeightM = NEATRACING_WHEEL_HUB_HEIGHT_M;
    const [frontLeft, frontRight, rearLeft, rearRight] =
      resolveVoxelWheelSockets();
    expect(frontLeft.offset).toEqual([-halfTrackM, hubHeightM, halfWheelbaseM]);
    expect(frontRight.offset).toEqual([halfTrackM, hubHeightM, halfWheelbaseM]);
    expect(rearLeft.offset).toEqual([-halfTrackM, hubHeightM, -halfWheelbaseM]);
    expect(rearRight.offset).toEqual([halfTrackM, hubHeightM, -halfWheelbaseM]);
  });

  it('pins the spec-unpinned F1-style wheelbase and track tuning constants', () => {
    expect(NEATRACING_WHEEL_WHEELBASE_M).toBe(3.2);
    expect(NEATRACING_WHEEL_TRACK_M).toBe(2);
  });

  it('honors a hub-height override while keeping the socket layout offsets', () => {
    const [override] = resolveVoxelWheelSockets(0.5);
    expect(override).toBeDefined();
    expect(override?.offset[1]).toBe(0.5);
    expect(override?.offset[2]).toBeCloseTo(
      NEATRACING_WHEEL_WHEELBASE_M / 2,
      METER_PRECISION,
    );
  });
});

describe('mount transform per socket (plan AC: sockets attach at offsets; G9)', () => {
  it('composes hub height, grid-center pivot and the +90° yaw per socket', () => {
    for (const socket of resolveVoxelWheelSockets()) {
      const transform = resolveVoxelWheelMountTransform(socket.id);
      expect(transform.position).toEqual(socket.offset);
      expect(transform.pivot).toEqual(NEATRACING_WHEEL_PIVOT_FRACTIONS);
      expect(transform.rotationYawRad).toBe(NEATRACING_WHEEL_MOUNT_YAW_RAD);
    }
  });

  it('keeps the transform contact patch at y = 0 for the normative wheel', () => {
    const transform = resolveVoxelWheelMountTransform('rear-right');
    expect(transform.position[1] - NEATRACING_WHEEL_WORLD_RADIUS_M).toBe(0);
  });
});

describe('true spin advances with ground speed (spec §7.4: ω = v / r)', () => {
  it('advances true spin ω with ground speed as ω = v / r', () => {
    const omega = trueWheelSpinOmega(10, NEATRACING_WHEEL_WORLD_RADIUS_M);
    expect(omega).toBeGreaterThan(0);
    expect(Math.abs(omega * NEATRACING_WHEEL_WORLD_RADIUS_M - 10)).toBeLessThan(
      1e-9,
    );
  });

  it('rolls in reverse for negative ground speed', () => {
    const omega = trueWheelSpinOmega(-10, NEATRACING_WHEEL_WORLD_RADIUS_M);
    expect(omega).toBeLessThan(0);
    expect(Math.abs(omega * NEATRACING_WHEEL_WORLD_RADIUS_M + 10)).toBeLessThan(
      1e-9,
    );
  });

  it('stops the wheel for zero ground speed', () => {
    expect(trueWheelSpinOmega(0, NEATRACING_WHEEL_WORLD_RADIUS_M)).toBe(0);
  });

  it('yields a deterministic no-spin fallback for non-finite or invalid inputs', () => {
    expect(
      trueWheelSpinOmega(Number.NaN, NEATRACING_WHEEL_WORLD_RADIUS_M),
    ).toBe(0);
    expect(
      trueWheelSpinOmega(
        Number.POSITIVE_INFINITY,
        NEATRACING_WHEEL_WORLD_RADIUS_M,
      ),
    ).toBe(0);
    expect(trueWheelSpinOmega(10, 0)).toBe(0);
  });
});

describe('visual rev clamp (spec §7.4 G9 strobing cap; plan AC: cap = 2 rev/s)', () => {
  it('pins the visual strobing cap at 2 revolutions per second', () => {
    expect(NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S).toBe(2);
  });

  it('exposes the cap in rad/s as 2 rev/s × 2π', () => {
    expect(NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S).toBe(
      NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S * 2 * Math.PI,
    );
  });

  it('passes true ω through unchanged below and exactly at the cap', () => {
    expect(clampVisualWheelSpinOmega(2 * Math.PI)).toBe(2 * Math.PI);
    expect(
      clampVisualWheelSpinOmega(NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S),
    ).toBe(NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S);
  });

  it('clamps over-cap true ω to exactly the 2 rev/s cap', () => {
    const clampedOmega = clampVisualWheelSpinOmega(270);
    expect(clampedOmega).toBe(NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S);
    expect(clampedOmega / (2 * Math.PI)).toBe(2);
  });

  it('preserves spin sign under clamping', () => {
    expect(clampVisualWheelSpinOmega(-270)).toBe(
      -NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S,
    );
  });

  it('yields a deterministic zero for non-finite true ω', () => {
    expect(clampVisualWheelSpinOmega(Number.NaN)).toBe(0);
    expect(clampVisualWheelSpinOmega(Number.POSITIVE_INFINITY)).toBe(0);
  });

  it('keeps the clamped display spin decoupled from physics over the cap', () => {
    const trueOmega = trueWheelSpinOmega(
      320 / 3.6,
      NEATRACING_WHEEL_WORLD_RADIUS_M,
    );
    const clampedOmega = clampVisualWheelSpinOmega(trueOmega);
    expect(clampedOmega).not.toBe(trueOmega);
    expect(trueOmega).toBeGreaterThan(clampedOmega);
  });

  it('stops the spec strobe example below the half-frame backward-alias floor', () => {
    const trueOmega = trueWheelSpinOmega(
      SPEC_STROBE_SPEED_KMH / 3.6,
      SPEC_STROBE_WHEEL_RADIUS_M,
    );
    const unclampedPerFrameDeg =
      ((trueOmega / SPEC_DISPLAY_FRAME_RATE_HZ) * 180) / Math.PI;
    const clampedPerFrameDeg =
      ((clampVisualWheelSpinOmega(trueOmega) / SPEC_DISPLAY_FRAME_RATE_HZ) *
        180) /
      Math.PI;
    expect(unclampedPerFrameDeg).toBeGreaterThan(180);
    expect(clampedPerFrameDeg).toBeLessThan(180);
    expect(clampedPerFrameDeg).toBeCloseTo(12, 6);
  });
});

describe('per-tick visual spin delta (Step 14B round-3: capped composer accumulation)', () => {
  it('advances the display spin by the clamped omega times the stepped duration', () => {
    // 100 km/h on the 0.5625 m wheel: true ω = 180.6 rad/s, clamped to 2 rev/s
    // (12.566 rad/s); one 60 Hz tick advances the display by 0.20944 rad.
    const delta = computeVisualWheelSpinDeltaRad(
      1,
      1 / 60,
      100 / 3.6,
      NEATRACING_WHEEL_WORLD_RADIUS_M,
    );
    expect(delta).toBeCloseTo(
      NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S / SPEC_DISPLAY_FRAME_RATE_HZ,
      9,
    );
  });

  it('aggregates proportional deltas for catch-up tick batches', () => {
    const delta = computeVisualWheelSpinDeltaRad(
      7,
      1 / 60,
      100 / 3.6,
      NEATRACING_WHEEL_WORLD_RADIUS_M,
    );
    expect(delta).toBeCloseTo(
      (7 * NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S) /
        SPEC_DISPLAY_FRAME_RATE_HZ,
      9,
    );
  });

  it('returns zero without ticking or for zero ground speed', () => {
    expect(
      computeVisualWheelSpinDeltaRad(
        0,
        1 / 60,
        50,
        NEATRACING_WHEEL_WORLD_RADIUS_M,
      ),
    ).toBe(0);
    expect(
      computeVisualWheelSpinDeltaRad(
        3,
        1 / 60,
        0,
        NEATRACING_WHEEL_WORLD_RADIUS_M,
      ),
    ).toBe(0);
  });

  it('advances in reverse under braking so the tire visibly counter-rotates', () => {
    // Sub-cap speed so the true omega passes the clamp unchanged (the cap
    // magnitude is asserted separately by the omega clamp tests above).
    const delta = computeVisualWheelSpinDeltaRad(
      2,
      1 / 60,
      -6,
      NEATRACING_WHEEL_WORLD_RADIUS_M,
    );
    expect(delta).toBeLessThan(0);
    expect(delta).toBeCloseTo(
      (2 / 60) * (-6 / NEATRACING_WHEEL_WORLD_RADIUS_M),
      9,
    );
  });
});
