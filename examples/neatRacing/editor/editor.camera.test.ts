/**
 * Red-first contract tests for the NeatRacing editor camera (slice P1-S5-D;
 * spec section 13 AC-019 + SL-28 in `plans/neatracing-voxel-racer.spec.md`).
 *
 * Locks the camera contract: orbit click-drag deltas accumulate on the target
 * pose while the smoothed pose stays still (inertial easing), orbit pitch is
 * clamped at the named CAD limits while horizontal yaw stays free, scroll
 * zoom scales the orbit distance exponentially per wheel line and clamps at
 * the configured minimum/maximum, each `tick(dt)` moves the smoothed pose one
 * deterministic exponential step toward the target pose and converges after
 * sustained ticks, CAD-style orthographic front/side/top views snap to exact
 * axis alignments in orthographic mode, dragging exits the CAD snap, and the
 * view/projection builders emit column-major matrices matching the game
 * engine's `uViewProjection` convention (Y-up world, column-major floats).
 *
 * The module is pure numeric state — no DOM, no WebGL context, and no RNG —
 * so every contract is exercised through the `createEditorCamera` factory in
 * the plain Node Jest environment; determinism here is same-runtime exact.
 */
import { describe, expect, it } from '@jest/globals';
import {
  createEditorCamera,
  NEATRACING_EDITOR_CAMERA_DEFAULTS,
  NEATRACING_EDITOR_CAMERA_MAX_DISTANCE,
  NEATRACING_EDITOR_CAMERA_MAX_ORBIT_PITCH_RAD,
  NEATRACING_EDITOR_CAMERA_MIN_DISTANCE,
  NEATRACING_EDITOR_CAMERA_ORTHO_HEIGHT_PER_DISTANCE,
  NEATRACING_EDITOR_CAMERA_PAN_FRACTION_OF_DISTANCE,
  NEATRACING_EDITOR_CAMERA_SMOOTHING_TAU_S,
  NEATRACING_EDITOR_CAMERA_ZOOM_LOG_PER_LINE,
} from './editor.camera';

/** Number of fractional digits used by the matrix closeness helper. */
const EXPECTED_MATRIX_PRECISION_DIGITS = 12;

/**
 * Asserts a column-major matrix matches element values at Float32 storage
 * precision: each expected element is pre-rounded through `Float32Array`,
 * because the builders return the game engine's column-major
 * `Float32Array(16)` whose entries are float32 roundings of the exact
 * float64 math — then compares at the configured digit precision.
 *
 * @param actual - Matrix returned by the camera builders.
 * @param expected - Element values in column-major order.
 */
function expectMatrixCloseTo(actual: Float32Array, expected: number[]): void {
  expected.forEach((entry, index) => {
    const expectedFloat32 = new Float32Array([entry])[0];
    expect(actual[index]).toBeCloseTo(
      expectedFloat32,
      EXPECTED_MATRIX_PRECISION_DIGITS,
    );
  });
}

describe('editor.camera (P1-S5-D; spec AC-019)', () => {
  describe('orbit deltas', () => {
    it('applies orbit deltas to the target pose without moving the smoothed pose', () => {
      const camera = createEditorCamera();
      camera.orbit(0.5, -0.2);
      const afterOrbit = camera.snapshot();
      expect(afterOrbit.target.yaw).toBeCloseTo(
        0.5,
        EXPECTED_MATRIX_PRECISION_DIGITS,
      );
      expect(afterOrbit.target.pitch).toBeCloseTo(
        NEATRACING_EDITOR_CAMERA_DEFAULTS.pitch - 0.2,
        EXPECTED_MATRIX_PRECISION_DIGITS,
      );
      expect(afterOrbit.current.yaw).toBe(
        NEATRACING_EDITOR_CAMERA_DEFAULTS.yaw,
      );
    });

    it('clamps orbit pitch at the named limits for overshooting deltas', () => {
      const camera = createEditorCamera();
      camera.orbit(0, 99);
      expect(camera.snapshot().target.pitch).toBe(
        NEATRACING_EDITOR_CAMERA_MAX_ORBIT_PITCH_RAD,
      );
      camera.orbit(0, -198);
      expect(camera.snapshot().target.pitch).toBe(
        -NEATRACING_EDITOR_CAMERA_MAX_ORBIT_PITCH_RAD,
      );
    });

    it('accumulates free horizontal yaw without clamping', () => {
      const camera = createEditorCamera();
      camera.orbit(100, 0);
      expect(camera.snapshot().target.yaw).toBeCloseTo(
        NEATRACING_EDITOR_CAMERA_DEFAULTS.yaw + 100,
        EXPECTED_MATRIX_PRECISION_DIGITS,
      );
    });
  });

  describe('zoom clamp', () => {
    it('scales the orbit distance exponentially per wheel line', () => {
      const camera = createEditorCamera();
      camera.zoom(1);
      expect(camera.snapshot().target.distance).toBeCloseTo(
        NEATRACING_EDITOR_CAMERA_DEFAULTS.distance *
          Math.exp(NEATRACING_EDITOR_CAMERA_ZOOM_LOG_PER_LINE),
        EXPECTED_MATRIX_PRECISION_DIGITS,
      );
    });

    it('clamps distance at the configured minimum on massive zoom-in', () => {
      const camera = createEditorCamera();
      camera.zoom(-1000);
      expect(camera.snapshot().target.distance).toBe(
        NEATRACING_EDITOR_CAMERA_MIN_DISTANCE,
      );
    });

    it('clamps distance at the configured maximum on massive zoom-out', () => {
      const camera = createEditorCamera();
      camera.zoom(1000);
      expect(camera.snapshot().target.distance).toBe(
        NEATRACING_EDITOR_CAMERA_MAX_DISTANCE,
      );
    });
  });

  describe('inertial smoothing', () => {
    it('eases the smoothed pose one deterministic exponential step per tick', () => {
      const camera = createEditorCamera();
      camera.orbit(0.5, 0);
      camera.tick(0.16);
      const eased =
        0.5 * (1 - Math.exp(-0.16 / NEATRACING_EDITOR_CAMERA_SMOOTHING_TAU_S));
      expect(camera.snapshot().current.yaw).toBeCloseTo(
        eased,
        EXPECTED_MATRIX_PRECISION_DIGITS,
      );
    });

    it('converges the smoothed pose onto the target pose after sustained ticks', () => {
      const camera = createEditorCamera();
      camera.orbit(0.5, -0.2);
      camera.tick(1e9);
      const converged = camera.snapshot();
      expect(converged.current.yaw).toBe(converged.target.yaw);
      expect(converged.current.pitch).toBe(converged.target.pitch);
    });

    it('leaves the smoothed pose unchanged on non-positive dt', () => {
      const camera = createEditorCamera();
      camera.orbit(0.5, 0);
      camera.tick(0);
      camera.tick(-5);
      expect(camera.snapshot().current.yaw).toBe(
        NEATRACING_EDITOR_CAMERA_DEFAULTS.yaw,
      );
    });
  });

  describe('CAD orthographic views', () => {
    it('snaps the front view to exact axis alignment in orthographic mode', () => {
      const camera = createEditorCamera();
      camera.setOrthoView('front');
      const snapped = camera.snapshot();
      expect(snapped.orthoView).toBe('front');
      expect(snapped.target.yaw).toBe(0);
      expect(snapped.target.pitch).toBe(0);
    });

    it('snaps the side view to the +X camera alignment', () => {
      const camera = createEditorCamera();
      camera.setOrthoView('side');
      const snapped = camera.snapshot();
      expect(snapped.orthoView).toBe('side');
      expect(snapped.target.yaw).toBeCloseTo(
        NEATRACING_EDITOR_CAMERA_DEFAULTS.yaw + Math.PI / 2,
        EXPECTED_MATRIX_PRECISION_DIGITS,
      );
      expect(snapped.target.pitch).toBe(0);
    });

    it('snaps the top view to the exact top-down pitch', () => {
      const camera = createEditorCamera();
      camera.setOrthoView('top');
      const snapped = camera.snapshot();
      expect(snapped.orthoView).toBe('top');
      expect(snapped.target.pitch).toBe(Math.PI / 2);
      expect(snapped.target.yaw).toBe(0);
    });

    it('returns to perspective mode on orbit drag', () => {
      const camera = createEditorCamera();
      camera.setOrthoView('side');
      camera.orbit(0.1, 0);
      const dragged = camera.snapshot();
      expect(dragged.mode).toBe('perspective');
      expect(dragged.orthoView).toBeNull();
    });
  });

  describe('matrices', () => {
    it('builds the column-major front-view look-at matrix', () => {
      const camera = createEditorCamera();
      camera.setOrthoView('front');
      camera.tick(1e9);
      const view = camera.buildViewMatrix();
      expect(view).toEqual(
        new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -192, 1]),
      );
    });

    it('builds the column-major side-view look-at matrix', () => {
      const camera = createEditorCamera();
      camera.setOrthoView('side');
      camera.tick(1e9);
      const view = camera.buildViewMatrix();
      expect(view).toEqual(
        new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, -192, 1]),
      );
    });

    it('builds a centered ortho projection from the current distance', () => {
      const camera = createEditorCamera();
      camera.setOrthoView('front');
      camera.tick(1e9);
      const halfHeight =
        NEATRACING_EDITOR_CAMERA_DEFAULTS.distance *
        NEATRACING_EDITOR_CAMERA_ORTHO_HEIGHT_PER_DISTANCE;
      const projection = camera.buildProjectionMatrix(1);
      expectMatrixCloseTo(projection, [
        1 / halfHeight,
        0,
        0,
        0,
        0,
        1 / halfHeight,
        0,
        0,
        0,
        0,
        -2 /
          (NEATRACING_EDITOR_CAMERA_DEFAULTS.far -
            NEATRACING_EDITOR_CAMERA_DEFAULTS.near),
        0,
        0,
        0,
        -(
          NEATRACING_EDITOR_CAMERA_DEFAULTS.far +
          NEATRACING_EDITOR_CAMERA_DEFAULTS.near
        ) /
          (NEATRACING_EDITOR_CAMERA_DEFAULTS.far -
            NEATRACING_EDITOR_CAMERA_DEFAULTS.near),
        1,
      ]);
    });

    it('builds a perspective projection for perspective mode', () => {
      const camera = createEditorCamera();
      const projection = camera.buildProjectionMatrix(2);
      const focal =
        1 / Math.tan(NEATRACING_EDITOR_CAMERA_DEFAULTS.fovYRadians / 2);
      const near = NEATRACING_EDITOR_CAMERA_DEFAULTS.near;
      const far = NEATRACING_EDITOR_CAMERA_DEFAULTS.far;
      expectMatrixCloseTo(projection, [
        focal / 2,
        0,
        0,
        0,
        0,
        focal,
        0,
        0,
        0,
        0,
        (far + near) / (near - far),
        -1,
        0,
        0,
        (2 * far * near) / (near - far),
        0,
      ]);
    });
  });

  describe('screen-plane pan', () => {
    it('shifts the orbit center opposite the drag along the camera basis', () => {
      const camera = createEditorCamera();
      camera.pan(0.5, 0);
      const panned = camera.snapshot();
      const shift =
        0.5 *
        NEATRACING_EDITOR_CAMERA_PAN_FRACTION_OF_DISTANCE *
        NEATRACING_EDITOR_CAMERA_DEFAULTS.distance;
      expect(panned.target.center[0]).toBeCloseTo(
        -shift,
        EXPECTED_MATRIX_PRECISION_DIGITS,
      );
      expect(panned.target.center[1]).toBe(0);
    });

    it('pans without degeneracy while snapped to the top view', () => {
      const camera = createEditorCamera();
      camera.setOrthoView('top');
      camera.tick(1e9);
      camera.pan(0.25, 0);
      const panned = camera.snapshot();
      expect(Number.isFinite(panned.target.center[0])).toBe(true);
      expect(Number.isFinite(panned.target.center[2])).toBe(true);
    });
  });
});
