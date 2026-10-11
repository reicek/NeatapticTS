/**
 * Editor camera controls for NeatRacing (slice P1-S5-D; spec section 13
 * AC-019, editor navigation SL-28 of `plans/neatracing-voxel-racer.spec.md`):
 * orbit on drag, scroll zoom with clamped distance, CAD-style orthographic
 * front/side/top views, screen-plane pan, and inertial exponential smoothing
 * between ticks.
 *
 * The module stays pure numeric state — it touches no DOM and no WebGL
 * context — so the state is testable inside the plain Node Jest environment
 * and the future editor shell (slice P1-S5-C) only feeds drag deltas, wheel
 * lines, and ticks into {@linkcode EditorCamera} and uploads the built
 * matrices into the game renderer's combined view-projection uniform slot.
 *
 * Conventions match the game engine passes the editor re-implements nothing
 * from: matrices are column-major 4x4 `Float32Array(16)` and the world is
 * Y-up, so the look-at basis is right = normalize(forward × worldUp), the
 * camera up frame completes the right-handed basis, and view space looks
 * down −Z. Camera yaw/pitch orbit a focus `center` on a spherical shell of
 * radius `distance`, exactly like the CAD panels the spec calls out in
 * SL-28, and every mutation lands on a *target* pose that
 * {@linkcode EditorCamera.tick} eases the *current* pose toward — the
 * inertial smoothing the AC demands. CAD orthographic snaps bypass the
 * orbit pitch clamp BY DESIGN: the top view pitch (π/2) legitimately points
 * straight down, which free-drag orbiting must never reach.
 */

/** Smallest camera distance in world units before zoom clamps (AC-019). */
export const NEATRACING_EDITOR_CAMERA_MIN_DISTANCE = 16;

/** Largest camera distance in world units before zoom clamps (AC-019). */
export const NEATRACING_EDITOR_CAMERA_MAX_DISTANCE = 1024;

/**
 * Free-drag orbit pitch clamp in radians (±80°) — CAD orthographic snaps
 * bypass this clamp on purpose so the top view can point straight down.
 */
export const NEATRACING_EDITOR_CAMERA_MAX_ORBIT_PITCH_RAD = 1.3962634015954636;

/**
 * Inertial smoothing time constant in seconds: each `tick(dtSeconds)` moves
 * the current pose by `ease = 1 − exp(−dt ÷ τ)` of the gap to the target
 * pose, the deterministic exponential ease the AC-019 tests lock in.
 */
export const NEATRACING_EDITOR_CAMERA_SMOOTHING_TAU_S = 0.08;

/**
 * Log-scale zoom per scroll-wheel line: the orbit distance multiplies by
 * `exp(k · lines)`, so positive lines zoom out and negative lines zoom in.
 */
export const NEATRACING_EDITOR_CAMERA_ZOOM_LOG_PER_LINE = 0.125;

/** Pan speed as a fraction of the current orbit distance per screen unit. */
export const NEATRACING_EDITOR_CAMERA_PAN_FRACTION_OF_DISTANCE = 1.6;

/** Ortho frustum half-height as a fraction of the current orbit distance. */
export const NEATRACING_EDITOR_CAMERA_ORTHO_HEIGHT_PER_DISTANCE = 0.35;

/** Trig values with magnitude below this snap to exact zero (CAD alignment). */
const NEATRACING_TRIG_SNAP_EPSILON = 1e-12;

/** Squared-length threshold under which a camera basis axis is degenerate. */
const NEATRACING_BASIS_DEGENERATE_EPSILON = 1e-18;

/** CAD orthographic projection the editor preview can snap to. */
export type EditorOrthoView = 'front' | 'side' | 'top';

/** Projection mode of the editor camera (AC-019 CAD views vs orbit view). */
export type EditorCameraMode = 'perspective' | 'orthographic';

/** Orbit pose of the camera: focus center, spherical angles, and distance. */
export interface EditorCameraPose {
  /** Focus point (world center) the camera orbits around. */
  center: [number, number, number];
  /** Horizontal orbit angle in radians (0 places the camera on +Z looking down −Z; π/2 places it on +X looking down −X). */
  yaw: number;
  /** Vertical orbit angle in radians (0 is horizon; +π/2 is straight above). */
  pitch: number;
  /** Orbit shell radius in world units (clamped between MIN and MAX). */
  distance: number;
}

/** Immutable camera tuning the factory spreads user overrides over. */
export interface EditorCameraConfig {
  /** Initial focus point. */
  center: [number, number, number];
  /** Initial yaw in radians. */
  yaw: number;
  /** Initial pitch in radians. */
  pitch: number;
  /** Initial orbit distance in world units. */
  distance: number;
  /** Perspective vertical field of view in radians. */
  fovYRadians: number;
  /** Near clip plane distance in world units. */
  near: number;
  /** Far clip plane distance in world units. */
  far: number;
}

/** Point-in-time view of both camera poses plus the active CAD state. */
export interface EditorCameraSnapshot {
  /** Active projection mode. */
  mode: EditorCameraMode;
  /** Active CAD override, or `null` while orbiting freely. */
  orthoView: EditorOrthoView | null;
  /** Smoothed (rendered) pose. */
  current: EditorCameraPose;
  /** Pose the smoothed pose is easing toward. */
  target: EditorCameraPose;
}

/** Public editor camera surface the canvas shell drives per frame. */
export interface EditorCamera {
  /**
   * Applies an orbit drag: deltas land on the target pose, clamp the pitch,
   * and exit any CAD orthographic snap back toward perspective mode.
   *
   * @param deltaYawRadians - Horizontal drag delta in radians.
   * @param deltaPitchRadians - Vertical drag delta in radians.
   * @example
   * ```ts
   * const camera = createEditorCamera();
   * camera.orbit(0.5, -0.2);
   * ```
   */
  orbit(deltaYawRadians: number, deltaPitchRadians: number): void;

  /**
   * Applies a screen-space pan: the focus center moves opposite the drag
   * along the current camera basis, scaled by the orbit distance, so CAD
   * slicing stays viewport-stable at any zoom. Safe in the degenerate top
   * view via the fallback basis.
   *
   * `dx`/`dy` are signed screen units where positive content-follows-cursor
   * semantics shift the center along `-right·dx − up·dy`.
   *
   * @param dx - Horizontal drag in screen units.
   * @param dy - Vertical drag in screen units.
   * @example
   * ```ts
   * const camera = createEditorCamera();
   * camera.pan(0.5, 0);
   * ```
   */
  pan(dx: number, dy: number): void;

  /**
   * Applies a scroll-wheel zoom: the orbit distance scales exponentially
   * per line and clamps into `[..., MIN, MAX]`. Positive lines zoom out.
   *
   * @param lines - Signed scroll-wheel line count.
   * @example
   * ```ts
   * const camera = createEditorCamera();
   * camera.zoom(-1);
   * ```
   */
  zoom(lines: number): void;

  /**
   * Snaps to a CAD orthographic view with exact axis-aligned angles:
   * front looks down −Z (yaw 0), side looks down −X (yaw π/2), top looks
   * straight down (pitch π/2). The orbit distance is preserved so the
   * CAD slice keeps its current framing.
   *
   * @param view - CAD view to snap to.
   * @example
   * ```ts
   * const camera = createEditorCamera();
   * camera.setOrthoView('front');
   * ```
   */
  setOrthoView(view: EditorOrthoView): void;

  /**
   * Returns to perspective orbit mode, clearing the CAD snap while keeping
   * the current angles.
   *
   * @example
   * ```ts
   * const camera = createEditorCamera();
   * camera.setOrthoView('top');
   * camera.setPerspectiveView();
   * ```
   */
  setPerspectiveView(): void;

  /**
   * Eases the smoothed (rendered) pose one deterministic exponential step
   * toward the target pose. Non-positive or NaN `dtSeconds` is a no-op so
   * paused frames never move the camera.
   *
   * @param dtSeconds - Frame delta in seconds.
   * @example
   * ```ts
   * const camera = createEditorCamera();
   * camera.orbit(0.5, 0);
   * camera.tick(0.016);
   * ```
   */
  tick(dtSeconds: number): void;

  /**
   * Returns an immutable view of both poses plus the active CAD state.
   *
   * @returns Snapshot copy safe to hold across a frame.
   * @example
   * ```ts
   * const camera = createEditorCamera();
   * const snapshot = camera.snapshot();
   * ```
   */
  snapshot(): EditorCameraSnapshot;

  /**
   * Builds the column-major view matrix from the smoothed pose, orthonormal
   * look-at basis, and engine convention (Y-up world, view down −Z).
   *
   * @returns 4x4 column-major view matrix.
   * @example
   * ```ts
   * const camera = createEditorCamera();
   * const view = camera.buildViewMatrix();
   * ```
   */
  buildViewMatrix(): Float32Array;

  /**
   * Builds the column-major projection matrix for the active mode:
   * perspective frustum in perspective mode, centered ortho frustum whose
   * half-height tracks the orbit distance in CAD mode.
   *
   * @param aspect - Width-over-height aspect ratio of the viewport.
   * @returns 4x4 column-major projection matrix.
   * @example
   * ```ts
   * const camera = createEditorCamera();
   * const projection = camera.buildProjectionMatrix(16 / 9);
   * ```
   */
  buildProjectionMatrix(aspect: number): Float32Array;
}

const NEATRACING_EDITOR_CAMERA_DEFAULTS: Readonly<EditorCameraConfig> =
  Object.freeze({
    center: Object.freeze([0, 0, 0]) as [number, number, number],
    yaw: 0,
    pitch: 0.42,
    distance: 192,
    fovYRadians: 1.0471975511965976,
    near: 0.5,
    far: 4096,
  });

export { NEATRACING_EDITOR_CAMERA_DEFAULTS };

interface EditorCameraState {
  config: Readonly<EditorCameraConfig>;
  mode: EditorCameraMode;
  orthoView: EditorOrthoView | null;
  current: EditorCameraPose;
  target: EditorCameraPose;
}

/**
 * Clamps a value into an inclusive range.
 *
 * @param value - Number to clamp.
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 * @returns Clamped number.
 */
function clampValue(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Snaps trig values that fall within float noise of zero onto exact zero so
 * CAD axis-aligned views reproduce exact integer matrices instead of values
 * like `cos(π/2) ≈ 6.1e−17`.
 *
 * @param value - Trig evaluation result.
 * @returns Exact zero when the magnitude is float-noise-small, else `value`.
 */
function snapTrigZero(value: number): number {
  return Math.abs(value) < NEATRACING_TRIG_SNAP_EPSILON ? 0 : value;
}

/**
 * Copies a pose deeply so snapshots and target writes never alias state.
 *
 * @param pose - Pose to copy.
 * @returns Detached copy.
 */
function clonePose(pose: EditorCameraPose): EditorCameraPose {
  return {
    center: [pose.center[0], pose.center[1], pose.center[2]],
    yaw: pose.yaw,
    pitch: pose.pitch,
    distance: pose.distance,
  };
}

/**
 * Crosses two 3-vectors into a new vector.
 *
 * @param a - First vector.
 * @param b - Second vector.
 * @returns `a × b`.
 */
function cross3(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * Normalizes a 3-vector in place into a fresh array.
 *
 * @param vector - Vector to normalize.
 * @returns Unit-length vector, or the input when the length is zero.
 */
function normalize3(
  vector: [number, number, number],
): [number, number, number] {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  if (length === 0) {
    return [vector[0], vector[1], vector[2]];
  }
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

/**
 * Dots two 3-vectors.
 *
 * @param a - First vector.
 * @param b - Second vector.
 * @returns `a · b`.
 */
function dot3(
  a: [number, number, number],
  b: [number, number, number],
): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Creates the editor camera state with defaults plus partial overrides.
 *
 * @param overrides - Partial camera config to spread over the defaults.
 * @returns Ready-to-drive camera implementing {@linkcode EditorCamera}.
 * @example
 * ```ts
 * import { createEditorCamera } from './editor.camera';
 * const camera = createEditorCamera({ distance: 64 });
 * camera.orbit(0.2, -0.1);
 * camera.tick(0.016);
 * ```
 */
export function createEditorCamera(
  overrides?: Partial<EditorCameraConfig>,
): EditorCamera {
  const config: Readonly<EditorCameraConfig> = {
    ...NEATRACING_EDITOR_CAMERA_DEFAULTS,
    ...overrides,
    center: [
      ...(overrides?.center ?? NEATRACING_EDITOR_CAMERA_DEFAULTS.center),
    ] as [number, number, number],
  };
  const state: EditorCameraState = {
    config,
    mode: 'perspective',
    orthoView: null,
    current: clonePose({
      center: [...config.center],
      yaw: config.yaw,
      pitch: config.pitch,
      distance: config.distance,
    }),
    target: clonePose({
      center: [...config.center],
      yaw: config.yaw,
      pitch: config.pitch,
      distance: config.distance,
    }),
  };

  return {
    orbit(deltaYawRadians, deltaPitchRadians): void {
      state.mode = 'perspective';
      state.orthoView = null;
      state.target.yaw += deltaYawRadians;
      state.target.pitch = clampValue(
        state.target.pitch + deltaPitchRadians,
        -NEATRACING_EDITOR_CAMERA_MAX_ORBIT_PITCH_RAD,
        NEATRACING_EDITOR_CAMERA_MAX_ORBIT_PITCH_RAD,
      );
    },

    pan(dx, dy): void {
      const c = state.current;
      const forward: [number, number, number] = [
        -snapTrigZero(Math.cos(c.pitch)) * snapTrigZero(Math.sin(c.yaw)),
        -snapTrigZero(Math.sin(c.pitch)),
        -snapTrigZero(Math.cos(c.pitch)) * snapTrigZero(Math.cos(c.yaw)),
      ];
      let right = cross3(forward, [0, 1, 0]);
      if (
        right[0] ** 2 + right[1] ** 2 + right[2] ** 2 <
        NEATRACING_BASIS_DEGENERATE_EPSILON
      ) {
        right = cross3(forward, [0, 0, -1]);
      }
      const rightUnit = normalize3(right);
      const upUnit = normalize3(cross3(rightUnit, forward));
      const scale =
        NEATRACING_EDITOR_CAMERA_PAN_FRACTION_OF_DISTANCE * c.distance;
      state.target.center[0] -= (rightUnit[0] * dx + upUnit[0] * dy) * scale;
      state.target.center[1] -= (rightUnit[1] * dx + upUnit[1] * dy) * scale;
      state.target.center[2] -= (rightUnit[2] * dx + upUnit[2] * dy) * scale;
    },

    zoom(lines): void {
      const scaled =
        state.target.distance *
        Math.exp(NEATRACING_EDITOR_CAMERA_ZOOM_LOG_PER_LINE * lines);
      state.target.distance = clampValue(
        scaled,
        NEATRACING_EDITOR_CAMERA_MIN_DISTANCE,
        NEATRACING_EDITOR_CAMERA_MAX_DISTANCE,
      );
    },

    setOrthoView(view): void {
      state.mode = 'orthographic';
      state.orthoView = view;
      if (view === 'front') {
        state.target.yaw = 0;
        state.target.pitch = 0;
      } else if (view === 'side') {
        state.target.yaw = Math.PI / 2;
        state.target.pitch = 0;
      } else {
        state.target.pitch = Math.PI / 2;
        state.target.yaw = 0;
      }
    },

    setPerspectiveView(): void {
      state.mode = 'perspective';
      state.orthoView = null;
    },

    tick(dtSeconds): void {
      if (!(dtSeconds > 0)) {
        return;
      }
      const ease =
        1 - Math.exp(-dtSeconds / NEATRACING_EDITOR_CAMERA_SMOOTHING_TAU_S);
      if (ease >= 1) {
        state.current = clonePose(state.target);
        return;
      }
      state.current.center[0] +=
        (state.target.center[0] - state.current.center[0]) * ease;
      state.current.center[1] +=
        (state.target.center[1] - state.current.center[1]) * ease;
      state.current.center[2] +=
        (state.target.center[2] - state.current.center[2]) * ease;
      state.current.yaw += (state.target.yaw - state.current.yaw) * ease;
      state.current.pitch += (state.target.pitch - state.current.pitch) * ease;
      state.current.distance +=
        (state.target.distance - state.current.distance) * ease;
    },

    snapshot(): EditorCameraSnapshot {
      return {
        mode: state.mode,
        orthoView: state.orthoView,
        current: clonePose(state.current),
        target: clonePose(state.target),
      };
    },

    buildViewMatrix(): Float32Array {
      const c = state.current;
      const sinYaw = snapTrigZero(Math.sin(c.yaw));
      const cosYaw = snapTrigZero(Math.cos(c.yaw));
      const sinPitch = snapTrigZero(Math.sin(c.pitch));
      const cosPitch = snapTrigZero(Math.cos(c.pitch));
      const eye: [number, number, number] = [
        c.center[0] + c.distance * cosPitch * sinYaw,
        c.center[1] + c.distance * sinPitch,
        c.center[2] + c.distance * cosPitch * cosYaw,
      ];
      const zAxis = normalize3([
        cosPitch * sinYaw,
        sinPitch,
        cosPitch * cosYaw,
      ]);
      const forward = normalize3([-zAxis[0], -zAxis[1], -zAxis[2]]);
      let right = cross3(forward, [0, 1, 0]);
      if (
        right[0] ** 2 + right[1] ** 2 + right[2] ** 2 <
        NEATRACING_BASIS_DEGENERATE_EPSILON
      ) {
        right = cross3(forward, [0, 0, -1]);
      }
      const xAxis = normalize3(right);
      const yAxis = normalize3(cross3(zAxis, xAxis));
      const matrix = new Float32Array(16);
      matrix[0] = xAxis[0];
      matrix[1] = yAxis[0];
      matrix[2] = zAxis[0];
      matrix[4] = xAxis[1];
      matrix[5] = yAxis[1];
      matrix[6] = zAxis[1];
      matrix[8] = xAxis[2];
      matrix[9] = yAxis[2];
      matrix[10] = zAxis[2];
      matrix[12] = -dot3(xAxis, eye);
      matrix[13] = -dot3(yAxis, eye);
      matrix[14] = -dot3(zAxis, eye);
      matrix[15] = 1;
      // Negated dot products can produce IEEE −0; emit exact +0 so strict
      // typed-array comparisons stay stable across runtimes.
      for (let i = 0; i < matrix.length; i += 1) {
        if (matrix[i] === 0) {
          matrix[i] = 0;
        }
      }
      return matrix;
    },

    buildProjectionMatrix(aspect): Float32Array {
      const matrix = new Float32Array(16);
      const halfHeight =
        state.current.distance *
        NEATRACING_EDITOR_CAMERA_ORTHO_HEIGHT_PER_DISTANCE;
      if (state.mode === 'orthographic') {
        matrix[0] = 1 / (halfHeight * aspect);
        matrix[5] = 1 / halfHeight;
        matrix[10] = -2 / (state.config.far - state.config.near);
        matrix[14] =
          -(state.config.far + state.config.near) /
          (state.config.far - state.config.near);
        matrix[15] = 1;
      } else {
        const focal = 1 / Math.tan(state.config.fovYRadians / 2);
        matrix[0] = focal / aspect;
        matrix[5] = focal;
        matrix[10] =
          (state.config.far + state.config.near) /
          (state.config.near - state.config.far);
        matrix[11] = -1;
        matrix[14] =
          (2 * state.config.far * state.config.near) /
          (state.config.near - state.config.far);
      }
      return matrix;
    },
  };
}
