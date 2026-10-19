/**
 * NeatRacing track grid draw pass (slice P1-grid-footprint, AC-045).
 *
 * This pass wraps the same flat-horizon fullscreen shader already shipped in
 * `../engine/passes/grid.pass.ts` but feeds it the larger track-grid spacing
 * (5.0 m, giving a 25 m² cell footprint area versus the engine grid's
 * 1 m² baseline). Everything else — the depth-disabled draw, the blending
 * mode, the alpha envelope, and the pulse pass — stays unchanged.
 */

import {
  NEATRACING_GRID_FRAGMENT_SHADER,
  NEATRACING_GRID_GLOW_ALPHA_MULTIPLIER,
  NEATRACING_GRID_GLOW_WIDTH_PX,
  NEATRACING_GRID_LINE_COLOR_RGB,
  NEATRACING_GRID_LINE_WIDTH_PX,
  NEATRACING_GRID_MAX_ALPHA,
  NEATRACING_GRID_MIN_ALPHA,
  NEATRACING_GRID_PLANE_Y_OFFSET_M,
  NEATRACING_GRID_VERTEX_SHADER,
  NEATRACING_DRAW_DISTANCE_M,
} from '../engine/passes/grid.pass';

export {
  NEATRACING_TRACK_GRID_BASELINE_CELL_FOOTPRINT_AREA_M2,
  NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_MULTIPLIER,
  NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_M2,
  NEATRACING_TRACK_GRID_SPACING_M,
  buildTrackGridSpacing,
  trackGridLineCountForExtent,
} from './grid.builder';

import type { ShaderWarmupProgram } from '../engine/context.service';

import { buildTrackGridSpacing } from './grid.builder';

/** Warmup/draw registry key for the track grid pass (distinct from engine grid). */
export const NEATRACING_TRACK_GRID_PROGRAM_KEY = 'track-grid';

/** Names of every uniform the track grid pass resolves and sets. */
const TRACK_GRID_UNIFORM_NAMES = [
  'uInvViewProjection',
  'uCameraWorld',
  'uPlaneY',
  'uSpacing',
  'uLineWidthPx',
  'uGlowWidthPx',
  'uGlowAlphaMultiplier',
  'uMinAlpha',
  'uMaxAlpha',
  'uDrawDistance',
  'uLineColor',
] as const;

/** Resolved uniform locations keyed by uniform name. */
type TrackGridUniformLocations = Record<
  (typeof TRACK_GRID_UNIFORM_NAMES)[number],
  WebGLUniformLocation
>;

/** Per-draw inputs for the track grid pass. */
export interface TrackGridPassDrawState {
  /** Column-major 16-element view-projection matrix. */
  readonly viewProjection: Float32Array;
  /** Camera world position `[x, y, z]` in meters. */
  readonly cameraWorld: readonly [number, number, number];
}

/** The track grid pass handle: create once, draw every frame. */
export interface TrackGridPass {
  /** Draws the track grid for the given camera state. */
  draw(state: TrackGridPassDrawState): void;
}

/** `gl.BLEND` enum value (0x0BE2) enabled during the track grid draw. */
const GL_BLEND = 0x0be2;

/** `gl.SRC_ALPHA` enum value (0x0302) for alpha blending. */
const GL_SRC_ALPHA = 0x0302;

/** `gl.ONE_MINUS_SRC_ALPHA` enum value (0x0303) for alpha blending. */
const GL_ONE_MINUS_SRC_ALPHA = 0x0303;

/** `gl.DEPTH_TEST` enum value (0x0B71) disabled during the track grid draw. */
const GL_DEPTH_TEST = 0x0b71;

/**
 * Compiles one shader stage or throws a descriptive error carrying the stage
 * label and the driver info log.
 */
function compileTrackGridShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error(
      `neatRacing track grid pass: createShader returned null for ${label} stage`,
    );
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const infoLog = gl.getShaderInfoLog(shader) ?? 'unknown error';
    throw new Error(
      `neatRacing track grid pass: ${label} shader compile failed: ${infoLog}`,
    );
  }
  return shader;
}

/**
 * Inverts a 4x4 column-major matrix into `out` using the adjugate/determinant
 * method. The source and destination must not alias.
 */
function invertMatrix44(out: Float32Array, m: Float32Array): boolean {
  const a00 = m[0];
  const a01 = m[1];
  const a02 = m[2];
  const a03 = m[3];
  const a10 = m[4];
  const a11 = m[5];
  const a12 = m[6];
  const a13 = m[7];
  const a20 = m[8];
  const a21 = m[9];
  const a22 = m[10];
  const a23 = m[11];
  const a30 = m[12];
  const a31 = m[13];
  const a32 = m[14];
  const a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  const det =
    b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (det === 0) {
    return false;
  }

  const invDet = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * invDet;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * invDet;
  out[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * invDet;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * invDet;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * invDet;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * invDet;
  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * invDet;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * invDet;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;

  return true;
}

/**
 * Creates the track grid pass: compiles and links the reused engine grid
 * program and resolves all uniform locations. The returned handle never
 * recompiles or re-uploads; `draw` only sets uniforms and issues the same
 * fullscreen-triangle draw the engine grid uses, but with the larger
 * track-grid spacing.
 *
 * @param gl - WebGL2 rendering context to bind the pass to.
 * @returns The track grid pass handle.
 */
export function createTrackGridPass(gl: WebGL2RenderingContext): TrackGridPass {
  const vertexShader = compileTrackGridShader(
    gl,
    gl.VERTEX_SHADER,
    NEATRACING_GRID_VERTEX_SHADER,
    'vertex',
  );
  const fragmentShader = compileTrackGridShader(
    gl,
    gl.FRAGMENT_SHADER,
    NEATRACING_GRID_FRAGMENT_SHADER,
    'fragment',
  );
  const program = gl.createProgram();
  if (!program) {
    throw new Error('neatRacing track grid pass: createProgram returned null');
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const infoLog = gl.getProgramInfoLog(program) ?? 'unknown error';
    throw new Error(
      `neatRacing track grid pass: program link failed: ${infoLog}`,
    );
  }

  const uniforms = {} as Partial<TrackGridUniformLocations>;
  for (const name of TRACK_GRID_UNIFORM_NAMES) {
    const location = gl.getUniformLocation(program, name);
    if (!location) {
      throw new Error(
        `neatRacing track grid pass: uniform ${name} not found in program`,
      );
    }
    uniforms[name] = location;
  }
  const uniformLocations = uniforms as TrackGridUniformLocations;
  const invViewProjection = new Float32Array(16);

  return {
    draw(state: TrackGridPassDrawState): void {
      if (state.viewProjection.length !== 16) {
        throw new Error(
          'neatRacing track grid pass draw: viewProjection must be a 16-element column-major matrix',
        );
      }
      if (!invertMatrix44(invViewProjection, state.viewProjection)) {
        throw new Error(
          'neatRacing track grid pass draw: viewProjection matrix is singular',
        );
      }

      // Reuse the engine grid spacing helper from grid.builder.ts.
      const spacing = buildTrackGridSpacing();

      gl.useProgram(program);
      gl.enable(GL_BLEND);
      gl.blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      gl.disable(GL_DEPTH_TEST);
      gl.depthMask(false);
      gl.uniformMatrix4fv(
        uniformLocations.uInvViewProjection,
        false,
        invViewProjection,
      );
      gl.uniform3f(
        uniformLocations.uCameraWorld,
        state.cameraWorld[0],
        state.cameraWorld[1],
        state.cameraWorld[2],
      );
      gl.uniform1f(uniformLocations.uPlaneY, NEATRACING_GRID_PLANE_Y_OFFSET_M);
      gl.uniform1f(uniformLocations.uSpacing, spacing);
      gl.uniform1f(
        uniformLocations.uLineWidthPx,
        NEATRACING_GRID_LINE_WIDTH_PX,
      );
      gl.uniform1f(
        uniformLocations.uGlowWidthPx,
        NEATRACING_GRID_GLOW_WIDTH_PX,
      );
      gl.uniform1f(
        uniformLocations.uGlowAlphaMultiplier,
        NEATRACING_GRID_GLOW_ALPHA_MULTIPLIER,
      );
      gl.uniform1f(uniformLocations.uMinAlpha, NEATRACING_GRID_MIN_ALPHA);
      gl.uniform1f(uniformLocations.uMaxAlpha, NEATRACING_GRID_MAX_ALPHA);
      gl.uniform1f(uniformLocations.uDrawDistance, NEATRACING_DRAW_DISTANCE_M);
      gl.uniform3f(
        uniformLocations.uLineColor,
        NEATRACING_GRID_LINE_COLOR_RGB[0],
        NEATRACING_GRID_LINE_COLOR_RGB[1],
        NEATRACING_GRID_LINE_COLOR_RGB[2],
      );
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
  };
}

/** Column-major identity matrix for the warmup offscreen draw. */
const NEATRACING_TRACK_GRID_WARMUP_IDENTITY_VIEW_PROJECTION = new Float32Array(
  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
);

/** Zero-pose draw state used for the track-grid shader warmup offscreen draw. */
const NEATRACING_TRACK_GRID_WARMUP_DRAW_STATE: TrackGridPassDrawState = {
  viewProjection: NEATRACING_TRACK_GRID_WARMUP_IDENTITY_VIEW_PROJECTION,
  cameraWorld: [0, 10, 20],
};

/**
 * Exposes the track grid pass as a shader warmup program (AC-040 seam).
 *
 * The returned entry keeps compilation lazy: `compileAndLink` builds the
 * pass, and `drawOffscreen` issues a zero-pose track-grid draw through the
 * shared pass path so the driver warms the exact render pipeline used on
 * screen. The context argument is accepted for interface parity; the pass is
 * bound to the context supplied at construction.
 *
 * @param context - WebGL2 context the pass resources will be built against.
 * @returns Warmup entry keyed to the `track-grid` registry slot.
 */
export function toTrackGridWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram {
  let pass: TrackGridPass | null = null;
  const ensurePass = (): TrackGridPass => {
    if (!pass) {
      pass = createTrackGridPass(context);
    }
    return pass;
  };

  return {
    key: NEATRACING_TRACK_GRID_PROGRAM_KEY,
    compileAndLink: (): void => {
      ensurePass();
    },
    drawOffscreen: (): void => {
      ensurePass().draw(NEATRACING_TRACK_GRID_WARMUP_DRAW_STATE);
    },
  };
}