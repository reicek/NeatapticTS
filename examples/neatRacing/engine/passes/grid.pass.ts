/**
 * NeatRacing world-space ground grid pass (spec P1-S2, slice P1-S2-B,
 * AC-005): a full-screen raycast ground plane whose fragment shader draws
 * the world-coherent 1.0-unit grid with constant *screen-space* line width.
 *
 * Contract highlights (spec section 6.4.2 + AC-005):
 * - Line half-widths are derived per fragment from `fwidth(vWorldXZ)` so a
 *   2 px core + 4 px halo stay constant in pixels at any zoom (G7: no
 *   `gl.lineWidth`, which WebGL2/ANGLE does not honor above 1).
 * - Line alpha fades with camera distance from 0.58 (near) to a 0.05 far
 *   floor across `NEATRACING_DRAW_DISTANCE_M` (320 m, TP-16).
 * - The grid is world-anchored: line phase comes from world-space XZ only,
 *   so it stays stable as the player moves.
 * - The plane sits at `NEATRACING_GRID_PLANE_Y_OFFSET_M` (-0.01 m) below the
 *   ground plane and the pass renders with depth test/write disabled, so it
 *   can never z-fight with track geometry drawn above it.
 *
 * The pure TypeScript functions below are the Jest-testable mirrors of the
 * fragment-shader math; the GLSL sources encode the identical formulas.
 */

import type { ShaderWarmupProgram } from '../context.service';

/** World spacing between grid lines, mirroring the 1.0 world-unit floor grid. */
export const NEATRACING_GRID_SPACING_WORLD = 1.0;

/** Far-floor alpha for grid lines (the visible-at-distance floor). */
export const NEATRACING_GRID_MIN_ALPHA = 0.05;

/** Near-camera alpha for grid line centers. */
export const NEATRACING_GRID_MAX_ALPHA = 0.58;

/** Core grid line width in screen pixels (constant via fwidth, not gl.lineWidth). */
export const NEATRACING_GRID_LINE_WIDTH_PX = 2;

/** Halo (glow) grid line width in screen pixels around the core line. */
export const NEATRACING_GRID_GLOW_WIDTH_PX = 4;

/** Alpha multiplier applied to the halo band relative to the line-center alpha. */
export const NEATRACING_GRID_GLOW_ALPHA_MULTIPLIER = 0.8;

/** Draw distance in meters spanning the depth fade (TP-16). */
export const NEATRACING_DRAW_DISTANCE_M = 320;

/** Grid plane height offset below the ground plane (anti-z-fighting, G9). */
export const NEATRACING_GRID_PLANE_Y_OFFSET_M = -0.01;

/** Warmup/draw registry key; must stay a member of the program registry. */
export const NEATRACING_GRID_PROGRAM_KEY = 'grid';

/**
 * Grid line color as normalized RGB. The spec pins no grid color (AC-005
 * references only the LD-24 neon family), so this mirrors the AC-004
 * starfield cyan rgba(95,255,255,1) to keep the palette small and coherent.
 */
export const NEATRACING_GRID_LINE_COLOR_RGB: readonly [number, number, number] =
  [95 / 255, 1, 1];

/**
 * Vertex shader (GLSL ES 3.00) for the grid pass: full-screen triangle that
 * covers normalized device coordinates. The vertex stage only forwards the
 * clip-space x/y as the `vClip` varying — clip coordinates are an affine
 * function of the interpolated barycentrics, so `vClip` interpolates exactly
 * across the giant triangle. Every projective quantity (unprojection, ray
 * direction, ground-plane hit, world position) is computed per fragment in
 * the fragment shader, because linearly interpolating per-vertex unprojections
 * across a fullscreen triangle is only exact at the 3 vertices and produces
 * inclined garbage lines everywhere else (AC-005: the grid must be flat,
 * world-anchored ground, never camera-tilted lines in the sky).
 */
export const NEATRACING_GRID_VERTEX_SHADER = `#version 300 es
const vec2 kTri[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vClip;
void main() {
  vec2 clip = kTri[gl_VertexID];
  vClip = clip;
  gl_Position = vec4(clip, 1.0, 1.0);
}`;

/**
 * Fragment shader (GLSL ES 3.00) for the grid pass: unprojects each fragment
 * through `uInvViewProjection` from its exact interpolated clip position,
 * casts the camera ray, and intersects the horizontal ground plane at
 * `uPlaneY`. Upward rays (sky) discard, so the grid stays flat on the ground
 * and the horizon is always horizontal. Line half-widths are derived per
 * fragment from `fwidth(worldXZ)` so a 2 px core + 4 px halo stay constant in
 * pixels at any zoom (G7: no `gl.lineWidth`, which WebGL2/ANGLE does not
 * honor above 1).
 */
export const NEATRACING_GRID_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 vClip;
uniform mat4 uInvViewProjection;
uniform vec3 uCameraWorld;
uniform float uPlaneY;
uniform float uSpacing;
uniform float uLineWidthPx;
uniform float uGlowWidthPx;
uniform float uGlowAlphaMultiplier;
uniform float uMinAlpha;
uniform float uMaxAlpha;
uniform float uDrawDistance;
uniform vec3 uLineColor;
out vec4 fragColor;

float distanceToAxis(float coord, float spacing) {
  float f = coord / spacing + 0.5;
  return abs(f - floor(f) - 0.5) * spacing;
}

float bandCoverage(float lineDistance, float threshold) {
  if (threshold <= 0.0) {
    return lineDistance <= 0.0 ? 1.0 : 0.0;
  }
  float t = clamp(lineDistance / threshold, 0.0, 1.0);
  return 1.0 - t * t * (3.0 - 2.0 * t);
}

void main() {
  vec4 far4 = uInvViewProjection * vec4(vClip, 1.0, 1.0);
  vec3 far = far4.xyz / far4.w;
  vec3 dir = far - uCameraWorld;
  if (dir.y >= 0.0) {
    discard;
  }
  float t = (uPlaneY - uCameraWorld.y) / dir.y;
  vec2 worldXZ = uCameraWorld.xz + dir.xz * t;
  float fwidthX = fwidth(worldXZ.x);
  float fwidthZ = fwidth(worldXZ.y);
  float lineX = distanceToAxis(worldXZ.x, uSpacing);
  float lineZ = distanceToAxis(worldXZ.y, uSpacing);
  float coreThresholdX = uLineWidthPx * fwidthX;
  float coreThresholdZ = uLineWidthPx * fwidthZ;
  float glowThresholdX = uGlowWidthPx * fwidthX;
  float glowThresholdZ = uGlowWidthPx * fwidthZ;
  float coreCoverage = max(
    bandCoverage(lineX, coreThresholdX),
    bandCoverage(lineZ, coreThresholdZ)
  );
  float glowCoverage = max(
    bandCoverage(lineX, glowThresholdX),
    bandCoverage(lineZ, glowThresholdZ)
  );
  float cameraDistance = length(worldXZ - uCameraWorld.xz);
  float fogFactor = clamp(cameraDistance / uDrawDistance, 0.0, 1.0);
  float baseAlpha = max(uMinAlpha, uMaxAlpha * (1.0 - fogFactor));
  float alpha = max(
    baseAlpha * coreCoverage,
    baseAlpha * uGlowAlphaMultiplier * glowCoverage
  );
  fragColor = vec4(uLineColor, alpha);
}`;

/** `gl.BLEND` enum value (0x0BE2) enabled during the grid draw. */
const GL_BLEND = 0x0be2;

/** `gl.SRC_ALPHA` enum value (0x0302) for premultiplied-free alpha blending. */
const GL_SRC_ALPHA = 0x0302;

/** `gl.ONE_MINUS_SRC_ALPHA` enum value (0x0303) for alpha blending. */
const GL_ONE_MINUS_SRC_ALPHA = 0x0303;

/** `gl.DEPTH_TEST` enum value (0x0B71) disabled during the grid draw. */
const GL_DEPTH_TEST = 0x0b71;

/** Names of every uniform the grid pass resolves and sets. */
const GRID_UNIFORM_NAMES = [
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
type GridUniformLocations = Record<
  (typeof GRID_UNIFORM_NAMES)[number],
  WebGLUniformLocation
>;

/**
 * Computes the world-space line half-width for a pixel width at the current
 * fragment footprint: the TS mirror of `threshold = widthPx * fwidth(coord)`.
 * Because `fwidth` is the per-pixel world-space derivative of the coordinate,
 * the resulting band is constant in screen pixels at any distance.
 *
 * @param widthPx Desired band width in screen pixels.
 * @param fwidthWorld `fwidth` derivative double of the world coordinate.
 * @returns World-space half-width threshold to feed the band coverage.
 */
export function gridLineThresholdWorld(
  widthPx: number,
  fwidthWorld: number,
): number {
  return widthPx * fwidthWorld;
}

/**
 * Normalized camera-distance fog factor clamped to [0, 1]: 0 at the camera,
 * 1 at (and beyond) the draw distance. TS mirror of the shader fog term.
 *
 * @param distanceM Camera distance in meters.
 * @param drawDistanceM Draw distance in meters (must be positive).
 * @returns Fog factor in [0, 1].
 */
export function gridFogFactor01(
  distanceM: number,
  drawDistanceM: number,
): number {
  const t = distanceM / drawDistanceM;
  return Math.min(1, Math.max(0, t));
}

/**
 * Depth-fade line-center alpha: `max(MIN_ALPHA, MAX_ALPHA * (1 - fogFactor))`,
 * i.e. 0.58 at the camera fading linearly to the 0.05 far floor across the
 * draw distance. TS mirror of the shader alpha envelope (G7 MIN/MAX form).
 *
 * @param distanceM Camera distance in meters.
 * @param drawDistanceM Draw distance in meters (must be positive).
 * @returns Line-center alpha in [MIN_ALPHA, MAX_ALPHA].
 */
export function gridLineAlpha(
  distanceM: number,
  drawDistanceM: number,
): number {
  const fog = gridFogFactor01(distanceM, drawDistanceM);
  return Math.max(
    NEATRACING_GRID_MIN_ALPHA,
    NEATRACING_GRID_MAX_ALPHA * (1 - fog),
  );
}

/**
 * Distance from a world coordinate to the nearest grid axis line for a given
 * spacing: the TS mirror of `abs(fract(coord / spacing + 0.5) - 0.5) *
 * spacing`. Periodic per spacing, symmetric around each line, and purely
 * world-anchored (camera-independent).
 *
 * @param worldCoord World-space coordinate along one axis.
 * @param spacing Grid spacing in world units (must be positive).
 * @returns Distance in world units to the nearest line, in [0, spacing / 2].
 */
export function gridAxisLineDistance(
  worldCoord: number,
  spacing: number,
): number {
  const scaled = worldCoord / spacing + 0.5;
  return Math.abs(scaled - Math.floor(scaled) - 0.5) * spacing;
}

/**
 * Smoothstep band coverage for a fragment at `lineDistance` from a line with
 * half-width `threshold`: 1 at the line center, 0.5 at half the threshold,
 * 0 at and beyond the threshold. Degenerate (non-positive) thresholds map to
 * 1 only exactly on the line. TS mirror of the shader band term.
 *
 * @param lineDistance Distance from the line center in world units.
 * @param threshold World-space band half-width (from the fwidth mirror).
 * @returns Coverage in [0, 1].
 */
export function gridBandCoverage01(
  lineDistance: number,
  threshold: number,
): number {
  if (threshold <= 0) {
    return lineDistance <= 0 ? 1 : 0;
  }
  const t = Math.min(1, Math.max(0, lineDistance / threshold));
  return 1 - t * t * (3 - 2 * t);
}

/**
 * Full fragment alpha for the grid at a world position: the maximum of the
 * core contribution (`baseAlpha * coreCoverage`) and the halo contribution
 * (`baseAlpha * GLOW_ALPHA_MULTIPLIER * glowCoverage`), where coverages are
 * the per-axis maxima over X and Z. TS mirror of the shader `main` body.
 *
 * @param worldX World-space X coordinate of the fragment.
 * @param worldZ World-space Z coordinate of the fragment.
 * @param fwidthWorld `fwidth` derivative double of the world coordinate.
 * @param distanceM Camera distance in meters.
 * @param drawDistanceM Draw distance in meters (must be positive).
 * @returns Fragment alpha in [0, MAX_ALPHA].
 */
export function gridFragmentAlpha(
  worldX: number,
  worldZ: number,
  fwidthWorld: number,
  distanceM: number,
  drawDistanceM: number,
): number {
  const baseAlpha = gridLineAlpha(distanceM, drawDistanceM);
  const lineX = gridAxisLineDistance(worldX, NEATRACING_GRID_SPACING_WORLD);
  const lineZ = gridAxisLineDistance(worldZ, NEATRACING_GRID_SPACING_WORLD);
  const coreThreshold = gridLineThresholdWorld(
    NEATRACING_GRID_LINE_WIDTH_PX,
    fwidthWorld,
  );
  const glowThreshold = gridLineThresholdWorld(
    NEATRACING_GRID_GLOW_WIDTH_PX,
    fwidthWorld,
  );
  const coreCoverage = Math.max(
    gridBandCoverage01(lineX, coreThreshold),
    gridBandCoverage01(lineZ, coreThreshold),
  );
  const glowCoverage = Math.max(
    gridBandCoverage01(lineX, glowThreshold),
    gridBandCoverage01(lineZ, glowThreshold),
  );
  return Math.max(
    baseAlpha * coreCoverage,
    baseAlpha * NEATRACING_GRID_GLOW_ALPHA_MULTIPLIER * glowCoverage,
  );
}

/** Per-draw inputs for the grid pass. */
export interface GridPassDrawState {
  /** Column-major 16-element view-projection matrix. */
  readonly viewProjection: Float32Array;
  /** Camera world position `[x, y, z]` in meters (fog origin + ray origin). */
  readonly cameraWorld: readonly [number, number, number];
}

/**
 * Inverts a 4x4 column-major matrix into `out` using the adjugate/determinant
 * method. The source and destination must not alias.
 *
 * @param out - 16-float destination (column-major).
 * @param m - 16-float source matrix (column-major).
 * @returns `true` when the matrix was invertible, `false` if singular.
 */
function invertMatrix44(out: Float32Array, m: Float32Array): boolean {
  const a00 = m[0],
    a01 = m[1],
    a02 = m[2],
    a03 = m[3];
  const a10 = m[4],
    a11 = m[5],
    a12 = m[6],
    a13 = m[7];
  const a20 = m[8],
    a21 = m[9],
    a22 = m[10],
    a23 = m[11];
  const a30 = m[12],
    a31 = m[13],
    a32 = m[14],
    a33 = m[15];

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

/** The grid pass handle: create once, draw every frame. */
export interface GridPass {
  /** Draws the ground grid quad for the given camera state. */
  draw(state: GridPassDrawState): void;
}

/**
 * Compiles one shader stage or throws a descriptive error carrying the stage
 * label and the driver info log.
 */
function compileGridShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error(
      `neatRacing grid pass: createShader returned null for ${label} stage`,
    );
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const infoLog = gl.getShaderInfoLog(shader) ?? 'unknown error';
    throw new Error(
      `neatRacing grid pass: ${label} shader compile failed: ${infoLog}`,
    );
  }
  return shader;
}

/**
 * Creates the grid pass: compiles and links the program exactly once and
 * resolves all uniform locations. The returned handle never recompiles or
 * re-uploads; `draw` only sets uniforms and issues a full-screen triangle
 * draw with blending enabled and depth test/write disabled (background-layer
 * composition, anti-z-fighting vs the track).
 *
 * @param gl WebGL2 rendering context to bind the pass to.
 * @returns The grid pass handle.
 */
export function createGridPass(gl: WebGL2RenderingContext): GridPass {
  const vertexShader = compileGridShader(
    gl,
    gl.VERTEX_SHADER,
    NEATRACING_GRID_VERTEX_SHADER,
    'vertex',
  );
  const fragmentShader = compileGridShader(
    gl,
    gl.FRAGMENT_SHADER,
    NEATRACING_GRID_FRAGMENT_SHADER,
    'fragment',
  );
  const program = gl.createProgram();
  if (!program) {
    throw new Error('neatRacing grid pass: createProgram returned null');
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const infoLog = gl.getProgramInfoLog(program) ?? 'unknown error';
    throw new Error(`neatRacing grid pass: program link failed: ${infoLog}`);
  }
  const uniforms = {} as Partial<GridUniformLocations>;
  for (const name of GRID_UNIFORM_NAMES) {
    const location = gl.getUniformLocation(program, name);
    if (!location) {
      throw new Error(
        `neatRacing grid pass: uniform ${name} not found in program`,
      );
    }
    uniforms[name] = location;
  }
  const uniformLocations = uniforms as GridUniformLocations;
  const invViewProjection = new Float32Array(16);
  return {
    draw(state: GridPassDrawState): void {
      if (state.viewProjection.length !== 16) {
        throw new Error(
          'neatRacing grid pass draw: viewProjection must be a 16-element column-major matrix',
        );
      }
      if (!invertMatrix44(invViewProjection, state.viewProjection)) {
        throw new Error(
          'neatRacing grid pass draw: viewProjection matrix is singular',
        );
      }
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
      gl.uniform1f(uniformLocations.uSpacing, NEATRACING_GRID_SPACING_WORLD);
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

/** Column-major identity matrix reused for the zero-pose warmup draw. */
const IDENTITY_VIEW_PROJECTION = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
];

/** Zero-pose warmup draw state: identity projection, camera above the origin. */
const ZERO_POSE_DRAW_STATE: GridPassDrawState = {
  viewProjection: new Float32Array(IDENTITY_VIEW_PROJECTION),
  cameraWorld: [0, 10, 20],
};

/**
 * Exposes the grid pass as a shader warmup program (AC-040 seam).
 *
 * The returned entry keeps compilation lazy: `compileAndLink` builds the
 * pass (compiling and linking the program), and `drawOffscreen` issues the
 * zero-pose ground-grid draw through the shared pass path so the driver
 * warms the exact render pipeline used on screen. The context passed to
 * `drawOffscreen` is accepted for interface parity; the pass is bound to
 * the context supplied at construction.
 *
 * @param context - WebGL2 context the pass resources will be built against.
 * @returns Warmup entry keyed to the grid registry slot (second warmed key).
 *
 * @example
 * const warmup = toGridWarmupProgram(gl);
 * warmup.compileAndLink();
 * warmup.drawOffscreen(gl);
 */
export function toGridWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram {
  let pass: GridPass | null = null;
  const ensurePass = (): GridPass => {
    if (!pass) {
      pass = createGridPass(context);
    }
    return pass;
  };

  return {
    key: NEATRACING_GRID_PROGRAM_KEY,
    compileAndLink: (): void => {
      ensurePass();
    },
    drawOffscreen: (): void => {
      ensurePass().draw(ZERO_POSE_DRAW_STATE);
    },
  };
}
