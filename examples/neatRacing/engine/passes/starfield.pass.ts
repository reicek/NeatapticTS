/// <reference lib="dom" />

/**
 * WebGL2 starfield background pass for the Neat Racing voxel racer engine
 * (slice P1-S2-A; spec §6.4.1 and AC-004).
 *
 * The module owns the full starfield layer behind one small seam:
 *
 * 1. Deterministic layout (AC-004): each of the three flappy-parity layers
 *    (far/mid/near) generates its stars once at pass construction from an
 *    xorshift32 generator seeded by the layer spec (1337/2777/4242), using
 *    the exact placement order of the flappy tile builder so the visual
 *    result ports one-to-one.
 * 2. Camera-driven scroll (G6): layer offsets are pure functions of the
 *    camera pose — `offset = −(revolution fraction) × tile extent × scroll
 *    ratio` — so yaw rotates the sky horizontally and pitch scrolls it
 *    vertically, with far/mid/near shearing at 3%/4%/5% rates for depth.
 * 3. Zero per-frame allocation (P9): `draw` only reads precomputed layouts
 *    and vertex arrays; nothing is allocated between frames. The layouts are
 *    also exposed via {@link StarfieldPass.getLayerLayout} so diagnostics can
 *    observe the stable references.
 *
 * The pass never writes depth (`DEPTH_TEST` disabled, `depthMask(false)`) —
 * it is a pure 2D backdrop painted first, before every 3D world pass. Stars
 * are `gl.POINTS` batches sized by `gl_PointSize` and tinted with the
 * flappy-parity cyan. Everything here is visual-only and exempt from replay
 * byte parity (spec §9.7.2); the seeds derive from the visual seed base
 * TP-20.
 *
 * @module
 */

import { NEATRACING_FIXED_RENDER_HEIGHT } from '../../browser-entry/constants';
import {
  NEATRACING_STARFIELD_INCLUSIVE_RANGE_OFFSET,
  NEATRACING_STARFIELD_LAYER_SPECS,
  NEATRACING_STARFIELD_TILE_HEIGHT_PX,
  NEATRACING_STARFIELD_TILE_WIDTH_PX,
  NEATRACING_STARFIELD_UNSIGNED_NORMALIZATION_DIVISOR,
  NEATRACING_STARFIELD_XORSHIFT_LEFT_SHIFT_FINAL,
  NEATRACING_STARFIELD_XORSHIFT_LEFT_SHIFT_PRIMARY,
  NEATRACING_STARFIELD_XORSHIFT_RIGHT_SHIFT,
} from '../../browser-entry/constants.starfield';
import type { StarfieldLayerSpec } from '../../browser-entry/constants.starfield';
import type { ShaderWarmupProgram } from '../context.service';

/** Full-revolution denominator shared by both scroll axes (G6 formulas). */
const TWO_PI = Math.PI * 2;

/** RGB triple matching `NEATRACING_STARFIELD_CYAN_FILL_STYLE` (AC-004). */
const STAR_COLOR_UNIFORM: readonly [number, number, number] = [
  95 / 255,
  255 / 255,
  255 / 255,
];

/** Vertex attribute location of the per-star tile position (vec2, pixels). */
const STAR_POSITION_ATTRIBUTE_LOCATION = 0;

/** Vertex attribute location of the per-star point size (pixels). */
const STAR_SIZE_ATTRIBUTE_LOCATION = 1;

/** Vertex attribute location of the per-star fill alpha. */
const STAR_ALPHA_ATTRIBUTE_LOCATION = 2;

/** Floats per star vertex: xPx, yPx, sizePx, alpha. */
const STAR_VERTEX_FLOATS_PER_STAR = 4;

/** Byte stride of one interleaved star vertex (4 floats × 4 bytes). */
const STAR_VERTEX_STRIDE_BYTES = 16;

/** Offscreen warmup pose: zero yaw and pitch so the sweep draws the base layout. */
const ZERO_CAMERA_POSE: StarfieldCameraPose = {
  yawRadians: 0,
  pitchRadians: 0,
};

/** One deterministic star placement inside a starfield tile (flappy parity). */
export interface StarPlacement {
  /** Horizontal pixel position within the tile, `[0, tileWidthPx)`. */
  readonly xPx: number;
  /** Vertical pixel position within the tile, `[0, tileHeightPx)`. */
  readonly yPx: number;
  /** Square star edge length in pixels, `[minSizePx, maxSizePx]`. */
  readonly sizePx: number;
  /** Star fill alpha, `[minAlpha, maxAlpha)`. */
  readonly alpha: number;
}

/** Camera pose subset the starfield reacts to (G6). */
export interface StarfieldCameraPose {
  /** Camera yaw in radians; positive rotates the camera to the right. */
  readonly yawRadians: number;
  /** Camera pitch in radians; positive pitches the camera down. */
  readonly pitchRadians: number;
}

/** Precomputed point layout for one parallax starfield layer. */
export interface StarfieldLayerLayout {
  /** Layer recipe the layout was generated from. */
  readonly spec: StarfieldLayerSpec;
  /** Deterministic placements, exactly `spec.starCount` entries. */
  readonly placements: readonly StarPlacement[];
}

/** Renderable handle for one starfield layer: layout plus its GPU buffers. */
interface StarfieldLayerRenderTarget {
  /** Layer index into the pinned spec table (draw and scroll order). */
  readonly layerIndex: number;
  /** Precomputed deterministic placements for the layer. */
  readonly layout: StarfieldLayerLayout;
  /** Vertex array object owning the layer's interleaved star buffer. */
  readonly vertexArrayObject: WebGLVertexArrayObject;
}

/** The starfield background pass seam consumed by the render loop. */
export interface StarfieldPass {
  /** Warmup registry key; pinned as the first warmed program (AC-040). */
  readonly key: 'starfield';
  /**
   * Draws all three star layers for the current camera pose.
   *
   * Allocates nothing (P9): every buffer, uniform, and layout value was
   * precomputed at construction.
   *
   * @param cameraPose - Current camera yaw/pitch in radians.
   */
  draw(cameraPose: StarfieldCameraPose): void;
  /**
   * Returns the precomputed layout for one layer (diagnostics seam).
   *
   * @param layerIndex - Layer index into the pinned spec order (0–2).
   * @returns The stable layout reference reused by every draw call.
   * @throws RangeError when `layerIndex` is outside the pinned layer count.
   */
  getLayerLayout(layerIndex: number): StarfieldLayerLayout;
}

/** Optional construction knobs for the starfield pass. */
export interface CreateStarfieldPassOptions {
  /**
   * Horizontal backdrop size in pixels used for the NDC mapping.
   *
   * Defaults to {@link NEATRACING_STARFIELD_TILE_WIDTH_PX}. Tests may
   * override this to probe wrap behavior, but production rendering should use
   * the pinned tile width so a single tiled star layer fills the viewport.
   */
  readonly backdropWidthPx?: number;
  /**
   * Vertical backdrop size in pixels used for the NDC mapping.
   *
   * Defaults to {@link NEATRACING_STARFIELD_TILE_HEIGHT_PX} (the sky band
   * height). Tests may override this to probe wrap behavior.
   */
  readonly backdropHeightPx?: number;
}

/**
 * Vertex shader for the starfield point batch (GLSL ES 3.00).
 *
 * Wraps each star's tile position by the per-layer camera scroll, then maps
 * the wrapped position into NDC over the backing store: horizontally across
 * the full width, vertically across the fixed sky band at the top of the
 * frame. `mod` produces the seamless infinite-tile wrap; negative scroll
 * offsets (the G6 sign convention) shift stars left/up as the camera turns
 * right / pitches down.
 */
const STARFIELD_VERTEX_SHADER_SOURCE = `#version 300 es
layout(location = 0) in vec2 a_tilePositionPx;
layout(location = 1) in float a_sizePx;
layout(location = 2) in float a_alpha;
uniform vec2 u_tileSizePx;
uniform vec2 u_backdropSizePx;
uniform vec2 u_scrollOffsetPx;
out float v_alpha;
void main() {
  float wrappedXPx = mod(a_tilePositionPx.x + u_scrollOffsetPx.x, u_tileSizePx.x);
  float wrappedYPx = mod(a_tilePositionPx.y + u_scrollOffsetPx.y, u_tileSizePx.y);
  float ndcX = (wrappedXPx / u_backdropSizePx.x) * 2.0 - 1.0;
  float ndcY = 1.0 - (wrappedYPx / u_backdropSizePx.y) * 2.0;
  gl_Position = vec4(ndcX, ndcY, 0.0, 1.0);
  gl_PointSize = a_sizePx;
  v_alpha = a_alpha;
}`;

/**
 * Fragment shader for the starfield point batch (GLSL ES 3.00).
 *
 * Outputs the flappy-parity cyan tint scaled by the per-star alpha; blending
 * is configured source-over at pass construction so stars composite over the
 * sky without depth interaction.
 */
const STARFIELD_FRAGMENT_SHADER_SOURCE = `#version 300 es
precision mediump float;
in float v_alpha;
uniform vec3 u_starColor;
out vec4 outColor;
void main() {
  outColor = vec4(u_starColor, v_alpha);
}`;

/**
 * Creates a deterministic pseudo-random generator for starfield layouts.
 *
 * Direct port of the flappy `createSeededRandom` xorshift32 generator so the
 * voxel racer reproduces the exact same value stream per seed. The seed is
 * coerced to unsigned 32-bit state; every call advances the state through
 * the pinned 13/17/5 shift triple and normalizes into `[0, 1)`.
 *
 * @param seed - Unsigned integer seed (layer spec seed or visual seed base).
 * @returns Function yielding deterministic values in `[0, 1)`.
 *
 * @example
 * const random = createStarfieldRng(NEATRACING_VISUAL_SEED);
 * random(); // → deterministic value in [0, 1)
 */
export function createStarfieldRng(seed: number): () => number {
  let randomState = seed >>> 0;

  return () => {
    // Step 1: Apply xorshift32 state transitions (flappy parity).
    randomState ^=
      randomState << NEATRACING_STARFIELD_XORSHIFT_LEFT_SHIFT_PRIMARY;
    randomState >>>= 0;
    randomState ^= randomState >> NEATRACING_STARFIELD_XORSHIFT_RIGHT_SHIFT;
    randomState >>>= 0;
    randomState ^=
      randomState << NEATRACING_STARFIELD_XORSHIFT_LEFT_SHIFT_FINAL;
    randomState >>>= 0;

    // Step 2: Normalize 32-bit unsigned state into [0, 1).
    return randomState / NEATRACING_STARFIELD_UNSIGNED_NORMALIZATION_DIVISOR;
  };
}

/**
 * Resolves one deterministic star placement and appearance from the RNG.
 *
 * Port of the flappy `resolveStarPlacement`: exactly four draws per star in
 * the pinned order (x, y, size, alpha) so any change to the sequence would
 * shift every later star. The inclusive size offset makes the upper size
 * bound reachable, matching the flappy tile builder.
 *
 * @param options - Random source, layer bounds, and tile extent.
 * @returns Pixel location, square size, and alpha for one rendered star.
 */
function resolveStarPlacement(options: {
  random: () => number;
  spec: StarfieldLayerSpec;
  tileWidthPx: number;
  tileHeightPx: number;
}): StarPlacement {
  const xPx = Math.floor(options.random() * options.tileWidthPx);
  const yPx = Math.floor(options.random() * options.tileHeightPx);
  const sizePx =
    options.spec.minSizePx +
    Math.floor(
      options.random() *
        (options.spec.maxSizePx -
          options.spec.minSizePx +
          NEATRACING_STARFIELD_INCLUSIVE_RANGE_OFFSET),
    );
  const alpha =
    options.spec.minAlpha +
    options.random() * (options.spec.maxAlpha - options.spec.minAlpha);

  return { xPx, yPx, sizePx, alpha };
}

/**
 * Builds the deterministic point layout for one starfield layer.
 *
 * Called once per layer at pass construction; repeated calls with the same
 * arguments return deep-equal layouts because the xorshift32 stream is fully
 * seeded by the layer spec.
 *
 * @param spec - Layer recipe (seed, counts, size/alpha ranges).
 * @param tileWidthPx - Tile width the x positions are generated against.
 * @param tileHeightPx - Tile height the y positions are generated against.
 * @returns Immutable layout holding exactly `spec.starCount` placements.
 *
 * @example
 * const layout = buildStarLayerLayout(
 *   NEATRACING_STARFIELD_LAYER_SPECS[0],
 *   NEATRACING_STARFIELD_TILE_WIDTH_PX,
 *   NEATRACING_STARFIELD_TILE_HEIGHT_PX,
 * );
 * // → { spec, placements: StarPlacement[35] }
 */
export function buildStarLayerLayout(
  spec: StarfieldLayerSpec,
  tileWidthPx: number,
  tileHeightPx: number,
): StarfieldLayerLayout {
  const random = createStarfieldRng(spec.seed);
  const placements: StarPlacement[] = [];
  for (let starIndex = 0; starIndex < spec.starCount; starIndex += 1) {
    placements.push(
      resolveStarPlacement({
        random,
        spec,
        tileWidthPx,
        tileHeightPx,
      }),
    );
  }
  return { spec, placements };
}

/**
 * Resolves the layer spec for an index, failing loudly on out-of-range use.
 *
 * @param layerIndex - Index into the pinned spec order (0 = far, 2 = near).
 * @returns The pinned layer spec.
 * @throws RangeError when the index is outside the pinned three layers.
 */
function requireLayerSpec(layerIndex: number): StarfieldLayerSpec {
  const spec = NEATRACING_STARFIELD_LAYER_SPECS[layerIndex];
  if (!spec) {
    throw new RangeError(
      `starfield layer index out of range: ${String(layerIndex)}`,
    );
  }
  return spec;
}

/**
 * Resolves the horizontal scroll offset for one layer from the camera yaw.
 *
 * G6 formula: `offsetX = −(yaw / 2π) × tileWidthPx × scrollRatio`. Positive
 * yaw (camera rotating right) shifts stars left, and each layer scrolls at
 * its pinned ratio (far 3%, mid 4%, near 5%) for the parallax shear.
 *
 * @param yawRadians - Camera yaw in radians.
 * @param layerIndex - Layer index into the pinned spec order.
 * @returns Horizontal scroll offset in pixels.
 *
 * @example
 * computeStarLayerOffsetX(Math.PI / 2, 0); // → −3.84 (quarter turn, far layer)
 */
export function computeStarLayerOffsetX(
  yawRadians: number,
  layerIndex: number,
): number {
  return (
    -(yawRadians / TWO_PI) *
    NEATRACING_STARFIELD_TILE_WIDTH_PX *
    requireLayerSpec(layerIndex).scrollRatio
  );
}

/**
 * Resolves the vertical scroll offset for one layer from the camera pitch.
 *
 * G6 formula: `offsetY = −(pitch / 2π) × tileHeightPx × scrollRatio`. A
 * camera pitching down (positive pitch) scrolls stars up the sky band so
 * the fixed 240-pixel band appears to tilt away with the view.
 *
 * @param pitchRadians - Camera pitch in radians.
 * @param layerIndex - Layer index into the pinned spec order.
 * @returns Vertical scroll offset in pixels.
 *
 * @example
 * computeStarLayerOffsetY(Math.PI / 2, 0); // → −1.8 (quarter pitch, far layer)
 */
export function computeStarLayerOffsetY(
  pitchRadians: number,
  layerIndex: number,
): number {
  return (
    -(pitchRadians / TWO_PI) *
    NEATRACING_STARFIELD_TILE_HEIGHT_PX *
    requireLayerSpec(layerIndex).scrollRatio
  );
}

/**
 * Resolves both scroll offsets for one layer from the camera pose.
 *
 * Pure convenience wrapper over {@link computeStarLayerOffsetX} and
 * {@link computeStarLayerOffsetY}; the draw path calls the two scalar
 * functions directly so the per-frame path never allocates.
 *
 * @param cameraPose - Current camera pose in radians.
 * @param layerIndex - Layer index into the pinned spec order.
 * @returns Both pixel scroll offsets for the layer.
 *
 * @example
 * computeStarLayerOffsets({ yawRadians: 0, pitchRadians: 0 }, 0);
 * // → { offsetX: -0, offsetY: -0 }
 */
export function computeStarLayerOffsets(
  cameraPose: StarfieldCameraPose,
  layerIndex: number,
): { offsetX: number; offsetY: number } {
  return {
    offsetX: computeStarLayerOffsetX(cameraPose.yawRadians, layerIndex),
    offsetY: computeStarLayerOffsetY(cameraPose.pitchRadians, layerIndex),
  };
}

/**
 * Packs one layer's placements into the interleaved star vertex buffer.
 *
 * @param layout - Precomputed layer layout.
 * @returns Float buffer laid out as `[xPx, yPx, sizePx, alpha]` per star.
 */
function packLayerVertices(layout: StarfieldLayerLayout): Float32Array {
  const vertices = new Float32Array(
    layout.placements.length * STAR_VERTEX_FLOATS_PER_STAR,
  );
  layout.placements.forEach((placement, starIndex) => {
    const vertexOffset = starIndex * STAR_VERTEX_FLOATS_PER_STAR;
    vertices[vertexOffset] = placement.xPx;
    vertices[vertexOffset + 1] = placement.yPx;
    vertices[vertexOffset + 2] = placement.sizePx;
    vertices[vertexOffset + 3] = placement.alpha;
  });
  return vertices;
}

/**
 * Uploads one layer's stars into a vertex array object.
 *
 * Runs once per layer at construction; the returned VAO owns the interleaved
 * buffer and both attribute pointers so the per-frame draw path only needs
 * `bindVertexArray` + `drawArrays` (P9 zero-allocation contract).
 *
 * @param context - WebGL2 context owning the GPU resources.
 * @param layout - Precomputed layer layout to upload.
 * @returns The bound vertex array object for the layer.
 * @throws Error when the context cannot allocate the VAO or buffer.
 */
function createLayerVertexArray(
  context: WebGL2RenderingContext,
  layout: StarfieldLayerLayout,
): WebGLVertexArrayObject {
  const vertexArrayObject = context.createVertexArray();
  if (!vertexArrayObject) {
    throw new Error('starfield pass could not create a vertex array object');
  }
  const buffer = context.createBuffer();
  if (!buffer) {
    throw new Error('starfield pass could not create a vertex buffer');
  }

  context.bindVertexArray(vertexArrayObject);
  context.bindBuffer(context.ARRAY_BUFFER, buffer);
  context.bufferData(
    context.ARRAY_BUFFER,
    packLayerVertices(layout),
    context.STATIC_DRAW,
  );
  context.enableVertexAttribArray(STAR_POSITION_ATTRIBUTE_LOCATION);
  context.vertexAttribPointer(
    STAR_POSITION_ATTRIBUTE_LOCATION,
    2,
    context.FLOAT,
    false,
    STAR_VERTEX_STRIDE_BYTES,
    0,
  );
  context.enableVertexAttribArray(STAR_SIZE_ATTRIBUTE_LOCATION);
  context.vertexAttribPointer(
    STAR_SIZE_ATTRIBUTE_LOCATION,
    1,
    context.FLOAT,
    false,
    STAR_VERTEX_STRIDE_BYTES,
    8,
  );
  context.enableVertexAttribArray(STAR_ALPHA_ATTRIBUTE_LOCATION);
  context.vertexAttribPointer(
    STAR_ALPHA_ATTRIBUTE_LOCATION,
    1,
    context.FLOAT,
    false,
    STAR_VERTEX_STRIDE_BYTES,
    12,
  );
  context.bindVertexArray(null);
  return vertexArrayObject;
}

/**
 * Compiles one shader stage and fails loudly with the driver info log.
 *
 * @param context - WebGL2 context owning the shader.
 * @param shaderType - `VERTEX_SHADER` or `FRAGMENT_SHADER`.
 * @param shaderSource - GLSL ES 3.00 source text.
 * @returns The compiled shader handle.
 * @throws Error when allocation or compilation fails.
 */
function compileStarfieldShader(
  context: WebGL2RenderingContext,
  shaderType: number,
  shaderSource: string,
): WebGLShader {
  const shader = context.createShader(shaderType);
  if (!shader) {
    throw new Error('starfield pass could not create a shader');
  }
  context.shaderSource(shader, shaderSource);
  context.compileShader(shader);
  if (!context.getShaderParameter(shader, context.COMPILE_STATUS)) {
    const infoLog = context.getShaderInfoLog(shader) ?? '(no info log)';
    throw new Error(`starfield pass shader compile failed: ${infoLog}`);
  }
  return shader;
}

/**
 * Compiles and links the starfield shader program.
 *
 * @param context - WebGL2 context owning the program.
 * @returns The linked program handle.
 * @throws Error when allocation, compilation, or linking fails.
 */
function compileStarfieldProgram(
  context: WebGL2RenderingContext,
): WebGLProgram {
  const vertexShader = compileStarfieldShader(
    context,
    context.VERTEX_SHADER,
    STARFIELD_VERTEX_SHADER_SOURCE,
  );
  const fragmentShader = compileStarfieldShader(
    context,
    context.FRAGMENT_SHADER,
    STARFIELD_FRAGMENT_SHADER_SOURCE,
  );
  const program = context.createProgram();
  if (!program) {
    throw new Error('starfield pass could not create the shader program');
  }
  context.attachShader(program, vertexShader);
  context.attachShader(program, fragmentShader);
  context.linkProgram(program);
  if (!context.getProgramParameter(program, context.LINK_STATUS)) {
    const infoLog = context.getProgramInfoLog(program) ?? '(no info log)';
    throw new Error(`starfield pass program link failed: ${infoLog}`);
  }
  return program;
}

/**
 * Resolves a uniform location, failing loudly when the driver drops it.
 *
 * @param context - WebGL2 context owning the program.
 * @param program - Linked starfield program.
 * @param name - Uniform name declared in the shader source.
 * @returns The live uniform location.
 * @throws Error when the uniform location is unavailable.
 */
function requireUniformLocation(
  context: WebGL2RenderingContext,
  program: WebGLProgram,
  name: string,
): WebGLUniformLocation {
  const location = context.getUniformLocation(program, name);
  if (!location) {
    throw new Error(`starfield pass missing uniform: ${name}`);
  }
  return location;
}

/**
 * Creates the starfield background pass (AC-004).
 *
 * Compiles the point-sprite program once, precomputes every layer layout and
 * vertex buffer once (P9), configures the static backdrop state (cyan tint,
 * alpha blending, no depth writes), and returns the draw seam. Every later
 * `draw` call only rebinds precomputed state and feeds the pure G6 scroll
 * offsets, so the per-frame path allocates nothing.
 *
 * @param context - WebGL2 context acquired by the rendering context service.
 * @param options - Optional backing-store width override for NDC mapping.
 * @returns The starfield pass seam.
 * @throws Error when program compilation, linking, or buffer setup fails.
 *
 * @example
 * const starfieldPass = createStarfieldPass(gl);
 * starfieldPass.draw({ yawRadians: camera.yaw, pitchRadians: camera.pitch });
 */
export function createStarfieldPass(
  context: WebGL2RenderingContext,
  options: CreateStarfieldPassOptions = {},
): StarfieldPass {
  const program = compileStarfieldProgram(context);
  const tileSizeLocation = requireUniformLocation(
    context,
    program,
    'u_tileSizePx',
  );
  const backdropSizeLocation = requireUniformLocation(
    context,
    program,
    'u_backdropSizePx',
  );
  const starColorLocation = requireUniformLocation(
    context,
    program,
    'u_starColor',
  );
  const scrollOffsetLocation = requireUniformLocation(
    context,
    program,
    'u_scrollOffsetPx',
  );
  // The shader maps a wrapped tile position directly into NDC. The horizontal
  // axis must span the full viewport, so the backdrop width is the tile width
  // (the wrapped tile repeats seamlessly across the width). The vertical axis
  // must keep stars inside the sky band, so the backdrop height is the fixed
  // render height: a 240 px tile then maps to the top ~33% of the frame
  // (NEATRACING_SKY_BAND_HEIGHT_PX default 240 / 720). Test fixtures may
  // override either axis.
  const backdropWidthPx =
    options.backdropWidthPx ?? NEATRACING_STARFIELD_TILE_WIDTH_PX;
  const backdropHeightPx =
    options.backdropHeightPx ?? NEATRACING_FIXED_RENDER_HEIGHT;

  // Step 1: Configure the static backdrop state (never depth-written).
  context.useProgram(program);
  context.uniform2f(
    tileSizeLocation,
    NEATRACING_STARFIELD_TILE_WIDTH_PX,
    NEATRACING_STARFIELD_TILE_HEIGHT_PX,
  );
  context.uniform2f(backdropSizeLocation, backdropWidthPx, backdropHeightPx);
  context.uniform3f(
    starColorLocation,
    STAR_COLOR_UNIFORM[0],
    STAR_COLOR_UNIFORM[1],
    STAR_COLOR_UNIFORM[2],
  );
  context.enable(context.BLEND);
  context.blendFunc(context.SRC_ALPHA, context.ONE_MINUS_SRC_ALPHA);
  context.disable(context.DEPTH_TEST);
  context.depthMask(false);

  // Step 2: Precompute every layer layout and vertex batch exactly once.
  const layerLayouts: StarfieldLayerLayout[] =
    NEATRACING_STARFIELD_LAYER_SPECS.map((spec) =>
      buildStarLayerLayout(
        spec,
        NEATRACING_STARFIELD_TILE_WIDTH_PX,
        NEATRACING_STARFIELD_TILE_HEIGHT_PX,
      ),
    );
  const layerRenderTargets: StarfieldLayerRenderTarget[] = layerLayouts.map(
    (layout, layerIndex) => ({
      layout,
      layerIndex,
      vertexArrayObject: createLayerVertexArray(context, layout),
    }),
  );

  return {
    key: 'starfield',
    draw(cameraPose: StarfieldCameraPose): void {
      context.useProgram(program);
      for (
        let layerIndex = 0;
        layerIndex < layerRenderTargets.length;
        layerIndex += 1
      ) {
        const target = layerRenderTargets[layerIndex];
        context.bindVertexArray(target.vertexArrayObject);
        context.uniform2f(
          scrollOffsetLocation,
          computeStarLayerOffsetX(cameraPose.yawRadians, target.layerIndex),
          computeStarLayerOffsetY(cameraPose.pitchRadians, target.layerIndex),
        );
        context.drawArrays(context.POINTS, 0, target.layout.placements.length);
      }
    },
    getLayerLayout(layerIndex: number): StarfieldLayerLayout {
      const layout = layerLayouts[layerIndex];
      if (!layout) {
        throw new RangeError(
          `starfield layer index out of range: ${String(layerIndex)}`,
        );
      }
      return layout;
    },
  };
}

/**
 * Exposes the starfield pass as a shader warmup program (AC-040 seam).
 *
 * The returned entry keeps compilation lazy: `compileAndLink` builds the
 * pass (compiling and linking the program), and `drawOffscreen` issues the
 * three-layer draw through the shared pass path so the driver warms the
 * exact render pipeline used on screen. The context passed to
 * `drawOffscreen` is accepted for interface parity; the pass is bound to the
 * context supplied at construction.
 *
 * @param context - WebGL2 context the pass resources will be built against.
 * @returns Warmup entry keyed as the first pinned registry key.
 *
 * @example
 * const warmup = toStarfieldWarmupProgram(gl);
 * warmup.compileAndLink();
 * warmup.drawOffscreen(gl);
 */
export function toStarfieldWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram {
  let pass: StarfieldPass | null = null;
  const ensurePass = (): StarfieldPass => {
    if (!pass) {
      pass = createStarfieldPass(context);
    }
    return pass;
  };

  return {
    key: 'starfield',
    compileAndLink: (): void => {
      ensurePass();
    },
    drawOffscreen: (): void => {
      ensurePass().draw(ZERO_CAMERA_POSE);
    },
  };
}
