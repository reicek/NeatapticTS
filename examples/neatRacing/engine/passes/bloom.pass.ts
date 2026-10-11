/**
 * NeatRacing cheap bloom post-processing pass (slice P1-S4-E; spec section
 * 6.6 G8, AC-016, AC-035).
 *
 * G8 concretizes the glow-byte bloom pipeline as THREE fullscreen programs
 * behind one pass and three warmup registry keys:
 *
 * 1. `bloom-threshold` (stage 1): the bright pass samples the scene texture
 *    and keeps only pixels above NEATRACING_BLOOM_BRIGHT_THRESHOLD, scaled
 *    by a Rec709 luminance dot (named weights 0.2126 / 0.7152 / 0.0722).
 *    Bright-pass input selection (glow voxels minus starfield/pulses per
 *    section 6.6) is the host's composition contract: this pass consumes
 *    exactly ONE scene texture.
 * 2. `bloom-blur` (stage 2): ONE separable blur, run twice — horizontally
 *    into ping-pong target A, then vertically into ping-pong target B — on
 *    the half-res 640x360 target with NEATRACING_BLOOM_MAX_TAPS = 13
 *    (the spec's named tap cap; the used NEATRACING_BLOOM_TAP_COUNT equals
 *    the cap exactly). Tap offsets/weights are gaussian with
 *    NEATRACING_BLOOM_BLUR_SIGMA_TEXELS, normalized to sum to one, built in
 *    TypeScript by {@link computeBloomBlurTaps} and interpolated as GLSL
 *    const arrays so the shader cannot drift from the TS mirror.
 * 3. `bloom-additive` (stage 3): the composite samples the scene texture
 *    and the blurred bloom and emits
 *    `scene + bloom * NEATRACING_BLOOM_COMPOSITE_INTENSITY` through
 *    `blendFunc(GL_ONE, GL_ONE)` additive blending onto the DEFAULT
 *    framebuffer — which the host has cleared and pre-filled with its
 *    additive background (starfield/pulses drawn direct, pulse-pass
 *    pattern). The world exists per-frame ONLY as the host-owned scene
 *    FBO texture, so this composite draw is the ONLY scene transfer
 *    to the display, and the world is rendered exactly once there.
 *    Never emit bloom-only output (the world would vanish from the
 *    display) and never pre-draw the world to the default framebuffer
 *    (the scene would double).
 *
 * Internal render targets are RGBA8 half-res 640x360 FBOs (bright, blur A,
 * blur B) validated with `checkFramebufferStatus` against
 * `GL_FRAMEBUFFER_COMPLETE` at creation. The AC-035 off-switch is the
 * `enabled === false` frame state: zero GL operations, no viewport change,
 * no state churn. Bloom sits FIRST on the drop list before any LOD swap.
 *
 * Structural sibling of `grid.pass.ts` / `car.pass.ts` (numeric GL enums,
 * exported GLSL ES 3.00 sources, compile helpers that throw with the pass
 * label, throw-on-missing uniform resolution, warmup seam) — with three
 * programs and no vertex-mesh meshing: every stage draws the same shared
 * fullscreen NDC TRIANGLE_STRIP quad.
 *
 * @module
 */

import {
  NEATRACING_WARMED_PROGRAM_KEYS,
  type ShaderWarmupProgram,
} from '../context.service';

/** `gl.COMPILE_STATUS` (0x8B81) checked after every shader compile. */
const GL_COMPILE_STATUS = 0x8b81;

/** `gl.LINK_STATUS` (0x8B82) checked after every program link. */
const GL_LINK_STATUS = 0x8b82;

/** `gl.VERTEX_SHADER` (0x8B31). */
const GL_VERTEX_SHADER_TYPE = 0x8b31;

/** `gl.FRAGMENT_SHADER` (0x8B30). */
const GL_FRAGMENT_SHADER_TYPE = 0x8b30;

/** `gl.ARRAY_BUFFER` (0x8892) target for the fullscreen quad upload. */
const GL_ARRAY_BUFFER = 0x8892;

/** `gl.STATIC_DRAW` (0x88E4) usage for the quad vertex data. */
const GL_STATIC_DRAW = 0x88e4;

/** `gl.FLOAT` (0x1406) attribute component type. */
const GL_FLOAT = 0x1406;

/** `gl.DEPTH_TEST` (0x0B71) disabled inside the fullscreen post pipeline. */
const GL_DEPTH_TEST = 0x0b71;

/** `gl.CULL_FACE` (0x0B44) disabled: fullscreen strips face both ways. */
const GL_CULL_FACE = 0x0b44;

/** `gl.BLEND` (0x0BE2) enabled only for the additive composite stage. */
const GL_BLEND = 0x0be2;

/** `gl.ONE` (1) blend source factor for the G8 additive composite. */
const GL_ONE = 1;

/** `gl.TRIANGLE_STRIP` (5) draw mode for every fullscreen stage. */
const GL_TRIANGLE_STRIP = 5;

/** `gl.TEXTURE_2D` (0x0DE1) target for every bloom texture. */
const GL_TEXTURE_2D = 0x0de1;

/** `gl.RGBA` (0x1908) internal format and pixel format of every target. */
const GL_RGBA = 0x1908;

/** `gl.UNSIGNED_BYTE` (0x1401) texel type of every bloom target. */
const GL_UNSIGNED_BYTE = 0x1401;

/** `gl.LINEAR` (0x2601) filter for blur/upsample bilinearity. */
const GL_LINEAR = 0x2601;

/** `gl.CLAMP_TO_EDGE` (0x812F) wrap mode for every bloom target. */
const GL_CLAMP_TO_EDGE = 0x812f;

/** `gl.TEXTURE_MIN_FILTER` (0x2801) sampler parameter. */
const GL_TEXTURE_MIN_FILTER = 0x2801;

/** `gl.TEXTURE_MAG_FILTER` (0x2800) sampler parameter. */
const GL_TEXTURE_MAG_FILTER = 0x2800;

/** `gl.TEXTURE_WRAP_S` (0x2802) sampler parameter. */
const GL_TEXTURE_WRAP_S = 0x2802;

/** `gl.TEXTURE_WRAP_T` (0x2803) sampler parameter. */
const GL_TEXTURE_WRAP_T = 0x2803;

/** `gl.FRAMEBUFFER` (0x8D40) target of every framebuffer bind. */
const GL_FRAMEBUFFER = 0x8d40;

/** `gl.COLOR_ATTACHMENT0` (0x8CE0) attachment point of every target. */
const GL_COLOR_ATTACHMENT0 = 0x8ce0;

/** `gl.FRAMEBUFFER_COMPLETE` (0x8CD5) required after every FBO setup. */
const GL_FRAMEBUFFER_COMPLETE = 0x8cd5;

/** `gl.TEXTURE0` (0x84C0) first sampler unit (scene). */
const GL_TEXTURE0 = 0x84c0;

/** `gl.TEXTURE1` (0x84C1) second sampler unit (blurred bloom). */
const GL_TEXTURE1 = 0x84c1;

/** Pass label prefixing every compile/link/FBO rejection message. */
const NEATRACING_BLOOM_PASS_LABEL = 'neatRacing bloom pass';

/**
 * Half-res bloom target width (G8: "half-res 640x360 target" derived from
 * the fixed 720p backing store's 1280 width; NEATRACING_FALLBACK_CANVAS_
 * WIDTH / 2).
 */
export const NEATRACING_BLOOM_TARGET_WIDTH = 640;

/**
 * Half-res bloom target height (G8: derived from the fixed 720p backing
 * store height; NEATRACING_FIXED_RENDER_HEIGHT / 2).
 */
export const NEATRACING_BLOOM_TARGET_HEIGHT = 360;

/**
 * Named tap cap (G8: "named tap cap <= 13 taps"); per-direction tap counts
 * must never exceed this value.
 */
export const NEATRACING_BLOOM_MAX_TAPS = 13;

/**
 * Tap count the separable blur actually uses per direction; pinned at the
 * cap (13, odd so the gaussian keeps a center tap).
 */
export const NEATRACING_BLOOM_TAP_COUNT = NEATRACING_BLOOM_MAX_TAPS;

/** Gaussian sigma of the separable blur kernel, in half-res texels. */
export const NEATRACING_BLOOM_BLUR_SIGMA_TEXELS = 2.5;

/**
 * Bright-pass luminance threshold in linear [0, 1] units (G8 stage 1):
 * pixels luminance-below this value contribute zero bloom.
 */
export const NEATRACING_BLOOM_BRIGHT_THRESHOLD = 0.72;

/**
 * Composite intensity multiplier for the additive bloom contribution
 * (G8 stage 3: additive composite onto the scene, TP-10 tuning value).
 */
export const NEATRACING_BLOOM_COMPOSITE_INTENSITY = 0.35;

/**
 * Rec709 luminance weights used by both the TS threshold mirror and the
 * interpolated GLSL const (shader cannot drift from the TS math).
 */
export const NEATRACING_BLOOM_LUMA_WEIGHTS: readonly [number, number, number] =
  [0.2126, 0.7152, 0.0722];

/** Warmup program key (AC-040): threshold stage, registry index 6. */
export const NEATRACING_BLOOM_THRESHOLD_PROGRAM_KEY =
  NEATRACING_WARMED_PROGRAM_KEYS[6];

/** Warmup program key (AC-040): separable blur stage, registry index 7. */
export const NEATRACING_BLOOM_BLUR_PROGRAM_KEY =
  NEATRACING_WARMED_PROGRAM_KEYS[7];

/** Warmup program key (AC-040): additive composite stage, registry index 8. */
export const NEATRACING_BLOOM_ADDITIVE_PROGRAM_KEY =
  NEATRACING_WARMED_PROGRAM_KEYS[8];

/** One blur tap: separable offset in texels plus its normalized weight. */
export interface BloomBlurTap {
  /** Offset of the tap in texels (integer, from -(N-1)/2 to +(N-1)/2). */
  readonly offsetTexels: number;
  /** Gaussian weight normalized so all taps sum to one. */
  readonly weight: number;
}

/**
 * Computes the Rec709 luminance of one color triple (pure, GL-free).
 *
 * @param red - Linear channel in [0, 1].
 * @param green - Linear channel in [0, 1].
 * @param blue - Linear channel in [0, 1].
 * @returns The Rec709 luminance dot with NEATRACING_BLOOM_LUMA_WEIGHTS.
 *
 * @example
 * computeBloomLuminance(1, 0, 0); // 0.2126
 */
export function computeBloomLuminance(
  red: number,
  green: number,
  blue: number,
): number {
  const [weightRed, weightGreen, weightBlue] = NEATRACING_BLOOM_LUMA_WEIGHTS;
  return weightRed * red + weightGreen * green + weightBlue * blue;
}

/**
 * Computes the bright-pass factor of one scene pixel (pure, GL-free mirror
 * of the threshold shader). Pixels below the threshold are dropped to zero;
 * above it, the factor ramps linearly to one at full white.
 *
 * @param red - Linear channel in [0, 1].
 * @param green - Linear channel in [0, 1].
 * @param blue - Linear channel in [0, 1].
 * @returns The bright-pass multiplier clamped to [0, 1].
 *
 * @example
 * computeBloomBrightFactor(0, 0, 0); // 0 — black never blooms (LD-23)
 * computeBloomBrightFactor(1, 1, 1); // 1
 */
export function computeBloomBrightFactor(
  red: number,
  green: number,
  blue: number,
): number {
  const luminance = computeBloomLuminance(red, green, blue);
  const thresholdSpan = 1 - NEATRACING_BLOOM_BRIGHT_THRESHOLD;
  if (luminance <= NEATRACING_BLOOM_BRIGHT_THRESHOLD) {
    return 0;
  }
  return Math.min(
    1,
    (luminance - NEATRACING_BLOOM_BRIGHT_THRESHOLD) / thresholdSpan,
  );
}

/**
 * Formats one tap number as a GLSL float literal (ES ES requires a decimal
 * point or exponent; integer offsets gain an explicit `.0` suffix).
 *
 * @param value - Tap value to format.
 * @returns A GLSL-compatible float literal.
 */
function formatBloomTapFloat(value: number): string {
  return value === Math.trunc(value) ? `${value}.0` : String(value);
}

/**
 * Builds the separable blur tap table (pure, GL-free): a symmetric gaussian
 * kernel of {@link NEATRACING_BLOOM_TAP_COUNT} taps centered on zero with
 * weights normalized to sum to one, capped by
 * {@link NEATRACING_BLOOM_MAX_TAPS}. GLSL const arrays interpolated from
 * this table guarantee the TS mirror and the shader cannot drift.
 *
 * @returns One tap per lattice offset from -6 to +6 texels (inclusive).
 *
 * @example
 * const taps = computeBloomBlurTaps();
 * // taps.length === 13; taps[0].offsetTexels === -6; center weight is the
 * // maximum; Math.hypot(taps.reduce(...) ... sums to one within 1e-12.
 */
export function computeBloomBlurTaps(): readonly BloomBlurTap[] {
  const halfWidth = Math.trunc(NEATRACING_BLOOM_TAP_COUNT / 2);
  const sigma = NEATRACING_BLOOM_BLUR_SIGMA_TEXELS;
  const unnormalized: number[] = [];
  let weightSum = 0;
  for (let offset = -halfWidth; offset <= halfWidth; offset += 1) {
    const scaled = offset / sigma;
    const weight = Math.exp(-0.5 * scaled * scaled);
    unnormalized.push(weight);
    weightSum += weight;
  }
  return unnormalized.map((weight, index) => ({
    offsetTexels: index - halfWidth,
    weight: weight / weightSum,
  }));
}

/** Threshold-program uniform names resolved with throw-on-missing semantics. */
const BLOOM_THRESHOLD_UNIFORM_NAMES: readonly string[] = ['uScene'];

/** Blur-program uniform names (throw-on-missing). */
const BLOOM_BLUR_UNIFORM_NAMES: readonly string[] = [
  'uBlurTexture',
  'uBlurDirection',
];

/** Composite-program uniform names (throw-on-missing). */
const BLOOM_ADDITIVE_UNIFORM_NAMES: readonly string[] = [
  'uScene',
  'uBloomTexture',
];

/**
 * Fullscreen vertex shader shared by all three bloom stages: maps the unit
 * NDC strip corner to UV space.
 */
export const NEATRACING_BLOOM_VERTEX_SHADER_SOURCE = `#version 300 es
in vec2 aCorner;
out vec2 vUv;
void main() {
  vUv = aCorner * 0.5 + vec2(0.5, 0.5);
  gl_Position = vec4(aCorner, 0.0, 1.0);
}`;

/**
 * Threshold fragment shader (G8 stage 1): Rec709 luminance dot interpolated
 * from the TS mirror constants so the shader cannot drift.
 */
export const NEATRACING_BLOOM_THRESHOLD_FRAGMENT_SHADER_SOURCE = `#version 300 es
precision highp float;
uniform sampler2D uScene;
const vec3 kLumaWeights = vec3(${NEATRACING_BLOOM_LUMA_WEIGHTS.map(String).join(', ')});
const float kBrightnessThreshold = ${NEATRACING_BLOOM_BRIGHT_THRESHOLD};
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec3 scene = texture(uScene, vUv).rgb;
  float luminance = dot(kLumaWeights, scene);
  float factor = clamp(
    (luminance - kBrightnessThreshold) / (1.0 - kBrightnessThreshold),
    0.0,
    1.0
  );
  fragColor = vec4(scene * factor, 1.0);
}`;

/**
 * Separable blur fragment shader (G8 stage 2): sums
 * {@link NEATRACING_BLOOM_TAP_COUNT} gaussian taps mirrored from
 * {@link computeBloomBlurTaps} as GLSL const arrays, offset along
 * `uBlurDirection` (set to (1/W, 0) or (0, 1/H) per draw).
 */
export const NEATRACING_BLOOM_BLUR_FRAGMENT_SHADER_SOURCE = `#version 300 es
precision highp float;
uniform sampler2D uBlurTexture;
uniform vec2 uBlurDirection;
const int kBlurTapCount = ${NEATRACING_BLOOM_TAP_COUNT};
const float kBlurOffsets[kBlurTapCount] = float[kBlurTapCount](${computeBloomBlurTaps()
  .map((tap) => formatBloomTapFloat(tap.offsetTexels))
  .join(', ')});
const float kBlurWeights[kBlurTapCount] = float[kBlurTapCount](${computeBloomBlurTaps()
  .map((tap) => formatBloomTapFloat(tap.weight))
  .join(', ')});
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec3 sum = vec3(0.0);
  for (int tap = 0; tap < kBlurTapCount; tap += 1) {
    sum += kBlurWeights[tap] * texture(
      uBlurTexture,
      vUv + uBlurDirection * kBlurOffsets[tap]
    ).rgb;
  }
  fragColor = vec4(sum, 1.0);
}`;

/**
 * Additive composite fragment shader (G8 stage 3): emits the full
 * `scene + bloom * kCompositeIntensity` color, which the pass draws
 * additively onto the default framebuffer that already holds the host's
 * additive background (starfield/pulses drawn direct). This composite is
 * the ONLY scene transfer to the display — the world is rendered exactly
 * once into the host-owned scene FBO texture this shader samples.
 */
export const NEATRACING_BLOOM_ADDITIVE_FRAGMENT_SHADER_SOURCE = `#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform sampler2D uBloomTexture;
const float kCompositeIntensity = ${NEATRACING_BLOOM_COMPOSITE_INTENSITY};
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec3 scene = texture(uScene, vUv).rgb;
  vec3 bloom = texture(uBloomTexture, vUv).rgb;
  fragColor = vec4(scene + bloom * kCompositeIntensity, 1.0);
}`;

/** One internal half-res render target: an RGBA texture plus complete FBO. */
interface BloomRenderTarget {
  /** Framebuffer the stage renders into. */
  readonly framebuffer: WebGLFramebuffer;
  /** Texture holding the color attachment. */
  readonly texture: WebGLTexture;
}

/** Compiled program bundle: the linked program plus its two live shaders. */
interface CompiledBloomProgram {
  /** Linked program handle. */
  readonly program: WebGLProgram;
  /** Live vertex shader handle (released by dispose). */
  readonly vertexShader: WebGLShader;
  /** Live fragment shader handle (released by dispose). */
  readonly fragmentShader: WebGLShader;
}

/** Per-frame draw state accepted by BloomPass.draw. */
export interface BloomPassDrawFrame {
  /** Scene texture the threshold + composite stages sample. */
  readonly sceneTexture: WebGLTexture;
  /** Composite viewport width (display backing store width in pixels). */
  readonly targetWidth: number;
  /** Height of the composite viewport in pixels. */
  readonly targetHeight: number;
  /**
   * AC-035 off-switch: `false` skips every GL operation (Bloom is first on
   * the adaptive-quality drop list, before any LOD swap).
   */
  readonly enabled: boolean;
}

/** Bloom pass handle returned by createBloomPass. */
export interface BloomPass {
  /**
   * Runs the three-stage pipeline when enabled: threshold -> separable blur
   * (horizontal + vertical on half-res ping-pong targets) -> additive
   * composite onto the default framebuffer at the frame's viewport. When
   * `enabled === false` ZERO GL operations are issued.
   *
   * @param frame - Frame draw state (see {@link BloomPassDrawState}).
   * @throws When targetWidth/targetHeight are not positive integers.
   */
  draw(frame: BloomPassDrawFrame): void;
  /** Releases all three programs, six shaders, targets, and the quad. */
  dispose(): void;
}

/**
 * Compiles one bloom pass shader stage, rejecting failures with the pass
 * label (grid.pass.ts parity).
 *
 * @param gl - WebGL2 context.
 * @param shaderType - gl.VERTEX_SHADER (0x8B31) or gl.FRAGMENT_SHADER (0x8B30).
 * @param shaderSource - GLSL ES 3.00 source.
 * @param shaderLabel - Stage label used in the rejection message.
 * @returns The compiled shader handle.
 * @throws When compilation fails; the message carries the pass label.
 */
function compileBloomShader(
  gl: WebGL2RenderingContext,
  shaderType: number,
  shaderSource: string,
  shaderLabel: string,
): WebGLShader {
  const shader = gl.createShader(shaderType);
  if (shader === null) {
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: failed to create the ${shaderLabel} shader`,
    );
  }
  gl.shaderSource(shader, shaderSource);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, GL_COMPILE_STATUS)) {
    const infoLog = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: failed to compile the ${shaderLabel} shader: ${String(infoLog)}`,
    );
  }
  return shader;
}

/**
 * Compiles and links one bloom stage program from its two stages.
 *
 * @param gl - WebGL2 context.
 * @param fragmentShaderSource - GLSL ES 3.00 fragment source.
 * @param programLabel - Label used in every rejection message.
 * @returns The linked program plus its two shader handles.
 * @throws When compilation or linking fails.
 */
function compileAndLinkBloomProgram(
  gl: WebGL2RenderingContext,
  fragmentShaderSource: string,
  programLabel: string,
): CompiledBloomProgram {
  const vertexShader = compileBloomShader(
    gl,
    GL_VERTEX_SHADER_TYPE,
    NEATRACING_BLOOM_VERTEX_SHADER_SOURCE,
    `${programLabel} vertex`,
  );
  const fragmentShader = compileBloomShader(
    gl,
    GL_FRAGMENT_SHADER_TYPE,
    fragmentShaderSource,
    `${programLabel} fragment`,
  );
  const program = gl.createProgram();
  if (program === null) {
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: failed to create the ${programLabel} program`,
    );
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, GL_LINK_STATUS)) {
    const infoLog = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: failed to link the ${programLabel} program: ${String(infoLog)}`,
    );
  }
  return { program, vertexShader, fragmentShader };
}

/**
 * Resolves one program's uniform locations with throw-on-missing semantics
 * so a shader/name drift fails loudly at pass creation.
 *
 * @param gl - WebGL2 context.
 * @param program - Linked program handle.
 * @param names - Uniform names to resolve, in feed order.
 * @returns Location table keyed by uniform name.
 * @throws When any uniform is missing from the linked program.
 */
function resolveBloomUniformLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  names: readonly string[],
): Record<string, WebGLUniformLocation> {
  const locations: Record<string, WebGLUniformLocation> = {};
  for (const name of names) {
    const location = gl.getUniformLocation(program, name);
    if (location === null) {
      throw new Error(
        `${NEATRACING_BLOOM_PASS_LABEL}: missing uniform ${name}`,
      );
    }
    locations[name] = location;
  }
  return locations;
}

/**
 * Resolves the `aCorner` fullscreen attribute for one program.
 *
 * @param gl - WebGL2 context.
 * @param program - Linked program handle.
 * @param programLabel - Stage label used in the rejection message.
 * @returns The attribute location for `aCorner`.
 * @throws When the attribute is missing from the linked program.
 */
function resolveBloomCornerAttributeId(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  programLabel: string,
): number {
  const attributeId = gl.getAttribLocation(program, 'aCorner');
  if (attributeId < 0) {
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: missing attribute aCorner (${programLabel})`,
    );
  }
  return attributeId;
}

/**
 * Creates one half-res RGBA8 render target validated with
 * `checkFramebufferStatus`.
 *
 * @param gl - WebGL2 context.
 * @param label - Label used in every rejection message.
 * @returns The framebuffer + texture bundle.
 * @throws When texture/framebuffer creation or completeness fails.
 */
function createBloomRenderTarget(
  gl: WebGL2RenderingContext,
  label: string,
): BloomRenderTarget {
  const texture = gl.createTexture();
  if (texture === null) {
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: failed to create the ${label} texture`,
    );
  }
  gl.bindTexture(GL_TEXTURE_2D, texture);
  gl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl.texParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  gl.texImage2D(
    GL_TEXTURE_2D,
    0,
    GL_RGBA,
    NEATRACING_BLOOM_TARGET_WIDTH,
    NEATRACING_BLOOM_TARGET_HEIGHT,
    0,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    null,
  );
  const framebuffer = gl.createFramebuffer();
  if (framebuffer === null) {
    gl.deleteTexture(texture);
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: failed to create the ${label} framebuffer`,
    );
  }
  gl.bindFramebuffer(GL_FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(
    GL_FRAMEBUFFER,
    GL_COLOR_ATTACHMENT0,
    GL_TEXTURE_2D,
    texture,
    0,
  );
  const status = gl.checkFramebufferStatus(GL_FRAMEBUFFER);
  if (status !== GL_FRAMEBUFFER_COMPLETE) {
    gl.deleteFramebuffer(framebuffer);
    gl.deleteTexture(texture);
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: ${label} framebuffer incomplete (status ${String(status)})`,
    );
  }
  return { framebuffer, texture };
}

/**
 * Uploads the shared fullscreen NDC TRIANGLE_STRIP quad corner data into a
 * STATIC_DRAW buffer used by all three stages.
 *
 * @param gl - WebGL2 context.
 * @returns The uploaded buffer handle.
 * @throws When gl.createBuffer returns null.
 */
function uploadBloomFullscreenQuad(gl: WebGL2RenderingContext): WebGLBuffer {
  const buffer = gl.createBuffer();
  if (buffer === null) {
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: failed to create the fullscreen quad buffer`,
    );
  }
  gl.bindBuffer(GL_ARRAY_BUFFER, buffer);
  const corners = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  gl.bufferData(GL_ARRAY_BUFFER, corners, GL_STATIC_DRAW);
  return buffer;
}

/**
 * Creates the bloom post-processing pass: compiles the threshold, separable
 * blur, and additive composite programs eagerly, resolves uniforms and the
 * `aCorner` attribute throw-on-missing, uploads the shared fullscreen quad,
 * and creates the three half-res 640x360 internal targets with
 * completeness checks (bright, blur ping-pong A, blur ping-pong B).
 *
 * @param gl - WebGL2 context.
 * @returns The bloom pass handle ({@link BloomPass}).
 * @throws When shaders fail to compile/link, uniforms are absent, or an
 *   internal framebuffer is incomplete.
 *
 * @example
 * const bloomPass = createBloomPass(gl);
 * bloomPass.draw({ sceneTexture, targetWidth: 1280, targetHeight: 720, enabled: true });
 * bloomPass.dispose();
 */
export function createBloomPass(gl: WebGL2RenderingContext): BloomPass {
  const thresholdCompiled = compileAndLinkBloomProgram(
    gl,
    NEATRACING_BLOOM_THRESHOLD_FRAGMENT_SHADER_SOURCE,
    'bloom threshold',
  );
  const blurCompiled = compileAndLinkBloomProgram(
    gl,
    NEATRACING_BLOOM_BLUR_FRAGMENT_SHADER_SOURCE,
    'bloom blur',
  );
  const additiveCompiled = compileAndLinkBloomProgram(
    gl,
    NEATRACING_BLOOM_ADDITIVE_FRAGMENT_SHADER_SOURCE,
    'bloom additive',
  );
  const thresholdUniforms = resolveBloomUniformLocations(
    gl,
    thresholdCompiled.program,
    BLOOM_THRESHOLD_UNIFORM_NAMES,
  );
  const blurUniforms = resolveBloomUniformLocations(
    gl,
    blurCompiled.program,
    BLOOM_BLUR_UNIFORM_NAMES,
  );
  const additiveUniforms = resolveBloomUniformLocations(
    gl,
    additiveCompiled.program,
    BLOOM_ADDITIVE_UNIFORM_NAMES,
  );
  const thresholdCornerAttributeId = resolveBloomCornerAttributeId(
    gl,
    thresholdCompiled.program,
    'bloom threshold',
  );
  const blurCornerAttributeId = resolveBloomCornerAttributeId(
    gl,
    blurCompiled.program,
    'bloom blur',
  );
  const additiveCornerAttributeId = resolveBloomCornerAttributeId(
    gl,
    additiveCompiled.program,
    'bloom additive',
  );
  const brightTarget = createBloomRenderTarget(gl, 'bloom bright');
  const blurTargetA = createBloomRenderTarget(gl, 'bloom blur A');
  const blurTargetB = createBloomRenderTarget(gl, 'bloom blur B');
  const quadBuffer = uploadBloomFullscreenQuad(gl);

  /**
   * Binds the shared quad on the currently active program and points the
   * two-component `aCorner` attribute pointer (buffer-coupled client state
   * without a VAO).
   *
   * @param cornerAttributeId - The program's `aCorner` location.
   */
  function pointFullscreenQuad(cornerAttributeId: number): void {
    gl.bindBuffer(GL_ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(cornerAttributeId);
    gl.vertexAttribPointer(cornerAttributeId, 2, GL_FLOAT, false, 0, 0);
  }

  /**
   * Binds one texture on the given sampler unit.
   *
   * @param unit - `gl.TEXTURE0` (0x84C0) or `gl.TEXTURE1` (0x84C1).
   * @param texture - Texture to bind on the unit.
   */
  function bindBloomTexture(unit: number, texture: WebGLTexture): void {
    gl.activeTexture(unit);
    gl.bindTexture(GL_TEXTURE_2D, texture);
  }

  return {
    draw(frame: BloomPassDrawFrame): void {
      // AC-035 off-switch: an explicit false skips EVERY GL operation so the
      // quality governor can drop bloom with no state churn.
      if (!frame.enabled) {
        return;
      }
      if (
        !Number.isInteger(frame.targetWidth) ||
        !Number.isInteger(frame.targetHeight) ||
        frame.targetWidth <= 0 ||
        frame.targetHeight <= 0
      ) {
        throw new Error(
          `${NEATRACING_BLOOM_PASS_LABEL}: target width/height must be positive integers, received ${String(frame.targetWidth)}x${String(frame.targetHeight)}`,
        );
      }
      // Pass-local depth state is intentionally not restored: the Step 08
      // host renderer owns frame-level state (glass pass G4 parity).
      gl.depthMask(false);
      gl.disable(GL_DEPTH_TEST);
      gl.disable(GL_CULL_FACE);
      gl.disable(GL_BLEND);
      gl.viewport(
        0,
        0,
        NEATRACING_BLOOM_TARGET_WIDTH,
        NEATRACING_BLOOM_TARGET_HEIGHT,
      );

      // Stage 1: bright pass, scene -> bright target.
      gl.bindFramebuffer(GL_FRAMEBUFFER, brightTarget.framebuffer);
      gl.useProgram(thresholdCompiled.program);
      pointFullscreenQuad(thresholdCornerAttributeId);
      bindBloomTexture(GL_TEXTURE0, frame.sceneTexture);
      gl.uniform1i(thresholdUniforms.uScene, 0);
      gl.drawArrays(GL_TRIANGLE_STRIP, 0, 4);

      // Separable blur direction: horizontal texel step, then into target A.
      gl.bindFramebuffer(GL_FRAMEBUFFER, blurTargetA.framebuffer);
      gl.useProgram(blurCompiled.program);
      pointFullscreenQuad(blurCornerAttributeId);
      bindBloomTexture(GL_TEXTURE0, brightTarget.texture);
      gl.uniform1i(blurUniforms.uBlurTexture, 0);
      gl.uniform2f(
        blurUniforms.uBlurDirection,
        1 / NEATRACING_BLOOM_TARGET_WIDTH,
        0,
      );
      gl.drawArrays(GL_TRIANGLE_STRIP, 0, 4);

      // Vertical texel step: A -> B completes the ONE separable pass.
      gl.bindFramebuffer(GL_FRAMEBUFFER, blurTargetB.framebuffer);
      bindBloomTexture(GL_TEXTURE0, blurTargetA.texture);
      gl.uniform2f(
        blurUniforms.uBlurDirection,
        0,
        1 / NEATRACING_BLOOM_TARGET_HEIGHT,
      );
      gl.drawArrays(GL_TRIANGLE_STRIP, 0, 4);

      // Additive composite onto the caller's display framebuffer — the
      // default FB holds only the host's additive background (starfield/
      // pulses drawn direct, pulse-pass pattern); the scene arrives only
      // as the scene texture, making this the sole scene transfer.
      gl.bindFramebuffer(GL_FRAMEBUFFER, null);
      gl.viewport(0, 0, frame.targetWidth, frame.targetHeight);
      gl.enable(GL_BLEND);
      gl.blendFunc(GL_ONE, GL_ONE);
      gl.useProgram(additiveCompiled.program);
      pointFullscreenQuad(additiveCornerAttributeId);
      bindBloomTexture(GL_TEXTURE0, frame.sceneTexture);
      bindBloomTexture(GL_TEXTURE1, blurTargetB.texture);
      gl.uniform1i(additiveUniforms.uScene, 0);
      gl.uniform1i(additiveUniforms.uBloomTexture, 1);
      gl.drawArrays(GL_TRIANGLE_STRIP, 0, 4);
      gl.disable(GL_BLEND);
    },
    dispose(): void {
      gl.deleteBuffer(quadBuffer);
      for (const target of [brightTarget, blurTargetA, blurTargetB]) {
        gl.deleteFramebuffer(target.framebuffer);
        gl.deleteTexture(target.texture);
      }
      gl.deleteProgram(thresholdCompiled.program);
      gl.deleteProgram(blurCompiled.program);
      gl.deleteProgram(additiveCompiled.program);
      gl.deleteShader(thresholdCompiled.vertexShader);
      gl.deleteShader(thresholdCompiled.fragmentShader);
      gl.deleteShader(blurCompiled.vertexShader);
      gl.deleteShader(blurCompiled.fragmentShader);
      gl.deleteShader(additiveCompiled.vertexShader);
      gl.deleteShader(additiveCompiled.fragmentShader);
    },
  };
}

/** Warmup-only display viewport (the shared runShaderWarmup 1x1 target). */
const NEATRACING_BLOOM_WARMUP_VIEWPORT_SIZE = 1;

/** Opaque-white RGBA bytes for the 1x1 warmup scene placeholder. */
const NEATRACING_BLOOM_WARMUP_SCENE_TEXELS = new Uint8Array([
  255, 255, 255, 255,
]);

/** Lazily hydrated singleton pass behind the warmup seam (car parity). */
let warmupBloomPass: BloomPass | null = null;

/** Context the current warmup singleton was built against. */
let warmupBloomPassContext: WebGL2RenderingContext | null = null;

/** Lazily created 1x1 white scene placeholder texture for warmup frames. */
let warmupBloomSceneTexture: WebGLTexture | null = null;

/**
 * Releases the warmup singleton (pass + placeholder texture) — used when
 * the context changes so stale GL handles are never reused.
 */
function disposeWarmupBloomPass(): void {
  if (warmupBloomPass === null) {
    return;
  }
  warmupBloomPass.dispose();
  if (warmupBloomSceneTexture !== null && warmupBloomPassContext !== null) {
    warmupBloomPassContext.deleteTexture(warmupBloomSceneTexture);
  }
  warmupBloomPass = null;
  warmupBloomSceneTexture = null;
  warmupBloomPassContext = null;
}

/**
 * Hydrates the singleton bloom pass for the warmup seam, rebuilding only
 * when the context changes. Compilation stays lazy until compileAndLink
 * runs (car.pass.ts parity); the scene placeholder is created on the passed
 * context.
 *
 * @param context - WebGL2 context the warmup shaders will run against.
 * @returns The hydrated singleton pass.
 */
function ensureWarmupBloomPass(context: WebGL2RenderingContext): BloomPass {
  if (
    warmupBloomPass !== null &&
    warmupBloomSceneTexture !== null &&
    warmupBloomPassContext === context
  ) {
    return warmupBloomPass;
  }
  disposeWarmupBloomPass();
  const texture = context.createTexture();
  if (texture === null) {
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: failed to create the warmup scene placeholder`,
    );
  }
  context.bindTexture(GL_TEXTURE_2D, texture);
  context.texImage2D(
    GL_TEXTURE_2D,
    0,
    GL_RGBA,
    NEATRACING_BLOOM_WARMUP_VIEWPORT_SIZE,
    NEATRACING_BLOOM_WARMUP_VIEWPORT_SIZE,
    0,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    NEATRACING_BLOOM_WARMUP_SCENE_TEXELS,
  );
  warmupBloomPass = createBloomPass(context);
  warmupBloomSceneTexture = texture;
  warmupBloomPassContext = context;
  return warmupBloomPass;
}

/**
 * Builds the three bloom warmup program descriptors for the shared warmup
 * registry (keys 'bloom-threshold', 'bloom-blur', 'bloom-additive' —
 * NEATRACING_WARMED_PROGRAM_KEYS[6..8]). Compilation is lazy: no GL program
 * exists until compileAndLink runs, and each drawOffscreen pushes the
 * enabled zero-pose frame through the shared four-draw pipeline so all
 * stages warm in the sweep.
 *
 * @param context - WebGL2 context the warmup shaders will run against.
 * @returns The three warmup program descriptors in registry-key order.
 *
 * @example
 * for (const program of toBloomWarmupPrograms(gl)) {
 *   programRegistry.register(program);
 * }
 */
export function toBloomWarmupPrograms(
  context: WebGL2RenderingContext,
): readonly [
  threshold: ShaderWarmupProgram,
  blur: ShaderWarmupProgram,
  additive: ShaderWarmupProgram,
] {
  return [
    {
      key: NEATRACING_BLOOM_THRESHOLD_PROGRAM_KEY,
      compileAndLink: (): void => {
        ensureWarmupBloomPass(context);
      },
      drawOffscreen: (): void => {
        const pass = ensureWarmupBloomPass(context);
        pass.draw(createWarmupZeroPoseFrame());
      },
    },
    {
      key: NEATRACING_BLOOM_BLUR_PROGRAM_KEY,
      compileAndLink: (): void => {
        ensureWarmupBloomPass(context);
      },
      drawOffscreen: (): void => {
        const pass = ensureWarmupBloomPass(context);
        pass.draw(createWarmupZeroPoseFrame());
      },
    },
    {
      key: NEATRACING_BLOOM_ADDITIVE_PROGRAM_KEY,
      compileAndLink: (): void => {
        ensureWarmupBloomPass(context);
      },
      drawOffscreen: (): void => {
        const pass = ensureWarmupBloomPass(context);
        pass.draw(createWarmupZeroPoseFrame());
      },
    },
  ];
}

/**
 * Builds the enabled zero-pose warmup frame (1x1 white placeholder source,
 * 1x1 composite viewport) against the hydrated singleton's placeholder
 * texture so the warmup drawOffscreen exercises the full 4-draw pipeline
 * (AC-040).
 *
 * @returns The `enabled` warmup draw frame.
 * @throws When the warmup singleton has not been hydrated.
 */
function createWarmupZeroPoseFrame(): BloomPassDrawFrame {
  if (warmupBloomPass === null || warmupBloomSceneTexture === null) {
    throw new Error(
      `${NEATRACING_BLOOM_PASS_LABEL}: warmup pass is not hydrated`,
    );
  }
  return {
    sceneTexture: warmupBloomSceneTexture,
    targetWidth: NEATRACING_BLOOM_WARMUP_VIEWPORT_SIZE,
    targetHeight: NEATRACING_BLOOM_WARMUP_VIEWPORT_SIZE,
    enabled: true,
  };
}
