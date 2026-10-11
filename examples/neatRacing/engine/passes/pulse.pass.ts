/**
 * NeatRacing ambient pulse pass (AC-006 / spec §6.4.3, slice P1-S2-C).
 *
 * Renders non-interactive, world-space pulse rings on the track ground
 * plane: each pulse is a `#B7FF00` (185, 255, 0) glow ring at alpha 0.42
 * (mirroring the neatenstein `PULSE_GLOW_ALPHA` constant) that expands from
 * the ring start radius to the ring end radius over a fixed lifetime while
 * its alpha follows a deterministic attack-then-decay envelope. Placement
 * is derived deterministically from the TP-20 visual seed, emission follows
 * a fixed cadence capped at the concurrency limit, and every pulse is
 * depth-tested (GL_LESS semantics) so world geometry occludes it.
 *
 * Structure follows the orchestration-first layout: the public pure-math
 * contract (constants, types, envelope, radius, placement, cadence,
 * visuals, depth predicate) is testable in a plain Node Jest environment,
 * while the WebGL2 orchestration (`createPulsePass`, `toPulseWarmupProgram`,
 * draw loop, GLSL sources, GL helpers) sits below the fold and is exercised
 * through the recording mock-context harness in `pulse.pass.test.ts`.
 *
 * The pass is deliberately stateless between frames: active pulses are
 * re-collected from the simulation timestamp every draw (pure time-window
 * model), which keeps the visual output deterministic and replayable and
 * avoids incremental spawn state on the render side.
 */
import type { ShaderWarmupProgram } from '../context.service';

/** Hex token of the pulse glow color, as written in spec §6.4.3. */
export const NEATRACING_PULSE_COLOR_HEX = '#B7FF00';

/**
 * RGB decomposition of {@link NEATRACING_PULSE_COLOR_HEX} (185, 255, 0),
 * normalized to the `[0, 1]` range the fragment shader consumes: the shader
 * multiplies the raw channels, so 0-255 values would clamp the glow to
 * yellow-white (the defect the starfield/grid passes avoid by feeding
 * normalized channels).
 *
 * Spec defect (flagged to the spec owner, intentionally not "fixed" here):
 * the spec pins the hex token `#B7FF00` (hex 0xB7 decodes to 183) and the
 * decimal triplet (185, 255, 0) in the same section. The decimal pin stays
 * authoritative for this pass, so 185 is kept.
 */
export const NEATRACING_PULSE_COLOR_RGB: readonly [number, number, number] = [
  185 / 255,
  255 / 255,
  0,
];

/** Glow alpha at the envelope peak; mirrors the neatenstein 0.42 constant. */
export const NEATRACING_PULSE_GLOW_ALPHA = 0.42;

/** Pulse lifetime in milliseconds; drives both envelope and cadence windows. */
export const NEATRACING_PULSE_LIFETIME_MS = 2_000;

/** Fixed cadence between pulse emissions in milliseconds. */
export const NEATRACING_PULSE_AMBIENT_INTERVAL_MS = 500;

/** Maximum number of pulses kept alive at once; defensive cap. */
export const NEATRACING_PULSE_MAX_CONCURRENT = 4;

/** Ring radius in meters at pulse birth. */
export const NEATRACING_PULSE_RING_START_RADIUS_M = 0.5;

/** Ring radius in meters at pulse end of lifetime. */
export const NEATRACING_PULSE_RING_END_RADIUS_M = 6;

/** Fraction of the lifetime spent ramping up to the glow alpha peak. */
export const NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION = 0.15;

/** Half extent in meters of the square spawn bounds around the origin. */
export const NEATRACING_PULSE_SPAWN_HALF_EXTENT_M = 40;

/** Default visual seed (TP-20) used to derive deterministic placement. */
export const NEATRACING_PULSE_VISUAL_SEED_DEFAULT = 1337;

/** Ground lift in meters keeping the ring coplanar-ish above the ground. */
export const NEATRACING_PULSE_GROUND_LIFT_M = 0.01;

/** Warmup/draw registry key; must stay a member of the program registry. */
export const NEATRACING_PULSE_PROGRAM_KEY = 'pulse';

/**
 * World-space origin of a pulse, derived from the visual seed.
 *
 * @property x - Origin X coordinate in meters within the spawn bounds.
 * @property z - Origin Z coordinate in meters within the spawn bounds.
 */
export interface PulseOrigin {
  x: number;
  z: number;
}

/**
 * A single active ambient pulse collected for a frame.
 *
 * @property spawnIndex - Monotonic emission index the pulse was born at.
 * @property originX - Deterministic origin X in meters.
 * @property originZ - Deterministic origin Z in meters.
 * @property ageMs - Milliseconds elapsed since emission at collection time.
 * @property active - Always `true` for collected pulses; alive by contract.
 */
export interface AmbientPulse {
  spawnIndex: number;
  originX: number;
  originZ: number;
  ageMs: number;
  active: boolean;
}

/**
 * Renderable visuals resolved for one pulse.
 *
 * @property alpha - Envelope alpha in `[0, NEATRACING_PULSE_GLOW_ALPHA]`.
 * @property radiusM - Expanded ring radius in meters for the pulse age.
 */
export interface PulseVisuals {
  alpha: number;
  radiusM: number;
}

/**
 * Frame payload handed to {@link NeatRacingPulsePass.draw}.
 *
 * @property pulses - Active pulses for the frame, newest first.
 * @property viewProjection - Column-major 4x4 view-projection matrix.
 */
export interface PulsePassDrawFrame {
  pulses: readonly AmbientPulse[];
  viewProjection: readonly number[];
}

/**
 * Draw-pass handle returned by {@link createPulsePass}.
 *
 * @property programKey - Registry key; matches the `'pulse'` warmup entry.
 * @property draw - Draws one frame of active pulses with glow state.
 * @property dispose - Releases the GPU program, shaders, and quad buffer.
 */
export interface NeatRacingPulsePass {
  readonly programKey: string;
  draw: (frame: PulsePassDrawFrame) => void;
  dispose: () => void;
}

/**
 * Computes the lifetime alpha envelope for a pulse.
 *
 * The envelope is zero at progress `0`, ramps linearly from `0` to the glow
 * alpha across the attack window, then decays linearly back to zero across
 * the remainder of the lifetime. It is continuous, peaks exactly at the
 * glow alpha at the attack boundary, and is fully deterministic, which lets
 * the plan's red test assert the expansion/decay curve.
 *
 * @param progress - Pulse lifetime progress; clamped into `[0, 1]`.
 * @returns Alpha in `[0, NEATRACING_PULSE_GLOW_ALPHA]`.
 * @example
 * computePulseEnvelope(0); // 0
 * computePulseEnvelope(NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION); // 0.42
 * computePulseEnvelope(1); // 0
 */
export function computePulseEnvelope(progress: number): number {
  const clamped = clamp01(progress);
  if (clamped <= NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION) {
    return (
      NEATRACING_PULSE_GLOW_ALPHA *
      (clamped / NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION)
    );
  }
  const decaySpan = 1 - NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION;
  return (
    NEATRACING_PULSE_GLOW_ALPHA *
    (1 - (clamped - NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION) / decaySpan)
  );
}

/**
 * Computes the expanding ring radius for a pulse.
 *
 * The radius grows linearly from the start radius at birth to the end
 * radius at end of lifetime, giving the deterministic expansion curve the
 * plan binds to AC-006. Progress outside `[0, 1]` clamps onto the
 * endpoints so out-of-window inputs cannot produce out-of-range geometry.
 *
 * @param progress - Pulse lifetime progress; clamped into `[0, 1]`.
 * @returns Ring radius in meters between start and end radius.
 * @example
 * computePulseRingRadius(0); // 0.5
 * computePulseRingRadius(1); // 6
 */
export function computePulseRingRadius(progress: number): number {
  const clamped = clamp01(progress);
  const span =
    NEATRACING_PULSE_RING_END_RADIUS_M - NEATRACING_PULSE_RING_START_RADIUS_M;
  return NEATRACING_PULSE_RING_START_RADIUS_M + span * clamped;
}

/**
 * Derives the deterministic world-space origin for a pulse emission.
 *
 * Two independent hash streams (X and Z) are seeded by the visual seed and
 * the spawn index, so the same seed always reproduces the same placement
 * while different seeds produce different layouts. Values map the `[0, 1)`
 * hash range onto the square spawn bounds centered on the world origin.
 *
 * @param seed - Visual seed (TP-20 base 1337 by default).
 * @param spawnIndex - Monotonic emission index to derive placement for.
 * @returns Origin in meters, bounded by the spawn half extent.
 * @example
 * derivePulseOrigin(1337, 0); // e.g. { x: -23.41, z: 12.07 }
 */
export function derivePulseOrigin(
  seed: number,
  spawnIndex: number,
): PulseOrigin {
  const streamX = hashPulseSeed(seed, spawnIndex * 2);
  const streamZ = hashPulseSeed(seed, spawnIndex * 2 + 1);
  return {
    x: (streamX * 2 - 1) * NEATRACING_PULSE_SPAWN_HALF_EXTENT_M,
    z: (streamZ * 2 - 1) * NEATRACING_PULSE_SPAWN_HALF_EXTENT_M,
  };
}

/**
 * Collects the ambient pulses alive at a simulation timestamp.
 *
 * The model is a pure time window rather than incremental state: spawn `k`
 * exists for `[k * interval, k * interval + lifetime)`. Iterating newest
 * first and capping at the concurrency limit yields the deterministic,
 * replayable alive-set the plan requires. Pulses whose age reached the
 * lifetime expire, and negative timestamps yield an empty set. Each call
 * returns a fresh array; inputs are never mutated.
 *
 * @param simTimeMs - Simulation time in milliseconds.
 * @param seed - Visual seed driving deterministic placement.
 * @returns Fresh array of active pulses ordered newest first.
 * @example
 * collectActivePulses(0, 1337); // exactly the first pulse, age 0
 * collectActivePulses(1_999, 1337); // 4 pulses at the concurrency cap
 */
export function collectActivePulses(
  simTimeMs: number,
  seed: number,
): AmbientPulse[] {
  if (simTimeMs < 0) {
    return [];
  }
  const newestSpawnIndex = Math.floor(
    simTimeMs / NEATRACING_PULSE_AMBIENT_INTERVAL_MS,
  );
  const oldestCandidateIndex = Math.max(
    0,
    newestSpawnIndex - (NEATRACING_PULSE_MAX_CONCURRENT - 1),
  );
  const pulses: AmbientPulse[] = [];
  for (
    let spawnIndex = newestSpawnIndex;
    spawnIndex >= oldestCandidateIndex &&
    pulses.length < NEATRACING_PULSE_MAX_CONCURRENT;
    spawnIndex -= 1
  ) {
    const ageMs = simTimeMs - spawnIndex * NEATRACING_PULSE_AMBIENT_INTERVAL_MS;
    if (ageMs >= NEATRACING_PULSE_LIFETIME_MS) {
      break;
    }
    const origin = derivePulseOrigin(seed, spawnIndex);
    pulses.push({
      spawnIndex,
      originX: origin.x,
      originZ: origin.z,
      ageMs,
      active: true,
    });
  }
  return pulses;
}

/**
 * Resolves the renderable visuals for one pulse from its age.
 *
 * Composition point binding the envelope and radius curves to a concrete
 * pulse: alpha comes from the lifetime envelope, radius from the expansion
 * curve, both driven by the same normalized progress.
 *
 * @param pulse - Pulse to resolve visuals for.
 * @returns Alpha and radius in meters for the pulse's current age.
 * @example
 * const pulse = collectActivePulses(1_000, 1337)[0];
 * resolvePulseVisuals(pulse); // { alpha: <decay tail>, radiusM: 3.25 }
 */
export function resolvePulseVisuals(pulse: AmbientPulse): PulseVisuals {
  const progress = pulse.ageMs / NEATRACING_PULSE_LIFETIME_MS;
  return {
    alpha: computePulseEnvelope(progress),
    radiusM: computePulseRingRadius(progress),
  };
}

/**
 * Depth-visibility predicate for a pulse against the depth buffer.
 *
 * Uses strict GL_LESS semantics matching the default WebGL2 depth test: a
 * pulse is visible only when its depth value is strictly nearer (smaller)
 * than the buffer value, so the sky clear depth of 1.0 never occludes a
 * pulse while equal or nearer geometry always does.
 *
 * @param pulseDepth - Pulse depth value in `[0, 1]` window space.
 * @param bufferDepth - Depth buffer value in `[0, 1]` window space.
 * @returns `true` when the pulse passes the depth test.
 * @example
 * isPulseDepthVisible(0.5, 1); // true (sky never occludes)
 * isPulseDepthVisible(0.5, 0.5); // false (equal depth occludes)
 */
export function isPulseDepthVisible(
  pulseDepth: number,
  bufferDepth: number,
): boolean {
  return pulseDepth < bufferDepth;
}

/**
 * Creates the WebGL2 pulse pass handle for AC-006 rendering.
 *
 * Compiles and links the ground-plane ring program once, uploads the unit
 * quad, and returns a stateless draw handle. The draw call owns the glow
 * state machine per frame: depth test on (world occlusion), depth writes
 * off, face culling off (ground-plane winding is view-dependent), and
 * premultiplied additive blending `blendFunc(ONE, ONE)` so overlapping
 * rings accumulate light without sorting. One `drawArrays` call is issued
 * per pulse in the frame; zero-alpha pulses are issued anyway and the
 * fragment shader discards them, keeping the call count deterministic.
 *
 * @param gl - WebGL2 context owned by the rendering context service.
 * @returns Pulse pass handle keyed to the `'pulse'` registry entry.
 * @example
 * const pass = createPulsePass(gl);
 * pass.draw({ pulses: collectActivePulses(simTimeMs, seed), viewProjection });
 * // later: pass.dispose();
 */
export function createPulsePass(
  gl: WebGL2RenderingContext,
): NeatRacingPulsePass {
  const vertexShader = compileShaderOrThrow(
    gl,
    gl.VERTEX_SHADER,
    PULSE_VERTEX_SHADER_SOURCE,
  );
  const fragmentShader = compileShaderOrThrow(
    gl,
    gl.FRAGMENT_SHADER,
    PULSE_FRAGMENT_SHADER_SOURCE,
  );
  const program = gl.createProgram();
  if (program === null) {
    throw new Error(
      'neatRacing pulse pass: WebGL2 refused to create a program',
    );
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error('neatRacing pulse pass: program failed to link');
  }

  const quadBuffer = gl.createBuffer();
  if (quadBuffer === null) {
    throw new Error('neatRacing pulse pass: WebGL2 refused to create a buffer');
  }
  gl.bindBuffer(GL_ARRAY_BUFFER, quadBuffer);
  gl.bufferData(
    GL_ARRAY_BUFFER,
    new Float32Array(UNIT_QUAD_CORNERS),
    GL_STATIC_DRAW,
  );

  const locations = {
    viewProjection: gl.getUniformLocation(program, 'uViewProjection'),
    originXZ: gl.getUniformLocation(program, 'uOriginXZ'),
    radiusM: gl.getUniformLocation(program, 'uRadiusM'),
    groundYM: gl.getUniformLocation(program, 'uGroundYM'),
    color: gl.getUniformLocation(program, 'uColor'),
    glow: gl.getUniformLocation(program, 'uGlow'),
  };

  const drawPulseFrame = (frame: PulsePassDrawFrame): void => {
    gl.useProgram(program);
    gl.enable(GL_DEPTH_TEST);
    gl.depthMask(false);
    gl.disable(GL_CULL_FACE);
    gl.enable(GL_BLEND);
    gl.blendFunc(GL_ONE, GL_ONE);

    gl.bindBuffer(GL_ARRAY_BUFFER, quadBuffer);
    gl.vertexAttribPointer(0, 2, GL_FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    if (locations.viewProjection !== null) {
      gl.uniformMatrix4fv(
        locations.viewProjection,
        false,
        Float32Array.from(frame.viewProjection),
      );
    }
    if (locations.color !== null) {
      gl.uniform3f(
        locations.color,
        NEATRACING_PULSE_COLOR_RGB[0],
        NEATRACING_PULSE_COLOR_RGB[1],
        NEATRACING_PULSE_COLOR_RGB[2],
      );
    }

    for (const pulse of frame.pulses) {
      const visuals = resolvePulseVisuals(pulse);
      if (locations.originXZ !== null) {
        gl.uniform2f(locations.originXZ, pulse.originX, pulse.originZ);
      }
      if (locations.groundYM !== null) {
        gl.uniform1f(locations.groundYM, NEATRACING_PULSE_GROUND_LIFT_M);
      }
      if (locations.radiusM !== null) {
        gl.uniform1f(locations.radiusM, visuals.radiusM);
      }
      if (locations.glow !== null) {
        gl.uniform1f(locations.glow, visuals.alpha);
      }
      gl.drawArrays(GL_TRIANGLE_STRIP, 0, UNIT_QUAD_VERTEX_COUNT);
    }
  };

  return {
    programKey: NEATRACING_PULSE_PROGRAM_KEY,
    draw: (frame: PulsePassDrawFrame): void => {
      drawPulseFrame(frame);
    },
    dispose: (): void => {
      gl.deleteProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      gl.deleteBuffer(quadBuffer);
    },
  };
}

/** Column-major identity matrix reused for the offscreen warmup draw. */
const IDENTITY_VIEW_PROJECTION = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
];

/** Unit quad corners as an XY triangle strip in `[-1, 1]` space. */
const UNIT_QUAD_CORNERS = [-1, -1, 1, -1, -1, 1, 1, 1];

/** Vertex count of the unit quad triangle strip. */
const UNIT_QUAD_VERTEX_COUNT = 4;

/**
 * Zero-pose warmup frame: a single pulse at birth under the identity
 * projection, drawn through the shared pass draw path during warmup.
 */
const ZERO_POSE_WARMUP_FRAME: PulsePassDrawFrame = {
  pulses: [
    {
      spawnIndex: 0,
      originX: 0,
      originZ: 0,
      ageMs: 0,
      active: true,
    },
  ],
  viewProjection: IDENTITY_VIEW_PROJECTION,
};

/**
 * Vertex shader placing the unit quad onto the XZ ground plane.
 *
 * Each corner is scaled by the pulse radius, offset by the pulse origin in
 * XZ and the ground lift in Y, then transformed by the view-projection
 * matrix. The corner is passed through as a varying for the radial band.
 */
const PULSE_VERTEX_SHADER_SOURCE = `#version 300 es
layout(location = 0) in vec2 aCorner;
uniform mat4 uViewProjection;
uniform vec2 uOriginXZ;
uniform float uRadiusM;
uniform float uGroundYM;
out vec2 vCorner;
void main() {
  vCorner = aCorner;
  vec3 world = vec3(
    uOriginXZ.x + aCorner.x * uRadiusM,
    uGroundYM,
    uOriginXZ.y + aCorner.y * uRadiusM
  );
  gl_Position = uViewProjection * vec4(world, 1.0);
}`;

/**
 * Fragment shader drawing the radial pulse ring band.
 *
 * The band peaks between the inner and peak fractions and fades out toward
 * the quad edge; glow below the epsilon is discarded so zero-alpha pulses
 * contribute no fragments. Output is premultiplied for additive blending.
 */
const PULSE_FRAGMENT_SHADER_SOURCE = `#version 300 es
precision mediump float;
in vec2 vCorner;
uniform vec3 uColor;
uniform float uGlow;
out vec4 fragColor;
const float RING_INNER_FRACTION = 0.70;
const float RING_PEAK_FRACTION = 0.86;
const float RING_OUTER_FRACTION = 1.0;
void main() {
  if (uGlow <= 0.0) {
    discard;
  }
  float dist = length(vCorner);
  float band = smoothstep(RING_INNER_FRACTION, RING_PEAK_FRACTION, dist)
    * (1.0 - smoothstep(RING_PEAK_FRACTION, RING_OUTER_FRACTION, dist));
  float glow = uGlow * band;
  if (glow <= 0.0) {
    discard;
  }
  fragColor = vec4(uColor * glow, glow);
}`;

/** WebGL2 GL_DEPTH_TEST constant. */
const GL_DEPTH_TEST = 0x0b71;

/** WebGL2 GL_CULL_FACE constant. */
const GL_CULL_FACE = 0x0b44;

/** WebGL2 GL_BLEND constant, enabled each frame before the blend factors. */
const GL_BLEND = 0x0be2;

/** WebGL2 GL_ONE blend factor. */
const GL_ONE = 1;

/** WebGL2 GL_ARRAY_BUFFER constant. */
const GL_ARRAY_BUFFER = 0x8892;

/** WebGL2 GL_STATIC_DRAW constant. */
const GL_STATIC_DRAW = 0x88e4;

/** WebGL2 GL_FLOAT constant. */
const GL_FLOAT = 0x1406;

/** WebGL2 GL_TRIANGLE_STRIP primitive. */
const GL_TRIANGLE_STRIP = 5;

/**
 * Clamps a scalar into the unit interval.
 *
 * @param value - Value to clamp.
 * @returns Value constrained to `[0, 1]`.
 */
function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/**
 * Hashes a seed and stream index into a deterministic `[0, 1)` fraction.
 *
 * Uses integer-only mixing (`Math.imul` with xor-shift finalization) so the
 * result is stable across platforms and Jest runs, giving placement its
 * replayability guarantee. Streams are independent integers derived from
 * the spawn index, so X and Z never collapse onto one stream.
 *
 * @param seed - Visual seed to salt the hash with.
 * @param stream - Independent stream index (X uses even, Z uses odd).
 * @returns Deterministic fraction in `[0, 1)`.
 */
function hashPulseSeed(seed: number, stream: number): number {
  let hash =
    Math.imul(seed | 0, 0x27d4eb2d) ^ Math.imul(stream | 0, 0x9e3779b9);
  hash = Math.imul(hash ^ (hash >>> 15), 0x85ebca6b);
  hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35);
  hash ^= hash >>> 16;
  return (hash >>> 0) / 4294967296;
}

/**
 * Compiles one shader stage, throwing with pass context on failure.
 *
 * @param gl - WebGL2 context to compile against.
 * @param type - GL shader stage constant.
 * @param source - GLSL ES 3.00 source for the stage.
 * @returns Compiled shader handle.
 */
function compileShaderOrThrow(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (shader === null) {
    throw new Error('neatRacing pulse pass: WebGL2 refused to create a shader');
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(
      `neatRacing pulse pass: shader failed to compile: ${source}`,
    );
  }
  return shader;
}

/**
 * Exposes the pulse pass as a shader warmup program (AC-040 seam).
 *
 * The returned entry keeps compilation lazy: `compileAndLink` builds the
 * pass (compiling and linking the program), and `drawOffscreen` issues the
 * zero-pose frame draw through the shared pass path so the driver warms the
 * exact render pipeline used on screen. The context passed to
 * `drawOffscreen` is accepted for interface parity; the pass is bound to
 * the context supplied at construction.
 *
 * @param context - WebGL2 context the pass resources will be built against.
 * @returns Warmup entry keyed to the `'pulse'` registry entry.
 *
 * @example
 * const warmup = toPulseWarmupProgram(gl);
 * warmup.compileAndLink();
 * warmup.drawOffscreen(gl);
 */
export function toPulseWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram {
  let pass: NeatRacingPulsePass | null = null;
  const ensurePass = (): NeatRacingPulsePass => {
    if (!pass) {
      pass = createPulsePass(context);
    }
    return pass;
  };

  return {
    key: NEATRACING_PULSE_PROGRAM_KEY,
    compileAndLink: (): void => {
      ensurePass();
    },
    drawOffscreen: (): void => {
      ensurePass().draw(ZERO_POSE_WARMUP_FRAME);
    },
  };
}
