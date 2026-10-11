/**
 * NeatRacing lighting model (slice P1-S4-D; spec sections 6.5-6.6).
 *
 * Pure TypeScript owner of the car-surface lighting model shared by every lit
 * pass: a single directional sun, hemisphere ambient that mixes ground into
 * sky over the surface normal's up component, and exp2 distance fog matching
 * the grid pass draw distance. Every function here is a numeric mirror of the
 * GLSL chunk, and `NEATRACING_LIGHTING_GLSL_CHUNK` ships the uniform names
 * plus the equivalent GLSL so `car.pass.ts` can embed the chunk itself and
 * the shader cannot drift from this mirror.
 *
 * Color discipline: all colors are authored as 0-255 bytes and normalized to
 * [0, 1] here once (byte / 255); no raw byte ever reaches a uniform feed.
 * Distances are meters; the fog density carries the 1/m unit in its name.
 */

/** RGB triple normalized to [0, 1]; also used for directions and normals. */
export type LightingVec3 = readonly [number, number, number];

/** Red byte of the sun tint (authored 0-255, normalized below). */
const NEATRACING_SUN_COLOR_R_BYTE = 255;

/** Green byte of the sun tint (authored 0-255, normalized below). */
const NEATRACING_SUN_COLOR_G_BYTE = 246;

/** Blue byte of the sun tint (authored 0-255, normalized below). */
const NEATRACING_SUN_COLOR_B_BYTE = 229;

/** Red byte of the hemisphere sky tint. */
const NEATRACING_HEMISPHERE_SKY_R_BYTE = 140;

/** Green byte of the hemisphere sky tint. */
const NEATRACING_HEMISPHERE_SKY_G_BYTE = 180;

/** Blue byte of the hemisphere sky tint. */
const NEATRACING_HEMISPHERE_SKY_B_BYTE = 255;

/** Red byte of the hemisphere ground tint. */
const NEATRACING_HEMISPHERE_GROUND_R_BYTE = 60;

/** Green byte of the hemisphere ground tint. */
const NEATRACING_HEMISPHERE_GROUND_G_BYTE = 55;

/** Blue byte of the hemisphere ground tint. */
const NEATRACING_HEMISPHERE_GROUND_B_BYTE = 50;

/** Red byte of the exp2 fog color. */
const NEATRACING_FOG_R_BYTE = 168;

/** Green byte of the exp2 fog color. */
const NEATRACING_FOG_G_BYTE = 195;

/** Blue byte of the exp2 fog color. */
const NEATRACING_FOG_B_BYTE = 225;

/**
 * Sun direction components, authored in integer space and normalized below.
 * Up-dominant (y = 2) so the directional term reads as daylight with soft
 * east/west bias; unit-length in the exported constant.
 */
const NEATRACING_SUN_DIRECTION_AUTHORED = [1, 2, 1] as const;

/** Fog extinction coefficient in 1/m (fogFactor = exp2(-(density * d)^2)). */
export const NEATRACING_FOG_DENSITY_1_PER_M = 0.006;

/** Byte divisor helper kept single-source for the color normalization rule. */
const NEATRACING_COLOR_BYTE_MAX = 255;

/** Distance in meters at which the exp2 fog transmittance is exactly 0.5. */
export const NEATRACING_FOG_HALF_TRANSMITTANCE_DISTANCE_M =
  1 / NEATRACING_FOG_DENSITY_1_PER_M;

/** Frozen, normalized sun direction (unit length, y-dominant). */
export const NEATRACING_SUN_DIRECTION: LightingVec3 = Object.freeze(
  (() => {
    const [ax, ay, az] = NEATRACING_SUN_DIRECTION_AUTHORED;
    const length = Math.hypot(ax, ay, az);
    return [ax / length, ay / length, az / length] as LightingVec3;
  })(),
);

/** Frozen normalized sun tint. */
export const NEATRACING_SUN_COLOR: LightingVec3 = Object.freeze([
  NEATRACING_SUN_COLOR_R_BYTE / NEATRACING_COLOR_BYTE_MAX,
  NEATRACING_SUN_COLOR_G_BYTE / NEATRACING_COLOR_BYTE_MAX,
  NEATRACING_SUN_COLOR_B_BYTE / NEATRACING_COLOR_BYTE_MAX,
] as LightingVec3);

/** Frozen normalized hemisphere sky tint (up-facing ambient). */
export const NEATRACING_HEMISPHERE_SKY_COLOR: LightingVec3 = Object.freeze([
  NEATRACING_HEMISPHERE_SKY_R_BYTE / NEATRACING_COLOR_BYTE_MAX,
  NEATRACING_HEMISPHERE_SKY_G_BYTE / NEATRACING_COLOR_BYTE_MAX,
  NEATRACING_HEMISPHERE_SKY_B_BYTE / NEATRACING_COLOR_BYTE_MAX,
] as LightingVec3);

/** Frozen normalized hemisphere ground tint (down-facing ambient). */
export const NEATRACING_HEMISPHERE_GROUND_COLOR: LightingVec3 = Object.freeze([
  NEATRACING_HEMISPHERE_GROUND_R_BYTE / NEATRACING_COLOR_BYTE_MAX,
  NEATRACING_HEMISPHERE_GROUND_G_BYTE / NEATRACING_COLOR_BYTE_MAX,
  NEATRACING_HEMISPHERE_GROUND_B_BYTE / NEATRACING_COLOR_BYTE_MAX,
] as LightingVec3);

/** Frozen normalized exp2 fog tint. */
export const NEATRACING_FOG_COLOR: LightingVec3 = Object.freeze([
  NEATRACING_FOG_R_BYTE / NEATRACING_COLOR_BYTE_MAX,
  NEATRACING_FOG_G_BYTE / NEATRACING_COLOR_BYTE_MAX,
  NEATRACING_FOG_B_BYTE / NEATRACING_COLOR_BYTE_MAX,
] as LightingVec3);

/**
 * Shared lighting uniform names resolved by every lit pass.
 *
 * `uCameraWorld` enables view-dependent shading inputs (spec 7.3.1 view-flip
 * uses the reconstructable view direction; the fog distance term uses the
 * world-fragment distance to the camera).
 */
export const NEATRACING_LIGHTING_UNIFORM_NAMES = [
  'uSunDirection',
  'uSunColor',
  'uHemisphereSky',
  'uHemisphereGround',
  'uFogDensity',
  'uFogColor',
  'uCameraWorld',
] as const;

/**
 * GLSL chunk for lit passes: the lighting uniforms from
 * {@link NEATRACING_LIGHTING_UNIFORM_NAMES} plus the mirrored shade
 * functions. Embedding this chunk in a fragment shader is the seam that
 * guarantees the GL and TypeScript lighting models cannot drift.
 */
export const NEATRACING_LIGHTING_GLSL_CHUNK = `uniform vec3 uSunDirection;
uniform vec3 uSunColor;
uniform vec3 uHemisphereSky;
uniform vec3 uHemisphereGround;
uniform float uFogDensity;
uniform vec3 uFogColor;
uniform vec3 uCameraWorld;

vec3 hemisphereAmbient(vec3 surfaceNormal) {
  float mixFactor = clamp(surfaceNormal.y * 0.5 + 0.5, 0.0, 1.0);
  return mix(uHemisphereGround, uHemisphereSky, mixFactor);
}

float sunDiffuse(vec3 surfaceNormal) {
  return max(0.0, dot(surfaceNormal, uSunDirection));
}

float fogFactor(float distanceMeters) {
  float opticalDepth = uFogDensity * distanceMeters;
  return exp2(-(opticalDepth * opticalDepth));
}

vec3 shadeLitSurface(vec3 albedo, vec3 surfaceNormal, float emissive) {
  vec3 lit = albedo * (hemisphereAmbient(surfaceNormal) + uSunColor * sunDiffuse(surfaceNormal));
  return lit + albedo * emissive;
}`;

/**
 * Computes the exp2 distance-fog transmittance: 1 at the camera, exactly 0.5
 * at 1/density, and ~0.53 at half the 320 m grid-pass draw distance.
 *
 * This mirrors the shader's `fogFactor(distanceMeters)` from
 * {@link NEATRACING_LIGHTING_GLSL_CHUNK}; keep both in lockstep.
 *
 * @param distanceMeters - Eye-space distance in meters (>= 0 expected).
 * @param density - Extinction coefficient in 1/m (defaults to the tuned
 *   {@link NEATRACING_FOG_DENSITY_1_PER_M}).
 * @returns Transmittance in [0, 1]: 1 = clear, 0 = fully fogged.
 *
 * @example
 * computeFogFactor(0);                       // 1
 * computeFogFactor(1 / 0.006);               // ~0.5
 * computeFogFactor(160);                     // ~0.53 at half draw distance
 */
export function computeFogFactor(
  distanceMeters: number,
  density: number = NEATRACING_FOG_DENSITY_1_PER_M,
): number {
  const opticalDepth = density * distanceMeters;
  return Math.pow(2, -(opticalDepth * opticalDepth));
}

/**
 * Computes hemisphere ambient: ground/sky mixed over the normal's up
 * component with the mirrored `clamp(normal.y * 0.5 + 0.5)` weight.
 *
 * @param normalY - World-space normal y component (unclamped range).
 * @returns Mixed ambient color, per channel, normalized [0, 1].
 *
 * @example
 * computeHemisphereAmbient(1);   // pure sky
 * computeHemisphereAmbient(-1);  // pure ground
 * computeHemisphereAmbient(0);   // ground/sky midpoint
 */
export function computeHemisphereAmbient(normalY: number): LightingVec3 {
  const mixFactor = Math.min(1, Math.max(0, normalY * 0.5 + 0.5));
  const [groundR, groundG, groundB] = NEATRACING_HEMISPHERE_GROUND_COLOR;
  const [skyR, skyG, skyB] = NEATRACING_HEMISPHERE_SKY_COLOR;
  return [
    groundR * (1 - mixFactor) + skyR * mixFactor,
    groundG * (1 - mixFactor) + skyG * mixFactor,
    groundB * (1 - mixFactor) + skyB * mixFactor,
  ];
}

/**
 * Computes the directional sun term: `max(0, dot(normal, sunDirection))`.
 *
 * @param normal - Unit surface normal in the same space as the sun direction.
 * @returns Clamped diffuse factor in [0, 1].
 *
 * @example
 * computeSunDiffuse(NEATRACING_SUN_DIRECTION); // 1
 * computeSunDiffuse([-1, 0, 0]);               // 0
 */
export function computeSunDiffuse(normal: LightingVec3): number {
  return Math.max(
    0,
    normal[0] * NEATRACING_SUN_DIRECTION[0] +
      normal[1] * NEATRACING_SUN_DIRECTION[1] +
      normal[2] * NEATRACING_SUN_DIRECTION[2],
  );
}

/**
 * Full fragment shade mirror: `albedo * (hemisphere + sunColor * sunDiffuse)
 * + albedo * emissive`. The emissive term is glow/255 resolved at vertex
 * build time, with LD-23's black-entry exception already folded in by
 * `buildVoxelMaterialTable` (black palette entries always emit 0).
 *
 * @param albedo - Vertex color normalized to [0, 1].
 * @param normal - Unit surface normal.
 * @param emissive - Additive emissive scalar in [0, 1] (glow byte / 255).
 * @returns Final lit color normalized [0, 1] (may exceed 1 before the
 *   shader's output clamp, mirroring GL behavior).
 *
 * @example
 * shadeLitSurface([0.5, 0.5, 0.5], [0, 1, 0], 0); // plain shaded white plate
 */
export function shadeLitSurface(
  albedo: LightingVec3,
  normal: LightingVec3,
  emissive: number,
): LightingVec3 {
  const hemisphere = computeHemisphereAmbient(normal[1]);
  const sunDiffuse = computeSunDiffuse(normal);
  return [
    albedo[0] * (hemisphere[0] + NEATRACING_SUN_COLOR[0] * sunDiffuse) +
      albedo[0] * emissive,
    albedo[1] * (hemisphere[1] + NEATRACING_SUN_COLOR[1] * sunDiffuse) +
      albedo[1] * emissive,
    albedo[2] * (hemisphere[2] + NEATRACING_SUN_COLOR[2] * sunDiffuse) +
      albedo[2] * emissive,
  ];
}
