/**
 * NeatRacing car render pass (slice P1-S4-D; spec sections 6.5, 6.6, 7.3.1,
 * 7.4 in the NeatRacing spec document).
 *
 * Owns the on-track car draw: body and wheel voxel-bin meshes under the
 * shared lighting model, plus the per-car blob shadow. Structural sibling of
 * `grid.pass.ts` (numeric GL enums, exported GLSL ES 3.00 sources, a compile
 * helper that throws with the pass label, throw-on-missing uniform
 * resolution, a lazy warmup seam) with two programs under one warmup key:
 *
 * 1. The lit voxel program draws bins-derived meshes two-sided (spec 7.3.1:
 *    CULL_FACE disabled, per-fragment view-flip when the world normal points
 *    away from the viewer), reconstructs the axis normal in world space
 *    through the model matrix, applies the draw-time team remap (AC-014:
 *    uTeamColor swaps albedo wherever paletteIndex equals
 *    NEATRACING_TEAM_REMAP_SLOT — never baked into vertex data), and embeds
 *    NEATRACING_LIGHTING_GLSL_CHUNK so the shader cannot drift from the
 *    lighting module's TS mirror. exp2 fog mixes to uFogColor over each
 *    fragment's world distance to the camera.
 * 2. The blob shadow program draws one ground quad per car at world
 *    y + NEATRACING_BLOB_SHADOW_Y_OFFSET_M (spec-pinned 0.01 m anti
 *    z-fight) with depth-write off, depth-test on, and SRC_ALPHA blending;
 *    a smoothstep soft edge falls off from named constants and the XZ scale
 *    covers the body's full XZ footprint with a margin.
 *
 * Draw order per frame: cars sort far -> near by squared XZ camera distance
 * inside a reused scratch array (strict-compare insertion sort keeps equal
 * distances stable), then all blob shadows draw (loop 1), then each car's
 * body plus its four wheel-mounted wheels draw (loop 2). No per-frame
 * allocation: every matrix and sort scratch is module-scope and reused (P9).
 *
 * @module
 */

import {
  NEATRACING_WARMED_PROGRAM_KEYS,
  type ShaderWarmupProgram,
} from '../context.service';
import {
  NEATRACING_FOG_COLOR,
  NEATRACING_FOG_DENSITY_1_PER_M,
  NEATRACING_HEMISPHERE_GROUND_COLOR,
  NEATRACING_HEMISPHERE_SKY_COLOR,
  NEATRACING_LIGHTING_GLSL_CHUNK,
  NEATRACING_LIGHTING_UNIFORM_NAMES,
  NEATRACING_SUN_COLOR,
  NEATRACING_SUN_DIRECTION,
  type LightingVec3,
} from '../lighting';
import {
  NEATRACING_TEAM_ID_CYAN,
  NEATRACING_TEAM_REMAP_SLOT,
  resolveVoxelTeamColor,
  type VoxelTeamColorId,
} from '../../voxel/team-remap';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../../voxel/lod.distance.utils';
import {
  resolveVoxelWheelMountTransform,
  resolveVoxelWheelSockets,
  type VoxelWheelMountTransform,
} from '../../voxel/wheel.mount.utils';
import {
  buildVoxelMaterialTable,
  type VoxelMaterial,
} from '../../voxel/materials';
import {
  VOXEL_EMPTY_PALETTE_ENTRY,
  type VoxelPaletteEntry,
} from '../../voxel/model.schema';
import type { VoxelBinModel, VoxelBinQuad } from '../../voxel/bin.codec';

/** `gl.COMPILE_STATUS` (0x8B81) checked after every shader compile. */
const GL_COMPILE_STATUS = 0x8b81;

/** `gl.LINK_STATUS` (0x8B82) checked after every program link. */
const GL_LINK_STATUS = 0x8b82;

/** `gl.VERTEX_SHADER` (0x8B31). */
const GL_VERTEX_SHADER_TYPE = 0x8b31;

/** `gl.FRAGMENT_SHADER` (0x8B30). */
const GL_FRAGMENT_SHADER_TYPE = 0x8b30;

/** `gl.ARRAY_BUFFER` (0x8892) target for every attribute upload. */
const GL_ARRAY_BUFFER = 0x8892;

/** `gl.STATIC_DRAW` (0x88E4) usage for pass-owned vertex data. */
const GL_STATIC_DRAW = 0x88e4;

/** `gl.FLOAT` (0x1406) attribute component type. */
const GL_FLOAT = 0x1406;

/** `gl.BLEND` (0x0BE2) toggled around the blob shadow phase. */
const GL_BLEND = 0x0be2;

/** `gl.SRC_ALPHA` (0x0302) shadow blend source factor. */
const GL_SRC_ALPHA = 0x0302;

/** `gl.ONE_MINUS_SRC_ALPHA` (0x0303) shadow blend destination factor. */
const GL_ONE_MINUS_SRC_ALPHA = 0x0303;

/** `gl.DEPTH_TEST` (0x0B71) enabled through both draw phases. */
const GL_DEPTH_TEST = 0x0b71;

/** `gl.CULL_FACE` (0x0B44) never enabled: bins meshes draw two-sided. */
const GL_CULL_FACE = 0x0b44;

/** `gl.TRIANGLES` (4) draw mode for every car stage. */
const GL_TRIANGLES = 4;

/** Pass label prefixing every compile/link/uniform rejection message. */
const NEATRACING_CAR_PASS_LABEL = 'neatRacing car pass';

/** Peak blob shadow alpha (soft ground-contact darkening). */
const NEATRACING_BLOB_SHADOW_MAX_ALPHA = 0.55;

/** Exponent applied to the smoothstep shadow falloff (2 = quadratic edge).
 * Interpolated into the shadow fragment shader behind a float() constructor:
 * GLSL ES forbids the const int -> const highp float initializer conversion,
 * and any TS-side decimal (2.0) is minified back to the integer `2` in the
 * shipped bundle, so the GLSL emission owns the float type. */
const NEATRACING_BLOB_SHADOW_FALLOFF_POWER = 2;

/** Dark blob shadow tint bytes; normalized to [0, 1] before any uniform feed. */
const NEATRACING_BLOB_SHADOW_COLOR_BYTES: readonly [number, number, number] = [
  8, 8, 10,
];

/** Byte divisor used when normalizing palette/tint colors into [0, 1]. */
const NEATRACING_CAR_COLOR_BYTE_MAX = 255;

/** NEATRACING_BLOB_SHADOW_COLOR_BYTES normalized into shader-space [0, 1]. */
const NEATRACING_BLOB_SHADOW_COLOR: readonly [number, number, number] = [
  NEATRACING_BLOB_SHADOW_COLOR_BYTES[0] / NEATRACING_CAR_COLOR_BYTE_MAX,
  NEATRACING_BLOB_SHADOW_COLOR_BYTES[1] / NEATRACING_CAR_COLOR_BYTE_MAX,
  NEATRACING_BLOB_SHADOW_COLOR_BYTES[2] / NEATRACING_CAR_COLOR_BYTE_MAX,
];

/**
 * World y offset for blob shadows (spec section 6.6: shadows render at
 * y + 0.01 m with depth-write off so they read as ground decals without
 * z-fighting the road grid).
 */
export const NEATRACING_BLOB_SHADOW_Y_OFFSET_M = 0.01;

/**
 * Blob shadow XZ scale margin over the body's FULL XZ footprint. The shadow
 * quad world width equals 2 x boundsHalfExtentXzM (the mesh bounds field is
 * a half extent) times this scale, so 1.15 covers the chassis footprint
 * plus a 15% margin.
 */
export const NEATRACING_BLOB_SHADOW_BOUNDS_SCALE = 1.15;

/** Floats per expanded car vertex: position, color, axis, emissive, slot. */
export const CAR_MESH_FLOATS_PER_VERTEX = 9;

/** Bytes per expanded car vertex (9 floats x 4 bytes). */
export const CAR_MESH_VERTEX_STRIDE_BYTES = CAR_MESH_FLOATS_PER_VERTEX * 4;

/** Position component byte offset inside an expanded car vertex. */
const CAR_MESH_POSITION_OFFSET_BYTES = 0;

/** Color component byte offset inside an expanded car vertex. */
const CAR_MESH_COLOR_OFFSET_BYTES = 12;

/** Normal-axis component byte offset inside an expanded car vertex. */
const CAR_MESH_NORMAL_AXIS_OFFSET_BYTES = 24;

/** Emissive component byte offset inside an expanded car vertex. */
const CAR_MESH_EMISSIVE_OFFSET_BYTES = 28;

/** Palette-index component byte offset inside an expanded car vertex. */
const CAR_MESH_PALETTE_INDEX_OFFSET_BYTES = 32;

/** Expanded vertices per source quad: two non-indexed triangles. */
const NEATRACING_CAR_VERTICES_PER_QUAD = 6;

/** Corner index order per quad: triangles (c0, c1, c2) and (c0, c2, c3). */
const NEATRACING_CAR_TRIANGLE_CORNER_INDICES = [0, 1, 2, 0, 2, 3] as const;

/** Quad corner (u, v) scale table in expansion order: 00, 10, 11, 01. */
const NEATRACING_CAR_QUAD_CORNER_SCALES: readonly (readonly [
  number,
  number,
])[] = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
];

/** Floats per blob shadow vertex (unit quad corner in [0, 1] x [0, 1]). */
const CAR_SHADOW_FLOATS_PER_VERTEX = 2;

/** Bytes per blob shadow vertex. */
const CAR_SHADOW_VERTEX_STRIDE_BYTES = CAR_SHADOW_FLOATS_PER_VERTEX * 4;

/** Blob shadow corner data: two triangles over the unit square. */
const CAR_SHADOW_QUAD_CORNERS = new Float32Array([
  0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1,
]);

/** Vertex count of one blob shadow quad (two triangles). */
const CAR_SHADOW_VERTEX_COUNT =
  CAR_SHADOW_QUAD_CORNERS.length / CAR_SHADOW_FLOATS_PER_VERTEX;

/** Lit-program uniform names: shared lighting uniforms + car-specific feeds. */
const CAR_LIT_UNIFORM_NAMES: readonly string[] = [
  ...NEATRACING_LIGHTING_UNIFORM_NAMES,
  'uViewProjection',
  'uModel',
  'uTeamColor',
  'uTeamSlot',
];

/** Blob-shadow program uniform names. */
const CAR_SHADOW_UNIFORM_NAMES: readonly string[] = [
  'uViewProjection',
  'uModel',
  'uShadowColor',
] as const;

/**
 * Lit vertex shader: transforms a car voxel vertex into world space and
 * passes color, axis, emissive, palette slot, and camera distance through.
 */
export const CAR_LIT_VERTEX_SHADER_SOURCE = `#version 300 es
in vec3 aPosition;
in vec3 aColor;
in float aNormalAxis;
in float aEmissive;
in float aPaletteIndex;
uniform mat4 uViewProjection;
uniform mat4 uModel;
uniform vec3 uCameraWorld;
out vec3 vWorld;
out vec3 vAlbedo;
out float vNormalAxis;
out float vEmissive;
out float vPaletteIndex;
out float vFogDistance;
void main() {
  vec4 world = uModel * vec4(aPosition, 1.0);
  vWorld = world.xyz;
  vAlbedo = aColor;
  vNormalAxis = aNormalAxis;
  vEmissive = aEmissive;
  vPaletteIndex = aPaletteIndex;
  vFogDistance = length(world.xyz - uCameraWorld);
  gl_Position = uViewProjection * world;
}`;

/**
 * Lit fragment shader: two-sided axis normal reconstructed through the model
 * matrix, per-fragment view flip (spec 7.3.1), draw-time team remap
 * (AC-014), and the shared lighting chunk over exp2 fog.
 */
export const CAR_LIT_FRAGMENT_SHADER_SOURCE = `#version 300 es
precision highp float;
uniform mat4 uModel;
uniform vec3 uTeamColor;
uniform float uTeamSlot;
${NEATRACING_LIGHTING_GLSL_CHUNK}
in vec3 vWorld;
in vec3 vAlbedo;
in float vNormalAxis;
in float vEmissive;
in float vPaletteIndex;
in float vFogDistance;
out vec4 fragColor;
void main() {
  vec3 axisNormal = vec3(1.0, 0.0, 0.0);
  if (vNormalAxis >= 0.5 && vNormalAxis < 1.5) {
    axisNormal = vec3(0.0, 1.0, 0.0);
  } else if (vNormalAxis >= 1.5) {
    axisNormal = vec3(0.0, 0.0, 1.0);
  }
  vec3 worldNormal = normalize(mat3(uModel) * axisNormal);
  vec3 viewDirection = normalize(vWorld - uCameraWorld);
  if (dot(worldNormal, viewDirection) > 0.0) {
    worldNormal = -worldNormal;
  }
  vec3 albedo = vAlbedo;
  if (abs(vPaletteIndex - uTeamSlot) < 0.5) {
    albedo = uTeamColor;
  }
  vec3 lit = shadeLitSurface(albedo, worldNormal, vEmissive);
  fragColor = vec4(mix(uFogColor, lit, fogFactor(vFogDistance)), 1.0);
}`;

/** Blob shadow vertex shader: unit quad centered on the car's XZ footprint. */
export const CAR_SHADOW_VERTEX_SHADER_SOURCE = `#version 300 es
in vec2 aCorner;
uniform mat4 uViewProjection;
uniform mat4 uModel;
out vec2 vCorner;
void main() {
  vCorner = aCorner;
  vec2 centered = aCorner - vec2(0.5, 0.5);
  gl_Position = uViewProjection * uModel * vec4(centered.x, 0.0, centered.y, 1.0);
}`;

/**
 * Blob shadow fragment shader: smoothstep soft edge over the dark tint,
 * peaking at NEATRACING_BLOB_SHADOW_MAX_ALPHA and shaped by
 * NEATRACING_BLOB_SHADOW_FALLOFF_POWER (both interpolated from named
 * constants so the GLSL cannot drift from the module contract).
 */
export const CAR_SHADOW_FRAGMENT_SHADER_SOURCE = `#version 300 es
precision highp float;
uniform vec3 uShadowColor;
in vec2 vCorner;
out vec4 fragColor;
const float kShadowCenter = 0.5;
const float kShadowRadiusScale = 2.0;
const float kShadowMaxAlpha = ${NEATRACING_BLOB_SHADOW_MAX_ALPHA};
const float kShadowFalloffPower = float(${NEATRACING_BLOB_SHADOW_FALLOFF_POWER});
void main() {
  float radial = length(vCorner - vec2(kShadowCenter, kShadowCenter)) * kShadowRadiusScale;
  float falloff = 1.0 - smoothstep(0.0, 1.0, radial);
  fragColor = vec4(uShadowColor, kShadowMaxAlpha * pow(falloff, kShadowFalloffPower));
}`;

/**
 * Fallback material for out-of-range palette indices: renders like LD-23
 * black (zero emissive) so a malformed index can never fabricate glow.
 */
const NEATRACING_CAR_MISSING_MATERIAL: VoxelMaterial = {
  kind: 'black',
  color: [0, 0, 0],
  alpha: 0,
  emissive: 0,
};

/**
 * Mesh produced by {@link expandVoxelBinModelToCarMesh}: interleaved
 * non-indexed triangles ready for one FLOAT-attributed VBO per model.
 */
export interface CarVoxelMesh {
  /** Interleaved vertex data on the CAR_MESH_VERTEX_STRIDE_BYTES stride. */
  readonly vertexData: Float32Array;
  /** Total expanded vertex count (6 per source quad). */
  readonly vertexCount: number;
  /** XZ half-extent of the expanded vertices in meters (shadow sizing). */
  readonly boundsHalfExtentXzM: number;
}

/** Car pose inside one draw frame. */
export interface CarPassCarState {
  /** Car world X in meters. */
  readonly worldX: number;
  /** Car world Z in meters. */
  readonly worldZ: number;
  /** Car yaw around world up in radians. */
  readonly yawRad: number;
  /** Team color id resolved through the draw-time AC-014 table. */
  readonly teamColorId: number;
  /** Wheel spin in radians shared by all four wheels. */
  readonly wheelSpinRad: number;
}

/** Per-frame draw state accepted by CarPass.draw. */
export interface CarPassDrawState {
  /** Column-major view-projection (16 floats). */
  readonly viewProjection: Float32Array;
  /** Camera world position [x, y, z] in meters. */
  readonly cameraWorld: readonly [number, number, number];
  /** On-track cars in any order; the pass sorts far -> near internally. */
  readonly cars: readonly CarPassCarState[];
}

/** Car pass handle returned by createCarPass. */
export interface CarPass {
  /**
   * Draws all blob shadows first (depth-write off, blended), then every car
   * body plus its four wheels (depth-write on, two-sided).
   *
   * @param state - Frame draw state (see {@link CarPassDrawState}).
   * @throws When viewProjection is not 16 elements or cameraWorld is not 3.
   */
  draw(state: CarPassDrawState): void;
  /** Releases both programs, all four shaders, and all three buffers. */
  dispose(): void;
}

/** Models the pass renders: the car body bin plus one wheel bin. */
export interface CarPassModels {
  /** Car body voxel bin (bins-derived mesh, drawn two-sided). */
  readonly body: VoxelBinModel;
  /** Wheel voxel bin (drawn four times via the wheel mount contract). */
  readonly wheel: VoxelBinModel;
}

/**
 * Expands a voxel bin into the car pass vertex format (pure, GL-free).
 *
 * Each quad becomes six non-indexed vertices (triangles c0-c1-c2 and
 * c0-c2-c3 over corners (0,0), (w,0), (w,h), (0,h)) on a 9-float stride
 * [x, y, z, r, g, b, normalAxis, emissive, paletteIndex]. X/Z grid positions
 * map to model-space meters centered on the model origin (lattice-centered);
 * the Y mapping is pinned by the `yAnchor` argument so the two authored
 * model kinds land where the G9 mount contract expects them:
 * - `'ground'` (car bodies, authored with row 0 down on the floor):
 *   `modelY = row * voxelSize`, so the body's floor row sits exactly on the
 *   car-frame ground plane `y = 0` and the body rises above the road.
 *   Centering the lattice instead would sink a bottom-authored body meters
 *   underground where every fragment fails the depth test against the road
 *   (invisible car bug).
 * - `'center'` (wheels, authored as a symmetric 32³ disc whose surface
 *   center is the lattice midpoint `dimY * 0.5` = 16 cells = the hub
 *   height): `modelY = (row - dimY * 0.5) * voxelSize`, so the disc's
 *   geometric center is the rotation pivot AND the tire's bottom face sits
 *   at exactly `-NEATRACING_WHEEL_HUB_HEIGHT_M`; the mount then raises the
 *   disc by the hub height, landing the contact patch at exactly `y = 0`
 *   (AC-015). A `(dimY - 1) * 0.5` offset would float the tire half a cell
 *   (17.6 mm) above the road.
 *
 * Colors resolve from `buildVoxelMaterialTable` (bytes / 255) and emissive is
 * glow / 255 with the LD-23 black exception already folded in (black palette
 * entries carry emissive 0). The NEATRACING_TEAM_REMAP_SLOT color stays
 * UNBAKED because the team remap happens at draw time (AC-014).
 *
 * Tangent rule (matching the shared mesher `FACE_TANGENT_AXES` exactly —
 * the mesher records `position[widthAxis] = u` and `position[heightAxis] =
 * v`, so a mismatch here transposes every non-square face and stretches it
 * to the swapped extents): width runs along the first tangent axis, height
 * along the second: axis 0 -> (y, z), axis 1 -> (x, z), axis 2 -> (x, y);
 * the fixed axis is `quad.position[axis]`.
 *
 * @param model - Decoded voxel bin model (dims, palette, quads).
 * @param yAnchor - Vertical anchoring contract: `'ground'` for bodies
 *   authored with row 0 on the floor, `'center'` for lattice-centered wheel
 *   discs. Defaults to `'ground'`.
 * @returns The expanded mesh plus its XZ half-extent in meters.
 *
 * @example
 * const mesh = expandVoxelBinModelToCarMesh(bodyBin);
 * // mesh.vertexCount === 6 * bodyBin.quads.length; 9 floats per vertex
 * const wheelMesh = expandVoxelBinModelToCarMesh(wheelBin, 'center');
 */
export type CarMeshYAnchor = 'ground' | 'center';

export function expandVoxelBinModelToCarMesh(
  model: VoxelBinModel,
  yAnchor: CarMeshYAnchor = 'ground',
): CarVoxelMesh {
  const materials = buildVoxelMaterialTable(model.palette);
  const voxelSize = NEATRACING_VOXEL_WORLD_SIZE;
  const [dimX, dimY, dimZ] = model.dims;
  const latticeCenterRows = dimY * 0.5;
  const vertexCount = model.quads.length * NEATRACING_CAR_VERTICES_PER_QUAD;
  const vertexData = new Float32Array(vertexCount * CAR_MESH_FLOATS_PER_VERTEX);
  let boundsHalfExtentXzM = 0;
  let writeOffset = 0;
  for (const quad of model.quads) {
    const material =
      materials[quad.paletteIndex] ?? NEATRACING_CAR_MISSING_MATERIAL;
    const red = material.color[0] / NEATRACING_CAR_COLOR_BYTE_MAX;
    const green = material.color[1] / NEATRACING_CAR_COLOR_BYTE_MAX;
    const blue = material.color[2] / NEATRACING_CAR_COLOR_BYTE_MAX;
    const emissive = material.emissive / NEATRACING_CAR_COLOR_BYTE_MAX;
    for (const cornerIndex of NEATRACING_CAR_TRIANGLE_CORNER_INDICES) {
      const [uScale, vScale] = NEATRACING_CAR_QUAD_CORNER_SCALES[cornerIndex];
      const u = uScale * quad.width;
      const v = vScale * quad.height;
      const gridX = quadCornerGridX(quad, u);
      const gridY = quadCornerGridY(quad, u, v);
      const gridZ = quadCornerGridZ(quad, v);
      const modelX = (gridX - dimX * 0.5) * voxelSize;
      const modelY =
        yAnchor === 'center'
          ? (gridY - latticeCenterRows) * voxelSize
          : gridY * voxelSize;
      const modelZ = (gridZ - dimZ * 0.5) * voxelSize;
      vertexData[writeOffset] = modelX;
      vertexData[writeOffset + 1] = modelY;
      vertexData[writeOffset + 2] = modelZ;
      vertexData[writeOffset + 3] = red;
      vertexData[writeOffset + 4] = green;
      vertexData[writeOffset + 5] = blue;
      vertexData[writeOffset + 6] = quad.normalAxis;
      vertexData[writeOffset + 7] = emissive;
      vertexData[writeOffset + 8] = quad.paletteIndex;
      writeOffset += CAR_MESH_FLOATS_PER_VERTEX;
      const xzExtent = Math.max(Math.abs(modelX), Math.abs(modelZ));
      if (xzExtent > boundsHalfExtentXzM) {
        boundsHalfExtentXzM = xzExtent;
      }
    }
  }
  return { vertexData, vertexCount, boundsHalfExtentXzM };
}

/**
 * Resolves a corner's grid x coordinate per the shared mesher tangent rule.
 *
 * @param quad - Source quad (minimum lattice corner + tangent extents).
 * @param u - Width extent along the quad's first tangent axis.
 * @returns Grid x of the expanded corner.
 */
function quadCornerGridX(quad: VoxelBinQuad, u: number): number {
  if (quad.normalAxis === 0) {
    return quad.position[0];
  }
  // Axes 1 and 2 both take x as their width axis: width extends by u.
  return quad.position[0] + u;
}

/**
 * Resolves a corner's grid y coordinate per the ascending tangent rule.
 *
 * @param quad - Source quad (minimum lattice corner + tangent extents).
 * @param u - Width extent along the quad's first tangent axis.
 * @param v - Height extent along the quad's second tangent axis.
 * @returns Grid y of the expanded corner.
 */
function quadCornerGridY(quad: VoxelBinQuad, u: number, v: number): number {
  if (quad.normalAxis === 0) {
    return quad.position[1] + u;
  }
  if (quad.normalAxis === 1) {
    return quad.position[1];
  }
  return quad.position[1] + v;
}

/**
 * Resolves a corner's grid z coordinate per the shared mesher tangent rule.
 *
 * @param quad - Source quad (minimum lattice corner + tangent extents).
 * @param v - Height extent along the quad's second tangent axis.
 * @returns Grid z of the expanded corner.
 */
function quadCornerGridZ(quad: VoxelBinQuad, v: number): number {
  if (quad.normalAxis === 2) {
    return quad.position[2];
  }
  // Axes 0 and 1 both take z as their height axis: height extends by v.
  return quad.position[2] + v;
}

/** Identity column-major matrix values reused by every helper below. */
const NEATRACING_CAR_MAT4_IDENTITY: readonly number[] = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
];

/**
 * Writes the identity matrix into a scratch buffer.
 *
 * @param out - Column-major 16-element destination.
 * @returns The same buffer for call-site chaining.
 */
function identityMat4(out: Float32Array): Float32Array {
  for (let index = 0; index < 16; index += 1) {
    out[index] = NEATRACING_CAR_MAT4_IDENTITY[index];
  }
  return out;
}

/**
 * Writes a translation matrix into a scratch buffer (column-major).
 *
 * @param out - Column-major 16-element scratch destination.
 * @param x - Translation x in meters.
 * @param y - Translation y in meters.
 * @param z - Translation z in meters.
 * @returns The same buffer for call-site chaining.
 */
function translationMat4(
  out: Float32Array,
  x: number,
  y: number,
  z: number,
): Float32Array {
  identityMat4(out);
  out[12] = x;
  out[13] = y;
  out[14] = z;
  return out;
}

/**
 * Writes an axis scale matrix into a scratch buffer (column-major).
 *
 * @param out - Column-major 16-element scratch destination.
 * @param sx - X scale factor.
 * @param sy - Y scale factor.
 * @param sz - Z scale factor.
 * @returns The same buffer for call-site chaining.
 */
function scaleXYZMat4(
  out: Float32Array,
  sx: number,
  sy: number,
  sz: number,
): Float32Array {
  identityMat4(out);
  out[0] = sx;
  out[5] = sy;
  out[10] = sz;
  return out;
}

/**
 * Writes a rotation-around-world-up matrix into a scratch buffer
 * (column-major: col0 = (cos, 0, -sin), col2 = (sin, 0, cos)).
 *
 * @param out - Column-major 16-element scratch destination.
 * @param angleRad - Yaw angle in radians.
 * @returns The same buffer for call-site chaining.
 */
function rotationYMat4(out: Float32Array, angleRad: number): Float32Array {
  identityMat4(out);
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  out[0] = cos;
  out[2] = -sin;
  out[8] = sin;
  out[10] = cos;
  return out;
}

/**
 * Writes a rotation-around-the-model-depth-axis matrix into a scratch buffer
 * (the pre-mount wheel model's spin axis; column-major).
 *
 * @param out - Column-major 16-element scratch destination.
 * @param angleRad - Spin angle in radians.
 * @returns The same buffer for call-site chaining.
 */
function rotationZMat4(out: Float32Array, angleRad: number): Float32Array {
  identityMat4(out);
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  out[0] = cos;
  out[1] = sin;
  out[4] = -sin;
  out[5] = cos;
  return out;
}

/**
 * Multiplies two column-major matrices into an output scratch buffer:
 * `out = a * b` (a point is transformed by b first, then a). Call sites must
 * pass a dedicated scratch that aliases neither input; every scratch here is
 * preallocated at module scope to keep the draw path allocation-free (P9).
 *
 * @param out - 16-element output scratch (must not alias a or b).
 * @param a - Left operand, applied second in the transform chain.
 * @param b - Right operand, applied first in the transform chain.
 * @returns The same output buffer for call-site chaining.
 */
function multiplyMat4(
  out: Float32Array,
  a: Float32Array,
  b: Float32Array,
): Float32Array {
  for (let column = 0; column < 4; column += 1) {
    const b0 = b[column * 4 + 0];
    const b1 = b[column * 4 + 1];
    const b2 = b[column * 4 + 2];
    const b3 = b[column * 4 + 3];
    for (let row = 0; row < 4; row += 1) {
      out[column * 4 + row] =
        a[row] * b0 + a[4 + row] * b1 + a[8 + row] * b2 + a[12 + row] * b3;
    }
  }
  return out;
}

/**
 * Feeds a normalized [0, 1] color triple into a vec3 uniform; no raw 0-255
 * byte ever reaches a uniform through this helper.
 *
 * @param gl - WebGL2 context.
 * @param location - Resolved uniform location.
 * @param color - Color already normalized to [0, 1].
 */
function uniform3fColor(
  gl: WebGL2RenderingContext,
  location: WebGLUniformLocation,
  color: LightingVec3,
): void {
  gl.uniform3f(location, color[0], color[1], color[2]);
}

/**
 * Compiles one car pass shader stage, rejecting failures with the pass label
 * (grid.pass.ts parity).
 *
 * @param gl - WebGL2 context.
 * @param shaderType - gl.VERTEX_SHADER (0x8B31) or gl.FRAGMENT_SHADER (0x8B30).
 * @param shaderSource - GLSL ES 3.00 source.
 * @param shaderLabel - Stage label used in the rejection message.
 * @returns The compiled shader handle.
 * @throws When compilation fails; the message carries the pass label.
 */
function compileCarShader(
  gl: WebGL2RenderingContext,
  shaderType: number,
  shaderSource: string,
  shaderLabel: string,
): WebGLShader {
  const shader = gl.createShader(shaderType);
  if (shader === null) {
    throw new Error(
      `${NEATRACING_CAR_PASS_LABEL}: failed to create the ${shaderLabel} shader`,
    );
  }
  gl.shaderSource(shader, shaderSource);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, GL_COMPILE_STATUS)) {
    const infoLog = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(
      `${NEATRACING_CAR_PASS_LABEL}: failed to compile the ${shaderLabel} shader: ${String(infoLog)}`,
    );
  }
  return shader;
}

/** Compiled program bundle: the linked program plus its two live shaders. */
interface CompiledCarProgram {
  /** Linked program handle. */
  readonly program: WebGLProgram;
  /** Live vertex shader handle (released by dispose). */
  readonly vertexShader: WebGLShader;
  /** Live fragment shader handle (released by dispose). */
  readonly fragmentShader: WebGLShader;
}

/**
 * Compiles and links one of the car pass programs from its two stages.
 *
 * @param gl - WebGL2 context.
 * @param vertexShaderSource - Source for the vertex stage.
 * @param fragmentShaderSource - Source for the fragment stage.
 * @param programLabel - Label used in every rejection message.
 * @returns The linked program plus its two shader handles.
 * @throws When compilation or linking fails.
 */
function compileAndLinkCarProgram(
  gl: WebGL2RenderingContext,
  vertexShaderSource: string,
  fragmentShaderSource: string,
  programLabel: string,
): CompiledCarProgram {
  const vertexShader = compileCarShader(
    gl,
    GL_VERTEX_SHADER_TYPE,
    vertexShaderSource,
    `${programLabel} vertex`,
  );
  const fragmentShader = compileCarShader(
    gl,
    GL_FRAGMENT_SHADER_TYPE,
    fragmentShaderSource,
    `${programLabel} fragment`,
  );
  const program = gl.createProgram();
  if (program === null) {
    throw new Error(
      `${NEATRACING_CAR_PASS_LABEL}: failed to create the ${programLabel} program`,
    );
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, GL_LINK_STATUS)) {
    const infoLog = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(
      `${NEATRACING_CAR_PASS_LABEL}: failed to link the ${programLabel} program: ${String(infoLog)}`,
    );
  }
  return { program, vertexShader, fragmentShader };
}

/**
 * Resolves one program's uniform locations with throw-on-missing semantics
 * so a shader/name drift fails loudly at pass creation (grid.pass.ts parity).
 *
 * @param gl - WebGL2 context.
 * @param program - Linked program handle.
 * @param names - Uniform names to resolve, in feed order.
 * @returns Location table keyed by uniform name.
 * @throws When any uniform is missing from the linked program.
 */
function resolveCarUniformLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  names: readonly string[],
): Record<string, WebGLUniformLocation> {
  const locations: Record<string, WebGLUniformLocation> = {};
  for (const name of names) {
    const location = gl.getUniformLocation(program, name);
    if (location === null) {
      throw new Error(`${NEATRACING_CAR_PASS_LABEL}: missing uniform ${name}`);
    }
    locations[name] = location;
  }
  return locations;
}

/** Lit-program attribute names resolved in expansion float order. */
const CAR_LIT_ATTRIBUTE_NAMES = [
  'aPosition',
  'aColor',
  'aNormalAxis',
  'aEmissive',
  'aPaletteIndex',
] as const;

/**
 * Resolves the five lit-program attribute locations in expansion float order
 * (position, color, normal axis, emissive, palette index).
 *
 * @param gl - WebGL2 context.
 * @param program - Linked lit program handle.
 * @returns Attribute location table used by vertexAttribPointer feeds.
 */
function resolveCarLitAttributeLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
): Record<(typeof CAR_LIT_ATTRIBUTE_NAMES)[number], number> {
  const locations: Record<string, number> = {};
  for (const name of CAR_LIT_ATTRIBUTE_NAMES) {
    const attributeId = gl.getAttribLocation(program, name);
    if (attributeId < 0) {
      throw new Error(
        `${NEATRACING_CAR_PASS_LABEL}: missing attribute ${name}`,
      );
    }
    locations[name] = attributeId;
  }
  return locations as Record<(typeof CAR_LIT_ATTRIBUTE_NAMES)[number], number>;
}

/**
 * Uploads one Float32 array into a STATIC_DRAW buffer.
 *
 * @param gl - WebGL2 context.
 * @param data - Vertex data to upload.
 * @returns The uploaded buffer handle.
 * @throws When gl.createBuffer returns null.
 */
function uploadCarFloat32Buffer(
  gl: WebGL2RenderingContext,
  data: Float32Array,
): WebGLBuffer {
  const buffer = gl.createBuffer();
  if (buffer === null) {
    throw new Error(
      `${NEATRACING_CAR_PASS_LABEL}: failed to create a car vertex buffer`,
    );
  }
  gl.bindBuffer(GL_ARRAY_BUFFER, buffer);
  gl.bufferData(GL_ARRAY_BUFFER, data, GL_STATIC_DRAW);
  return buffer;
}

/** Car model matrix scratch (T x RY per car body). */
const NEATRACING_CAR_TRANSLATION_MATRIX = new Float32Array(16);

/** Car yaw scratch fed by state.yawRad. */
const NEATRACING_CAR_YAW_MATRIX = new Float32Array(16);

/** Composed car model matrix fed to uModel for body and wheel chains. */
const NEATRACING_CAR_MODEL_MATRIX = new Float32Array(16);

/** First matrix-multiply scratch (inputs never alias outputs). */
const NEATRACING_CAR_MATRIX_SCRATCH_A = new Float32Array(16);

/** Second matrix-multiply scratch (kept separate so inputs never alias). */
const NEATRACING_CAR_MATRIX_SCRATCH_B = new Float32Array(16);

/** Wheel hub translation scratch. */
const NEATRACING_WHEEL_TRANSLATION_MATRIX = new Float32Array(16);

/** Wheel mount yaw scratch. */
const NEATRACING_WHEEL_MOUNT_YAW_MATRIX = new Float32Array(16);

/** Wheel spin scratch (about the pre-mount model depth axis). */
const NEATRACING_WHEEL_SPIN_MATRIX = new Float32Array(16);

/** Blob shadow translation scratch (x, 0.01, z). */
const NEATRACING_SHADOW_TRANSLATION_MATRIX = new Float32Array(16);

/** Blob shadow yaw scratch. */
const NEATRACING_SHADOW_YAW_MATRIX = new Float32Array(16);

/** Blob shadow scale scratch (s, 1, s). */
const NEATRACING_SHADOW_SCALE_MATRIX = new Float32Array(16);

/** Composed blob shadow matrix fed to uModel per car. */
const NEATRACING_SHADOW_MATRIX = new Float32Array(16);

/** Reused insertion-sort index scratch (grown once per fleet size, no churn). */
const NEATRACING_CAR_SORT_ORDER_SCRATCH: number[] = [];

/** Reused squared-distance scratch for the far -> near sort (P9). */
let NEATRACING_CAR_SORT_DISTANCE_SCRATCH = new Float64Array(0);

/**
 * Grows the sort scratches to the fleet size once; steady-state frames never
 * reallocate because scratch capacity only ever grows (P9).
 *
 * @param carCount - Cars in the current frame.
 */
function ensureCarSortScratchCapacity(carCount: number): void {
  if (NEATRACING_CAR_SORT_DISTANCE_SCRATCH.length < carCount) {
    NEATRACING_CAR_SORT_DISTANCE_SCRATCH = new Float64Array(carCount);
  }
  if (NEATRACING_CAR_SORT_ORDER_SCRATCH.length < carCount) {
    NEATRACING_CAR_SORT_ORDER_SCRATCH.length = carCount;
  }
}

/**
 * Fills the sort scratch with the far -> near draw order. Stable insertion
 * sort over squared XZ camera distances: equal distances keep array order,
 * so the ordering is deterministic frame to frame.
 *
 * @param state - Frame draw state (cars + camera world position).
 */
function composeSortedCarOrder(state: CarPassDrawState): void {
  const carCount = state.cars.length;
  const cameraX = state.cameraWorld[0];
  const cameraZ = state.cameraWorld[2];
  ensureCarSortScratchCapacity(carCount);
  for (let index = 0; index < carCount; index += 1) {
    const car = state.cars[index];
    const deltaX = car.worldX - cameraX;
    const deltaZ = car.worldZ - cameraZ;
    NEATRACING_CAR_SORT_ORDER_SCRATCH[index] = index;
    NEATRACING_CAR_SORT_DISTANCE_SCRATCH[index] =
      deltaX * deltaX + deltaZ * deltaZ;
  }
  for (let slot = 1; slot < carCount; slot += 1) {
    const movingIndex = NEATRACING_CAR_SORT_ORDER_SCRATCH[slot];
    const movingDistance = NEATRACING_CAR_SORT_DISTANCE_SCRATCH[movingIndex];
    let target = slot - 1;
    while (
      target >= 0 &&
      NEATRACING_CAR_SORT_DISTANCE_SCRATCH[
        NEATRACING_CAR_SORT_ORDER_SCRATCH[target]
      ] < movingDistance
    ) {
      NEATRACING_CAR_SORT_ORDER_SCRATCH[target + 1] =
        NEATRACING_CAR_SORT_ORDER_SCRATCH[target];
      target -= 1;
    }
    NEATRACING_CAR_SORT_ORDER_SCRATCH[target + 1] = movingIndex;
  }
}

/**
 * Warmup-only bins: a 2-cubed synthetic body and wheel so the warmup path
 * exercises the real compile + draw pipeline without loaded track assets.
 */
const NEATRACING_WARMUP_MODEL_DIMS = [2, 2, 2] as const;

/** Warmup palette: the EMPTY sentinel plus one neutral solid entry. */
const NEATRACING_WARMUP_PALETTE: readonly VoxelPaletteEntry[] = [
  [...VOXEL_EMPTY_PALETTE_ENTRY] as VoxelPaletteEntry,
  [180, 40, 40, 255, 0] as VoxelPaletteEntry,
];

/** Warmup body bin (axis 0 face across the 2-cubed grid). */
const NEATRACING_WARMUP_BODY_BIN: VoxelBinModel = {
  dims: NEATRACING_WARMUP_MODEL_DIMS,
  palette: NEATRACING_WARMUP_PALETTE,
  quads: [
    {
      position: [0, 0, 0],
      normalAxis: 0,
      width: 2,
      height: 2,
      paletteIndex: 1,
    },
  ],
};

/** Warmup wheel bin (axis 1 face across the 2-cubed grid). */
const NEATRACING_WARMUP_WHEEL_BIN: VoxelBinModel = {
  dims: NEATRACING_WARMUP_MODEL_DIMS,
  palette: NEATRACING_WARMUP_PALETTE,
  quads: [
    {
      position: [0, 0, 0],
      normalAxis: 1,
      width: 2,
      height: 2,
      paletteIndex: 1,
    },
  ],
};

/** Warmup models consumed by the lazy car pass instantiation. */
const NEATRACING_WARMUP_MODELS: CarPassModels = {
  body: NEATRACING_WARMUP_BODY_BIN,
  wheel: NEATRACING_WARMUP_WHEEL_BIN,
};

/**
 * Zero-pose warmup frame: identity projection, camera at the origin, one car
 * parked at the origin on team cyan with no spin. drawOffscreen pushes this
 * through the shared draw path so warmup proves the full pipeline works.
 */
const NEATRACING_ZERO_POSE_DRAW_STATE: CarPassDrawState = {
  viewProjection: new Float32Array(NEATRACING_CAR_MAT4_IDENTITY),
  cameraWorld: [0, 0, 0],
  cars: [
    {
      worldX: 0,
      worldZ: 0,
      yawRad: 0,
      teamColorId: NEATRACING_TEAM_ID_CYAN,
      wheelSpinRad: 0,
    },
  ],
};

/**
 * Warmup program key: registry index 4 of NEATRACING_WARMED_PROGRAM_KEYS
 * ('car') is the contract pinned by the warmup seam test (AC-040).
 */
const NEATRACING_CAR_PROGRAM_KEY = NEATRACING_WARMED_PROGRAM_KEYS[4];

/** Lazily hydrated singleton pass behind the warmup seam (grid parity). */
let warmupCarPass: CarPass | null = null;

/** Context the current warmup singleton was built against. */
let warmupCarPassContext: WebGL2RenderingContext | null = null;

/**
 * Hydrates the singleton car pass for the warmup seam, rebuilding only when
 * the context changes. Compilation stays lazy until compileAndLink runs
 * (grid.pass.ts parity).
 *
 * @param context - WebGL2 context the warmup shader will run against.
 * @returns The hydrated singleton pass.
 */
function ensureWarmupCarPass(context: WebGL2RenderingContext): CarPass {
  if (warmupCarPass !== null && warmupCarPassContext === context) {
    return warmupCarPass;
  }
  if (warmupCarPass !== null) {
    warmupCarPass.dispose();
  }
  warmupCarPass = createCarPass(context, NEATRACING_WARMUP_MODELS);
  warmupCarPassContext = context;
  return warmupCarPass;
}

/**
 * Builds the car warmup program descriptor for the shared warmup registry
 * (key 'car', NEATRACING_WARMED_PROGRAM_KEYS[4]). Compilation is lazy: no GL
 * program exists until compileAndLink runs, and drawOffscreen pushes the
 * zero-pose frame through the same draw path production uses.
 *
 * @param context - WebGL2 context the warmup shader will run against.
 * @returns The warmup program descriptor keyed 'car'.
 *
 * @example
 * programRegistry.register(toCarWarmupProgram(gl));
 */
export function toCarWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram {
  return {
    key: NEATRACING_CAR_PROGRAM_KEY,
    compileAndLink: (): void => {
      ensureWarmupCarPass(context);
    },
    drawOffscreen: (): void => {
      ensureWarmupCarPass(context).draw(NEATRACING_ZERO_POSE_DRAW_STATE);
    },
  };
}

/**
 * Creates the car render pass: compiles the lit + blob shadow programs
 * eagerly, resolves uniforms throw-on-missing, expands and uploads both
 * voxel bins plus the shadow quad, and resolves the four wheel mount
 * transforms once at creation (they allocate per call; keeping them
 * pre-resolved keeps the draw path allocation-free, P9).
 *
 * @param gl - WebGL2 context.
 * @param models - Body and wheel voxel bins to render ({@link CarPassModels}).
 * @returns The car pass handle ({@link CarPass}).
 * @throws When models are missing, shaders fail, or a uniform is absent.
 *
 * @example
 * const carPass = createCarPass(gl, { body: bodyBin, wheel: wheelBin });
 * carPass.draw({ viewProjection, cameraWorld, cars });
 * carPass.dispose();
 */
export function createCarPass(
  gl: WebGL2RenderingContext,
  models: CarPassModels,
): CarPass {
  if (
    models === null ||
    typeof models !== 'object' ||
    models.body === null ||
    models.wheel === null
  ) {
    throw new Error(
      `${NEATRACING_CAR_PASS_LABEL}: models must provide a body and a wheel VoxelBinModel`,
    );
  }
  const litCompiled = compileAndLinkCarProgram(
    gl,
    CAR_LIT_VERTEX_SHADER_SOURCE,
    CAR_LIT_FRAGMENT_SHADER_SOURCE,
    'car lit',
  );
  const shadowCompiled = compileAndLinkCarProgram(
    gl,
    CAR_SHADOW_VERTEX_SHADER_SOURCE,
    CAR_SHADOW_FRAGMENT_SHADER_SOURCE,
    'car shadow',
  );
  const litUniforms = resolveCarUniformLocations(
    gl,
    litCompiled.program,
    CAR_LIT_UNIFORM_NAMES,
  );
  const shadowUniforms = resolveCarUniformLocations(
    gl,
    shadowCompiled.program,
    CAR_SHADOW_UNIFORM_NAMES,
  );
  const litAttributes = resolveCarLitAttributeLocations(
    gl,
    litCompiled.program,
  );
  const shadowCornerAttributeId = gl.getAttribLocation(
    shadowCompiled.program,
    'aCorner',
  );
  const bodyMesh = expandVoxelBinModelToCarMesh(models.body);
  const wheelMesh = expandVoxelBinModelToCarMesh(models.wheel, 'center');
  const bodyBuffer = uploadCarFloat32Buffer(gl, bodyMesh.vertexData);
  const wheelBuffer = uploadCarFloat32Buffer(gl, wheelMesh.vertexData);
  const shadowBuffer = uploadCarFloat32Buffer(gl, CAR_SHADOW_QUAD_CORNERS);
  const wheelMounts: readonly VoxelWheelMountTransform[] =
    resolveVoxelWheelSockets().map((socket) =>
      resolveVoxelWheelMountTransform(socket.id),
    );
  // The mesh bounds field is a HALF extent, so the body's full XZ footprint
  // is 2 x boundsHalfExtentXzM; the shadow quad world width equals this
  // scale (the unit quad is centered by the vertex shader), targeting full
  // footprint x NEATRACING_BLOB_SHADOW_BOUNDS_SCALE.
  const shadowScale =
    bodyMesh.boundsHalfExtentXzM * 2 * NEATRACING_BLOB_SHADOW_BOUNDS_SCALE;

  /**
   * Enables and points the five lit attributes on the currently bound
   * buffer. Called after each bindBuffer because attribute pointers are
   * buffer-coupled client state without a bound VAO.
   */
  function pointLitMeshAttributes(): void {
    gl.enableVertexAttribArray(litAttributes.aPosition);
    gl.enableVertexAttribArray(litAttributes.aColor);
    gl.enableVertexAttribArray(litAttributes.aNormalAxis);
    gl.enableVertexAttribArray(litAttributes.aEmissive);
    gl.enableVertexAttribArray(litAttributes.aPaletteIndex);
    gl.vertexAttribPointer(
      litAttributes.aPosition,
      3,
      GL_FLOAT,
      false,
      CAR_MESH_VERTEX_STRIDE_BYTES,
      CAR_MESH_POSITION_OFFSET_BYTES,
    );
    gl.vertexAttribPointer(
      litAttributes.aColor,
      3,
      GL_FLOAT,
      false,
      CAR_MESH_VERTEX_STRIDE_BYTES,
      CAR_MESH_COLOR_OFFSET_BYTES,
    );
    gl.vertexAttribPointer(
      litAttributes.aNormalAxis,
      1,
      GL_FLOAT,
      false,
      CAR_MESH_VERTEX_STRIDE_BYTES,
      CAR_MESH_NORMAL_AXIS_OFFSET_BYTES,
    );
    gl.vertexAttribPointer(
      litAttributes.aEmissive,
      1,
      GL_FLOAT,
      false,
      CAR_MESH_VERTEX_STRIDE_BYTES,
      CAR_MESH_EMISSIVE_OFFSET_BYTES,
    );
    gl.vertexAttribPointer(
      litAttributes.aPaletteIndex,
      1,
      GL_FLOAT,
      false,
      CAR_MESH_VERTEX_STRIDE_BYTES,
      CAR_MESH_PALETTE_INDEX_OFFSET_BYTES,
    );
  }

  /**
   * Draws loop 1: one blob shadow quad per car, depth-write off, blended, at
   * world y + NEATRACING_BLOB_SHADOW_Y_OFFSET_M, scaled to the body's full
   * XZ footprint plus the NEATRACING_BLOB_SHADOW_BOUNDS_SCALE margin.
   *
   * @param state - Frame draw state.
   * @param carCount - Number of cars sorted into the scratch order.
   */
  function drawBlobShadows(state: CarPassDrawState, carCount: number): void {
    gl.useProgram(shadowCompiled.program);
    gl.enable(GL_DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(GL_BLEND);
    gl.blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    gl.bindBuffer(GL_ARRAY_BUFFER, shadowBuffer);
    gl.enableVertexAttribArray(shadowCornerAttributeId);
    gl.vertexAttribPointer(
      shadowCornerAttributeId,
      CAR_SHADOW_FLOATS_PER_VERTEX,
      GL_FLOAT,
      false,
      CAR_SHADOW_VERTEX_STRIDE_BYTES,
      0,
    );
    gl.uniformMatrix4fv(
      shadowUniforms.uViewProjection,
      false,
      state.viewProjection,
    );
    uniform3fColor(
      gl,
      shadowUniforms.uShadowColor,
      NEATRACING_BLOB_SHADOW_COLOR,
    );
    for (let slot = 0; slot < carCount; slot += 1) {
      const carIndex = NEATRACING_CAR_SORT_ORDER_SCRATCH[slot];
      const car = state.cars[carIndex];
      translationMat4(
        NEATRACING_SHADOW_TRANSLATION_MATRIX,
        car.worldX,
        NEATRACING_BLOB_SHADOW_Y_OFFSET_M,
        car.worldZ,
      );
      rotationYMat4(NEATRACING_SHADOW_YAW_MATRIX, car.yawRad);
      multiplyMat4(
        NEATRACING_CAR_MATRIX_SCRATCH_A,
        NEATRACING_SHADOW_TRANSLATION_MATRIX,
        NEATRACING_SHADOW_YAW_MATRIX,
      );
      scaleXYZMat4(NEATRACING_SHADOW_SCALE_MATRIX, shadowScale, 1, shadowScale);
      multiplyMat4(
        NEATRACING_SHADOW_MATRIX,
        NEATRACING_CAR_MATRIX_SCRATCH_A,
        NEATRACING_SHADOW_SCALE_MATRIX,
      );
      gl.uniformMatrix4fv(
        shadowUniforms.uModel,
        false,
        NEATRACING_SHADOW_MATRIX,
      );
      gl.drawArrays(GL_TRIANGLES, 0, CAR_SHADOW_VERTEX_COUNT);
    }
  }

  /**
   * Feeds the draw-time team remap uniforms for one car (AC-014): the
   * normalized team color plus the shared remap slot. Body and wheels share
   * the rule because the shader swaps albedo wherever the vertex flags the
   * slot palette index.
   *
   * @param car - Car being drawn.
   */
  function feedTeamRemapUniforms(car: CarPassCarState): void {
    const teamRgb = resolveVoxelTeamColor(car.teamColorId as VoxelTeamColorId);
    gl.uniform3f(
      litUniforms.uTeamColor,
      teamRgb[0] / NEATRACING_CAR_COLOR_BYTE_MAX,
      teamRgb[1] / NEATRACING_CAR_COLOR_BYTE_MAX,
      teamRgb[2] / NEATRACING_CAR_COLOR_BYTE_MAX,
    );
    gl.uniform1f(litUniforms.uTeamSlot, NEATRACING_TEAM_REMAP_SLOT);
  }

  /**
   * Draws one lit mesh (body or wheel) with the composed model matrix and
   * the current car's team remap uniforms.
   *
   * @param matrix - Column-major model matrix to feed via uModel.
   * @param vertexCount - Vertex count of the mesh being drawn.
   * @param car - Car whose team remap feeds this draw.
   */
  function drawLitMesh(
    matrix: Float32Array,
    vertexCount: number,
    car: CarPassCarState,
  ): void {
    gl.uniformMatrix4fv(litUniforms.uModel, false, matrix);
    feedTeamRemapUniforms(car);
    gl.drawArrays(GL_TRIANGLES, 0, vertexCount);
  }

  /**
   * Draws loop 2: body then four wheels per car (far -> near), two-sided
   * with the shared lighting model. Wheels compose
   * carModel x translate(socket) x rotY(mountYaw) x spin(model-depth axis).
   * The body VBO rebinds before every body draw because the wheel phase
   * leaves the wheel VBO bound and the attribute pointers are buffer-coupled
   * client state without a bound VAO.
   *
   * @param state - Frame draw state.
   * @param carCount - Number of cars sorted into the scratch order.
   */
  function drawLitCars(state: CarPassDrawState, carCount: number): void {
    gl.depthMask(true);
    gl.disable(GL_BLEND);
    gl.useProgram(litCompiled.program);
    gl.disable(GL_CULL_FACE);
    gl.enable(GL_DEPTH_TEST);
    gl.uniformMatrix4fv(
      litUniforms.uViewProjection,
      false,
      state.viewProjection,
    );
    uniform3fColor(gl, litUniforms.uSunDirection, NEATRACING_SUN_DIRECTION);
    uniform3fColor(gl, litUniforms.uSunColor, NEATRACING_SUN_COLOR);
    uniform3fColor(
      gl,
      litUniforms.uHemisphereSky,
      NEATRACING_HEMISPHERE_SKY_COLOR,
    );
    uniform3fColor(
      gl,
      litUniforms.uHemisphereGround,
      NEATRACING_HEMISPHERE_GROUND_COLOR,
    );
    gl.uniform1f(litUniforms.uFogDensity, NEATRACING_FOG_DENSITY_1_PER_M);
    uniform3fColor(gl, litUniforms.uFogColor, NEATRACING_FOG_COLOR);
    gl.uniform3f(
      litUniforms.uCameraWorld,
      state.cameraWorld[0],
      state.cameraWorld[1],
      state.cameraWorld[2],
    );
    for (let slot = 0; slot < carCount; slot += 1) {
      const carIndex = NEATRACING_CAR_SORT_ORDER_SCRATCH[slot];
      const car = state.cars[carIndex];
      translationMat4(
        NEATRACING_CAR_TRANSLATION_MATRIX,
        car.worldX,
        0,
        car.worldZ,
      );
      rotationYMat4(NEATRACING_CAR_YAW_MATRIX, car.yawRad);
      multiplyMat4(
        NEATRACING_CAR_MODEL_MATRIX,
        NEATRACING_CAR_TRANSLATION_MATRIX,
        NEATRACING_CAR_YAW_MATRIX,
      );
      // Rebind per car: the wheel phase left the wheel VBO bound, so the
      // body draw must rebind the body VBO and its attribute pointers
      // (buffer-coupled client state, no VAO).
      gl.bindBuffer(GL_ARRAY_BUFFER, bodyBuffer);
      pointLitMeshAttributes();
      drawLitMesh(NEATRACING_CAR_MODEL_MATRIX, bodyMesh.vertexCount, car);
      gl.bindBuffer(GL_ARRAY_BUFFER, wheelBuffer);
      pointLitMeshAttributes();
      for (const mount of wheelMounts) {
        translationMat4(
          NEATRACING_WHEEL_TRANSLATION_MATRIX,
          mount.position[0],
          mount.position[1],
          mount.position[2],
        );
        multiplyMat4(
          NEATRACING_CAR_MATRIX_SCRATCH_A,
          NEATRACING_CAR_MODEL_MATRIX,
          NEATRACING_WHEEL_TRANSLATION_MATRIX,
        );
        rotationYMat4(NEATRACING_WHEEL_MOUNT_YAW_MATRIX, mount.rotationYawRad);
        multiplyMat4(
          NEATRACING_CAR_MATRIX_SCRATCH_B,
          NEATRACING_CAR_MATRIX_SCRATCH_A,
          NEATRACING_WHEEL_MOUNT_YAW_MATRIX,
        );
        rotationZMat4(NEATRACING_WHEEL_SPIN_MATRIX, car.wheelSpinRad);
        multiplyMat4(
          NEATRACING_CAR_MATRIX_SCRATCH_A,
          NEATRACING_CAR_MATRIX_SCRATCH_B,
          NEATRACING_WHEEL_SPIN_MATRIX,
        );
        drawLitMesh(
          NEATRACING_CAR_MATRIX_SCRATCH_A,
          wheelMesh.vertexCount,
          car,
        );
      }
    }
  }

  return {
    draw(state: CarPassDrawState): void {
      if (state.viewProjection.length !== 16) {
        throw new Error(
          `${NEATRACING_CAR_PASS_LABEL}: viewProjection must contain exactly 16 elements, received ${String(state.viewProjection.length)}`,
        );
      }
      if (state.cameraWorld.length !== 3) {
        throw new Error(
          `${NEATRACING_CAR_PASS_LABEL}: cameraWorld must contain exactly 3 components, received ${String(state.cameraWorld.length)}`,
        );
      }
      composeSortedCarOrder(state);
      const carCount = state.cars.length;
      drawBlobShadows(state, carCount);
      drawLitCars(state, carCount);
    },
    dispose(): void {
      gl.deleteProgram(litCompiled.program);
      gl.deleteProgram(shadowCompiled.program);
      gl.deleteShader(litCompiled.vertexShader);
      gl.deleteShader(litCompiled.fragmentShader);
      gl.deleteShader(shadowCompiled.vertexShader);
      gl.deleteShader(shadowCompiled.fragmentShader);
      gl.deleteBuffer(bodyBuffer);
      gl.deleteBuffer(wheelBuffer);
      gl.deleteBuffer(shadowBuffer);
    },
  };
}
