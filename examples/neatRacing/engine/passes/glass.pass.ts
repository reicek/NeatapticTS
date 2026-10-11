/**
 * NeatRacing glass render pass (slice P1-S4-E; spec section 6.5 G4, AC-013
 * part 3, and the section 7.3.2 winding/glass exemption note).
 *
 * G4 concretizes the LD-23 glass voxels: a dedicated glass pass runs after
 * ALL opaque geometry with **depth-test ON, depth-write OFF, back-face
 * culling ON**, and every glass quad is tinted with **premultiplied-alpha
 * color** plus an **additive glow contribution** sampled from the palette's
 * glow byte. Cars draw far -> near as cheap insurance; stacked glass inside
 * one car double-tints, which the spec explicitly locks as accepted.
 *
 * Structural sibling of `car.pass.ts` (numeric GL enums, exported GLSL ES
 * 3.00 sources, compile helpers that throw with the pass label,
 * throw-on-missing uniform resolution, a lazy warmup seam) with ONE program:
 *
 * 1. The glass tint program draws bins-INDEPENDENT geometry. Glass cells are
 *    alpha-0 air for the mesher (section 7.3.2), so the greedy mesher emits
 *    no glass quads and this pass derives its quads straight from
 *    `DecodedVoxelModel` slices: six full-faces per glass cell, emitted in a
 *    deterministic z-asc -> y-asc -> x-asc lattice scan with a fixed face
 *    order (-x, +x, -y, +y, -z, +z) and TRUE winding (front faces point
 *    outward; verified by cross-product tests below).
 *
 * Fragment math is premultiplied alpha: the shader emits
 * `fragColor = vec4(tint.rgb * (kTintAlpha + vEmissive), kTintAlpha)` and the
 * pass pins `gl.blendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)`, so a glass layer
 * contributes its tint plus glow additively while softly attenuating only
 * what is behind it. Black palette entries classify as `black` BEFORE glass
 * in the shared material table (LD-23), so black can never fabricate glass
 * geometry: black never shines in the glass pass either (AC-013).
 *
 * No per-frame allocation: every matrix, sort scratch, and mesh is
 * preallocated at module or creation scope (P9).
 *
 * @module
 */

import {
  NEATRACING_WARMED_PROGRAM_KEYS,
  type ShaderWarmupProgram,
} from '../context.service';
import {
  buildVoxelMaterialTable,
  type VoxelMaterial,
} from '../../voxel/materials';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../../voxel/lod.distance.utils';
import type { DecodedVoxelModel } from '../../voxel/model.decode.utils';
import {
  resolveVoxelWheelMountTransform,
  resolveVoxelWheelSockets,
  type VoxelWheelMountTransform,
} from '../../voxel/wheel.mount.utils';
import {
  VOXEL_EMPTY_PALETTE_ENTRY,
  type VoxelPaletteEntry,
} from '../../voxel/model.schema';
import type { CarMeshYAnchor } from './car.pass';

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

/** `gl.DEPTH_TEST` (0x0B71) enabled for the glass pass (G4). */
const GL_DEPTH_TEST = 0x0b71;

/** `gl.CULL_FACE` (0x0B44) enabled with back-face culling ON (G4). */
const GL_CULL_FACE = 0x0b44;

/** `gl.BACK` (0x0405) cull target: only back faces are culled. */
const GL_CULL_FACE_BACK = 0x0405;

/** `gl.CCW` (0x0901) front-face order expected by the true glass winding. */
const GL_FRONT_FACE_CCW = 0x0901;

/** `gl.BLEND` (0x0BE2) enabled for premultiplied glass tinting. */
const GL_BLEND = 0x0be2;

/** `gl.ONE` (1) blend source factor for premultiplied-alpha glass. */
const GL_ONE = 1;

/**
 * `gl.ONE_MINUS_SRC_ALPHA` (0x0303) blend destination factor: the G4 pinned
 * pairing with {@link GL_ONE} (spec section 6.5 glass policy).
 */
const GL_ONE_MINUS_SRC_ALPHA = 0x0303;

/** `gl.TRIANGLES` (4) draw mode for every glass stage. */
const GL_TRIANGLES = 4;

/** Pass label prefixing every compile/link/uniform rejection message. */
const NEATRACING_GLASS_PASS_LABEL = 'neatRacing glass pass';

/**
 * Mild premultiplied glass tint alpha (G4: "arcade glass is thin and tint
 * is mild"; double-tinting stacked panes is explicitly accepted, so the
 * per-layer alpha stays low). Pinned so GLSL interpolation cannot drift.
 */
export const NEATRACING_GLASS_TINT_ALPHA = 0.18;

/** Byte divisor used when normalizing palette colors/glow into [0, 1]. */
const NEATRACING_GLASS_COLOR_BYTE_MAX = 255;

/** Floats per expanded glass vertex: position, color, emissive. */
export const GLASS_MESH_FLOATS_PER_VERTEX = 7;

/** Bytes per expanded glass vertex (7 floats x 4 bytes). */
export const GLASS_MESH_VERTEX_STRIDE_BYTES = GLASS_MESH_FLOATS_PER_VERTEX * 4;

/** Position component byte offset inside an expanded glass vertex. */
const GLASS_MESH_POSITION_OFFSET_BYTES = 0;

/** Color component byte offset inside an expanded glass vertex. */
const GLASS_MESH_COLOR_OFFSET_BYTES = 12;

/** Emissive component byte offset inside an expanded glass vertex. */
const GLASS_MESH_EMISSIVE_OFFSET_BYTES = 24;

/** Expanded vertices per glass face: two non-indexed triangles. */
export const NEATRACING_GLASS_VERTICES_PER_FACE = 6;

/** Corner index order per face: triangles (c0, c1, c2) and (c0, c2, c3). */
const NEATRACING_GLASS_TRIANGLE_CORNER_INDICES = [0, 1, 2, 0, 2, 3] as const;

/**
 * Glass faces emitted per glass cell, in fixed face order
 * (-x, +x, -y, +y, -z, +z); derived from the face table length so the
 * count and the winding table can never drift.
 */
export const NEATRACING_GLASS_FACES_PER_CELL = 6;

/**
 * Cell-local corner table for glass faces, in emission face order with
 * *true winding in MODEL SPACE*: the model mapping keeps the grid Y axis
 * monotonic up (`modelY` grows with `gridY`; row 0 stays ground/down — the
 * same anchors as the car pass), so every face's corner quad is listed in
 * natural lattice-CCW order and
 * `cross(v1 - v0, v2 - v0) = outward model-space normal` exactly
 * (asserted by the winding contract tests, so `frontFace(GL_CCW)` +
 * `cullFace(GL_BACK)` keep only outward-facing glass under G4). Face
 * planes sit at the cell's minus corner for minus faces and at
 * `cell[axis] + 1` for plus faces.
 */
const GLASS_FACE_CORNERS: readonly (readonly (readonly [
  number,
  number,
  number,
])[])[] = [
  // Face 0: -x (plane at cell.x); outward model normal (-1, 0, 0).
  [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 0],
  ],
  // Face 1: +x (plane at cell.x + 1); outward model normal (+1, 0, 0).
  [
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
    [1, 0, 1],
  ],
  // Face 2: -y (plane at cell.y; model bottom); outward model normal (0, -1, 0).
  [
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 1],
    [0, 0, 1],
  ],
  // Face 3: +y (plane at cell.y + 1; model top); outward model normal (0, +1, 0).
  [
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 0],
  ],
  // Face 4: -z (plane at cell.z); outward model normal (0, 0, -1).
  [
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
  ],
  // Face 5: +z (plane at cell.z + 1); outward model normal (0, 0, +1).
  [
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
  ],
];

/**
 * Fallback material for out-of-range palette indices: renders like LD-23
 * black (never glass, zero glow) so a malformed voxel byte can never
 * fabricate glass geometry.
 */
const GLASS_MISSING_MATERIAL: VoxelMaterial = {
  kind: 'black',
  color: [0, 0, 0],
  alpha: 255,
  emissive: 0,
};

/** Glass-program uniform names resolved with throw-on-missing semantics. */
const GLASS_UNIFORM_NAMES: readonly string[] = ['uViewProjection', 'uModel'];

/** Glass-program attribute names resolved in expansion float order. */
const GLASS_ATTRIBUTE_NAMES = ['aPosition', 'aColor', 'aEmissive'] as const;

/**
 * Glass vertex shader: transforms one glass face corner into clip space and
 * passes the palette tint plus glow contribution through.
 */
export const NEATRACING_GLASS_VERTEX_SHADER_SOURCE = `#version 300 es
in vec3 aPosition;
in vec3 aColor;
in float aEmissive;
uniform mat4 uViewProjection;
uniform mat4 uModel;
out vec3 vTint;
out float vEmissive;
void main() {
  vTint = aColor;
  vEmissive = aEmissive;
  gl_Position = uViewProjection * uModel * vec4(aPosition, 1.0);
}`;

/**
 * Glass fragment shader (G4): premultiplied-alpha tint plus additive glow.
 * `fragColor.rgb = tint.rgb * (kTintAlpha + vEmissive)` with
 * `fragColor.a = kTintAlpha` and the pass blending
 * `blendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)`, so the tint softens what is
 * behind the pane while the palette glow contribution adds on top.
 */
export const NEATRACING_GLASS_FRAGMENT_SHADER_SOURCE = `#version 300 es
precision highp float;
const float kTintAlpha = ${NEATRACING_GLASS_TINT_ALPHA};
in vec3 vTint;
in float vEmissive;
out vec4 fragColor;
void main() {
  fragColor = vec4(vTint * (kTintAlpha + vEmissive), kTintAlpha);
}`;

/** Glass pose inside one draw frame. */
export interface GlassPassCarState {
  /** Car world X in meters. */
  readonly worldX: number;
  /** Car world Z in meters. */
  readonly worldZ: number;
  /** Car yaw around world up in radians. */
  readonly yawRad: number;
  /** Wheel spin in radians shared by all four wheels (LD-21 mount parity). */
  readonly wheelSpinRad: number;
}

/** Per-frame draw state accepted by GlassPass.draw. */
export interface GlassPassDrawState {
  /** Column-major view-projection (16 floats). */
  readonly viewProjection: Float32Array;
  /** Camera world position [x, y, z] in meters (far -> near sort anchor). */
  readonly cameraWorld: readonly [number, number, number];
  /** On-track cars in any order; the pass sorts far -> near internally. */
  readonly cars: readonly GlassPassCarState[];
}

/** Models the pass renders: the car body and wheel DECODED voxel models. */
export interface GlassPassModels {
  /** Car body decoded voxel model (glass quads derived from slices). */
  readonly body: DecodedVoxelModel;
  /** Wheel decoded voxel model (glass quads derived from slices). */
  readonly wheel: DecodedVoxelModel;
}

/**
 * Mesh produced by {@link expandGlassModelToGlassMesh}: interleaved
 * non-indexed triangles ready for one FLOAT-attributed VBO per model.
 */
export interface GlassVoxelMesh {
  /** Interleaved vertex data on the GLASS_MESH_VERTEX_STRIDE_BYTES stride. */
  readonly vertexData: Float32Array;
  /** Total expanded vertex count (6 per glass face). */
  readonly vertexCount: number;
  /** Glass cells counted by the shared material table (AC-013 filter). */
  readonly glassCellCount: number;
}

/** Glass pass handle returned by createGlassPass. */
export interface GlassPass {
  /**
   * Draws far -> near sorted glass for every car: the body glass mesh and
   * the wheel glass meshes on the section 7.4 mount transforms, all through
   * the single G4 state machine (depth-test ON, depth-write OFF, back-face
   * culling ON, premultiplied-alpha blend).
   *
   * @param state - Frame draw state (see {@link GlassPassDrawState}).
   * @throws When viewProjection is not 16 elements or cameraWorld is not 3.
   */
  draw(state: GlassPassDrawState): void;
  /** Releases the program, both shaders, and both vertex buffers. */
  dispose(): void;
}

/**
 * Expands a decoded voxel model into the glass pass vertex format (pure,
 * GL-free).
 *
 * Every palette entry classified `glass` by the shared material table
 * (fully transparent, colored — spec section 7.2) contributes six full
 * faces with TRUE winding; every other kind — including `black`, which the
 * shared table resolves BEFORE glass with zero emissive — contributes
 * nothing, so black can never shine in the glass pass (AC-013).
 *
 * Grid positions map to model-space meters exactly like the car pass
 * (`expandVoxelBinModelToCarMesh`): `modelX = (gx - dimX * 0.5) *
 * voxelSize` and `modelZ = (gz - dimZ * 0.5) * voxelSize` are
 * lattice-centered, while `modelY` follows the `yAnchor` argument —
 * `'ground'` (`modelY = gridY * voxelSize`, bodies authored with row 0 on
 * the floor) or `'center'` (`modelY = (gridY - dimY * 0.5) * voxelSize`,
 * lattice-centered wheel discs whose tire bottom must land at the negative
 * hub height) — so glass aligns one-to-one with the bins-derived body
 * instead of floating a Y-flipped copy above the car. Colors resolve from
 * `buildVoxelMaterialTable` (bytes / 255) and emissive is glow / 255. The
 * lattice scan runs z-asc -> y-asc -> x-asc with the fixed face order
 * (-x, +x, -y, +y, -z, +z), so the same model always expands to the same
 * byte-stable vertex data.
 *
 * @param model - Decoded voxel model to filter.
 * @param yAnchor - Vertical anchoring contract: `'ground'` for bodies,
 *   `'center'` for wheel discs. Defaults to `'ground'`.
 * @returns The expanded glass mesh plus its glass cell count.
 *
 * @example
 * const mesh = expandGlassModelToGlassMesh(decodedBody);
 * // mesh.vertexCount === mesh.glassCellCount * 36; 7 floats per vertex
 */
export function expandGlassModelToGlassMesh(
  model: DecodedVoxelModel,
  yAnchor: CarMeshYAnchor = 'ground',
): GlassVoxelMesh {
  const materials = buildVoxelMaterialTable(model.palette);
  const voxelSize = NEATRACING_VOXEL_WORLD_SIZE;
  const [dimX, dimY, dimZ] = model.dims;
  const latticeCenterRows = dimY * 0.5;
  let glassCellCount = 0;
  for (let z = 0; z < dimZ; z += 1) {
    const slice = model.slices[z];
    for (let y = 0; y < dimY; y += 1) {
      const rowBase = y * dimX;
      for (let x = 0; x < dimX; x += 1) {
        const material =
          materials[slice[rowBase + x]] ?? GLASS_MISSING_MATERIAL;
        if (material.kind === 'glass') {
          glassCellCount += 1;
        }
      }
    }
  }
  const faceVertexCount =
    glassCellCount *
    NEATRACING_GLASS_FACES_PER_CELL *
    NEATRACING_GLASS_VERTICES_PER_FACE;
  const vertexData = new Float32Array(
    faceVertexCount * GLASS_MESH_FLOATS_PER_VERTEX,
  );
  let writeOffset = 0;
  const pushCorner = (
    x: number,
    y: number,
    z: number,
    material: VoxelMaterial,
    corner: readonly [number, number, number],
  ): void => {
    const gridX = x + corner[0];
    const gridY = y + corner[1];
    const gridZ = z + corner[2];
    vertexData[writeOffset] = (gridX - dimX * 0.5) * voxelSize;
    vertexData[writeOffset + 1] =
      yAnchor === 'center'
        ? (gridY - latticeCenterRows) * voxelSize
        : gridY * voxelSize;
    vertexData[writeOffset + 2] = (gridZ - dimZ * 0.5) * voxelSize;
    vertexData[writeOffset + 3] =
      material.color[0] / NEATRACING_GLASS_COLOR_BYTE_MAX;
    vertexData[writeOffset + 4] =
      material.color[1] / NEATRACING_GLASS_COLOR_BYTE_MAX;
    vertexData[writeOffset + 5] =
      material.color[2] / NEATRACING_GLASS_COLOR_BYTE_MAX;
    vertexData[writeOffset + 6] =
      material.emissive / NEATRACING_GLASS_COLOR_BYTE_MAX;
    writeOffset += GLASS_MESH_FLOATS_PER_VERTEX;
  };
  for (let z = 0; z < dimZ; z += 1) {
    const slice = model.slices[z];
    for (let y = 0; y < dimY; y += 1) {
      const rowBase = y * dimX;
      for (let x = 0; x < dimX; x += 1) {
        const material =
          materials[slice[rowBase + x]] ?? GLASS_MISSING_MATERIAL;
        if (material.kind !== 'glass') {
          continue;
        }
        for (let face = 0; face < NEATRACING_GLASS_FACES_PER_CELL; face += 1) {
          const corners = GLASS_FACE_CORNERS[face];
          for (const cornerIndex of NEATRACING_GLASS_TRIANGLE_CORNER_INDICES) {
            pushCorner(x, y, z, material, corners[cornerIndex]);
          }
        }
      }
    }
  }
  return { vertexData, vertexCount: faceVertexCount, glassCellCount };
}

/**
 * Compiles one glass pass shader stage, rejecting failures with the pass
 * label (grid.pass.ts parity).
 *
 * @param gl - WebGL2 context.
 * @param shaderType - gl.VERTEX_SHADER (0x8B31) or gl.FRAGMENT_SHADER (0x8B30).
 * @param shaderSource - GLSL ES 3.00 source.
 * @param shaderLabel - Stage label used in the rejection message.
 * @returns The compiled shader handle.
 * @throws When compilation fails; the message carries the pass label.
 */
function compileGlassShader(
  gl: WebGL2RenderingContext,
  shaderType: number,
  shaderSource: string,
  shaderLabel: string,
): WebGLShader {
  const shader = gl.createShader(shaderType);
  if (shader === null) {
    throw new Error(
      `${NEATRACING_GLASS_PASS_LABEL}: failed to create the ${shaderLabel} shader`,
    );
  }
  gl.shaderSource(shader, shaderSource);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, GL_COMPILE_STATUS)) {
    const infoLog = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(
      `${NEATRACING_GLASS_PASS_LABEL}: failed to compile the ${shaderLabel} shader: ${String(infoLog)}`,
    );
  }
  return shader;
}

/** Compiled program bundle: the linked program plus its two live shaders. */
interface CompiledGlassProgram {
  /** Linked program handle. */
  readonly program: WebGLProgram;
  /** Live vertex shader handle (released by dispose). */
  readonly vertexShader: WebGLShader;
  /** Live fragment shader handle (released by dispose). */
  readonly fragmentShader: WebGLShader;
}

/**
 * Compiles and links the glass pass program from its two stages.
 *
 * @param gl - WebGL2 context.
 * @param vertexShaderSource - Source for the vertex stage.
 * @param fragmentShaderSource - Source for the fragment stage.
 * @param programLabel - Label used in every rejection message.
 * @returns The linked program plus its two shader handles.
 * @throws When compilation or linking fails.
 */
function compileAndLinkGlassProgram(
  gl: WebGL2RenderingContext,
  vertexShaderSource: string,
  fragmentShaderSource: string,
  programLabel: string,
): CompiledGlassProgram {
  const vertexShader = compileGlassShader(
    gl,
    GL_VERTEX_SHADER_TYPE,
    vertexShaderSource,
    `${programLabel} vertex`,
  );
  const fragmentShader = compileGlassShader(
    gl,
    GL_FRAGMENT_SHADER_TYPE,
    fragmentShaderSource,
    `${programLabel} fragment`,
  );
  const program = gl.createProgram();
  if (program === null) {
    throw new Error(
      `${NEATRACING_GLASS_PASS_LABEL}: failed to create the ${programLabel} program`,
    );
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, GL_LINK_STATUS)) {
    const infoLog = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(
      `${NEATRACING_GLASS_PASS_LABEL}: failed to link the ${programLabel} program: ${String(infoLog)}`,
    );
  }
  return { program, vertexShader, fragmentShader };
}

/**
 * Resolves the glass program's uniform locations with throw-on-missing
 * semantics so a shader/name drift fails loudly at pass creation (grid and
 * car pass parity).
 *
 * @param gl - WebGL2 context.
 * @param program - Linked program handle.
 * @param names - Uniform names to resolve, in feed order.
 * @returns Location table keyed by uniform name.
 * @throws When any uniform is missing from the linked program.
 */
function resolveGlassUniformLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  names: readonly string[],
): Record<string, WebGLUniformLocation> {
  const locations: Record<string, WebGLUniformLocation> = {};
  for (const name of names) {
    const location = gl.getUniformLocation(program, name);
    if (location === null) {
      throw new Error(
        `${NEATRACING_GLASS_PASS_LABEL}: missing uniform ${name}`,
      );
    }
    locations[name] = location;
  }
  return locations;
}

/**
 * Resolves the glass program's attribute locations in expansion float order
 * (position, color, emissive).
 *
 * @param gl - WebGL2 context.
 * @param program - Linked glass program handle.
 * @returns Attribute location table used by vertexAttribPointer feeds.
 * @throws When any attribute is missing from the linked program.
 */
function resolveGlassAttributeLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
): Record<(typeof GLASS_ATTRIBUTE_NAMES)[number], number> {
  const locations: Record<string, number> = {};
  for (const name of GLASS_ATTRIBUTE_NAMES) {
    const attributeId = gl.getAttribLocation(program, name);
    if (attributeId < 0) {
      throw new Error(
        `${NEATRACING_GLASS_PASS_LABEL}: missing attribute ${name}`,
      );
    }
    locations[name] = attributeId;
  }
  return locations as Record<(typeof GLASS_ATTRIBUTE_NAMES)[number], number>;
}

/**
 * Uploads one Float32 array into a STATIC_DRAW buffer.
 *
 * @param gl - WebGL2 context.
 * @param data - Vertex data to upload.
 * @returns The uploaded buffer handle.
 * @throws When gl.createBuffer returns null.
 */
function uploadGlassFloat32Buffer(
  gl: WebGL2RenderingContext,
  data: Float32Array,
): WebGLBuffer {
  const buffer = gl.createBuffer();
  if (buffer === null) {
    throw new Error(
      `${NEATRACING_GLASS_PASS_LABEL}: failed to create a glass vertex buffer`,
    );
  }
  gl.bindBuffer(GL_ARRAY_BUFFER, buffer);
  gl.bufferData(GL_ARRAY_BUFFER, data, GL_STATIC_DRAW);
  return buffer;
}

/** Car model matrix scratch (T x RY per car body). */
const NEATRACING_GLASS_TRANSLATION_MATRIX = new Float32Array(16);

/** Car yaw scratch fed by state.yawRad. */
const NEATRACING_GLASS_YAW_MATRIX = new Float32Array(16);

/** Composed car model matrix fed to uModel for the body glass draw. */
const NEATRACING_GLASS_MODEL_MATRIX = new Float32Array(16);

/** First matrix-multiply scratch (inputs never alias outputs). */
const NEATRACING_GLASS_MATRIX_SCRATCH_A = new Float32Array(16);

/** Second matrix-multiply scratch (kept separate so inputs never alias). */
const NEATRACING_GLASS_MATRIX_SCRATCH_B = new Float32Array(16);

/** Wheel hub translation scratch. */
const NEATRACING_GLASS_WHEEL_TRANSLATION_MATRIX = new Float32Array(16);

/** Wheel mount yaw scratch. */
const NEATRACING_GLASS_WHEEL_MOUNT_YAW_MATRIX = new Float32Array(16);

/** Wheel spin scratch (about the pre-mount model depth axis). */
const NEATRACING_GLASS_WHEEL_SPIN_MATRIX = new Float32Array(16);

/** Identity column-major matrix values reused by every helper below. */
const NEATRACING_GLASS_MAT4_IDENTITY: readonly number[] = [
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
    out[index] = NEATRACING_GLASS_MAT4_IDENTITY[index];
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
 * Writes a rotation-around-the-model-depth-axis matrix into a scratch
 * buffer (the pre-mount wheel model's spin axis; column-major).
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
 * `out = a * b` (a point is transformed by b first, then a). Call sites
 * must pass a dedicated scratch that aliases neither input; every scratch
 * here is preallocated at module scope to keep the draw path
 * allocation-free (P9).
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

/** Reused insertion-sort index scratch (grown once per fleet size, no churn). */
const NEATRACING_GLASS_SORT_ORDER_SCRATCH: number[] = [];

/** Reused squared-distance scratch for the far -> near sort (P9). */
let NEATRACING_GLASS_SORT_DISTANCE_SCRATCH = new Float64Array(0);

/**
 * Grows the sort scratches to the fleet size once; steady-state frames
 * never reallocate because scratch capacity only ever grows (P9).
 *
 * @param carCount - Cars in the current frame.
 */
function ensureGlassSortScratchCapacity(carCount: number): void {
  if (NEATRACING_GLASS_SORT_DISTANCE_SCRATCH.length < carCount) {
    NEATRACING_GLASS_SORT_DISTANCE_SCRATCH = new Float64Array(carCount);
  }
  if (NEATRACING_GLASS_SORT_ORDER_SCRATCH.length < carCount) {
    NEATRACING_GLASS_SORT_ORDER_SCRATCH.length = carCount;
  }
}

/**
 * Fills the sort scratch with the far -> near glass draw order, anchored at
 * the camera's XZ position. Stable insertion sort over squared camera
 * distances: equal distances keep array order, so the ordering is
 * deterministic frame to frame (spec section 6.5 "cheap insurance").
 *
 * @param state - Frame draw state (cars + camera world position).
 */
function composeSortedGlassOrder(state: GlassPassDrawState): void {
  const carCount = state.cars.length;
  const cameraX = state.cameraWorld[0];
  const cameraZ = state.cameraWorld[2];
  ensureGlassSortScratchCapacity(carCount);
  for (let index = 0; index < carCount; index += 1) {
    const car = state.cars[index];
    const deltaX = car.worldX - cameraX;
    const deltaZ = car.worldZ - cameraZ;
    NEATRACING_GLASS_SORT_ORDER_SCRATCH[index] = index;
    NEATRACING_GLASS_SORT_DISTANCE_SCRATCH[index] =
      deltaX * deltaX + deltaZ * deltaZ;
  }
  for (let slot = 1; slot < carCount; slot += 1) {
    const movingIndex = NEATRACING_GLASS_SORT_ORDER_SCRATCH[slot];
    const movingDistance = NEATRACING_GLASS_SORT_DISTANCE_SCRATCH[movingIndex];
    let target = slot - 1;
    while (
      target >= 0 &&
      NEATRACING_GLASS_SORT_DISTANCE_SCRATCH[
        NEATRACING_GLASS_SORT_ORDER_SCRATCH[target]
      ] < movingDistance
    ) {
      NEATRACING_GLASS_SORT_ORDER_SCRATCH[target + 1] =
        NEATRACING_GLASS_SORT_ORDER_SCRATCH[target];
      target -= 1;
    }
    NEATRACING_GLASS_SORT_ORDER_SCRATCH[target + 1] = movingIndex;
  }
}

/**
 * Warmup-only decoded model: a 2-cubed grid whose palette carries exactly
 * one glass entry so the warmup path exercises the real compile + glass mesh
 * draw pipeline without loaded track assets.
 */
const NEATRACING_GLASS_WARMUP_PALETTE: readonly VoxelPaletteEntry[] = [
  [...VOXEL_EMPTY_PALETTE_ENTRY] as VoxelPaletteEntry,
  [190, 225, 255, 0, 160] as VoxelPaletteEntry,
];

/** First warmup slice: all four cells reference the glass palette entry. */
const NEATRACING_GLASS_WARMUP_SLICE = new Uint8Array([1, 1, 1, 1]);

/** Warmup body model (every cell glass — full six-face derivation drawn). */
const NEATRACING_GLASS_WARMUP_BODY_MODEL: DecodedVoxelModel = {
  name: 'neatRacing glass warmup body',
  dims: [2, 2, 2],
  symmetric: false,
  palette: NEATRACING_GLASS_WARMUP_PALETTE,
  slices: [
    NEATRACING_GLASS_WARMUP_SLICE,
    NEATRACING_GLASS_WARMUP_SLICE,
    NEATRACING_GLASS_WARMUP_SLICE,
  ],
};

/** Warmup wheel model (same synthetic glass cube as the body). */
const NEATRACING_GLASS_WARMUP_WHEEL_MODEL: DecodedVoxelModel = {
  name: 'neatRacing glass warmup wheel',
  dims: [2, 2, 2],
  symmetric: false,
  palette: NEATRACING_GLASS_WARMUP_PALETTE,
  slices: [
    NEATRACING_GLASS_WARMUP_SLICE,
    NEATRACING_GLASS_WARMUP_SLICE,
    NEATRACING_GLASS_WARMUP_SLICE,
  ],
};

/** Warmup models consumed by the lazy glass pass instantiation. */
const NEATRACING_GLASS_WARMUP_MODELS: GlassPassModels = {
  body: NEATRACING_GLASS_WARMUP_BODY_MODEL,
  wheel: NEATRACING_GLASS_WARMUP_WHEEL_MODEL,
};

/**
 * Zero-pose warmup frame: identity projection, camera at the origin, one
 * car parked at the origin with no spin. drawOffscreen pushes this through
 * the shared draw path so warmup proves the full glass pipeline works.
 */
const NEATRACING_GLASS_ZERO_POSE_DRAW_STATE: GlassPassDrawState = {
  viewProjection: new Float32Array(NEATRACING_GLASS_MAT4_IDENTITY),
  cameraWorld: [0, 0, 0],
  cars: [
    {
      worldX: 0,
      worldZ: 0,
      yawRad: 0,
      wheelSpinRad: 0,
    },
  ],
};

/**
 * Warmup program key: registry index 5 of NEATRACING_WARMED_PROGRAM_KEYS
 * ('glass') is the contract pinned by the warmup seam test (AC-040).
 */
export const NEATRACING_GLASS_PROGRAM_KEY = NEATRACING_WARMED_PROGRAM_KEYS[5];

/** Lazily hydrated singleton pass behind the warmup seam (car parity). */
let warmupGlassPass: GlassPass | null = null;

/** Context the current warmup singleton was built against. */
let warmupGlassPassContext: WebGL2RenderingContext | null = null;

/**
 * Hydrates the singleton glass pass for the warmup seam, rebuilding only
 * when the context changes. Compilation stays lazy until compileAndLink
 * runs (car.pass.ts parity).
 *
 * @param context - WebGL2 context the warmup shader will run against.
 * @returns The hydrated singleton pass.
 */
function ensureWarmupGlassPass(context: WebGL2RenderingContext): GlassPass {
  if (warmupGlassPass !== null && warmupGlassPassContext === context) {
    return warmupGlassPass;
  }
  if (warmupGlassPass !== null) {
    warmupGlassPass.dispose();
  }
  warmupGlassPass = createGlassPass(context, NEATRACING_GLASS_WARMUP_MODELS);
  warmupGlassPassContext = context;
  return warmupGlassPass;
}

/**
 * Builds the glass warmup program descriptor for the shared warmup registry
 * (key 'glass', NEATRACING_WARMED_PROGRAM_KEYS[5]). Compilation is lazy: no
 * GL program exists until compileAndLink runs, and drawOffscreen pushes the
 * zero-pose frame through the same draw path production uses (G4 state
 * machine included).
 *
 * @param context - WebGL2 context the warmup shader will run against.
 * @returns The warmup program descriptor keyed 'glass'.
 *
 * @example
 * programRegistry.register(toGlassWarmupProgram(gl));
 */
export function toGlassWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram {
  return {
    key: NEATRACING_GLASS_PROGRAM_KEY,
    compileAndLink: (): void => {
      ensureWarmupGlassPass(context);
    },
    drawOffscreen: (): void => {
      ensureWarmupGlassPass(context).draw(
        NEATRACING_GLASS_ZERO_POSE_DRAW_STATE,
      );
    },
  };
}

/**
 * Creates the glass render pass: compiles the tint program eagerly, resolves
 * uniforms and attributes throw-on-missing, expands both decoded models into
 * glass meshes (AC-013 material-table filter; black contributes nothing),
 * and resolves the wheel mount transforms once at creation (they allocate
 * per call; keeping them pre-resolved keeps the draw path allocation-free,
 * P9).
 *
 * @param gl - WebGL2 context.
 * @param models - Body and wheel decoded models to render
 *   ({@link GlassPassModels}).
 * @returns The glass pass handle ({@link GlassPass}).
 * @throws When models are missing, shaders fail, or a uniform is absent.
 *
 * @example
 * const glassPass = createGlassPass(gl, { body: decodedBody, wheel: decodedWheel });
 * glassPass.draw({ viewProjection, cameraWorld, cars });
 * glassPass.dispose();
 */
export function createGlassPass(
  gl: WebGL2RenderingContext,
  models: GlassPassModels,
): GlassPass {
  if (
    models === null ||
    typeof models !== 'object' ||
    models.body === null ||
    models.wheel === null
  ) {
    throw new Error(
      `${NEATRACING_GLASS_PASS_LABEL}: models must provide a body and a wheel DecodedVoxelModel`,
    );
  }
  const compiled = compileAndLinkGlassProgram(
    gl,
    NEATRACING_GLASS_VERTEX_SHADER_SOURCE,
    NEATRACING_GLASS_FRAGMENT_SHADER_SOURCE,
    'glass tint',
  );
  const uniforms = resolveGlassUniformLocations(
    gl,
    compiled.program,
    GLASS_UNIFORM_NAMES,
  );
  const attributes = resolveGlassAttributeLocations(gl, compiled.program);
  const bodyMesh = expandGlassModelToGlassMesh(models.body);
  const wheelMesh = expandGlassModelToGlassMesh(models.wheel, 'center');
  const bodyBuffer = uploadGlassFloat32Buffer(gl, bodyMesh.vertexData);
  const wheelBuffer = uploadGlassFloat32Buffer(gl, wheelMesh.vertexData);
  const wheelMounts: readonly VoxelWheelMountTransform[] =
    resolveVoxelWheelSockets().map((socket) =>
      resolveVoxelWheelMountTransform(socket.id),
    );

  /**
   * Enables and points the three glass attributes on the currently bound
   * buffer. Called after each bindBuffer because attribute pointers are
   * buffer-coupled client state without a bound VAO.
   */
  function pointGlassMeshAttributes(): void {
    gl.enableVertexAttribArray(attributes.aPosition);
    gl.enableVertexAttribArray(attributes.aColor);
    gl.enableVertexAttribArray(attributes.aEmissive);
    gl.vertexAttribPointer(
      attributes.aPosition,
      3,
      GL_FLOAT,
      false,
      GLASS_MESH_VERTEX_STRIDE_BYTES,
      GLASS_MESH_POSITION_OFFSET_BYTES,
    );
    gl.vertexAttribPointer(
      attributes.aColor,
      3,
      GL_FLOAT,
      false,
      GLASS_MESH_VERTEX_STRIDE_BYTES,
      GLASS_MESH_COLOR_OFFSET_BYTES,
    );
    gl.vertexAttribPointer(
      attributes.aEmissive,
      1,
      GL_FLOAT,
      false,
      GLASS_MESH_VERTEX_STRIDE_BYTES,
      GLASS_MESH_EMISSIVE_OFFSET_BYTES,
    );
  }

  /**
   * Draws the G4 glass state machine exactly once per frame in the pinned
   * op order: depth-test ON, depth-write OFF, back-face culling ON
   * (frontFace CCW + cullFace BACK — the true winding), then the
   * premultiplied-alpha blend pair blendFunc(ONE, ONE_MINUS_SRC_ALPHA).
   */
  function applyGlassState(): void {
    gl.useProgram(compiled.program);
    gl.enable(GL_DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(GL_CULL_FACE);
    gl.cullFace(GL_CULL_FACE_BACK);
    gl.frontFace(GL_FRONT_FACE_CCW);
    gl.enable(GL_BLEND);
    gl.blendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  }

  /**
   * Draws one car's glass far -> near: the body glass mesh first, then the
   * four wheel glass meshes on the section 7.4 mount chain
   * (carModel x translate(socket) x rotY(mountYaw) x spin(model-depth
   * axis)) — the same chain the opaque wheel pass uses, minus spin-visible
   * lighting.
   *
   * @param car - Car being drawn.
   */
  function drawGlassCar(car: GlassPassCarState): void {
    translationMat4(
      NEATRACING_GLASS_TRANSLATION_MATRIX,
      car.worldX,
      0,
      car.worldZ,
    );
    rotationYMat4(NEATRACING_GLASS_YAW_MATRIX, car.yawRad);
    multiplyMat4(
      NEATRACING_GLASS_MODEL_MATRIX,
      NEATRACING_GLASS_TRANSLATION_MATRIX,
      NEATRACING_GLASS_YAW_MATRIX,
    );
    gl.bindBuffer(GL_ARRAY_BUFFER, bodyBuffer);
    pointGlassMeshAttributes();
    gl.uniformMatrix4fv(uniforms.uModel, false, NEATRACING_GLASS_MODEL_MATRIX);
    gl.drawArrays(GL_TRIANGLES, 0, bodyMesh.vertexCount);
    gl.bindBuffer(GL_ARRAY_BUFFER, wheelBuffer);
    pointGlassMeshAttributes();
    for (const mount of wheelMounts) {
      translationMat4(
        NEATRACING_GLASS_WHEEL_TRANSLATION_MATRIX,
        mount.position[0],
        mount.position[1],
        mount.position[2],
      );
      multiplyMat4(
        NEATRACING_GLASS_MATRIX_SCRATCH_A,
        NEATRACING_GLASS_MODEL_MATRIX,
        NEATRACING_GLASS_WHEEL_TRANSLATION_MATRIX,
      );
      rotationYMat4(
        NEATRACING_GLASS_WHEEL_MOUNT_YAW_MATRIX,
        mount.rotationYawRad,
      );
      multiplyMat4(
        NEATRACING_GLASS_MATRIX_SCRATCH_B,
        NEATRACING_GLASS_MATRIX_SCRATCH_A,
        NEATRACING_GLASS_WHEEL_MOUNT_YAW_MATRIX,
      );
      rotationZMat4(NEATRACING_GLASS_WHEEL_SPIN_MATRIX, car.wheelSpinRad);
      multiplyMat4(
        NEATRACING_GLASS_MATRIX_SCRATCH_A,
        NEATRACING_GLASS_MATRIX_SCRATCH_B,
        NEATRACING_GLASS_WHEEL_SPIN_MATRIX,
      );
      gl.uniformMatrix4fv(
        uniforms.uModel,
        false,
        NEATRACING_GLASS_MATRIX_SCRATCH_A,
      );
      gl.drawArrays(GL_TRIANGLES, 0, wheelMesh.vertexCount);
    }
  }

  return {
    draw(state: GlassPassDrawState): void {
      if (state.viewProjection.length !== 16) {
        throw new Error(
          `${NEATRACING_GLASS_PASS_LABEL}: viewProjection must contain exactly 16 elements, received ${String(state.viewProjection.length)}`,
        );
      }
      if (state.cameraWorld.length !== 3) {
        throw new Error(
          `${NEATRACING_GLASS_PASS_LABEL}: cameraWorld must contain exactly 3 components, received ${String(state.cameraWorld.length)}`,
        );
      }
      composeSortedGlassOrder(state);
      applyGlassState();
      const carCount = state.cars.length;
      for (let slot = 0; slot < carCount; slot += 1) {
        drawGlassCar(state.cars[NEATRACING_GLASS_SORT_ORDER_SCRATCH[slot]]);
      }
      // Restore the cull state the glass state machine turned on: leaving
      // CULL_FACE enabled leaks into the next frame's passes and silently
      // culls geometry that never sets its own cull state (the track ribbon),
      // making the track blink out after the first frame.
      gl.disable(GL_CULL_FACE);
    },
    dispose(): void {
      gl.deleteProgram(compiled.program);
      gl.deleteShader(compiled.vertexShader);
      gl.deleteShader(compiled.fragmentShader);
      gl.deleteBuffer(bodyBuffer);
      gl.deleteBuffer(wheelBuffer);
    },
  };
}
