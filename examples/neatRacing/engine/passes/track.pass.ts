/**
 * NeatRacing track slab draw pass (spec section 6.7, slice P1-ground-composition,
 * AC-017). Uploads the deterministic {@link buildTrackSlabs} ribbon as an
 * interleaved position+color VBO and draws it with depth testing enabled so
 * the track sits on the ground behind/on top of the grid plane.
 */

import type { ShaderWarmupProgram } from '../context.service';
import { buildTrackSlabs, type TrackSlabMesh } from '../../track/track.slabs';

/** Warmup/draw registry key for the track pass (AC-040). */
export const NEATRACING_TRACK_PROGRAM_KEY = 'track';

/**
 * Default deterministic track build input used by {@link createTrackPass}.
 * The seed, layout version, and size bucket are pinned so the ribbon is
 * identical on every page load and in every test run.
 */
const DEFAULT_TRACK_BUILD_INPUT = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
} as const;

/**
 * Vertex shader (GLSL ES 3.00) for the track pass: position + color in,
 * view-projected position plus interpolated color out.
 */
const NEATRACING_TRACK_VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
in vec3 aColor;
uniform mat4 uViewProjection;
out vec3 vColor;
void main() {
  gl_Position = uViewProjection * vec4(aPosition, 1.0);
  vColor = aColor;
}`;

/** Fragment shader (GLSL ES 3.00) for the track pass: flat vertex color. */
const NEATRACING_TRACK_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 fragColor;
void main() {
  fragColor = vec4(vColor, 1.0);
}`;

/** `gl.ARRAY_BUFFER` enum value (0x8892) for the track VBO. */
const GL_ARRAY_BUFFER = 0x8892;

/** `gl.STATIC_DRAW` enum value (0x88E4) for the one-time geometry upload. */
const GL_STATIC_DRAW = 0x88e4;

/** `gl.FLOAT` enum value (0x1406) for the position/color attribute pointers. */
const GL_FLOAT = 0x1406;

/** `gl.DEPTH_TEST` enum value (0x0B71) enabled during the track draw. */
const GL_DEPTH_TEST = 0x0b71;

/** `gl.CULL_FACE` enum value (0x0B44) explicitly disabled for the two-sided track. */
const GL_CULL_FACE = 0x0b44;

/** `gl.TRIANGLES` enum value (0x0004) for the triangulated track quads. */
const GL_TRIANGLES = 0x0004;

/** Floats per interleaved vertex: position (xyz) + color (rgb). */
const FLOATS_PER_VERTEX = 6;

/** Vertices emitted per slab quad when triangulated as two triangles. */
const VERTICES_PER_QUAD = 6;

/** Bytes per float for VBO size calculations. */
const BYTES_PER_FLOAT = 4;

/** Byte stride of one interleaved vertex. */
const VERTEX_STRIDE_BYTES = FLOATS_PER_VERTEX * BYTES_PER_FLOAT;

/** Byte offset of the position attribute within one vertex. */
const POSITION_OFFSET_BYTES = 0;

/** Byte offset of the color attribute within one vertex. */
const COLOR_OFFSET_BYTES = 3 * BYTES_PER_FLOAT;

/** Per-draw inputs for the track pass. */
export interface TrackPassDrawState {
  /** Column-major 16-element view-projection matrix. */
  readonly viewProjection: Float32Array;
  /** Camera world position [x, y, z]. */
  readonly cameraWorld: readonly [number, number, number];
}

/** The track pass handle: create once, draw every frame. */
export interface TrackPass {
  /** Draws the track slab geometry for the given camera state. */
  draw(state: TrackPassDrawState): void;
}

/**
 * Compiles one shader stage or throws a descriptive error carrying the stage
 * label and the driver info log.
 */
function compileTrackShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error(
      `neatRacing track pass: createShader returned null for ${label} stage`,
    );
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const infoLog = gl.getShaderInfoLog(shader) ?? 'unknown error';
    throw new Error(
      `neatRacing track pass: ${label} shader compile failed: ${infoLog}`,
    );
  }
  return shader;
}

/**
 * Converts 0-255 RGB bytes to 0..1 normalized floats for the vertex color
 * attribute.
 */
function normalizeColorBytes(
  color: readonly [number, number, number],
): readonly [number, number, number] {
  return [color[0] / 255, color[1] / 255, color[2] / 255];
}

/**
 * Builds an interleaved Float32Array of position+color vertices from the
 * deterministic track slab mesh. Each greedy-meshed quad is triangulated as
 * two triangles (six vertices) with a consistent counter-clockwise winding.
 */
function buildTrackVboData(mesh: TrackSlabMesh): Float32Array {
  const vertexCount = mesh.quads.length * VERTICES_PER_QUAD;
  const data = new Float32Array(vertexCount * FLOATS_PER_VERTEX);

  for (let quadIndex = 0; quadIndex < mesh.quads.length; quadIndex++) {
    const quad = mesh.quads[quadIndex];
    const paletteEntry = mesh.palette[quad.paletteIndex];
    if (!paletteEntry) {
      throw new Error(
        `neatRacing track pass: quad paletteIndex ${quad.paletteIndex} out of bounds`,
      );
    }

    const [corner0, corner1, corner2, corner3] = expandSlabQuadCorners(quad);
    const color = normalizeColorBytes(paletteEntry.color);
    const vertices = [
      corner0,
      corner1,
      corner2,
      corner0,
      corner2,
      corner3,
    ] as const;

    for (let vertexIndex = 0; vertexIndex < VERTICES_PER_QUAD; vertexIndex++) {
      const vertex = vertices[vertexIndex];
      const base =
        (quadIndex * VERTICES_PER_QUAD + vertexIndex) * FLOATS_PER_VERTEX;
      data[base + 0] = vertex[0];
      data[base + 1] = vertex[1];
      data[base + 2] = vertex[2];
      data[base + 3] = color[0];
      data[base + 4] = color[1];
      data[base + 5] = color[2];
    }
  }

  return data;
}

/**
 * Expands a slab quad into its four corner positions in consistent winding
 * order. The quad lies on the plane perpendicular to its normal axis, with
 * width along the first tangent axis and height along the second.
 */
function expandSlabQuadCorners(
  quad: TrackSlabMesh['quads'][number],
): readonly [
  readonly [number, number, number],
  readonly [number, number, number],
  readonly [number, number, number],
  readonly [number, number, number],
] {
  const [widthAxis, heightAxis] = resolveTangentAxes(quad.normalAxis);
  const min = quad.position;
  const widthDelta = [0, 0, 0] as [number, number, number];
  const heightDelta = [0, 0, 0] as [number, number, number];
  widthDelta[widthAxis] = quad.width;
  heightDelta[heightAxis] = quad.height;

  const corner0: readonly [number, number, number] = [min[0], min[1], min[2]];
  const corner1: readonly [number, number, number] = [
    min[0] + widthDelta[0],
    min[1] + widthDelta[1],
    min[2] + widthDelta[2],
  ];
  const corner2: readonly [number, number, number] = [
    min[0] + widthDelta[0] + heightDelta[0],
    min[1] + widthDelta[1] + heightDelta[1],
    min[2] + widthDelta[2] + heightDelta[2],
  ];
  const corner3: readonly [number, number, number] = [
    min[0] + heightDelta[0],
    min[1] + heightDelta[1],
    min[2] + heightDelta[2],
  ];

  return [corner0, corner1, corner2, corner3];
}

/**
 * Resolves the width and height tangent axes for a given face normal axis,
 * mirroring the {@link FACE_TANGENT_AXES} mapping in the shared greedy mesher.
 */
function resolveTangentAxes(normalAxis: number): readonly [number, number] {
  if (normalAxis === 0) {
    return [1, 2];
  }
  if (normalAxis === 1) {
    return [0, 2];
  }
  return [0, 1];
}

/**
 * Creates the track slab draw pass: builds the default deterministic ribbon,
 * compiles and links the position+color program, uploads the interleaved VBO,
 * and returns a handle whose `draw(state)` sets the view-projection uniform,
 * enables depth testing, and issues one `gl.drawArrays(GL_TRIANGLES, 0, count)`.
 *
 * @param gl WebGL2 rendering context to bind the pass to.
 * @returns The track pass handle.
 */
export function createTrackPass(gl: WebGL2RenderingContext): TrackPass {
  const mesh = buildTrackSlabs(DEFAULT_TRACK_BUILD_INPUT);
  const vboData = buildTrackVboData(mesh);
  const vertexCount = vboData.length / FLOATS_PER_VERTEX;

  const vertexShader = compileTrackShader(
    gl,
    gl.VERTEX_SHADER,
    NEATRACING_TRACK_VERTEX_SHADER,
    'vertex',
  );
  const fragmentShader = compileTrackShader(
    gl,
    gl.FRAGMENT_SHADER,
    NEATRACING_TRACK_FRAGMENT_SHADER,
    'fragment',
  );

  const program = gl.createProgram();
  if (!program) {
    throw new Error('neatRacing track pass: createProgram returned null');
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const infoLog = gl.getProgramInfoLog(program) ?? 'unknown error';
    throw new Error(`neatRacing track pass: program link failed: ${infoLog}`);
  }

  const vbo = gl.createBuffer();
  if (!vbo) {
    throw new Error('neatRacing track pass: createBuffer returned null');
  }
  gl.bindBuffer(GL_ARRAY_BUFFER, vbo);
  gl.bufferData(GL_ARRAY_BUFFER, vboData, GL_STATIC_DRAW);

  const positionAttrib = gl.getAttribLocation(program, 'aPosition');
  const colorAttrib = gl.getAttribLocation(program, 'aColor');
  if (positionAttrib < 0) {
    throw new Error(
      'neatRacing track pass: aPosition attribute not found in program',
    );
  }
  if (colorAttrib < 0) {
    throw new Error(
      'neatRacing track pass: aColor attribute not found in program',
    );
  }

  const viewProjectionLocation = gl.getUniformLocation(
    program,
    'uViewProjection',
  );
  if (!viewProjectionLocation) {
    throw new Error(
      'neatRacing track pass: uViewProjection uniform not found in program',
    );
  }

  return {
    draw(state: TrackPassDrawState): void {
      if (state.viewProjection.length !== 16) {
        throw new Error(
          'neatRacing track pass draw: viewProjection must be a 16-element column-major matrix',
        );
      }

      gl.useProgram(program);
      gl.bindBuffer(GL_ARRAY_BUFFER, vbo);
      gl.enableVertexAttribArray(positionAttrib);
      gl.vertexAttribPointer(
        positionAttrib,
        3,
        GL_FLOAT,
        false,
        VERTEX_STRIDE_BYTES,
        POSITION_OFFSET_BYTES,
      );
      gl.enableVertexAttribArray(colorAttrib);
      gl.vertexAttribPointer(
        colorAttrib,
        3,
        GL_FLOAT,
        false,
        VERTEX_STRIDE_BYTES,
        COLOR_OFFSET_BYTES,
      );

      gl.enable(GL_DEPTH_TEST);
      gl.depthMask(true);
      // Two-sided ribbon: the track is flat ground geometry viewed from above,
      // so face culling must stay OFF. Set it explicitly — inheriting a leaked
      // CULL_FACE=ON from a previously drawn pass (glass) fully culls every
      // track quad and the ribbon blinks out of existence (AC-017).
      gl.disable(GL_CULL_FACE);
      gl.uniformMatrix4fv(viewProjectionLocation, false, state.viewProjection);
      gl.drawArrays(GL_TRIANGLES, 0, vertexCount);
    },
  };
}

/** Column-major identity matrix reused for the zero-pose warmup draw. */
const IDENTITY_VIEW_PROJECTION = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
];

/** Zero-pose warmup draw state: identity projection, camera above the origin. */
const ZERO_POSE_DRAW_STATE: TrackPassDrawState = {
  viewProjection: new Float32Array(IDENTITY_VIEW_PROJECTION),
  cameraWorld: [0, 10, 20],
};

/**
 * Exposes the track pass as a shader warmup program (AC-040 seam).
 *
 * The returned entry keeps compilation lazy: `compileAndLink` builds the
 * pass (compiling and linking the program), and `drawOffscreen` issues the
 * zero-pose track draw through the shared pass path so the driver warms the
 * exact render pipeline used on screen. The context argument is accepted for
 * interface parity; the pass is bound to the context supplied at construction.
 *
 * @param context - WebGL2 context the pass resources will be built against.
 * @returns Warmup entry keyed to the track registry slot (fourth warmed key).
 */
export function toTrackWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram {
  let pass: TrackPass | null = null;
  const ensurePass = (): TrackPass => {
    if (!pass) {
      pass = createTrackPass(context);
    }
    return pass;
  };

  return {
    key: NEATRACING_TRACK_PROGRAM_KEY,
    compileAndLink: (): void => {
      ensurePass();
    },
    drawOffscreen: (_context: WebGL2RenderingContext): void => {
      void _context;
      ensurePass().draw(ZERO_POSE_DRAW_STATE);
    },
  };
}
