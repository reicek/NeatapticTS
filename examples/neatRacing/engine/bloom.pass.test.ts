/**
 * Red-first contract tests for slice P1-S4-E (spec section 6.5 G4 glass +
 * section 6.6 G8 bloom, AC-013 / AC-016 / AC-035 / AC-040).
 *
 * Locks BOTH passes of the slice through pure functions and a recording
 * mock WebGL2 context:
 *
 * 1. The glass pass (G4): the bins-independent glass mesh derivation with
 *    the AC-013 material-table filter (black never shines), TRUE
 *    model-space winding per glass face (cross-product contract), the
 *    deterministic z-asc -> y-asc -> x-asc emission, the exact G4 draw
 *    state machine (depth-test ON, depth-write OFF, back-face culling ON
 *    with `frontFace(GL_CCW)` + `cullFace(GL_BACK)`, premultiplied-alpha
 *    blend `blendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)`), far -> near
 *    camera-anchored car sorting, malformed input rejection,
 *    zero-pose warmup draws, and the 'glass' warmup key parity.
 *
 * 2. The bloom pass (G8): the half-res 640x360 render targets, the AC-016
 *    bright math (Rec709 luminance + thresholded/clamped bright factor),
 *    the 13-tap symmetric gaussian separable blur table, the four-stage
 *    draw pipeline (`threshold -> blur A -> blur B -> additive composite`
 *    with `blendFunc(GL_ONE, GL_ONE)` on the composite stage only), the
 *    AC-035 explicit off-switch (zero GL ops when disabled), dispose
 *    counts, and the keys[6..8] warmup seam with the 1x1 white placeholder.
 *
 * The Jest project runs in a plain Node environment, so no GPU or DOM is
 * required (pulse.pass.test.ts harness parity).
 */
import { describe, expect, it } from '@jest/globals';
import { NEATRACING_WARMED_PROGRAM_KEYS } from './context.service';
import { resolveVoxelWheelSockets } from '../voxel/wheel.mount.utils';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';
import {
  VOXEL_EMPTY_PALETTE_ENTRY,
  type VoxelPaletteEntry,
} from '../voxel/model.schema';
import type { DecodedVoxelModel } from '../voxel/model.decode.utils';
import {
  createGlassPass,
  expandGlassModelToGlassMesh,
  toGlassWarmupProgram,
  GLASS_MESH_FLOATS_PER_VERTEX,
  GLASS_MESH_VERTEX_STRIDE_BYTES,
  NEATRACING_GLASS_FACES_PER_CELL,
  NEATRACING_GLASS_FRAGMENT_SHADER_SOURCE,
  NEATRACING_GLASS_PROGRAM_KEY,
  NEATRACING_GLASS_TINT_ALPHA,
  NEATRACING_GLASS_VERTEX_SHADER_SOURCE,
  NEATRACING_GLASS_VERTICES_PER_FACE,
  type GlassPassCarState,
  type GlassPassDrawState,
  type GlassPassModels,
} from './passes/glass.pass';
import {
  computeBloomBrightFactor,
  computeBloomBlurTaps,
  computeBloomLuminance,
  createBloomPass,
  toBloomWarmupPrograms,
  NEATRACING_BLOOM_ADDITIVE_FRAGMENT_SHADER_SOURCE,
  NEATRACING_BLOOM_ADDITIVE_PROGRAM_KEY,
  NEATRACING_BLOOM_BLUR_FRAGMENT_SHADER_SOURCE,
  NEATRACING_BLOOM_BLUR_PROGRAM_KEY,
  NEATRACING_BLOOM_BLUR_SIGMA_TEXELS,
  NEATRACING_BLOOM_BRIGHT_THRESHOLD,
  NEATRACING_BLOOM_COMPOSITE_INTENSITY,
  NEATRACING_BLOOM_LUMA_WEIGHTS,
  NEATRACING_BLOOM_MAX_TAPS,
  NEATRACING_BLOOM_TARGET_HEIGHT,
  NEATRACING_BLOOM_TARGET_WIDTH,
  NEATRACING_BLOOM_TAP_COUNT,
  NEATRACING_BLOOM_THRESHOLD_FRAGMENT_SHADER_SOURCE,
  NEATRACING_BLOOM_THRESHOLD_PROGRAM_KEY,
  NEATRACING_BLOOM_VERTEX_SHADER_SOURCE,
  type BloomPassDrawFrame,
} from './passes/bloom.pass';

/** `gl.BLEND` enum value (0x0BE2) asserted on both state machines. */
const MOCK_GL_BLEND = 0x0be2;

/** `gl.CULL_FACE` enum value (0x0B44) asserted on both state machines. */
const MOCK_GL_CULL_FACE = 0x0b44;

/** `gl.BACK` cull-face enum value (0x0405) asserted on the G4 glass state. */
const MOCK_GL_CULL_FACE_BACK = 0x0405;

/** `gl.CCW` front-face enum value (0x0901) asserted on the G4 glass state. */
const MOCK_GL_FRONT_FACE_CCW = 0x0901;

/** `gl.DEPTH_TEST` enum value (0x0B71) asserted on the G4 glass state. */
const MOCK_GL_DEPTH_TEST = 0x0b71;

/** `gl.ONE` blend factor (1) shared by the glass and blend pairs. */
const MOCK_GL_ONE = 1;

/** `gl.ONE_MINUS_SRC_ALPHA` blend factor (0x0303) of the glass pass. */
const MOCK_GL_ONE_MINUS_SRC_ALPHA = 0x0303;

/** `gl.TRIANGLES` draw mode (4) asserted on glass meshes. */
const MOCK_GL_TRIANGLES = 4;

/** `gl.TRIANGLE_STRIP` draw mode (5) asserted on the bloom quad stages. */
const MOCK_GL_TRIANGLE_STRIP = 5;

/** `gl.TEXTURE_2D` enum value (0x0DE1) asserted on bloom texture binds. */
const MOCK_GL_TEXTURE_2D = 0x0de1;

/** `gl.TEXTURE0` sampler unit (0x84C0) asserted on bloom binds. */
const MOCK_GL_TEXTURE0 = 0x84c0;

/** `gl.TEXTURE1` sampler unit (0x84C1) asserted on the composite bloom texture. */
const MOCK_GL_TEXTURE1 = 0x84c1;

/** `gl.FRAMEBUFFER_COMPLETE` status (0x8CD5) returned by the default mock. */
const MOCK_GL_FRAMEBUFFER_COMPLETE = 0x8cd5;

/** Column-major identity view-projection used by the mock draw frames. */
const IDENTITY_VIEW_PROJECTION_FIXTURE = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
]);

/** Shared test palette: 0 EMPTY, 1 fully-transparent glow glass, 2 black, 3 solid. */
const TEST_PALETTE_FIXTURE: readonly VoxelPaletteEntry[] = [
  [...VOXEL_EMPTY_PALETTE_ENTRY] as VoxelPaletteEntry,
  [190, 225, 255, 0, 160] as VoxelPaletteEntry,
  [0, 0, 0, 255, 220] as VoxelPaletteEntry,
  [200, 80, 40, 255, 0] as VoxelPaletteEntry,
];

/** Palette index of the fully transparent glow glass entry. */
const TEST_PALETTE_GLASS_INDEX = 1;

/** Palette index of the LD-23 black entry (opaque, glow byte forced dead). */
const TEST_PALETTE_BLACK_INDEX = 2;

/** Palette index of the opaque colored solid entry. */
const TEST_PALETTE_SOLID_INDEX = 3;

/** Placeholder scene texture handle the bloom mock frames draw from. */
const SCENE_TEXTURE_FIXTURE = {} as WebGLTexture;

/** Expected model-space outward normals in glass emission face order. */
const EXPECTED_MODEL_SPACE_FACE_NORMALS: readonly (readonly [
  number,
  number,
  number,
])[] = [
  [-1, 0, 0],
  [1, 0, 0],
  [0, -1, 0],
  [0, 1, 0],
  [0, 0, -1],
  [0, 0, 1],
];

/**
 * Builds a decoded voxel model fixture over the shared 4-entry palette:
 * each slice array holds `dims[0] * dims[1]` palette indices on the
 * `x + y * dims[0]` layout the glass pass scans.
 *
 * @param name - Model name echoed into the fixture.
 * @param dims - Grid dims `[width, height, depth]`.
 * @param slices - One row-major palette-index array per depth slice.
 * @returns The decoded model fixture.
 */
function buildTestVoxelModel(
  name: string,
  dims: readonly [number, number, number],
  slices: readonly number[][],
): DecodedVoxelModel {
  return {
    name,
    dims,
    symmetric: false,
    palette: TEST_PALETTE_FIXTURE,
    slices: slices.map((slice) => new Uint8Array(slice)),
  };
}

/**
 * Builds an all-glass body + wheel model pair of the requested dims: every
 * cell references the transparent glow palette entry.
 *
 * @param name - Model name echoed into the fixture.
 * @param dims - Grid dims shared by body and wheel models.
 * @returns The body + wheel model pair for {@link createGlassPass}.
 */
function buildGlassTestModels(
  name: string,
  dims: readonly [number, number, number],
): GlassPassModels {
  const cellCount = dims[0] * dims[1];
  const glassSlice = new Array<number>(cellCount).fill(
    TEST_PALETTE_GLASS_INDEX,
  );
  const model = buildTestVoxelModel(
    name,
    dims,
    Array.from({ length: dims[2] }, () => glassSlice),
  );
  return { body: model, wheel: model };
}

/**
 * Builds a glass draw frame with the identity projection and the camera
 * anchored at the world origin.
 *
 * @param cars - Cars to draw (any order; the pass sorts far -> near).
 * @returns The glass draw state fixture.
 */
function buildGlassDrawState(
  cars: readonly GlassPassCarState[],
): GlassPassDrawState {
  return {
    viewProjection: new Float32Array(IDENTITY_VIEW_PROJECTION_FIXTURE),
    cameraWorld: [0, 0, 0],
    cars,
  };
}

/**
 * Builds a bloom draw frame against the shared placeholder scene texture
 * with the fixed 720p display viewport.
 *
 * @param enabled - Whether the frame renders (AC-035 off-switch probe).
 * @returns The bloom draw frame fixture.
 */
function buildBloomDrawFrame(enabled: boolean): BloomPassDrawFrame {
  return {
    sceneTexture: SCENE_TEXTURE_FIXTURE,
    targetWidth: 1280,
    targetHeight: 720,
    enabled,
  };
}

/** One recorded frame-state GL op from the glass or bloom draw. */
interface MockGlStateOp {
  readonly op:
    'enable' | 'disable' | 'depthMask' | 'blendFunc' | 'cullFace' | 'frontFace';
  readonly args: readonly (number | boolean)[];
}

/** One recorded uniform1i feed keyed by uniform name. */
interface MockUniform1iFeed {
  readonly name: string;
  readonly value: number;
}

/** One recorded uniform2f feed keyed by uniform name. */
interface MockUniform2fFeed {
  readonly name: string;
  readonly x: number;
  readonly y: number;
}

/** One recorded uniformMatrix4fv feed keyed by uniform name. */
interface MockUniformMatrix4fvFeed {
  readonly name: string;
  readonly values: readonly number[];
}

/** One recorded texImage2D upload with its extent and source data. */
interface MockTexImage2DRecord {
  readonly target: number;
  readonly width: number;
  readonly height: number;
  readonly source: unknown;
}

/** One recorded drawArrays call with mode/first/count. */
interface MockDrawArraysRecord {
  readonly mode: number;
  readonly first: number;
  readonly count: number;
}

/** One recorded bindFramebuffer call with its framebuffer handle. */
interface MockBindFramebufferRecord {
  readonly target: number;
  readonly framebuffer: unknown;
}

/** One recorded bindTexture call with its target and texture handle. */
interface MockBindTextureRecord {
  readonly target: number;
  readonly texture: unknown;
}

/** Mutable recording mock WebGL2 harness (shared by glass + bloom tests). */
interface MockGlHarness {
  readonly gl: WebGL2RenderingContext;
  readonly calls: Record<
    | 'drawArrays'
    | 'depthMask'
    | 'disable'
    | 'enable'
    | 'blendFunc'
    | 'cullFace'
    | 'frontFace'
    | 'useProgram'
    | 'bindBuffer'
    | 'createProgram'
    | 'createShader'
    | 'createBuffer'
    | 'createTexture'
    | 'createFramebuffer'
    | 'deleteProgram'
    | 'deleteShader'
    | 'deleteBuffer'
    | 'deleteTexture'
    | 'deleteFramebuffer'
    | 'bindFramebuffer'
    | 'activeTexture'
    | 'bindTexture'
    | 'texParameteri'
    | 'texImage2D'
    | 'framebufferTexture2D'
    | 'checkFramebufferStatus'
    | 'viewport'
    | 'uniform1i'
    | 'uniform2f'
    | 'vertexAttribPointer'
    | 'enableVertexAttribArray',
    number
  >;
  /** Frame-state ops in call order (G4 + G8 state machine pins). */
  readonly stateOps: ReadonlyArray<MockGlStateOp>;
  /** Uniform1i feeds keyed by uniform name in call order. */
  readonly uniform1iFeeds: ReadonlyArray<MockUniform1iFeed>;
  /** Uniform2f feeds keyed by uniform name in call order. */
  readonly uniform2fFeeds: ReadonlyArray<MockUniform2fFeed>;
  /** UniformMatrix4fv feeds keyed by uniform name in call order. */
  readonly uniformMatrix4fvFeeds: ReadonlyArray<MockUniformMatrix4fvFeed>;
  /** TexImage2D uploads in call order (target dims + source). */
  readonly texImage2DRecords: ReadonlyArray<MockTexImage2DRecord>;
  /** Viewport feeds in call order (pipeline + composite pins). */
  readonly viewportFeeds: ReadonlyArray<MockViewportFeed>;
  /** DrawArrays records in call order (mode/first/count pins). */
  readonly drawArraysRecords: ReadonlyArray<MockDrawArraysRecord>;
  /** BindFramebuffer targets in call order (stage order pin). */
  readonly bindFramebufferRecords: ReadonlyArray<MockBindFramebufferRecord>;
  /** BindTexture handles in call order (sampler-binding pin). */
  readonly bindTextureRecords: ReadonlyArray<MockBindTextureRecord>;
  /** Active texture units in call order. */
  readonly activeTextureRecords: ReadonlyArray<number>;
  /** Created texture handles in creation order. */
  readonly createdTextures: readonly object[];
  /** Created framebuffer handles in creation order. */
  readonly createdFramebuffers: readonly object[];
}

/** One recorded viewport call. */
interface MockViewportFeed {
  readonly width: number;
  readonly height: number;
}

/**
 * Creates a recording mock WebGL2 context covering every GL entry point the
 * glass and bloom passes touch, mirroring the pulse pass harness pattern.
 *
 * @param options - Overrides; `checkFramebufferStatusResult` returns a
 *   failing status for the incomplete-framebuffer rejection test.
 * @returns The recording harness.
 */
function createMockGlHarness(
  options: { readonly checkFramebufferStatusResult?: number } = {},
): MockGlHarness {
  const calls: MockGlHarness['calls'] = {
    drawArrays: 0,
    depthMask: 0,
    disable: 0,
    enable: 0,
    blendFunc: 0,
    cullFace: 0,
    frontFace: 0,
    useProgram: 0,
    bindBuffer: 0,
    createProgram: 0,
    createShader: 0,
    createBuffer: 0,
    createTexture: 0,
    createFramebuffer: 0,
    deleteProgram: 0,
    deleteShader: 0,
    deleteBuffer: 0,
    deleteTexture: 0,
    deleteFramebuffer: 0,
    bindFramebuffer: 0,
    activeTexture: 0,
    bindTexture: 0,
    texParameteri: 0,
    texImage2D: 0,
    framebufferTexture2D: 0,
    checkFramebufferStatus: 0,
    viewport: 0,
    uniform1i: 0,
    uniform2f: 0,
    vertexAttribPointer: 0,
    enableVertexAttribArray: 0,
  };
  const stateOps: MockGlStateOp[] = [];
  const uniform1iFeeds: MockUniform1iFeed[] = [];
  const uniform2fFeeds: MockUniform2fFeed[] = [];
  const uniformMatrix4fvFeeds: MockUniformMatrix4fvFeed[] = [];
  const texImage2DRecords: MockTexImage2DRecord[] = [];
  const viewportFeeds: MockViewportFeed[] = [];
  const drawArraysRecords: MockDrawArraysRecord[] = [];
  const bindFramebufferRecords: MockBindFramebufferRecord[] = [];
  const bindTextureRecords: MockBindTextureRecord[] = [];
  const activeTextureRecords: number[] = [];
  const createdTextures: object[] = [];
  const createdFramebuffers: object[] = [];
  const checkFramebufferStatusResult =
    options.checkFramebufferStatusResult ?? MOCK_GL_FRAMEBUFFER_COMPLETE;
  const gl = {
    createShader: (): object => {
      calls.createShader += 1;
      return {};
    },
    shaderSource: (): void => undefined,
    compileShader: (): void => undefined,
    getShaderParameter: (): boolean => true,
    getShaderInfoLog: (): string | null => null,
    createProgram: (): object => {
      calls.createProgram += 1;
      return {};
    },
    attachShader: (): void => undefined,
    linkProgram: (): void => undefined,
    getProgramParameter: (): boolean => true,
    getProgramInfoLog: (): string | null => null,
    getAttribLocation: (): number => 0,
    getUniformLocation: (_program: unknown, name: string): object => ({ name }),
    useProgram: (): void => {
      calls.useProgram += 1;
    },
    enable: (target: number): void => {
      calls.enable += 1;
      stateOps.push({ op: 'enable', args: [target] });
    },
    disable: (target: number): void => {
      calls.disable += 1;
      stateOps.push({ op: 'disable', args: [target] });
    },
    depthMask: (flag: boolean): void => {
      calls.depthMask += 1;
      stateOps.push({ op: 'depthMask', args: [flag] });
    },
    cullFace: (mode: number): void => {
      calls.cullFace += 1;
      stateOps.push({ op: 'cullFace', args: [mode] });
    },
    frontFace: (mode: number): void => {
      calls.frontFace += 1;
      stateOps.push({ op: 'frontFace', args: [mode] });
    },
    blendFunc: (sfactor: number, dfactor: number): void => {
      calls.blendFunc += 1;
      stateOps.push({ op: 'blendFunc', args: [sfactor, dfactor] });
    },
    createBuffer: (): object => {
      calls.createBuffer += 1;
      return {};
    },
    bindBuffer: (): void => {
      calls.bindBuffer += 1;
    },
    bufferData: (): void => undefined,
    vertexAttribPointer: (): void => {
      calls.vertexAttribPointer += 1;
    },
    enableVertexAttribArray: (): void => {
      calls.enableVertexAttribArray += 1;
    },
    uniformMatrix4fv: (
      location: { name: string },
      _transpose: boolean,
      value: Float32Array,
    ): void => {
      uniformMatrix4fvFeeds.push({
        name: location.name,
        values: Array.from(value),
      });
    },
    uniform1i: (location: { name: string }, value: number): void => {
      calls.uniform1i += 1;
      uniform1iFeeds.push({ name: location.name, value });
    },
    uniform2f: (location: { name: string }, x: number, y: number): void => {
      calls.uniform2f += 1;
      uniform2fFeeds.push({ name: location.name, x, y });
    },
    createTexture: (): object => {
      calls.createTexture += 1;
      const texture = {};
      createdTextures.push(texture);
      return texture;
    },
    bindTexture: (target: number, texture: unknown): void => {
      calls.bindTexture += 1;
      bindTextureRecords.push({ target, texture });
    },
    activeTexture: (unit: number): void => {
      calls.activeTexture += 1;
      activeTextureRecords.push(unit);
    },
    texParameteri: (): void => {
      calls.texParameteri += 1;
    },
    texImage2D: (
      target: number,
      _level: number,
      _internalFormat: number,
      width: number,
      height: number,
      _border: number,
      _sourceFormat: number,
      _sourceType: number,
      source: unknown,
    ): void => {
      calls.texImage2D += 1;
      texImage2DRecords.push({ target, width, height, source });
    },
    createFramebuffer: (): object => {
      calls.createFramebuffer += 1;
      const framebuffer = {};
      createdFramebuffers.push(framebuffer);
      return framebuffer;
    },
    bindFramebuffer: (target: number, framebuffer: unknown): void => {
      calls.bindFramebuffer += 1;
      bindFramebufferRecords.push({ target, framebuffer });
    },
    framebufferTexture2D: (): void => {
      calls.framebufferTexture2D += 1;
    },
    checkFramebufferStatus: (): number => {
      calls.checkFramebufferStatus += 1;
      return checkFramebufferStatusResult;
    },
    viewport: (x: number, y: number, width: number, height: number): void => {
      calls.viewport += 1;
      void x;
      void y;
      viewportFeeds.push({ width, height });
    },
    drawArrays: (mode: number, first: number, count: number): void => {
      calls.drawArrays += 1;
      drawArraysRecords.push({ mode, first, count });
    },
    deleteProgram: (): void => {
      calls.deleteProgram += 1;
    },
    deleteShader: (): void => {
      calls.deleteShader += 1;
    },
    deleteBuffer: (): void => {
      calls.deleteBuffer += 1;
    },
    deleteTexture: (): void => {
      calls.deleteTexture += 1;
    },
    deleteFramebuffer: (): void => {
      calls.deleteFramebuffer += 1;
    },
  };
  return {
    gl: gl as unknown as WebGL2RenderingContext,
    calls,
    stateOps,
    uniform1iFeeds,
    uniform2fFeeds,
    uniformMatrix4fvFeeds,
    texImage2DRecords,
    viewportFeeds,
    drawArraysRecords,
    bindFramebufferRecords,
    bindTextureRecords,
    activeTextureRecords,
    createdTextures,
    createdFramebuffers,
  };
}

/**
 * Counts every observable mock GL op (counter, feed array, or record array
 * entry) so "zero GL ops" assertions cover the whole harness surface.
 *
 * @param harness - Recording harness to measure.
 * @returns The total observable op count.
 */
function countMockGlOps(harness: MockGlHarness): number {
  const callTotal = Object.values(harness.calls).reduce(
    (total, count) => total + count,
    0,
  );
  return (
    callTotal +
    harness.stateOps.length +
    harness.uniform1iFeeds.length +
    harness.uniform2fFeeds.length +
    harness.uniformMatrix4fvFeeds.length +
    harness.texImage2DRecords.length +
    harness.viewportFeeds.length +
    harness.drawArraysRecords.length +
    harness.bindFramebufferRecords.length +
    harness.bindTextureRecords.length +
    harness.activeTextureRecords.length
  );
}

describe('neatRacing glass pass constants (G4 / AC-013)', () => {
  it('registers under the glass warmup key from the program registry', () => {
    expect(NEATRACING_GLASS_PROGRAM_KEY).toBe('glass');
    expect(NEATRACING_GLASS_PROGRAM_KEY).toBe(
      NEATRACING_WARMED_PROGRAM_KEYS[5],
    );
    expect(NEATRACING_WARMED_PROGRAM_KEYS).toContain(
      NEATRACING_GLASS_PROGRAM_KEY,
    );
  });

  it('pins the interleaved glass vertex stride math', () => {
    expect(GLASS_MESH_FLOATS_PER_VERTEX).toBe(7);
    expect(GLASS_MESH_VERTEX_STRIDE_BYTES).toBe(28);
  });

  it('pins the six true-winding faces and six vertices per face', () => {
    expect(NEATRACING_GLASS_FACES_PER_CELL).toBe(6);
    expect(NEATRACING_GLASS_VERTICES_PER_FACE).toBe(6);
  });

  it('pins the premultiplied tint alpha at 0.18', () => {
    expect(NEATRACING_GLASS_TINT_ALPHA).toBe(0.18);
  });
});

describe('glass mesh expansion (AC-013)', () => {
  it('contributes zero geometry for a glow-carrying black cell (black never shines)', () => {
    const model = buildTestVoxelModel(
      'black only',
      [1, 1, 1],
      [[TEST_PALETTE_BLACK_INDEX]],
    );
    const mesh = expandGlassModelToGlassMesh(model);
    expect(mesh.glassCellCount).toBe(0);
    expect(mesh.vertexCount).toBe(0);
    expect(mesh.vertexData.length).toBe(0);
  });

  it('expands one glass cell into six full faces on the 7-float stride', () => {
    const model = buildTestVoxelModel(
      'one glass cell',
      [1, 1, 1],
      [[TEST_PALETTE_GLASS_INDEX]],
    );
    const mesh = expandGlassModelToGlassMesh(model);
    expect(mesh.glassCellCount).toBe(1);
    expect(mesh.vertexCount).toBe(36);
    expect(mesh.vertexData.length).toBe(252);
  });

  it('filters solid and black cells out of a mixed grid', () => {
    const model = buildTestVoxelModel(
      'mixed grid',
      [2, 2, 1],
      [
        [
          TEST_PALETTE_GLASS_INDEX,
          TEST_PALETTE_SOLID_INDEX,
          TEST_PALETTE_BLACK_INDEX,
          TEST_PALETTE_GLASS_INDEX,
        ],
      ],
    );
    const mesh = expandGlassModelToGlassMesh(model);
    expect(mesh.glassCellCount).toBe(2);
    expect(mesh.vertexCount).toBe(72);
  });

  it('emits the first vertex at the face-0 model-space corner (ground anchor)', () => {
    const model = buildTestVoxelModel(
      'corner probe',
      [1, 1, 1],
      [[TEST_PALETTE_GLASS_INDEX]],
    );
    const { vertexData } = expandGlassModelToGlassMesh(model);
    const voxelSize = NEATRACING_VOXEL_WORLD_SIZE;
    expect(vertexData[0]).toBe(-0.5 * voxelSize);
    expect(vertexData[1]).toBe(0);
    expect(vertexData[2]).toBe(-0.5 * voxelSize);
  });

  it('normalizes the palette tint bytes per glass vertex to f32', () => {
    const model = buildTestVoxelModel(
      'tint probe',
      [1, 1, 1],
      [[TEST_PALETTE_GLASS_INDEX]],
    );
    const { vertexData } = expandGlassModelToGlassMesh(model);
    expect(vertexData[3]).toBeCloseTo(190 / 255, 6);
    expect(vertexData[4]).toBeCloseTo(225 / 255, 6);
    expect(vertexData[5]).toBeCloseTo(255 / 255, 6);
  });

  it('carries the palette glow byte as the per-vertex emissive', () => {
    const model = buildTestVoxelModel(
      'glow probe',
      [1, 1, 1],
      [[TEST_PALETTE_GLASS_INDEX]],
    );
    const { vertexData } = expandGlassModelToGlassMesh(model);
    expect(vertexData[6]).toBeCloseTo(160 / 255, 6);
  });

  it('scans cells z-asc -> y-asc -> x-asc with the fixed face order', () => {
    const model = buildTestVoxelModel(
      'scan order',
      [2, 2, 1],
      [[0, TEST_PALETTE_GLASS_INDEX, 0, 0]],
    );
    const { vertexData } = expandGlassModelToGlassMesh(model);
    const voxelSize = NEATRACING_VOXEL_WORLD_SIZE;
    expect(vertexData[0]).toBe(0);
    expect(vertexData[1]).toBe(0);
    expect(vertexData[2]).toBe(-0.5 * voxelSize);
  });

  it('anchors the center variant so the tire bottom lands at the negative hub height', () => {
    const glassColumn = new Array<number>(3).fill(TEST_PALETTE_GLASS_INDEX);
    const model = buildTestVoxelModel(
      'center anchor',
      [1, 3, 1],
      [glassColumn],
    );
    const voxelSize = NEATRACING_VOXEL_WORLD_SIZE;
    const floatsPerCell =
      NEATRACING_GLASS_FACES_PER_CELL *
      NEATRACING_GLASS_VERTICES_PER_FACE *
      GLASS_MESH_FLOATS_PER_VERTEX;
    const { vertexData } = expandGlassModelToGlassMesh(model, 'center');
    // Cell y=0, face 2 (-y plane, gridY = 0): modelY = (0 - 1.5) * voxelSize.
    const bottomFaceBase =
      2 * NEATRACING_GLASS_VERTICES_PER_FACE * GLASS_MESH_FLOATS_PER_VERTEX;
    expect(vertexData[bottomFaceBase + 1]).toBe(-1.5 * voxelSize);
    // Cell y=2, face 3 (+y plane, gridY = 3): modelY = (3 - 1.5) * voxelSize.
    const topFaceBase =
      2 * floatsPerCell +
      3 * NEATRACING_GLASS_VERTICES_PER_FACE * GLASS_MESH_FLOATS_PER_VERTEX;
    expect(vertexData[topFaceBase + 1]).toBe(1.5 * voxelSize);
    // The ground anchor keeps row 0 at modelY = 0 on the same model.
    const grounded = expandGlassModelToGlassMesh(model);
    expect(grounded.vertexData[bottomFaceBase + 1]).toBe(0);
  });

  it('emits byte-stable vertex data across repeated expansions', () => {
    const model = buildTestVoxelModel(
      'determinism',
      [2, 1, 1],
      [[TEST_PALETTE_GLASS_INDEX, TEST_PALETTE_SOLID_INDEX]],
    );
    const first = expandGlassModelToGlassMesh(model);
    const second = expandGlassModelToGlassMesh(model);
    expect(first.vertexData).toEqual(second.vertexData);
  });

  it('winds every glass face outward in model space (natural lattice-CCW, Y up)', () => {
    const model = buildTestVoxelModel(
      'winding',
      [1, 1, 1],
      [[TEST_PALETTE_GLASS_INDEX]],
    );
    const { vertexData } = expandGlassModelToGlassMesh(model);
    const voxelSize = NEATRACING_VOXEL_WORLD_SIZE;
    const squaredVoxelSize = voxelSize * voxelSize;
    let windingMismatches = 0;
    for (let face = 0; face < NEATRACING_GLASS_FACES_PER_CELL; face += 1) {
      const read = (vertexIndex: number, component: number): number =>
        vertexData[
          face *
            NEATRACING_GLASS_VERTICES_PER_FACE *
            GLASS_MESH_FLOATS_PER_VERTEX +
            vertexIndex * GLASS_MESH_FLOATS_PER_VERTEX +
            component
        ];
      const edgeA = [
        read(1, 0) - read(0, 0),
        read(1, 1) - read(0, 1),
        read(1, 2) - read(0, 2),
      ];
      const edgeB = [
        read(2, 0) - read(0, 0),
        read(2, 1) - read(0, 1),
        read(2, 2) - read(0, 2),
      ];
      const cross = [
        edgeA[1] * edgeB[2] - edgeA[2] * edgeB[1],
        edgeA[2] * edgeB[0] - edgeA[0] * edgeB[2],
        edgeA[0] * edgeB[1] - edgeA[1] * edgeB[0],
      ];
      const expected = EXPECTED_MODEL_SPACE_FACE_NORMALS[face].map(
        (component) => component * squaredVoxelSize,
      );
      const withinTolerance = cross.every(
        (component, index) => Math.abs(component - expected[index]) <= 1e-12,
      );
      if (!withinTolerance) {
        windingMismatches += 1;
      }
    }
    expect(windingMismatches).toBe(0);
  });

  it('keeps the GLSL ES 3.00 sources and interpolated tint alpha pinned', () => {
    expect(
      NEATRACING_GLASS_VERTEX_SHADER_SOURCE.startsWith('#version 300 es'),
    ).toBe(true);
    expect(
      NEATRACING_GLASS_FRAGMENT_SHADER_SOURCE.startsWith('#version 300 es'),
    ).toBe(true);
    expect(NEATRACING_GLASS_FRAGMENT_SHADER_SOURCE).toContain(
      `const float kTintAlpha = ${NEATRACING_GLASS_TINT_ALPHA};`,
    );
  });
});

describe('glass draw state machine (G4 / AC-013)', () => {
  it('applies the exact G4 state in pinned op order and restores cull after the draw', () => {
    const harness = createMockGlHarness();
    const pass = createGlassPass(
      harness.gl,
      buildGlassTestModels('state', [1, 1, 1]),
    );
    pass.draw(buildGlassDrawState([]));
    expect(harness.stateOps).toEqual([
      { op: 'enable', args: [MOCK_GL_DEPTH_TEST] },
      { op: 'depthMask', args: [false] },
      { op: 'enable', args: [MOCK_GL_CULL_FACE] },
      { op: 'cullFace', args: [MOCK_GL_CULL_FACE_BACK] },
      { op: 'frontFace', args: [MOCK_GL_FRONT_FACE_CCW] },
      { op: 'enable', args: [MOCK_GL_BLEND] },
      { op: 'blendFunc', args: [MOCK_GL_ONE, MOCK_GL_ONE_MINUS_SRC_ALPHA] },
      // G4 restore: the leaked CULL_FACE=ON would cull the next frame's
      // two-sided track ribbon (AC-017), so glass re-disables culling after
      // the last car draw.
      { op: 'disable', args: [MOCK_GL_CULL_FACE] },
    ]);
    expect(harness.calls.drawArrays).toBe(0);
  });

  it('rejects a malformed view-projection with the pass label', () => {
    const harness = createMockGlHarness();
    const pass = createGlassPass(
      harness.gl,
      buildGlassTestModels('vp guard', [1, 1, 1]),
    );
    expect(() =>
      pass.draw({
        viewProjection: new Float32Array(15),
        cameraWorld: [0, 0, 0],
        cars: [],
      }),
    ).toThrow(
      'neatRacing glass pass: viewProjection must contain exactly 16 elements, received 15',
    );
  });

  it('rejects a malformed camera sort anchor', () => {
    const harness = createMockGlHarness();
    const pass = createGlassPass(
      harness.gl,
      buildGlassTestModels('camera guard', [1, 1, 1]),
    );
    const malformedState = {
      viewProjection: IDENTITY_VIEW_PROJECTION_FIXTURE,
      cameraWorld: [0, 0],
      cars: [],
    } as unknown as GlassPassDrawState;
    expect(() => pass.draw(malformedState)).toThrow(
      'neatRacing glass pass: cameraWorld must contain exactly 3 components, received 2',
    );
  });

  it('draws one body plus the wheel mounts per car on the TRIANGLES mode', () => {
    const harness = createMockGlHarness();
    const pass = createGlassPass(
      harness.gl,
      buildGlassTestModels('budget', [1, 1, 1]),
    );
    pass.draw(
      buildGlassDrawState([
        { worldX: 0, worldZ: 0, yawRad: 0, wheelSpinRad: 0 },
      ]),
    );
    expect(harness.drawArraysRecords.length).toBe(
      1 + resolveVoxelWheelSockets().length,
    );
    expect(harness.drawArraysRecords[0]).toEqual({
      mode: MOCK_GL_TRIANGLES,
      first: 0,
      count: 36,
    });
  });

  it('sorts cars far -> near anchored at the camera XZ position', () => {
    const harness = createMockGlHarness();
    const pass = createGlassPass(
      harness.gl,
      buildGlassTestModels('sorted', [1, 1, 1]),
    );
    pass.draw(
      buildGlassDrawState([
        { worldX: 3, worldZ: 0, yawRad: 0, wheelSpinRad: 0 },
        { worldX: 0, worldZ: 0, yawRad: 0, wheelSpinRad: 0 },
        { worldX: 6, worldZ: 0, yawRad: 0, wheelSpinRad: 0 },
      ]),
    );
    const modelFeeds = harness.uniformMatrix4fvFeeds.filter(
      (feed) => feed.name === 'uModel',
    );
    expect(modelFeeds.length).toBe(15);
    expect([
      modelFeeds[0].values[12],
      modelFeeds[5].values[12],
      modelFeeds[10].values[12],
    ]).toEqual([6, 3, 0]);
  });

  it('disposes the program, both shaders, and both vertex buffers', () => {
    const harness = createMockGlHarness();
    const pass = createGlassPass(
      harness.gl,
      buildGlassTestModels('dispose', [1, 1, 1]),
    );
    pass.dispose();
    expect(harness.calls.deleteProgram).toBe(1);
    expect(harness.calls.deleteShader).toBe(2);
    expect(harness.calls.deleteBuffer).toBe(2);
  });
});

describe('glass warmup seam (AC-040)', () => {
  it('registers the warmup descriptor under the glass registry key', () => {
    const harness = createMockGlHarness();
    expect(toGlassWarmupProgram(harness.gl).key).toBe(
      NEATRACING_GLASS_PROGRAM_KEY,
    );
  });

  it('stays lazy until compileAndLink hydrates exactly one program', () => {
    const harness = createMockGlHarness();
    const warmup = toGlassWarmupProgram(harness.gl);
    expect(harness.calls.createProgram).toBe(0);
    warmup.compileAndLink();
    expect(harness.calls.createProgram).toBe(1);
  });

  it('draws the zero-pose body and wheel meshes through the shared path', () => {
    const harness = createMockGlHarness();
    const warmup = toGlassWarmupProgram(harness.gl);
    warmup.compileAndLink();
    expect(harness.calls.drawArrays).toBe(0);
    warmup.drawOffscreen(harness.gl);
    expect(harness.calls.drawArrays).toBe(
      1 + resolveVoxelWheelSockets().length,
    );
    expect(harness.drawArraysRecords[0]).toEqual({
      mode: MOCK_GL_TRIANGLES,
      first: 0,
      count: 288,
    });
  });
});

describe('neatRacing bloom pass constants (G8 / AC-016)', () => {
  it('pins the half-res target derived from the fixed 720p backing store', () => {
    expect(NEATRACING_BLOOM_TARGET_WIDTH).toBe(640);
    expect(NEATRACING_BLOOM_TARGET_HEIGHT).toBe(360);
  });

  it('pins the named blur tap cap and its exact tap count', () => {
    expect(NEATRACING_BLOOM_MAX_TAPS).toBe(13);
    expect(NEATRACING_BLOOM_TAP_COUNT).toBe(NEATRACING_BLOOM_MAX_TAPS);
  });

  it('pins the gaussian sigma, bright threshold, and composite intensity', () => {
    expect(NEATRACING_BLOOM_BLUR_SIGMA_TEXELS).toBe(2.5);
    expect(NEATRACING_BLOOM_BRIGHT_THRESHOLD).toBe(0.72);
    expect(NEATRACING_BLOOM_COMPOSITE_INTENSITY).toBe(0.35);
  });

  it('pins the Rec709 luma weight triple', () => {
    expect(NEATRACING_BLOOM_LUMA_WEIGHTS).toEqual([0.2126, 0.7152, 0.0722]);
  });

  it('registers the threshold, blur, and additive warmup keys in order', () => {
    expect(NEATRACING_BLOOM_THRESHOLD_PROGRAM_KEY).toBe('bloom-threshold');
    expect(NEATRACING_BLOOM_BLUR_PROGRAM_KEY).toBe('bloom-blur');
    expect(NEATRACING_BLOOM_ADDITIVE_PROGRAM_KEY).toBe('bloom-additive');
  });

  it('mirrors the warmup keys against the registry slice 6..8', () => {
    expect([
      NEATRACING_BLOOM_THRESHOLD_PROGRAM_KEY,
      NEATRACING_BLOOM_BLUR_PROGRAM_KEY,
      NEATRACING_BLOOM_ADDITIVE_PROGRAM_KEY,
    ]).toEqual(NEATRACING_WARMED_PROGRAM_KEYS.slice(6, 9));
    expect(NEATRACING_WARMED_PROGRAM_KEYS).toContain(
      NEATRACING_BLOOM_BLUR_PROGRAM_KEY,
    );
  });
});

describe('bloom bright math (AC-016)', () => {
  it('computes Rec709 luminance on pure channels', () => {
    expect(computeBloomLuminance(1, 0, 0)).toBe(
      NEATRACING_BLOOM_LUMA_WEIGHTS[0],
    );
    expect(computeBloomLuminance(0, 1, 0)).toBe(
      NEATRACING_BLOOM_LUMA_WEIGHTS[1],
    );
    expect(computeBloomLuminance(0, 0, 1)).toBe(
      NEATRACING_BLOOM_LUMA_WEIGHTS[2],
    );
  });

  it('anchors luminance at 0 for black and near 1 for white', () => {
    expect(computeBloomLuminance(0, 0, 0)).toBe(0);
    expect(computeBloomLuminance(1, 1, 1)).toBeCloseTo(1, 12);
  });

  it('returns zero bright factor at or below the brightness threshold', () => {
    expect(computeBloomBrightFactor(0.5, 0.5, 0.5)).toBe(0);
    expect(computeBloomBrightFactor(0, 0, 0)).toBe(0);
  });

  it('clamps the bright factor at one for saturated scene texels', () => {
    expect(computeBloomBrightFactor(1, 1, 1)).toBeCloseTo(1, 12);
  });

  it('ramps linearly between threshold and 1 with the named span', () => {
    const luminance = computeBloomLuminance(1, 0.9, 0.8);
    const expected =
      (luminance - NEATRACING_BLOOM_BRIGHT_THRESHOLD) /
      (1 - NEATRACING_BLOOM_BRIGHT_THRESHOLD);
    expect(computeBloomBrightFactor(1, 0.9, 0.8)).toBeCloseTo(expected, 12);
  });

  it('monotonically increases the factor between threshold probes', () => {
    const low = computeBloomBrightFactor(0.8, 0.8, 0.8);
    const high = computeBloomBrightFactor(0.9, 0.9, 0.9);
    expect(high).toBeGreaterThan(low);
    expect(low).toBeGreaterThan(0);
  });
});

describe('bloom blur tap table (AC-016)', () => {
  it('spans NEATRACING_BLOOM_TAP_COUNT integer offsets from -6 to +6', () => {
    const taps = computeBloomBlurTaps();
    expect(taps.length).toBe(NEATRACING_BLOOM_TAP_COUNT);
    expect(taps.map((tap) => tap.offsetTexels)).toEqual([
      -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
    ]);
  });

  it('symmetrizes the weights and normalizes their sum to one', () => {
    const taps = computeBloomBlurTaps();
    const symmetryMismatches = taps.filter(
      (tap, index) => tap.weight !== taps[taps.length - 1 - index].weight,
    );
    expect(symmetryMismatches).toEqual([]);
    expect(taps.reduce((total, tap) => total + tap.weight, 0)).toBeCloseTo(
      1,
      9,
    );
  });

  it('rises monotonically to the center tap under the pinned sigma', () => {
    const weights = computeBloomBlurTaps().map((tap) => tap.weight);
    const centerWeight = weights[6];
    const outwardViolations = weights
      .slice(0, 6)
      .filter((weight, index) => weight >= weights[index + 1]);
    expect(centerWeight).toBe(Math.max(...weights));
    expect(outwardViolations).toEqual([]);
  });
});

describe('bloom GLSL sources (G8)', () => {
  it('keeps every stage source in GLSL ES 3.00', () => {
    const sources = [
      NEATRACING_BLOOM_VERTEX_SHADER_SOURCE,
      NEATRACING_BLOOM_THRESHOLD_FRAGMENT_SHADER_SOURCE,
      NEATRACING_BLOOM_BLUR_FRAGMENT_SHADER_SOURCE,
      NEATRACING_BLOOM_ADDITIVE_FRAGMENT_SHADER_SOURCE,
    ];
    const nonEs300 = sources.filter(
      (source) => !source.startsWith('#version 300 es'),
    );
    expect(nonEs300).toEqual([]);
  });

  it('pins the threshold luminance weights and brightness threshold', () => {
    const source = NEATRACING_BLOOM_THRESHOLD_FRAGMENT_SHADER_SOURCE;
    expect(source).toContain('uniform sampler2D uScene');
    expect(source).toContain('vec3(0.2126, 0.7152, 0.0722)');
    expect(source).toContain(
      `const float kBrightnessThreshold = ${NEATRACING_BLOOM_BRIGHT_THRESHOLD};`,
    );
  });

  it('pins the 13-tap GLSL const arrays and the blur uniforms', () => {
    const source = NEATRACING_BLOOM_BLUR_FRAGMENT_SHADER_SOURCE;
    expect(source).toContain(
      `const int kBlurTapCount = ${NEATRACING_BLOOM_TAP_COUNT};`,
    );
    expect(source).toContain(
      'const float kBlurOffsets[kBlurTapCount] = float[kBlurTapCount](-6.0',
    );
    expect(source).toContain('uBlurDirection * kBlurOffsets[tap]');
  });

  // Mock-GL cannot evaluate GLSL algebra, so this test locks the
  // single-scene-transfer composite contract as source pins (consistent
  // with the source-pin suite in this describe block): the composite
  // emits scene + bloom * kCompositeIntensity and is the ONLY scene
  // transfer to the display.
  it('pins the single-scene-transfer composite contract', () => {
    const source = NEATRACING_BLOOM_ADDITIVE_FRAGMENT_SHADER_SOURCE;
    expect(source).toContain('uniform sampler2D uBloomTexture');
    expect(source).toContain(
      `const float kCompositeIntensity = ${NEATRACING_BLOOM_COMPOSITE_INTENSITY};`,
    );
    expect(source).toContain(
      'fragColor = vec4(scene + bloom * kCompositeIntensity, 1.0);',
    );
  });
});

describe('bloom target creation (AC-016)', () => {
  it('allocates three internal targets with completeness checks', () => {
    const harness = createMockGlHarness();
    createBloomPass(harness.gl);
    expect(harness.calls.createFramebuffer).toBe(3);
    expect(harness.calls.createTexture).toBe(3);
    expect(harness.calls.checkFramebufferStatus).toBe(3);
  });

  it('sizes every internal RGBA8 target at the half-res dims', () => {
    const harness = createMockGlHarness();
    createBloomPass(harness.gl);
    const mismatched = harness.texImage2DRecords.filter(
      (record) =>
        record.width !== NEATRACING_BLOOM_TARGET_WIDTH ||
        record.height !== NEATRACING_BLOOM_TARGET_HEIGHT,
    );
    expect(harness.texImage2DRecords.length).toBe(3);
    expect(mismatched).toEqual([]);
  });

  it('rejects an incomplete internal framebuffer with the pass label', () => {
    const harness = createMockGlHarness({ checkFramebufferStatusResult: 0 });
    expect(() => createBloomPass(harness.gl)).toThrow(
      'neatRacing bloom pass: bloom bright framebuffer incomplete (status 0)',
    );
  });
});

describe('bloom draw pipeline (AC-016 / AC-035)', () => {
  it('skips every GL operation when the frame is disabled', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    const opsBefore = countMockGlOps(harness);
    pass.draw(buildBloomDrawFrame(false));
    expect(countMockGlOps(harness)).toBe(opsBefore);
  });

  it('rejects non-positive or non-integer composite viewports before GL ops', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    expect(() =>
      pass.draw({ ...buildBloomDrawFrame(true), targetWidth: 0 }),
    ).toThrow(
      'neatRacing bloom pass: target width/height must be positive integers, received 0x720',
    );
    expect(harness.stateOps).toEqual([]);
  });

  it('draws the four pipeline stages as fullscreen TRIANGLE_STRIP quads', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    const drawsBefore = harness.calls.drawArrays;
    pass.draw(buildBloomDrawFrame(true));
    expect(harness.calls.drawArrays - drawsBefore).toBe(4);
    expect(harness.drawArraysRecords).toEqual([
      { mode: MOCK_GL_TRIANGLE_STRIP, first: 0, count: 4 },
      { mode: MOCK_GL_TRIANGLE_STRIP, first: 0, count: 4 },
      { mode: MOCK_GL_TRIANGLE_STRIP, first: 0, count: 4 },
      { mode: MOCK_GL_TRIANGLE_STRIP, first: 0, count: 4 },
    ]);
  });

  it('binds the bright, blur A, blur B, then display framebuffers in order', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    const bindsBefore = harness.bindFramebufferRecords.length;
    const [brightFramebuffer, blurFramebufferA, blurFramebufferB] =
      harness.createdFramebuffers;
    pass.draw(buildBloomDrawFrame(true));
    const stageBinds = harness.bindFramebufferRecords
      .slice(bindsBefore)
      .map((record) => record.framebuffer);
    expect(stageBinds).toEqual([
      brightFramebuffer,
      blurFramebufferA,
      blurFramebufferB,
      null,
    ]);
  });

  it('pins the half-res pipeline viewport then the composite viewport', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    const viewportsBefore = harness.viewportFeeds.length;
    pass.draw(buildBloomDrawFrame(true));
    expect(harness.viewportFeeds.slice(viewportsBefore)).toEqual([
      {
        width: NEATRACING_BLOOM_TARGET_WIDTH,
        height: NEATRACING_BLOOM_TARGET_HEIGHT,
      },
      { width: 1280, height: 720 },
    ]);
  });

  it('applies the exact G8 fullscreen-post state machine once per draw', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    const opsBefore = harness.stateOps.length;
    pass.draw(buildBloomDrawFrame(true));
    expect(harness.stateOps.slice(opsBefore)).toEqual([
      { op: 'depthMask', args: [false] },
      { op: 'disable', args: [MOCK_GL_DEPTH_TEST] },
      { op: 'disable', args: [MOCK_GL_CULL_FACE] },
      { op: 'disable', args: [MOCK_GL_BLEND] },
      { op: 'enable', args: [MOCK_GL_BLEND] },
      { op: 'blendFunc', args: [MOCK_GL_ONE, MOCK_GL_ONE] },
      { op: 'disable', args: [MOCK_GL_BLEND] },
    ]);
  });

  it('feeds the sampler units in stage order', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    const feedsBefore = harness.uniform1iFeeds.length;
    pass.draw(buildBloomDrawFrame(true));
    expect(harness.uniform1iFeeds.slice(feedsBefore)).toEqual([
      { name: 'uScene', value: 0 },
      { name: 'uBlurTexture', value: 0 },
      { name: 'uScene', value: 0 },
      { name: 'uBloomTexture', value: 1 },
    ]);
  });

  it('binds scene and blurred textures on sampler units 0 and 1', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    const bindsBefore = harness.bindTextureRecords.length;
    const activeBefore = harness.activeTextureRecords.length;
    const [brightTexture, blurTextureA, blurTextureB] = harness.createdTextures;
    pass.draw(buildBloomDrawFrame(true));
    const stageBinds = harness.bindTextureRecords
      .slice(bindsBefore)
      .map((record) => record.texture);
    expect(stageBinds).toEqual([
      SCENE_TEXTURE_FIXTURE,
      brightTexture,
      blurTextureA,
      SCENE_TEXTURE_FIXTURE,
      blurTextureB,
    ]);
    expect(harness.activeTextureRecords.slice(activeBefore)).toEqual([
      MOCK_GL_TEXTURE0,
      MOCK_GL_TEXTURE0,
      MOCK_GL_TEXTURE0,
      MOCK_GL_TEXTURE0,
      MOCK_GL_TEXTURE1,
    ]);
  });

  it('drives the separable blur directions along one texel per axis', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    const feedsBefore = harness.uniform2fFeeds.length;
    pass.draw(buildBloomDrawFrame(true));
    const feeds = harness.uniform2fFeeds.slice(feedsBefore);
    expect(feeds.length).toBe(2);
    expect(feeds[0].x).toBeCloseTo(1 / NEATRACING_BLOOM_TARGET_WIDTH, 12);
    expect(feeds[1].y).toBeCloseTo(1 / NEATRACING_BLOOM_TARGET_HEIGHT, 12);
  });
});

describe('bloom dispose (P9)', () => {
  it('releases the three stage programs and six shaders', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    pass.dispose();
    expect(harness.calls.deleteProgram).toBe(3);
    expect(harness.calls.deleteShader).toBe(6);
  });

  it('releases the quad buffer and the three internal targets', () => {
    const harness = createMockGlHarness();
    const pass = createBloomPass(harness.gl);
    pass.dispose();
    expect(harness.calls.deleteBuffer).toBe(1);
    expect(harness.calls.deleteTexture).toBe(3);
    expect(harness.calls.deleteFramebuffer).toBe(3);
  });
});

describe('bloom warmup seam (AC-040)', () => {
  it('registers descriptors under the keys[6..8] warmup keys', () => {
    const harness = createMockGlHarness();
    const warmups = toBloomWarmupPrograms(harness.gl);
    expect(warmups.map((program) => program.key)).toEqual([
      NEATRACING_BLOOM_THRESHOLD_PROGRAM_KEY,
      NEATRACING_BLOOM_BLUR_PROGRAM_KEY,
      NEATRACING_BLOOM_ADDITIVE_PROGRAM_KEY,
    ]);
  });

  it('compiles lazily into exactly three programs on the first hydrate', () => {
    const harness = createMockGlHarness();
    const warmups = toBloomWarmupPrograms(harness.gl);
    expect(harness.calls.createProgram).toBe(0);
    warmups[0].compileAndLink();
    expect(harness.calls.createProgram).toBe(3);
    warmups[1].compileAndLink();
    warmups[2].compileAndLink();
    expect(harness.calls.createProgram).toBe(3);
  });

  it('uploads the 1x1 white placeholder scene texture before the targets', () => {
    const harness = createMockGlHarness();
    const warmups = toBloomWarmupPrograms(harness.gl);
    warmups[0].compileAndLink();
    expect(harness.texImage2DRecords[0]).toEqual({
      target: MOCK_GL_TEXTURE_2D,
      width: 1,
      height: 1,
      source: new Uint8Array([255, 255, 255, 255]),
    });
    expect(harness.texImage2DRecords.length).toBe(4);
  });

  it('pushes the full four-draw pipeline through each drawOffscreen', () => {
    const harness = createMockGlHarness();
    const warmups = toBloomWarmupPrograms(harness.gl);
    for (const warmup of warmups) {
      warmup.compileAndLink();
    }
    const perStageDeltas: number[] = [];
    for (const warmup of warmups) {
      const drawsBefore = harness.calls.drawArrays;
      warmup.drawOffscreen(harness.gl);
      perStageDeltas.push(harness.calls.drawArrays - drawsBefore);
    }
    expect(perStageDeltas).toEqual([4, 4, 4]);
  });

  it('viewports the half-res pipeline and the 1x1 warmup display target', () => {
    const harness = createMockGlHarness();
    const warmups = toBloomWarmupPrograms(harness.gl);
    warmups[0].compileAndLink();
    const viewportsBefore = harness.viewportFeeds.length;
    warmups[0].drawOffscreen(harness.gl);
    expect(harness.viewportFeeds.slice(viewportsBefore)).toEqual([
      {
        width: NEATRACING_BLOOM_TARGET_WIDTH,
        height: NEATRACING_BLOOM_TARGET_HEIGHT,
      },
      { width: 1, height: 1 },
    ]);
  });
});
