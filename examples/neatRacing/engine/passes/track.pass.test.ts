/**
 * Red-phase contract tests for the Neat Racing track slab draw pass
 * (slice P1-ground-composition; AC-017). The pass owns the GPU seam that
 * turns the deterministic {@link buildTrackSlabs} mesh into visible
 * geometry in the composed frame.
 *
 * RED PREMISE: this suite must fail while `./track.pass` is absent — the
 * sanctioned module-absence red seam. The expected failure is a TypeScript
 * diagnostic such as `Cannot find module './track.pass'` (TS2307). Any
 * other failure mode (fixture typo, mock error, etc.) is a defective red
 * and must be fixed in the test fixture.
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - Exports the pinned warmup key `'track'` matching
 *   {@link NEATRACING_WARMED_PROGRAM_KEYS}.
 * - {@link createTrackPass} returns a draw-only handle whose `draw(state)`
 *   accepts the same view-projection + camera-world bag the composer sends.
 * - A single draw call issues at least one geometry draw call, uses the
 *   camera/view-projection uniforms, and enables depth testing so the track
 *   sits on the ground rather than bleeding through the sky.
 * - {@link toTrackWarmupProgram} exposes the AC-040 seam: compile once,
 *   draw offscreen once, keyed to the track registry slot.
 */

import { describe, expect, it } from '@jest/globals';
import { NEATRACING_WARMED_PROGRAM_KEYS } from '../context.service';
import {
  NEATRACING_LIGHTING_UNIFORM_NAMES,
  NEATRACING_SUN_COLOR,
  NEATRACING_SUN_DIRECTION,
} from '../lighting';
import type { TrackSlabMesh } from '../../track/track.slabs';
import {
  NEATRACING_TRACK_PROGRAM_KEY,
  type TrackPass,
  type TrackPassDrawState,
  buildTrackVboData,
  createTrackPass,
  toTrackWarmupProgram,
} from './track.pass';

/** Identity view-projection matrix used by the mock draw state. */
const IDENTITY_VIEW_PROJECTION = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
] as const;

/** Fixture camera world position 10 m above the origin. */
const CAMERA_WORLD: TrackPassDrawState['cameraWorld'] = [0, 10, 20];

/** `gl.DEPTH_TEST` enum value (0x0B71) asserted on the mock draw state. */
const MOCK_GL_DEPTH_TEST = 0x0b71;

/** `gl.CULL_FACE` enum value (0x0B44) asserted as disabled every draw. */
const MOCK_GL_CULL_FACE = 0x0b44;

/** `gl.BLEND` enum value (0x0BE2) asserted as disabled every draw. */
const MOCK_GL_BLEND = 0x0be2;

/** `gl.TRIANGLES` enum value (0x0004) asserted as a plausible geometry mode. */
const MOCK_GL_TRIANGLES = 0x0004;

/** Minimal recording mock of the GL entry points the track pass touches. */
interface TrackMockGlHarness {
  readonly gl: WebGL2RenderingContext;
  readonly drawCalls: ReadonlyArray<{
    readonly mode: number;
    readonly first: number;
    readonly count: number;
  }>;
  readonly enableCalls: readonly number[];
  readonly disableCalls: readonly number[];
  readonly uniformMatrixCalls: readonly Float32Array[];
  readonly uniformLocationRequests: readonly string[];
  readonly uniform3fFeeds: ReadonlyArray<{
    readonly name: string;
    readonly channels: readonly [number, number, number];
  }>;
  readonly uniform1fFeeds: ReadonlyArray<{
    readonly name: string;
    readonly value: number;
  }>;
}

/** Filters recorded uniform3f feeds down to one uniform name. */
function uniform3fFeedsNamed(
  harness: TrackMockGlHarness,
  name: string,
): ReadonlyArray<{
  readonly name: string;
  readonly channels: readonly [number, number, number];
}> {
  return harness.uniform3fFeeds.filter((feed) => feed.name === name);
}

/** Filters recorded uniform1f feeds down to one uniform name. */
function uniform1fFeedsNamed(
  harness: TrackMockGlHarness,
  name: string,
): ReadonlyArray<{ readonly name: string; readonly value: number }> {
  return harness.uniform1fFeeds.filter((feed) => feed.name === name);
}

/**
 * Builds a recording mock WebGL2 context. The pass is expected to compile
 * and link a program, upload slab geometry once, and issue indexed or
 * non-indexed draw calls each frame; only the observable draw/state/uniform
 * contract is asserted here.
 */
function createTrackMockGlHarness(): TrackMockGlHarness {
  const drawCalls: Array<{
    readonly mode: number;
    readonly first: number;
    readonly count: number;
  }> = [];
  const enableCalls: number[] = [];
  const disableCalls: number[] = [];
  const uniformMatrixCalls: Float32Array[] = [];
  const uniformLocationRequests: string[] = [];
  const uniform3fFeeds: Array<{
    readonly name: string;
    readonly channels: readonly [number, number, number];
  }> = [];
  const uniform1fFeeds: Array<{
    readonly name: string;
    readonly value: number;
  }> = [];
  const gl = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b20,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    ARRAY_BUFFER: 0x8892,
    ELEMENT_ARRAY_BUFFER: 0x8893,
    STATIC_DRAW: 0x88e4,
    FLOAT: 0x1406,
    UNSIGNED_SHORT: 0x1403,
    DEPTH_TEST: MOCK_GL_DEPTH_TEST,
    TRIANGLES: MOCK_GL_TRIANGLES,
    createShader: (): object => ({}),
    shaderSource: (): void => undefined,
    compileShader: (): void => undefined,
    getShaderParameter: (): boolean => true,
    getShaderInfoLog: (): string => '',
    createProgram: (): object => ({}),
    attachShader: (): void => undefined,
    linkProgram: (): void => undefined,
    getProgramParameter: (): boolean => true,
    getProgramInfoLog: (): string => '',
    getAttribLocation: (): number => 0,
    getUniformLocation: (_program: unknown, name: string): object => {
      uniformLocationRequests.push(name);
      return { name };
    },
    createBuffer: (): object => ({}),
    bindBuffer: (): void => undefined,
    bufferData: (): void => undefined,
    enableVertexAttribArray: (): void => undefined,
    vertexAttribPointer: (): void => undefined,
    enable: (target: number): void => {
      enableCalls.push(target);
    },
    disable: (target: number): void => {
      disableCalls.push(target);
    },
    depthMask: (): void => undefined,
    depthFunc: (): void => undefined,
    useProgram: (): void => undefined,
    uniformMatrix4fv: (
      _location: unknown,
      _transpose: boolean,
      matrix: Float32Array,
    ): void => {
      uniformMatrixCalls.push(matrix);
    },
    uniform3f: (
      location: { name: string },
      red: number,
      green: number,
      blue: number,
    ): void => {
      uniform3fFeeds.push({
        name: location.name,
        channels: [red, green, blue],
      });
    },
    uniform3fv: (
      location: { name: string },
      channels: readonly [number, number, number],
    ): void => {
      uniform3fFeeds.push({ name: location.name, channels });
    },
    uniform1f: (location: { name: string }, value: number): void => {
      uniform1fFeeds.push({ name: location.name, value });
    },
    uniform1fv: (
      location: { name: string },
      values: readonly number[],
    ): void => {
      for (const value of values) {
        uniform1fFeeds.push({ name: location.name, value });
      }
    },
    drawArrays: (mode: number, first: number, count: number): void => {
      drawCalls.push({ mode, first, count });
    },
    drawElements: (
      mode: number,
      count: number,
      _type: number,
      offset: number,
    ): void => {
      drawCalls.push({ mode, first: offset, count });
    },
  };
  return {
    gl: gl as unknown as WebGL2RenderingContext,
    drawCalls,
    enableCalls,
    disableCalls,
    uniformMatrixCalls,
    uniformLocationRequests,
    uniform3fFeeds,
    uniform1fFeeds,
  };
}

/** Builds a deterministic draw state fixture. */
function createDrawState(): TrackPassDrawState {
  return {
    viewProjection: new Float32Array(IDENTITY_VIEW_PROJECTION),
    cameraWorld: CAMERA_WORLD,
  };
}

describe('neatRacing track slab pass (AC-017)', () => {
  it('registers under the track warmup key from the program registry', () => {
    expect(NEATRACING_TRACK_PROGRAM_KEY).toBe('track');
    expect(NEATRACING_WARMED_PROGRAM_KEYS).toContain(
      NEATRACING_TRACK_PROGRAM_KEY,
    );
  });

  it('returns a pass handle with a draw method from createTrackPass', () => {
    const harness = createTrackMockGlHarness();
    const pass: TrackPass = createTrackPass(harness.gl);

    expect(typeof pass.draw).toBe('function');
  });

  it('draws at least one geometry call when the camera is above the track', () => {
    const harness = createTrackMockGlHarness();
    const pass = createTrackPass(harness.gl);

    pass.draw(createDrawState());

    expect(harness.drawCalls.length).toBeGreaterThanOrEqual(1);
  });

  it('enables depth testing so the track ribbon sits on the ground', () => {
    const harness = createTrackMockGlHarness();
    const pass = createTrackPass(harness.gl);

    pass.draw(createDrawState());

    expect(harness.enableCalls).toContain(MOCK_GL_DEPTH_TEST);
  });

  it('explicitly disables face culling every draw so leaked cull state cannot hide the ribbon', () => {
    const harness = createTrackMockGlHarness();
    const pass = createTrackPass(harness.gl);

    pass.draw(createDrawState());

    expect(harness.disableCalls).toContain(MOCK_GL_CULL_FACE);
  });

  it('disables blending so the opaque track ribbon overwrites prior pixels', () => {
    const harness = createTrackMockGlHarness();
    const pass = createTrackPass(harness.gl);

    pass.draw(createDrawState());

    expect(harness.disableCalls).toContain(MOCK_GL_BLEND);
  });

  it('feeds the camera view-projection matrix to the track shader', () => {
    const harness = createTrackMockGlHarness();
    const pass = createTrackPass(harness.gl);
    const state = createDrawState();

    pass.draw(state);

    expect(
      harness.uniformMatrixCalls.some(
        (matrix) => matrix === state.viewProjection,
      ),
    ).toBe(true);
  });

  it('exposes the AC-040 warmup seam under the track registry key', () => {
    const harness = createTrackMockGlHarness();

    const warmup = toTrackWarmupProgram(harness.gl);

    expect(warmup.key).toBe(NEATRACING_WARMED_PROGRAM_KEYS[3]);
  });

  it('draws the warmed pipeline offscreen through the shared pass draw path', () => {
    const harness = createTrackMockGlHarness();
    const warmup = toTrackWarmupProgram(harness.gl);
    warmup.compileAndLink();
    const drawsBeforeWarmup = harness.drawCalls.length;

    warmup.drawOffscreen(harness.gl);

    expect(harness.drawCalls.length - drawsBeforeWarmup).toBeGreaterThanOrEqual(
      1,
    );
  });

  it('renders slab top faces as a flat near-black matte base', () => {
    const matte = 12 / 255;
    const roadColor: readonly [number, number, number] = [64, 64, 68];
    const mesh: TrackSlabMesh = {
      palette: [{ type: 'road', color: roadColor }],
      quads: [
        // Top face at world y = 0, normalAxis = 1.
        {
          position: [0, 0, 0],
          normalAxis: 1,
          width: 1,
          height: 1,
          paletteIndex: 0,
        },
      ],
    };

    const vbo = buildTrackVboData(mesh);

    expect(vbo[3]).toBeCloseTo(matte, 6);
    expect(vbo[4]).toBeCloseTo(matte, 6);
    expect(vbo[5]).toBeCloseTo(matte, 6);
  });

  it('keeps palette colors on slab side and bottom faces', () => {
    const curbColor: readonly [number, number, number] = [214, 44, 44];
    const grassColor: readonly [number, number, number] = [46, 156, 70];
    const mesh: TrackSlabMesh = {
      palette: [
        { type: 'curb', color: curbColor },
        { type: 'grass', color: grassColor },
      ],
      quads: [
        // Vertical curb face, not a top face.
        {
          position: [0, 0, 0],
          normalAxis: 0,
          width: 1,
          height: 0.5,
          paletteIndex: 0,
        },
        // Bottom grass face at world y = -0.5.
        {
          position: [0, -0.5, 0],
          normalAxis: 1,
          width: 1,
          height: 1,
          paletteIndex: 1,
        },
      ],
    };

    const vbo = buildTrackVboData(mesh);

    // Vertical face keeps curb color (first vertex color after position xyz).
    expect(vbo[3]).toBeCloseTo(curbColor[0] / 255, 6);
    expect(vbo[4]).toBeCloseTo(curbColor[1] / 255, 6);
    expect(vbo[5]).toBeCloseTo(curbColor[2] / 255, 6);

    // Bottom face keeps grass color (second quad starts at vertex 6).
    const base = 6 * 6;
    expect(vbo[base + 3]).toBeCloseTo(grassColor[0] / 255, 6);
    expect(vbo[base + 4]).toBeCloseTo(grassColor[1] / 255, 6);
    expect(vbo[base + 5]).toBeCloseTo(grassColor[2] / 255, 6);
  });
});

describe('neatRacing track pass shared sun lighting (P1-world-lighting-coherence)', () => {
  it('resolves the shared sun direction uniform when building the track program', () => {
    const harness = createTrackMockGlHarness();
    createTrackPass(harness.gl);
    expect(harness.uniformLocationRequests).toContain('uSunDirection');
  });

  it('resolves all shared lighting uniforms when building the track program', () => {
    const harness = createTrackMockGlHarness();
    createTrackPass(harness.gl);
    for (const name of NEATRACING_LIGHTING_UNIFORM_NAMES) {
      expect(harness.uniformLocationRequests).toContain(name);
    }
  });

  it('feeds the shared sun direction and color during track draw', () => {
    const harness = createTrackMockGlHarness();
    const pass = createTrackPass(harness.gl);
    pass.draw(createDrawState());
    const sunDirectionFeeds = uniform3fFeedsNamed(harness, 'uSunDirection');
    const sunColorFeeds = uniform3fFeedsNamed(harness, 'uSunColor');
    expect(sunDirectionFeeds.length).toBeGreaterThan(0);
    expect(sunColorFeeds.length).toBeGreaterThan(0);
    expect(sunDirectionFeeds[0].channels).toEqual(NEATRACING_SUN_DIRECTION);
    expect(sunColorFeeds[0].channels).toEqual(NEATRACING_SUN_COLOR);
  });

  it('feeds the camera world position and fog density during track draw', () => {
    const harness = createTrackMockGlHarness();
    const pass = createTrackPass(harness.gl);
    const state = createDrawState();
    pass.draw(state);
    const cameraFeeds = uniform3fFeedsNamed(harness, 'uCameraWorld');
    const fogDensityFeeds = uniform1fFeedsNamed(harness, 'uFogDensity');
    expect(cameraFeeds.length).toBeGreaterThan(0);
    expect(cameraFeeds[0].channels).toEqual(state.cameraWorld);
    expect(fogDensityFeeds.length).toBeGreaterThan(0);
  });
});
