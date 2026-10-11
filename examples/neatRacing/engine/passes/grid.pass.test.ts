/**
 * Red-phase contract tests for the NeatRacing world-space ground grid pass
 * (slice P1-S2-B; AC-005 constant screen-space line width via `fwidth`,
 * depth-fade alpha 0.05 (far) -> 0.58 (near), and no z-fighting against the
 * track).
 *
 * The Jest project runs in a plain Node environment, so the shader math is
 * mirrored as pure TypeScript functions driven by fixture `fwidth` derivative
 * doubles, and the GL interop runs against a hand-rolled recording mock of a
 * `WebGL2RenderingContext` (no real GPU, no DOM). The pass must never depend
 * on `gl.lineWidth` (unsupported > 1 in WebGL2/ANGLE, G7), so both shader
 * sources are asserted to be free of it.
 */
import { describe, expect, it } from '@jest/globals';
import { NEATRACING_WARMED_PROGRAM_KEYS } from '../context.service';
import {
  NEATRACING_DRAW_DISTANCE_M,
  NEATRACING_GRID_FRAGMENT_SHADER,
  NEATRACING_GRID_GLOW_ALPHA_MULTIPLIER,
  NEATRACING_GRID_GLOW_WIDTH_PX,
  NEATRACING_GRID_LINE_COLOR_RGB,
  NEATRACING_GRID_LINE_WIDTH_PX,
  NEATRACING_GRID_MAX_ALPHA,
  NEATRACING_GRID_MIN_ALPHA,
  NEATRACING_GRID_PLANE_Y_OFFSET_M,
  NEATRACING_GRID_PROGRAM_KEY,
  NEATRACING_GRID_SPACING_WORLD,
  NEATRACING_GRID_VERTEX_SHADER,
  createGridPass,
  gridAxisLineDistance,
  gridBandCoverage01,
  gridFogFactor01,
  gridFragmentAlpha,
  gridLineAlpha,
  gridLineThresholdWorld,
  toGridWarmupProgram,
} from './grid.pass';
import type { GridPassDrawState } from './grid.pass';

/** Fixture fwidth derivative doubles emulating near/mid/far viewports. */
const FWIDTH_FIXTURES_WORLD = [0.02, 0.008, 0.002] as const;

/** Fixture distances (m) across the AC-005 depth-fade envelope. */
const NEAR_DISTANCE_M = 0;
const MID_DISTANCE_M = NEATRACING_DRAW_DISTANCE_M / 2;
const FAR_DISTANCE_M = NEATRACING_DRAW_DISTANCE_M;

/** `gl.BLEND` enum value (0x0BE2) asserted on the mock draw state. */
const MOCK_GL_BLEND = 0x0be2;

/** `gl.SRC_ALPHA` enum value (0x0302) asserted on the mock blend function. */
const MOCK_GL_SRC_ALPHA = 0x0302;

/** `gl.ONE_MINUS_SRC_ALPHA` enum value (0x0303) asserted on the blend fn. */
const MOCK_GL_ONE_MINUS_SRC_ALPHA = 0x0303;

/** `gl.DEPTH_TEST` enum value (0x0B71) asserted on the mock draw state. */
const MOCK_GL_DEPTH_TEST = 0x0b71;

/** `gl.TRIANGLES` enum value (0x0004) asserted on the mock draw call. */
const MOCK_GL_TRIANGLES = 0x0004;

/** `gl.TRIANGLE_STRIP` enum value (0x0005) used by other passes. */
const MOCK_GL_TRIANGLE_STRIP = 0x0005;

/** Number of uniforms the pass resolves at creation (one call per uniform). */
const GRID_UNIFORM_COUNT = 11;

/** One recorded mock-GL call. */
interface RecordedGlCall {
  readonly op: string;
  readonly args: readonly unknown[];
}

/** Options controlling the mock-GL success/failure paths. */
interface MockGlOptions {
  readonly shaderCompileResult?: boolean;
  readonly programLinkResult?: boolean;
}

/** Harness returned by `createMockGl`. */
interface MockGlHarness {
  readonly gl: WebGL2RenderingContext;
  callsFor(op: string): ReadonlyArray<readonly unknown[]>;
}

/**
 * Creates a hand-rolled recording mock of a `WebGL2RenderingContext`. Calls
 * are appended to a flat log; uniform locations carry their own name so the
 * tests can attribute `uniform*` calls to specific uniforms.
 */
function createMockGl(options: MockGlOptions = {}): MockGlHarness {
  const log: RecordedGlCall[] = [];
  const record =
    (op: string) =>
    (...args: unknown[]): void => {
      log.push({ op, args });
    };
  const shaderHandle = { kind: 'shader' };
  const programHandle = { kind: 'program' };
  const bufferHandle = { kind: 'buffer' };
  const gl = {
    createShader(type: number): unknown {
      log.push({ op: 'createShader', args: [type] });
      return shaderHandle;
    },
    shaderSource: record('shaderSource'),
    compileShader: record('compileShader'),
    getShaderParameter(): boolean {
      log.push({ op: 'getShaderParameter', args: [] });
      return options.shaderCompileResult ?? true;
    },
    getShaderInfoLog(): string {
      return 'mock shader info log';
    },
    createProgram(): unknown {
      log.push({ op: 'createProgram', args: [] });
      return programHandle;
    },
    attachShader: record('attachShader'),
    linkProgram: record('linkProgram'),
    getProgramParameter(): boolean {
      log.push({ op: 'getProgramParameter', args: [] });
      return options.programLinkResult ?? true;
    },
    getProgramInfoLog(): string {
      return 'mock program info log';
    },
    TRIANGLES: MOCK_GL_TRIANGLES,
    createBuffer(): unknown {
      log.push({ op: 'createBuffer', args: [] });
      return bufferHandle;
    },
    bindBuffer: record('bindBuffer'),
    bufferData: record('bufferData'),
    getAttribLocation(_program: unknown, name: string): number {
      log.push({ op: 'getAttribLocation', args: [name] });
      return 0;
    },
    getUniformLocation(_program: unknown, name: string): unknown {
      log.push({ op: 'getUniformLocation', args: [name] });
      return { name };
    },
    enableVertexAttribArray: record('enableVertexAttribArray'),
    vertexAttribPointer: record('vertexAttribPointer'),
    useProgram: record('useProgram'),
    enable: record('enable'),
    disable: record('disable'),
    depthMask: record('depthMask'),
    blendFunc: record('blendFunc'),
    uniform1f: record('uniform1f'),
    uniform2f: record('uniform2f'),
    uniform3f: record('uniform3f'),
    uniformMatrix4fv: record('uniformMatrix4fv'),
    drawArrays: record('drawArrays'),
  } as unknown as WebGL2RenderingContext;
  return {
    gl,
    callsFor(op: string): ReadonlyArray<readonly unknown[]> {
      return log.filter((call) => call.op === op).map((call) => call.args);
    },
  };
}

/** Reads the single float a `uniform1f` call set for `uniformName`. */
function uniform1fValue(harness: MockGlHarness, uniformName: string): number {
  const args = harness
    .callsFor('uniform1f')
    .find((call) => (call[0] as { name: string }).name === uniformName);
  if (!args) {
    throw new Error(`uniform ${uniformName} was never set`);
  }
  return args[1] as number;
}

/** Reads the vec2 a `uniform2f` call set for `uniformName`. */
function uniform2fValue(
  harness: MockGlHarness,
  uniformName: string,
): readonly [number, number] {
  const args = harness
    .callsFor('uniform2f')
    .find((call) => (call[0] as { name: string }).name === uniformName);
  if (!args) {
    throw new Error(`uniform ${uniformName} was never set`);
  }
  return [args[1] as number, args[2] as number];
}

/** Reads the vec3 a `uniform3f` call set for `uniformName`. */
function uniform3fValue(
  harness: MockGlHarness,
  uniformName: string,
): readonly [number, number, number] {
  const args = harness
    .callsFor('uniform3f')
    .find((call) => (call[0] as { name: string }).name === uniformName);
  if (!args) {
    throw new Error(`uniform ${uniformName} was never set`);
  }
  return [args[1] as number, args[2] as number, args[3] as number];
}

/** Reads the column-major matrix a `uniformMatrix4fv` call set for `uniformName`. */
function uniformMatrix4fvValue(
  harness: MockGlHarness,
  uniformName: string,
): readonly number[] {
  const args = harness
    .callsFor('uniformMatrix4fv')
    .find((call) => (call[0] as { name: string }).name === uniformName);
  if (!args) {
    throw new Error(`uniform ${uniformName} was never set`);
  }
  return Array.from(args[2] as Float32Array | number[]);
}

/** Builds a draw-state fixture with an identity view-projection matrix. */
function createDrawState(x: number, y: number, z: number): GridPassDrawState {
  return {
    viewProjection: new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]),
    cameraWorld: [x, y, z],
  };
}

describe('NeatRacing grid pass (AC-005)', () => {
  describe('spec-pinned constants', () => {
    it('pins the AC-005 spacing, alpha envelope, widths, distance, and plane offset', () => {
      expect({
        spacingWorld: NEATRACING_GRID_SPACING_WORLD,
        minAlpha: NEATRACING_GRID_MIN_ALPHA,
        maxAlpha: NEATRACING_GRID_MAX_ALPHA,
        lineWidthPx: NEATRACING_GRID_LINE_WIDTH_PX,
        glowWidthPx: NEATRACING_GRID_GLOW_WIDTH_PX,
        glowAlphaMultiplier: NEATRACING_GRID_GLOW_ALPHA_MULTIPLIER,
        drawDistanceM: NEATRACING_DRAW_DISTANCE_M,
        planeYOffsetM: NEATRACING_GRID_PLANE_Y_OFFSET_M,
        lineColorChannelsInRange:
          NEATRACING_GRID_LINE_COLOR_RGB.every(
            (channel) => channel > 0 && channel <= 1,
          ) && NEATRACING_GRID_LINE_COLOR_RGB[2] === 1,
      }).toEqual({
        spacingWorld: 1.0,
        minAlpha: 0.05,
        maxAlpha: 0.58,
        lineWidthPx: 2,
        glowWidthPx: 4,
        glowAlphaMultiplier: 0.8,
        drawDistanceM: 320,
        planeYOffsetM: -0.01,
        lineColorChannelsInRange: true,
      });
    });

    it('registers under the grid warmup key from the program registry', () => {
      expect(NEATRACING_GRID_PROGRAM_KEY).toBe('grid');
      expect(NEATRACING_WARMED_PROGRAM_KEYS).toContain(
        NEATRACING_GRID_PROGRAM_KEY,
      );
    });
  });

  describe('fwidth-derived line widths (pure mirror)', () => {
    it('scales both band thresholds by the fixture fwidth derivative doubles', () => {
      const core = FWIDTH_FIXTURES_WORLD.map((fwidthWorld) =>
        gridLineThresholdWorld(NEATRACING_GRID_LINE_WIDTH_PX, fwidthWorld),
      );
      const glow = FWIDTH_FIXTURES_WORLD.map((fwidthWorld) =>
        gridLineThresholdWorld(NEATRACING_GRID_GLOW_WIDTH_PX, fwidthWorld),
      );
      expect({ core, glow }).toEqual({
        core: [0.04, 0.016, 0.004],
        glow: [0.08, 0.032, 0.008],
      });
    });

    it('keeps the halo threshold exactly the core threshold times the width ratio at every fixture', () => {
      const ratios = FWIDTH_FIXTURES_WORLD.map((fwidthWorld) => {
        const core = gridLineThresholdWorld(
          NEATRACING_GRID_LINE_WIDTH_PX,
          fwidthWorld,
        );
        const glow = gridLineThresholdWorld(
          NEATRACING_GRID_GLOW_WIDTH_PX,
          fwidthWorld,
        );
        return glow / core;
      });
      expect(ratios).toEqual([
        NEATRACING_GRID_GLOW_WIDTH_PX / NEATRACING_GRID_LINE_WIDTH_PX,
        NEATRACING_GRID_GLOW_WIDTH_PX / NEATRACING_GRID_LINE_WIDTH_PX,
        NEATRACING_GRID_GLOW_WIDTH_PX / NEATRACING_GRID_LINE_WIDTH_PX,
      ]);
    });
  });

  describe('depth-fade alpha envelope', () => {
    it('fades line alpha 0.58 (near) -> 0.29 (mid) -> 0.05 (far floor, clamped beyond draw distance)', () => {
      expect({
        near: gridLineAlpha(NEAR_DISTANCE_M, NEATRACING_DRAW_DISTANCE_M),
        mid: gridLineAlpha(MID_DISTANCE_M, NEATRACING_DRAW_DISTANCE_M),
        far: gridLineAlpha(FAR_DISTANCE_M, NEATRACING_DRAW_DISTANCE_M),
        beyond: gridLineAlpha(
          NEATRACING_DRAW_DISTANCE_M * 2,
          NEATRACING_DRAW_DISTANCE_M,
        ),
      }).toEqual({
        near: NEATRACING_GRID_MAX_ALPHA,
        mid: 0.29,
        far: NEATRACING_GRID_MIN_ALPHA,
        beyond: NEATRACING_GRID_MIN_ALPHA,
      });
    });

    it('is monotonically non-increasing across the full fade sweep', () => {
      const violations: Array<{ at: number; drop: number }> = [];
      for (let distanceM = 0; distanceM <= 640; distanceM += 16) {
        const current = gridLineAlpha(distanceM, NEATRACING_DRAW_DISTANCE_M);
        const previous = gridLineAlpha(
          Math.max(0, distanceM - 16),
          NEATRACING_DRAW_DISTANCE_M,
        );
        if (current > previous) {
          violations.push({ at: distanceM, drop: current - previous });
        }
      }
      expect(violations).toEqual([]);
    });

    it('clamps the fog factor to [0, 1] inside and beyond the draw distance', () => {
      expect([
        gridFogFactor01(0, NEATRACING_DRAW_DISTANCE_M),
        gridFogFactor01(MID_DISTANCE_M, NEATRACING_DRAW_DISTANCE_M),
        gridFogFactor01(FAR_DISTANCE_M, NEATRACING_DRAW_DISTANCE_M),
        gridFogFactor01(
          NEATRACING_DRAW_DISTANCE_M * 2,
          NEATRACING_DRAW_DISTANCE_M,
        ),
        gridFogFactor01(-5, NEATRACING_DRAW_DISTANCE_M),
      ]).toEqual([0, 0.5, 1, 1, 0]);
    });
  });

  describe('world-anchored line phase', () => {
    it('derives axis-line distance from world coordinates only (periodic, symmetric, camera-independent)', () => {
      const fixtures: ReadonlyArray<readonly [number, number]> = [
        [0, 0],
        [0.5, 0.5],
        [1.5, 0.5],
        [2.25, 0.25],
        [1.25, 0.25],
        [-0.25, 0.25],
        [-1, 0],
        [320.5, 0.5],
      ];
      expect(
        fixtures.map(([worldCoord]) =>
          gridAxisLineDistance(worldCoord, NEATRACING_GRID_SPACING_WORLD),
        ),
      ).toEqual(fixtures.map(([, expected]) => expected));
    });
  });

  describe('band coverage and fragment alpha', () => {
    it('maps line distance to coverage endpoints 1 -> 0.5 -> 0 with a degenerate-threshold guard', () => {
      const coreThreshold = gridLineThresholdWorld(
        NEATRACING_GRID_LINE_WIDTH_PX,
        FWIDTH_FIXTURES_WORLD[1],
      );
      expect({
        atCenter: gridBandCoverage01(0, coreThreshold),
        atMidpoint: gridBandCoverage01(coreThreshold / 2, coreThreshold),
        atEdge: gridBandCoverage01(coreThreshold, coreThreshold),
        beyondEdge: gridBandCoverage01(coreThreshold * 2, coreThreshold),
        degenerateOffLine: gridBandCoverage01(0.5, 0),
        degenerateAtCenter: gridBandCoverage01(0, 0),
      }).toEqual({
        atCenter: 1,
        atMidpoint: 0.5,
        atEdge: 0,
        beyondEdge: 0,
        degenerateOffLine: 0,
        degenerateAtCenter: 1,
      });
    });

    it('renders full base alpha at a line center near the camera', () => {
      expect(
        gridFragmentAlpha(
          2,
          5,
          FWIDTH_FIXTURES_WORLD[0],
          NEAR_DISTANCE_M,
          NEATRACING_DRAW_DISTANCE_M,
        ),
      ).toBeCloseTo(NEATRACING_GRID_MAX_ALPHA, 12);
    });

    it('renders zero alpha mid-cell between lines', () => {
      expect(
        gridFragmentAlpha(
          2.5,
          5.5,
          FWIDTH_FIXTURES_WORLD[0],
          NEAR_DISTANCE_M,
          NEATRACING_DRAW_DISTANCE_M,
        ),
      ).toBeCloseTo(0, 12);
    });

    it('renders the halo-only band at the glow multiplier times band coverage', () => {
      // 2.024 sits 0.024 world units off the x line: outside the 0.016 core
      // threshold, inside the 0.032 glow threshold (t = 0.75 under fwidth
      // 0.008), so only the 0.8x halo contributes.
      expect(
        gridFragmentAlpha(
          2.024,
          5.5,
          FWIDTH_FIXTURES_WORLD[1],
          NEAR_DISTANCE_M,
          NEATRACING_DRAW_DISTANCE_M,
        ),
      ).toBeCloseTo(
        NEATRACING_GRID_MAX_ALPHA *
          NEATRACING_GRID_GLOW_ALPHA_MULTIPLIER *
          gridBandCoverage01(
            0.024,
            NEATRACING_GRID_GLOW_WIDTH_PX * FWIDTH_FIXTURES_WORLD[1],
          ),
        10,
      );
    });

    it('fades line centers along the depth-fade envelope at mid and far distances', () => {
      expect({
        mid: gridFragmentAlpha(
          3,
          4,
          FWIDTH_FIXTURES_WORLD[1],
          MID_DISTANCE_M,
          NEATRACING_DRAW_DISTANCE_M,
        ),
        far: gridFragmentAlpha(
          3,
          4,
          FWIDTH_FIXTURES_WORLD[2],
          FAR_DISTANCE_M,
          NEATRACING_DRAW_DISTANCE_M,
        ),
      }).toEqual({
        mid: 0.29,
        far: NEATRACING_GRID_MIN_ALPHA,
      });
    });
  });

  describe('GLSL source contract', () => {
    it('declares GLSL ES 3.00, uses fwidth, and never references gl.lineWidth', () => {
      expect({
        vertexVersionFirst:
          NEATRACING_GRID_VERTEX_SHADER.startsWith('#version 300 es'),
        fragmentVersionFirst:
          NEATRACING_GRID_FRAGMENT_SHADER.startsWith('#version 300 es'),
        fragmentUsesFwidth: NEATRACING_GRID_FRAGMENT_SHADER.includes('fwidth('),
        vertexFreeOfLineWidth:
          !NEATRACING_GRID_VERTEX_SHADER.includes('gl.lineWidth'),
        fragmentFreeOfLineWidth:
          !NEATRACING_GRID_FRAGMENT_SHADER.includes('gl.lineWidth'),
        fragmentUsesInvViewProjection:
          NEATRACING_GRID_FRAGMENT_SHADER.includes('uInvViewProjection'),
        vertexUsesVertexID:
          NEATRACING_GRID_VERTEX_SHADER.includes('gl_VertexID'),
        fragmentUsesCameraWorld:
          NEATRACING_GRID_FRAGMENT_SHADER.includes('uCameraWorld'),
        fragmentUsesLineColor:
          NEATRACING_GRID_FRAGMENT_SHADER.includes('uLineColor'),
        fragmentUsesDrawDistance:
          NEATRACING_GRID_FRAGMENT_SHADER.includes('uDrawDistance'),
        fragmentUsesAlphaEnvelope:
          NEATRACING_GRID_FRAGMENT_SHADER.includes('uMinAlpha') &&
          NEATRACING_GRID_FRAGMENT_SHADER.includes('uMaxAlpha') &&
          NEATRACING_GRID_FRAGMENT_SHADER.includes('uGlowAlphaMultiplier'),
        fragmentDiscardsUpwardRays:
          NEATRACING_GRID_FRAGMENT_SHADER.includes('discard'),
        vertexForwardsClipVaryingOnly:
          NEATRACING_GRID_VERTEX_SHADER.includes('out vec2 vClip') &&
          NEATRACING_GRID_VERTEX_SHADER.includes('vClip = clip'),
        vertexFreeOfProjectiveMath:
          !NEATRACING_GRID_VERTEX_SHADER.includes('uInvViewProjection') &&
          !NEATRACING_GRID_VERTEX_SHADER.includes('vWorldXZ') &&
          !NEATRACING_GRID_VERTEX_SHADER.includes('vHit'),
        fragmentUnprojectsPerFragment:
          NEATRACING_GRID_FRAGMENT_SHADER.includes(
            'uInvViewProjection * vec4(vClip',
          ) && NEATRACING_GRID_FRAGMENT_SHADER.includes('dir.y >= 0.0'),
      }).toEqual({
        vertexVersionFirst: true,
        fragmentVersionFirst: true,
        fragmentUsesFwidth: true,
        vertexFreeOfLineWidth: true,
        fragmentFreeOfLineWidth: true,
        fragmentUsesInvViewProjection: true,
        vertexUsesVertexID: true,
        fragmentUsesCameraWorld: true,
        fragmentUsesLineColor: true,
        fragmentUsesDrawDistance: true,
        fragmentUsesAlphaEnvelope: true,
        fragmentDiscardsUpwardRays: true,
        vertexForwardsClipVaryingOnly: true,
        vertexFreeOfProjectiveMath: true,
        fragmentUnprojectsPerFragment: true,
      });
    });
  });

  describe('GL interop (mock WebGL2)', () => {
    it('compiles and links the program exactly once at creation without geometry buffers', () => {
      const harness = createMockGl();
      createGridPass(harness.gl);
      expect({
        createShader: harness.callsFor('createShader').length,
        shaderSource: harness.callsFor('shaderSource').length,
        compileShader: harness.callsFor('compileShader').length,
        createProgram: harness.callsFor('createProgram').length,
        attachShader: harness.callsFor('attachShader').length,
        linkProgram: harness.callsFor('linkProgram').length,
        createBuffer: harness.callsFor('createBuffer').length,
        bufferData: harness.callsFor('bufferData').length,
        getAttribLocation: harness.callsFor('getAttribLocation').length,
        getUniformLocation: harness.callsFor('getUniformLocation').length,
      }).toEqual({
        createShader: 2,
        shaderSource: 2,
        compileShader: 2,
        createProgram: 1,
        attachShader: 2,
        linkProgram: 1,
        createBuffer: 0,
        bufferData: 0,
        getAttribLocation: 0,
        getUniformLocation: GRID_UNIFORM_COUNT,
      });
    });

    it('throws a descriptive error when a shader fails to compile', () => {
      const harness = createMockGl({ shaderCompileResult: false });
      expect(() => createGridPass(harness.gl)).toThrow(
        /vertex shader compile failed/,
      );
    });

    it('throws a descriptive error when the program fails to link', () => {
      const harness = createMockGl({ programLinkResult: false });
      expect(() => createGridPass(harness.gl)).toThrow(/link failed/);
    });

    it('sets the spec-constant uniforms from the draw state', () => {
      const harness = createMockGl();
      const pass = createGridPass(harness.gl);
      const state = createDrawState(12, 2.5, -7);
      pass.draw(state);
      expect({
        spacing: uniform1fValue(harness, 'uSpacing'),
        lineWidthPx: uniform1fValue(harness, 'uLineWidthPx'),
        glowWidthPx: uniform1fValue(harness, 'uGlowWidthPx'),
        glowAlphaMultiplier: uniform1fValue(harness, 'uGlowAlphaMultiplier'),
        minAlpha: uniform1fValue(harness, 'uMinAlpha'),
        maxAlpha: uniform1fValue(harness, 'uMaxAlpha'),
        drawDistance: uniform1fValue(harness, 'uDrawDistance'),
        planeY: uniform1fValue(harness, 'uPlaneY'),
        cameraWorld: uniform3fValue(harness, 'uCameraWorld'),
        lineColor: uniform3fValue(harness, 'uLineColor'),
        invViewProjectionIsIdentity: uniformMatrix4fvValue(
          harness,
          'uInvViewProjection',
        ),
      }).toEqual({
        spacing: 1.0,
        lineWidthPx: 2,
        glowWidthPx: 4,
        glowAlphaMultiplier: 0.8,
        minAlpha: 0.05,
        maxAlpha: 0.58,
        drawDistance: 320,
        planeY: -0.01,
        cameraWorld: [12, 2.5, -7],
        lineColor: [95 / 255, 1, 1],
        invViewProjectionIsIdentity: [
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        ],
      });
    });

    it('draws a blended, depth-neutral full-screen triangle without vertex attributes', () => {
      const harness = createMockGl();
      const pass = createGridPass(harness.gl);
      pass.draw(createDrawState(0, 10, 20));
      expect({
        blendEnabled: harness
          .callsFor('enable')
          .some((call) => call[0] === MOCK_GL_BLEND),
        premultipliedFreeBlend: harness
          .callsFor('blendFunc')
          .some(
            (call) =>
              call[0] === MOCK_GL_SRC_ALPHA &&
              call[1] === MOCK_GL_ONE_MINUS_SRC_ALPHA,
          ),
        depthTestDisabled: harness
          .callsFor('disable')
          .some((call) => call[0] === MOCK_GL_DEPTH_TEST),
        depthWritesOff: harness
          .callsFor('depthMask')
          .some((call) => call[0] === false),
        fullscreenTriangleDraw: harness
          .callsFor('drawArrays')
          .some(
            (call) =>
              call[0] === MOCK_GL_TRIANGLES && call[1] === 0 && call[2] === 3,
          ),
        noVertexAttribArray:
          harness.callsFor('enableVertexAttribArray').length === 0,
        noVertexAttribPointer:
          harness.callsFor('vertexAttribPointer').length === 0,
        noBufferBound: harness.callsFor('bindBuffer').length === 0,
      }).toEqual({
        blendEnabled: true,
        premultipliedFreeBlend: true,
        depthTestDisabled: true,
        depthWritesOff: true,
        fullscreenTriangleDraw: true,
        noVertexAttribArray: true,
        noVertexAttribPointer: true,
        noBufferBound: true,
      });
    });

    it('never recompiles or re-uploads across repeated draws', () => {
      const harness = createMockGl();
      const pass = createGridPass(harness.gl);
      pass.draw(createDrawState(0, 10, 20));
      pass.draw(createDrawState(40, 10, -120));
      expect({
        createShader: harness.callsFor('createShader').length,
        bufferData: harness.callsFor('bufferData').length,
        useProgram: harness.callsFor('useProgram').length,
        drawArrays: harness.callsFor('drawArrays').length,
      }).toEqual({
        createShader: 2,
        bufferData: 0,
        useProgram: 2,
        drawArrays: 2,
      });
    });

    it('rejects a malformed view-projection matrix', () => {
      const harness = createMockGl();
      const pass = createGridPass(harness.gl);
      const badState = {
        viewProjection: new Float32Array(15),
        cameraWorld: [0, 10, 20],
      } as GridPassDrawState;
      expect(() => pass.draw(badState)).toThrow(/viewProjection/);
    });
  });
});

describe('shader warmup seam (AC-040)', () => {
  it('exposes the warmup program under the grid registry key', () => {
    const harness = createMockGl();

    const warmup = toGridWarmupProgram(harness.gl);

    expect(warmup.key).toBe(NEATRACING_WARMED_PROGRAM_KEYS[1]);
  });

  it('draws the warmed pipeline offscreen through the shared pass draw path', () => {
    const harness = createMockGl();
    const warmup = toGridWarmupProgram(harness.gl);
    warmup.compileAndLink();
    const drawsBeforeWarmup = harness.callsFor('drawArrays').length;

    warmup.drawOffscreen(harness.gl);

    expect(harness.callsFor('drawArrays').length - drawsBeforeWarmup).toBe(1);
  });
});
