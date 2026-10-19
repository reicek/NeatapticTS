/**
 * Contract tests for the NeatRacing track grid (slice P1-grid-footprint;
 * AC-045): the track grid cell footprint area is enlarged versus the Step 14B
 * / engine ground-grid baseline while the flat-horizon draw semantics and the
 * unrelated pulse pass stay unchanged.
 *
 * The engine ground grid pins `NEATRACING_GRID_SPACING_WORLD = 1.0`, so a
 * 1 m x 1 m baseline cell has area 1 m². User visual feedback (2026-10-12)
 * requested 5 m cells, so the multiplier is 25, giving a linear spacing of
 * 5.0 m. The track grid pass reuses the same flat-horizon fullscreen shader as
 * the engine grid but feeds it the larger spacing uniform.
 */

import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_TRACK_GRID_BASELINE_CELL_FOOTPRINT_AREA_M2,
  NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_MULTIPLIER,
  NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_M2,
  NEATRACING_TRACK_GRID_PROGRAM_KEY,
  NEATRACING_TRACK_GRID_SPACING_M,
  buildTrackGridSpacing,
  createTrackGridPass,
  trackGridLineCountForExtent,
} from './grid.pass';

/** `gl.TRIANGLES` enum value (0x0004) used by the fullscreen grid draw. */
const MOCK_GL_TRIANGLES = 0x0004;

/** `gl.BLEND` enum value (0x0BE2) asserted on the mock draw state. */
const MOCK_GL_BLEND = 0x0be2;

/** `gl.SRC_ALPHA` enum value (0x0302) asserted on the mock blend function. */
const MOCK_GL_SRC_ALPHA = 0x0302;

/** `gl.ONE_MINUS_SRC_ALPHA` enum value (0x0303) for the blend function. */
const MOCK_GL_ONE_MINUS_SRC_ALPHA = 0x0303;

/** `gl.DEPTH_TEST` enum value (0x0B71) asserted disabled for flat horizon. */
const MOCK_GL_DEPTH_TEST = 0x0b71;

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
    BLEND: MOCK_GL_BLEND,
    SRC_ALPHA: MOCK_GL_SRC_ALPHA,
    ONE_MINUS_SRC_ALPHA: MOCK_GL_ONE_MINUS_SRC_ALPHA,
    DEPTH_TEST: MOCK_GL_DEPTH_TEST,
    getUniformLocation(_program: unknown, name: string): unknown {
      log.push({ op: 'getUniformLocation', args: [name] });
      return { name };
    },
    useProgram: record('useProgram'),
    enable: record('enable'),
    disable: record('disable'),
    depthMask: record('depthMask'),
    blendFunc: record('blendFunc'),
    uniform1f: record('uniform1f'),
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

/** Builds a draw-state fixture with an identity view-projection matrix. */
function createDrawState(): {
  readonly viewProjection: Float32Array;
  readonly cameraWorld: readonly [number, number, number];
} {
  return {
    viewProjection: new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]),
    cameraWorld: [0, 10, 20] as const,
  };
}

describe('NeatRacing track grid (AC-045)', () => {
  describe('cell footprint constants', () => {
    it('uses a 25 m² cell footprint area versus the 1 m² Step 14B baseline', () => {
      expect({
        baselineAreaM2: NEATRACING_TRACK_GRID_BASELINE_CELL_FOOTPRINT_AREA_M2,
        multiplier: NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_MULTIPLIER,
        targetAreaM2: NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_M2,
        spacingM: NEATRACING_TRACK_GRID_SPACING_M,
      }).toEqual({
        baselineAreaM2: 1,
        multiplier: 25,
        targetAreaM2: 25,
        spacingM: 5,
      });
    });

    it('derives the same spacing from the builder helper', () => {
      expect(buildTrackGridSpacing()).toBe(5);
    });
  });

  describe('line-count scaling for a fixed world extent', () => {
    it('produces five times fewer lines than the 1 m baseline', () => {
      const extentM = 100;
      const baselineCount = trackGridLineCountForExtent(extentM, 1.0);
      const trackCount = trackGridLineCountForExtent(
        extentM,
        buildTrackGridSpacing(),
      );
      expect(trackCount).toBe(Math.round(extentM / 5));
      expect(baselineCount / trackCount).toBeCloseTo(5, 1);
    });
  });

  describe('track grid pass WebGL contract', () => {
    it('registers under a distinct track-grid warmup key', () => {
      expect(NEATRACING_TRACK_GRID_PROGRAM_KEY).toBe('track-grid');
    });

    it('compiles the engine flat-horizon shaders and sets the 5.0 m spacing uniform', () => {
      const harness = createMockGl();
      const pass = createTrackGridPass(harness.gl);
      pass.draw(createDrawState());

      const spacingUniform = uniform1fValue(harness, 'uSpacing');
      expect(spacingUniform).toBe(5);

      const shaderSources = harness
        .callsFor('shaderSource')
        .map((call) => call[1] as string);
      const vertexSource = shaderSources[0];
      const fragmentSource = shaderSources[1];
      expect(vertexSource).toContain('#version 300 es');
      expect(fragmentSource).toContain('#version 300 es');
      expect(fragmentSource).toContain('fwidth(');
      expect(vertexSource).not.toContain('gl.lineWidth');
      expect(fragmentSource).not.toContain('gl.lineWidth');
    });

    it('draws a fullscreen triangle with blending enabled and depth disabled (flat horizon)', () => {
      const harness = createMockGl();
      const pass = createTrackGridPass(harness.gl);
      pass.draw(createDrawState());

      expect(harness.callsFor('enable')).toContainEqual([MOCK_GL_BLEND]);
      expect(harness.callsFor('disable')).toContainEqual([MOCK_GL_DEPTH_TEST]);
      expect(harness.callsFor('depthMask')).toContainEqual([false]);
      expect(harness.callsFor('blendFunc')).toContainEqual([
        MOCK_GL_SRC_ALPHA,
        MOCK_GL_ONE_MINUS_SRC_ALPHA,
      ]);
      expect(harness.callsFor('drawArrays')).toContainEqual([
        MOCK_GL_TRIANGLES,
        0,
        3,
      ]);
    });

    it('rejects a malformed view-projection matrix', () => {
      const harness = createMockGl();
      const pass = createTrackGridPass(harness.gl);
      expect(() =>
        pass.draw({
          viewProjection: new Float32Array([1, 0, 0]),
          cameraWorld: [0, 10, 20],
        }),
      ).toThrow(/viewProjection must be a 16-element column-major matrix/);
    });
  });
});
