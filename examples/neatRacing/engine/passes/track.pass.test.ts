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
  NEATRACING_TRACK_PROGRAM_KEY,
  type TrackPass,
  type TrackPassDrawState,
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
    getUniformLocation: (_program: unknown, name: string): object => ({ name }),
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
    uniform3f: (): void => undefined,
    uniform1f: (): void => undefined,
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
});
