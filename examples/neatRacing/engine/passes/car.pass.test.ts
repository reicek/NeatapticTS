/**
 * Green-phase contract tests for the NeatRacing car pass visual drift-yaw
 * offset (slice P1-arcade-drift-visual; AC-047C).
 *
 * The physics step (sibling slice P1-arcade-drift-model) already stores the
 * visual drift offset in `PhysicsCarState.driftYawOffset`. This slice only
 * consumes that offset and adds it to the car *body* transform; wheels and
 * blob shadow stay aligned to the base yaw. No per-frame `Math.atan2` is
 * reintroduced here.
 *
 * The tests drive the pass against a hand-rolled WebGL2 recording mock and
 * inspect the `uModel` matrices uploaded during the lit draw phase.
 */
import { describe, expect, it } from '@jest/globals';
import { VOXEL_EMPTY_PALETTE_ENTRY } from '../../voxel/model.schema';
import type { VoxelBinModel, VoxelBinQuad } from '../../voxel/bin.codec';
import {
  NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
  NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD,
} from '../../physics/drift.model';
import { createCarPass, type CarPassDrawState } from './car.pass';

const MOCK_GL_TRIANGLES = 0x0004;
const MOCK_GL_BLEND = 0x0be2;
const MOCK_GL_DEPTH_TEST = 0x0b71;
const MOCK_GL_SRC_ALPHA = 0x0302;
const MOCK_GL_ONE_MINUS_SRC_ALPHA = 0x0303;
const MOCK_GL_ARRAY_BUFFER = 0x8892;
const MOCK_GL_STATIC_DRAW = 0x88e4;
const MOCK_GL_VERTEX_SHADER = 0x8b31;
const MOCK_GL_FRAGMENT_SHADER = 0x8b30;
const MOCK_GL_FLOAT = 0x1406;
const MOCK_GL_COMPILE_STATUS = 0x8b81;
const MOCK_GL_LINK_STATUS = 0x8b82;

/** One recorded mock-GL call. */
interface RecordedGlCall {
  readonly op: string;
  readonly args: readonly unknown[];
}

/** Hand-rolled WebGL2 recording harness. */
interface MockGlHarness {
  readonly gl: WebGL2RenderingContext;
  callsFor(op: string): ReadonlyArray<readonly unknown[]>;
  /** Returns every `uModel` matrix uploaded after the last `useProgram`. */
  modelMatricesForLitPass(): readonly number[][];
}

/**
 * Creates a hand-rolled recording mock of a `WebGL2RenderingContext`.
 * Uniform locations carry their own name so calls can be attributed back to
 * specific uniforms without real GPU/DOM dependencies.
 */
function createMockGl(): MockGlHarness {
  const log: RecordedGlCall[] = [];
  const record =
    (op: string) =>
    (...args: unknown[]): void => {
      log.push({ op, args });
    };
  const recordUniformMatrix4fv =
    () =>
    (location: unknown, transpose: unknown, matrix: unknown): void => {
      // Clone the Float32Array at record time so later draw-path reuse of
      // module-scope scratch buffers cannot rewrite history in the log.
      const clone =
        matrix instanceof Float32Array
          ? new Float32Array(matrix)
          : Array.isArray(matrix)
            ? [...(matrix as readonly unknown[])]
            : matrix;
      log.push({ op: 'uniformMatrix4fv', args: [location, transpose, clone] });
    };
  const shaderHandle = { kind: 'shader' };
  const programHandle = { kind: 'program' };
  const bufferHandle = { kind: 'buffer' };
  const gl = {
    VERTEX_SHADER: MOCK_GL_VERTEX_SHADER,
    FRAGMENT_SHADER: MOCK_GL_FRAGMENT_SHADER,
    ARRAY_BUFFER: MOCK_GL_ARRAY_BUFFER,
    STATIC_DRAW: MOCK_GL_STATIC_DRAW,
    FLOAT: MOCK_GL_FLOAT,
    COMPILE_STATUS: MOCK_GL_COMPILE_STATUS,
    LINK_STATUS: MOCK_GL_LINK_STATUS,
    TRIANGLES: MOCK_GL_TRIANGLES,
    BLEND: MOCK_GL_BLEND,
    DEPTH_TEST: MOCK_GL_DEPTH_TEST,
    SRC_ALPHA: MOCK_GL_SRC_ALPHA,
    ONE_MINUS_SRC_ALPHA: MOCK_GL_ONE_MINUS_SRC_ALPHA,
    createShader(): unknown {
      log.push({ op: 'createShader', args: [] });
      return shaderHandle;
    },
    shaderSource: record('shaderSource'),
    compileShader: record('compileShader'),
    getShaderParameter(): boolean {
      log.push({ op: 'getShaderParameter', args: [] });
      return true;
    },
    getShaderInfoLog(): string {
      return 'mock shader info log';
    },
    deleteShader: record('deleteShader'),
    createProgram(): unknown {
      log.push({ op: 'createProgram', args: [] });
      return programHandle;
    },
    attachShader: record('attachShader'),
    linkProgram: record('linkProgram'),
    getProgramParameter(): boolean {
      log.push({ op: 'getProgramParameter', args: [] });
      return true;
    },
    getProgramInfoLog(): string {
      return 'mock program info log';
    },
    deleteProgram: record('deleteProgram'),
    detachShader: record('detachShader'),
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
    uniformMatrix4fv: recordUniformMatrix4fv(),
    drawArrays: record('drawArrays'),
  } as unknown as WebGL2RenderingContext;
  return {
    gl,
    callsFor(op: string): ReadonlyArray<readonly unknown[]> {
      return log.filter((call) => call.op === op).map((call) => call.args);
    },
    modelMatricesForLitPass(): readonly number[][] {
      let lastUseProgramIndex = -1;
      for (let i = 0; i < log.length; i += 1) {
        if (log[i].op === 'useProgram') {
          lastUseProgramIndex = i;
        }
      }
      if (lastUseProgramIndex < 0) {
        throw new Error('no useProgram call recorded');
      }
      const matrices: number[][] = [];
      for (let i = lastUseProgramIndex + 1; i < log.length; i += 1) {
        if (log[i].op !== 'uniformMatrix4fv') {
          continue;
        }
        const args = log[i].args;
        const location = args[0] as { readonly name: string };
        const value = args[2] as Float32Array | readonly number[];
        if (location.name === 'uModel') {
          matrices.push(Array.from(value));
        }
      }
      return matrices;
    },
  };
}

/** 16-element identity matrix (column-major). */
function identityMat4(out: Float32Array): void {
  out.fill(0);
  out[0] = 1;
  out[5] = 1;
  out[10] = 1;
  out[15] = 1;
}

/** Column-major translation matrix. */
function translationMat4(
  out: Float32Array,
  x: number,
  y: number,
  z: number,
): void {
  identityMat4(out);
  out[12] = x;
  out[13] = y;
  out[14] = z;
}

/** Column-major rotation around world up (Y axis). */
function rotationYMat4(out: Float32Array, angleRad: number): void {
  identityMat4(out);
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  out[0] = cos;
  out[2] = -sin;
  out[8] = sin;
  out[10] = cos;
}

/** Column-major matrix multiply `out = a * b`. */
function multiplyMat4(
  out: Float32Array,
  a: Float32Array,
  b: Float32Array,
): void {
  for (let column = 0; column < 4; column += 1) {
    const b0 = b[column * 4 + 0];
    const b1 = b[column * 4 + 1];
    const b2 = b[column * 4 + 2];
    const b3 = b[column * 4 + 3];
    for (let row = 0; row < 4; row += 1) {
      out[column * 4 + row] =
        a[row] * b0 +
        a[4 + row] * b1 +
        a[8 + row] * b2 +
        a[12 + row] * b3;
    }
  }
}

/** Expected car body model matrix = T(x,0,z) * R(yaw + offset). */
function expectedBodyMatrix(
  worldX: number,
  worldZ: number,
  yawRad: number,
  driftOffsetRad: number,
): readonly number[] {
  const t = new Float32Array(16);
  const r = new Float32Array(16);
  const out = new Float32Array(16);
  translationMat4(t, worldX, 0, worldZ);
  rotationYMat4(r, yawRad + driftOffsetRad);
  multiplyMat4(out, t, r);
  return Array.from(out);
}

/** Asserts two 16-element matrices are element-wise close. */
function expectMatricesClose(
  received: readonly number[],
  expected: readonly number[],
  precision = 7,
): void {
  expect(received).toHaveLength(16);
  expect(expected).toHaveLength(16);
  for (let i = 0; i < 16; i += 1) {
    expect(received[i]).toBeCloseTo(expected[i], precision);
  }
}

/** Minimal car body/wheel bins used only for the mock pipeline. */
const TEST_PALETTE: VoxelBinModel['palette'] = [
  [...VOXEL_EMPTY_PALETTE_ENTRY],
  [180, 40, 40, 255, 0],
];

const TEST_BODY_QUAD: VoxelBinQuad = {
  position: [0, 0, 0],
  normalAxis: 0,
  width: 2,
  height: 2,
  paletteIndex: 1,
};

const TEST_WHEEL_QUAD: VoxelBinQuad = {
  position: [0, 0, 0],
  normalAxis: 1,
  width: 2,
  height: 2,
  paletteIndex: 1,
};

const TEST_BODY_BIN: VoxelBinModel = {
  dims: [2, 2, 2],
  palette: TEST_PALETTE,
  quads: [TEST_BODY_QUAD],
};

const TEST_WHEEL_BIN: VoxelBinModel = {
  dims: [2, 2, 2],
  palette: TEST_PALETTE,
  quads: [TEST_WHEEL_QUAD],
};

const TEST_MODELS = { body: TEST_BODY_BIN, wheel: TEST_WHEEL_BIN };

/** Identity view-projection fixture. */
const TEST_VIEW_PROJECTION = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
]);

/** Natural absolute visual-yaw limit from the drift model constants. */
const DRIFT_VISUAL_YAW_LIMIT_RAD =
  NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD;

describe('NeatRacing car pass visual drift yaw (AC-047C)', () => {
  it('applies driftYawOffsetRad only to the body uModel matrix', () => {
    const harness = createMockGl();
    const carPass = createCarPass(harness.gl, TEST_MODELS);
    const state: CarPassDrawState = {
      viewProjection: TEST_VIEW_PROJECTION,
      cameraWorld: [0, 5, 0],
      cars: [
        {
          worldX: 3,
          worldZ: 5,
          yawRad: 0.7,
          teamColorId: 0,
          wheelSpinRad: 0,
          driftYawOffsetRad: 0.35,
        },
      ],
    };
    carPass.draw(state);

    const litMatrices = harness.modelMatricesForLitPass();
    expect(litMatrices.length).toBeGreaterThanOrEqual(5);
    const bodyMatrix = litMatrices[0];
    expectMatricesClose(bodyMatrix, expectedBodyMatrix(3, 5, 0.7, 0.35));
  });

  it('keeps wheels and shadow using the base yaw when a drift offset is present', () => {
    const offsetHarness = createMockGl();
    const zeroHarness = createMockGl();
    const baseCar = {
      worldX: 2,
      worldZ: -1,
      yawRad: 0.4,
      teamColorId: 0,
      wheelSpinRad: 0,
    };
    const offsetPass = createCarPass(offsetHarness.gl, TEST_MODELS);
    const zeroPass = createCarPass(zeroHarness.gl, TEST_MODELS);
    const baseState = (driftYawOffsetRad: number): CarPassDrawState => ({
      viewProjection: TEST_VIEW_PROJECTION,
      cameraWorld: [0, 5, 0],
      cars: [{ ...baseCar, driftYawOffsetRad }],
    });

    offsetPass.draw(baseState(0.5));
    zeroPass.draw(baseState(0));

    const offsetLit = offsetHarness.modelMatricesForLitPass();
    const zeroLit = zeroHarness.modelMatricesForLitPass();
    // Lit phase: body first, then four wheels. Wheels must be identical
    // across the two frames because they ignore the drift offset.
    expect(offsetLit.length).toBeGreaterThanOrEqual(5);
    expect(zeroLit.length).toBeGreaterThanOrEqual(5);
    for (let i = 0; i < 4; i += 1) {
      expectMatricesClose(offsetLit[i + 1], zeroLit[i + 1]);
    }

    // Body matrices must differ because only the body consumes the offset.
    expect(offsetLit[0]).not.toEqual(zeroLit[0]);

    // The wheel matrices must be the same as the no-offset body matrix would
    // be (the wheels mount off the base car model matrix).
    expectMatricesClose(zeroLit[0], expectedBodyMatrix(2, -1, 0.4, 0));
  });

  it('zero drift offset leaves the body matrix equal to the base-yaw matrix', () => {
    const harness = createMockGl();
    const carPass = createCarPass(harness.gl, TEST_MODELS);
    const state: CarPassDrawState = {
      viewProjection: TEST_VIEW_PROJECTION,
      cameraWorld: [0, 5, 0],
      cars: [
        {
          worldX: -2,
          worldZ: 4,
          yawRad: 1.1,
          teamColorId: 0,
          wheelSpinRad: 0,
          driftYawOffsetRad: 0,
        },
      ],
    };
    carPass.draw(state);

    const litMatrices = harness.modelMatricesForLitPass();
    expectMatricesClose(litMatrices[0], expectedBodyMatrix(-2, 4, 1.1, 0));
  });

  it('maps drift offset monotonically within the natural slip-angle bounds', () => {
    const offsets = [
      -DRIFT_VISUAL_YAW_LIMIT_RAD,
      -DRIFT_VISUAL_YAW_LIMIT_RAD / 2,
      0,
      DRIFT_VISUAL_YAW_LIMIT_RAD / 2,
      DRIFT_VISUAL_YAW_LIMIT_RAD,
    ];
    const extractedYaws: number[] = [];
    for (const driftYawOffsetRad of offsets) {
      const harness = createMockGl();
      const carPass = createCarPass(harness.gl, TEST_MODELS);
      const state: CarPassDrawState = {
        viewProjection: TEST_VIEW_PROJECTION,
        cameraWorld: [0, 5, 0],
        cars: [
          {
            worldX: 0,
            worldZ: 0,
            yawRad: 0,
            teamColorId: 0,
            wheelSpinRad: 0,
            driftYawOffsetRad,
          },
        ],
      };
      carPass.draw(state);
      const bodyMatrix = harness.modelMatricesForLitPass()[0];
      const sin = bodyMatrix[8];
      const cos = bodyMatrix[0];
      extractedYaws.push(Math.atan2(sin, cos));
    }

    for (let i = 1; i < extractedYaws.length; i += 1) {
      expect(extractedYaws[i]).toBeGreaterThan(extractedYaws[i - 1]);
    }
    expect(extractedYaws[0]).toBeCloseTo(-DRIFT_VISUAL_YAW_LIMIT_RAD, 10);
    expect(extractedYaws[extractedYaws.length - 1]).toBeCloseTo(
      DRIFT_VISUAL_YAW_LIMIT_RAD,
      10,
    );
  });
});
