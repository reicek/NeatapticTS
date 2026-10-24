/**
 * Contract tests for the NeatRacing car pass visual drift-yaw offset
 * (slice P1-arcade-drift-visual; AC-047C) and for the attached tires that
 * follow the body during a drift (slice P1-drift-tire-follow; AC-078).
 *
 * The physics step (sibling slice P1-arcade-drift-model) already stores the
 * visual drift offset in `PhysicsCarState.driftYawOffset`. The pass consumes
 * that offset on the car body transform, and because the tires are ATTACHED
 * to the body, the wheel chain is parented onto the SAME drift-yawed matrix
 * (AC-078), so the rendered wheel transform carries the same drift yaw as
 * the body. No per-frame `Math.atan2` is reintroduced here.
 *
 * The tests drive the pass against a hand-rolled WebGL2 recording mock and
 * inspect the `uModel` matrices uploaded during the lit draw phase.
 */
import { describe, expect, it } from '@jest/globals';
import { VOXEL_EMPTY_PALETTE_ENTRY } from '../../voxel/model.schema';
import type { VoxelBinModel, VoxelBinQuad } from '../../voxel/bin.codec';
import {
  resolveVoxelWheelMountTransform,
  resolveVoxelWheelSockets,
  type VoxelWheelMountTransform,
} from '../../voxel/wheel.mount.utils';
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

/** Column-major rotation around the model depth axis (the pre-mount spin axis). */
function rotationZMat4(out: Float32Array, angleRad: number): void {
  identityMat4(out);
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  out[0] = cos;
  out[1] = sin;
  out[4] = -sin;
  out[5] = cos;
}

/**
 * Expected wheel model matrix for one socket =
 * `T(x,0,z) * R(yaw + drift) * T(mount) * R(mount yaw) * R(steer) * R(spin)`.
 * The drift offset is part of the car yaw the tire is mounted on, so the
 * rendered tire inherits it.
 */
function expectedWheelMatrix(
  worldX: number,
  worldZ: number,
  yawRad: number,
  driftOffsetRad: number,
  mount: VoxelWheelMountTransform,
  steerRad: number,
  spinRad: number,
): readonly number[] {
  const a = new Float32Array(16);
  const b = new Float32Array(16);
  const out = new Float32Array(16);
  translationMat4(a, worldX, 0, worldZ);
  rotationYMat4(b, yawRad + driftOffsetRad);
  multiplyMat4(out, a, b);
  translationMat4(a, mount.position[0], mount.position[1], mount.position[2]);
  multiplyMat4(b, out, a);
  rotationYMat4(a, mount.rotationYawRad);
  multiplyMat4(out, b, a);
  rotationYMat4(a, steerRad);
  multiplyMat4(b, out, a);
  rotationZMat4(a, spinRad);
  multiplyMat4(out, b, a);
  return Array.from(out);
}

/**
 * Extracts the yaw around world up from a column-major model matrix using
 * its third column (the model's local +Z axis). The wheel chain ends with a
 * spin rotation about the model depth axis, which leaves that column
 * untouched, so the extracted yaw is the car yaw plus the mount yaw.
 */
function yawFromModelColumn(model: readonly number[]): number {
  return Math.atan2(model[8], model[10]);
}

/** Wraps an angle difference into the (-pi, pi] range. */
function wrapAngle(angleRad: number): number {
  const twoPi = 2 * Math.PI;
  const wrapped = (((angleRad + Math.PI) % twoPi) + twoPi) % twoPi;
  return wrapped - Math.PI;
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

describe('NeatRacing car pass visual drift yaw (AC-047C + AC-078)', () => {
  it('applies driftYawOffsetRad to the car body uModel matrix', () => {
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

  it('parents the attached wheel chain onto the drift-yawed body matrix', () => {
    const harness = createMockGl();
    const carPass = createCarPass(harness.gl, TEST_MODELS);
    const state: CarPassDrawState = {
      viewProjection: TEST_VIEW_PROJECTION,
      cameraWorld: [0, 5, 0],
      cars: [
        {
          worldX: 2,
          worldZ: -1,
          yawRad: 0.4,
          teamColorId: 0,
          wheelSpinRad: 1.25,
          driftYawOffsetRad: 0.5,
        },
      ],
    };
    carPass.draw(state);

    const litMatrices = harness.modelMatricesForLitPass();
    expect(litMatrices.length).toBeGreaterThanOrEqual(5);
    const bodyYawRad = yawFromModelColumn(litMatrices[0]);
    expect(bodyYawRad).toBeCloseTo(0.4 + 0.5, 6);

    const mounts = resolveVoxelWheelSockets().map((socket) =>
      resolveVoxelWheelMountTransform(socket.id),
    );
    for (let index = 0; index < mounts.length; index += 1) {
      const mount = mounts[index];
      const wheelMatrix = litMatrices[index + 1];
      // The rendered wheel transform carries the SAME drift yaw as the body:
      // the tire's own mount yaw is the only difference between the two.
      expect(
        wrapAngle(yawFromModelColumn(wheelMatrix) - bodyYawRad),
      ).toBeCloseTo(mount.rotationYawRad, 6);
      // The tire's mount position orbits with the drifting body, so the tire
      // is still attached to the body instead of sliding off it. Precision 6
      // absorbs the float32 rounding of the composed chain.
      expectMatricesClose(
        wheelMatrix,
        expectedWheelMatrix(2, -1, 0.4, 0.5, mount, 0, 1.25),
        6,
      );
    }
  });

  it('keeps the attached wheel chain on the base yaw when the drift offset is zero', () => {
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
    const mounts = resolveVoxelWheelSockets().map((socket) =>
      resolveVoxelWheelMountTransform(socket.id),
    );
    for (let index = 0; index < mounts.length; index += 1) {
      expectMatricesClose(
        litMatrices[index + 1],
        expectedWheelMatrix(-2, 4, 1.1, 0, mounts[index], 0, 0),
        6,
      );
    }
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
