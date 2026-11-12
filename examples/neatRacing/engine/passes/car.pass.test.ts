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
 *
 * RED-14R (slice P1-drift-presentation-red, AC-088): the car pass must
 * CONSUME the PUBLISHED traction severity - the ONE shared scalar published
 * by Step 14Q - instead of presenting the fixed-sun lighting band and the
 * raw steer echo. Four RED contracts are authored here, and all of them
 * are observable through the recording mock alone (no browser, no CDP):
 *
 * 1. [RED-14R-CAR-TRACTION-CONSUMED] the SAME pose rendered with a
 *    DIFFERENT published severity must feed DIFFERENT uniforms. The
 *    severity is a quantity the player does NOT control, so the only way
 *    the response can exist is the physics reaching the screen.
 * 2. [RED-14R-CAR-PER-TIRE-GRIP] the four tires must NOT share ONE grip:
 *    at least one scalar/vector uniform fed for the wheels must be fed
 *    with more than one distinct value (the rears breaking away).
 * 3. [RED-14R-CAR-STEER-FROM-PHYSICS] the rendered front-wheel steer yaw
 *    must be driven by the PUBLISHED physics state, not by the raw steer
 *    echo: the SAME echo with a DIFFERENT published severity must render
 *    a DIFFERENT front-wheel yaw.
 * 4. [RED-14R-CAR-STEER-NOT-ECHO] the rendered front-wheel steer yaw
 *    must NOT be the raw steer echo while the tires are sliding, so the
 *    steering cannot be a lip-service echo of the input.
 *
 * RED DECISIONS (the shape the green slice must honour):
 * - the `CarPassCarState` traction channel is NAMED `tractionSeverity` -
 *   the SAME name the physics state and the browser seam use - and it
 *   carries the PUBLISHED severity in [0, 1], never a re-derived one;
 * - the channel is OPTIONAL in the production type (like the existing
 *   `steerAngleRad` / `driftYawOffsetRad` visual channels), because the
 *   out-of-scope producers of `CarPassCarState` (`editor/editor-entry.ts`
 *   `PREVIEW_CARS`, `host/frame-composer.ts`) are NOT owned by the green
 *   slice and must keep compiling. This suite always carries the channel
 *   explicitly, so a missing severity is never silently green here;
 * - the visible drift yaw is bounded by the PUBLISHED pose envelope
 *   (`NEATRACING_DRIFT_POSE_ENVELOPE_RAD` = 0.30185 rad), NOT by the
 *   retired committed-slip product (`MAX_SLIP * GAIN` = 0.2025 rad). The
 *   old pin is DELETED in this same change and REPLACED by the
 *   published-slip contract - never weakened.
 *
 * RED PREMISE: the car pass has NO traction channel today - it feeds the
 * fixed-sun lighting band and echoes the raw steer input straight into the
 * front-wheel matrix - so every fixture below renders IDENTICALLY whatever
 * the published severity is. The four contracts therefore fail on their
 * ASSERTIONS (not on a compile error), each with a named expectation, and
 * the green slice turns them green by consuming the named channel. The
 * channel NAME is pinned by the BEHAVIOUR (the fixture type is
 * `CarPassCarState & { tractionSeverity: number }`), never by a
 * placeholder export.
 */
import { describe, expect, it } from '@jest/globals';
import { VOXEL_EMPTY_PALETTE_ENTRY } from '../../voxel/model.schema';
import type { VoxelBinModel, VoxelBinQuad } from '../../voxel/bin.codec';
import {
  resolveVoxelWheelMountTransform,
  resolveVoxelWheelSockets,
  type VoxelWheelMountTransform,
} from '../../voxel/wheel.mount.utils';
import { NEATRACING_DRIFT_POSE_ENVELOPE_RAD } from '../../physics/drift.model';
import {
  createCarPass,
  type CarPassCarState,
  type CarPassDrawState,
} from './car.pass';

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

/**
 * One recorded scalar/vector uniform feed in the lit pass: the uniform
 * name and the values it was fed with. The matrix uploads (`uModel`,
 * `uViewProjection`) are GEOMETRY, not shader inputs, so they are not
 * feeds.
 */
interface RecordedUniformFeed {
  readonly name: string;
  readonly values: readonly number[];
}

/**
 * One draw group of the lit pass: the uniform feeds recorded between two
 * `drawArrays` calls (the group before the first `drawArrays` carries the
 * lighting uniforms and the car body feed).
 */
interface RecordedDrawGroup {
  readonly feeds: readonly RecordedUniformFeed[];
}

/** Hand-rolled WebGL2 recording harness. */
interface MockGlHarness {
  readonly gl: WebGL2RenderingContext;
  callsFor(op: string): ReadonlyArray<readonly unknown[]>;
  /** Returns every `uModel` matrix uploaded after the last `useProgram`. */
  modelMatricesForLitPass(): readonly number[][];
  /**
   * Returns the lit-pass draw groups: the scalar/vector uniform feeds
   * recorded between two `drawArrays` calls after the last `useProgram`.
   */
  litDrawGroups(): readonly RecordedDrawGroup[];
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
    uniform4f: record('uniform4f'),
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
    litDrawGroups(): readonly RecordedDrawGroup[] {
      let lastUseProgramIndex = -1;
      for (let i = 0; i < log.length; i += 1) {
        if (log[i].op === 'useProgram') {
          lastUseProgramIndex = i;
        }
      }
      if (lastUseProgramIndex < 0) {
        throw new Error('no useProgram call recorded');
      }
      const groups: RecordedDrawGroup[] = [];
      let feeds: RecordedUniformFeed[] = [];
      for (let i = lastUseProgramIndex + 1; i < log.length; i += 1) {
        const call = log[i];
        if (
          call.op === 'uniform1f' ||
          call.op === 'uniform2f' ||
          call.op === 'uniform3f' ||
          call.op === 'uniform4f'
        ) {
          const location = call.args[0] as { readonly name: string };
          const values = call.args
            .slice(1)
            .map((value) => Number(value as number));
          feeds.push({ name: location.name, values });
        } else if (call.op === 'drawArrays') {
          groups.push({ feeds });
          feeds = [];
        }
      }
      if (feeds.length > 0) {
        groups.push({ feeds });
      }
      return groups;
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
        a[row] * b0 + a[4 + row] * b1 + a[8 + row] * b2 + a[12 + row] * b3;
    }
  }
}

/**
 * Expected car body model matrix = T(x,0,z) * R(yaw) * R(offset).
 *
 * The drift offset is composed as a SEPARATE rotation because that is how the
 * pass applies it (a post-translation rotation, AC-047C), so the expected
 * matrix follows the SAME float32 pipeline as the renderer. A single
 * `R(yaw + offset)` is mathematically equal but rounds differently in float32
 * - the two differ by up to one float32 ULP (5.96e-8 at this magnitude) - so
 * composing the offset separately keeps the comparison exact instead of
 * tolerance-bound.
 */
function expectedBodyMatrix(
  worldX: number,
  worldZ: number,
  yawRad: number,
  driftOffsetRad: number,
): readonly number[] {
  const t = new Float32Array(16);
  const r = new Float32Array(16);
  const d = new Float32Array(16);
  const model = new Float32Array(16);
  const out = new Float32Array(16);
  translationMat4(t, worldX, 0, worldZ);
  rotationYMat4(r, yawRad);
  multiplyMat4(model, t, r);
  rotationYMat4(d, driftOffsetRad);
  multiplyMat4(out, model, d);
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

/**
 * The PUBLISHED pose envelope of the visible drift yaw (RED-14R, AC-088):
 * the PUBLISHED slip ceiling times the visual yaw gain plus the steer-yaw
 * articulation term, published by the drift model as
 * `NEATRACING_DRIFT_POSE_ENVELOPE_RAD` = 0.30185 rad.
 *
 * This REPLACES the retired committed-slip product (`MAX_SLIP * GAIN` =
 * 0.2025 rad): the visible body yaw is bounded by the PUBLISHED slip
 * basis, not by the committed slip the car happens to carry. The retired
 * pin is DELETED (never weakened) by this same change.
 */
const PUBLISHED_POSE_YAW_ENVELOPE_RAD = NEATRACING_DRIFT_POSE_ENVELOPE_RAD;

/**
 * The drift-yaw fixtures: both INSIDE the published pose envelope
 * (0.30185 rad).
 */
const REBASED_DRIFT_YAW_FIXTURES_RAD = [0.12, 0.2] as const;

/**
 * The raw steer echo the fixture feeds to the front wheels: the value the
 * production seam carries today (`steerAngleRad`, an input echo - G-V5).
 * The RED contracts hold this value FIXED so the rendered steer yaw can
 * only change when the PUBLISHED physics state changes.
 */
const RAW_STEER_ECHO_RAD = 0.35;

/**
 * The visible-modulation floor (rad) the steer contracts use: a rendered
 * steer yaw that differs from the raw echo by less than this is not a
 * VISIBLE difference (1e-3 rad is about 0.06 degrees).
 */
const STEER_RESPONSE_TOLERANCE_RAD = 1e-3;

/**
 * Adds the PUBLISHED traction severity to a car state.
 *
 * The seam does not carry the channel yet, so the fixture cannot spell it
 * as a plain object literal (the excess-property check would reject the
 * unknown property). The helper keeps the channel NAME in the fixture's
 * type - the returned state is `CarPassCarState & { tractionSeverity:
 * number }` - so the production must read EXACTLY `tractionSeverity` for
 * the contracts to turn green. The name is pinned by the BEHAVIOUR, never
 * by a placeholder export.
 *
 * @param base - The car state without the traction channel.
 * @param tractionSeverity - The published traction severity in [0, 1].
 * @returns The car state carrying the published severity.
 */
function withTractionSeverity(
  base: CarPassCarState,
  tractionSeverity: number,
): CarPassCarState & { tractionSeverity: number } {
  return Object.assign({}, base, { tractionSeverity });
}

/**
 * Renders one car with the given published traction severity and the given
 * raw steer echo, and returns the recording harness of the lit pass.
 *
 * The pose is FIXED across the fixtures on purpose: the published severity
 * is a quantity the player does NOT control, so every traction contract
 * compares two renders that differ ONLY in the severity (the R1 mitigation:
 * a contract that only compares two player-controlled inputs cannot tell a
 * physics-driven response from an input-driven one).
 *
 * The car state is built through {@link withTractionSeverity}: the seam does
 * not carry the channel yet, so the fixture cannot spell it as a plain
 * object literal (the excess-property check would reject it). The helper
 * keeps the channel NAME in the fixture's type, so the production must read
 * EXACTLY `tractionSeverity` for the contracts to turn green - the name is
 * pinned by the BEHAVIOUR, not by a placeholder export.
 *
 * @param tractionSeverity - The published traction severity in [0, 1].
 * @param steerAngleRad - The raw steer echo fed to the front wheels.
 * @returns The recording harness of the rendered lit pass.
 */
function renderCarWithTraction(
  tractionSeverity: number,
  steerAngleRad: number,
): MockGlHarness {
  const harness = createMockGl();
  const carPass = createCarPass(harness.gl, TEST_MODELS);
  const car = withTractionSeverity(
    {
      worldX: 0,
      worldZ: 0,
      yawRad: 0,
      teamColorId: 0,
      wheelSpinRad: 0,
      steerAngleRad,
    },
    tractionSeverity,
  );
  carPass.draw({
    viewProjection: TEST_VIEW_PROJECTION,
    cameraWorld: [0, 5, 0],
    cars: [car],
  });
  return harness;
}

/** Flattens the draw groups into the recorded uniform feed order. */
function uniformFeedsOf(
  groups: readonly RecordedDrawGroup[],
): readonly RecordedUniformFeed[] {
  const feeds: RecordedUniformFeed[] = [];
  for (const group of groups) {
    for (const feed of group.feeds) {
      feeds.push(feed);
    }
  }
  return feeds;
}

/**
 * Maps every uniform name fed in the given draw groups to the number of
 * DISTINCT tire-facing values it carried.
 *
 * A tire uniform that is fed with ONE value for all four tires is a SHARED
 * grip (the fixed-sun band); a tire uniform that is fed with several
 * distinct values is a PER-TIRE grip. The count is measured over the value
 * TUPLES of the feeds and NEVER over the components of one tuple: the
 * shared `vec3` team color carries three different components but is still
 * ONE shared value, and counting components would report a shared uniform
 * as differentiated.
 *
 * A single four-component feed is ALSO accepted as a per-tire vector (one
 * vector carrying the four per-tire grips).
 *
 * @param groups - The draw groups to measure.
 * @param excludedNames - Uniform names that are NOT grip carriers (the
 * gradient TOGGLE is a band switch: 0 for the body, 1 for the wheels).
 * @returns The distinct-value count per uniform name.
 */
function uniformValueVariety(
  groups: readonly RecordedDrawGroup[],
  excludedNames: readonly string[] = [],
): ReadonlyMap<string, number> {
  const tuples = new Map<string, Set<string>>();
  for (const group of groups) {
    for (const feed of group.feeds) {
      if (excludedNames.includes(feed.name)) continue;
      const bucket = tuples.get(feed.name) ?? new Set<string>();
      bucket.add(feed.values.join(','));
      tuples.set(feed.name, bucket);
    }
  }
  const counts = new Map<string, number>();
  for (const [name, bucket] of tuples) {
    let count = bucket.size;
    if (count === 1) {
      const [only] = [...bucket];
      const components = only?.split(',') ?? [];
      if (components.length === 4) {
        count = new Set(components).size;
      }
    }
    counts.set(name, count);
  }
  return counts;
}

/**
 * Extracts the steer yaw the front-left wheel was rendered with: the
 * rendered wheel yaw minus the body yaw minus the socket mount yaw.
 *
 * The socket order is front-left, front-right, rear-left, rear-right, so
 * the first mount is a FRONT wheel - the only steered axle.
 *
 * @param harness - The recording harness of a rendered lit pass.
 * @returns The rendered front-wheel steer yaw in radians.
 */
function frontWheelSteerYawRad(harness: MockGlHarness): number {
  const matrices = harness.modelMatricesForLitPass();
  const bodyYawRad = yawFromModelColumn(matrices[0]);
  const frontMount = resolveVoxelWheelMountTransform('front-left');
  const frontWheelYawRad = yawFromModelColumn(matrices[1]);
  return wrapAngle(frontWheelYawRad - bodyYawRad - frontMount.rotationYawRad);
}

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
          driftYawOffsetRad: 0.12,
        },
      ],
    };
    carPass.draw(state);

    const litMatrices = harness.modelMatricesForLitPass();
    expect(litMatrices.length).toBeGreaterThanOrEqual(5);
    const bodyMatrix = litMatrices[0];
    expectMatricesClose(bodyMatrix, expectedBodyMatrix(3, 5, 0.7, 0.12));
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
          driftYawOffsetRad: 0.2,
        },
      ],
    };
    carPass.draw(state);

    const litMatrices = harness.modelMatricesForLitPass();
    expect(litMatrices.length).toBeGreaterThanOrEqual(5);
    const bodyYawRad = yawFromModelColumn(litMatrices[0]);
    expect(bodyYawRad).toBeCloseTo(0.4 + 0.2, 6);

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
        expectedWheelMatrix(2, -1, 0.4, 0.2, mount, 0, 1.25),
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

  it('maps drift offset monotonically within the published pose envelope', () => {
    const offsets = [
      -PUBLISHED_POSE_YAW_ENVELOPE_RAD,
      -PUBLISHED_POSE_YAW_ENVELOPE_RAD / 2,
      0,
      PUBLISHED_POSE_YAW_ENVELOPE_RAD / 2,
      PUBLISHED_POSE_YAW_ENVELOPE_RAD,
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
    // LEDGER NOTE (fix-packet-P1-drift-feel-tune-state-iteration-1, OBS-3 /
    // GAP j): the yaw makes a float32 round-trip through the uploaded
    // `uModel` matrix, so the extracted value carries the float32 epsilon
    // (the retired 0.2025 rad bound measured 0.20249999177779054, i.e.
    // ~8.22e-9 off; the published 0.30185 rad bound rounds the same way).
    // Precision 10 demands < 5e-11 and is unreachable in float32; precision
    // 7 demands < 5e-7 and still pins the envelope to seven decimals. The
    // asserted VALUES are unchanged - only the precision digit is relaxed.
    expect(extractedYaws[0]).toBeCloseTo(-PUBLISHED_POSE_YAW_ENVELOPE_RAD, 7);
    expect(extractedYaws[extractedYaws.length - 1]).toBeCloseTo(
      PUBLISHED_POSE_YAW_ENVELOPE_RAD,
      7,
    );
  });
});

describe('drift yaw envelope (AC-047C, RED G-15 + RED 14R)', () => {
  // [RED-14R-PUBLISHED-SLIP-ENVELOPE]
  it('bounds the visible drift yaw by the PUBLISHED pose envelope, not the retired committed-slip product', () => {
    // The PUBLISHED basis: the published slip ceiling (0.493 rad, the
    // regime-independent maximum) times the visual yaw gain (0.45) plus
    // the steer-yaw articulation term (0.08) = 0.30185 rad. The retired
    // pin was the COMMITTED slip ceiling times the same gain (0.45 * 0.45
    // = 0.2025 rad), which the published slip (the committed slip plus the
    // chase-lag overshoot) can BREACH - so the committed-slip product is
    // DELETED here and the published-slip contract replaces it.
    expect(PUBLISHED_POSE_YAW_ENVELOPE_RAD).toBeCloseTo(0.30185, 10);
  });

  // [RED-G15-FIXTURES-INSIDE-ENVELOPE]
  it('feeds only in-envelope drift yaw fixtures to the visual path', () => {
    expect(
      REBASED_DRIFT_YAW_FIXTURES_RAD.every(
        (value) => Math.abs(value) <= PUBLISHED_POSE_YAW_ENVELOPE_RAD,
      ),
    ).toBe(true);
  });

  // [RED-G15-MONOTONIC-INSIDE-ENVELOPE]
  it('renders the body yaw inside the published envelope at the envelope boundary', () => {
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
          driftYawOffsetRad: PUBLISHED_POSE_YAW_ENVELOPE_RAD,
        },
      ],
    };
    carPass.draw(state);

    const bodyMatrix = harness.modelMatricesForLitPass()[0];
    expect(Math.abs(yawFromModelColumn(bodyMatrix))).toBeLessThanOrEqual(
      PUBLISHED_POSE_YAW_ENVELOPE_RAD + 1e-6,
    );
  });
});

describe('published traction severity (AC-088, RED 14R)', () => {
  // [RED-14R-CAR-TRACTION-CONSUMED]
  it('feeds the lit pass from the published traction severity', () => {
    // Arrange — the SAME pose and the SAME raw steer echo in both renders;
    // only the published severity differs (0 = a straight line, 0.85 = the
    // tires are sliding). The severity is NOT a player input, so the only
    // way the two renders can differ at all is the physics reaching the
    // screen.
    const straight = renderCarWithTraction(0, RAW_STEER_ECHO_RAD);
    const drifting = renderCarWithTraction(0.85, RAW_STEER_ECHO_RAD);
    // Act
    const straightFeeds = uniformFeedsOf(straight.litDrawGroups());
    const driftingFeeds = uniformFeedsOf(drifting.litDrawGroups());
    // Assert — the lit pass consumes the published severity.
    expect(driftingFeeds).not.toEqual(straightFeeds);
  });

  // [RED-14R-CAR-PER-TIRE-GRIP]
  it('gives the four tires their own grip instead of one shared grip', () => {
    // Arrange — one car with a published severity above zero.
    const harness = renderCarWithTraction(0.85, RAW_STEER_ECHO_RAD);
    // Act — the tire-facing uniforms of the whole lit pass. The gradient
    // TOGGLE is excluded: it is a band switch (0 for the body, 1 for the
    // wheels), never a grip. Every other uniform the pass feeds today
    // carries ONE shared value for all four tires.
    const variety = uniformValueVariety(harness.litDrawGroups(), [
      'uApplyTireGradient',
    ]);
    // Assert — at least one tire-facing uniform is fed with more than one
    // distinct value (the fronts and the rears do NOT share one grip).
    expect({
      differentiatedTireUniforms: [...variety.entries()]
        .filter(([, count]) => count > 1)
        .map(([name]) => name),
    }).toEqual({
      differentiatedTireUniforms: expect.arrayContaining([expect.any(String)]),
    });
  });

  // [RED-14R-CAR-STEER-FROM-PHYSICS]
  it('drives the front-wheel steer yaw from the published physics state, not the raw steer echo', () => {
    // Arrange — the SAME raw steer echo in both renders; only the published
    // severity differs.
    const straight = renderCarWithTraction(0, RAW_STEER_ECHO_RAD);
    const drifting = renderCarWithTraction(0.85, RAW_STEER_ECHO_RAD);
    // Act
    const straightSteerRad = frontWheelSteerYawRad(straight);
    const driftingSteerRad = frontWheelSteerYawRad(drifting);
    // Assert — the front-wheel yaw RESPONDS to the published physics state.
    expect(Math.abs(driftingSteerRad - straightSteerRad)).toBeGreaterThan(
      STEER_RESPONSE_TOLERANCE_RAD,
    );
  });

  // [RED-14R-CAR-STEER-NOT-ECHO]
  it('never renders the raw steer echo as the front-wheel steer yaw while the tires are sliding', () => {
    // Arrange — the raw steer echo, with the published severity saying the
    // tires are sliding (the physics state, not the player's key, says so).
    const harness = renderCarWithTraction(0.85, RAW_STEER_ECHO_RAD);
    // Act
    const renderedSteerRad = frontWheelSteerYawRad(harness);
    // Assert — G-V5: the tire yaw-follow is an INPUT ECHO today (the raw
    // axis times 35 degrees), so a dramatic-looking turn is rendered even
    // when the physics state never latched a slide.
    expect(Math.abs(renderedSteerRad - RAW_STEER_ECHO_RAD)).toBeGreaterThan(
      STEER_RESPONSE_TOLERANCE_RAD,
    );
  });
});
