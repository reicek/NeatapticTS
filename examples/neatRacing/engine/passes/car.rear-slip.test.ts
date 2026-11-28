/**
 * RED home for the VISIBLE REAR SLIP (Step 30, slice `P5-S30-red`,
 * AC-110 / RED-S30-5) plus the THREE invariants that MUST survive the
 * green.
 *
 * THE DEFECT (GAP-5-07, AC-110): the car is visibly sideways but the REAR
 * wheels still point straight ahead, so the slide is unreadable at a glance.
 * The renderer today yaws the rear axle by EXACTLY 0
 * (`isFrontWheel ? computeFrontWheelSteerYawRad(car) : 0` in the wheel draw
 * loop of `examples/neatRacing/engine/passes/car.pass.ts`), so the rear
 * wheels cannot show the slide the physics already publishes.
 *
 * THE RED CONTRACT ([RED-S30-5]): the rear wheels carry the PUBLISHED slip
 * angle VERBATIM. The published slip is the drift model's
 * `atan2(lateralVelocity, speed)`, and the production sign convention is
 * LEFT = NEGATIVE (a LEFT turn commits a NEGATIVE slip), so the fixture
 * publishes -0.25 rad and the rendered rear-wheel yaw must read back as
 * -0.25 rad - the SAME sign and the SAME magnitude, never a scaled, a
 * negated or a clamped copy.
 *
 * THE RED DECISIONS (the shape the green slice must honour):
 * - the car state carries the published slip through a NEW OPTIONAL channel
 *   named `slipAngleRad` - the SAME name the browser debug seam already
 *   publishes (`examples/neatRacing/browser-entry/browser-entry.ts`, the
 *   FOURTH seam field) and the SAME `...Rad` convention as
 *   `steerAngleRad` / `driftYawOffsetRad` / `yawRad`. The channel is
 *   OPTIONAL, so the out-of-scope producers of `CarPassCarState` keep
 *   compiling;
 * - the fixture therefore CANNOT spell the channel as a plain object
 *   literal (the excess-property check would reject the unknown property),
 *   so it goes through the house helper {@link withPublishedSlip}, which
 *   keeps the channel NAME in the fixture's TYPE
 *   (`CarPassCarState & { slipAngleRad: number }`) - the production must
 *   read EXACTLY `slipAngleRad` for the contract to turn green. The name is
 *   pinned by the BEHAVIOUR, never by a placeholder export;
 * - the TOLERANCE: the published slip is asserted VERBATIM to six decimals
 *   (`expect.closeTo(..., 6)`), which is far above the float32 rounding of
 *   the composed wheel chain (the MEASURED error of the extracted yaw is
 *   4.75e-9 rad) and far below any visible difference; the SEPARATION
 *   between the front and the rear axle is asserted with the house
 *   visible-modulation floor (1e-3 rad, about 0.06 degrees) - the rear and
 *   the front must differ by MORE than shading.
 *
 * THE MEASURED BASELINE (read off the LIVE bytes on 2026-11-22, BEFORE any
 * production edit):
 * - the extracted rear-left wheel yaw (relative to the body yaw and to the
 *   rear mount yaw) is EXACTLY 0 rad, because the rear axle is yawed by
 *   exactly 0 today -> [RED-S30-5] FAILS with "expected -0.25, received 0";
 * - the front-left wheel yaw is +0.35 rad (the raw steer echo scaled by the
 *   front grip share, and the grip share is 1 because the car carries no
 *   traction severity), so the front-vs-rear gap is 0.35 rad today and
 *   0.6 rad after the green - the separation invariant PASSES today and
 *   MUST survive the green;
 * - the committed slip after the 120-tick build phase is
 *   -0.33407350581129147 rad and the MAXIMUM committed slip during the
 *   120-tick countersteer phase is exactly 0 -> the AC-087 never-mirrors
 *   pin PASSES today (and it is NOT vacuous: the slide IS engaged);
 * - the peak |pose| of the visual pose spring over 300 ticks with a 1.0 rad
 *   slip input is 0.5807730854115001 rad, well ABOVE the published envelope
 *   (0.30185 rad) -> the unclamped-pose pin PASSES today.
 *
 * THE SCOPE: this home probes the EXISTING car pass and the EXISTING drift
 * model ONLY - it imports NO module that does not exist yet, so it is RED
 * BY ASSERTION and NEVER by an import, a compile or a type error. The
 * browser debug seam is NOT probed here: its EIGHT-FIELD pin belongs to
 * `examples/neatRacing/browser-entry/browser-entry.test.ts` (an
 * already-passing invariant, not a red contract) and that module is OUT OF
 * SCOPE of this slice.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';
import { VOXEL_EMPTY_PALETTE_ENTRY } from '../../voxel/model.schema';
import type { VoxelBinModel, VoxelBinQuad } from '../../voxel/bin.codec';
import {
  NEATRACING_WHEEL_SOCKET_ORDER,
  resolveVoxelWheelMountTransform,
  type VoxelWheelSocketId,
} from '../../voxel/wheel.mount.utils';
import {
  NEATRACING_DRIFT_POSE_ENVELOPE_RAD,
  createZeroDriftChannelState,
  stepDriftCommitment,
  stepDriftPoseSpring,
} from '../../physics/drift.model';
import { createCarPass, type CarPassCarState } from './car.pass';

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
 * Hand-rolled WebGL2 recording harness - the house mock, trimmed to the one
 * probe this home needs (the `uModel` matrices uploaded during the lit
 * pass). Uniform locations carry their own name so the recorded matrix
 * uploads can be attributed to `uModel` without a real GPU or DOM.
 */
interface MockGlHarness {
  readonly gl: WebGL2RenderingContext;
  /** Returns every `uModel` matrix uploaded after the last `useProgram`. */
  modelMatricesForLitPass(): readonly number[][];
}

/**
 * Creates a hand-rolled recording mock of a `WebGL2RenderingContext`.
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
 * The PUBLISHED slip the fixture publishes to the renderer (rad).
 *
 * The sign is the PRODUCTION convention (LEFT = NEGATIVE slip: the nose
 * leads and the velocity lags, so `atan2(lateralVelocity, speed)` is
 * negative in a LEFT turn), and the magnitude is comfortably inside the
 * published slip ceilings (0.452 rad on road, 0.493 rad off road).
 */
const PUBLISHED_SLIP_ANGLE_RAD = -0.25;

/**
 * The raw steer echo the fixture feeds to the front wheels (rad): the value
 * the production seam carries today (`steerAngleRad`, an input echo). It is
 * held FIXED so the rendered rear yaw can only change when the PUBLISHED
 * physics state changes.
 */
const RAW_STEER_ECHO_RAD = 0.35;

/**
 * The visible-modulation floor (rad): a front-vs-rear yaw difference below
 * this is not a VISIBLE difference (1e-3 rad is about 0.06 degrees), so the
 * separation contract demands MORE than shading.
 */
const STEER_RESPONSE_TOLERANCE_RAD = 1e-3;

/** The fixed simulation tick of the drift model (1/60 s). */
const DRIFT_TICK_DT_S = 1 / 60;

/** The build phase of the never-mirrors probe (120 ticks of a left turn). */
const COMMIT_PHASE_TICKS = 120;

/** The countersteer phase of the never-mirrors probe (120 ticks). */
const COUNTERSTEER_PHASE_TICKS = 120;

/** The pose-spring probe length (300 ticks). */
const POSE_SPRING_TICKS = 300;

/**
 * The slip the pose-spring probe feeds ON PURPOSE (rad).
 *
 * The published slip ceiling is 0.493 rad, and the pose target at that
 * ceiling is `0.45 * 0.493` = 0.22185 rad, whose MEASURED peak is 0.2863
 * rad - BELOW the published envelope (0.30185 rad). A probe at the
 * published ceiling therefore CANNOT prove the absence of a clamp (the
 * clipped and the unclamped model would both stay under the envelope), so
 * only an ABOVE-envelope input can. 1.0 rad is such an input.
 */
const POSE_PROBE_SLIP_ANGLE_RAD = 1;

/**
 * Adds the PUBLISHED slip angle to a car state.
 *
 * The seam does not carry the channel yet, so the fixture cannot spell it
 * as a plain object literal (the excess-property check would reject the
 * unknown property). The helper keeps the channel NAME in the fixture's
 * type - the returned state is `CarPassCarState & { slipAngleRad: number }`
 * - so the production must read EXACTLY `slipAngleRad` for the contract to
 * turn green. The name is pinned by the BEHAVIOUR, never by a placeholder
 * export.
 *
 * @param base - The car state without the published-slip channel.
 * @param slipAngleRad - The published slip angle in radians.
 * @returns The car state carrying the published slip.
 */
function withPublishedSlip(
  base: CarPassCarState,
  slipAngleRad: number,
): CarPassCarState & { slipAngleRad: number } {
  return Object.assign({}, base, { slipAngleRad });
}

/**
 * Extracts the yaw around world up from a column-major model matrix using
 * its third column (the model's local +Z axis). The wheel chain ends with a
 * spin rotation about the model depth axis, which leaves that column
 * untouched, so the extracted yaw is the car yaw plus the mount yaw plus the
 * axle's own yaw.
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

/**
 * The index of one wheel inside the lit-pass `uModel` matrices: the body
 * is index 0 and the wheels follow in the deterministic socket order
 * (front-left, front-right, rear-left, rear-right).
 *
 * @param socketId - The socket to locate.
 * @returns The index of that socket's `uModel` matrix.
 */
function wheelMatrixIndex(socketId: VoxelWheelSocketId): number {
  return NEATRACING_WHEEL_SOCKET_ORDER.indexOf(socketId) + 1;
}

/**
 * Extracts the yaw of one rendered wheel RELATIVE to the car body and to
 * that socket's own mount yaw - i.e. the axle's OWN yaw, with the shared
 * body yaw and the shared mount yaw (the same +pi/2 for all four sockets)
 * removed.
 *
 * @param matrices - The lit-pass `uModel` matrices.
 * @param bodyYawRad - The rendered body yaw.
 * @param socketId - The wheel socket to measure.
 * @returns The axle yaw in radians.
 */
function wheelYawRadRelativeToBody(
  matrices: readonly number[][],
  bodyYawRad: number,
  socketId: VoxelWheelSocketId,
): number {
  const mount = resolveVoxelWheelMountTransform(socketId);
  const wheelYawRad = yawFromModelColumn(
    matrices[wheelMatrixIndex(socketId)],
  );
  return wrapAngle(wheelYawRad - bodyYawRad - mount.rotationYawRad);
}

/**
 * Renders ONE car that carries the PUBLISHED slip and returns the recorded
 * lit-pass matrices.
 *
 * The pose is FIXED (world origin, no yaw, no drift offset): the drift
 * offset is left at 0 on purpose, so the measured wheel yaw is attributable
 * to the slip ALONE.
 *
 * @param slipAngleRad - The published slip angle in radians.
 * @returns The lit-pass `uModel` matrices of the rendered car.
 */
function renderCarWithPublishedSlip(slipAngleRad: number): readonly number[][] {
  const harness = createMockGl();
  const carPass = createCarPass(harness.gl, TEST_MODELS);
  const car = withPublishedSlip(
    {
      worldX: 0,
      worldZ: 0,
      yawRad: 0,
      teamColorId: 0,
      wheelSpinRad: 0,
      steerAngleRad: RAW_STEER_ECHO_RAD,
    },
    slipAngleRad,
  );
  carPass.draw({
    viewProjection: TEST_VIEW_PROJECTION,
    cameraWorld: [0, 5, 0],
    cars: [car],
  });
  return harness.modelMatricesForLitPass();
}

describe('the visible rear slip (AC-110, RED-S30-5)', () => {
  // [RED-S30-5]
  it('carries the PUBLISHED slip angle verbatim on the rear wheels', () => {
    // Arrange - one car at the origin with the published slip (-0.25 rad,
    // the production LEFT-turn sign) and the raw steer echo (0.35 rad).
    const matrices = renderCarWithPublishedSlip(PUBLISHED_SLIP_ANGLE_RAD);

    // Act - the rendered body yaw and the two axle yaws, each measured
    // RELATIVE to the body and to its own mount yaw (the mount yaw is the
    // same +pi/2 for all four sockets, so it cancels out of the difference).
    const bodyYawRad = yawFromModelColumn(matrices[0]);
    const rearWheelYawRad = wheelYawRadRelativeToBody(
      matrices,
      bodyYawRad,
      'rear-left',
    );
    const frontWheelYawRad = wheelYawRadRelativeToBody(
      matrices,
      bodyYawRad,
      'front-left',
    );

    // Assert - the rear axle carries the published slip VERBATIM: the SAME
    // sign and the SAME magnitude. MEASURED TODAY: the rear axle is yawed
    // by EXACTLY 0 (the draw loop yaws the rear wheels by 0), so this
    // fails with "expected -0.25, received 0".
    expect({
      publishedSlipAngleRad: PUBLISHED_SLIP_ANGLE_RAD,
      measuredRearWheelYawRad: rearWheelYawRad,
    }).toEqual({
      publishedSlipAngleRad: PUBLISHED_SLIP_ANGLE_RAD,
      measuredRearWheelYawRad: expect.closeTo(PUBLISHED_SLIP_ANGLE_RAD, 6),
    });

    // Assert - the MUST-SURVIVE invariant: the rear and the front axle
    // differ by MORE than shading. MEASURED TODAY: |0.35 - 0| = 0.35 rad;
    // after the green: |0.35 - (-0.25)| = 0.6 rad.
    expect(Math.abs(frontWheelYawRad - rearWheelYawRad)).toBeGreaterThan(
      STEER_RESPONSE_TOLERANCE_RAD,
    );
  });
});

describe('the invariants that must survive the green (AC-087 / the unclamped pose / the pure model)', () => {
  // [RED-S30-5-NEVER-MIRRORS]
  it('never mirrors the committed slide while the turn sign is latched', () => {
    // Arrange - the AC-087 probe: 120 ticks of a LEFT turn (steer = +1)
    // build the slide, then 120 ticks of countersteer (steer = -1) unwind
    // it. The drift model is probed DIRECTLY (the renderer is not involved
    // in the commitment law).
    let committed = createZeroDriftChannelState().committed;
    for (let tick = 0; tick < COMMIT_PHASE_TICKS; tick += 1) {
      committed = stepDriftCommitment(committed, {
        steer: 1,
        speed: 60,
        speedPercent: 1,
        dt: DRIFT_TICK_DT_S,
        lateralVelocity: 0,
        slipAngle: 0,
      });
    }
    const committedSlipAfterBuildRad = committed.slip;

    // Act - the countersteer phase, tracking the MAXIMUM committed slip.
    let maxCommittedSlipDuringCountersteerRad = Number.NEGATIVE_INFINITY;
    for (let tick = 0; tick < COUNTERSTEER_PHASE_TICKS; tick += 1) {
      committed = stepDriftCommitment(committed, {
        steer: -1,
        speed: 60,
        speedPercent: 1,
        dt: DRIFT_TICK_DT_S,
        lateralVelocity: 0,
        slipAngle: 0,
      });
      maxCommittedSlipDuringCountersteerRad = Math.max(
        maxCommittedSlipDuringCountersteerRad,
        committed.slip,
      );
    }

    // Assert - the AC-087 never-mirrors law: the committed slide NEVER
    // flips to the mirrored sign while the turn sign is latched. The pin is
    // NOT vacuous: the build phase really engages the slide (MEASURED:
    // -0.33407350581129147 rad), and the MEASURED maximum during the
    // countersteer phase is exactly 0.
    expect({
      committedSlipAfterBuildRad,
      maxCommittedSlipDuringCountersteerRad,
      slideEngagedDuringBuild: committedSlipAfterBuildRad < 0,
      mirroredDuringCountersteer: maxCommittedSlipDuringCountersteerRad > 0,
    }).toEqual({
      committedSlipAfterBuildRad: expect.any(Number),
      maxCommittedSlipDuringCountersteerRad: expect.any(Number),
      slideEngagedDuringBuild: true,
      mirroredDuringCountersteer: false,
    });
  });

  // [RED-S30-5-UNCLAMPED-POSE]
  it('keeps the visual pose spring UNCLAMPED above the published envelope', () => {
    // Arrange - the visual pose spring stepped with the ON-PURPOSE
    // above-envelope slip input (1.0 rad): at the published slip ceiling
    // (0.493 rad) the pose target (0.45 * 0.493) peaks at 0.2863 rad,
    // BELOW the published envelope (0.30185 rad), so only an above-envelope
    // input can PROVE the absence of a clamp. The steer is 0, so the
    // articulation term stays out of the pose target.
    let pose = createZeroDriftChannelState().pose;
    let peakAbsPoseRad = 0;
    for (let tick = 0; tick < POSE_SPRING_TICKS; tick += 1) {
      pose = stepDriftPoseSpring(pose, {
        steer: 0,
        speedPercent: 1,
        slipAngle: POSE_PROBE_SLIP_ANGLE_RAD,
        dt: DRIFT_TICK_DT_S,
      });
      peakAbsPoseRad = Math.max(peakAbsPoseRad, Math.abs(pose.pose));
    }

    // Assert - the pose spring is NOT clipped to the declared envelope. A
    // clamp would fight the spring and flatten the pop, which IS the
    // feature. MEASURED TODAY: the peak is 0.5807730854115001 rad against
    // the 0.30185 rad envelope.
    expect({
      peakAbsPoseRad,
      publishedEnvelopeRad: NEATRACING_DRIFT_POSE_ENVELOPE_RAD,
      exceedsPublishedEnvelope:
        peakAbsPoseRad > NEATRACING_DRIFT_POSE_ENVELOPE_RAD,
    }).toEqual({
      peakAbsPoseRad: expect.any(Number),
      publishedEnvelopeRad: NEATRACING_DRIFT_POSE_ENVELOPE_RAD,
      exceedsPublishedEnvelope: true,
    });
  });

  // [RED-S30-5-PURE-MODEL]
  it('keeps the drift model PURE: the same input yields the same output and the caller state is untouched', () => {
    // Arrange - one commitment tick on the SPAWN state, with a snapshot of
    // the caller's committed state taken BEFORE the call.
    const committed = createZeroDriftChannelState().committed;
    const committedSnapshot = { ...committed };
    const input = {
      steer: 1,
      speed: 60,
      speedPercent: 1,
      dt: DRIFT_TICK_DT_S,
      lateralVelocity: 0,
      slipAngle: 0,
    };

    // Act - the SAME (state, input) pair stepped twice.
    const firstOutput = stepDriftCommitment(committed, input);
    const secondOutput = stepDriftCommitment(committed, input);

    // Assert - the model is PURE: it never reads the track, never reads a
    // mutable global and never mutates the caller's state. The same input
    // yields the same output twice, and the caller's committed state is
    // exactly what it was before the calls.
    expect({
      firstOutput,
      secondOutput,
      callerStateAfterBothCalls: committed,
    }).toEqual({
      firstOutput: expect.any(Object),
      secondOutput: firstOutput,
      callerStateAfterBothCalls: committedSnapshot,
    });
  });
});
