/**
 * NeatRacing NGE controller — the recycled racing-curriculum network-controller
 * wrapper (spec P2-S3-A) migrated realm-free onto the NeatRacing observation
 * ladder and physics control contract.
 *
 * THE E11 OWNERSHIP (spec §9.7): this module implements the SIM-side per-tick
 * control path of the two-worker split — observe -> activate -> steer — while
 * growth, variant evaluation, score windows, exhaustion state, oscillation
 * state, and commit/rollback bookkeeping stay with the separate eval-worker
 * adaptation engine (`./runtime.adaptation.ts`). The controller is
 * inference-only: it never mutates physics state, never draws ambient
 * randomness (D1: the network's own seeded stream drives any stochastic
 * forward pass), and never reads the adaptation engine's state.
 *
 * ## Recycled contract
 *
 * The module is the realm-free migration of the racing-curriculum
 * `controller/nge.controller.ts`: the same activate-only network seam, the
 * same tier-guide evidence channels (optimal-line lateral offset at scalar
 * index 16, heading error at index 17), and the same degenerate single-car
 * self-radio seam. The NeatRacing deltas are the observation source and the
 * control sink:
 *
 * - Observation: `assembleNeatRacingObservationVector` (the 82/89/110 channel
 *   ladder, AC-027) replaces the recycled 63/77-wide assembler. The recycled
 *   Tier-2 self-radio mapping becomes the ladder's self-radio tail, carried
 *   on levels 2 and 3.
 * - Control: the output is a NeatRacing `PhysicsStepInput`
 *   (`throttle`, `brake`, `steer`). THE STEER AXIS IS THE SOLE TURN CHANNEL
 *   (AC-093): no `turnRatePercent` is ever emitted. The throttle axis is
 *   clamped to its physics domain `[0, 1]` — negative network output commands
 *   zero thrust because reverse thrust is not an axis of this game — while
 *   the steer axis keeps the recycled `[-1, 1]` domain.
 *
 * ## Background reading
 *
 * - NEAT and topology-evolving neuroevolution:
 *   K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through
 *   Augmenting Topologies," *Evolutionary Computation*, vol. 10, no. 2,
 *   pp. 99-127, 2002.
 *   [NEAT publications](https://nn.cs.utexas.edu/?neat-papers)
 *
 * @module
 */

import type { TrackSpec } from '../../racing_curriculum/track/track.generator.types';
import type { PhysicsStepInput } from '../physics/physics.step.service';
import {
  assembleNeatRacingObservationVector,
  NEATRACING_OBSERVATION_LATERAL_SPEED_WORLD_SCALE,
  NEATRACING_OBSERVATION_OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE,
  NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
  NEATRACING_OBSERVATION_WHEEL_SLIP_SLIP_ANGLE_SCALE,
  NEATRACING_OBSERVATION_YAW_RATE_RADIANS_PER_SECOND_SCALE,
  type NeatRacingObservationLevel,
  type NeatRacingObservationState,
} from './observation.assembler';

// ── Recycled scales and seams (mirrored owner-local VERBATIM from the racing
// curriculum `nge.controller.ts`) ─────────────────────────────────────────────

/**
 * Default observation level for the NGE controller seam — the recycled solo
 * NGE driver default tier (`DEFAULT_CONTROLLER_TIER`).
 */
export const NEATRACING_NGE_DEFAULT_OBSERVATION_LEVEL: NeatRacingObservationLevel =
  1;

/**
 * Number of controller output channels mapped to the control axes.
 * Recycled value verbatim (`CONTROLLER_OUTPUT_COUNT`): the network emits
 * `[throttle, steer]` (plus the self-radio write tail on radio-augmented
 * levels).
 */
export const NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT = 2;

/**
 * Observation channel index of the normalized optimal-line lateral offset.
 * Recycled value verbatim; the NeatRacing assembler keeps the recycled
 * 20-scalar base block order, so the index is level-independent.
 */
export const NEATRACING_NGE_OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET = 16;

/**
 * Observation channel index of the normalized optimal-line heading error.
 * Recycled value verbatim; see
 * {@link NEATRACING_NGE_OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET}.
 */
export const NEATRACING_NGE_OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR = 17;

/**
 * Guidance overlay alpha for the solo NGE level: a faint optimal-line hint
 * while the network learns to drive. Recycled value verbatim
 * (`TIER_ONE_GUIDANCE_ALPHA`). Radio-augmented levels turn the hint off
 * entirely (alpha 0) so the network self-navigates.
 */
export const NEATRACING_NGE_GUIDANCE_ALPHA_TIER_ONE = 0.35;

/**
 * Default self-radio channel count for the degenerate single-car radio seam.
 * Owner-local single source: the recycled radio width (7) is the assembler's
 * self-radio tail width, so the controller derives it instead of restating.
 */
export const NEATRACING_NGE_RADIO_DIMENSION =
  NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT;

// ── Public types ──────────────────────────────────────────────────────────────

/**
 * Public network surface required by the NeatRacing NGE controller seam.
 * Recycled owner-local contract verbatim: the controller is inference-only and
 * accepts any object exposing `activate(...)`, so the NGE network stays free
 * to grow structure behind the seam.
 */
export interface NeatRacingControllerNetwork {
  /**
   * Runs one forward pass over the assembled observation vector.
   *
   * @param inputVector - Normalized observation vector from the assembler.
   * @returns Controller output vector (or a scalar shorthand).
   */
  activate(
    inputVector: readonly number[] | Float32Array,
  ): readonly number[] | number;
}

/** Mutable single-car self-radio seam (recycled `SingleCarRadioChannel`). */
export interface NeatRacingSingleCarRadioChannel {
  /**
   * Writes the latest self-monitoring payload into the channel.
   *
   * @param radioValues - Ordered self-radio payload.
   */
  writeSelf(radioValues: readonly number[] | Float32Array): void;
  /**
   * Reads the most recent self-radio payload.
   *
   * @returns Ordered self-radio payload.
   */
  readSelf(): readonly number[] | Float32Array;
}

/**
 * Runtime options for the NeatRacing NGE controller seam. All fields are
 * optional. Level 1 is the default; the self-radio seam is exercised on
 * ladder levels 2 and 3 (the ladder's self-radio tail).
 */
export interface NeatRacingNgeControllerOptions {
  /** Observation ladder level that decides the vector width and guidance fade. */
  readonly level?: NeatRacingObservationLevel;
  /** Optional shared self-radio seam for radio-augmented levels. */
  readonly radioChannel?: NeatRacingSingleCarRadioChannel;
  /** Radio channel width; defaults to the assembler's self-radio tail width. */
  readonly radioDim?: number;
}

/**
 * Tier-guide evidence sampled from one controller tick. Recycled structure
 * verbatim; the signals are read from the assembled observation vector's
 * optimal-line scalar channels.
 */
export interface NeatRacingNgeControllerTickEvidence {
  /** Absolute normalized lateral error against the optimal line. */
  lateralErrorNormalized: number;
  /** Heading alignment against the optimal line in `[0, 1]`. */
  headingAlignment01: number;
  /** Combined guidance-need signal in `[0, 1]`. */
  centerGuideNeed01: number;
}

/** Controller tick result including control and adaptation evidence. */
export interface NeatRacingNgeControllerTickResult {
  /** Controller action for the current fixed-timestep tick. */
  control: PhysicsStepInput;
  /** Tier-guide evidence sampled from the same assembled observation vector. */
  evidence: NeatRacingNgeControllerTickEvidence;
}

/** Public controller surface consumed by the sim-worker per-tick wiring. */
export interface NeatRacingNgeController {
  /**
   * Produces the physics control input for the next simulation step.
   *
   * @param envState - Focal car observation seam state.
   * @param trackSpec - Frozen track geometry.
   * @returns Physics control input for the next fixed-timestep integration.
   */
  computeControl(
    envState: NeatRacingObservationState,
    trackSpec: TrackSpec,
  ): PhysicsStepInput;
  /**
   * Produces the physics control input plus tier-guide evidence sampled from
   * the active observation seam on this tick.
   *
   * @param envState - Focal car observation seam state.
   * @param trackSpec - Frozen track geometry.
   * @returns Control input plus tier-guide evidence for adaptation scoring.
   */
  computeControlWithEvidence(
    envState: NeatRacingObservationState,
    trackSpec: TrackSpec,
  ): NeatRacingNgeControllerTickResult;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Creates the NeatRacing NGE controller wrapper around the public
 * `activate(...)` inference surface.
 *
 * Per tick the controller assembles the level-shaped observation vector, runs
 * one public forward pass, maps the first two outputs onto the physics control
 * axes (throttle in `[0, 1]`, brake fixed at 0, steer in `[-1, 1]`, and NO
 * `turnRatePercent` — the steer axis is the sole turn channel, AC-093), and —
 * on radio-augmented levels — feeds the degenerate self-radio seam. The
 * recycled write-back order is preserved: the seam is written with the derived
 * self-monitoring payload BEFORE assembly, and with the network's own excess
 * outputs AFTER activation, so an externally supplied seam can stream both.
 *
 * @param network - Public network object that exposes `activate(...)`.
 * @param options - Optional observation level and radio-channel overrides.
 * @returns Stateful controller that maps network outputs to engine commands.
 *
 * @example
 * ```ts
 * const controller = createNeatRacingNgeController(network, { level: 2 });
 * const { control, evidence } = controller.computeControlWithEvidence(
 *   { carX: 0, carY: 0, carHeading: 0 },
 *   trackSpec,
 * );
 * physics.step(control);
 * ```
 */
export function createNeatRacingNgeController(
  network: NeatRacingControllerNetwork,
  options: NeatRacingNgeControllerOptions = {},
): NeatRacingNgeController {
  const controllerLevel = options.level ?? NEATRACING_NGE_DEFAULT_OBSERVATION_LEVEL;
  const radioDimension = options.radioDim ?? NEATRACING_NGE_RADIO_DIMENSION;
  const radioChannel =
    options.radioChannel ?? createNeatRacingSingleCarRadioChannel(radioDimension);

  const computeTickResult = (
    envState: NeatRacingObservationState,
    trackSpec: TrackSpec,
  ): NeatRacingNgeControllerTickResult => {
    // Step 1: Bring the self-radio seam online before radio-augmented assembly.
    //         The recycled mapping is level-based: the ladder's self-radio
    //         tail starts at level 2 (the recycled Tier 2).
    const observationState =
      controllerLevel >= 2
        ? prepareRadioSeamState(envState, radioChannel)
        : envState;

    // Step 2: Build the level-shaped observation vector and run a forward pass.
    const observationVector = assembleNeatRacingObservationVector(
      observationState,
      trackSpec,
      {
        level: controllerLevel,
      },
    );
    const controllerOutputs = normalizeNeatRacingControllerOutputs(
      network.activate(observationVector),
    );

    // Step 3: On radio-augmented levels, stream the network's excess output
    //         channels into the self-radio seam (recycled Tier-2 write-back).
    if (controllerLevel >= 2) {
      radioChannel.writeSelf(
        controllerOutputs.slice(
          NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
          NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT + radioDimension,
        ),
      );
    }

    // Step 4: Map control outputs and emit evidence from the same vector.
    return {
      control: {
        throttle: clampNeatRacingThrottleAxis(controllerOutputs[0] ?? 0),
        brake: 0,
        steer: clampNeatRacingSteerAxis(controllerOutputs[1] ?? 0),
      },
      evidence: resolveTickEvidence(observationVector),
    };
  };

  return {
    computeControl(envState, trackSpec): PhysicsStepInput {
      return computeTickResult(envState, trackSpec).control;
    },
    computeControlWithEvidence(envState, trackSpec) {
      return computeTickResult(envState, trackSpec);
    },
  };
}

/**
 * Resolves the optimal-line overlay alpha for each NGE controller level.
 *
 * The recycled `resolveGuidanceAlphaForTier` mapping, expressed on the
 * NeatRacing ladder: the solo NGE level keeps a faint hint, and the
 * radio-augmented levels turn the hint off entirely so the network
 * self-navigates once the self-radio seam is live. The scripted-baseline
 * full-overlay case (recycled alpha 1.0) belongs to the scripted controller
 * domain, not to this NGE seam.
 *
 * @param level - Observation ladder level driving the vector width.
 * @returns Overlay alpha in `[0, 1]`.
 *
 * @example
 * ```ts
 * resolveNeatRacingNgeGuidanceAlpha(1); // 0.35 — faint hint on the solo level
 * resolveNeatRacingNgeGuidanceAlpha(2); // 0 — self-navigation on radio levels
 * ```
 */
export function resolveNeatRacingNgeGuidanceAlpha(
  level: NeatRacingObservationLevel,
): number {
  if (level === 1) {
    return NEATRACING_NGE_GUIDANCE_ALPHA_TIER_ONE;
  }
  return 0;
}

/**
 * Creates the single-car radio seam used by radio-augmented controller levels.
 * Recycled `createSingleCarRadioChannel` verbatim: a fixed-width zero buffer
 * that pads and truncates through `writeSelf` and exposes `readSelf`.
 *
 * @param radioDim - Number of channels retained in the radio buffer.
 * @returns Read/write self-monitoring radio channel.
 */
export function createNeatRacingSingleCarRadioChannel(
  radioDim: number,
): NeatRacingSingleCarRadioChannel {
  const channelWidth = Math.max(0, Math.floor(radioDim));
  const radioBuffer = new Array<number>(channelWidth).fill(0);

  return {
    writeSelf(radioValues: readonly number[] | Float32Array): void {
      radioBuffer.fill(0);

      for (let channelIndex = 0; channelIndex < channelWidth; channelIndex++) {
        radioBuffer[channelIndex] = radioValues[channelIndex] ?? 0;
      }
    },
    readSelf(): readonly number[] {
      return radioBuffer;
    },
  };
}

// ── Private helpers (recycled structure) ──────────────────────────────────────

/**
 * Enriches the observation state with the current self-radio field before
 * radio-augmented assembly. Recycled `prepareObservationState` mapping: the
 * seam is first written with the derived self-monitoring payload, then the
 * buffer is published through the state's `radioField` tail.
 *
 * @param envState - Focal car observation seam state.
 * @param radioChannel - Single-car self-monitoring seam.
 * @returns Observation-ready seam state carrying the radio field.
 */
function prepareRadioSeamState(
  envState: NeatRacingObservationState,
  radioChannel: NeatRacingSingleCarRadioChannel,
): NeatRacingObservationState {
  // Step 1: Derive a stable seven-channel self-monitoring payload.
  radioChannel.writeSelf(resolveNeatRacingSelfMonitoringPayload(envState));

  // Step 2: Expose that payload through the state's self-radio tail.
  return {
    ...envState,
    radioField: Float32Array.from(radioChannel.readSelf()),
  };
}

/**
 * Builds the seven-channel self-monitoring payload consumed by the self-radio
 * seam. Recycled payload verbatim, with the recycled scales sourced from the
 * assembler's owner-local constants: forward speed (108), lateral speed (54),
 * total speed (108), yaw rate (1), slip angle (pi/2), centered progress, and
 * the raw optimal-line lateral offset (18).
 *
 * @param envState - Focal car observation seam state.
 * @returns Ordered seven-channel self-monitoring payload.
 */
function resolveNeatRacingSelfMonitoringPayload(
  envState: NeatRacingObservationState,
): Float32Array {
  const forwardSpeedWorld = envState.forwardSpeedWorld ?? 0;
  const lateralSpeedWorld = envState.lateralSpeedWorld ?? 0;
  const speedWorld =
    envState.speedWorld ?? Math.hypot(forwardSpeedWorld, lateralSpeedWorld);

  return Float32Array.from([
    clampNeatRacingControlValue(
      forwardSpeedWorld / NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
    ),
    clampNeatRacingControlValue(
      lateralSpeedWorld / NEATRACING_OBSERVATION_LATERAL_SPEED_WORLD_SCALE,
    ),
    clampNeatRacingControlValue(
      speedWorld / NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
    ),
    clampNeatRacingControlValue(
      (envState.yawRateRadiansPerSecond ?? 0) /
        NEATRACING_OBSERVATION_YAW_RATE_RADIANS_PER_SECOND_SCALE,
    ),
    clampNeatRacingControlValue(
      (envState.slipAngleRadians ?? 0) /
        NEATRACING_OBSERVATION_WHEEL_SLIP_SLIP_ANGLE_SCALE,
    ),
    clampNeatRacingControlValue((envState.progress01 ?? 0) * 2 - 1),
    clampNeatRacingControlValue(
      (envState.optimalLineLateralOffsetWorld ?? 0) /
        NEATRACING_OBSERVATION_OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE,
    ),
  ]);
}

/**
 * Normalizes raw network outputs into a flat, finite array. Recycled
 * `normalizeControllerOutputs` mapping: scalar shorthand becomes a two-channel
 * vector and non-finite entries collapse to 0 so downstream slicing stays safe.
 *
 * @param controllerOutputs - Raw controller network output.
 * @returns Finite controller output vector.
 */
function normalizeNeatRacingControllerOutputs(
  controllerOutputs: readonly number[] | number,
): readonly number[] {
  if (typeof controllerOutputs === 'number') {
    return [controllerOutputs, 0];
  }

  return controllerOutputs.map((outputValue) => {
    return Number.isFinite(outputValue) ? outputValue : 0;
  });
}

/**
 * Clamps a raw controller output into the `[-1, 1]` domain. Recycled
 * `clampControlValue` verbatim.
 *
 * @param value - Raw controller output.
 * @returns Value clamped to `[-1, 1]`.
 */
function clampNeatRacingControlValue(value: number): number {
  return Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0));
}

/**
 * Clamps the network's first output onto the physics throttle axis.
 *
 * The `PhysicsStepInput.throttle` axis is defined in `[0, 1]` (positive
 * forward thrust), while NGE networks emit signed outputs. The clamp floors
 * the negative half at 0 — reverse thrust is not an axis of this game — and
 * keeps the recycled [-1, 1] saturation behavior on the positive half.
 *
 * @param value - Raw network throttle output.
 * @returns Throttle command clamped to `[0, 1]`.
 */
function clampNeatRacingThrottleAxis(value: number): number {
  return Math.max(0, clampNeatRacingControlValue(value));
}

/**
 * Clamps the network's second output onto the physics steer axis in
 * `[-1, 1]` (positive steers counterclockwise). The steer axis is the sole
 * turn channel (AC-093), so no `turnRatePercent` is ever emitted.
 *
 * @param value - Raw network steer output.
 * @returns Steer command clamped to `[-1, 1]`.
 */
function clampNeatRacingSteerAxis(value: number): number {
  return clampNeatRacingControlValue(value);
}

/**
 * Extracts the optimal-line guidance evidence channels from one assembled
 * observation vector. Recycled `resolveTickEvidence` verbatim: the lateral
 * error is the absolute normalized offset (channel 16), the heading error is
 * the absolute normalized angular error (channel 17), and the combined
 * guidance-need signal is the max of the two.
 *
 * @param observationVector - Active assembled observation vector.
 * @returns Lateral error, heading alignment, and combined guidance-need signal.
 */
function resolveTickEvidence(
  observationVector: Float32Array,
): NeatRacingNgeControllerTickEvidence {
  const lateralErrorNormalized = Math.abs(
    observationVector[
      NEATRACING_NGE_OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET
    ] ?? 0,
  );
  const headingErrorNormalized = Math.abs(
    observationVector[
      NEATRACING_NGE_OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR
    ] ?? 0,
  );
  const headingAlignment01 = 1 - Math.min(1, headingErrorNormalized);
  const centerGuideNeed01 = Math.max(
    lateralErrorNormalized,
    headingErrorNormalized,
  );

  return {
    lateralErrorNormalized,
    headingAlignment01,
    centerGuideNeed01,
  };
}
