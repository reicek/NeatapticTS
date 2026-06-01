import type { CarControlOutput } from '../environment/environment.types';
import type { TrackSpec } from '../track/track.generator.types';
import {
  assembleNormalizedObservationVector,
  type ObservationTier,
  type RacingObservationState,
} from './observation.assembler';

/** Default controller tier for the browser harness after the scripted seam. */
const DEFAULT_CONTROLLER_TIER: ObservationTier = 1;
/** Number of output channels expected from the public controller network. */
const CONTROLLER_OUTPUT_COUNT = 2;
/** Observation channel index for normalized optimal-line lateral offset. */
const OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET = 16;
/** Observation channel index for normalized optimal-line heading error. */
const OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR = 17;
/** Tier 1 keeps a faint optimal-line hint while Tier 2 turns it off. */
const TIER_ONE_GUIDANCE_ALPHA = 0.35;
/** Size of the degenerate single-car radio seam. */
const DEFAULT_RADIO_DIMENSION = 7;

/** Public network surface required by the owner-local NGE controller seam. */
export interface RacingControllerNetwork {
  /**
   * Runs one forward pass over the normalized observation vector.
   *
   * @param inputVector - Normalized observation vector from the assembler.
   * @returns Controller output pair or a scalar shorthand.
   */
  activate(
    inputVector: readonly number[] | Float32Array,
  ): readonly number[] | number;
}

/** Mutable self-radio seam used by Tier 2 single-car self-monitoring. */
export interface SingleCarRadioChannel {
  /**
   * Writes the latest self-monitoring payload into the channel.
   *
   * @param radioValues - Ordered self-monitoring payload.
   */
  writeSelf(radioValues: readonly number[] | Float32Array): void;
  /**
   * Reads the most recent self-monitoring payload.
   *
   * @returns Ordered self-monitoring payload.
   */
  readSelf(): readonly number[] | Float32Array;
}

/**
 * Runtime options for the owner-local NGE controller seam.
 *
 * All fields are optional. Tier 1 is the default; providing `radioChannel`
 * and `radioDim` enables the degenerate single-car self-monitoring path used
 * by Tier 2.
 */
export interface NgeControllerOptions {
  /** Active curriculum tier that decides observation width and guidance fade. */
  readonly tier?: ObservationTier;
  /** Optional shared radio seam for Tier 2 self-monitoring. */
  readonly radioChannel?: SingleCarRadioChannel;
  /** Radio channel width; defaults to seven channels. */
  readonly radioDim?: number;
}

/** Public controller surface consumed by the browser harness. */
export interface NgeController {
  /**
   * Produces throttle and steering for the next simulation step.
   *
   * @param envState - Current environment snapshot.
   * @param trackSpec - Frozen track geometry.
   * @returns Controller output for the next fixed-timestep integration.
   */
  computeControl(
    envState: RacingObservationState,
    trackSpec: TrackSpec,
  ): CarControlOutput;
  /**
   * Produces control output plus tier-guide evidence sampled from the active
   * observation seam on this tick.
   *
   * @param envState - Current environment snapshot.
   * @param trackSpec - Frozen track geometry.
   * @returns Control plus tier-guide evidence for adaptation scoring.
   */
  computeControlWithEvidence(
    envState: RacingObservationState,
    trackSpec: TrackSpec,
  ): NgeControllerTickResult;
}

/** Tier-guide evidence sampled from one controller tick. */
export interface NgeControllerTickEvidence {
  /** Absolute normalized lateral error against the center guide. */
  lateralErrorNormalized: number;
  /** Heading alignment against the center guide in [0, 1]. */
  headingAlignment01: number;
  /** Combined guidance need signal in [0, 1]. */
  centerGuideNeed01: number;
}

/** Controller tick result including control and adaptation evidence. */
export interface NgeControllerTickResult {
  /** Controller action for the current fixed-timestep tick. */
  control: CarControlOutput;
  /** Tier-guide evidence sampled from the same observation vector. */
  evidence: NgeControllerTickEvidence;
}

/**
 * Creates the owner-local NGE controller wrapper around the public `activate(...)`
 * inference surface.
 *
 * @param network - Public network object that exposes `activate(...)`.
 * @param options - Optional tier and radio-channel overrides.
 * @returns Stateful controller that maps network outputs to throttle and steer.
 */
export function createNgeController(
  network: RacingControllerNetwork,
  options: NgeControllerOptions = {},
): NgeController {
  const controllerTier = options.tier ?? DEFAULT_CONTROLLER_TIER;
  const radioDimension = options.radioDim ?? DEFAULT_RADIO_DIMENSION;
  const radioChannel =
    options.radioChannel ?? createSingleCarRadioChannel(radioDimension);

  const computeTickResult = (
    envState: RacingObservationState,
    trackSpec: TrackSpec,
  ): NgeControllerTickResult => {
    // Step 1: Bring the degenerate self-radio seam online before Tier 2 assembly.
    const observationState = prepareObservationState(
      envState,
      controllerTier,
      radioChannel,
    );

    // Step 2: Build the normalized observation vector and run a public forward pass.
    const observationVector = assembleNormalizedObservationVector(
      observationState,
      trackSpec,
      {
        tier: controllerTier,
      },
    );
    const controllerOutputs = normalizeControllerOutputs(
      network.activate(observationVector),
    );

    // Step 3: Map outputs and emit evidence from the same observation seam.
    return {
      control: {
        throttle: clampControlValue(controllerOutputs[0] ?? 0),
        steer: clampControlValue(controllerOutputs[1] ?? 0),
      },
      evidence: resolveTickEvidence(observationVector),
    };
  };

  return {
    computeControl(
      envState: RacingObservationState,
      trackSpec: TrackSpec,
    ): CarControlOutput {
      return computeTickResult(envState, trackSpec).control;
    },
    computeControlWithEvidence(
      envState: RacingObservationState,
      trackSpec: TrackSpec,
    ): NgeControllerTickResult {
      return computeTickResult(envState, trackSpec);
    },
  };
}

/**
 * Resolves the optimal-line overlay alpha for each curriculum tier.
 *
 * - `0` — full overlay (alpha 1.0); used for the scripted baseline before NGE
 *   control is active.
 * - `1` — faint overlay (alpha 0.35); keeps a subtle optimal-line hint while
 *   the solo NGE driver learns.
 * - `2` — overlay off (alpha 0); the network must self-navigate without the
 *   hint once the radio seam is live.
 *
 * @param tier - Curriculum tier index (0 = scripted baseline, 1 = solo NGE, 2 = radio-augmented).
 * @returns Overlay alpha in [0, 1].
 */
export function resolveGuidanceAlphaForTier(tier: 0 | 1 | 2): number {
  if (tier === 0) {
    return 1;
  }

  if (tier === 1) {
    return TIER_ONE_GUIDANCE_ALPHA;
  }

  return 0;
}

/**
 * Creates the owner-local single-car radio seam used by Tier 2 self-monitoring.
 *
 * @param radioDim - Number of channels retained in the radio buffer.
 * @returns Read/write self-monitoring radio channel.
 */
export function createSingleCarRadioChannel(
  radioDim: number,
): SingleCarRadioChannel {
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

/**
 * Enriches the environment snapshot with the current self-radio field when Tier 2
 * is active.
 *
 * @param envState - Current environment snapshot.
 * @param controllerTier - Active controller tier.
 * @param radioChannel - Single-car self-monitoring seam.
 * @returns Observation-ready environment snapshot.
 */
function prepareObservationState(
  envState: RacingObservationState,
  controllerTier: ObservationTier,
  radioChannel: SingleCarRadioChannel,
): RacingObservationState {
  if (controllerTier !== 2) {
    return envState;
  }

  // Step 1: Derive a stable seven-channel self-monitoring payload.
  radioChannel.writeSelf(resolveSelfMonitoringPayload(envState));

  // Step 2: Expose that payload through the environment radio field.
  return {
    ...envState,
    radioField: Float32Array.from(radioChannel.readSelf()),
  };
}

/**
 * Builds the seven-channel self-monitoring payload consumed by the Tier 2 radio seam.
 *
 * @param envState - Current environment snapshot.
 * @returns Ordered seven-channel self-monitoring payload.
 */
function resolveSelfMonitoringPayload(
  envState: RacingObservationState,
): Float32Array {
  const forwardSpeedWorld = envState.forwardSpeedWorld ?? 0;
  const lateralSpeedWorld = envState.lateralSpeedWorld ?? 0;
  const speedWorld =
    envState.speedWorld ?? Math.hypot(forwardSpeedWorld, lateralSpeedWorld);

  return Float32Array.from([
    clampControlValue(forwardSpeedWorld / 108),
    clampControlValue(lateralSpeedWorld / 54),
    clampControlValue(speedWorld / 108),
    clampControlValue((envState.yawRateRadiansPerSecond ?? 0) / 1),
    clampControlValue((envState.slipAngleRadians ?? 0) / (Math.PI / 2)),
    clampControlValue((envState.progress01 ?? 0) * 2 - 1),
    clampControlValue((envState.optimalLineLateralOffsetWorld ?? 0) / 18),
  ]);
}

/**
 * Normalizes controller outputs into a two-element array.
 *
 * @param controllerOutputs - Raw network output.
 * @returns Two-element control vector.
 */
function normalizeControllerOutputs(
  controllerOutputs: readonly number[] | number,
): readonly number[] {
  if (typeof controllerOutputs === 'number') {
    return [controllerOutputs, 0];
  }

  return Array.from({ length: CONTROLLER_OUTPUT_COUNT }, (_, outputIndex) => {
    const outputValue = controllerOutputs[outputIndex] ?? 0;
    return Number.isFinite(outputValue) ? outputValue : 0;
  });
}

/**
 * Clamps controller outputs into the accepted environment range.
 *
 * @param value - Raw controller output.
 * @returns Value clamped to [-1, 1].
 */
function clampControlValue(value: number): number {
  return Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0));
}

/**
 * Extracts Tier 1 center-guide evidence channels from one normalized observation vector.
 *
 * @param observationVector - Active normalized observation vector.
 * @returns Lateral error, heading alignment, and combined guidance-need signal.
 */
function resolveTickEvidence(
  observationVector: Float32Array,
): NgeControllerTickEvidence {
  const lateralErrorNormalized = Math.abs(
    observationVector[OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET] ?? 0,
  );
  const headingErrorNormalized = Math.abs(
    observationVector[OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR] ?? 0,
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
