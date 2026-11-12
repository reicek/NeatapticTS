/**
 * In-band live simulation worker for the neatRacing browser demo (Phase 2,
 * Step 01, slice P2-S1-A).
 *
 * Educational note — why this file exists:
 * The phase-1 race loop lived on the browser main thread (`host/race-session.ts`):
 * every rAF frame advanced physics, race rules, and controllers inline, which
 * couples the frame budget of the render pipeline to the cost of the simulation.
 * The Phase-2 learning-document contract (LD P2-S1) moves that authority to a
 * single in-band live-worker file: the worker owns ALL simulation state, and
 * the host only paces frames and forwards the once-per-frame human control
 * sample (decision D5) as a message. Every piece of racing truth now crosses a
 * typed message boundary, so the ownership question ("which side owns this
 * state?") has one visible answer for every field.
 *
 * What this worker owns per slice P2-S1-A:
 * - the 60 Hz fixed-step accumulator (`host/render-loop.utils.ts` reuse),
 * - the car fleet physics (`stepPhysics` per car through `stepPhysicsFleet`,
 *   whose single live-car path stays `stepPhysics`),
 * - the race-rules state machine (`createRaceRulesState` / `stepRaceRules`),
 * - the agent-car control path: the INJECTED NGE control seam
 *   (`./nge.fleet.control.ts` — observe -> activate -> steer through the
 *   LATEST adopted per-car network). The LEGACY SCRIPTED LANE is RETIRED
 *   (AC-029 — no dual path, no deferred cleanup): an agent car with NO
 *   injected controller has NO driver, and the fleet tick-input resolver
 *   raises the NAMED no-controller error instead of falling back to a
 *   scripted line,
 * - the per-tick forward lap-line crossing detector (ported phase-1 host
 *   boot logic, now owned where the car state it reads lives).
 *
 * What the host keeps until later phases: rendering, HUD, selector, and the
 * keyboard-sampling seam — the host samples driving input once per drawn
 * frame (D5) and ships it to the worker with the frame timestamp.
 *
 * Message protocol contract (slice A):
 * - host → worker `{ type: 'init', payload }` boots the session. The payload
 *   carries the AC-026 reproducibility tuple fields (`seed`, `agentCount`,
 *   `packSchemaVersion`, `opponentSnapshot`, `trackId`, `featureFlags`) plus
 *   the `trackSpec` geometry the sim needs to own its state. Replies exactly
 *   one `{ type: 'ready', payload }`.
 * - host → worker `{ type: 'frame', payload }` paces one drawn frame
 *   (`timestampMs` drives the accumulator; `sample` is the D5 human control
 *   sample for car 0 in `manual` mode). Replies exactly one
 *   `{ type: 'frames', payload }` whose payload snapshots the released ticks
 *   (`fromTick`..`toTick`), the post-frame fleet, and the post-frame race
 *   rules. Slice B replaces this payload with the packed
 *   `neatracing-packed-v1` budget; the message shape is the stable seam.
 *   The payload additionally carries every rival-swap event released during
 *   that frame's ticks as an OPTIONAL `rivalSwaps` list (P4-S2, AC-032 —
 *   the live rival winner propagation); frames with NO rival crossing omit
 *   the field entirely, so untouched frames keep their exact prior shape.
 * - the NAMED no-controller error is the ONE error that converts into a
 *   protocol reply (M1, slice `P4-S2-rival-propagation`): a frame resolved
 *   for a session booted WITHOUT an injected controller replies one
 *   `{ type: 'error', payload: { message } }` carrying the SAME message
 *   text the resolver raises (`NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR`
 *   plus the ` (race mode: <mode>)` suffix) instead of escaping
 *   `handleRequest` as an uncaught error. Every OTHER worker error keeps
 *   its current behavior.
 * - host → worker `{ type: 'stop' }` halts the session. Replies nothing
 *   (fire-and-forget lifecycle message, matching the flappy stop precedent).
 * - anything else (unknown type, malformed payload, bad ordering) replies one
 *   `{ type: 'error', payload: { message } }`.
 *
 * Determinism discipline (LD D1 / D5, AC-026):
 * - no `Math.random` anywhere in this file; every spawn is derived from the
 *   init seed and car index, and every controller is a pure step function.
 * - no clock reads inside the tick pipeline: wall-clock timestamps enter ONLY
 *   through `frame` messages and feed the release accumulator, never tick math.
 * - identical tuple + identical frame pacing + identical sample streams
 *   produce identical tick content (Level-2 ordered determinism), which the
 *   byte-parity slice (P2-S1-C) formalizes.
 */

/// <reference lib="webworker" />

import { createTrackSurfaceResolver } from '../../physics/collision.utils';
import {
  createInitialPhysicsCarState,
  stepPhysicsFleet,
  type PhysicsCarSpawnPose,
  type PhysicsCarState,
  type PhysicsStepInput,
} from '../../physics/physics.step.service';
import {
  createRaceRulesState,
  NEATRACING_HUMAN_MAX_LEVEL,
  NEATRACING_HUMAN_START_LEVEL,
  stepRaceRules,
  type RaceRulesState,
} from '../../physics/race-rules';
import {
  resolveTrackEdgeHalfWidthM,
  resolveTrackEdgeProfile,
  resolveTrackEdgeSideAt,
  resolveTrackSplineEdges,
  type TrackEdgeProfile,
  type TrackSplineEdge,
} from '../../track/track.edge';
import { NEATRACING_VOXEL_CAR_LENGTH_M } from '../../voxel/lod.distance.utils';
import {
  advanceFrameSteps,
  createNeatRacingRenderLoopState,
  NEATRACING_FIXED_TIMESTEP_SECONDS,
  type NeatRacingRenderLoopState,
} from '../../host/render-loop.utils';
import type {
  SplineSample,
  TrackSegment,
  TrackSpec,
} from '../../../racing_curriculum/track/track.generator.types';
import {
  mapNeatRacingFleetControl,
  type NeatRacingFleetControlSeam,
} from './nge.fleet.control';
import {
  resolveNeatRacingObservationChannelCount,
  NEATRACING_TEAM_LAYOUTS,
  type NeatRacingObservationLevel,
} from '../../ai/observation.assembler';
import { NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT } from '../../ai/nge.controller';
import {
  propagateNeatRacingRivalWinnerWeights,
  toNeatRacingRivalSwapPayload,
  type NeatRacingRivalSwapPayload,
} from '../../ai/rival.propagation';
import { deriveNeatRacingCarNetworkChildSeed } from './reproducibility.utils';
import Network from '../../../../src/architecture/network';

/** Smallest legal fleet size for a live sim-worker session (just the car). */
export const NEATRACING_SIM_WORKER_MIN_AGENTS = 1;

/**
 * Largest legal fleet size for a live sim-worker session.
 *
 * Educational note: the AC-033 demo fleet is three cars (human car 0 plus
 * two evolving agents), so this cap is deliberately generous rather than
 * load-bearing. The NGE eval-worker slice revisits the bound when real
 * populations flow through instead of hard-coding a tuning guess here.
 */
export const NEATRACING_SIM_WORKER_MAX_AGENTS = 32;

/**
 * The clearance (meters) the fleet grid stands FULLY BEHIND the green start
 * line, measured to the car CENTRE.
 *
 * THE DERIVED CLEARANCE: the value is NEVER a literal - it IS the ONE
 * published car length (`NEATRACING_VOXEL_CAR_LENGTH_M`, 4.5 m), so a drift
 * in the car's published size moves the grid WITH it. Measured to the
 * centre, the car's NOSE stands 2.25 m upstream of the plane and its TAIL
 * 6.75 m upstream, so the WHOLE footprint stays behind the line and can
 * never overlap the forward green band `[0, 4.0] m`.
 *
 * The constant is EXPORTED so the GREEN lane can cross-check the production
 * value against the RED home's test-local literal.
 */
export const NEATRACING_SIM_WORKER_SPAWN_BEHIND_CLEARANCE_M =
  NEATRACING_VOXEL_CAR_LENGTH_M;

/**
 * Anchor tolerance (meters) under which the spline's first sample counts as
 * sitting exactly on segment 0's start point. The generated layouts publish
 * their first spline sample AT the segment-0 control point (the generator's
 * interpolation factor is zero there), while the hand-authored fixture rings
 * hang their sample list elsewhere - the tolerance separates the two without
 * any heuristic geometry.
 */
export const NEATRACING_SIM_WORKER_SPAWN_PATH_ANCHOR_EPS_M = 0.01;

/**
 * Lateral gate (meters from segment 0's chord) inside which a pose counts as
 * crossing the lap line. Carried verbatim from the phase-1 host detector so
 * the worker-owned port stays behaviorally identical.
 */
export const NEATRACING_SIM_WORKER_LAP_LINE_LATERAL_GATE_M = 5;

/**
 * Along-track deadband (meters from the lap-line plane) below which samples
 * do not update the detector baseline. Carried verbatim from the phase-1 host
 * detector.
 */
export const NEATRACING_SIM_WORKER_LAP_LINE_ALONG_EPS_M = 0.5;

/**
 * Neutral control sample replayed for car 0 in BOTH race modes before the
 * host has sent any sample. Frozen: it is a constant, never a live
 * accumulator.
 */
export const NEUTRAL_SIM_WORKER_CONTROL_SAMPLE: Readonly<PhysicsStepInput> =
  Object.freeze({ throttle: 0, brake: 0, steer: 0 });

/** Race-mode knob the host boots the session with (selector state at init). */
export type NeatRacingSimWorkerRaceMode = 'manual' | 'ai';

/** Lifecycle phases of one sim-worker session. */
export type NeatRacingSimWorkerPhase = 'idle' | 'running' | 'stopped';

/**
 * Init payload for the sim worker.
 *
 * The AC-026 reproducibility tuple components ride alongside the track
 * geometry so one init message carries everything a replay bootstrap needs:
 * `trackId` names the tuple's track while `trackSpec` supplies the actual
 * owned geometry (segments for the surface resolver and lap detector).
 * The spline samples ride along for the REPLAY tuple, but the worker no
 * longer follows a scripted racing line (AC-029 — the legacy scripted lane
 * is retired).
 */
export interface NeatRacingSimWorkerInitPayload {
  /** AC-026 tuple: deterministic run seed driving per-car spawn states. */
  seed: number;
  /** AC-026 tuple: fleet size (car 0 plus `agentCount - 1` agent cars). */
  agentCount: number;
  /** AC-026 tuple: pack schema version string carried for later slices. */
  packSchemaVersion: string;
  /**
   * AC-026 tuple: opponent snapshot carried verbatim (opaque until the NGE
   * network-controller slice consumes it; not validated here).
   */
  opponentSnapshot: unknown;
  /** AC-026 tuple: reproducibility track identifier. */
  trackId: string;
  /** AC-026 tuple: feature flags carried verbatim (shape pinned by P2-S1-C). */
  featureFlags: Readonly<Record<string, boolean>> | undefined;
  /** Owned track geometry: segments + spline samples + generation seed. */
  trackSpec: TrackSpec;
  /** Boot race mode for car 0 (`'ai'` default; `'manual'` consumes samples). */
  raceMode?: NeatRacingSimWorkerRaceMode;
  /**
   * The INJECTED NGE control seam that drives the agent cars (cars 1..n).
   *
   * Educational note — the transport decision (slice `P4-S1-worker-seam`):
   * the seam rides the init payload IN BAND. The sim session is owned
   * IN-PROCESS by the host (`browser-entry.ts` imports
   * `createSimWorkerSession` directly), so the payload never crosses a
   * structured-clone boundary and a LIVE seam object (a function cannot be
   * cloned) is legal here. A future REAL worker realm would need a
   * snapshot-relay transport instead — the seam's relay surface
   * (`readCarController`) is the named seam for that swap.
   *
   * A payload WITHOUT a seam leaves the agent cars WITHOUT a driver: the
   * LEGACY SCRIPTED LANE is retired (AC-029), so the first fleet tick
   * raises the NAMED no-controller error — which the session converts into
   * a `{ type: 'error' }` REPLY carrying the same message text instead of
   * letting it escape `handleRequest` (M1, slice
   * `P4-S2-rival-propagation`); every other worker error keeps its current
   * behavior.
   */
  controller?: NeatRacingFleetControlSeam;
}

/** Per-frame pace message payload (one drawn frame, one reply). */
export interface NeatRacingSimWorkerFramePayload {
  /** rAF-style monotonic timestamp driving the fixed-step accumulator. */
  timestampMs: number;
  /** D5 once-per-frame human control sample for car 0 (`manual` mode). */
  sample?: PhysicsStepInput;
}

/** Host → worker request message union (the sim-worker protocol inbox). */
export type NeatRacingSimWorkerRequest =
  | { type: 'init'; payload: NeatRacingSimWorkerInitPayload }
  | { type: 'frame'; payload: NeatRacingSimWorkerFramePayload }
  | { type: 'stop' }
  | { type: 'network'; carIndex: number };

/**
 * The self-describing envelope kind the panel-network reply carries (P4-S2C,
 * Step 24C — the docked-right panel network transport).
 *
 * Educational note — WHY A NAMED KIND: the reply travels over the SAME
 * outbox as the frame/ready/error messages, so the consumer must be able to
 * tell a panel-network envelope from any other payload WITHOUT guessing the
 * payload's shape. The discriminator is a named constant on the worker's own
 * surface (the one serialization path per payload law — the host codec
 * compares against this exact constant, never against a bare literal).
 */
export const NEATRACING_PANEL_NETWORK_ENVELOPE_KIND = 'panel-network' as const;

/**
 * The panel-network reply payload (P4-S2C, Step 24C — the docked-right panel
 * network transport).
 *
 * Educational note — THE CARRIER LAW: the envelope names the car it carries
 * (the elected rival), the two network dimensions of that car's rung (the
 * observation channel count and the controller output count) and the car's
 * serialized connection weights in the established hex law (lowercase, two
 * characters per byte, no separators, native byte order). The worker NEVER
 * mutates the ledger to answer the request: the reply is an observational
 * COPY of the stored rival mind.
 */
export interface NeatRacingSimWorkerPanelNetworkReplyPayload {
  /** The self-describing envelope kind (the discriminator). */
  kind: typeof NEATRACING_PANEL_NETWORK_ENVELOPE_KIND;
  /** The rival car index the envelope carries (the carrier law). */
  carIndex: number;
  /** The rung's observation channel count (the network's input width). */
  channelCount: number;
  /** The rung's controller output count (the network's output width). */
  outputCount: number;
  /** The serialized connection weights (the established hex law). */
  weights: string;
}

/** Boot acknowledgement payload echoing the session identity tuple. */
export interface NeatRacingSimWorkerReadyPayload {
  /** Seed the session booted with. */
  seed: number;
  /** Fleet size the session booted with. */
  agentCount: number;
  /** Pack schema version string the session booted with. */
  packSchemaVersion: string;
  /** Reproducibility track identifier the session booted with. */
  trackId: string;
  /** Effective race mode (`'ai'` when omitted at init). */
  raceMode: NeatRacingSimWorkerRaceMode;
}

/**
 * Per-frame reply payload (slice A structured snapshots).
 *
 * Slice B swaps this payload for the packed `neatracing-packed-v1` budget;
 * the tick-range bookkeeping below is what the pack must preserve.
 */
export interface NeatRacingSimWorkerFramesPayload {
  /** Total-tick counter before this frame's released ticks (car 0 basis). */
  fromTick: number;
  /** Total-tick counter after this frame's released ticks. */
  toTick: number;
  /** Post-frame fleet snapshot (structured clone; plain objects). */
  cars: PhysicsCarState[];
  /** Post-frame race-rules snapshot (structured clone; plain object). */
  rules: RaceRulesState;
  /**
   * The rival-swap events released during this frame's ticks (P4-S2,
   * AC-032 — the LIVE rival winner propagation). Each event is the
   * JSON-safe tuple record the operator produces: the crossing rival's
   * serialized-network snapshot (byte-exact hex weights), the event's
   * derived mutation seed, and the replaced rivals' car indices. Frames
   * with NO rival crossing OMIT this field entirely, so untouched frames
   * keep their exact prior shape (the byte-parity mandate).
   */
  rivalSwaps?: readonly NeatRacingRivalSwapPayload[];
}

/** Error payload for any rejected request. */
export interface NeatRacingSimWorkerErrorPayload {
  /** Static human-readable reason (never contains dynamic state). */
  message: string;
}

/** Worker → host response message union (the sim-worker protocol outbox). */
export type NeatRacingSimWorkerResponse =
  | { type: 'ready'; payload: NeatRacingSimWorkerReadyPayload }
  | { type: 'frames'; payload: NeatRacingSimWorkerFramesPayload }
  | { type: 'network'; payload: NeatRacingSimWorkerPanelNetworkReplyPayload }
  | { type: 'error'; payload: NeatRacingSimWorkerErrorPayload };

/** Read-model diagnostics snapshot of one sim-worker session. */
export interface NeatRacingSimWorkerSessionSnapshot {
  /** Current lifecycle phase. */
  phase: NeatRacingSimWorkerPhase;
  /** Total ticks released since boot (car 0 tick basis). */
  tick: number;
  /** Live fleet (copy; safe to hold without aliasing tick output). */
  cars: readonly PhysicsCarState[];
  /** Live race-rules state (copy). */
  rules: RaceRulesState;
  /** Lap-line detector baseline (latched along-track side, or null). */
  lapLinePrevSide: number | null;
}

/**
 * The worker-plane rival propagation state (P4-S2, AC-032 — the LIVE
 * rival winner propagation).
 *
 * Educational note — WHY THE FLAT SERIALIZED PLANE: the worker's drive path
 * never owns a `Network` instance per car (the controller seam holds the
 * adopted networks), so the session keeps the rival MINDS as the recycled
 * `CarGenome.serialize()` Float32Arrays — the exact flat serialized plane
 * the propagation operator computes on. Each materialized rival mind is the
 * spawn-mind reconstruction the adaptation plane itself uses
 * (`createNeatRacingAdaptationCarStates`): a network of the level's
 * observation channel count and controller output count, seeded from the
 * per-car child seed, serialized in positional connection order.
 */
interface SimWorkerRivalPropagationState {
  /** The session's rival car indices (ascending; car 0 never included). */
  readonly rivalCarIndices: readonly number[];
  /** The ladder rung the fleet boots into (observation width + layout). */
  readonly level: NeatRacingObservationLevel;
  /** Serialized rival mind per rival car index (the mutable ledger). */
  readonly rivalMinds: Map<number, Float32Array>;
  /**
   * Lap-line detector baseline per fleet car (latched along-track side, or
   * null). The car 0 slot mirrors `lapLinePrevSide` and stays unused — car
   * 0 keeps its own detector slot and feeds the race rules.
   */
  readonly rivalPrevSides: (number | null)[];
  /**
   * The lap-line detector's ARMING flag per fleet car (the AC-105
   * behind-the-line law): false while the car still waits BEHIND the
   * start-line plane, true once the car has crossed the plane forward. The
   * car 0 slot mirrors `lapLineStarted` and stays unused — car 0 keeps its
   * own detector slot.
   */
  readonly rivalStarted: boolean[];
  /** The next rival-swap event index (starts at 0; +1 per resolved event). */
  nextEventIndex: number;
}

/** Internal mutable session bag: all sim truth owned by the worker lives here. */
interface SimWorkerSessionState {
  phase: NeatRacingSimWorkerPhase;
  renderLoop: NeatRacingRenderLoopState;
  cars: PhysicsCarState[];
  rules: RaceRulesState;
  lapLinePrevSide: number | null;
  /**
   * The lap-line detector's ARMING flag (the AC-105 behind-the-line law):
   * false while car 0 still waits BEHIND the start-line plane, true once the
   * car has crossed the plane forward. The flag lives in the INTERNAL bag so
   * the PUBLIC `NeatRacingSimWorkerSessionSnapshot` shape never grows a
   * field.
   */
  lapLineStarted: boolean;
  raceMode: NeatRacingSimWorkerRaceMode;
  humanSample: PhysicsStepInput | undefined;
  controller: NeatRacingFleetControlSeam | undefined;
  surfaceResolver: ReturnType<typeof createTrackSurfaceResolver> | undefined;
  firstSegment: TrackSegment | undefined;
  /**
   * The rival propagation lane (P4-S2): undefined for boots whose snapshot
   * names no rivals — the 1-3-car baseline boots keep their exact prior
   * behavior.
   */
  rivalPropagation: SimWorkerRivalPropagationState | undefined;
  readyDetail:
    | {
        seed: number;
        agentCount: number;
        packSchemaVersion: string;
        trackId: string;
      }
    | undefined;
}

/** Public session handle: a pure request/response state machine over the bag. */
export interface NeatRacingSimWorkerSession {
  /**
   * Handles one request and returns the responses it should post.
   *
   * Educational note: the session deliberately RETURNS its replies instead of
   * posting them itself. The pure protocol machine stays testable headless
   * (no worker realm needed), and the worker entrypoint below is the only
   * place that touches `self.postMessage`.
   *
   * @param request - Host request message (validated defensively at runtime).
   * @returns An ordered list of responses to post (0-2 entries).
   * @throws Error for every worker error EXCEPT the NAMED no-controller one
   *   (M1, slice `P4-S2-rival-propagation`): that named error is the ONE
   *   error that converts into a protocol `{ type: 'error' }` reply carrying
   *   the SAME message text (the constant plus the ` (race mode: <mode>)`
   *   suffix), so a consumer matching
   *   `message.includes(NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR)` still
   *   works and the uncaught escape stays closed. Every other worker error
   *   keeps its current behavior — it still escapes as an uncaught error.
   */
  handleRequest(
    request: NeatRacingSimWorkerRequest,
  ): NeatRacingSimWorkerResponse[];
  /** Read-model snapshot of the live session (diagnostics + tests). */
  readonly snapshot: NeatRacingSimWorkerSessionSnapshot;
}

/** Validation failure carries a static rejection reason string. */
type SimWorkerValidationError = { error: string };

/**
 * Validates an init payload and returns a rejection reason, or null when the
 * payload may boot. The carried tuple fields (`opponentSnapshot`,
 * `featureFlags`) are intentionally unvalidated: slice A transports them, the
 * byte-parity slice pins their serialized shape.
 *
 * @param payload - Candidate init payload.
 * @returns Rejection reason, or null when valid.
 */
function validateSimWorkerInitPayload(
  payload: unknown,
): SimWorkerValidationError | null {
  // Step 1: Reject non-object payloads wholesale.
  if (
    payload === null ||
    typeof payload !== 'object' ||
    Array.isArray(payload)
  ) {
    return { error: 'sim worker init failed: payload must be an object' };
  }
  const candidate = payload as Partial<NeatRacingSimWorkerInitPayload>;

  // Step 2: Validate the numeric identity fields.
  if (typeof candidate.seed !== 'number' || !Number.isFinite(candidate.seed)) {
    return { error: 'sim worker init failed: seed must be a finite number' };
  }
  if (
    typeof candidate.agentCount !== 'number' ||
    !Number.isInteger(candidate.agentCount) ||
    candidate.agentCount < NEATRACING_SIM_WORKER_MIN_AGENTS ||
    candidate.agentCount > NEATRACING_SIM_WORKER_MAX_AGENTS
  ) {
    return {
      error: `sim worker init failed: agentCount must be an integer in [${NEATRACING_SIM_WORKER_MIN_AGENTS}, ${NEATRACING_SIM_WORKER_MAX_AGENTS}]`,
    };
  }

  // Step 3: Validate the two string identity fields.
  if (
    typeof candidate.packSchemaVersion !== 'string' ||
    candidate.packSchemaVersion.length === 0
  ) {
    return {
      error:
        'sim worker init failed: packSchemaVersion must be a non-empty string',
    };
  }
  if (typeof candidate.trackId !== 'string' || candidate.trackId.length === 0) {
    return {
      error: 'sim worker init failed: trackId must be a non-empty string',
    };
  }

  // Step 4: Validate the owned geometry (non-empty closed-loop inputs).
  if (candidate.trackSpec === null || typeof candidate.trackSpec !== 'object') {
    return { error: 'sim worker init failed: trackSpec must be an object' };
  }
  if (
    !Array.isArray(candidate.trackSpec.segments) ||
    candidate.trackSpec.segments.length === 0
  ) {
    return {
      error:
        'sim worker init failed: trackSpec.segments must be a non-empty array',
    };
  }
  if (
    !Array.isArray(candidate.trackSpec.splineSamples) ||
    candidate.trackSpec.splineSamples.length === 0
  ) {
    return {
      error:
        'sim worker init failed: trackSpec.splineSamples must be a non-empty array',
    };
  }

  // Step 5: Validate the optional race-mode knob.
  if (
    candidate.raceMode !== undefined &&
    candidate.raceMode !== 'manual' &&
    candidate.raceMode !== 'ai'
  ) {
    return {
      error:
        "sim worker init failed: raceMode must be 'manual' or 'ai' when present",
    };
  }

  // Step 6: Validate the optional NGE control seam.
  //
  // Educational note: the seam is a LIVE object (its `computeControl` is a
  // function), so the check is a shape check, not a data check. An object OR
  // a function carrying a callable `computeControl` is accepted — the seam
  // double of the contract tests is a callable proxy over a function.
  if (
    candidate.controller !== undefined &&
    !isSimWorkerFleetControlSeam(candidate.controller)
  ) {
    return {
      error:
        'sim worker init failed: controller must expose a computeControl function',
    };
  }
  return null;
}

/**
 * Checks whether a candidate value is a usable NGE control seam.
 *
 * Educational note: the seam is a LIVE object, so the init payload carries a
 * shape check instead of a data check. A callable proxy over a function is
 * accepted as well as a plain object — the contract tests inject exactly
 * that shape.
 *
 * @param candidate - The candidate seam value.
 * @returns True when the candidate exposes a callable `computeControl`.
 */
function isSimWorkerFleetControlSeam(
  candidate: unknown,
): candidate is NeatRacingFleetControlSeam {
  // Step 1: A seam is an object OR a callable.
  if (
    candidate === null ||
    (typeof candidate !== 'object' && typeof candidate !== 'function')
  ) {
    return false;
  }

  // Step 2: The seam must expose a callable `computeControl`.
  const seam = candidate as unknown as { computeControl?: unknown };
  return typeof seam.computeControl === 'function';
}

/**
 * Validates a frame payload (object with a finite timestamp; an optional
 * sample carrying finite axes).
 *
 * @param payload - Candidate frame payload.
 * @returns Rejection reason, or null when valid.
 */
function validateSimWorkerFramePayload(
  payload: unknown,
): SimWorkerValidationError | null {
  // Step 1: Reject non-object payloads wholesale.
  if (
    payload === null ||
    typeof payload !== 'object' ||
    Array.isArray(payload)
  ) {
    return { error: 'sim worker frame failed: payload must be an object' };
  }
  const candidate = payload as Partial<NeatRacingSimWorkerFramePayload>;

  // Step 2: The pacing timestamp must be a usable number.
  if (
    typeof candidate.timestampMs !== 'number' ||
    !Number.isFinite(candidate.timestampMs)
  ) {
    return {
      error: 'sim worker frame failed: timestampMs must be a finite number',
    };
  }

  // Step 3: When a D5 sample rides along, its axes must be finite numbers.
  if (candidate.sample !== undefined) {
    const sample = candidate.sample;
    if (sample === null || typeof sample !== 'object') {
      return { error: 'sim worker frame failed: sample must be an object' };
    }
    for (const axis of ['throttle', 'brake', 'steer'] as const) {
      const value = sample[axis];
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        return {
          error: `sim worker frame failed: sample.${axis} must be a finite number`,
        };
      }
    }
  }
  return null;
}

/**
 * One step of the worker-owned lap-line detector: decides whether the pose
 * ENTERING this tick crossed the finish plane forward, and carries the
 * detector baseline forward.
 *
 * Educational note: this is the exact port of the phase-1 host detector
 * (browser-entry boot glue) — the finish line is the plane through segment
 * 0's start point perpendicular to its chord; a crossing counts only inside
 * the lateral gate, samples inside the along-track deadband never update the
 * baseline, and leaving the gate resets it (a car that wandered off can never
 * bank a stale side). Phase-1 semantics kept: direction filtering is a
 * carried gap, so ANY sign flip counts.
 *
 * @param car - Fleet car state (the tick's entering pose).
 * @param prevSide - Latched along-track side from the previous qualifying
 *   sample, or null before the first one.
 * @param firstSegment - Segment 0 of the owned track (or undefined when the
 *   geometry is degenerate; the detector then never reports a crossing).
 * @returns The crossing verdict plus the baseline to carry forward.
 */
function resolveSimWorkerLapLineDetectorStep(
  car: PhysicsCarState,
  prevSide: number | null,
  firstSegment: TrackSegment | undefined,
): { crossing: boolean; nextSide: number | null } {
  // Step 1: Degenerate geometry never counts a lap.
  if (firstSegment === undefined) {
    return { crossing: false, nextSide: prevSide };
  }
  const chordX = firstSegment.endX - firstSegment.startX;
  const chordZ = firstSegment.endY - firstSegment.startY;
  const chordLength = Math.hypot(chordX, chordZ);
  if (chordLength === 0) {
    return { crossing: false, nextSide: prevSide };
  }

  // Step 2: Project the pose onto the chord's along/lateral axes.
  const relX = car.x - firstSegment.startX;
  const relY = car.y - firstSegment.startY;
  const alongM = (relX * chordX + relY * chordZ) / chordLength;
  const lateralM = (chordX * relY - chordZ * relX) / chordLength;

  // Step 3: Outside the gate, the baseline resets (stale side dropped).
  if (Math.abs(lateralM) > NEATRACING_SIM_WORKER_LAP_LINE_LATERAL_GATE_M) {
    return { crossing: false, nextSide: null };
  }
  // Step 4: Deadband samples neither count nor update the baseline.
  if (Math.abs(alongM) < NEATRACING_SIM_WORKER_LAP_LINE_ALONG_EPS_M) {
    return { crossing: false, nextSide: prevSide };
  }

  // Step 5: First qualifying sample only latches the side.
  const baseline = prevSide;
  if (baseline === null) {
    return { crossing: false, nextSide: alongM };
  }
  return { crossing: baseline * alongM < 0, nextSide: alongM };
}

/**
 * One step of the ARMED worker lap-line detector: the phase-1 detector plus
 * the PRE-PLANE arming the AC-105 behind-the-line law demands.
 *
 * Educational note — WHY THE ARMING EXISTS: the AC-105 start grid moves the
 * WHOLE fleet UPSTREAM of the start-line plane, so at t=0 every car sits on
 * the NEGATIVE side of it. The phase-1 detector latches the side of the
 * FIRST QUALIFYING SAMPLE and counts ANY sign flip, so a fleet that starts
 * behind the line latches `-1` and banks a SPURIOUS lap the moment it
 * crosses the plane forward. The arming splits "crossing the line at the
 * START" from "completing a lap": while the detector is NOT armed, a
 * QUALIFYING sample BEHIND the plane never latches a side and never banks a
 * crossing, and the first qualifying sample IN FRONT of the plane arms the
 * detector WITHOUT banking a lap. Once armed the phase-1 semantics run
 * UNCHANGED (the lateral-gate reset, the along deadband, the
 * first-qualifying-sample latch and the sign-flip crossing), so the first
 * GENUINE lap completion banks lap 1.
 *
 * Educational note — WHY THE ARMING SURVIVES THE GATE RESET: leaving the
 * lateral gate drops the latched side (a car that wandered off can never
 * bank a stale side), but it must NOT un-arm the detector — the arming
 * records that the car has already crossed the plane once, which the gate
 * reset says nothing about. The arming is therefore PER CAR (car 0 AND
 * every rival: the rival-propagation tick runs the SAME detector for every
 * rival, so a fleet that starts behind the line would otherwise fire a
 * SPURIOUS rival-swap event at the race start) and it lives in the INTERNAL
 * session state, so the PUBLIC snapshot shape never grows a field.
 *
 * @param car - Fleet car state (the tick's entering pose).
 * @param prevSide - Latched along-track side from the previous qualifying
 *   sample, or null before the first one.
 * @param started - Whether the detector has already been armed by a forward
 *   crossing of the start-line plane.
 * @param firstSegment - Segment 0 of the owned track (or undefined when the
 *   geometry is degenerate; the detector then never reports a crossing).
 * @returns The crossing verdict plus the baseline and the arming flag to
 *   carry forward.
 */
function resolveSimWorkerArmedLapLineDetectorStep(
  car: PhysicsCarState,
  prevSide: number | null,
  started: boolean,
  firstSegment: TrackSegment | undefined,
): { crossing: boolean; nextSide: number | null; started: boolean } {
  // Step 1: While the detector is NOT armed the baseline is EMPTY (a
  //         qualifying sample behind the plane never latched one) and only
  //         a sample IN FRONT of the plane can arm the detector.
  if (!started) {
    const preArm = resolveSimWorkerLapLineDetectorStep(car, null, firstSegment);
    // Step 1a: Not a qualifying sample (outside the lateral gate, inside
    //          the along deadband or a degenerate geometry): the arming
    //          stays pending and the baseline stays empty.
    if (preArm.nextSide === null) {
      return { crossing: false, nextSide: null, started: false };
    }
    // Step 1b: BEHIND the plane: no latch, no crossing, still not armed.
    if (preArm.nextSide < 0) {
      return { crossing: false, nextSide: null, started: false };
    }
    // Step 1c: IN FRONT of the plane: arm the detector and latch the side
    //          (the phase-1 first-qualifying-sample latch) WITHOUT banking
    //          a lap.
    return { crossing: false, nextSide: preArm.nextSide, started: true };
  }

  // Step 2: Once armed the phase-1 detector semantics run UNCHANGED.
  const armed = resolveSimWorkerLapLineDetectorStep(
    car,
    prevSide,
    firstSegment,
  );
  return {
    crossing: armed.crossing,
    nextSide: armed.nextSide,
    started: true,
  };
}

/**
 * Resolves the road-anchored spawn path of one track, or `null` when the
 * spline does not anchor at the segment-0 start point.
 *
 * THE LAW (the P4-S2 live rival lane): the generated layout's spline IS the
 * road every controller reads - the racing-line table, the observation
 * assembler and the shipped autopilot all walk it - and its FIRST sample
 * carries an interpolation factor of ZERO, so it sits exactly on segment 0's
 * control point while the samples continue along the road. A fleet grid laid
 * on the SEGMENT-0 CHORD instead drifts OFF that road as the chord diverges
 * from the lane-center curve (the chord is the control straight, the spline
 * is the sampled road), spawning cars where the shipped autopilot's own
 * throttle gate never opens: the aim error at pose start already exceeds the
 * full-throttle error while the car stands still, so it can neither
 * accelerate nor turn (a standing car never rotates). The spawn path
 * therefore walks the spline when the spline is road-anchored, and keeps the
 * phase-1 chord walk when it is not (the hand-authored fixture rings, whose
 * pinned poses the contract holds verbatim).
 *
 * @param trackSpec - Owned track geometry.
 * @returns The ordered spline samples plus the closed-loop length, or `null`
 *   when the spline does not anchor at segment 0's start (or degenerates).
 */
function resolveSimWorkerSplineSpawnPath(
  trackSpec: TrackSpec,
): { samples: readonly SplineSample[]; loopLengthM: number } | null {
  const firstSegment = trackSpec.segments[0];
  const samples = trackSpec.splineSamples;
  if (firstSegment === undefined || samples.length < 2) {
    return null;
  }
  const anchor = samples[0];
  const anchorMissM = Math.hypot(
    anchor.x - firstSegment.startX,
    anchor.y - firstSegment.startY,
  );
  if (anchorMissM > NEATRACING_SIM_WORKER_SPAWN_PATH_ANCHOR_EPS_M) {
    return null;
  }
  let loopLengthM = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const from = samples[index];
    const to = samples[(index + 1) % samples.length];
    loopLengthM += Math.hypot(to.x - from.x, to.y - from.y);
  }
  if (loopLengthM <= 0) {
    return null;
  }
  return { samples, loopLengthM };
}

/**
 * The published road room at ONE spawn station: the half-width on each side
 * of the road centreline, read AT the station.
 *
 * THE STATION-DEPENDENT ROOM: the road widens on the OUTER side of a corner,
 * so the half-width is NOT a constant of the track - it is read at the
 * station the fleet waits on (the lap tail), never at the start line.
 */
interface SimWorkerSpawnStation {
  /** Index of the chord the station's side is measured against. */
  readonly segmentIndex: number;
  /** The published inner half-width at the station, in meters. */
  readonly innerHalfWidthM: number;
  /** The published outer half-width at the station, in meters. */
  readonly outerHalfWidthM: number;
}

/**
 * Resolves the published room on each side of the road centreline at one
 * spawn station.
 *
 * THE ASYMMETRIC ROOM: the road is WIDER on the outer side of a corner, so
 * the room is asymmetric - and WHICH side is the outer one is a property of
 * the WHOLE loop (the profile publishes the outer side's sign), not of the
 * local turn. On a counter-clockwise loop the outer side is the RIGHT of
 * travel, so the LEFT room is the inner half-width; on a clockwise loop the
 * two sides swap.
 *
 * @param station - The published station edge (the half-width pair).
 * @param profile - The published edge profile of the loop.
 * @returns The room to the LEFT and to the RIGHT of the path, in meters.
 */
function resolveSimWorkerSpawnRoomM(
  station: SimWorkerSpawnStation,
  profile: TrackEdgeProfile,
): { readonly leftRoomM: number; readonly rightRoomM: number } {
  const leftIsInner = profile.outerSideSign === -1;
  return {
    leftRoomM: leftIsInner ? station.innerHalfWidthM : station.outerHalfWidthM,
    rightRoomM: leftIsInner ? station.outerHalfWidthM : station.innerHalfWidthM,
  };
}

/**
 * Resolves the side-by-side lane offsets of one fleet grid.
 *
 * THE DERIVED LATERAL PITCH: the pitch is NOT a constant - it is the
 * SMALLEST of the published car length and the station's own room per car,
 * so a big fleet still fits inside the road while a small fleet keeps the
 * cars one car-length apart. The lanes are SYMMETRIC about the path (the
 * road centreline), so the grid is centred on the racing line, and car `k`
 * takes lane `k`.
 *
 * @param roomWidthM - The station's total room (left + right), in meters.
 * @param agentCount - Fleet size (>= 1).
 * @returns One lateral offset per fleet slot, in car order.
 */
function resolveSimWorkerSpawnLaneOffsetsM(
  roomWidthM: number,
  agentCount: number,
): number[] {
  const pitchM = Math.min(
    NEATRACING_VOXEL_CAR_LENGTH_M,
    roomWidthM / agentCount,
  );
  const offsetsM: number[] = [];
  for (let carIndex = 0; carIndex < agentCount; carIndex += 1) {
    offsetsM.push((carIndex - (agentCount - 1) / 2) * pitchM);
  }
  return offsetsM;
}

/**
 * Resolves the published half-width bound at one spawn station, for the side
 * the pose actually sits on.
 *
 * @param profile - The published edge profile of the loop.
 * @param station - The published station edge (the half-width pair).
 * @param x - The pose's generator-plane x, in meters.
 * @param y - The pose's generator-plane y, in meters.
 * @returns The side-resolved half-width at the station, in meters.
 */
function resolveSimWorkerSpawnBoundM(
  profile: TrackEdgeProfile,
  station: SimWorkerSpawnStation,
  x: number,
  y: number,
): number {
  return resolveTrackEdgeHalfWidthM(
    station,
    resolveTrackEdgeSideAt(profile, station.segmentIndex, x, y),
  );
}

/**
 * Places one lane's car on the road, clamped to the published half-width at
 * the spawn station.
 *
 * THE STATION-READ BOUND: the bound is the SIDE-RESOLVED published
 * half-width at the car's OWN station (the road widens on the outer side of
 * a corner, so the two sides differ) - never the width at the start line and
 * never a hard-coded road width. The side is resolved against the station's
 * own chord, and the bound is re-read after the clamp moves the car, so the
 * published pose is inside the road under the side it actually sits on.
 *
 * @param centreX - The station's centre point x, in meters.
 * @param centreY - The station's centre point y, in meters.
 * @param leftNormalX - The path's LEFT normal x component (unit length).
 * @param leftNormalY - The path's LEFT normal y component (unit length).
 * @param laneOffsetM - The lane's lateral offset, in meters.
 * @param profile - The published edge profile of the loop.
 * @param station - The published station edge (the half-width pair).
 * @returns The car's pose and the lateral offset it settled at.
 */
function resolveSimWorkerSpawnLanePose(
  centreX: number,
  centreY: number,
  leftNormalX: number,
  leftNormalY: number,
  laneOffsetM: number,
  profile: TrackEdgeProfile,
  station: SimWorkerSpawnStation,
): { readonly x: number; readonly y: number; readonly lateralM: number } {
  let lateralM = laneOffsetM;
  let x = centreX + leftNormalX * lateralM;
  let y = centreY + leftNormalY * lateralM;
  // The clamp settles in a couple of passes: the side is resolved AT the
  // pose, and moving the car along the normal can change that side, so the
  // bound is re-read until the pose sits inside the room it is measured on.
  for (let pass = 0; pass < 3; pass += 1) {
    const boundM = resolveSimWorkerSpawnBoundM(profile, station, x, y);
    if (Math.abs(lateralM) <= boundM) {
      return { x, y, lateralM };
    }
    lateralM = Math.sign(lateralM) * boundM;
    x = centreX + leftNormalX * lateralM;
    y = centreY + leftNormalY * lateralM;
  }
  return { x, y, lateralM };
}

/**
 * Resolves the published edge AT one spline station.
 *
 * THE STATION-READ EDGE: the road's width changes along the loop, so the
 * half-width is READ AT THE STATION - the linear interpolation between the
 * two published sample edges that own it - instead of the nearest sample's
 * edge (which can be a metre away and centimetres narrower or wider).
 *
 * @param edges - The published per-sample edges of the loop.
 * @param boundaryIndex - Index of the sample whose span owns the station.
 * @param fraction - The station's fraction along that span (0..1).
 * @returns The published station edge.
 */
function resolveSimWorkerSplineSpawnStation(
  edges: readonly TrackSplineEdge[],
  boundaryIndex: number,
  fraction: number,
): SimWorkerSpawnStation {
  const fromEdge = edges[boundaryIndex];
  const toEdge = edges[(boundaryIndex + 1) % edges.length];
  return {
    segmentIndex: fromEdge.segmentIndex,
    innerHalfWidthM:
      fromEdge.innerHalfWidthM +
      (toEdge.innerHalfWidthM - fromEdge.innerHalfWidthM) * fraction,
    outerHalfWidthM:
      fromEdge.outerHalfWidthM +
      (toEdge.outerHalfWidthM - fromEdge.outerHalfWidthM) * fraction,
  };
}

/**
 * Lays the side-by-side grid on ONE station of the road-anchored spawn path.
 *
 * Educational note: the law is the AC-105 START GRID - the WHOLE fleet (the
 * player AND every rival) waits FULLY BEHIND the green start line, SIDE BY
 * SIDE on ONE shared station, so no car starts on or above the line. The
 * upstream direction of the closed-loop arc frame is the lap TAIL, so the
 * station is `loopLengthM - clearance` (the wrap is MANDATORY: the raw
 * `-clearance` would clamp onto sample 0, which is ON the line). Each lane
 * is offset along the SPAN CHORD's normal - the frame the pose and the
 * heading already use - so a lateral offset can never slip the car along the
 * road, and the lane's lateral offset is clamped to the published
 * half-width READ AT THIS STATION.
 *
 * @param path - The road-anchored spawn path (samples + loop length).
 * @param profile - The published edge profile of the loop.
 * @param agentCount - Fleet size (>= 1).
 * @returns One spawn pose per fleet slot, ordered car 0 first.
 */
function resolveSimWorkerSplineSpawnGrid(
  path: { samples: readonly SplineSample[]; loopLengthM: number },
  profile: TrackEdgeProfile,
  agentCount: number,
): PhysicsCarSpawnPose[] {
  const { samples, loopLengthM } = path;
  const edges = resolveTrackSplineEdges(samples, profile);

  // Step 1: THE ONE SHARED STATION - the lap tail (upstream of the line).
  //
  // Educational note — the SIGN-CORRECTED modulo: the JavaScript percent
  // keeps the dividend's sign, so a loop SHORTER than the clearance would
  // walk the station NEGATIVE and land the fleet ON the start line. The
  // double modulo folds the station into `[0, loopLengthM)` for EVERY loop
  // length and stays byte-identical for the shipped layouts (a loop at or
  // above the clearance).
  const stationArcM =
    (((loopLengthM - NEATRACING_SIM_WORKER_SPAWN_BEHIND_CLEARANCE_M) %
      loopLengthM) +
      loopLengthM) %
    loopLengthM;

  // Step 2: The span that owns the station. A span the distance lands
  // EXACTLY at the start of resolves the pose AT the span's sample with the
  // span's own forward direction.
  let accumulatedM = 0;
  let boundaryIndex = 0;
  let spanLengthM = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const from = samples[index];
    const to = samples[(index + 1) % samples.length];
    const spanM = Math.hypot(to.x - from.x, to.y - from.y);
    if (spanM > 0 && accumulatedM + spanM > stationArcM) {
      boundaryIndex = index;
      spanLengthM = spanM;
      break;
    }
    accumulatedM += spanM;
  }
  const from = samples[boundaryIndex];
  const to = samples[(boundaryIndex + 1) % samples.length];
  const fraction =
    spanLengthM > 0
      ? Math.min(1, Math.max(0, (stationArcM - accumulatedM) / spanLengthM))
      : 0;

  // Step 3: The station's centre, its heading and its LEFT normal. The
  // heading IS the span chord's direction - the road the autopilot reads at
  // this pose - and the fraction interpolates the pose onto the road
  // between the two samples.
  const centreX = from.x + (to.x - from.x) * fraction;
  const centreY = from.y + (to.y - from.y) * fraction;
  const heading = Math.atan2(to.y - from.y, to.x - from.x);
  const leftNormalX = spanLengthM > 0 ? -(to.y - from.y) / spanLengthM : 0;
  const leftNormalY = spanLengthM > 0 ? (to.x - from.x) / spanLengthM : 0;

  // Step 4: The published room AT the station, then the lane offsets.
  const station = resolveSimWorkerSplineSpawnStation(
    edges,
    boundaryIndex,
    fraction,
  );
  const room = resolveSimWorkerSpawnRoomM(station, profile);
  const laneOffsetsM = resolveSimWorkerSpawnLaneOffsetsM(
    room.leftRoomM + room.rightRoomM,
    agentCount,
  );

  // Step 5: One pose per lane - every car on the SAME station.
  return laneOffsetsM.map((laneOffsetM) => {
    const pose = resolveSimWorkerSpawnLanePose(
      centreX,
      centreY,
      leftNormalX,
      leftNormalY,
      laneOffsetM,
      profile,
      station,
    );
    return { x: pose.x, y: pose.y, heading };
  });
}

/**
 * Lays the side-by-side grid on the ONE station of the legacy chord walk.
 *
 * Educational note: the hand-authored fixture rings do NOT anchor their
 * spline at segment 0's start, so their grid walks segment 0's chord - and
 * the upstream direction of that walk is the chord's BACKWARD direction, so
 * the ONE shared station sits `-clearance` behind segment 0's start point.
 * The lanes, the room and the clamp are the SAME law the road-anchored walk
 * uses.
 *
 * @param profile - The published edge profile of the loop.
 * @param firstSegment - Segment 0 of the owned track.
 * @param agentCount - Fleet size (>= 1).
 * @returns One spawn pose per fleet slot, ordered car 0 first.
 */
function resolveSimWorkerChordSpawnGrid(
  profile: TrackEdgeProfile,
  firstSegment: TrackSegment,
  agentCount: number,
): PhysicsCarSpawnPose[] {
  const chordX = firstSegment.endX - firstSegment.startX;
  const chordY = firstSegment.endY - firstSegment.startY;
  const chordLengthM = Math.hypot(chordX, chordY);
  const heading = Math.atan2(chordY, chordX);
  const directionX = chordLengthM > 0 ? chordX / chordLengthM : 0;
  const directionY = chordLengthM > 0 ? chordY / chordLengthM : 0;

  // THE ONE SHARED STATION: `clearance` BEHIND segment 0's start point.
  const centreX =
    firstSegment.startX -
    directionX * NEATRACING_SIM_WORKER_SPAWN_BEHIND_CLEARANCE_M;
  const centreY =
    firstSegment.startY -
    directionY * NEATRACING_SIM_WORKER_SPAWN_BEHIND_CLEARANCE_M;
  const leftNormalX = -directionY;
  const leftNormalY = directionX;

  // The published edge of the chord the station sits on.
  const chordEdge = profile.chords[0];
  const station: SimWorkerSpawnStation = {
    segmentIndex: chordEdge.segmentIndex,
    innerHalfWidthM: chordEdge.innerHalfWidthM,
    outerHalfWidthM: chordEdge.outerHalfWidthM,
  };
  const room = resolveSimWorkerSpawnRoomM(station, profile);
  const laneOffsetsM = resolveSimWorkerSpawnLaneOffsetsM(
    room.leftRoomM + room.rightRoomM,
    agentCount,
  );

  return laneOffsetsM.map((laneOffsetM) => {
    const pose = resolveSimWorkerSpawnLanePose(
      centreX,
      centreY,
      leftNormalX,
      leftNormalY,
      laneOffsetM,
      profile,
      station,
    );
    return { x: pose.x, y: pose.y, heading };
  });
}

/**
 * Resolves the deterministic fleet spawn poses from the owned track.
 *
 * Educational note: the grid is the AC-105 START GRID - the WHOLE fleet
 * (the player AND every rival) waits FULLY BEHIND the green start line,
 * SIDE BY SIDE on ONE shared station, one distinct lane each. The
 * placement is a PURE function of the track geometry and the fleet size (no
 * RNG, no clock anywhere), and every pose stays inside the published
 * asymmetric half-width READ AT ITS OWN SPAWN STATION. THE PATH: on the
 * generated layouts the station sits on the ROAD-ANCHORED spline (the road
 * the autopilot reads) at the lap tail; on the hand-authored fixture rings
 * (whose spline does not anchor at segment 0's start) it sits on segment 0's
 * chord, `clearance` behind its start point.
 *
 * @param trackSpec - Owned track geometry.
 * @param agentCount - Fleet size (>= 1).
 * @returns One spawn pose per fleet slot, ordered car 0 first.
 */
function resolveSimWorkerFleetSpawnPoses(
  trackSpec: TrackSpec,
  agentCount: number,
): PhysicsCarSpawnPose[] {
  const firstSegment = trackSpec.segments[0];
  if (firstSegment === undefined) {
    // Degenerate geometry (the init rejects a segment-less track spec).
    return [];
  }
  const profile = resolveTrackEdgeProfile(trackSpec.segments);

  // THE ROAD-ANCHORED SPAWN PATH - the generated layouts.
  const splinePath = resolveSimWorkerSplineSpawnPath(trackSpec);
  if (splinePath !== null) {
    return resolveSimWorkerSplineSpawnGrid(splinePath, profile, agentCount);
  }

  // THE LEGACY CHORD WALK - the hand-authored fixture rings.
  return resolveSimWorkerChordSpawnGrid(profile, firstSegment, agentCount);
}

/**
 * The NAMED no-controller error (AC-029 — the retired scripted lane).
 *
 * Educational note: the LEGACY SCRIPTED LANE is retired, so an agent car
 * without the injected NGE control seam has NO driver. The message NAMES the
 * missing controller (the contract pins the REASON, not one exact wording)
 * instead of falling back to a scripted drive.
 *
 * Educational note — how to MATCH it: the THROWN message is this reason text
 * plus a diagnostic suffix (` (race mode: <mode>)`), so a consumer asserts
 * with `message.includes(NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR)` — an
 * equality check against the constant alone does NOT match. The suffix stays
 * because the resolver's race-mode parameter exists only to name the mode
 * here, and the resolver arity is pinned by the byte-frozen RED homes.
 */
export const NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR =
  'sim worker: no controller injected for the agent cars';

/**
 * The retired scripted lane's racing-line slot (always empty).
 *
 * Educational note — why the slot survives while the lane is gone: the
 * resolver's FOURTH positional parameter carries the racing line the
 * scripted autopilot used to follow, and the five-argument call sites (the
 * byte-frozen seam RED home and the re-pinned green test) still fill that
 * slot while the injected controller rides in the FIFTH slot. The worker no
 * longer OWNS a racing line — the injected seam derives its driving line
 * from the adopted network — so the session carries no spline samples and
 * this empty list fills the retired slot.
 */
const RETIRED_SCRIPTED_LANE_SPLINE_SAMPLES: readonly SplineSample[] =
  Object.freeze([]);

/**
 * Resolves the per-tick control inputs for the whole fleet.
 *
 * Educational note — the ONE live lane (AC-029, the retired scripted lane):
 * - car 0 is the HUMAN car in BOTH race modes (LD-12 lane isolation): it
 *   replays the latest host sample — or the neutral sample before the first
 *   one — and it is NEVER handed to a network.
 * - every AGENT car (index >= 1) runs the INJECTED NGE control seam
 *   (observe -> activate -> steer through the LATEST adopted network). An
 *   agent car with no adopted network YET gets the neutral control sample,
 *   never a scripted drive.
 * - there is NO scripted fallback lane: an agent car whose session carries
 *   no injected controller raises the NAMED no-controller error instead of
 *   driving a scripted line.
 *
 * Every published input is a FRESH object: a resolved tick's inputs stay
 * observable after the next tick resolves (the adopted-snapshot round-trip
 * law), so the seam's reused control buffer is copied at this boundary.
 *
 * @param cars - Fleet states entering this tick.
 * @param raceMode - Session race mode (the retired scripted lane made the
 *   mode irrelevant to the control decision — car 0 is the human car in both
 *   modes — so the slot only names the mode in the no-controller error).
 * @param humanSample - Latest host sample (may be undefined pre-first sample).
 * @param splineSamples - The RETIRED scripted lane's racing line (the
 *   positional slot the five-argument call sites still fill; the worker
 *   never reads it).
 * @param controller - The injected NGE control seam driving the agent cars
 *   (undefined only when the session booted without one).
 * @returns Exactly one input per fleet car, ordered like the fleet.
 * @throws Error When an agent car has no injected controller (the NAMED
 *   no-controller error).
 */
function resolveSimWorkerFleetTickInputs(
  cars: readonly PhysicsCarState[],
  raceMode: NeatRacingSimWorkerRaceMode,
  humanSample: PhysicsStepInput | undefined,
  splineSamples: readonly SplineSample[],
  controller: NeatRacingFleetControlSeam | undefined,
): PhysicsStepInput[] {
  // Step 1: Resolve every slot in fleet order (car 0 first).
  return cars.map((_car, carIndex) => {
    // Step 2: Car 0 keeps the HUMAN lane in BOTH race modes (the human lane
    //         is never handed to a network).
    if (carIndex === 0) {
      return humanSample ?? NEUTRAL_SIM_WORKER_CONTROL_SAMPLE;
    }

    // Step 3: An agent car with NO injected controller has NO driver (the
    //         legacy scripted lane is retired — AC-029).
    if (controller === undefined) {
      throw new Error(
        `${NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR} (race mode: ${raceMode})`,
      );
    }

    // Step 4: The agent cars run the injected seam.
    const seamControl = mapNeatRacingFleetControl(
      controller.computeControl(cars, carIndex),
    );
    return seamControl ?? NEUTRAL_SIM_WORKER_CONTROL_SAMPLE;
  });
}

/**
 * Serializes one network's connection weights positionally — the recycled
 * `CarGenome.serialize()` plane: `network.connections[i].weight` in
 * connection order, mirroring the promotion slice's private
 * `serializeNetworkWeightsToFloat32Array` walk (fresh array; the caller
 * owns its bytes).
 *
 * @param network - The spawned network to serialize.
 * @returns The positional connection-weight array.
 */
function serializeSimWorkerRivalMindWeights(network: Network): Float32Array {
  const weights = new Float32Array(network.connections.length);
  for (let weightIndex = 0; weightIndex < weights.length; weightIndex += 1) {
    weights[weightIndex] = network.connections[weightIndex].weight;
  }
  return weights;
}

/**
 * Encodes one serialized weight plane as a byte-exact hex string (P4-S2C,
 * Step 24C — the panel-network envelope).
 *
 * Educational note — THE OWNER-LOCAL MIRROR: the `promotion.rules.ts` and
 * `rival.propagation.ts` encoders are private to THEIR payloads; this
 * payload owns its own serialization path (the one serialization path per
 * payload law — never a shared dual path). Lowercase hex, two characters
 * per byte, no separators — the exact byte pattern the `Float32Array` holds,
 * never a `toJSON`.
 *
 * @param weights - The serialized weight plane to encode.
 * @returns The hex string of the array's raw bytes in NATIVE byte order
 *   (same-runtime byte parity only, never cross-environment exact).
 */
function encodeSimWorkerPanelWeightsHex(weights: Float32Array): string {
  const bytes = new Uint8Array(
    weights.buffer as ArrayBuffer,
    weights.byteOffset,
    weights.byteLength,
  );
  let hex = '';
  for (const byte of bytes) {
    hex += byte.toString(16).padStart(2, '0');
  }
  return hex;
}

/**
 * Materializes one rival's SPAWN mind as the flat serialized plane (P4-S2,
 * AC-032).
 *
 * Educational note — the DETERMINISTIC RECONSTRUCTION precedent (the
 * promotion slice and the adaptation session both build agent-car minds
 * this way): the worker never reaches a live evolved network through the
 * seam — the controller seam holds the adopted networks behind
 * `computeControl` — so the worker reconstructs the rival's spawn mind from
 * the tuple: `new Network(level channel count, controller output count,
 * { seed: deriveNeatRacingCarNetworkChildSeed(seed, carIndex) })`, the SAME
 * construction `createNeatRacingAdaptationCarStates` uses for the same car.
 * The instance is serialized positionally and discarded: only the flat
 * SerializedWeights plane enters the ledger, and a later replacement entry
 * swaps the ledger's bytes, not an instance.
 *
 * @param seed - The AC-026 tuple seed the session booted with.
 * @param level - The ladder rung (selects the observation channel count).
 * @param carIndex - The rival's fleet car index (the per-car child seed).
 * @returns The rival's serialized spawn mind.
 */
function materializeSimWorkerRivalSpawnMind(
  seed: number,
  level: NeatRacingObservationLevel,
  carIndex: number,
): Float32Array {
  return serializeSimWorkerRivalMindWeights(
    new Network(
      resolveNeatRacingObservationChannelCount(level),
      NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
      {
        seed: deriveNeatRacingCarNetworkChildSeed(seed, carIndex),
      },
    ),
  );
}

/**
 * Resolves the worker-plane rival propagation state from one init payload
 * (P4-S2, AC-032).
 *
 * Defensive resolution (the init `opponentSnapshot` is opaque at this
 * boundary): the level is read from the snapshot only when it is an integer
 * naming a pinned ladder rung (`NEATRACING_TEAM_LAYOUTS`), and the rivals
 * are that rung's non-zero team slots within the booted fleet size. A
 * snapshot that names no rivals (the solo rung, a missing snapshot, or a
 * malformed one) leaves the lane UNDEFINED — the 1-3-car baseline boots
 * never construct a network and never fire a swap.
 *
 * @param payload - Validated init payload.
 * @returns The rival-propagation state, or undefined when the boot has no
 *   rival lane.
 */
function createSimWorkerRivalPropagationState(
  payload: NeatRacingSimWorkerInitPayload,
): SimWorkerRivalPropagationState | undefined {
  // Step 1: Read the ladder rung defensively from the opaque snapshot.
  const snapshot = payload.opponentSnapshot;
  const snapshotLevel =
    snapshot !== null && typeof snapshot === 'object'
      ? (snapshot as { level?: unknown }).level
      : undefined;
  if (
    typeof snapshotLevel !== 'number' ||
    !Number.isInteger(snapshotLevel) ||
    (snapshotLevel !== 1 && snapshotLevel !== 2 && snapshotLevel !== 3)
  ) {
    return undefined;
  }
  const level: NeatRacingObservationLevel = snapshotLevel;
  const layout = NEATRACING_TEAM_LAYOUTS[level];

  // Step 2: The rivals are the rung's non-zero team slots inside the booted
  //         fleet (car 0 is the main agent's slot at every rung — AC-033;
  //         the operator rejects any rival index at or below car 0).
  const rivalCarIndices: number[] = [];
  const fleetSlotCeiling = Math.min(layout.length, payload.agentCount);
  for (let carIndex = 1; carIndex < fleetSlotCeiling; carIndex += 1) {
    if (layout[carIndex] !== 0) {
      rivalCarIndices.push(carIndex);
    }
  }
  if (rivalCarIndices.length === 0) {
    return undefined;
  }

  // Step 3: Materialize the serialized spawn minds (the ledger seeding).
  const rivalMinds = new Map<number, Float32Array>();
  for (const rivalCarIndex of rivalCarIndices) {
    rivalMinds.set(
      rivalCarIndex,
      materializeSimWorkerRivalSpawnMind(payload.seed, level, rivalCarIndex),
    );
  }

  // Step 4: One detector baseline slot per fleet car (car 0's slot mirrors
  //         `lapLinePrevSide` and stays unused).
  const rivalPrevSides: (number | null)[] = new Array(payload.agentCount).fill(
    null,
  );

  // Step 5: One ARMING slot per fleet car (the AC-105 behind-the-line law):
  //         a rival that still waits BEHIND the start-line plane never
  //         latches a side and never banks a crossing.
  const rivalStarted: boolean[] = new Array(payload.agentCount).fill(false);

  return {
    rivalCarIndices,
    level,
    rivalMinds,
    rivalPrevSides,
    rivalStarted,
    nextEventIndex: 0,
  };
}

/**
 * Resolves the level the worker rules machine seeds from: the round's
 * carried boot level.
 *
 * Educational note — the carried level (fix packet
 * `P4-S2A-debug-rung-seam-green` iteration 2): the host copies the round's
 * OWN boot rung into the init payload's `opponentSnapshot`
 * (`{ level, carCount }`), so the rules state machine — not just the fleet
 * ladder and the rival-propagation lane — seeds at the SAME rung. The
 * browser's mode-switch re-boot now carries a `?level=` debug rung across
 * the switch, and a solo lvl-1 rules state beside a rival-bearing lvl-2
 * fleet would desync the HUD read model from the round's honest layout.
 *
 * Defensive resolution (the init `opponentSnapshot` is opaque at this
 * boundary, mirroring the rival-propagation lane's read): the level is
 * read only when it is an integer naming an in-ladder rung
 * (1..NEATRACING_HUMAN_MAX_LEVEL); a snapshot that names no rung (null, a
 * non-object, a malformed level or an out-of-band integer) keeps the
 * byte-identical prior default (NEATRACING_HUMAN_START_LEVEL — the worker
 * started every race at lvl 1 before the rung carried across the switch).
 *
 * @param payload - Validated init payload.
 * @returns The rung the rules machine seeds from.
 */
function resolveSimWorkerRulesSeedLevel(
  payload: NeatRacingSimWorkerInitPayload,
): number {
  const snapshot = payload.opponentSnapshot;
  const snapshotLevel =
    snapshot !== null && typeof snapshot === 'object'
      ? (snapshot as { level?: unknown }).level
      : undefined;
  if (
    typeof snapshotLevel === 'number' &&
    Number.isInteger(snapshotLevel) &&
    snapshotLevel >= NEATRACING_HUMAN_START_LEVEL &&
    snapshotLevel <= NEATRACING_HUMAN_MAX_LEVEL
  ) {
    return snapshotLevel;
  }
  return NEATRACING_HUMAN_START_LEVEL;
}

/**
 * Boots the sim state for one session from a validated init payload.
 *
 * @param payload - Validated init payload.
 * @returns The fresh internal session bag in the `running` phase, with the
 *   boot detail guaranteed non-undefined (the running bag always carries it).
 */
function createSimWorkerRunningState(
  payload: NeatRacingSimWorkerInitPayload,
): SimWorkerSessionState & {
  readyDetail: NonNullable<SimWorkerSessionState['readyDetail']>;
} {
  // Step 1: Preallocate the accumulator through the default allocator seam.
  const renderLoop = createNeatRacingRenderLoopState();

  // Step 2: Spawn the fleet deterministically (seed + index offset per car).
  const spawnPoses = resolveSimWorkerFleetSpawnPoses(
    payload.trackSpec,
    payload.agentCount,
  );
  const cars = spawnPoses.map((pose, carIndex) =>
    createInitialPhysicsCarState(payload.seed + carIndex, pose),
  );

  // Step 3: Seed the rules state machine at the ROUND'S CARRIED boot level
  //         (the rung the init payload's `opponentSnapshot` names — the
  //         fleet ladder, the rival-propagation lane and the rules machine
  //         ride the SAME rung; a snapshot-less boot keeps the
  //         byte-identical lvl-1 seed behind the defensive read).
  const rules = createRaceRulesState(resolveSimWorkerRulesSeedLevel(payload));

  // Step 4: Resolve the worker-plane rival propagation lane (P4-S2, AC-032):
  //         undefined for boots whose snapshot names no rivals, so the 1-3-car
  //         baseline boots keep their exact prior behavior.
  const rivalPropagation = createSimWorkerRivalPropagationState(payload);

  return {
    phase: 'running',
    renderLoop,
    cars,
    rules,
    lapLinePrevSide: null,
    lapLineStarted: false,
    raceMode: payload.raceMode ?? 'ai',
    humanSample: undefined,
    controller: payload.controller,
    surfaceResolver: createTrackSurfaceResolver(payload.trackSpec.segments),
    firstSegment: payload.trackSpec.segments[0],
    rivalPropagation,
    readyDetail: {
      seed: payload.seed,
      agentCount: payload.agentCount,
      packSchemaVersion: payload.packSchemaVersion,
      trackId: payload.trackId,
    },
  };
}

/**
 * The shared empty rival-swaps list: frames with no rival crossing reuse it
 * so the tick loop never allocates (the 1-3-car baseline boots every frame).
 */
const SIM_WORKER_NO_RIVAL_SWAPS: readonly NeatRacingRivalSwapPayload[] =
  Object.freeze([]);

/**
 * Runs ONE tick of the worker-plane rival propagation lane (P4-S2, AC-032).
 *
 * Each rival's lap-line detector sees its ENTERING pose (the same port the
 * car-0 detector runs: segment-0 chord projection, lateral gate, along
 * deadband, first-sample latch, sign-flip crossing), and a crossing rival
 * resolves ONE swap event through the operator — the crossing rival's
 * serialized ledger mind becomes the propagation base, every OTHER rival's
 * ledger entry is replaced by its bounded seeded nudge, and the event's
 * tuple record is collected for the frame's `frames` reply.
 *
 * Educational note — the lane's place in the tick: the detectors run in the
 * SAME tick position as car 0's (the pose entering the tick, before fleet
 * physics), but rival crossings feed ONLY this lane — `stepRaceRules` keeps
 * car 0's detector as its sole input, so the main agent carries its own
 * lap state and is independent of rival success (AC-032). The operator
 * REFUSES a mid-lap call, so the lane passes `atLapLineBoundary: true` only
 * at a detected line crossing.
 *
 * @param sessionState - Running session bag (mutates the rival ledger, the
 *   event counter and the per-rival detector baselines in place).
 * @param swaps - The frame-local event list being collected (events pushed
 *   in resolved order: rivals ascending per tick, ticks in session order).
 */
function stepSimWorkerRivalPropagationTick(
  sessionState: SimWorkerSessionState,
  swaps: NeatRacingRivalSwapPayload[],
): void {
  // Step 1: The lane is inert without a rival-bearing boot (undefined lane)
  //         or without its boot detail (the tuple seed the event derives
  //         from). Both are guaranteed while the session is `running`, so
  //         these guards are defensive only.
  const rival = sessionState.rivalPropagation;
  const tupleSeed = sessionState.readyDetail?.seed;
  const firstSegment = sessionState.firstSegment;
  if (
    rival === undefined ||
    tupleSeed === undefined ||
    firstSegment === undefined
  ) {
    return;
  }

  // Step 2: One detector per rival, in ascending car order, at the same tick
  //         position as car 0's detector.
  for (const rivalCarIndex of rival.rivalCarIndices) {
    const detector = resolveSimWorkerArmedLapLineDetectorStep(
      sessionState.cars[rivalCarIndex],
      rival.rivalPrevSides[rivalCarIndex],
      rival.rivalStarted[rivalCarIndex],
      firstSegment,
    );
    rival.rivalPrevSides[rivalCarIndex] = detector.nextSide;
    rival.rivalStarted[rivalCarIndex] = detector.started;
    if (!detector.crossing) {
      continue;
    }

    // Step 3: ONE swap event per crossing rival. The ledger mind IS the
    //         serialized crossing network (eagerly materialized at boot, so
    //         the lookup cannot miss — the guard keeps a ledger gap from
    //         ever reaching the operator, which would throw instead).
    const crossingMind = rival.rivalMinds.get(rivalCarIndex);
    if (crossingMind === undefined) {
      continue;
    }
    const result = propagateNeatRacingRivalWinnerWeights({
      rivalCarIndices: rival.rivalCarIndices,
      crossingCarIndex: rivalCarIndex,
      winnerWeights: crossingMind,
      tupleSeed,
      eventIndex: rival.nextEventIndex,
      atLapLineBoundary: true,
    });

    // Step 4: Apply the replacement minds to the ledger — THE propagation
    //         application in the fleet loop: the replacement weights are
    //         what each rival's NEXT crossing propagates from, so a swap
    //         chain compounds through the serialized plane deterministically.
    for (const replacement of result.replacements) {
      rival.rivalMinds.set(replacement.carIndex, replacement.weights);
    }
    rival.nextEventIndex += 1;

    // Step 5: Publish the event's tuple record (JSON-safe; the crossing
    //         snapshot + the mutation seed ride the frame reply).
    swaps.push(toNeatRacingRivalSwapPayload(result.snapshot));
  }
}

/**
 * Releases 0-4 fixed steps for one paced frame and mutates the running session.
 *
 * @param sessionState - Running session bag mutated in place.
 * @param timestampMs - rAF-style frame timestamp driving the accumulator.
 * @param framePayload - The paced frame payload (its optional D5 sample is
 *   latched BEFORE ticking so zero-step frames still refresh the held manual
 *   sample, matching the phase-1 sampling-outside-the-step-guard behavior).
 * @returns The rival-swap events released during this frame's ticks (the
 *   shared empty list when no rival crossed; never a live tick-internal
 *   array alias).
 */
function driveSimWorkerFrame(
  sessionState: SimWorkerSessionState,
  timestampMs: number,
  framePayload: NeatRacingSimWorkerFramePayload,
): readonly NeatRacingRivalSwapPayload[] {
  // Step 1: Latch the D5 sample FIRST (in BOTH race modes) so every
  // catch-up tick below replays the SAME once-per-frame sample.
  //
  // Educational note: the human lane owns car 0 in BOTH race modes (LD-12),
  // so the sample must be latched in `ai` mode too.
  if (framePayload.sample !== undefined) {
    sessionState.humanSample = framePayload.sample;
  }

  // Step 2: Advance the accumulator and release the frame's tick count.
  const stepCount = advanceFrameSteps(sessionState.renderLoop, timestampMs);

  // Step 3: Run each released tick in the phase-1 order: the detector sees
  // the pose ENTERING the tick, then fleet physics, then race rules.
  const frameSwaps: NeatRacingRivalSwapPayload[] = [];
  for (let stepIndex = 0; stepIndex < stepCount; stepIndex += 1) {
    const detector = resolveSimWorkerArmedLapLineDetectorStep(
      sessionState.cars[0],
      sessionState.lapLinePrevSide,
      sessionState.lapLineStarted,
      sessionState.firstSegment,
    );
    sessionState.lapLinePrevSide = detector.nextSide;
    sessionState.lapLineStarted = detector.started;
    stepSimWorkerRivalPropagationTick(sessionState, frameSwaps);
    const inputs = resolveSimWorkerFleetTickInputs(
      sessionState.cars,
      sessionState.raceMode,
      sessionState.humanSample,
      RETIRED_SCRIPTED_LANE_SPLINE_SAMPLES,
      sessionState.controller,
    );
    // One fleet pass per tick: every member steps through `stepPhysics`,
    // then ONE collision-resolution pass keeps overlap bookkeeping in sync.
    sessionState.cars = stepPhysicsFleet(
      sessionState.cars,
      inputs,
      sessionState.surfaceResolver,
    );
    sessionState.rules = stepRaceRules(sessionState.rules, {
      dtS: NEATRACING_FIXED_TIMESTEP_SECONDS,
      forwardLapLineCrossing: detector.crossing,
    });
  }
  return frameSwaps.length > 0 ? frameSwaps : SIM_WORKER_NO_RIVAL_SWAPS;
}

/**
 * Builds the structured `frames` reply for one paced frame.
 *
 * @param fromTick - Total-tick counter before the frame's released ticks.
 * @param sessionState - Running session bag (post-tick state).
 * @param rivalSwaps - The frame's rival-swap events (empty when no rival
 *   crossed; an empty list OMITS the optional payload field so untouched
 *   frames keep their exact prior shape).
 * @returns The reply payload snapshot (fresh copies, no aliasing into later
 *   tick output).
 */
function buildSimWorkerFramesPayload(
  fromTick: number,
  sessionState: SimWorkerSessionState,
  rivalSwaps: readonly NeatRacingRivalSwapPayload[],
): NeatRacingSimWorkerFramesPayload {
  return {
    fromTick,
    toTick: sessionState.cars[0].tick,
    cars: sessionState.cars.map((car) => ({ ...car })),
    rules: { ...sessionState.rules },
    ...(rivalSwaps.length > 0 ? { rivalSwaps } : {}),
  };
}

/**
 * Answers ONE panel-network request with the elected car's network envelope
 * (P4-S2C, Step 24C — the docked-right panel network transport).
 *
 * Educational note — THE OBSERVATIONAL COPY: the request is a READ. The
 * dispatcher never touches the RNG, never advances the simulation and never
 * mutates the rival ledger — it copies the stored rival mind's bytes into a
 * fresh hex string, so two identical requests answer byte-identically while
 * the ledger keeps its own bytes (the copy law).
 *
 * Educational note — THE M1 REPLY LAW: every guard below answers with an
 * ERROR REPLY; no rejection ever escapes `handleRequest` as a throw (the
 * same shape the frame case uses for its own guards).
 *
 * @param state - Internal session bag (read-only on this path).
 * @param request - The panel-network request (the carrier car index).
 * @returns The single reply: the panel-network envelope or an error reply.
 */
function dispatchSimWorkerNetworkRequest(
  state: SimWorkerSessionState,
  request: Extract<NeatRacingSimWorkerRequest, { type: 'network' }>,
): NeatRacingSimWorkerResponse {
  // Step 1: A session that never booted owns no rival lane to serve.
  if (state.phase === 'idle') {
    return {
      type: 'error',
      payload: {
        message: 'sim worker network failed: the session is not initialized',
      },
    };
  }

  // Step 2: A stopped session is terminal — it answers nothing else.
  if (state.phase === 'stopped') {
    return {
      type: 'error',
      payload: {
        message: 'sim worker network failed: the session is stopped',
      },
    };
  }

  // Step 3: The rival lane owns the transported minds (a boot without a
  //         rival ladder has none to carry).
  const rivalPropagation = state.rivalPropagation;
  if (
    rivalPropagation === undefined ||
    rivalPropagation.rivalMinds.size === 0
  ) {
    return {
      type: 'error',
      payload: {
        message: 'sim worker network failed: the session has no rival lane',
      },
    };
  }

  // Step 4: The carrier index must name a rival slot (car 0 is the human's
  //         slot and is never carried).
  if (!Number.isInteger(request.carIndex) || request.carIndex < 1) {
    return {
      type: 'error',
      payload: {
        message: 'sim worker network failed: malformed network request',
      },
    };
  }

  // Step 5: An index the ledger does not know is a loud rejection.
  const rivalMind = rivalPropagation.rivalMinds.get(request.carIndex);
  if (rivalMind === undefined) {
    return {
      type: 'error',
      payload: {
        message: 'sim worker network failed: unknown rival car index',
      },
    };
  }

  // Step 6: The happy path — one observational copy of the stored mind.
  return {
    type: 'network',
    payload: {
      kind: NEATRACING_PANEL_NETWORK_ENVELOPE_KIND,
      carIndex: request.carIndex,
      channelCount: resolveNeatRacingObservationChannelCount(
        rivalPropagation.level,
      ),
      outputCount: NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
      weights: encodeSimWorkerPanelWeightsHex(rivalMind),
    },
  };
}

/**
 * Routes one validated sim-worker request to its response list (the
 * `handleRequest` dispatch). Pure over the session bag apart from the
 * mutation the routed request performs.
 *
 * Educational note — WHY the dispatch is a standalone function: `handleRequest`
 * wraps this call in the M1 selective catch (the NAMED no-controller error is
 * the ONE error that converts into a protocol `{ type: 'error' }` reply), and
 * the routing switch needs a provably-returning shape for that wrapper to own
 * the whole dispatch.
 *
 * @param state - Internal session bag (mutated by the routed request).
 * @param request - Host request message (non-junk; type-routed below).
 * @returns The ordered responses to post for this request.
 */
function dispatchSimWorkerRequest(
  state: SimWorkerSessionState,
  request: NeatRacingSimWorkerRequest,
): NeatRacingSimWorkerResponse[] {
  // Route by message type; `stop` stays deliberately reply-free.
  switch (request.type) {
    case 'init': {
      if (state.phase !== 'idle') {
        return [
          {
            type: 'error',
            payload: {
              message: `sim worker init failed: session already ${state.phase}`,
            },
          },
        ];
      }
      const validation = validateSimWorkerInitPayload(request.payload);
      if (validation !== null) {
        return [{ type: 'error', payload: { message: validation.error } }];
      }
      const runningState = createSimWorkerRunningState(request.payload);
      Object.assign(state, runningState);
      return [
        {
          type: 'ready',
          payload: {
            seed: runningState.readyDetail.seed,
            agentCount: runningState.readyDetail.agentCount,
            packSchemaVersion: runningState.readyDetail.packSchemaVersion,
            trackId: runningState.readyDetail.trackId,
            raceMode: runningState.raceMode,
          },
        },
      ];
    }
    case 'frame': {
      if (state.phase === 'idle') {
        return [
          {
            type: 'error',
            payload: {
              message: 'sim worker frame failed: session not initialized',
            },
          },
        ];
      }
      if (state.phase === 'stopped') {
        return [
          {
            type: 'error',
            payload: {
              message: 'sim worker frame failed: session stopped',
            },
          },
        ];
      }
      const validation = validateSimWorkerFramePayload(request.payload);
      if (validation !== null) {
        return [{ type: 'error', payload: { message: validation.error } }];
      }
      const fromTick = state.cars[0].tick;
      const frameRivalSwaps = driveSimWorkerFrame(
        state,
        request.payload.timestampMs,
        request.payload,
      );
      return [
        {
          type: 'frames',
          payload: buildSimWorkerFramesPayload(
            fromTick,
            state,
            frameRivalSwaps,
          ),
        },
      ];
    }
    case 'network': {
      // The panel-network transport: exactly ONE reply (the envelope or the
      // error reply — never a throw).
      return [dispatchSimWorkerNetworkRequest(state, request)];
    }
    case 'stop': {
      // Lifecycle message, reply-free by contract (flappy precedent).
      state.phase = 'stopped';
      return [];
    }
    default: {
      const unknownType = String((request as { type?: unknown }).type);
      return [
        {
          type: 'error',
          payload: {
            message: `sim worker: unknown message type ${unknownType}`,
          },
        },
      ];
    }
  }
}

/**
 * Creates a pure sim-worker session state machine.
 *
 * Educational note: one factory per session keeps every boot test isolated
 * without touching `globalThis.self`. The worker entrypoint below builds the
 * module-level session and is the ONLY place that wires `self.onmessage` /
 * `self.postMessage` — the transport adapter stays three lines.
 *
 * @returns A session whose `handleRequest` folds messages into state and
 *   returns the ordered responses to post.
 *
 * @example
 * ```ts
 * const session = createSimWorkerSession();
 * const [ready] = session.handleRequest({ type: 'init', payload });
 * ```
 */
export function createSimWorkerSession(): NeatRacingSimWorkerSession {
  // Step 1: Start the protocol machine in the idle phase.
  const state: SimWorkerSessionState = {
    phase: 'idle',
    renderLoop: createNeatRacingRenderLoopState(),
    cars: [],
    rules: createRaceRulesState(),
    lapLinePrevSide: null,
    lapLineStarted: false,
    raceMode: 'ai',
    humanSample: undefined,
    controller: undefined,
    surfaceResolver: undefined,
    firstSegment: undefined,
    rivalPropagation: undefined,
    readyDetail: undefined,
  };

  return {
    handleRequest(
      request: NeatRacingSimWorkerRequest,
    ): NeatRacingSimWorkerResponse[] {
      // Step 2: Reject non-object messages outright (transport-level junk).
      if (request === null || typeof request !== 'object') {
        return [
          {
            type: 'error',
            payload: { message: 'sim worker: malformed message' },
          },
        ];
      }

      // Step 3: Route by message type inside a SELECTIVE catch (M1, slice
      // `P4-S2-rival-propagation`): the NAMED no-controller error is the ONE
      // error that converts into a protocol `{ type: 'error' }` reply, and
      // the reply carries the SAME message text the resolver raises (the
      // constant plus the ` (race mode: <mode>)` suffix), so a consumer
      // matching `message.includes(NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR)`
      // still works. Every OTHER worker error keeps its current behavior —
      // it still escapes `handleRequest` as an uncaught error.
      try {
        return dispatchSimWorkerRequest(state, request);
      } catch (error) {
        if (
          error instanceof Error &&
          error.message.includes(NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR)
        ) {
          return [{ type: 'error', payload: { message: error.message } }];
        }
        throw error;
      }
    },
    get snapshot(): NeatRacingSimWorkerSessionSnapshot {
      return {
        phase: state.phase,
        tick: state.cars.length > 0 ? state.cars[0].tick : 0,
        cars: state.cars.map((car) => ({ ...car })),
        rules: { ...state.rules },
        lapLinePrevSide: state.lapLinePrevSide,
      };
    },
  };
}

/**
 * Pure-function test seam for the sim worker (P2-S1-A green tests).
 *
 * Educational note: like every established worker module in this repo
 * (flappy evolution, asciiMaze), the entrypoint exposes its internals so
 * Jest can drive the protocol contract headless without a worker realm,
 * while production still boots through the module side effect below.
 */
export const NEATRACING_SIMULATION_WORKER_INTERNALS = {
  createSimWorkerSession,
  resolveSimWorkerLapLineDetectorStep,
  resolveSimWorkerFleetSpawnPoses,
  resolveSimWorkerFleetTickInputs,
  NEUTRAL_SIM_WORKER_CONTROL_SAMPLE,
  NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR,
} as const;

// Boot: one session per worker realm; the message adapter is the only
// impurity in this file. Responses are posted in order, exactly once each.
const simulationWorkerSession = createSimWorkerSession();

self.onmessage = (event: MessageEvent<NeatRacingSimWorkerRequest>): void => {
  for (const response of simulationWorkerSession.handleRequest(event.data)) {
    self.postMessage(response, []);
  }
};
